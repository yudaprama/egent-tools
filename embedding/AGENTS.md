# AGENTS.md

## What this is

Workspace-aware embedding provider library. Solves the cross-model embedding
space contamination problem: different embedding models produce incompatible
vector spaces, so mixing models in the same vector DB corrupts search quality.

**Solution:** Each workspace (tenant) is deterministically assigned to one
embedding provider via `DJB2(workspace_id) % num_providers`. Same workspace
always uses the same model. Zero DB lookup, fully deterministic.

## Package layout

```
embedding/
  djb2.go              # DJB2 hash function (deterministic, uniform)
  provider_openai.go   # OpenAI-compatible (OpenRouter, text-embedding-3-small)
  provider_nvidia.go   # NVIDIA NIM (nv-embed-v1)
  provider_gemini.go   # Google Gemini (gemini-embedding-001)
  tenant_aware.go      # TenantAwareEmbedder — core logic
  registry.go          # BuildProvidersFromEnvWithKeys() — env-based factory
  billing.go           # BillingHook interface for usage tracking
  embedding_test.go    # Unit tests
```

## Key types

```go
// TenantAwareEmbedder selects provider per workspace using DJB2 hash.
type TenantAwareEmbedder struct { ... }

func NewTenantAwareEmbedder(providers []fp.Embedder) *TenantAwareEmbedder
func (t *TenantAwareEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error)
func (t *TenantAwareEmbedder) Dimension() int

// Provider constructors (all return fp.Embedder, dim=1024)
func NewOpenAICompatibleEmbedder(url, apiKey, model string, dim int) fp.Embedder
func NewNvidiaEmbedder(url, apiKey, model string, dim int) fp.Embedder
func NewGeminiEmbedder(apiKey, model string, dim int) fp.Embedder

// Factory — builds providers from .env comma-separated API keys
func BuildProvidersFromEnvWithKeys() []fp.Embedder
```

## How provider selection works

```go
// 1. TenantAwareEmbedder reads workspace ID from context
tenantID := memory.TenantIDFromContext(ctx)

// 2. DJB2 hash → deterministic index
idx := int(djb2(tenantID)) % len(providers)

// 3. Delegate to selected provider
return providers[idx].Embed(ctx, texts)
```

DJB2 is deterministic: same workspace ID → same provider, always.
Distribution is uniform across providers (~33% each for 3 providers).

## Ingestion consistency (egent-jobs)

Retrieval is not the only embedding writer. `egent-jobs` file-ingest also embeds
chunks into `public.embeddings`. To keep ONE workspace in ONE vector space, the
ingest path uses the SAME `TenantAwareEmbedder`:

- `egent-jobs/embeddings/tenant_aware.go` — `TenantAwareBatchEmbedder` adapts
  `TenantAwareEmbedder` to the local `embeddings.Embedder` (`EmbedBatch`).
- `embed_worker.go` injects the resolved `workspace_id` via
  `memory.WithTenantID(ctx, workspaceID)` before embedding, so DJB2 selects the
  same provider the retrieval side would for that workspace.
- `TenantAwareEmbedder.ModelForTenant` stamps the per-workspace model name on
  each `embeddings.Result.Model`, which the worker persists to the `model` column.

If ingestion and retrieval ever diverge (one fixed model on one side), stored
vectors live in different spaces and search silently degrades — keep both sides
on the tenant-aware router.

## Fallback behavior

When the selected provider fails:
1. Log the failure
2. Try next provider in the list (circular)
3. If all fail, return error from first provider

```go
for attempt := 0; attempt < len(providers); attempt++ {
    pidx := (idx + attempt) % len(providers)
    vecs, err := providers[pidx].Embed(ctx, texts)
    if err == nil {
        return vecs, nil  // success (may have fallen back)
    }
    // log and continue
}
```

## Provider API shapes

| Provider | Endpoint | Auth | Batch | input_type | dimensions |
|---|---|---|---|---|---|
| OpenAI | `POST /v1/embeddings` | `Authorization: Bearer` | Yes | No | Yes |
| NVIDIA | `POST /embeddings` | `Authorization: Bearer` | No (single) | Required | No |
| Gemini | `POST /models/{m}:embedContent` | `x-goog-api-key` | No (single) | No | No (truncate) |

All providers output **1024 dimensions** to match `public.embeddings vector(1024)`.

## Key rotation

Each provider has multiple API keys (comma-separated in `.env`):
- OpenRouter: `OPENROUTER_API_KEYS` (4 keys)
- NVIDIA: `NVIDIA_API_KEYS` (3 keys)
- Gemini: `GEMINI_API_KEYS` (13 keys)

A random key is picked per provider at startup via
`egent-common/envutil.PickRandomKey` (same utility as the NVIDIA reranker and
egent tool_builder). Each restart may use a different key, spreading rate-limit
usage across the pool.

## Billing integration

`billing.go` defines a `BillingHook` callback interface. The embedding package
does NOT depend on `plano-usage` directly. Callers wire the hook through
`TenantAwareEmbedder.SetBillingHook` (propagates to every wrapped
`billingEmbedder`):

```go
import (
    "context"
    usage "github.com/yudaprama/plano-usage"
    "github.com/yudaprama/egent-tools/embedding"
)

ta := embedding.NewTenantAwareEmbedder(providers)
ta.SetBillingHook(func(ctx context.Context, model string) {
    usage.Record(ctx, "embedding", model) // no-op when no actor in ctx
})
```

The actor ID is taken from the request context (`x-arch-actor-id` →
`usage.WithActorID` in the egent main.go), so billing is a no-op for
background/shared paths with no actor.

## Context propagation

Workspace ID flows through context via `memory.WithTenantID` / `memory.TenantIDFromContext`.
The `TenantAwareEmbedder.Embed()` reads it automatically.

```
HTTP request → Oathkeeper injects X-Tenant-Id → memory.WithTenantID(ctx) → TenantAwareEmbedder
```

## Dependencies

- `github.com/kawai-network/fileprocessor` — `Embedder` interface
- `github.com/yudaprama/egent-tools/memory` — context tenant ID
- `github.com/yudaprama/egent-common/envutil` — `PickRandomKey` for key rotation

## Gotchas

- **All providers MUST output 1024 dims.** OpenAI/NVIDIA are native 1024. Gemini truncates from 3072.
- **NVIDIA requires `input_type`.** Use `"passage"` for indexing, `"query"` for search.
- **Gemini only supports single-text requests.** Concurrent calls for batches.
- **No tenant ID = first provider.** Backward compatible with existing code.
- **Don't mix models in one workspace.** This is the whole point — DJB2 ensures consistency.

## Testing

```bash
cd egent-tools && go test ./embedding/...
```

Tests cover:
- DJB2 distribution uniformity
- DJB2 determinism
- TenantAwareEmbedder provider selection
- Fallback behavior
- No-tenant backward compat
