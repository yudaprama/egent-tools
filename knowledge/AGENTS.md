# AGENTS.md

Guidance for AI agents working on the **knowledge base / RAG read path** (the
`knowledge_search` tool and its retrieval pipeline).

This package owns the **read path**. The **write path** (ingest → chunk → embed)
lives in `egent-jobs/` + the `fileprocessor/` library. See "Related docs" below.

## What this is

`egent-tools/knowledge` is a Go library that exposes `knowledge_search` as an
Eino tool. It runs **hybrid retrieval** (pgvector HNSW ‖ BM25 keyword, fused with
weighted Reciprocal Rank Fusion) over a user's uploaded files, optionally
reranks with NVIDIA, dedupes per-file, and formats hits as an LLM context block.

- Module: `github.com/yudaprama/egent-tools` (Go 1.26)
- Consumed by every egent via a local `replace` directive. It is **not built
  standalone** — editing it here is picked up when the consuming egent recompiles.
- Degrades gracefully: no DB pool → no tool; no embedder → search skipped
  (`knowledge_get` stays); no `NVIDIA_API_KEYS` → rerank skipped silently.

## Related docs (read these — they have the depth this file points to)

| Doc | Covers |
|---|---|
| `docs/knowledge-base-architecture-review.md` | The authoritative architecture + a live correctness bug (B1), design risks, prioritized recommendations. **Read before changing retrieval.** |
| `fileprocessor/AGENTS.md` | The `fileprocessor` library internals (stores, chunker, loader, pgvector, hybrid retriever). |
| `docs/knowledge-base-comparison-weknora-vs-kawai.md` | Schema + pipeline comparison vs WeKnora (analytical, not how-to). |
| `docs/knowledge-r6-measurement.md` | R6 recall@K measurement procedure — how to build the eval set, run the benchmark, and the decision rule. |
| `API_MAP.md` | The ingest→embed→retrieve data flow across services. |

## File map

### This package (`egent-tools/knowledge/`)
| File | Purpose |
|---|---|
| `tools.go` | `KnowledgeSearchTool` (Eino tool) — parses args, scopes per-user, calls retriever, reranks, dedupes, formats. **The entrypoint: `InvokableRun` (`tools.go:88`).** Also holds `FormatResults`, `deduplicateByFile`, `filterByMinScore`. |
| `service.go` | `Service` — wires pgvector store + chunk store + keyword index + hybrid retriever; long-lived. Created at egent startup. |
| `service_test.go`, `tools_test.go` | Tool/service tests with `fakeService`/`fakeRetriever` (no DB). |

### Eval harness (`egent-tools/knowledge/eval/`)
| File | Purpose |
|---|---|
| `benchmark_test.go` | `TestBenchmark_R6Recall` — gated benchmark comparing single-query vs multi-query (R6) recall@K + MRR over a labeled dataset. Skips without `KAWAI_PG_DSN` + `EVAL_DATASET`. |
| `dataset.example.json` | Starter eval set format (~12 queries; extend to ~30 with real chunk IDs). |
| `seed/main.go` | Seeding tool: generates 20 files × 5 chunks with random embeddings + writes `dataset.json` with deterministic relevant-chunk IDs. `go run ./knowledge/eval/seed -dsn $KAWAI_PG_DSN -out dataset.json` |

### Sibling: `egent-tools/rerank/`
| File | Purpose |
|---|---|
| `reranker.go` | `Reranker` interface + `RerankerConfig`. |
| `nvidia_reranker.go` | NVIDIA `llama-nemotron-rerank-vl-1b-v2` impl (key from `NVIDIA_API_KEYS`, comma-split, random pick). |

### The retrieval engine (`fileprocessor/`)
| File | Purpose |
|---|---|
| `eino_retriever.go` | `Retriever` — the Eino `retriever.Retriever`. Fans out to vector + keyword sub-retrievers, weighted RRF fusion, parent-context expansion. **`NewRetriever` (`:56`), `Retrieve` (`:179`).** |
| `eino_retriever_router.go` | `vectorRetriever`, `keywordRetriever`, `WeightedRRFFusion`. |
| `rag_searcher.go` | Legacy `Searcher` (older, pre-Eino). Also hybrid-capable but **not** the production path — the `Service` uses `Retriever`, not `Searcher`. |
| `sqlite_keyword_index.go` | `SQLiteKeywordIndex` — local FTS5 BM25 projection. **Unwired** (library code); production uses PG FTS. |
| `public_embeddings_store.go` | `PublicEmbeddingsStore` — pgvector `VectorStore` over `public.embeddings`, **plus** the live PG-FTS `KeywordSearch` (`:307`). |
| `ragcore.go` | Public types: `VectorStore`, `ChunkStore`, `Embedder`, `KeywordSearcher`, `VectorMatch`, metadata keys. |

## How it's wired (the full call chain)

```
egent startup
  egent-public-apis/sharedtools.go:34  buildSharedTools(ctx)
    ├─ initSharedDBPool         ← KAWAI_PG_DSN (fallback KNOWLEDGE_PG_DSN)
    ├─ buildSharedEmbedder      ← OPENAI_EMBEDDINGS_URL / MODEL_BASE_URL, model text-embedding-3-small, 1024-dim
    │                              wrapped in billingEmbedder (usage.Record per call)
    ├─ buildSharedReranker      ← NVIDIA_API_KEYS  (nil if unset)
    ├─ knowledge.NewServiceWithRerank(ctx, pool, embedder, rerankModel)   ← service.go:128
    └─ knowledge.NewKnowledgeSearchTool(kSvc, rerankModel)                ← tools.go:58
```

## The two paths

### Write path (NOT in this package — `egent-jobs/` owns it)
```
AList upload (web → :5244)
  → egent-jobs River worker  (parse_file_to_chunks, embed_file_chunks)
  → FileProcessorChunker.Chunk  (egent-jobs/fileingest/)  — extract text → TextChunker
  → EmbedBatch  (egent-jobs/embeddings/) — 1024-dim, text-embedding-3-small
  → public.embeddings  (pgvector HNSW, cosine)   +   public.chunks / public.file_chunks (fts_vector)
```
Status ledger: `async_tasks` (read by the BFF via pREST). River rows live in `river_job`.

### Read path (`knowledge_search`, `tools.go:88`)
```
1. userID / tenantID / projectID  ← context (memory.UserIDFromContext …)
2. UserFileIDs                     ← SELECT id FROM public.files WHERE user_id AND tenant_id [AND project_id]
                                     (service.go:170 — the ONLY ACL; loads IDs into memory)
3. fp.Retriever.Retrieve(query, WithTopK, WithFileIDs, WithTenantID)
   ├── vectorRetriever   → embed query → PublicEmbeddingsStore.Search (HNSW, SET LOCAL app.tenant_id) → threshold(0.15) → hydrate
   ├── keywordRetriever  → PublicEmbeddingsStore.KeywordSearch (PG FTS BM25, SET LOCAL app.tenant_id)  → threshold(0.3)  → hydrate
   └── WeightedRRFFusion (vector 0.7, keyword 0.3, K=60)
4. (optional) rerankResults   → NVIDIA rerank → Eino score transformer
5. filterByMinScore (default 0.3) → deduplicateByFile (word-Jaccard > 0.8) → ExpandParent → FormatResults
   (parent expansion runs LAST, after dedupe — fixes architecture-review B1)
```

**R3 enforcement:** `SearchParams.TenantID` is set by `withTenantGuard`, which wraps
each sub-retriever's call in a PG transaction with `SET LOCAL app.tenant_id = $1`.
If the env var is empty, the guard is a no-op (safe for dev/CI without RLS). When
the egent runs as `kawai_kb_reader` (NOBYPASSSRLS), RLS policies on
`public.embeddings` and `public.chunks` enforce tenant isolation at the DB level.
The in-memory ACL (`UserFileIDs` → `file_id = ANY(...)`) remains as a secondary
guard.

## The keyword index — Postgres FTS (was: dual SQLite + PG)

Previously this subsystem ran **two** BM25 implementations, with a local SQLite
FTS5 file rebuilt every 30s shadowing an authoritative Postgres FTS index. That
duplication is gone (architecture review R2). Keyword search now runs against
**Postgres full-text search** exclusively:

| Path | Status |
|---|---|
| `PublicEmbeddingsStore.KeywordSearch` (`public_embeddings_store.go:307`) — `ts_rank_cd` over `chunks.fts_vector` | **LIVE.** Auto-wired: `NewRetriever` detects that `PublicEmbeddingsStore` implements `KeywordSearcher` when `RetrieverConfig.Keyword` is nil (`eino_retriever.go:80`). `fts_vector` is a STORED generated column (migration `20260802000001`), always in sync with chunk text — no rebuild, no local file, no env var. |
| `SQLiteKeywordIndex` (`fileprocessor/sqlite_keyword_index.go`) — FTS5 BM25 | **Unwired** (kept as library code). No longer constructed by `Service`; the 30s `refreshKeywordIndex` goroutine and `KAWAI_KNOWLEDGE_FTS_PATH` are gone. |

If you change keyword behavior, you're now changing `PublicEmbeddingsStore.KeywordSearch`.

## Key interfaces / extension points

```go
// egent-tools/knowledge — the tool's view of the service (tools.go:35)
type KnowledgeBackend interface {
    UserFileIDs(ctx, userID, tenantID, projectID string) ([]string, error)
    Retriever() retriever.Retriever
    Rerank(ctx, query string, documents []*schema.Document) ([]*schema.Document, error)
}

// fileprocessor — the pluggable stores (ragcore.go)
type VectorStore interface { /* Upsert, UpsertBatch, Search, DeleteByID, DeleteByFileID, Close */ }
type KeywordSearcher interface { KeywordSearch(ctx, query, SearchParams) ([]VectorMatch, error) }  // optional
type Embedder interface { Embed(ctx, texts) ([][]float32, error); Dimension() int }
type ChunkStore interface { GetDocument, CreateChunk, GetChunksByIDs, GetFile, UpdateFileChunkStats }
```

- A `VectorStore` that also implements `KeywordSearcher` is auto-used for keyword
  search **unless** an explicit `Keyword` is passed to `RetrieverConfig`.
- Per-call retrieval options: `fp.WithFileIDs`, `fp.WithTopK` (via `retriever`),
  `fp.WithVectorThreshold`, `fp.WithKeywordThreshold`, `fp.WithMetric`,
  `fp.WithExpandParent`.

## Config / env vars

| Var | Default | Effect |
|---|---|---|
| `KAWAI_PG_DSN` | — | kawai DB pool. Unset → no knowledge tools. (fallback: `KNOWLEDGE_PG_DSN`) |
| `OPENAI_EMBEDDINGS_URL` | `MODEL_BASE_URL`+`/embeddings` | Embeddings endpoint. Unset → search skipped. |
| `OPENAI_EMBEDDINGS_MODEL` | `text-embedding-3-small` | Embedding model. |
| `OPENAI_API_KEY` / `MODEL_API_KEY` | — | Embeddings auth. |
| `NVIDIA_API_KEYS` | — | Comma-separated rerank keys. Unset → no rerank (silent). |
| `KAWAI_KNOWLEDGE_QUERY_REWRITE` | unset | `1`/`true`/`on` enables LLM query rewriting + multi-query RRF (R6). Also needs `MODEL_BASE_URL` + `MODEL_NAME`. **Default off** — measure recall@K before enabling. |
| `EVAL_DATASET` | — | Path to the JSON eval dataset for `TestBenchmark_R6Recall`. Unset → benchmark skips. |

## Gotchas

- **✅ B1 — fixed.** Parent-context expansion now runs *after* per-file dedupe (`tools.go` calls `svc.ExpandParent` last), so sibling chunks sharing a parent are no longer collapsed. Guarded by `TestKnowledgeSearch_DoesNotCollapseSiblingSections`. The expansion logic is exported as `fp.ExpandParentContext` (`fileprocessor/eino_retriever.go`); the in-`Retrieve` path is disabled (`ExpandParent: false` in `RetrieverConfig`).
- **✅ R3 — active.** `SearchParams.TenantID` is set by `withTenantGuard` on every sub-retriever call; `SET LOCAL app.tenant_id` is applied inside a PG transaction. When the egent connects as `kawai_kb_reader` (NOBYPASSSRLS), RLS policies enforce tenant isolation at the DB level. The in-memory ACL (`UserFileIDs`) is still the primary filter and loads into memory; the RLS guard is the safety net.
- **Keyword index is PG FTS now** — see above. The SQLite index is unwired (library code only). Editing `SQLiteKeywordIndex` has no production effect.
- **1024-dim is pinned** in 4 places (`service.go:51`,
  `public_embeddings_store.go:55/174/251`) + the schema `vector(1024)`. Changing
  embedding model/dim = migration + full re-embed, no coexistence.
- **Rerank degrades observably now** (`tools.go`): span attrs `rerank.configured`/`rerank.applied`/`rerank.failed` are set, and a note is prepended to the tool output when rerank was configured but failed (R7).
- **Query rewriting is opt-in (R6).** `LLMQueryRewriter` + `SetQueryRewriter`; enabled via `KAWAI_KNOWLEDGE_QUERY_REWRITE`. When on, the tool retrieves per rewrite in parallel with the original and fuses with RRF (1 extra LLM call + N× retrieval latency). Default off — measure recall@K before enabling.
- **ACL is two-layer: in-memory allowlist + RLS.** `UserFileIDs` (`service.go:170`) loads file IDs then passes `file_id = ANY(...)` into HNSW (primary filter). `withTenantGuard` sets `app.tenant_id` per transaction so RLS policies also filter (safety net). When egent runs as `kawai_kb_reader` (NOBYPASSSRLS), both layers are live. When running as `postgres`, the RLS layer is bypassed but the in-memory ACL still applies.
- **`deduplicateByFile` is O(n²) per file** (`tools.go:292`) — fine at limit ≤50.
- **`embeddings.model` is always `"fileprocessor"`** regardless of real model
  (`public_embeddings_store.go:44` `modelTag`) — blocks future model migration.
- **`Searcher` (rag_searcher.go) is legacy.** Production uses `Retriever`
  (eino_retriever.go). Don't wire new behavior into `Searcher`.
- **Egents ignore SIGTERM** — use `planoctl down` / `kill -9` when iterating.

## How to make changes

**Edit the retrieval pipeline (fusion weights, thresholds, expansion):**
`fileprocessor/eino_retriever.go` + `eino_retriever_router.go`. Constants
(`defaultVectorWeight 0.7`, `defaultKeywordWeight 0.3`, `defaultRRFK 60`,
thresholds) live in `rag_searcher.go:12-33`.

**Change keyword search:** edit `PublicEmbeddingsStore.KeywordSearch` (`fileprocessor/public_embeddings_store.go:307`) — the live PG-FTS path. The SQLite index (`sqlite_keyword_index.go`) is unwired library code; changing it has no production effect.

**Change the tool's post-processing (rerank, dedupe, min-score, formatting):**
`egent-tools/knowledge/tools.go` (`InvokableRun`).

**Add a new rerank provider:** add to the `switch` in `rerank.NewReranker`
(`reranker.go:93`) + implement `Reranker`.

**Change embedding model:** update `OPENAI_EMBEDDINGS_MODEL` callers + the
4 dim-1024 sites + a `vector(N)` migration + full re-embed. Single-source the
dim first (architecture review R5).

## Build / test

This is a library — build & test in-module (it has its own `go.mod`), then recompile the consuming egent:
```bash
# from inside egent-tools/ (its own module)
go test ./knowledge/... ./rerank/...
go vet   ./knowledge/... ./rerank/...

# run R6 recall@K benchmark (requires KAWAI_PG_DSN + EVAL_DATASET + embeddings endpoint)
# first, seed synthetic data + generate eval dataset:
go run ./knowledge/eval/seed -dsn "$KAWAI_PG_DSN" -out knowledge/eval/dataset.json
EVAL_DATASET=knowledge/eval/dataset.json go test -v ./knowledge/eval/ -run TestBenchmark_R6Recall

# exercise end-to-end (recompiles egent-public-apis from source)
./planoctl down && ./planoctl up
# direct probe of the knowledge agent (default on :10507)
curl -N -d '{"model":"kawai-pro-max","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"search my files for X"}]}]}' http://localhost:10507/v1/chat/completions
```
fileprocessor integration tests need `FILEPROCESSOR_TEST_PG_DSN` + `-tags=integration`.

## DB schema (the lobehub `public` tables this reads)

| Table | Role |
|---|---|
| `public.files` | File records (`user_id`, `tenant_id`, `project_id`) — ACL source |
| `public.chunks` | Chunk text + `fts_vector` (TSVECTOR, GIN-indexed, **used by PG-FTS keyword search**) |
| `public.file_chunks` | chunk ↔ file link (hydrates `file_id` on results) |
| `public.embeddings` | `vector(1024)` + HNSW `public_embeddings_hnsw_idx` (`vector_cosine_ops`). `chunk_id` UNIQUE, FK→`chunks.id` `ON DELETE CASCADE` |

Live schema = `db/migrations/*.sql`. Query the DB directly (`psql "\d <table>"`)
— `db/lobehub-schema.sql` is a stale snapshot.
