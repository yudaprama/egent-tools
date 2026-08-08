package embedding

import (
	"context"
	"log/slog"

	fp "github.com/kawai-network/fileprocessor"
	"github.com/yudaprama/egent-tools/memory"
)

// modelNamer is an optional interface that concrete provider types satisfy.
type modelNamer interface{ Model() string }

// TenantAwareEmbedder selects an embedding provider based on the tenant
// (workspace) ID from context using DJB2 hash. Each workspace is permanently
// assigned to one provider, ensuring all embeddings within a workspace use
// the same model — preventing cross-model embedding space contamination.
//
// On provider failure, it falls back to the next provider in the list.
type TenantAwareEmbedder struct {
	providers []fp.Embedder
	dim       int
}

// NewTenantAwareEmbedder creates a tenant-aware embedder from a list of
// providers. All providers MUST produce the same dimension (1024).
// The provider list is indexed by DJB2(tenantID) % len(providers).
func NewTenantAwareEmbedder(providers []fp.Embedder) *TenantAwareEmbedder {
	if len(providers) == 0 {
		panic("embedding: NewTenantAwareEmbedder requires at least one provider")
	}
	dim := providers[0].Dimension()
	return &TenantAwareEmbedder{
		providers: providers,
		dim:       dim,
	}
}

// providerIndex returns the deterministic provider index for a tenant ID.
func (t *TenantAwareEmbedder) providerIndex(tenantID string) int {
	return int(djb2(tenantID)) % len(t.providers)
}

// Embed embeds texts using the provider assigned to the current workspace.
// Falls back to the next provider on failure.
func (t *TenantAwareEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	tenantID := memory.TenantIDFromContext(ctx)
	if tenantID == "" {
		// No tenant context — use first provider (backward compat).
		return t.providers[0].Embed(ctx, texts)
	}

	idx := t.providerIndex(tenantID)

	// Try the selected provider, then fallback chain.
	for attempt := 0; attempt < len(t.providers); attempt++ {
		pidx := (idx + attempt) % len(t.providers)
		provider := t.providers[pidx]

		vecs, err := provider.Embed(ctx, texts)
		if err == nil {
			if attempt > 0 {
				slog.Warn("embedding: fallback provider used",
					"tenant_id", tenantID,
					"selected", idx,
					"used", pidx,
					"attempt", attempt)
			}
			return vecs, nil
		}

		slog.Warn("embedding: provider failed, trying next",
			"tenant_id", tenantID,
			"provider_index", pidx,
			"error", err)
	}

	// All providers failed — try the first one and return its error.
	return t.providers[0].Embed(ctx, texts)
}

// Dimension returns the embedding dimension (same for all providers).
func (t *TenantAwareEmbedder) Dimension() int {
	return t.dim
}

// ModelForTenant returns the provider model name for the given tenant.
// Falls back to the first provider when tenant ID is empty (backward compat).
func (t *TenantAwareEmbedder) ModelForTenant(tenantID string) string {
	idx := 0
	if tenantID != "" {
		idx = t.providerIndex(tenantID)
	}
	if n, ok := t.providers[idx].(modelNamer); ok {
		return n.Model()
	}
	return ""
}

// SetBillingHook propagates a billing hook to all wrapped billingEmbedders.
func (t *TenantAwareEmbedder) SetBillingHook(hook BillingHook) {
	for _, p := range t.providers {
		if be, ok := p.(*billingEmbedder); ok {
			be.SetBillingHook(hook)
		}
	}
}
