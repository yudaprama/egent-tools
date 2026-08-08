package embedding

import (
	"context"

	fp "github.com/kawai-network/fileprocessor"
)

// BillingHook is called after each successful embedding call.
// Implementations should record usage (e.g. to plano-usage/Talos).
// If nil, billing is skipped.
type BillingHook func(ctx context.Context, model string)

// billingEmbedder wraps an fp.Embedder and calls a BillingHook after
// each successful embedding. Billing failures are logged but never
// block the embedding result.
type billingEmbedder struct {
	inner fp.Embedder
	model string
	hook  BillingHook
}

func newBillingEmbedder(inner fp.Embedder, model string) fp.Embedder {
	return &billingEmbedder{inner: inner, model: model}
}

// SetBillingHook sets the billing callback. Safe to call before first Embed.
func (b *billingEmbedder) SetBillingHook(hook BillingHook) {
	b.hook = hook
}

func (b *billingEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	vecs, err := b.inner.Embed(ctx, texts)
	if err != nil {
		return nil, err
	}
	if b.hook != nil {
		b.hook(ctx, b.model)
	}
	return vecs, nil
}

func (b *billingEmbedder) Dimension() int {
	return b.inner.Dimension()
}
