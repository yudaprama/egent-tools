package embedding

import (
	"context"
	"testing"

	fp "github.com/kawai-network/fileprocessor"
	"github.com/yudaprama/egent-tools/memory"
)

func TestDJB2_Distribution(t *testing.T) {
	// Test that DJB2 produces reasonably uniform distribution across providers.
	providers := 3
	counts := make([]int, providers)
	n := 10000

	for i := 0; i < n; i++ {
		id := generateTestWorkspaceID(i)
		idx := int(djb2(id)) % providers
		counts[idx]++
	}

	// Check that no provider gets more than 50% or less than 15% of workspaces.
	// A perfectly uniform distribution would be 33.3%, but with DJB2 and
	// structured test IDs, we accept some variance.
	for i, count := range counts {
		percentage := float64(count) / float64(n) * 100
		if percentage < 15 || percentage > 50 {
			t.Errorf("provider %d got %.1f%% of workspaces (expected 15-50%%)", i, percentage)
		}
	}
	t.Logf("distribution: %v", counts)
}

func TestDJB2_Deterministic(t *testing.T) {
	id := "550e8400-e29b-41d4-a716-446655440000"
	h1 := djb2(id)
	h2 := djb2(id)
	if h1 != h2 {
		t.Errorf("djb2 not deterministic: %d != %d", h1, h2)
	}
}

func TestTenantAwareEmbedder(t *testing.T) {
	// Create mock embedders that return different values.
	provider0 := &mockEmbedder{dim: 1024, providerIdx: 0}
	provider1 := &mockEmbedder{dim: 1024, providerIdx: 1}
	provider2 := &mockEmbedder{dim: 1024, providerIdx: 2}

	ta := NewTenantAwareEmbedder([]fp.Embedder{provider0, provider1, provider2})

	// Test that same tenant always gets same provider.
	idx1 := ta.providerIndex("tenant-abc")
	idx2 := ta.providerIndex("tenant-abc")
	if idx1 != idx2 {
		t.Errorf("provider index not deterministic: %d != %d", idx1, idx2)
	}

	// Test dimension.
	if ta.Dimension() != 1024 {
		t.Errorf("expected dimension 1024, got %d", ta.Dimension())
	}
}

func TestTenantAwareEmbedder_WithTenant(t *testing.T) {
	provider0 := &mockEmbedder{dim: 1024, providerIdx: 0}
	provider1 := &mockEmbedder{dim: 1024, providerIdx: 1}

	ta := NewTenantAwareEmbedder([]fp.Embedder{provider0, provider1})

	// With tenant ID, should use the deterministically selected provider.
	ctx := memory.WithTenantID(context.Background(), "test-tenant-xyz")
	vecs, err := ta.Embed(ctx, []string{"hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(vecs))
	}
}

func TestTenantAwareEmbedder_NoTenant(t *testing.T) {
	provider0 := &mockEmbedder{dim: 1024, providerIdx: 0}
	provider1 := &mockEmbedder{dim: 1024, providerIdx: 1}

	ta := NewTenantAwareEmbedder([]fp.Embedder{provider0, provider1})

	// Without tenant ID, should use first provider.
	ctx := context.Background()
	vecs, err := ta.Embed(ctx, []string{"hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(vecs))
	}
	// First element should be 0 (providerIdx of first provider).
	if vecs[0][0] != 0 {
		t.Errorf("expected provider 0, got provider %v", vecs[0][0])
	}
}

// generateTestWorkspaceID generates a deterministic test workspace ID.
// Uses UUID-like format to simulate real workspace IDs.
func generateTestWorkspaceID(i int) string {
	// Generate varied IDs using different patterns.
	hex := "0123456789abcdef"
	a := hex[i%16]
	b := hex[(i/16)%16]
	c := hex[(i/256)%16]
	d := hex[(i/4096)%16]
	return "ws-" + string(a) + string(b) + string(c) + string(d) + "-0000-0000-0000-000000000000"
}

// mockEmbedder is a test double that records which provider was called.
type mockEmbedder struct {
	dim         int
	providerIdx int
}

func (m *mockEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i := range texts {
		result[i] = make([]float32, m.dim)
		result[i][0] = float32(m.providerIdx)
	}
	return result, nil
}

func (m *mockEmbedder) Dimension() int {
	return m.dim
}
