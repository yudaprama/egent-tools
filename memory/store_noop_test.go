package memory

import (
	"context"
	"testing"
)

// TestNoopStore_Contract verifies the NoopStore fallback (used when
// MUNINN_URL is unset) implements Store correctly and degrades gracefully:
// nothing persists, nothing recalls, and Get returns (nil, nil) so the
// get-memory tool yields a friendly "no memory found" instead of erroring.
func TestNoopStore_Contract(t *testing.T) {
	ctx := context.Background()
	s := NoopStore{}

	if err := s.Set(ctx, "u", "k", "v"); err != nil {
		t.Fatalf("Set: want nil, got %v", err)
	}
	if err := s.Delete(ctx, "u", "k"); err != nil {
		t.Fatalf("Delete: want nil, got %v", err)
	}

	if entry, err := s.Get(ctx, "u", "k"); err != nil || entry != nil {
		t.Errorf("Get: want (nil,nil); got (%+v,%v)", entry, err)
	}
	if entries, err := s.Search(ctx, "u", "q", 10); err != nil || len(entries) != 0 {
		t.Errorf("Search: want (empty,nil); got (%v,%v)", entries, err)
	}
	if entries, err := s.List(ctx, "u"); err != nil || len(entries) != 0 {
		t.Errorf("List: want (empty,nil); got (%v,%v)", entries, err)
	}
}

// TestNoopStore_ManagerRecallNoop verifies a Manager backed by NoopStore
// recalls nothing (returns "") and that AllTools does not panic — the
// behaviour the binary relies on when it boots without MUNINN_URL.
func TestNoopStore_ManagerRecallNoop(t *testing.T) {
	mgr := NewManager(NoopStore{})
	if got := mgr.Recall(context.Background(), "u", "anything"); got != "" {
		t.Errorf("Recall via NoopStore: want empty, got %q", got)
	}
	// The binary registers memory tools unconditionally on boot; ensure the
	// NoopStore-backed manager exposes them without panicking.
	tools := mgr.AllTools()
	if len(tools) == 0 {
		t.Fatal("AllTools: expected tools, got none")
	}
}
