package memory

import (
	"context"
	"testing"
)

func TestNoopStore_Contract(t *testing.T) {
	ctx := context.Background()
	s := NoopStore{}

	if err := s.Set(ctx, "t1", "u1", "s1", "k", "v"); err != nil {
		t.Fatalf("Set: want nil, got %v", err)
	}
	if err := s.Delete(ctx, "t1", "u1", "s1", "k"); err != nil {
		t.Fatalf("Delete: want nil, got %v", err)
	}

	if entry, err := s.Get(ctx, "t1", "u1", "s1", "k"); err != nil || entry != nil {
		t.Errorf("Get: want (nil,nil); got (%+v,%v)", entry, err)
	}
	if entries, err := s.Search(ctx, "t1", "q", 10, ByUserID("u1"), BySessionID("s1")); err != nil || len(entries) != 0 {
		t.Errorf("Search: want (empty,nil); got (%v,%v)", entries, err)
	}
	if entries, err := s.List(ctx, "t1", ByUserID("u1"), BySessionID("s1")); err != nil || len(entries) != 0 {
		t.Errorf("List: want (empty,nil); got (%v,%v)", entries, err)
	}
}

func TestNoopStore_ManagerRecallNoop(t *testing.T) {
	mgr := NewManager(NoopStore{})
	if got := mgr.Recall(context.Background(), "t1", "u1", "s1", "anything"); got != "" {
		t.Errorf("Recall via NoopStore: want empty, got %q", got)
	}
	tools := mgr.AllTools()
	if len(tools) == 0 {
		t.Fatal("AllTools: expected tools, got none")
	}
}
