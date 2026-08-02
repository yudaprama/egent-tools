package memory

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

// mockExtractor returns canned facts for testing.
type mockExtractor struct {
	facts map[string]string
	err   error
}

func (e *mockExtractor) Extract(_ context.Context, _ string) (map[string]string, error) {
	return e.facts, e.err
}

func TestExtractAndStoreReturnsExtractionError(t *testing.T) {
	mgr := NewManager(NoopStore{}).WithExtractor(&mockExtractor{
		err: errors.New("extractor unavailable"),
	})

	err := mgr.ExtractAndStore(context.Background(), "t1", "u1", "s1", "hello")
	if err == nil || !strings.Contains(err.Error(), "extractor unavailable") {
		t.Fatalf("expected extraction error to be returned, got %v", err)
	}
}

// batchRecordingStore embeds NoopStore (satisfying Store) and records
// SetBatch calls so async extraction can be verified deterministically.
type batchRecordingStore struct {
	NoopStore
	mu  sync.Mutex
	got []map[string]string
	ch  chan struct{}
}

func (s *batchRecordingStore) SetBatch(ctx context.Context, tenantID, userID, sessionID string, entries map[string]string) error {
	s.mu.Lock()
	s.got = append(s.got, entries)
	s.mu.Unlock()
	if s.ch != nil {
		select {
		case s.ch <- struct{}{}:
		default:
		}
	}
	return nil
}

func (s *batchRecordingStore) batches() []map[string]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]map[string]string(nil), s.got...)
}

// waitForBatch blocks until the store records a SetBatch or the timeout hits.
func waitForBatch(t *testing.T, s *batchRecordingStore, d time.Duration) []map[string]string {
	t.Helper()
	select {
	case <-s.ch:
		return s.batches()
	case <-time.After(d):
		t.Fatalf("timed out waiting for async batch write")
		return nil
	}
}

func TestExtractAndStoreAsync_WritesExtractedFacts(t *testing.T) {
	store := &batchRecordingStore{ch: make(chan struct{}, 1)}
	mgr := NewManager(store).WithExtractor(&mockExtractor{
		facts: map[string]string{
			"user.name":        "Budi",
			"user.location":    "Bandung",
			"preferences.kopi": "kopi",
		},
	})

	memID := Identity{TenantID: "t1", UserID: "u1"}
	mgr.ExtractAndStoreAsync(context.Background(), memID, "I'm Budi. I live in Bandung. I like kopi.")

	batches := waitForBatch(t, store, 2*time.Second)
	if len(batches) != 1 {
		t.Fatalf("want 1 batch, got %d", len(batches))
	}
	got := batches[0]
	for key, want := range map[string]string{
		"user.name":        "Budi",
		"user.location":    "Bandung",
		"preferences.kopi": "kopi",
	} {
		if v := got[key]; v != want {
			t.Errorf("fact %q: want %q, got %q", key, want, v)
		}
	}
}

func TestExtractAndStoreAsync_NoopWhenEmptyText(t *testing.T) {
	store := &batchRecordingStore{ch: make(chan struct{}, 1)}
	mgr := NewManager(store).WithExtractor(&mockExtractor{
		facts: map[string]string{"k": "v"},
	})

	mgr.ExtractAndStoreAsync(context.Background(), Identity{TenantID: "t1", UserID: "u1"}, "")

	select {
	case <-store.ch:
		t.Fatalf("expected no batch write for empty text")
	case <-time.After(100 * time.Millisecond):
	}
	if got := store.batches(); len(got) != 0 {
		t.Fatalf("expected no writes, got %v", got)
	}
}

func TestExtractAndStoreAsync_NoopWhenIdentityIncomplete(t *testing.T) {
	store := &batchRecordingStore{ch: make(chan struct{}, 1)}
	mgr := NewManager(store).WithExtractor(&mockExtractor{
		facts: map[string]string{"k": "v"},
	})

	mgr.ExtractAndStoreAsync(context.Background(), Identity{TenantID: "", UserID: "u1"}, "I like kopi")
	mgr.ExtractAndStoreAsync(context.Background(), Identity{TenantID: "t1", UserID: ""}, "I like kopi")

	select {
	case <-store.ch:
		t.Fatalf("expected no batch write for incomplete identity")
	case <-time.After(100 * time.Millisecond):
	}
	if got := store.batches(); len(got) != 0 {
		t.Fatalf("expected no writes, got %v", got)
	}
}

func TestExtractAndStoreAsync_NoopWhenNoFactsExtracted(t *testing.T) {
	store := &batchRecordingStore{ch: make(chan struct{}, 1)}
	mgr := NewManager(store).WithExtractor(&mockExtractor{
		facts: map[string]string{},
	})

	// Mock extractor returns nothing, so no store write should occur.
	mgr.ExtractAndStoreAsync(context.Background(), Identity{TenantID: "t1", UserID: "u1"}, "What is the weather today?")

	select {
	case <-store.ch:
		t.Fatalf("expected no batch write when no facts extracted")
	case <-time.After(100 * time.Millisecond):
	}
	if got := store.batches(); len(got) != 0 {
		t.Fatalf("expected no writes, got %v", got)
	}
}
