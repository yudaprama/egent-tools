package knowledge

import (
	"context"
	"testing"
)

func TestNewService_NilPool(t *testing.T) {
	svc, err := NewService(context.Background(), nil, nil)
	if err != nil {
		t.Fatalf("expected nil error for nil pool, got %v", err)
	}
	if svc != nil {
		t.Fatal("expected nil service for nil pool")
	}
}

func TestNewService_NilPoolShortCircuitsBeforeEmbedderCheck(t *testing.T) {
	// nil pool short-circuits before the embedder nil check.
	svc, err := NewService(context.Background(), nil, nil)
	if err != nil {
		t.Fatalf("nil pool should short-circuit, got %v", err)
	}
	if svc != nil {
		t.Fatal("expected nil service")
	}
}

func TestService_CloseIsNoop(t *testing.T) {
	svc := &Service{}
	if err := svc.Close(); err != nil {
		t.Fatalf("Close should be a no-op, got %v", err)
	}

	var nilSvc *Service
	if err := nilSvc.Close(); err != nil {
		t.Fatalf("Close on nil should be a no-op, got %v", err)
	}
}

func TestNewServiceWithSearcher(t *testing.T) {
	srch := &fakeSearcher{}
	svc := NewServiceWithSearcher(nil, srch)
	if svc == nil {
		t.Fatal("expected non-nil service")
	}
	if svc.Searcher() != srch {
		t.Error("expected searcher to match")
	}
	if svc.Pool() != nil {
		t.Error("expected nil pool")
	}
}
