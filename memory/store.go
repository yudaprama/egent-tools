package memory

import (
	"context"
	"time"
)

// MemoryEntry represents a single stored memory fact.
type MemoryEntry struct {
	Key       string    `json:"key"`
	Value     string    `json:"value"`
	CreatedAt time.Time `json:"createdAt"`
	UpdatedAt time.Time `json:"updatedAt"`
}

// Store is the memory persistence interface.
// Maps to LobeHub's userMemory service: agents read/write user
// preferences, facts, and context during conversations.
type Store interface {
	// Set stores a memory entry for a user.
	Set(ctx context.Context, userID, key, value string) error
	// Get retrieves a specific memory entry.
	Get(ctx context.Context, userID, key string) (*MemoryEntry, error)
	// Delete removes a memory entry.
	Delete(ctx context.Context, userID, key string) error
	// Search finds memory entries matching a query.
	Search(ctx context.Context, userID, query string, limit int) ([]MemoryEntry, error)
	// List returns all memory entries for a user.
	List(ctx context.Context, userID string) ([]MemoryEntry, error)
}

// NoopStore is a stateless Store that persists nothing and recalls nothing.
// It is the MUNINN_URL-unset fallback so the binary can boot in environments
// that don't run MuninnDB (dev/CI). Every consumer degrades gracefully:
// Manager.Recall returns "" (no system-prompt injection) and the memory tools
// report "no memory found" / empty results instead of erroring. It deliberately
// does not implement CognitiveStore, so Manager skips the activation path.
//
// A MUNINN_URL that is set but points at an unreachable MuninnDB remains a
// fatal startup error (see main.go) — NoopStore only covers the *unset* case.
// Production should run MuninnDB (github.com/scrypster/muninndb) and set MUNINN_URL.
type NoopStore struct{}

// Compile-time guarantee that NoopStore satisfies Store.
var _ Store = NoopStore{}

func (NoopStore) Set(_ context.Context, _, _, _ string) error { return nil }

// Get returns (nil, nil) so the get-memory tool yields a friendly "no memory
// found" rather than surfacing an error to the agent — matching the
// behaviour callers expect for an absent key under degraded memory.
func (NoopStore) Get(_ context.Context, _, _ string) (*MemoryEntry, error) {
	return nil, nil
}

func (NoopStore) Delete(_ context.Context, _, _ string) error { return nil }

func (NoopStore) Search(_ context.Context, _ string, _ string, _ int) ([]MemoryEntry, error) {
	return nil, nil
}

func (NoopStore) List(_ context.Context, _ string) ([]MemoryEntry, error) {
	return nil, nil
}
