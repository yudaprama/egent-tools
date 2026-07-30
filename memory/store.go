package memory

import (
	"context"
	"time"
)

// MemoryEntry represents a single stored memory fact.
type MemoryEntry struct {
	Key       string    `json:"key"`
	Value     string    `json:"value"`
	UserID    string    `json:"userID"`
	SessionID string    `json:"sessionID"`
	CreatedAt time.Time `json:"createdAt"`
	UpdatedAt time.Time `json:"updatedAt"`
}

// Store is the memory persistence interface.
// tenantID is used as the vault name (data isolation boundary).
// userID and sessionID are stored as tags on each engram for filtering.
type Store interface {
	Set(ctx context.Context, tenantID, userID, sessionID, key, value string) error
	Get(ctx context.Context, tenantID, userID, sessionID, key string) (*MemoryEntry, error)
	Delete(ctx context.Context, tenantID, userID, sessionID, key string) error
	Search(ctx context.Context, tenantID, query string, limit int, opts ...SearchOption) ([]MemoryEntry, error)
	List(ctx context.Context, tenantID string, opts ...SearchOption) ([]MemoryEntry, error)
	// SaveTurn persists a raw conversation turn: question becomes the concept,
	// answer becomes the content, tagged with [user:<userID>, session:<sessionID>].
	// Unlike Set (key→concept), SaveTurn stores the verbatim Q&A pair so it can
	// be recalled as conversation history rather than an extracted fact.
	SaveTurn(ctx context.Context, tenantID, userID, sessionID, question, answer string) error
}

// SearchFilter carries optional filter parameters for Search and List.
// When both are empty, results span the entire tenant (cross-user).
type SearchFilter struct {
	UserID    string
	SessionID string
}

// SearchOption configures a SearchFilter.
type SearchOption func(*SearchFilter)

// ByUserID filters results to memories belonging to a specific user.
func ByUserID(uid string) SearchOption {
	return func(f *SearchFilter) { f.UserID = uid }
}

// BySessionID filters results to memories belonging to a specific session.
func BySessionID(sid string) SearchOption {
	return func(f *SearchFilter) { f.SessionID = sid }
}

// NoopStore is a stateless Store that persists nothing and recalls nothing.
// It is the MUNINN_URL-unset fallback so the binary can boot in environments
// that don't run MuninnDB (dev/CI). Every consumer degrades gracefully.
type NoopStore struct{}

var _ Store = NoopStore{}

func (NoopStore) Set(_ context.Context, _, _, _, _, _ string) error { return nil }

func (NoopStore) Get(_ context.Context, _, _, _, _ string) (*MemoryEntry, error) {
	return nil, nil
}

func (NoopStore) Delete(_ context.Context, _, _, _, _ string) error { return nil }

func (NoopStore) Search(_ context.Context, _ string, _ string, _ int, _ ...SearchOption) ([]MemoryEntry, error) {
	return nil, nil
}

func (NoopStore) List(_ context.Context, _ string, _ ...SearchOption) ([]MemoryEntry, error) {
	return nil, nil
}

func (NoopStore) SaveTurn(_ context.Context, _, _, _, _, _ string) error { return nil }
