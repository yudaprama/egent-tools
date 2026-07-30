package memory

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"
)

// Manager coordinates memory extraction, persistence, and retrieval
// during an agent conversation. Mirrors LobeHub's UserMemory service
// at the orchestration level (not the DB-level persona service).
//
// Usage:
//
//	mgr := NewManager(store)
//	// During conversation:
//	mgr.ExtractAndStore(ctx, tenantID, userID, sessionID, conversationText)
//	memories := mgr.Recall(ctx, tenantID, userID, sessionID, userQuery)
//	// memories is injected into the agent's system prompt.
type Manager struct {
	store     Store
	cognitive CognitiveStore
	batch     BatchStore
	extractor Extractor
	mu        sync.RWMutex
	OnExtract func(tenantID, userID string, factCount int)
	OnRecall  func(tenantID, userID string, query string, resultCount int)
}

// BatchStore is an optional interface that Store implementations may
// satisfy when they can persist multiple facts in one request.
type BatchStore interface {
	SetBatch(ctx context.Context, tenantID, userID, sessionID string, entries map[string]string) error
}

// NewManager creates a memory manager backed by the given store.
func NewManager(store Store) *Manager {
	mgr := &Manager{
		store:     store,
		extractor: NewHeuristicExtractor(),
	}
	if cognitive, ok := store.(CognitiveStore); ok {
		mgr.cognitive = cognitive
	}
	if batch, ok := store.(BatchStore); ok {
		mgr.batch = batch
	}
	return mgr
}

// WithExtractor sets the extractor used by ExtractAndStore.
func (m *Manager) WithExtractor(e Extractor) *Manager {
	m.extractor = e
	return m
}

// Recall retrieves relevant memories for a query scoped to a tenant/user/session.
// Returns a formatted string suitable for injection into a system prompt.
// Returns empty string if no memories or store error.
func (m *Manager) Recall(ctx context.Context, tenantID, userID, sessionID, query string) string {
	if m.cognitive != nil {
		activated, err := m.cognitive.Activate(ctx, tenantID, userID, sessionID, []string{query}, 10)
		if err == nil && len(activated) > 0 {
			if m.OnRecall != nil {
				m.OnRecall(tenantID, userID, query, len(activated))
			}
			return FormatActivatedMemories(activated)
		}
		if err != nil {
			slog.Warn("memory activate failed, falling back to search", "error", err)
		}
	}

	entries, err := m.store.Search(ctx, tenantID, query, 10, ByUserID(userID), BySessionID(sessionID))
	if err != nil {
		slog.Warn("memory recall failed", "error", err)
		return ""
	}
	if m.OnRecall != nil {
		m.OnRecall(tenantID, userID, query, len(entries))
	}
	if len(entries) == 0 {
		return ""
	}
	return FormatMemories(entries)
}

// FormatMemories renders memory entries as a context block.
func FormatMemories(entries []MemoryEntry) string {
	if len(entries) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("[User Memory]\n")
	for _, e := range entries {
		fmt.Fprintf(&b, "- %s: %s\n", e.Key, e.Value)
	}
	return b.String()
}

// ExtractAndStore uses the configured extractor to pull facts from a user
// message and stores them in the backing store, scoped to the given
// tenant/user/session.
func (m *Manager) ExtractAndStore(ctx context.Context, tenantID, userID, sessionID, text string) error {
	facts, err := m.extractor.Extract(ctx, text)
	if err != nil {
		slog.Warn("memory extraction failed", "error", err)
		return nil
	}
	if len(facts) == 0 {
		return nil
	}
	stored := len(facts)
	if m.batch != nil {
		if err := m.batch.SetBatch(ctx, tenantID, userID, sessionID, facts); err != nil {
			slog.Warn("memory batch store failed", "error", err)
			stored = 0
		}
	} else {
		for key, value := range facts {
			if err := m.store.Set(ctx, tenantID, userID, sessionID, key, value); err != nil {
				slog.Warn("memory store failed", "key", key, "error", err)
				stored--
				continue
			}
		}
	}
	if m.OnExtract != nil && stored > 0 {
		m.OnExtract(tenantID, userID, stored)
	}
	return nil
}

// PurgeOlderThan removes entries older than the given duration within a tenant.
func (m *Manager) PurgeOlderThan(ctx context.Context, tenantID string, age time.Duration) (int, error) {
	entries, err := m.store.List(ctx, tenantID)
	if err != nil {
		return 0, err
	}
	cutoff := time.Now().Add(-age)
	removed := 0
	for _, e := range entries {
		if e.UpdatedAt.Before(cutoff) {
			if err := m.store.Delete(ctx, tenantID, e.UserID, e.SessionID, e.Key); err != nil {
				continue
			}
			removed++
		}
	}
	return removed, nil
}

// ErrStoreUnavailable is returned when the memory store is not initialized.
var ErrStoreUnavailable = errors.New("memory store not available")
