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
//	mgr.ExtractAndStore(ctx, userID, conversationText)
//	memories := mgr.Recall(ctx, userID, userQuery)
//	// memories is injected into the agent's system prompt.
type Manager struct {
	store     Store
	cognitive CognitiveStore
	batch     BatchStore
	mu        sync.RWMutex
	// Optional hooks for observability (Phase 4: tracing).
	OnExtract func(userID string, factCount int)
	OnRecall  func(userID string, query string, resultCount int)
}

// BatchStore is an optional interface that Store implementations may
// satisfy when they can persist multiple facts in one request.
type BatchStore interface {
	SetBatch(ctx context.Context, userID string, entries map[string]string) error
}

// NewManager creates a memory manager backed by the given store.
func NewManager(store Store) *Manager {
	mgr := &Manager{store: store}
	if cognitive, ok := store.(CognitiveStore); ok {
		mgr.cognitive = cognitive
	}
	if batch, ok := store.(BatchStore); ok {
		mgr.batch = batch
	}
	return mgr
}

// Recall retrieves relevant memories for a query.
// Returns a formatted string suitable for injection into a system prompt.
// Returns empty string if no memories or store error.
func (m *Manager) Recall(ctx context.Context, userID, query string) string {
	if m.cognitive != nil {
		activated, err := m.cognitive.Activate(ctx, userID, []string{query}, 10)
		if err == nil && len(activated) > 0 {
			if m.OnRecall != nil {
				m.OnRecall(userID, query, len(activated))
			}
			return FormatActivatedMemories(activated)
		}
		if err != nil {
			slog.Warn("memory activate failed, falling back to search", "error", err)
		}
	}

	entries, err := m.store.Search(ctx, userID, query, 10)
	if err != nil {
		slog.Warn("memory recall failed", "error", err)
		return ""
	}
	if m.OnRecall != nil {
		m.OnRecall(userID, query, len(entries))
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

// ExtractAndStore pulls simple facts from a user message and stores them.
// This is a lightweight heuristic extractor; the full LobeHub version uses
// an embedding model + LLM extraction. We provide:
//  1. "My name is X" → user.name = X
//  2. "I live in X" → user.location = X
//  3. "I like X" / "I prefer X" → preferences.X = true
func (m *Manager) ExtractAndStore(ctx context.Context, userID, text string) error {
	facts := extractHeuristic(text)
	if len(facts) == 0 {
		return nil
	}
	stored := len(facts)
	if m.batch != nil {
		if err := m.batch.SetBatch(ctx, userID, facts); err != nil {
			slog.Warn("memory batch store failed", "error", err)
			stored = 0
		}
	} else {
		for key, value := range facts {
			if err := m.store.Set(ctx, userID, key, value); err != nil {
				slog.Warn("memory store failed", "key", key, "error", err)
				stored--
				continue
			}
		}
	}
	if m.OnExtract != nil && stored > 0 {
		m.OnExtract(userID, stored)
	}
	return nil
}

// extractHeuristic returns key/value pairs derived from text.
// This is intentionally minimal; production systems should use an LLM.
func extractHeuristic(text string) map[string]string {
	lower := strings.ToLower(text)
	facts := map[string]string{}

	// Name patterns
	if idx := strings.Index(lower, "my name is "); idx >= 0 {
		rest := text[idx+len("my name is "):]
		name := firstWord(rest)
		if name != "" {
			facts["user.name"] = name
		}
	}
	if idx := strings.Index(lower, "i'm "); idx >= 0 {
		rest := text[idx+len("i'm "):]
		name := firstWord(rest)
		if isValidName(name) {
			facts["user.name"] = name
		}
	}

	// Location patterns
	if idx := strings.Index(lower, "i live in "); idx >= 0 {
		rest := text[idx+len("i live in "):]
		loc := firstPhrase(rest)
		if loc != "" {
			facts["user.location"] = loc
		}
	}
	if idx := strings.Index(lower, "i'm from "); idx >= 0 {
		rest := text[idx+len("i'm from "):]
		loc := firstPhrase(rest)
		if loc != "" {
			facts["user.location"] = loc
		}
	}

	// Preferences
	for _, prefix := range []string{"i like ", "i prefer ", "i love ", "i use "} {
		if idx := strings.Index(lower, prefix); idx >= 0 {
			rest := text[idx+len(prefix):]
			item := firstPhrase(rest)
			if item != "" {
				key := "preferences." + strings.ReplaceAll(strings.ToLower(item), " ", "_")
				facts[key] = item
			}
			break
		}
	}

	return facts
}

func firstWord(s string) string {
	s = strings.TrimSpace(s)
	for i, ch := range s {
		if ch == ' ' || ch == ',' || ch == '.' || ch == '!' || ch == '\n' {
			return s[:i]
		}
	}
	return s
}

func firstPhrase(s string) string {
	s = strings.TrimSpace(s)
	for i, ch := range s {
		if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
			return s[:i]
		}
	}
	// Cut at trailing comma if phrase looks complete
	if idx := strings.Index(s, ", "); idx > 0 {
		return s[:idx]
	}
	return s
}

func isValidName(s string) bool {
	if len(s) < 2 || len(s) > 30 {
		return false
	}
	// Reject common false positives
	switch strings.ToLower(s) {
	case "not", "sure", "sorry", "happy", "sad", "tired", "hungry", "going":
		return false
	}
	// Should start with a letter
	if !isAlpha(s[0]) {
		return false
	}
	return true
}

func isAlpha(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

// PurgeOlderThan removes entries older than the given duration.
// Useful for periodic cleanup.
func (m *Manager) PurgeOlderThan(ctx context.Context, userID string, age time.Duration) (int, error) {
	entries, err := m.store.List(ctx, userID)
	if err != nil {
		return 0, err
	}
	cutoff := time.Now().Add(-age)
	removed := 0
	for _, e := range entries {
		if e.UpdatedAt.Before(cutoff) {
			if err := m.store.Delete(ctx, userID, e.Key); err != nil {
				continue
			}
			removed++
		}
	}
	return removed, nil
}

// ErrStoreUnavailable is returned when the memory store is not initialized.
var ErrStoreUnavailable = errors.New("memory store not available")
