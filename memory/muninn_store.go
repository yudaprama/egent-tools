package memory

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/scrypster/muninndb/sdk/go/muninn"
)

// MuninnStore is a memory.Store backed by MuninnDB — a cognitive database
// that strengthens memories with use (Hebbian learning), fades unused ones,
// and returns contextually relevant results via semantic activation scoring.
//
// Each user gets their own vault (namespace). Memories are stored as engrams
// with concept (key) + content (value). Search uses Activate() for
// context-aware retrieval instead of simple substring matching.
//
// An in-memory ID cache maps concept→engramID so that Get and Delete can
// call Read / Forget (exact operations) instead of Activate + filter.
// The cache is populated on Set and Delete and is safe for concurrent use.
//
// When MuninnDB is unavailable (not running, network error), all operations
// fail with an error — callers should treat these as fatal: the binary panics
// at startup if MuninnDB is unreachable, and there is no in-memory fallback.
type MuninnStore struct {
	client *muninn.Client

	// idCache stores known engram IDs for direct Read/Forget.
	// Structure: vault → concept → engramID.
	mu      sync.RWMutex
	idCache map[string]map[string]string
}

// CognitiveStore is an optional interface that Store implementations may
// satisfy when they support semantic/Hebbian memory activation. The Manager
// uses it to pick the richer retrieval path when available.
type CognitiveStore interface {
	Activate(ctx context.Context, userID string, ctxWords []string, limit int) ([]ActivatedMemory, error)
}

// NewMuninnStore creates a MuninnDB-backed memory store.
// baseURL is the MuninnDB HTTP endpoint (e.g. "http://localhost:8475").
// token is the API key (empty string for no auth).
func NewMuninnStore(baseURL, token string) *MuninnStore {
	return &MuninnStore{
		client:  muninn.NewClient(baseURL, token),
		idCache: make(map[string]map[string]string),
	}
}

// NewMuninnStoreFromClient creates a MuninnStore from an existing client.
func NewMuninnStoreFromClient(client *muninn.Client) *MuninnStore {
	return &MuninnStore{
		client:  client,
		idCache: make(map[string]map[string]string),
	}
}

// setIDCache records the engram ID for a concept in a vault.
func (s *MuninnStore) setIDCache(vault, concept, id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.idCache[vault] == nil {
		s.idCache[vault] = make(map[string]string)
	}
	s.idCache[vault][concept] = id
}

// lookupIDCache returns the cached engram ID for a concept, or "".
func (s *MuninnStore) lookupIDCache(vault, concept string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.idCache[vault] == nil {
		return ""
	}
	return s.idCache[vault][concept]
}

// deleteIDCache removes a concept from the ID cache.
func (s *MuninnStore) deleteIDCache(vault, concept string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.idCache[vault], concept)
}

// Set stores a memory as an engram. The userID becomes the vault name.
// The key becomes the concept, the value becomes the content.
// Tags include "memory" and the key prefix (e.g. "user", "preferences")
// for association building. The engram ID is cached for subsequent
// direct Read / Forget operations.
func (s *MuninnStore) Set(ctx context.Context, userID, key, value string) error {
	tags := tagsForKey(key)

	id, err := s.client.Write(ctx, userID, key, value, tags)
	if err != nil {
		return fmt.Errorf("muninn write: %w", err)
	}
	s.setIDCache(userID, key, id)
	return nil
}

// SetBatch stores multiple memories in a single MuninnDB batch call.
func (s *MuninnStore) SetBatch(ctx context.Context, userID string, entries map[string]string) error {
	if len(entries) == 0 {
		return nil
	}

	requests := make([]muninn.WriteRequest, 0, len(entries))
	keys := make([]string, 0, len(entries))
	for key, value := range entries {
		requests = append(requests, muninn.WriteRequest{
			Vault:      userID,
			Concept:    key,
			Content:    value,
			Tags:       tagsForKey(key),
			Confidence: 0.9,
			Stability:  0.5,
		})
		keys = append(keys, key)
	}

	for start := 0; start < len(requests); start += 50 {
		end := min(start+50, len(requests))
		resp, err := s.client.WriteBatch(ctx, userID, requests[start:end])
		if err != nil {
			return fmt.Errorf("muninn batch write: %w", err)
		}
		for _, result := range resp.Results {
			if result.Error != "" {
				return fmt.Errorf("muninn batch write item %d: %s", result.Index, result.Error)
			}
			if result.ID == "" {
				continue
			}
			s.setIDCache(userID, keys[start+result.Index], result.ID)
		}
	}
	return nil
}

// Get retrieves a specific memory by concept (key). It first tries the
// local ID cache for an exact engram Read, avoiding the cost and
// semantic uncertainty of an activation sweep. If the ID is not cached
// (e.g. after a server restart), it falls back to Activate and filters
// for an exact concept match.
//
// Returns (nil, nil) when no entry is found (consistent with the
// Store interface contract).
func (s *MuninnStore) Get(ctx context.Context, userID, key string) (*MemoryEntry, error) {
	if id := s.lookupIDCache(userID, key); id != "" {
		engram, err := s.client.Read(ctx, id, userID)
		if err != nil {
			return nil, fmt.Errorf("muninn read: %w", err)
		}
		return engramToEntryFromEngram(engram), nil
	}

	// Fallback: activate and filter for exact concept.
	resp, err := s.client.Activate(ctx, userID, []string{key}, 20)
	if err != nil {
		return nil, fmt.Errorf("muninn activate: %w", err)
	}
	for _, item := range resp.Activations {
		if item.Concept == key {
			s.setIDCache(userID, key, item.ID)
			return engramToEntryFromActivation(item), nil
		}
	}
	return nil, nil
}

// Delete forgets an engram by concept (key). It first tries the ID
// cache for a direct Forget, falling back to Activate when the ID is
// unknown. Returns ErrMemoryNotFound when neither path locates the
// engram (unlike the old implementation which silently returned nil).
func (s *MuninnStore) Delete(ctx context.Context, userID, key string) error {
	if id := s.lookupIDCache(userID, key); id != "" {
		if err := s.client.Forget(ctx, id, userID); err != nil {
			return fmt.Errorf("muninn forget: %w", err)
		}
		s.deleteIDCache(userID, key)
		return nil
	}

	// Fallback: activate to discover the engram ID.
	resp, err := s.client.Activate(ctx, userID, []string{key}, 20)
	if err != nil {
		return fmt.Errorf("muninn activate for delete: %w", err)
	}
	for _, item := range resp.Activations {
		if item.Concept == key {
			if err := s.client.Forget(ctx, item.ID, userID); err != nil {
				return fmt.Errorf("muninn forget: %w", err)
			}
			s.deleteIDCache(userID, key)
			return nil
		}
	}
	return ErrMemoryNotFound
}

// Search uses MuninnDB's Activate for context-aware retrieval. Unlike
// semantic ranking. Unlike a naive substring store, this returns memories
// ranked by MuninnDB's activation score (ACT-R base level + Hebbian boost).
// by relevance score — combining recency, frequency, and semantic
// similarity (Hebbian-weighted).
func (s *MuninnStore) Search(ctx context.Context, userID, query string, limit int) ([]MemoryEntry, error) {
	if limit <= 0 {
		limit = 10
	}

	resp, err := s.client.Activate(ctx, userID, []string{query}, limit)
	if err != nil {
		return nil, fmt.Errorf("muninn activate: %w", err)
	}

	entries := make([]MemoryEntry, 0, len(resp.Activations))
	for _, item := range resp.Activations {
		s.setIDCache(userID, item.Concept, item.ID)
		entries = append(entries, *engramToEntryFromActivation(item))
	}
	return entries, nil
}

// List returns all memories for a user via paginated ListEngrams.
func (s *MuninnStore) List(ctx context.Context, userID string) ([]MemoryEntry, error) {
	var all []MemoryEntry
	offset := 0
	pageSize := 100

	for {
		resp, err := s.client.ListEngrams(ctx, userID, pageSize, offset)
		if err != nil {
			return nil, fmt.Errorf("muninn list: %w", err)
		}

		for _, item := range resp.Engrams {
			s.setIDCache(userID, item.Concept, item.ID)
			all = append(all, MemoryEntry{
				Key:       item.Concept,
				Value:     item.Content,
				CreatedAt: unixToTime(item.CreatedAt),
				UpdatedAt: unixToTime(item.CreatedAt),
			})
		}

		if len(resp.Engrams) < pageSize || len(all) >= resp.Total {
			break
		}
		offset += pageSize
	}
	return all, nil
}

// Activate performs context-aware memory retrieval with scoring.
// This is the enhanced version of Search that exposes MuninnDB's
// activation scores and "why" explanations. Used by the Manager
// for injecting relevant memories into the system prompt.
//
// ctxWords replaces the former "context" parameter name to avoid
// shadowing Go's context.Context package.
func (s *MuninnStore) Activate(ctx context.Context, userID string, ctxWords []string, limit int) ([]ActivatedMemory, error) {
	if limit <= 0 {
		limit = 10
	}

	resp, err := s.client.Activate(ctx, userID, ctxWords, limit)
	if err != nil {
		return nil, fmt.Errorf("muninn activate: %w", err)
	}

	results := make([]ActivatedMemory, 0, len(resp.Activations))
	for _, item := range resp.Activations {
		s.setIDCache(userID, item.Concept, item.ID)
		why := ""
		if item.Why != nil {
			why = *item.Why
		}
		results = append(results, ActivatedMemory{
			Key:        item.Concept,
			Value:      item.Content,
			Score:      item.Score,
			Confidence: item.Confidence,
			Why:        why,
		})
	}
	return results, nil
}

// ActivatedMemory is a memory entry enriched with activation metadata.
type ActivatedMemory struct {
	Key        string  `json:"key"`
	Value      string  `json:"value"`
	Score      float64 `json:"score"`
	Confidence float64 `json:"confidence"`
	Why        string  `json:"why,omitempty"`
}

// FormatActivatedMemories renders activated memories as a context block
// with relevance scores. Higher-scored memories appear first.
func FormatActivatedMemories(memories []ActivatedMemory) string {
	if len(memories) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("[User Memory]\n")
	for _, m := range memories {
		fmt.Fprintf(&b, "- %s: %s [score=%.3f conf=%.2f]\n",
			m.Key, m.Value, m.Score, m.Confidence)
	}
	return b.String()
}

// Health checks if the MuninnDB server is reachable.
func (s *MuninnStore) Health(ctx context.Context) bool {
	ok, err := s.client.Health(ctx)
	if err != nil {
		slog.Debug("muninn health check failed", "error", err)
	}
	return ok
}

// Ensure interface compliance.
var _ Store = (*MuninnStore)(nil)
var _ CognitiveStore = (*MuninnStore)(nil)

// --- helpers ---

// engramToEntryFromActivation converts an ActivationItem to a MemoryEntry.
// ActivationItem does not carry CreatedAt/UpdatedAt, so timestamps are
// zero-valued. This is intentional — callers that need timestamps should
// use List or Read directly.
func engramToEntryFromActivation(item muninn.ActivationItem) *MemoryEntry {
	return &MemoryEntry{
		Key:   item.Concept,
		Value: item.Content,
	}
}

// engramToEntryFromEngram converts a full Engram (from Read) to a
// MemoryEntry, preserving real server-side timestamps.
func engramToEntryFromEngram(engram *muninn.Engram) *MemoryEntry {
	return &MemoryEntry{
		Key:       engram.Concept,
		Value:     engram.Content,
		CreatedAt: unixToTime(engram.CreatedAt),
		UpdatedAt: unixToTime(engram.UpdatedAt),
	}
}

func tagsForKey(key string) []string {
	tags := []string{"memory"}
	if idx := strings.IndexByte(key, '.'); idx > 0 {
		tags = append(tags, key[:idx])
	}
	return tags
}

// unixToTime converts a Unix timestamp (seconds) to time.Time.
func unixToTime(ts int64) time.Time {
	if ts <= 0 {
		return time.Time{}
	}
	return time.Unix(ts, 0)
}

// IsErrNotFound checks whether err wraps ErrMemoryNotFound.
func IsErrNotFound(err error) bool {
	return errors.Is(err, ErrMemoryNotFound)
}
