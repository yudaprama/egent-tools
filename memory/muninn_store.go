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
// Each tenant gets their own vault. userID and sessionID are stored as tags
// (e.g. "user:usr123", "session:sess456") so that Search and List can filter
// by user/session using MuninnDB's tags_all server-side filter.
//
// An in-memory ID cache maps (tenantID, userID, sessionID, concept) → engramID
// so that Get and Delete can call Read / Forget (exact operations) instead of
// Activate + filter. The cache is populated on Set and Delete and is safe for
// concurrent use.
type MuninnStore struct {
	client *muninn.Client

	// trustHeader, when non-empty, is the trusted identity header (e.g.
	// "X-Tenant-Id") the MuninnDB server trusts as the vault in edge-auth
	// mode. Each call attaches it (value = tenantID) so the trusted egent
	// backend can reach a per-tenant vault without a per-tenant Bearer token.
	trustHeader string

	// idCache stores known engram IDs for direct Read/Forget.
	// Structure: tenantID → (userID:sessionID:concept) → engramID.
	mu      sync.RWMutex
	idCache map[string]map[string]string
}

// CognitiveStore is an optional interface that Store implementations may
// satisfy when they support semantic/Hebbian memory activation. The Manager
// uses it to pick the richer retrieval path when available.
type CognitiveStore interface {
	Activate(ctx context.Context, tenantID, userID, sessionID string, ctxWords []string, limit int) ([]ActivatedMemory, error)
}

// NewMuninnStore creates a MuninnDB-backed memory store.
// baseURL is the MuninnDB HTTP endpoint (e.g. "http://localhost:8475").
// token is the API key (empty string for no auth).
func NewMuninnStore(baseURL, token string) *MuninnStore {
	return NewMuninnStoreWithTrustHeader(baseURL, token, muninn.DefaultTrustEdgeHeader)
}

// NewMuninnStoreWithTrustHeader creates a MuninnDB-backed memory store where
// the trusted-backend identity header is set to trustHeader, defaulting to
// "X-Tenant-Id". Use this when MuninnDB runs in edge-auth mode
// (MUNINN_TRUST_EDGE_HEADER) so a trusted backend (the egent) can reach
// per-tenant vaults without a per-tenant Bearer token. Set trustHeader to
// "" to disable.
func NewMuninnStoreWithTrustHeader(baseURL, token, trustHeader string) *MuninnStore {
	return &MuninnStore{
		client:      muninn.NewClient(baseURL, token),
		trustHeader: trustHeader,
		idCache:     make(map[string]map[string]string),
	}
}

// NewMuninnStoreFromClient creates a MuninnStore from an existing client.
func NewMuninnStoreFromClient(client *muninn.Client) *MuninnStore {
	return &MuninnStore{
		client:  client,
		idCache: make(map[string]map[string]string),
	}
}

// ctxWithVault attaches the trusted identity header (value = tenantID) when
// the store was configured with one. No-op otherwise.
func (s *MuninnStore) ctxWithVault(ctx context.Context, tenantID string) context.Context {
	if s.trustHeader == "" || tenantID == "" {
		return ctx
	}
	return muninn.WithTrustedVaultHeader(ctx, s.trustHeader, tenantID)
}

// cacheKey builds a deterministic key for the ID cache.
func cacheKey(userID, sessionID, concept string) string {
	return userID + ":" + sessionID + ":" + concept
}

// setIDCache records the engram ID for a (tenant, user, session, concept).
func (s *MuninnStore) setIDCache(tenant, userID, sessionID, concept, id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.idCache[tenant] == nil {
		s.idCache[tenant] = make(map[string]string)
	}
	s.idCache[tenant][cacheKey(userID, sessionID, concept)] = id
}

// lookupIDCache returns the cached engram ID for a (tenant, user, session, concept), or "".
func (s *MuninnStore) lookupIDCache(tenant, userID, sessionID, concept string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.idCache[tenant] == nil {
		return ""
	}
	return s.idCache[tenant][cacheKey(userID, sessionID, concept)]
}

// deleteIDCache removes a (tenant, user, session, concept) entry from the cache.
func (s *MuninnStore) deleteIDCache(tenant, userID, sessionID, concept string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.idCache[tenant], cacheKey(userID, sessionID, concept))
}

// Set stores a memory as an engram. The tenantID becomes the vault name.
// The key becomes the concept, the value becomes the content.
// Tags include "user:<userID>", "session:<sessionID>", and the
// key prefix (e.g. "user", "preferences").
func (s *MuninnStore) Set(ctx context.Context, tenantID, userID, sessionID, key, value string) error {
	tags := muninn.TagsForKey(key, userID, sessionID)

	id, err := s.client.Write(s.ctxWithVault(ctx, tenantID), tenantID, key, value, tags)
	if err != nil {
		return fmt.Errorf("muninn write: %w", err)
	}
	s.setIDCache(tenantID, userID, sessionID, key, id)
	return nil
}

// SaveTurn stores a raw conversation turn as a single engram. The question
// becomes the concept (byte-capped via muninn.TruncateConcept to fit the
// server's Concept field), the answer becomes the content, and tags are
// exactly [user:<userID>, session:<sessionID>]. The production chat handler
// does not call this method; the web client owns conversation persistence and
// resends the full history.
func (s *MuninnStore) SaveTurn(ctx context.Context, tenantID, userID, sessionID, question, answer string) error {
	concept := muninn.TruncateConcept(question)
	tags := []string{"user:" + userID, "session:" + sessionID}

	id, err := s.client.Write(s.ctxWithVault(ctx, tenantID), tenantID, concept, answer, tags)
	if err != nil {
		return fmt.Errorf("muninn save turn: %w", err)
	}
	s.setIDCache(tenantID, userID, sessionID, concept, id)
	return nil
}

// SetBatch stores multiple memories in a single MuninnDB batch call.
func (s *MuninnStore) SetBatch(ctx context.Context, tenantID, userID, sessionID string, entries map[string]string) error {
	return s.writeBatchTags(ctx, tenantID, userID, sessionID, entries, nil)
}

// SetProfile stores session-independent profile facts (name, email, ...) each
// tagged with "user:<userID>" and an explicit "profile" tag so RecallProfile
// can select them without inferring profile-ness from key naming or an empty
// session. Profile facts carry no session tag by design.
func (s *MuninnStore) SetProfile(ctx context.Context, tenantID, userID string, facts map[string]string) error {
	return s.writeBatchTags(ctx, tenantID, userID, "", facts, []string{muninn.ProfileTag})
}

// writeBatchTags persists entries as engrams in batched MuninnDB writes,
// deriving per-key tags via muninn.TagsForKey and appending extraTags (e.g.
// muninn.ProfileTag) to every engram. Server-returned IDs are cached so
// subsequent Get/Delete can use direct Read/Forget.
func (s *MuninnStore) writeBatchTags(ctx context.Context, tenantID, userID, sessionID string, entries map[string]string, extraTags []string) error {
	if len(entries) == 0 {
		return nil
	}

	requests := make([]muninn.WriteRequest, 0, len(entries))
	keys := make([]string, 0, len(entries))
	for key, value := range entries {
		tags := muninn.TagsForKey(key, userID, sessionID)
		tags = append(tags, extraTags...)
		requests = append(requests, muninn.WriteRequest{
			Vault:      tenantID,
			Concept:    key,
			Content:    value,
			Tags:       tags,
			Confidence: 0.9,
			Stability:  0.5,
		})
		keys = append(keys, key)
	}

	for start := 0; start < len(requests); start += 50 {
		end := min(start+50, len(requests))
		resp, err := s.client.WriteBatch(s.ctxWithVault(ctx, tenantID), tenantID, requests[start:end])
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
			s.setIDCache(tenantID, userID, sessionID, keys[start+result.Index], result.ID)
		}
	}
	return nil
}

// Get retrieves a specific memory by concept (key). It first tries the
// local ID cache for an exact engram Read. If the ID is not cached
// (e.g. after a server restart), it falls back to Activate with a
// tags_all filter and filters for an exact concept match.
//
// Returns (nil, nil) when no entry is found.
func (s *MuninnStore) Get(ctx context.Context, tenantID, userID, sessionID, key string) (*MemoryEntry, error) {
	if id := s.lookupIDCache(tenantID, userID, sessionID, key); id != "" {
		engram, err := s.client.Read(s.ctxWithVault(ctx, tenantID), id, tenantID)
		if err != nil {
			return nil, fmt.Errorf("muninn read: %w", err)
		}
		return engramToEntryFromEngram(engram), nil
	}

	// Fallback: activate with tag filter + exact concept match.
	filters := tagsAllFilter(userID, sessionID)
	resp, err := s.client.Activate(s.ctxWithVault(ctx, tenantID), tenantID, []string{key}, 20, filters...)
	if err != nil {
		return nil, fmt.Errorf("muninn activate: %w", err)
	}
	for _, item := range resp.Activations {
		if item.Concept == key {
			s.setIDCache(tenantID, userID, sessionID, key, item.ID)
			return engramToEntryFromActivation(item), nil
		}
	}
	return nil, nil
}

// Delete forgets an engram by concept (key). It first tries the ID
// cache for a direct Forget, falling back to Activate when the ID is
// unknown. Returns ErrMemoryNotFound when neither path locates the engram.
func (s *MuninnStore) Delete(ctx context.Context, tenantID, userID, sessionID, key string) error {
	if id := s.lookupIDCache(tenantID, userID, sessionID, key); id != "" {
		if err := s.client.Forget(s.ctxWithVault(ctx, tenantID), id, tenantID); err != nil {
			return fmt.Errorf("muninn forget: %w", err)
		}
		s.deleteIDCache(tenantID, userID, sessionID, key)
		return nil
	}

	filters := tagsAllFilter(userID, sessionID)
	resp, err := s.client.Activate(s.ctxWithVault(ctx, tenantID), tenantID, []string{key}, 20, filters...)
	if err != nil {
		return fmt.Errorf("muninn activate for delete: %w", err)
	}
	for _, item := range resp.Activations {
		if item.Concept == key {
			if err := s.client.Forget(ctx, item.ID, tenantID); err != nil {
				return fmt.Errorf("muninn forget: %w", err)
			}
			s.deleteIDCache(tenantID, userID, sessionID, key)
			return nil
		}
	}
	return ErrMemoryNotFound
}

// Search uses MuninnDB's Activate for context-aware retrieval. When
// ByUserID, BySessionID, or ByTag options are provided, they are converted
// into a tags_all server-side filter so the engine only considers engrams
// matching those tags.
func (s *MuninnStore) Search(ctx context.Context, tenantID, query string, limit int, opts ...SearchOption) ([]MemoryEntry, error) {
	if limit <= 0 {
		limit = 10
	}

	var filter SearchFilter
	for _, o := range opts {
		o(&filter)
	}

	filters := tagsAllFilterCombined(filter.UserID, filter.SessionID, filter.Tags)
	effectiveLimit := limit
	if len(filters) > 0 {
		effectiveLimit = limit * 3
	}

	resp, err := s.client.Activate(s.ctxWithVault(ctx, tenantID), tenantID, []string{query}, effectiveLimit, filters...)
	if err != nil {
		return nil, fmt.Errorf("muninn activate: %w", err)
	}

	entries := make([]MemoryEntry, 0, len(resp.Activations))
	for _, item := range resp.Activations {
		uid, sid := muninn.ExtractUserSession(item.Tags)
		s.setIDCache(tenantID, uid, sid, item.Concept, item.ID)
		entries = append(entries, MemoryEntry{
			Key:       item.Concept,
			Value:     item.Content,
			UserID:    uid,
			SessionID: sid,
		})
	}
	return entries, nil
}

// List returns all memories for a tenant via paginated ListEngrams.
// When ByUserID, BySessionID, or ByTag options are provided, results are
// filtered client-side (MuninnDB ListEngrams does not support tag filters).
func (s *MuninnStore) List(ctx context.Context, tenantID string, opts ...SearchOption) ([]MemoryEntry, error) {
	var filter SearchFilter
	for _, o := range opts {
		o(&filter)
	}

	var all []MemoryEntry
	offset := 0
	pageSize := 100

	for {
		resp, err := s.client.ListEngrams(s.ctxWithVault(ctx, tenantID), tenantID, pageSize, offset)
		if err != nil {
			return nil, fmt.Errorf("muninn list: %w", err)
		}

		for _, item := range resp.Engrams {
			uid, sid := muninn.ExtractUserSession(item.Tags)
			if !passesUserSessionFilter(uid, sid, filter) {
				continue
			}
			if len(filter.Tags) > 0 && !muninn.HasAllTags(item.Tags, filter.Tags) {
				continue
			}
			s.setIDCache(tenantID, uid, sid, item.Concept, item.ID)
			all = append(all, MemoryEntry{
				Key:       item.Concept,
				Value:     item.Content,
				UserID:    uid,
				SessionID: sid,
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
func (s *MuninnStore) Activate(ctx context.Context, tenantID, userID, sessionID string, ctxWords []string, limit int) ([]ActivatedMemory, error) {
	if limit <= 0 {
		limit = 10
	}

	filters := tagsAllFilter(userID, sessionID)

	resp, err := s.client.Activate(s.ctxWithVault(ctx, tenantID), tenantID, ctxWords, limit, filters...)
	if err != nil {
		return nil, fmt.Errorf("muninn activate: %w", err)
	}

	results := make([]ActivatedMemory, 0, len(resp.Activations))
	for _, item := range resp.Activations {
		uid, sid := muninn.ExtractUserSession(item.Tags)
		s.setIDCache(tenantID, uid, sid, item.Concept, item.ID)
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
var _ ProfileStore = (*MuninnStore)(nil)

// --- helpers ---

// tagsAllFilter builds tags_all filters when userID or sessionID is set.
func tagsAllFilter(userID, sessionID string) []muninn.Filter {
	var tagsAll []string
	if userID != "" {
		tagsAll = append(tagsAll, "user:"+userID)
	}
	if sessionID != "" {
		tagsAll = append(tagsAll, "session:"+sessionID)
	}
	if len(tagsAll) == 0 {
		return nil
	}
	return []muninn.Filter{{Field: "tags_all", Op: "all", Value: tagsAll}}
}

// tagsAllFilterCombined builds a single tags_all filter merging user/session
// identity tags with any additional caller-supplied tags.
func tagsAllFilterCombined(userID, sessionID string, extraTags []string) []muninn.Filter {
	var tagsAll []string
	if userID != "" {
		tagsAll = append(tagsAll, "user:"+userID)
	}
	if sessionID != "" {
		tagsAll = append(tagsAll, "session:"+sessionID)
	}
	tagsAll = append(tagsAll, extraTags...)
	if len(tagsAll) == 0 {
		return nil
	}
	return []muninn.Filter{{Field: "tags_all", Op: "all", Value: tagsAll}}
}

// passesUserSessionFilter checks whether the entry's tags match the filter.
func passesUserSessionFilter(tagUserID, tagSessionID string, filter SearchFilter) bool {
	if filter.UserID != "" && tagUserID != filter.UserID {
		return false
	}
	if filter.SessionID != "" && tagSessionID != filter.SessionID {
		return false
	}
	return true
}

// engramToEntryFromActivation converts an ActivationItem to a MemoryEntry.
func engramToEntryFromActivation(item muninn.ActivationItem) *MemoryEntry {
	uid, sid := muninn.ExtractUserSession(item.Tags)
	return &MemoryEntry{
		Key:       item.Concept,
		Value:     item.Content,
		UserID:    uid,
		SessionID: sid,
	}
}

// engramToEntryFromEngram converts a full Engram (from Read) to a
// MemoryEntry, preserving real server-side timestamps.
func engramToEntryFromEngram(engram *muninn.Engram) *MemoryEntry {
	uid, sid := muninn.ExtractUserSession(engram.Tags)
	return &MemoryEntry{
		Key:       engram.Concept,
		Value:     engram.Content,
		UserID:    uid,
		SessionID: sid,
		CreatedAt: unixToTime(engram.CreatedAt),
		UpdatedAt: unixToTime(engram.UpdatedAt),
	}
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
