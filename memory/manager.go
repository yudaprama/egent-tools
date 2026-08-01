package memory

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/cloudwego/eino-ext/components/model/openai"
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
		store: store,
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

// SaveTurn persists a raw Q&A conversation turn scoped to the given
// tenant/user/session: the question becomes the concept, the answer becomes
// the content, tagged with [user:<userID>, session:<sessionID>]. Unlike
// ExtractAndStore (which runs an LLM/heuristic extractor over the text), this
// stores the verbatim exchange so it can be recalled as conversation history.
// Best-effort: callers should run it asynchronously and never block a
// response on it.
func (m *Manager) SaveTurn(ctx context.Context, tenantID, userID, sessionID, question, answer string) error {
	return m.store.SaveTurn(ctx, tenantID, userID, sessionID, question, answer)
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
	if m.extractor == nil {
		return nil
	}
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

// IngestProfile writes structured profile facts (name, email, join date)
// directly to the store. Unlike ExtractAndStore (which runs an LLM over
// conversation text), this stores caller-provided key/value pairs with no
// extraction step — intended for registration-time data that is already
// known.
//
// Tags are set automatically: "user:<userID>" and the key prefix
// (e.g. "user" for "user.email"). The caller should pass an empty sessionID
// since profile facts are session-independent.
func (m *Manager) IngestProfile(ctx context.Context, tenantID, userID string, facts map[string]string) error {
	if len(facts) == 0 {
		return nil
	}
	stored := 0
	if m.batch != nil {
		if err := m.batch.SetBatch(ctx, tenantID, userID, "", facts); err != nil {
			slog.Warn("memory profile batch store failed", "error", err)
		} else {
			stored = len(facts)
		}
	} else {
		for key, value := range facts {
			if err := m.store.Set(ctx, tenantID, userID, "", key, value); err != nil {
				slog.Warn("memory profile store failed", "key", key, "error", err)
				continue
			}
			stored++
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

// NewManagerFromEnv builds a memory.Manager from the MUNINN_URL env var. When
// MUNINN_URL is unset, returns a Manager backed by NoopStore (every recall/save
// is a no-op) so an egent boots in dev/CI without MuninnDB. When set, it
// constructs a MuninnStore using edge-auth via MUNINN_TRUST_EDGE_HEADER
// (default "X-Tenant-Id") and fatals if the server is unreachable. The
// returned cleanup func is always non-nil and safe to defer.
//
// The extractor uses an LLM-backed extractor (OpenAI-compatible via
// PLANO_LLM_GATEWAY) with MODEL_NAME as the extraction model. Fatal if
// MODEL_NAME or PLANO_LLM_GATEWAY is not set, or if model creation fails.
func NewManagerFromEnv(ctx context.Context) (*Manager, func()) {
	muninnURL := os.Getenv("MUNINN_URL")
	if muninnURL == "" {
		slog.Warn("memory: MUNINN_URL unset — using NoopStore (recall/save are no-ops)")
		return NewManager(NoopStore{}), func() {}
	}
	trustHeader := os.Getenv("MUNINN_TRUST_EDGE_HEADER")
	if trustHeader == "" {
		trustHeader = "X-Tenant-Id"
	}
	s := NewMuninnStoreWithTrustHeader(muninnURL, os.Getenv("MUNINN_TOKEN"), trustHeader)
	if !s.Health(ctx) {
		slog.Error("memory: MuninnDB configured but unreachable at startup", "url", muninnURL)
		os.Exit(1)
	}
	slog.Info("memory: MuninnDB store enabled", "url", muninnURL)

	// Build LLM extractor — the only extraction path. Always uses MODEL_NAME
	// (the egent's primary model) via PLANO_LLM_GATEWAY.
	modelName := os.Getenv("MODEL_NAME")
	if modelName == "" {
		slog.Error("memory: MODEL_NAME must be set for LLM extraction")
		os.Exit(1)
	}
	baseURL := os.Getenv("PLANO_LLM_GATEWAY")
	if baseURL == "" {
		slog.Error("memory: PLANO_LLM_GATEWAY must be set for LLM extraction")
		os.Exit(1)
	}
	apiKey := os.Getenv("PLANO_INTERNAL_KEY")
	if apiKey == "" {
		apiKey = "EMPTY"
	}
	chatModel, err := openai.NewChatModel(ctx, &openai.ChatModelConfig{
		BaseURL: baseURL,
		Model:   modelName,
		APIKey:  apiKey,
	})
	if err != nil {
		slog.Error("memory: failed to create LLM extractor", "error", err)
		os.Exit(1)
	}
	mgr := NewManager(s).WithExtractor(NewLLMExtractor(chatModel))
	slog.Info("memory: LLM extractor enabled", "model", modelName)

	return mgr, func() {}
}

// RecallProfile retrieves profile-scoped memories (name, email, etc.)
// for the given user and returns them as a context block. Returns empty
// string when no profile data exists or the store is unavailable.
func (m *Manager) RecallProfile(ctx context.Context, tenantID, userID string) string {
	entries, err := m.store.List(ctx, tenantID, ByUserID(userID), ByTag("profile"))
	if err != nil {
		slog.Warn("memory profile recall failed", "error", err)
		return ""
	}
	return FormatMemories(entries)
}

// ErrStoreUnavailable is returned when the memory store is not initialized.
var ErrStoreUnavailable = errors.New("memory store not available")
