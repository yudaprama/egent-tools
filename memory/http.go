package memory

import (
	"context"
	"log/slog"
	"net/http"
)

// Identity holds the memory scoping ids extracted from request headers. Each
// egent handler resolves it once per request and reuses it for both recall
// (RecallPrefix) and persistence (SaveTurn) so the vault (tenant) and tags
// (user+session) stay consistent across the turn.
type Identity struct {
	TenantID  string
	UserID    string
	SessionID string
}

// InjectFromRequest reads the memory scoping ids from the request headers and
// stamps them onto the context (so the memory tools + Store calls can resolve
// vault/tags from context), returning the resolved Identity for later use.
//
// The edge (Oathkeeper) injects these authoritatively:
//   - x-tenant-id      → vault name (Kratos session active_workspace_id)
//   - x-arch-actor-id  → user id (whoami)
//   - x-model-affinity → session id (the chat session)
func InjectFromRequest(ctx context.Context, r *http.Request) (context.Context, Identity) {
	tenantID := r.Header.Get("x-tenant-id")
	userID := r.Header.Get("x-arch-actor-id")
	sessionID := r.Header.Get("x-model-affinity")
	ctx = WithTenantID(ctx, tenantID)
	ctx = WithUserID(ctx, userID)
	ctx = WithSessionID(ctx, sessionID)
	return ctx, Identity{TenantID: tenantID, UserID: userID, SessionID: sessionID}
}

// RecallPrefix retrieves memories relevant to query (scoped to the identity's
// tenant/user/session) and, when non-empty, returns them as a context block
// followed by the query so the agent sees prior conversation context without
// the client resending chat history. When recall yields nothing (empty vault,
// NoopStore, or no match), query is returned unchanged.
func (m *Manager) RecallPrefix(ctx context.Context, id Identity, query string) string {
	mem := m.Recall(ctx, id.TenantID, id.UserID, id.SessionID, query)
	if mem == "" {
		return query
	}
	return mem + "\n\n" + query
}

// SaveTurnAsync persists a raw Q&A conversation turn in a background goroutine
// so a response is never blocked on memory persistence. Errors are logged only.
// It is a no-op when the manager is backed by a NoopStore (MUNINN_URL unset).
// The request context is detached (context.WithoutCancel) so the save survives
// the request completing — values are preserved, cancellation/deadline is not.
func (m *Manager) SaveTurnAsync(ctx context.Context, id Identity, question, answer string) {
	go func() {
		if answer == "" {
			return
		}
		ctx := context.WithoutCancel(ctx)
		if err := m.store.SaveTurn(ctx, id.TenantID, id.UserID, id.SessionID, question, answer); err != nil {
			// Best-effort: log and move on. The response already went out.
			slog.Warn("memory: save turn failed", "err", err)
		}
	}()
}
