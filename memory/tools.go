package memory

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
)

// ManagerStore is the minimal interface tools need from the manager.
type ManagerStore interface {
	Store(ctx context.Context, tenantID, userID, sessionID, key, value string) error
	Get(ctx context.Context, tenantID, userID, sessionID, key string) (*MemoryEntry, error)
	Delete(ctx context.Context, tenantID, userID, sessionID, key string) error
	Search(ctx context.Context, tenantID, query string, limit int, opts ...SearchOption) ([]MemoryEntry, error)
	List(ctx context.Context, tenantID string, opts ...SearchOption) ([]MemoryEntry, error)
}

// context key types
type tenantIDKey struct{}
type userIDKey struct{}
type sessionIDKey struct{}
type projectIDKey struct{}

func WithTenantID(ctx context.Context, tenantID string) context.Context {
	return context.WithValue(ctx, tenantIDKey{}, tenantID)
}

func TenantIDFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(tenantIDKey{}).(string); ok {
		return v
	}
	return ""
}

func WithUserID(ctx context.Context, userID string) context.Context {
	return context.WithValue(ctx, userIDKey{}, userID)
}

func UserIDFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(userIDKey{}).(string); ok {
		return v
	}
	return ""
}

func WithSessionID(ctx context.Context, sessionID string) context.Context {
	return context.WithValue(ctx, sessionIDKey{}, sessionID)
}

func SessionIDFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(sessionIDKey{}).(string); ok {
		return v
	}
	return ""
}

func WithProjectID(ctx context.Context, projectID string) context.Context {
	return context.WithValue(ctx, projectIDKey{}, projectID)
}

func ProjectIDFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(projectIDKey{}).(string); ok {
		return v
	}
	return ""
}

// MemorySetTool lets the agent store a fact about the user.
type MemorySetTool struct {
	mgr *Manager
}

func NewMemorySetTool(mgr *Manager) *MemorySetTool {
	return &MemorySetTool{mgr: mgr}
}

func (t *MemorySetTool) Info(_ context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: "userMemory_set",
		Desc: "Store a fact about the current user for future conversations. Use key/value pairs (e.g. key='user.name' value='Alice').",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"key":   {Desc: "Memory key (e.g. 'user.name', 'user.location', 'preferences.dark_mode')", Type: schema.String, Required: true},
			"value": {Desc: "The fact to remember", Type: schema.String, Required: true},
		}),
	}, nil
}

func (t *MemorySetTool) InvokableRun(ctx context.Context, argsJSON string, _ ...tool.Option) (string, error) {
	var args struct {
		Key   string `json:"key"`
		Value string `json:"value"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("parse args: %w", err)
	}
	tenantID := TenantIDFromContext(ctx)
	if tenantID == "" {
		return "", fmt.Errorf("no tenant_id in context")
	}
	userID := UserIDFromContext(ctx)
	if userID == "" {
		return "", fmt.Errorf("no user_id in context")
	}
	sessionID := SessionIDFromContext(ctx)
	if err := t.mgr.store.Set(ctx, tenantID, userID, sessionID, args.Key, args.Value); err != nil {
		return "", err
	}
	return fmt.Sprintf("Stored memory: %s = %s", args.Key, args.Value), nil
}

// MemoryGetTool retrieves a specific memory entry.
type MemoryGetTool struct {
	mgr *Manager
}

func NewMemoryGetTool(mgr *Manager) *MemoryGetTool {
	return &MemoryGetTool{mgr: mgr}
}

func (t *MemoryGetTool) Info(_ context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: "userMemory_get",
		Desc: "Retrieve a specific memory entry by key. Returns null/empty if not found.",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"key": {Desc: "Memory key to retrieve", Type: schema.String, Required: true},
		}),
	}, nil
}

func (t *MemoryGetTool) InvokableRun(ctx context.Context, argsJSON string, _ ...tool.Option) (string, error) {
	var args struct {
		Key string `json:"key"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("parse args: %w", err)
	}
	tenantID := TenantIDFromContext(ctx)
	if tenantID == "" {
		return "", fmt.Errorf("no tenant_id in context")
	}
	userID := UserIDFromContext(ctx)
	if userID == "" {
		return "", fmt.Errorf("no user_id in context")
	}
	sessionID := SessionIDFromContext(ctx)
	entry, err := t.mgr.store.Get(ctx, tenantID, userID, sessionID, args.Key)
	if err != nil {
		return "", err
	}
	if entry == nil {
		return fmt.Sprintf("No memory found for key: %s", args.Key), nil
	}
	return fmt.Sprintf("%s = %s", entry.Key, entry.Value), nil
}

// MemorySearchTool searches user memories for relevant facts.
type MemorySearchTool struct {
	mgr *Manager
}

func NewMemorySearchTool(mgr *Manager) *MemorySearchTool {
	return &MemorySearchTool{mgr: mgr}
}

func (t *MemorySearchTool) Info(_ context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: "userMemory_search",
		Desc: "Search the user's stored memories. Use at the start of a conversation to recall user preferences, name, location, and history.",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"query": {Desc: "Search terms (e.g. 'name', 'location', 'preferences')", Type: schema.String, Required: true},
			"limit": {Desc: "Max results to return (default 10)", Type: schema.Integer, Required: false},
		}),
	}, nil
}

func (t *MemorySearchTool) InvokableRun(ctx context.Context, argsJSON string, _ ...tool.Option) (string, error) {
	var args struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("parse args: %w", err)
	}
	tenantID := TenantIDFromContext(ctx)
	if tenantID == "" {
		return "", fmt.Errorf("no tenant_id in context")
	}
	if args.Limit == 0 {
		args.Limit = 10
	}
	entries, err := t.mgr.store.Search(ctx, tenantID, args.Query, args.Limit, ByUserID(UserIDFromContext(ctx)), BySessionID(SessionIDFromContext(ctx)))
	if err != nil {
		return "", err
	}
	if len(entries) == 0 {
		return fmt.Sprintf("No memories found matching: %s", args.Query), nil
	}
	return FormatMemories(entries), nil
}

// MemoryListTool lists all stored memories for the current user and session.
type MemoryListTool struct {
	mgr *Manager
}

func NewMemoryListTool(mgr *Manager) *MemoryListTool {
	return &MemoryListTool{mgr: mgr}
}

func (t *MemoryListTool) Info(_ context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name:        "userMemory_list",
		Desc:        "List all stored memories for the current user and session.",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{}),
	}, nil
}

func (t *MemoryListTool) InvokableRun(ctx context.Context, _ string, _ ...tool.Option) (string, error) {
	tenantID := TenantIDFromContext(ctx)
	if tenantID == "" {
		return "", fmt.Errorf("no tenant_id in context")
	}
	entries, err := t.mgr.store.List(ctx, tenantID, ByUserID(UserIDFromContext(ctx)), BySessionID(SessionIDFromContext(ctx)))
	if err != nil {
		return "", err
	}
	if len(entries) == 0 {
		return "No memories stored yet.", nil
	}
	return FormatMemories(entries), nil
}

// MemoryDeleteTool deletes a specific memory entry by key.
type MemoryDeleteTool struct {
	mgr *Manager
}

func NewMemoryDeleteTool(mgr *Manager) *MemoryDeleteTool {
	return &MemoryDeleteTool{mgr: mgr}
}

func (t *MemoryDeleteTool) Info(_ context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: "userMemory_delete",
		Desc: "Delete a specific memory entry by key. Returns confirmation or an error if not found.",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"key": {Desc: "Memory key to delete (e.g. 'user.name', 'preferences.dark_mode')", Type: schema.String, Required: true},
		}),
	}, nil
}

func (t *MemoryDeleteTool) InvokableRun(ctx context.Context, argsJSON string, _ ...tool.Option) (string, error) {
	var args struct {
		Key string `json:"key"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("parse args: %w", err)
	}
	tenantID := TenantIDFromContext(ctx)
	if tenantID == "" {
		return "", fmt.Errorf("no tenant_id in context")
	}
	userID := UserIDFromContext(ctx)
	if userID == "" {
		return "", fmt.Errorf("no user_id in context")
	}
	sessionID := SessionIDFromContext(ctx)
	if err := t.mgr.store.Delete(ctx, tenantID, userID, sessionID, args.Key); err != nil {
		return "", err
	}
	return fmt.Sprintf("Deleted memory: %s", args.Key), nil
}

// AllTools returns the standard set of memory tools for the agent.
func (m *Manager) AllTools() []tool.BaseTool {
	return []tool.BaseTool{
		NewMemorySetTool(m),
		NewMemoryGetTool(m),
		NewMemorySearchTool(m),
		NewMemoryListTool(m),
		NewMemoryDeleteTool(m),
	}
}
