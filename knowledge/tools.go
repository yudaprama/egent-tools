package knowledge

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	scoretransformer "github.com/cloudwego/eino-ext/components/document/transformer/reranker/score"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
	fp "github.com/kawai-network/fileprocessor"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/yudaprama/egent-tools/memory"
	"github.com/yudaprama/egent-tools/rerank"
)

// tracerName is the OpenTelemetry tracer name for this package. It resolves
// lazily from the global provider on each span, so the host's
// TracerProvider (registered once at startup) is shared without this module
// depending on any tracing glue package.
const tracerName = "egent-tools/knowledge"

// KnowledgeBackend is the subset of *Service the tool needs. Declared as an
// interface so tests can inject a fake without touching the pool.
type KnowledgeBackend interface {
	UserFileIDs(ctx context.Context, userID, tenantID, projectID string) ([]string, error)
	Searcher() Searcher
	Rerank(ctx context.Context, query string, documents []*schema.Document) ([]*schema.Document, error)
	// GetChunksByIDs fetches parent chunks for context expansion.
	GetChunksByIDs(ctx context.Context, ids []string) ([]fp.Chunk, error)
}

// KnowledgeSearchTool performs semantic search over the current user's
// documents, knowledge bases, and ingested files. It is a thin wrapper over
// fileprocessor.Searcher that adds per-user file scoping. Retrieval results
// flow through the pipeline as Eino schema.Documents until they are formatted
// into the tool's string response.
//
// The user_id is read from context via memory.UserIDFromContext. The host
// agent runtime must inject it per-request with memory.WithUserID so the
// tool scopes per-user.
type KnowledgeSearchTool struct {
	svc         KnowledgeBackend
	rerankModel rerank.Reranker
}

// NewKnowledgeSearchTool wraps a Service (or any KnowledgeBackend) as an
// Eino tool. Pass nil to create a tool that always returns a "not
// configured" message — useful when the vector DB is absent.
func NewKnowledgeSearchTool(svc KnowledgeBackend, rerankModel rerank.Reranker) *KnowledgeSearchTool {
	return &KnowledgeSearchTool{svc: svc, rerankModel: rerankModel}
}

func (t *KnowledgeSearchTool) Info(_ context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: "knowledge_search",
		Desc: "Search the current user's documents, knowledge bases, and ingested files by semantic similarity. " +
			"Returns the most relevant chunks with source filenames. Use this when the user asks about " +
			"their files, notes, or knowledge base content.",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"query": {
				Desc:     "Natural-language search query (e.g. 'project timeline', 'how do I deploy').",
				Type:     schema.String,
				Required: true,
			},
			"limit": {
				Desc:     "Max chunks to return. Defaults to 10, max 50.",
				Type:     schema.Integer,
				Required: false,
			},
		}),
	}, nil
}

func (t *KnowledgeSearchTool) InvokableRun(ctx context.Context, argsJSON string, _ ...tool.Option) (string, error) {
	ctx, span := otel.Tracer(tracerName).Start(ctx, "knowledge_search",
		trace.WithAttributes(
			attribute.String("args", argsJSON),
		),
	)
	defer span.End()

	var args struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		span.SetStatus(codes.Error, "parse args")
		span.RecordError(err)
		return "", fmt.Errorf("knowledge_search: parse args: %w", err)
	}
	if strings.TrimSpace(args.Query) == "" {
		span.SetStatus(codes.Error, "empty query")
		return "", fmt.Errorf("knowledge_search: query is required")
	}
	if args.Limit <= 0 {
		args.Limit = 10
	}
	if args.Limit > 50 {
		args.Limit = 50
	}

	span.SetAttributes(
		attribute.String("query", args.Query),
		attribute.Int("limit", args.Limit),
	)

	if t.svc == nil || t.svc.Searcher() == nil {
		span.SetAttributes(attribute.Bool("configured", false))
		span.SetStatus(codes.Ok, "not configured")
		return "knowledge search is not configured on this server", nil
	}

	userID := memory.UserIDFromContext(ctx)
	if userID == "" {
		span.SetStatus(codes.Error, "no user_id")
		return "", fmt.Errorf("knowledge_search: no user_id in context")
	}
	span.SetAttributes(attribute.String("user.id", userID))
	tenantID := memory.TenantIDFromContext(ctx)
	if tenantID == "" {
		span.SetStatus(codes.Error, "no tenant_id")
		return "", fmt.Errorf("knowledge_search: no tenant_id in context")
	}
	span.SetAttributes(attribute.String("tenant.id", tenantID))

	projectID := memory.ProjectIDFromContext(ctx)
	span.SetAttributes(attribute.String("project.id", projectID))

	fileIDs, err := t.svc.UserFileIDs(ctx, userID, tenantID, projectID)
	if err != nil {
		span.SetStatus(codes.Error, "list user files")
		span.RecordError(err)
		return "", fmt.Errorf("knowledge_search: list user files: %w", err)
	}
	span.SetAttributes(attribute.Int("user.file_count", len(fileIDs)))

	if len(fileIDs) == 0 {
		span.SetStatus(codes.Ok, "no files")
		return "No documents found for this user. Upload files via the AList integration to populate the knowledge base.", nil
	}

	results, err := t.svc.Searcher().SemanticSearch(ctx, fp.SearchParamsSearcher{
		Query:   args.Query,
		FileIDs: fileIDs,
		Limit:   args.Limit,
	})
	if err != nil {
		span.SetStatus(codes.Error, "semantic search")
		span.RecordError(err)
		slog.Warn("knowledge_search: semantic search failed", "error", err, "user_id", userID)
		return "", fmt.Errorf("knowledge_search: search: %w", err)
	}
	span.SetAttributes(attribute.Int("results.count", len(results)))

	if t.rerankModel != nil && len(results) > 0 {
		reranked, err := t.rerankResults(ctx, args.Query, results)
		if err != nil {
			slog.Warn("knowledge_search: rerank failed, using original results", "error", err)
		} else if len(reranked) > 0 {
			results = reranked
		}
	}

	// Expand results with parent context for richer retrieval.
	results = t.expandParentContext(ctx, results)

	span.SetStatus(codes.Ok, "")
	if len(results) == 0 {
		return fmt.Sprintf("No relevant documents found for query: %q", args.Query), nil
	}

	return FormatResults(results, args.Query), nil
}

// rerankResults applies the rerank model and then positions the scored
// documents for LLM context using Eino's score transformer.
func (t *KnowledgeSearchTool) rerankResults(ctx context.Context, query string, results []*schema.Document) ([]*schema.Document, error) {
	ranked, err := t.svc.Rerank(ctx, query, results)
	if err != nil {
		return nil, err
	}
	if len(ranked) == 0 {
		return nil, nil
	}
	transformer, err := scoretransformer.NewReranker(ctx, &scoretransformer.Config{})
	if err != nil {
		return nil, fmt.Errorf("create score transformer: %w", err)
	}
	placed, err := transformer.Transform(ctx, ranked)
	if err != nil {
		return nil, fmt.Errorf("place reranked documents: %w", err)
	}
	return placed, nil
}

// expandParentContext fetches parent chunks for results that have a ParentID,
// prepending the parent text as context. This gives the LLM broader section
// context even when only a child chunk matched.
func (t *KnowledgeSearchTool) expandParentContext(ctx context.Context, results []*schema.Document) []*schema.Document {
	if t.svc == nil || len(results) == 0 {
		return results
	}

	// Collect unique parent IDs.
	parentIDs := make(map[string]bool)
	for _, r := range results {
		parentID := fp.DocumentStringMetadata(r, fp.DocumentMetaParentID)
		if parentID != "" {
			parentIDs[parentID] = true
		}
	}
	if len(parentIDs) == 0 {
		return results
	}

	ids := make([]string, 0, len(parentIDs))
	for id := range parentIDs {
		ids = append(ids, id)
	}

	parents, err := t.svc.GetChunksByIDs(ctx, ids)
	if err != nil {
		slog.Warn("knowledge_search: expand parent context failed", "error", err)
		return results
	}

	// Build parent text lookup.
	parentText := make(map[string]string, len(parents))
	for _, p := range parents {
		parentText[p.ID] = p.Text
	}

	// Prepend parent context to child results.
	expanded := make([]*schema.Document, 0, len(results))
	for _, r := range results {
		parentID := fp.DocumentStringMetadata(r, fp.DocumentMetaParentID)
		if pt, ok := parentText[parentID]; ok && pt != "" {
			r.Content = "[Context from section]\n" + pt + "\n\n[Excerpt]\n" + r.Content
		}
		expanded = append(expanded, r)
	}
	return expanded
}

// FormatResults renders a list of search hits as an LLM-friendly context
// block. Source filename and similarity score are included so the model can
// reason about provenance.
func FormatResults(results []*schema.Document, query string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Knowledge base results for query: %q (%d hits)\n\n", query, len(results))
	for i, r := range results {
		fmt.Fprintf(&b, "[%d] (similarity=%.3f) %s\n", i+1, r.Score(), sourceLabel(r))
		fmt.Fprintf(&b, "%s\n\n", strings.TrimSpace(r.Content))
	}
	return b.String()
}

func sourceLabel(r *schema.Document) string {
	if fileName := fp.DocumentStringMetadata(r, fp.DocumentMetaFileName); fileName != "" {
		return "file: " + fileName
	}
	if fileID := fp.DocumentStringMetadata(r, fp.DocumentMetaFileID); fileID != "" {
		return "file_id: " + fileID
	}
	return "chunk: " + r.ID
}
