package knowledge

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"strings"

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
	Rerank(ctx context.Context, query string, documents []string) ([]rerank.RankResult, error)
}

// KnowledgeSearchTool performs semantic search over the current user's
// documents, knowledge bases, and ingested files. It is a thin wrapper over
// fileprocessor.Searcher that adds per-user file scoping.
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

	span.SetStatus(codes.Ok, "")
	if len(results) == 0 {
		return fmt.Sprintf("No relevant documents found for query: %q", args.Query), nil
	}

	return FormatResults(results, args.Query), nil
}

// rerankResults applies the rerank model to search results,
// returning them sorted by relevance score descending, limited
// to the original result count.
func (t *KnowledgeSearchTool) rerankResults(ctx context.Context, query string, results []fp.SearchResult) ([]fp.SearchResult, error) {
	documents := make([]string, len(results))
	for i, r := range results {
		documents[i] = r.Text
	}
	rankResults, err := t.svc.Rerank(ctx, query, documents)
	if err != nil {
		return nil, err
	}
	if len(rankResults) == 0 {
		return nil, nil
	}
	for _, rr := range rankResults {
		if rr.Index >= 0 && rr.Index < len(results) {
			results[rr.Index].Similarity = rr.RelevanceScore
		}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})
	return results, nil
}

// FormatResults renders a list of search hits as an LLM-friendly context
// block. Source filename and similarity score are included so the model can
// reason about provenance.
func FormatResults(results []fp.SearchResult, query string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Knowledge base results for query: %q (%d hits)\n\n", query, len(results))
	for i, r := range results {
		fmt.Fprintf(&b, "[%d] (similarity=%.3f) %s\n", i+1, r.Similarity, sourceLabel(r))
		fmt.Fprintf(&b, "%s\n\n", strings.TrimSpace(r.Text))
	}
	return b.String()
}

func sourceLabel(r fp.SearchResult) string {
	if r.FileName != "" {
		return "file: " + r.FileName
	}
	if r.FileID != "" {
		return "file_id: " + r.FileID
	}
	return "chunk: " + r.ID
}
