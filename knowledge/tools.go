package knowledge

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"unicode"

	scoretransformer "github.com/cloudwego/eino-ext/components/document/transformer/reranker/score"
	"github.com/cloudwego/eino/components/retriever"
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

const defaultMinScore = 0.3

// KnowledgeBackend is the subset of *Service the tool needs. Declared as an
// interface so tests can inject a fake without touching the pool.
type KnowledgeBackend interface {
	UserFileIDs(ctx context.Context, userID, tenantID, projectID string) ([]string, error)
	Retriever() retriever.Retriever
	Rerank(ctx context.Context, query string, documents []*schema.Document) ([]*schema.Document, error)
	// ExpandParent prepends parent-chunk context. Called AFTER per-file
	// deduplication so siblings sharing a parent survive (architecture review B1).
	ExpandParent(ctx context.Context, docs []*schema.Document) []*schema.Document
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
	rewriter    QueryRewriter
}

// SetQueryRewriter enables multi-query retrieval (architecture review R6).
// When set, the tool rewrites the query, retrieves per variant in parallel with
// the original, and fuses the ranked lists with RRF. Leave nil (default) for the
// single-query path. Should be measured against recall@K before enabling.
func (t *KnowledgeSearchTool) SetQueryRewriter(r QueryRewriter) { t.rewriter = r }

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
			"min_score": {
				Desc:     "Minimum relevance score (0-1). Chunks below this are filtered out. Defaults to 0.3.",
				Type:     schema.Number,
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
		Query    string  `json:"query"`
		Limit    int     `json:"limit"`
		MinScore float64 `json:"min_score"`
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

	if t.svc == nil || t.svc.Retriever() == nil {
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

	// Build the query set: the original plus any rewrites (R6). Query
	// rewriting is opt-in; when no rewriter is wired this is just [args.Query].
	queries := []string{args.Query}
	if t.rewriter != nil {
		if rewrites, rerr := t.rewriter.Rewrite(ctx, args.Query); rerr != nil {
			slog.Warn("knowledge_search: query rewrite failed, using original query only", "error", rerr)
		} else if len(rewrites) > 0 {
			queries = append(queries, rewrites...)
			span.SetAttributes(attribute.Int("query.rewrite_count", len(rewrites)))
		}
	}

	// Retrieve per query and fuse ranked lists with RRF. Equal weight is given
	// to the original and each rewrite. When there is only one query, the
	// fusion function returns that list unchanged.
	grouped := make(map[string][]*schema.Document, len(queries))
	weights := make(map[string]float64, len(queries))
	for i, q := range queries {
		docs, derr := t.svc.Retriever().Retrieve(ctx, q,
			retriever.WithTopK(args.Limit),
			fp.WithFileIDs(fileIDs...),
		)
		if derr != nil {
			if i == 0 {
				span.SetStatus(codes.Error, "semantic search")
				span.RecordError(derr)
				slog.Warn("knowledge_search: semantic search failed", "error", derr, "user_id", userID)
				return "", fmt.Errorf("knowledge_search: search: %w", derr)
			}
			slog.Warn("knowledge_search: rewrite sub-query retrieve failed", "query_index", i, "error", derr)
			continue
		}
		key := fmt.Sprintf("q%d", i)
		grouped[key] = docs
		weights[key] = 1.0
	}
	if len(grouped) == 0 {
		return "", fmt.Errorf("knowledge_search: all retrieve attempts failed")
	}
	fusion := fp.WeightedRRFFusion(weights, 60)
	results, err := fusion(ctx, grouped)
	if err != nil {
		span.SetStatus(codes.Error, "rrf fusion")
		span.RecordError(err)
		return "", fmt.Errorf("knowledge_search: fuse: %w", err)
	}
	span.SetAttributes(attribute.Int("results.count", len(results)))

	// Rerank is optional and degrades silently by design (architecture review
	// M1/R7). Track the outcome so it is observable on the span and surfaced to
	// the agent when it was expected but did not apply.
	rerankApplied := false
	rerankFailed := false
	if t.rerankModel != nil && len(results) > 0 {
		reranked, err := t.rerankResults(ctx, args.Query, results)
		switch {
		case err != nil:
			rerankFailed = true
			slog.Warn("knowledge_search: rerank failed, using unranked results", "error", err)
		case len(reranked) > 0:
			results = reranked
			rerankApplied = true
		}
	}
	span.SetAttributes(
		attribute.Bool("rerank.configured", t.rerankModel != nil),
		attribute.Bool("rerank.applied", rerankApplied),
		attribute.Bool("rerank.failed", rerankFailed),
	)

	minScore := args.MinScore
	if minScore <= 0 {
		minScore = defaultMinScore
	}
	results = filterByMinScore(results, minScore)
	results = deduplicateByFile(results)
	// Parent-context expansion runs LAST, after per-file dedupe, so distinct
	// sibling chunks sharing a parent are not collapsed (architecture review B1).
	results = t.svc.ExpandParent(ctx, results)

	span.SetAttributes(
		attribute.Float64("min_score", minScore),
		attribute.Int("results.filtered_count", len(results)),
	)

	span.SetStatus(codes.Ok, "")
	if len(results) == 0 {
		return fmt.Sprintf("No relevant documents found for query: %q", args.Query), nil
	}

	out := FormatResults(results, args.Query)
	if rerankFailed {
		// Surface the silent degradation so the agent can decide whether to
		// widen the query — rerank was configured but did not run.
		out = "(note: reranking was unavailable, so results are in retrieval order; consider a more specific query.)\n\n" + out
	}
	return out, nil
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

// filterByMinScore removes documents whose score is below the threshold.
func filterByMinScore(docs []*schema.Document, minScore float64) []*schema.Document {
	if minScore <= 0 || len(docs) == 0 {
		return docs
	}
	out := make([]*schema.Document, 0, len(docs))
	for _, d := range docs {
		if d.Score() >= minScore {
			out = append(out, d)
		}
	}
	return out
}

// deduplicateByFile removes near-identical chunks from the same file.
// Within each file, chunks whose word-level Jaccard similarity > 0.8 are
// considered duplicates — only the highest-scored one is kept.
func deduplicateByFile(docs []*schema.Document) []*schema.Document {
	if len(docs) <= 1 {
		return docs
	}

	type entry struct {
		doc   *schema.Document
		words map[string]struct{}
	}

	// Group by file.
	byFile := make(map[string][]entry)
	var order []string
	for _, d := range docs {
		fid := fp.DocumentStringMetadata(d, fp.DocumentMetaFileID)
		if fid == "" {
			fid = "__no_file__"
		}
		if _, exists := byFile[fid]; !exists {
			order = append(order, fid)
		}
		byFile[fid] = append(byFile[fid], entry{doc: d, words: tokenizeWords(d.Content)})
	}

	out := make([]*schema.Document, 0, len(docs))
	for _, fid := range order {
		group := byFile[fid]
		// Mark duplicates within this group.
		kept := make([]bool, len(group))
		for i := range kept {
			kept[i] = true
		}
		for i := 0; i < len(group); i++ {
			if !kept[i] {
				continue
			}
			for j := i + 1; j < len(group); j++ {
				if !kept[j] {
					continue
				}
				if jaccard(group[i].words, group[j].words) > 0.8 {
					// Keep the one with higher score.
					if group[i].doc.Score() >= group[j].doc.Score() {
						kept[j] = false
					} else {
						kept[i] = false
						break
					}
				}
			}
		}
		for i, k := range kept {
			if k {
				out = append(out, group[i].doc)
			}
		}
	}
	return out
}

// tokenizeWords splits text on non-letter boundaries and returns a set.
func tokenizeWords(text string) map[string]struct{} {
	words := make(map[string]struct{})
	var buf strings.Builder
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			buf.WriteRune(unicode.ToLower(r))
		} else if buf.Len() > 0 {
			w := buf.String()
			if len(w) > 2 { // skip trivial words
				words[w] = struct{}{}
			}
			buf.Reset()
		}
	}
	if buf.Len() > 0 {
		w := buf.String()
		if len(w) > 2 {
			words[w] = struct{}{}
		}
	}
	return words
}

// jaccard computes Jaccard similarity between two word sets.
func jaccard(a, b map[string]struct{}) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}
	inter := 0
	for w := range a {
		if _, ok := b[w]; ok {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0.0
	}
	return float64(inter) / float64(union)
}
