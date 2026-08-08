package knowledge

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
	fp "github.com/kawai-network/fileprocessor"

	"github.com/yudaprama/egent-tools/memory"
)

func newTestDocument(id, content string, score float64, fileID, fileName string) *schema.Document {
	doc := (&schema.Document{ID: id, Content: content, MetaData: map[string]any{}}).WithScore(score)
	if fileID != "" {
		doc.MetaData[fp.DocumentMetaFileID] = fileID
	}
	if fileName != "" {
		doc.MetaData[fp.DocumentMetaFileName] = fileName
	}
	return doc
}

// fakeRetriever is a stand-in for eino retriever.Retriever used to verify the
// tool wiring without a real Postgres + pgvector backend.
type fakeRetriever struct {
	results    []*schema.Document
	err        error
	lastQuery  string
	lastFileIDs []string
	lastLimit  int
}

func (f *fakeRetriever) Retrieve(_ context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	f.lastQuery = query
	// Decode options to capture what the tool passed.
	co := retriever.GetCommonOptions(&retriever.Options{}, opts...)
	if co != nil && co.TopK != nil {
		f.lastLimit = *co.TopK
	}
	io := retriever.GetImplSpecificOptions(&fp.HybridOptions{}, opts...)
	if io != nil {
		f.lastFileIDs = io.FileIDs
	}
	if f.err != nil {
		return nil, f.err
	}
	return f.results, nil
}

// fakeService implements KnowledgeBackend without needing a real pool.
type fakeService struct {
	ret            retriever.Retriever
	fileIDs        []string
	projectFileIDs map[string][]string
	fileErr        error
	rerankResults  []*schema.Document
	parentText     string
}

func (f *fakeService) UserFileIDs(_ context.Context, _, _, projectID string) ([]string, error) {
	if f.fileErr != nil {
		return nil, f.fileErr
	}
	if projectID != "" && f.projectFileIDs != nil {
		return f.projectFileIDs[projectID], nil
	}
	return f.fileIDs, nil
}

func (f *fakeService) Retriever() retriever.Retriever { return f.ret }

func (f *fakeService) Rerank(_ context.Context, _ string, _ []*schema.Document) ([]*schema.Document, error) {
	return f.rerankResults, nil
}

// ExpandParent mimics fp.ExpandParentContext: when parentText is set, it
// prepends the shared parent context to any doc carrying a DocumentMetaParentID.
func (f *fakeService) ExpandParent(_ context.Context, docs []*schema.Document) []*schema.Document {
	if f.parentText == "" {
		return docs
	}
	for _, doc := range docs {
		if fp.DocumentStringMetadata(doc, fp.DocumentMetaParentID) != "" {
			doc.Content = "[Context from section]\n" + f.parentText + "\n\n[Excerpt]\n" + doc.Content
		}
	}
	return docs
}

func TestKnowledgeSearchTool_RerankPlacesHighScoresAtContextEdges(t *testing.T) {
	svc := &fakeService{
		rerankResults: []*schema.Document{
			newTestDocument("high", "high", 0.9, "", ""),
			newTestDocument("medium", "medium", 0.6, "", ""),
			newTestDocument("low", "low", 0.2, "", ""),
		},
	}
	tl := &KnowledgeSearchTool{svc: svc}

	got, err := tl.rerankResults(context.Background(), "query", svc.rerankResults)
	if err != nil {
		t.Fatalf("rerankResults: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 documents, got %d", len(got))
	}
	// score transformer order for [high, medium, low] is [high, low, medium].
	if got[0].ID != "high" || got[1].ID != "low" || got[2].ID != "medium" {
		t.Fatalf("unexpected context placement: %q, %q, %q", got[0].ID, got[1].ID, got[2].ID)
	}
}

func TestKnowledgeSearchTool_Info(t *testing.T) {
	tl := NewKnowledgeSearchTool(nil, nil)
	info, err := tl.Info(context.Background())
	if err != nil {
		t.Fatalf("Info: %v", err)
	}
	if info.Name != "knowledge_search" {
		t.Errorf("expected name knowledge_search, got %q", info.Name)
	}
	if info.ParamsOneOf == nil {
		t.Fatal("expected ParamsOneOf to be set")
	}
}

func TestKnowledgeSearchTool_NoService(t *testing.T) {
	tl := NewKnowledgeSearchTool(nil, nil)
	out, err := tl.InvokableRun(context.Background(), `{"query":"x"}`)
	if err != nil {
		t.Fatalf("expected nil err when no service, got %v", err)
	}
	if !strings.Contains(out, "not configured") {
		t.Errorf("expected 'not configured' message, got %q", out)
	}
}

func TestKnowledgeSearchTool_NoUserID(t *testing.T) {
	ret := &fakeRetriever{}
	svc := &fakeService{ret: ret, fileIDs: []string{"f1"}}
	tl := NewKnowledgeSearchTool(svc, nil)
	_, err := tl.InvokableRun(context.Background(), `{"query":"hello"}`)
	if err == nil {
		t.Fatal("expected error when no user_id in context")
	}
	if !strings.Contains(err.Error(), "user_id") {
		t.Errorf("expected user_id error, got %v", err)
	}
}

func TestKnowledgeSearchTool_NoFiles(t *testing.T) {
	ret := &fakeRetriever{}
	svc := &fakeService{ret: ret, fileIDs: nil}
	ctx := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	tl := NewKnowledgeSearchTool(svc, nil)
	out, err := tl.InvokableRun(ctx, `{"query":"hello"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "No documents found") {
		t.Errorf("expected 'No documents found' message, got %q", out)
	}
}

func TestKnowledgeSearchTool_SearchAndFormat(t *testing.T) {
	ret := &fakeRetriever{
		results: []*schema.Document{
			newTestDocument("chunk-1", "Project deadline is next Friday.", 0.91, "file-1", "plan.md"),
			newTestDocument("chunk-2", "Deploy via `make run`.", 0.78, "file-2", "README.md"),
		},
	}
	svc := &fakeService{
		ret:     ret,
		fileIDs: []string{"file-1", "file-2"},
	}
	ctx := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	tl := NewKnowledgeSearchTool(svc, nil)
	out, err := tl.InvokableRun(ctx, `{"query":"deadline","limit":5}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(ret.lastFileIDs) != 2 {
		t.Errorf("expected 2 fileIDs passed to retriever, got %d", len(ret.lastFileIDs))
	}
	if ret.lastLimit != 5 {
		t.Errorf("expected limit 5, got %d", ret.lastLimit)
	}
	if !strings.Contains(out, "plan.md") {
		t.Errorf("expected output to contain 'plan.md', got %q", out)
	}
	if !strings.Contains(out, "README.md") {
		t.Errorf("expected output to contain 'README.md', got %q", out)
	}
	if !strings.Contains(out, "0.910") {
		t.Errorf("expected similarity score 0.910, got %q", out)
	}
}

func TestKnowledgeSearchTool_EmptyQuery(t *testing.T) {
	tl := NewKnowledgeSearchTool(nil, nil)
	_, err := tl.InvokableRun(context.Background(), `{"query":"  "}`)
	if err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestKnowledgeSearchTool_SearchError(t *testing.T) {
	ret := &fakeRetriever{err: errors.New("boom")}
	svc := &fakeService{ret: ret, fileIDs: []string{"f1"}}
	ctx := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	tl := NewKnowledgeSearchTool(svc, nil)
	_, err := tl.InvokableRun(ctx, `{"query":"x"}`)
	if err == nil {
		t.Fatal("expected error from retriever")
	}
	if !strings.Contains(err.Error(), "boom") {
		t.Errorf("expected 'boom' in error, got %v", err)
	}
}

func TestKnowledgeSearchTool_LimitClampedToMax(t *testing.T) {
	ret := &fakeRetriever{}
	svc := &fakeService{ret: ret, fileIDs: []string{"f1"}}
	ctx := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	tl := NewKnowledgeSearchTool(svc, nil)
	if _, err := tl.InvokableRun(ctx, `{"query":"x","limit":999}`); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ret.lastLimit != 50 {
		t.Errorf("expected limit clamped to 50, got %d", ret.lastLimit)
	}
}

func TestKnowledgeSearchTool_DefaultLimit(t *testing.T) {
	ret := &fakeRetriever{results: []*schema.Document{newTestDocument("c1", "x", 0, "", "")}}
	svc := &fakeService{ret: ret, fileIDs: []string{"f1"}}
	ctx := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	tl := NewKnowledgeSearchTool(svc, nil)
	if _, err := tl.InvokableRun(ctx, `{"query":"x"}`); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ret.lastLimit != 10 {
		t.Errorf("expected default limit 10, got %d", ret.lastLimit)
	}
}

func TestFormatResults_Empty(t *testing.T) {
	out := FormatResults(nil, "nothing")
	if !strings.Contains(out, "0 hits") {
		t.Errorf("expected 0 hits in output, got %q", out)
	}
}

func TestFormatResults_SourceLabelFallback(t *testing.T) {
	out := FormatResults([]*schema.Document{
		{ID: "chunk-x", Content: "no file"},
	}, "q")
	if !strings.Contains(out, "chunk: chunk-x") {
		t.Errorf("expected fallback source label, got %q", out)
	}
}

// TestKnowledgeSearchTool_ProjectScoping verifies that a project id carried in
// context scopes retrieval to that project's files rather than the whole user.
func TestKnowledgeSearchTool_ProjectScoping(t *testing.T) {
	ret := &fakeRetriever{results: []*schema.Document{newTestDocument("c1", "x", 0, "", "")}}
	svc := &fakeService{
		ret:     ret,
		fileIDs: []string{"user-wide-1"},
		projectFileIDs: map[string][]string{
			"prj-a": {"project-a-1", "project-a-2"},
		},
	}
	tl := NewKnowledgeSearchTool(svc, nil)

	ctxNoProject := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	if _, err := tl.InvokableRun(ctxNoProject, `{"query":"x"}`); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ret.lastFileIDs) != 1 || ret.lastFileIDs[0] != "user-wide-1" {
		t.Errorf("expected user-wide file scope, got %v", ret.lastFileIDs)
	}

	ctxProject := memory.WithProjectID(ctxNoProject, "prj-a")
	if _, err := tl.InvokableRun(ctxProject, `{"query":"x"}`); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ret.lastFileIDs) != 2 || ret.lastFileIDs[0] != "project-a-1" {
		t.Errorf("expected project-a file scope, got %v", ret.lastFileIDs)
	}
}

func TestFilterByMinScore(t *testing.T) {
	docs := []*schema.Document{
		newTestDocument("high", "high", 0.9, "", ""),
		newTestDocument("mid", "mid", 0.5, "", ""),
		newTestDocument("low", "low", 0.1, "", ""),
	}
	got := filterByMinScore(docs, 0.3)
	if len(got) != 2 {
		t.Fatalf("expected 2 docs after min_score=0.3, got %d", len(got))
	}
	if got[0].ID != "high" || got[1].ID != "mid" {
		t.Fatalf("unexpected remaining docs: %v, %v", got[0].ID, got[1].ID)
	}
}

func TestFilterByMinScore_NoThreshold(t *testing.T) {
	docs := []*schema.Document{
		newTestDocument("a", "a", 0.1, "", ""),
		newTestDocument("b", "b", 0.9, "", ""),
	}
	got := filterByMinScore(docs, 0)
	if len(got) != 2 {
		t.Fatalf("expected all docs when threshold=0, got %d", len(got))
	}
}

func TestDeduplicateByFile(t *testing.T) {
	// These two chunks share >80% words (only differs in one word).
	docs := []*schema.Document{
		newTestDocument("c1", "The project deadline is next Friday and the budget has been approved by the team", 0.9, "f1", "plan.md"),
		newTestDocument("c2", "The project deadline is next Friday and the budget has been approved by management", 0.8, "f1", "plan.md"),
		newTestDocument("c3", "Deploy via make run in the CI pipeline", 0.7, "f2", "README.md"),
	}
	got := deduplicateByFile(docs)
	if len(got) != 2 {
		t.Fatalf("expected 2 docs after dedup (2 similar in f1 → 1), got %d", len(got))
	}
	// The higher-scored one should be kept.
	found := false
	for _, d := range got {
		if d.ID == "c1" {
			found = true
		}
	}
	if !found {
		t.Fatal("expected c1 (higher score) to be kept after dedup")
	}
}

func TestDeduplicateByFile_DifferentFiles(t *testing.T) {
	docs := []*schema.Document{
		newTestDocument("c1", "The project deadline is next Friday", 0.9, "f1", "a.md"),
		newTestDocument("c2", "The project deadline is next Friday", 0.8, "f2", "b.md"),
	}
	got := deduplicateByFile(docs)
	if len(got) != 2 {
		t.Fatalf("expected no dedup across different files, got %d", len(got))
	}
}

func TestTokenizeWords(t *testing.T) {
	words := tokenizeWords("Hello, World! This is a test.")
	if _, ok := words["hello"]; !ok {
		t.Error("expected 'hello' in token set")
	}
	if _, ok := words["world"]; !ok {
		t.Error("expected 'world' in token set")
	}
	// "a" and "is" are too short (<=2 chars) and should be skipped.
	if _, ok := words["a"]; ok {
		t.Error("expected 'a' to be skipped (too short)")
	}
}

func TestJaccard(t *testing.T) {
	a := map[string]struct{}{"foo": {}, "bar": {}, "baz": {}}
	b := map[string]struct{}{"foo": {}, "bar": {}, "qux": {}}
	got := jaccard(a, b)
	// intersection = {foo, bar} = 2, union = {foo, bar, baz, qux} = 4 → 0.5
	if got != 0.5 {
		t.Fatalf("expected jaccard=0.5, got %v", got)
	}
}

func TestJaccard_Empty(t *testing.T) {
	if jaccard(nil, nil) != 1.0 {
		t.Fatal("expected 1.0 for two empty sets")
	}
	if jaccard(map[string]struct{}{"a": {}}, nil) != 0.0 {
		t.Fatal("expected 0.0 for non-empty vs empty")
	}
}

// TestKnowledgeSearch_DoesNotCollapseSiblingSections is the regression test for
// architecture-review B1: parent-context expansion must run AFTER per-file
// deduplication. The three chunks below are distinct siblings (child-only
// Jaccard ≈ 0) sharing one parent. Once the long parent text is prepended they
// become >0.8 similar, so if dedupe ran on the expanded content two siblings
// would be dropped. The fix expands after dedupe, so all three survive.
func TestKnowledgeSearch_DoesNotCollapseSiblingSections(t *testing.T) {
	// Long shared parent section (30 distinct words). Short, distinct child
	// excerpts whose only overlap with each other is via the parent.
	parentText := "quarterly financial report revenue expenses projections " +
		"departments marketing operations engineering sales growth decline " +
		"margin volume capacity utilization forecast budget allocation " +
		"resources staffing hiring compliance audit treasury payroll " +
		"logistics procurement facilities"

	mk := func(id, content string, score float64) *schema.Document {
		return (&schema.Document{
			ID:      id,
			Content: content,
			MetaData: map[string]any{
				fp.DocumentMetaFileID:   "file-1",
				fp.DocumentMetaFileName: "report.md",
				fp.DocumentMetaParentID: "parent-1",
			},
		}).WithScore(score)
	}

	ret := &fakeRetriever{results: []*schema.Document{
		mk("c1", "revenue increased significantly", 0.9),
		mk("c2", "expenses decreased notably", 0.8),
		mk("c3", "projections improved markedly", 0.7),
	}}
	svc := &fakeService{
		ret:        ret,
		fileIDs:    []string{"file-1"},
		parentText: parentText,
	}
	ctx := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	tl := NewKnowledgeSearchTool(svc, nil)

	out, err := tl.InvokableRun(ctx, `{"query":"financial","limit":10}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Each child has a unique word absent from the parent; all three must
	// survive. If expansion ran before dedupe, only one sibling would remain.
	for _, want := range []string{"increased", "decreased", "improved"} {
		if !strings.Contains(out, want) {
			t.Errorf("B1 regression: %q missing — sibling section was collapsed by dedupe-after-expand.\noutput:\n%s", want, out)
		}
	}
}
