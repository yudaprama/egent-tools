package knowledge

import (
	"context"
	"errors"
	"strings"
	"testing"

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

// fakeSearcher is a stand-in for fileprocessor.Searcher used to verify the
// tool wiring without a real Postgres + pgvector backend.
type fakeSearcher struct {
	results []*schema.Document
	err     error
	lastP   fp.SearchParamsSearcher
}

func (f *fakeSearcher) SemanticSearch(_ context.Context, p fp.SearchParamsSearcher) ([]*schema.Document, error) {
	f.lastP = p
	if f.err != nil {
		return nil, f.err
	}
	return f.results, nil
}

// fakeService implements KnowledgeBackend without needing a real pool.
type fakeService struct {
	searcher       Searcher
	fileIDs        []string
	projectFileIDs map[string][]string
	fileErr        error
	rerankResults  []*schema.Document
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

func (f *fakeService) Searcher() Searcher { return f.searcher }

func (f *fakeService) GetChunksByIDs(_ context.Context, _ []string) ([]fp.Chunk, error) {
	return nil, nil
}

func (f *fakeService) Rerank(_ context.Context, _ string, _ []*schema.Document) ([]*schema.Document, error) {
	return f.rerankResults, nil
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
	srch := &fakeSearcher{}
	svc := &fakeService{searcher: srch, fileIDs: []string{"f1"}}
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
	srch := &fakeSearcher{}
	svc := &fakeService{searcher: srch, fileIDs: nil}
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
	srch := &fakeSearcher{
		results: []*schema.Document{
			newTestDocument("chunk-1", "Project deadline is next Friday.", 0.91, "file-1", "plan.md"),
			newTestDocument("chunk-2", "Deploy via `make run`.", 0.78, "file-2", "README.md"),
		},
	}
	svc := &fakeService{
		searcher: srch,
		fileIDs:  []string{"file-1", "file-2"},
	}
	ctx := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	tl := NewKnowledgeSearchTool(svc, nil)
	out, err := tl.InvokableRun(ctx, `{"query":"deadline","limit":5}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(srch.lastP.FileIDs) != 2 {
		t.Errorf("expected 2 fileIDs passed to searcher, got %d", len(srch.lastP.FileIDs))
	}
	if srch.lastP.Limit != 5 {
		t.Errorf("expected limit 5, got %d", srch.lastP.Limit)
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
	srch := &fakeSearcher{err: errors.New("boom")}
	svc := &fakeService{searcher: srch, fileIDs: []string{"f1"}}
	ctx := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	tl := NewKnowledgeSearchTool(svc, nil)
	_, err := tl.InvokableRun(ctx, `{"query":"x"}`)
	if err == nil {
		t.Fatal("expected error from searcher")
	}
	if !strings.Contains(err.Error(), "boom") {
		t.Errorf("expected 'boom' in error, got %v", err)
	}
}

func TestKnowledgeSearchTool_LimitClampedToMax(t *testing.T) {
	srch := &fakeSearcher{}
	svc := &fakeService{searcher: srch, fileIDs: []string{"f1"}}
	ctx := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	tl := NewKnowledgeSearchTool(svc, nil)
	if _, err := tl.InvokableRun(ctx, `{"query":"x","limit":999}`); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if srch.lastP.Limit != 50 {
		t.Errorf("expected limit clamped to 50, got %d", srch.lastP.Limit)
	}
}

func TestKnowledgeSearchTool_DefaultLimit(t *testing.T) {
	srch := &fakeSearcher{results: []*schema.Document{newTestDocument("c1", "x", 0, "", "")}}
	svc := &fakeService{searcher: srch, fileIDs: []string{"f1"}}
	ctx := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	tl := NewKnowledgeSearchTool(svc, nil)
	if _, err := tl.InvokableRun(ctx, `{"query":"x"}`); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if srch.lastP.Limit != 10 {
		t.Errorf("expected default limit 10, got %d", srch.lastP.Limit)
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
	srch := &fakeSearcher{results: []*schema.Document{newTestDocument("c1", "x", 0, "", "")}}
	svc := &fakeService{
		searcher: srch,
		fileIDs:  []string{"user-wide-1"},
		projectFileIDs: map[string][]string{
			"prj-a": {"project-a-1", "project-a-2"},
		},
	}
	tl := NewKnowledgeSearchTool(svc, nil)

	ctxNoProject := memory.WithTenantID(memory.WithUserID(context.Background(), "u-1"), "t-1")
	if _, err := tl.InvokableRun(ctxNoProject, `{"query":"x"}`); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(srch.lastP.FileIDs) != 1 || srch.lastP.FileIDs[0] != "user-wide-1" {
		t.Errorf("expected user-wide file scope, got %v", srch.lastP.FileIDs)
	}

	ctxProject := memory.WithProjectID(ctxNoProject, "prj-a")
	if _, err := tl.InvokableRun(ctxProject, `{"query":"x"}`); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(srch.lastP.FileIDs) != 2 || srch.lastP.FileIDs[0] != "project-a-1" {
		t.Errorf("expected project-a file scope, got %v", srch.lastP.FileIDs)
	}
}
