// Package eval contains the R6 recall@K measurement harness. It is gated: the
// test skips when KAWAI_PG_DSN or EVAL_DATASET is unset, so the unit suite stays
// green without a populated DB. See docs/knowledge-r6-measurement.md.
package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
	"github.com/jackc/pgx/v5/pgxpool"
	fp "github.com/kawai-network/fileprocessor"

	"github.com/yudaprama/egent-tools/knowledge"
)

type evalQuery struct {
	Query    string   `json:"query"`
	Relevant []string `json:"relevant"`
}

type dataset struct {
	TenantID string     `json:"tenant_id"`
	UserID   string     `json:"user_id"`
	Queries  []evalQuery `json:"queries"`
}

func loadDataset(path string) (*dataset, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var d dataset
	if err := json.Unmarshal(raw, &d); err != nil {
		return nil, err
	}
	return &d, nil
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// buildService wires a knowledge Service against the configured DB + embeddings
// endpoint. Returns the service, its file IDs for the dataset user/tenant, and
// a cleanup.
func buildService(ctx context.Context, t *testing.T, ds *dataset) (*knowledge.Service, []string, func()) {
	t.Helper()
	dsn := os.Getenv("KAWAI_PG_DSN")
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("pgxpool: %v", err)
	}
	embedderURL := envOr("OPENAI_EMBEDDINGS_URL", "")
	if embedderURL == "" {
		if base := os.Getenv("MODEL_BASE_URL"); base != "" {
			embedderURL = strings.TrimRight(base, "/") + "/embeddings"
		}
	}
	embedder := fp.NewOpenAIEmbedder(
		embedderURL,
		envOr("OPENAI_API_KEY", os.Getenv("MODEL_API_KEY")),
		envOr("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"),
		fp.DefaultEmbeddingDim,
	)
	svc, err := knowledge.NewService(ctx, pool, embedder)
	if err != nil {
		pool.Close()
		t.Fatalf("knowledge.NewService: %v", err)
	}
	if svc == nil {
		pool.Close()
		t.Fatalf("knowledge service nil (pool closed)")
	}
	fileIDs, err := svc.UserFileIDs(ctx, ds.UserID, ds.TenantID, "")
	if err != nil {
		t.Fatalf("UserFileIDs: %v", err)
	}
	return svc, fileIDs, func() { _ = svc.Close(); pool.Close() }
}

func docIDs(docs []*schema.Document) []string {
	out := make([]string, 0, len(docs))
	for _, d := range docs {
		out = append(out, d.ID)
	}
	return out
}

func retrieve(ctx context.Context, ret retriever.Retriever, query string, fileIDs []string, tenantID string, k int) ([]*schema.Document, error) {
	return ret.Retrieve(ctx, query,
		retriever.WithTopK(k),
		fp.WithFileIDs(fileIDs...),
		fp.WithTenantID(tenantID),
	)
}

// runMulti does the tool's multi-query RRF path: the original query plus
// rewrites, each retrieved, fused with equal-weight RRF.
func runMulti(ctx context.Context, ret retriever.Retriever, rw knowledge.QueryRewriter, query string, fileIDs []string, tenantID string, k int) ([]*schema.Document, error) {
	queries := []string{query}
	if rw != nil {
		if rewrites, err := rw.Rewrite(ctx, query); err == nil {
			queries = append(queries, rewrites...)
		}
	}
	grouped := make(map[string][]*schema.Document, len(queries))
	weights := make(map[string]float64, len(queries))
	for i, q := range queries {
		docs, err := retrieve(ctx, ret, q, fileIDs, tenantID, k)
		if err != nil || len(docs) == 0 {
			continue
		}
		key := fmt.Sprintf("q%d", i)
		grouped[key] = docs
		weights[key] = 1.0
	}
	if len(grouped) == 0 {
		return nil, nil
	}
	if len(grouped) == 1 {
		for _, docs := range grouped {
			return docs, nil
		}
	}
	return fp.WeightedRRFFusion(weights, 60)(ctx, grouped)
}

func recallAtK(returned, relevant []string, k int) float64 {
	if len(relevant) == 0 {
		return 0
	}
	rel := make(map[string]bool, len(relevant))
	for _, id := range relevant {
		rel[id] = true
	}
	if k > len(returned) {
		k = len(returned)
	}
	hits := 0
	for _, id := range returned[:k] {
		if rel[id] {
			hits++
		}
	}
	return float64(hits) / float64(len(relevant))
}

func mrr(returned, relevant []string) float64 {
	rel := make(map[string]bool, len(relevant))
	for _, id := range relevant {
		rel[id] = true
	}
	for i, id := range returned {
		if rel[id] {
			return 1.0 / float64(i+1)
		}
	}
	return 0
}

// TestBenchmark_R6Recall measures single-query vs multi-query (R6) retrieval
// recall@K and MRR over EVAL_DATASET. Skips without KAWAI_PG_DSN + EVAL_DATASET.
func TestBenchmark_R6Recall(t *testing.T) {
	dsn := os.Getenv("KAWAI_PG_DSN")
	dsPath := os.Getenv("EVAL_DATASET")
	if dsn == "" || dsPath == "" {
		t.Skip("KAWAI_PG_DSN or EVAL_DATASET unset; skipping R6 recall benchmark — seed data with: go run ./knowledge/eval/seed -dsn $KAWAI_PG_DSN -out dataset.json")
	}
	ds, err := loadDataset(dsPath)
	if err != nil {
		t.Fatalf("load dataset: %v", err)
	}
	if len(ds.Queries) == 0 {
		t.Fatal("dataset has no queries")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	svc, fileIDs, cleanup := buildService(ctx, t, ds)
	defer cleanup()

	var rw knowledge.QueryRewriter
	if base := os.Getenv("MODEL_BASE_URL"); base != "" && os.Getenv("MODEL_NAME") != "" {
		rw = knowledge.NewLLMQueryRewriter(knowledge.LLMQueryRewriterConfig{
			BaseURL: base,
			APIKey:  envOr("MODEL_API_KEY", os.Getenv("OPENAI_API_KEY")),
			Model:   os.Getenv("MODEL_NAME"),
		})
	}
	if rw == nil {
		t.Log("MODEL_BASE_URL/MODEL_NAME unset — multi-query will run WITHOUT rewriting (single-query only); set them for a real R6 comparison")
	}

	const k = 10
	var sumRecallSingle, sumRecallMulti, sumMRRSingle, sumMRRMulti float64
	fmt.Printf("\n%-50s %8s %8s %8s %8s\n", "query", "R@K_sg", "R@K_mq", "MRR_sg", "MRR_mq")
	for _, q := range ds.Queries {
		single, err := retrieve(ctx, svc.Retriever(), q.Query, fileIDs, ds.TenantID, k)
		if err != nil {
			t.Logf("single retrieve %q: %v", q.Query, err)
		}
		multi, err := runMulti(ctx, svc.Retriever(), rw, q.Query, fileIDs, ds.TenantID, k)
		if err != nil {
			t.Logf("multi retrieve %q: %v", q.Query, err)
		}
		sIDs, mIDs := docIDs(single), docIDs(multi)
		rs, rm := recallAtK(sIDs, q.Relevant, k), recallAtK(mIDs, q.Relevant, k)
		ms, mm := mrr(sIDs, q.Relevant), mrr(mIDs, q.Relevant)
		sumRecallSingle += rs
		sumRecallMulti += rm
		sumMRRSingle += ms
		sumMRRMulti += mm
		label := q.Query
		if len(label) > 48 {
			label = label[:48] + ".."
		}
		fmt.Printf("%-50s %8.2f %8.2f %8.2f %8.2f\n", label, rs, rm, ms, mm)
	}
	n := float64(len(ds.Queries))
	fmt.Printf("\n%-50s %8.2f %8.2f %8.2f %8.2f\n",
		"MEAN", sumRecallSingle/n, sumRecallMulti/n, sumMRRSingle/n, sumMRRMulti/n)

	ratio := 0.0
	if sumRecallSingle > 0 {
		ratio = sumRecallMulti / sumRecallSingle
	}
	fmt.Printf("recall@%d ratio (multi/single) = %.2f (enable R6 if >= 1.10)\n", k, ratio)
}
