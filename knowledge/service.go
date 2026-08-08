package knowledge

import (
	"context"
	"errors"
	"fmt"

	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	fp "github.com/kawai-network/fileprocessor"
	"github.com/yudaprama/egent-tools/rerank"
)

// Service wires a pgvector-backed semantic search over the existing lobehub
// schema (public.files, public.file_chunks, public.embeddings). It is scoped
// per user by looking up the user's file IDs and passing them as a filter to
// the eino Retriever.
//
// Keyword (BM25) search runs against Postgres full-text search
// (public.chunks.fts_vector) via PublicEmbeddingsStore.KeywordSearch — the
// authoritative, always-in-sync index. There is no separate local keyword
// projection. See the knowledge-base architecture review, recommendation R2.
//
// Service is long-lived; create once at process startup and Close on shutdown.
type Service struct {
	pool        *pgxpool.Pool
	retriever   retriever.Retriever
	rerankModel rerank.Reranker
	chunks      fp.ChunkStore
}

// NewService creates a knowledge service using the given shared Postgres pool.
// The embedder must produce 1024-dim vectors because PublicEmbeddingsStore
// targets public.embeddings which is constrained to vector(1024).
//
// If pool is nil, NewService returns nil, nil so callers can skip wiring
// the tool when no database is configured.
//
// The pool lifecycle is managed by the caller — Close does NOT close it.
func NewService(ctx context.Context, pool *pgxpool.Pool, embedder fp.Embedder) (*Service, error) {
	if pool == nil {
		return nil, nil
	}
	if embedder == nil {
		return nil, errors.New("knowledge: embedder is required")
	}
	if embedder.Dimension() != 0 && embedder.Dimension() != fp.DefaultEmbeddingDim {
		return nil, fmt.Errorf("knowledge: embedder dimension must be %d to match public.embeddings, got %d", fp.DefaultEmbeddingDim, embedder.Dimension())
	}

	vecStore, err := fp.NewPublicEmbeddingsStoreWithPool(ctx, pool, nil)
	if err != nil {
		return nil, fmt.Errorf("knowledge: create vector store: %w", err)
	}
	fileStore, err := fp.NewPostgresFileStoreWithPool(pool, fp.PostgresFileStoreOwner{UserID: "system-knowledge"})
	if err != nil {
		return nil, fmt.Errorf("knowledge: create file store: %w", err)
	}
	chunks := fileStore.ChunkStore()

	// Keyword (BM25) search is NOT passed explicitly: NewRetriever auto-detects
	// that PublicEmbeddingsStore implements KeywordSearcher (PG FTS over
	// chunks.fts_vector) and wires it. The fts_vector column is a STORED
	// generated column (migration 20260802000001), so it is always in sync
	// with chunk text — no rebuild goroutine needed. See architecture review R2.
	//
	// ExpandParent is false here: parent-context expansion runs as a
	// post-retrieval step in the tool (after per-file dedupe) so sibling
	// chunks sharing a parent are not collapsed. See architecture review B1.
	ret, err := fp.NewRetriever(ctx, &fp.RetrieverConfig{
		Store:        vecStore,
		Chunks:       chunks,
		Embedder:     embedder,
		ExpandParent: false,
	})
	if err != nil {
		return nil, fmt.Errorf("knowledge: create retriever: %w", err)
	}

	return &Service{
		pool:      pool,
		retriever: ret,
		chunks:    chunks,
	}, nil
}

// NewServiceWithRetriever is a test hook that bypasses DB setup and injects a
// custom Retriever. Pass nil for pool when the retriever does not need one.
func NewServiceWithRetriever(pool *pgxpool.Pool, ret retriever.Retriever) *Service {
	return &Service{
		pool:      pool,
		retriever: ret,
	}
}

// NewServiceWithRerank creates a knowledge service with an optional rerank model.
// The rerank model is used to rerank semantic search results for improved relevance.
func NewServiceWithRerank(ctx context.Context, pool *pgxpool.Pool, embedder fp.Embedder, rerankModel rerank.Reranker) (*Service, error) {
	svc, err := NewService(ctx, pool, embedder)
	if err != nil {
		return nil, err
	}
	if svc != nil {
		svc.rerankModel = rerankModel
	}
	return svc, nil
}

// Rerank applies the configured rerank model to Eino documents. Providers
// preserve each document's ID and metadata and update its Score(). Context
// placement is applied by KnowledgeSearchTool after this method returns.
func (s *Service) Rerank(ctx context.Context, query string, documents []*schema.Document) ([]*schema.Document, error) {
	if s == nil || s.rerankModel == nil || len(documents) == 0 {
		return nil, nil
	}
	return s.rerankModel.Rerank(ctx, query, documents)
}

// ExpandParent prepends parent-chunk context to each result. The tool calls
// this AFTER per-file deduplication (not inside Retrieve) so distinct sibling
// chunks that share a parent are not collapsed by the dedupe step. See the
// knowledge-base architecture review, finding B1.
func (s *Service) ExpandParent(ctx context.Context, docs []*schema.Document) []*schema.Document {
	if s == nil || s.chunks == nil {
		return docs
	}
	return fp.ExpandParentContext(ctx, s.chunks, docs)
}

// Close releases Service resources. The shared Postgres pool lifecycle is
// managed by the caller (it is not closed here).
func (s *Service) Close() error {
	return nil
}

// UserFileIDs returns the file IDs eligible for retrieval for the given user.
// This is the tenant filter applied before semantic search so users cannot
// read other users' chunks. When projectID is non-empty, the result is further
// scoped to files linked to that project (public.files.project_id), so a
// project's knowledge_search only surfaces that project's own attachments.
func (s *Service) UserFileIDs(ctx context.Context, userID, tenantID, projectID string) ([]string, error) {
	if s == nil || s.pool == nil {
		return nil, errors.New("knowledge: service not initialized")
	}
	if userID == "" {
		return nil, errors.New("knowledge: userID is required")
	}
	if tenantID == "" {
		return nil, errors.New("knowledge: tenantID is required")
	}
	var (
		rows pgx.Rows
		err  error
	)
	if projectID != "" {
		rows, err = s.pool.Query(ctx,
			`SELECT id FROM public.files
			 WHERE project_id = $1
			   AND tenant_id = $2
			   AND (user_id = $3 OR EXISTS (
			     SELECT 1 FROM project_members
			      WHERE project_id = $1 AND user_id = $3
			   ))`, projectID, tenantID, userID)
	} else {
		rows, err = s.pool.Query(ctx,
			`SELECT id FROM public.files WHERE user_id = $1 AND tenant_id = $2`, userID, tenantID)
	}
	if err != nil {
		return nil, fmt.Errorf("knowledge: list user files: %w", err)
	}
	defer rows.Close()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("knowledge: scan file id: %w", err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("knowledge: iterate user files: %w", err)
	}
	return ids, nil
}

// Retriever returns the eino Retriever for knowledge search.
func (s *Service) Retriever() retriever.Retriever {
	return s.retriever
}

// Pool returns the underlying pgx pool, mainly for tests.
func (s *Service) Pool() *pgxpool.Pool {
	return s.pool
}

// IsNotFound is a helper to detect pgx no-rows errors.
func IsNotFound(err error) bool {
	return errors.Is(err, pgx.ErrNoRows)
}
