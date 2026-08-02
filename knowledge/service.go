package knowledge

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/cloudwego/eino/schema"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	fp "github.com/kawai-network/fileprocessor"
	"github.com/yudaprama/egent-tools/rerank"
)

// Searcher is the subset of fileprocessor.Searcher that the knowledge tool
// needs. Defined as an interface so tests can inject a fake.
type Searcher interface {
	SemanticSearch(ctx context.Context, p fp.SearchParamsSearcher) ([]*schema.Document, error)
}

// Service wires a pgvector-backed semantic search over the existing lobehub
// schema (public.files, public.file_chunks, public.embeddings). It is scoped
// per user by looking up the user's file IDs and passing them as a filter to
// Searcher.SemanticSearch.
//
// Service is long-lived; create once at process startup and Close on shutdown.
type Service struct {
	pool         *pgxpool.Pool
	embedder     fp.Embedder
	vecStore     fp.VectorStore
	chunkStore   fp.ChunkStore
	searcher     Searcher
	rerankModel  rerank.Reranker
	keywordIndex *fp.SQLiteKeywordIndex
	keywordStop  chan struct{}
	keywordWG    sync.WaitGroup
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
	if embedder.Dimension() != 0 && embedder.Dimension() != 1024 {
		return nil, fmt.Errorf("knowledge: embedder dimension must be 1024 to match public.embeddings, got %d", embedder.Dimension())
	}

	vecStore, err := fp.NewPublicEmbeddingsStoreWithPool(ctx, pool, nil)
	if err != nil {
		return nil, fmt.Errorf("knowledge: create vector store: %w", err)
	}
	fileStore, err := fp.NewPostgresFileStoreWithPool(pool, fp.PostgresFileStoreOwner{UserID: "system-knowledge"})
	if err != nil {
		return nil, fmt.Errorf("knowledge: create file store: %w", err)
	}
	keywordIndexPath := os.Getenv("KAWAI_KNOWLEDGE_FTS_PATH")
	if keywordIndexPath == "" {
		keywordIndexPath = filepath.Join(".plano", "run", "knowledge-fts.db")
	}
	keywordIndex, err := fp.NewSQLiteKeywordIndex(keywordIndexPath)
	if err != nil {
		return nil, fmt.Errorf("knowledge: create SQLite keyword index: %w", err)
	}
	if err := keywordIndex.RefreshFromPostgres(ctx, pool); err != nil {
		_ = keywordIndex.Close()
		return nil, fmt.Errorf("knowledge: initial SQLite keyword index refresh: %w", err)
	}
	searcher := fp.NewSearcherWithKeywordSearcher(vecStore, fileStore.ChunkStore(), embedder, keywordIndex)

	svc := &Service{
		pool:         pool,
		embedder:     embedder,
		vecStore:     vecStore,
		chunkStore:   fileStore.ChunkStore(),
		searcher:     searcher,
		keywordIndex: keywordIndex,
		keywordStop:  make(chan struct{}),
	}
	svc.keywordWG.Add(1)
	go svc.refreshKeywordIndex()
	return svc, nil
}

func (s *Service) refreshKeywordIndex() {
	defer s.keywordWG.Done()
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
			if err := s.keywordIndex.RefreshFromPostgres(ctx, s.pool); err != nil {
				slog.Warn("knowledge: refresh SQLite keyword index failed", "error", err)
			}
			cancel()
		case <-s.keywordStop:
			return
		}
	}
}

// NewServiceWithSearcher is a test hook that bypasses DB setup and injects a
// custom Searcher. Pass nil for pool when the searcher does not need one.
func NewServiceWithSearcher(pool *pgxpool.Pool, searcher Searcher) *Service {
	return &Service{
		pool:     pool,
		searcher: searcher,
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

// Rerank applies the rerank model to search results if configured.
// Returns reranked results sorted by relevance score, or the original
// results if no rerank model is configured or the rerank call fails.
func (s *Service) Rerank(ctx context.Context, query string, documents []*schema.Document) ([]*schema.Document, error) {
	if s == nil || s.rerankModel == nil || len(documents) == 0 {
		return nil, nil
	}
	return s.rerankModel.Rerank(ctx, query, documents)
}

// Close stops the keyword-index refresher and closes its local SQLite file.
// The shared Postgres pool lifecycle is managed by the caller.
func (s *Service) Close() error {
	if s == nil {
		return nil
	}
	if s.keywordStop != nil {
		close(s.keywordStop)
		s.keywordWG.Wait()
	}
	if s.keywordIndex != nil {
		return s.keywordIndex.Close()
	}
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

// Searcher returns the underlying Searcher.
func (s *Service) Searcher() Searcher {
	return s.searcher
}

// GetChunksByIDs fetches chunks by their IDs, used for parent context expansion.
func (s *Service) GetChunksByIDs(ctx context.Context, ids []string) ([]fp.Chunk, error) {
	if s == nil || s.chunkStore == nil {
		return nil, nil
	}
	return s.chunkStore.GetChunksByIDs(ctx, ids)
}

// Pool returns the underlying pgx pool, mainly for tests.
func (s *Service) Pool() *pgxpool.Pool {
	return s.pool
}

// IsNotFound is a helper to detect pgx no-rows errors.
func IsNotFound(err error) bool {
	return errors.Is(err, pgx.ErrNoRows)
}
