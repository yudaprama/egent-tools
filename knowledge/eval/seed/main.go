// Command seed generates a synthetic corpus + labeled eval dataset for R6
// recall@K benchmarking. It inserts test files/chunks/embeddings into the DB
// and writes dataset.json with deterministic relevant-chunk IDs.
//
// Usage:
//   go run ./knowledge/eval/seed -dsn "$KAWAI_PG_DSN" -out dataset.json
package main

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/big"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	tenantID = "00000000-0000-0000-0000-000000000001"
	userID   = "00000000-0000-0000-0000-000000000099"
	numFiles = 20
	chunksPerFile = 5
	dim = 1024
)

var chunkTexts = [chunksPerFile]string{
	"Kubernetes deployment uses Helm charts. The chart templates are in deploy/charts/kawai. Run helm upgrade to deploy.",
	"CI/CD pipeline runs on GitHub Actions. The workflow is defined in .github/workflows/ci.yml. Push to main triggers deploy.",
	"Database connection pooling uses pgxpool. The pool is initialized at startup with KAWAI_PG_DSN. Max connections default 20.",
	"Authentication flows use Oathkeeper for edge auth. Sessions carry active_workspace_id in metadata_public. No bearer tokens.",
	"Rate limiting is configured in envoy.yaml. Per-workspace quotas are enforced at the edge. Default 100 req/s per user.",
}

type evalQuery struct {
	Query    string   `json:"query"`
	Relevant []string `json:"relevant"`
}

type dataset struct {
	TenantID string     `json:"tenant_id"`
	UserID   string     `json:"user_id"`
	Queries  []evalQuery `json:"queries"`
}

func randVec() []float32 {
	v := make([]float32, dim)
	for i := range v {
		n, _ := rand.Int(rand.Reader, big.NewInt(20000))
		v[i] = float32(n.Int64()-10000) / 10000.0
	}
	return v
}

func vecLiteral(v []float32) string {
	s := "["
	for i, f := range v {
		if i > 0 {
			s += ","
		}
		s += fmt.Sprintf("%f", f)
	}
	return s + "]"
}

func main() {
	dsn := flag.String("dsn", "", "Postgres DSN (or set KAWAI_PG_DSN)")
	out := flag.String("out", "dataset.json", "Output eval dataset JSON path")
	flag.Parse()
	if *dsn == "" {
		*dsn = os.Getenv("KAWAI_PG_DSN")
	}
	if *dsn == "" {
		log.Fatal("provide -dsn or set KAWAI_PG_DSN")
	}

	ctx := context.Background()
	pool, err := pgxpool.New(ctx, *dsn)
	if err != nil {
		log.Fatalf("pgxpool: %v", err)
	}
	defer pool.Close()

	tx, err := pool.Begin(ctx)
	if err != nil {
		log.Fatalf("begin: %v", err)
	}
	defer tx.Rollback(ctx)

	// Ensure tenant + user exist (idempotent).
	_, _ = tx.Exec(ctx, `INSERT INTO public.tenants (id,name,slug,created_at,updated_at) VALUES ($1,'eval-tenant','eval',now(),now()) ON CONFLICT (id) DO NOTHING`, tenantID)
	_, _ = tx.Exec(ctx, `INSERT INTO public.tenant_members (tenant_id,user_id,role,created_at) VALUES ($1,$2,'OWNER',now()) ON CONFLICT (tenant_id,user_id) DO NOTHING`, tenantID, userID)

	// Insert files.
	fileIDs := make([]string, numFiles)
	for i := range fileIDs {
		var id string
		err := tx.QueryRow(ctx,
			`INSERT INTO public.files (id,user_id,tenant_id,name,status,created_at,updated_at)
			 VALUES (gen_random_uuid(),$1,$2,$3,'DONE',now(),now()) RETURNING id`,
			userID, tenantID, fmt.Sprintf("eval-doc-%02d.txt", i+1),
		).Scan(&id)
		if err != nil {
			log.Fatalf("insert file %d: %v", i+1, err)
		}
		fileIDs[i] = id
	}

	// Insert chunks + file_chunks + random embeddings.
	// Track chunk IDs per file for the eval dataset.
	fileChunks := make([][]string, numFiles) // fileChunks[i] = chunk IDs for file i
	for i := range fileChunks {
		fileChunks[i] = make([]string, chunksPerFile)
	}

	for fi, fid := range fileIDs {
		for ci := 0; ci < chunksPerFile; ci++ {
			var chunkID string
			err := tx.QueryRow(ctx,
				`INSERT INTO public.chunks (id,document_id,"text","index","type",created_at,updated_at,metadata)
				 VALUES (gen_random_uuid(),$1,$2,$3,'text',now(),now(),'{}') RETURNING id`,
				fid, chunkTexts[ci], ci,
			).Scan(&chunkID)
			if err != nil {
				log.Fatalf("insert chunk: %v", err)
			}
			fileChunks[fi][ci] = chunkID

			_, err = tx.Exec(ctx,
				`INSERT INTO public.file_chunks (chunk_id,file_id) VALUES ($1,$2)`,
				chunkID, fid,
			)
			if err != nil {
				log.Fatalf("insert file_chunk: %v", err)
			}

			vec := randVec()
			_, err = tx.Exec(ctx,
				`INSERT INTO public.embeddings (chunk_id,embeddings,model,created_at)
				 VALUES ($1,$2::vector(1024),'fileprocessor',now())`,
				chunkID, vecLiteral(vec),
			)
			if err != nil {
				log.Fatalf("insert embedding: %v", err)
			}
		}
	}

	// Update file stats.
	_, err = tx.Exec(ctx, `
		UPDATE public.files SET
			chunk_count = (SELECT COUNT(*) FROM public.file_chunks WHERE file_id = public.files.id),
			chunking_status = 'DONE', embedding_status = 'DONE', updated_at = now()
		WHERE user_id = $1 AND tenant_id = $2`, userID, tenantID)
	if err != nil {
		log.Fatalf("update stats: %v", err)
	}

	if err := tx.Commit(ctx); err != nil {
		log.Fatalf("commit: %v", err)
	}

	// Build eval dataset: each query targets chunk index 0 of file i (deterministic).
	queries := make([]evalQuery, numFiles)
	for i := range queries {
		queries[i] = evalQuery{
			Query:    fmt.Sprintf("topic %d deployment or configuration", i+1),
			Relevant: []string{fileChunks[i][0]}, // first chunk of each file
		}
	}

	ds := dataset{
		TenantID: tenantID,
		UserID:   userID,
		Queries:  queries,
	}
	raw, _ := json.MarshalIndent(ds, "", "  ")
	if err := os.WriteFile(*out, raw, 0644); err != nil {
		log.Fatalf("write dataset: %v", err)
	}
	fmt.Printf("seeded %d files × %d chunks = %d chunks\n", numFiles, chunksPerFile, numFiles*chunksPerFile)
	fmt.Printf("wrote eval dataset to %s (%d queries)\n", *out, len(queries))
}
