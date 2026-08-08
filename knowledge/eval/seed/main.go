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
	tenantID      = "00000000-0000-0000-0000-000000000001"
	userID        = "00000000-0000-0000-0000-000000000099"
	numFiles      = 20
	chunksPerFile = 5
	dim           = 1024
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

	// Idempotent: delete previous eval data.
	_, _ = pool.Exec(ctx, `DELETE FROM public.embeddings WHERE user_id = $1`, userID)
	_, _ = pool.Exec(ctx, `DELETE FROM public.file_chunks WHERE user_id = $1`, userID)
	_, _ = pool.Exec(ctx, `DELETE FROM public.chunks WHERE user_id = $1`, userID)
	_, _ = pool.Exec(ctx, `DELETE FROM public.files WHERE user_id = $1 AND tenant_id = $2`, userID, tenantID)
	_, _ = pool.Exec(ctx, `DELETE FROM public.tenant_members WHERE user_id = $1`, userID)
	_, _ = pool.Exec(ctx, `DELETE FROM public.tenants WHERE id = $1`, tenantID)

	// Insert tenant + member.
	_, err = pool.Exec(ctx,
		`INSERT INTO public.tenants (id,name,slug,created_by,created_at,updated_at)
		 VALUES ($1,'eval-tenant','eval',$1,now(),now())`, tenantID)
	if err != nil {
		log.Fatalf("insert tenant: %v", err)
	}
	_, err = pool.Exec(ctx,
		`INSERT INTO public.tenant_members (tenant_id,user_id,email,role,created_at)
		 VALUES ($1,$2,'eval@kawai.test','owner',now())`, tenantID, userID)
	if err != nil {
		log.Fatalf("insert tenant_member: %v", err)
	}

	// Insert files (real schema: file_type, size, url are NOT NULL).
	fileIDs := make([]string, numFiles)
	for i := range fileIDs {
		var id string
		err := pool.QueryRow(ctx,
			`INSERT INTO public.files (id,user_id,tenant_id,name,file_type,size,url,created_at,updated_at)
			 VALUES (gen_random_uuid(),$1,$2,$3,'text/plain',1024,$4,now(),now()) RETURNING id`,
			userID, tenantID,
			fmt.Sprintf("eval-doc-%02d.txt", i+1),
			fmt.Sprintf("file:///eval-doc-%02d.txt", i+1),
		).Scan(&id)
		if err != nil {
			log.Fatalf("insert file %d: %v", i+1, err)
		}
		fileIDs[i] = id
	}

	// Insert chunks + file_chunks + random embeddings.
	fileChunks := make([][]string, numFiles)
	for i := range fileChunks {
		fileChunks[i] = make([]string, chunksPerFile)
	}

	for fi, fid := range fileIDs {
		for ci := 0; ci < chunksPerFile; ci++ {
			var chunkID string
			// chunks has user_id, tenant_id; no document_id column.
			err := pool.QueryRow(ctx,
				`INSERT INTO public.chunks (id,"text","index","type",user_id,tenant_id,created_at,updated_at,metadata)
				 VALUES (gen_random_uuid(),$1,$2,'text',$3,$4,now(),now(),'{}') RETURNING id`,
				chunkTexts[ci], ci, userID, tenantID,
			).Scan(&chunkID)
			if err != nil {
				log.Fatalf("insert chunk: %v", err)
			}
			fileChunks[fi][ci] = chunkID

			// file_chunks has user_id, tenant_id.
			_, err = pool.Exec(ctx,
				`INSERT INTO public.file_chunks (file_id,chunk_id,user_id,tenant_id) VALUES ($1,$2,$3,$4)`,
				fid, chunkID, userID, tenantID,
			)
			if err != nil {
				log.Fatalf("insert file_chunk: %v", err)
			}

			// embeddings has user_id, tenant_id.
			vec := randVec()
			_, err = pool.Exec(ctx,
				`INSERT INTO public.embeddings (chunk_id,embeddings,model,user_id,tenant_id,created_at)
				 VALUES ($1,$2::vector(1024),'fileprocessor',$3,$4,now())`,
				chunkID, vecLiteral(vec), userID, tenantID,
			)
			if err != nil {
				log.Fatalf("insert embedding: %v", err)
			}
		}
	}

	// Build eval dataset: each query targets chunk i of file 1 (deterministic).
	// Queries are written to match the chunk text for BM25 keyword recall.
	queryTargets := []struct {
		query     string
		chunkIdx  int
	}{
		{"Kubernetes deployment Helm charts helm upgrade", 0},
		{"CI/CD pipeline GitHub Actions ci.yml push main deploy", 1},
		{"Database connection pooling pgxpool KAWAI_PG_DSN max connections", 2},
		{"Authentication Oathkeeper edge auth sessions workspace", 3},
		{"Rate limiting envoy.yaml per-workspace quotas", 4},
	}
	queries := make([]evalQuery, len(queryTargets))
	for i, qt := range queryTargets {
		queries[i] = evalQuery{
			Query:    qt.query,
			Relevant: []string{fileChunks[0][qt.chunkIdx]}, // chunk from file 1
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
