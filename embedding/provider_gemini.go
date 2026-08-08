package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	fp "github.com/kawai-network/fileprocessor"
)

// geminiEmbedder calls Google Gemini embedding models.
// Auth: x-goog-api-key header. Gemini only supports single-text requests.
type geminiEmbedder struct {
	apiKey string
	model  string
	dim    int
	client *http.Client
}

// NewGeminiEmbedder creates a Gemini embedder.
// dim should be 1024 to match public.embeddings schema (truncated from native).
func NewGeminiEmbedder(apiKey, model string, dim int) fp.Embedder {
	if model == "" {
		model = "gemini-embedding-001"
	}
	if dim <= 0 {
		dim = 1024
	}
	return &geminiEmbedder{
		apiKey: apiKey,
		model:  model,
		dim:    dim,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *geminiEmbedder) embedURL() string {
	return fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:embedContent", e.model)
}

func (e *geminiEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Gemini only supports single-text requests — issue concurrent calls.
	results := make([][]float32, len(texts))
	errs := make([]error, len(texts))
	var wg sync.WaitGroup

	for i, t := range texts {
		wg.Add(1)
		go func(idx int, text string) {
			defer wg.Done()
			vec, err := e.embedOne(ctx, text)
			if err != nil {
				errs[idx] = err
				return
			}
			results[idx] = vec
		}(i, t)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}

func (e *geminiEmbedder) embedOne(ctx context.Context, text string) ([]float32, error) {
	payload := map[string]any{
		"model": "models/" + e.model,
		"content": map[string]any{
			"parts": []map[string]any{{"text": text}},
		},
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("gemini_embedder: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.embedURL(), bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("gemini_embedder: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("x-goog-api-key", e.apiKey)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gemini_embedder: call: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("gemini_embedder: read body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini_embedder: HTTP %d: %.200s", resp.StatusCode, raw)
	}

	var api struct {
		Embedding struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
	}
	if err := json.Unmarshal(raw, &api); err != nil {
		return nil, fmt.Errorf("gemini_embedder: decode: %w", err)
	}
	vals := api.Embedding.Values
	if e.dim > 0 && len(vals) > e.dim {
		vals = vals[:e.dim]
	}
	return vals, nil
}

func (e *geminiEmbedder) Model() string  { return e.model }
func (e *geminiEmbedder) Dimension() int { return e.dim }