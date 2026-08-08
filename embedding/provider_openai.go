package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	fp "github.com/kawai-network/fileprocessor"
)

// openaiEmbedder calls any OpenAI-compatible embeddings API
// (OpenAI, OpenRouter, Azure OpenAI, etc.).
type openaiEmbedder struct {
	url    string
	apiKey string
	model  string
	dim    int
	client *http.Client
}

// NewOpenAICompatibleEmbedder creates an embedder for OpenAI-compatible endpoints.
// dim should be 1024 to match public.embeddings schema.
func NewOpenAICompatibleEmbedder(url, apiKey, model string, dim int) fp.Embedder {
	if model == "" {
		model = "openai/text-embedding-3-small"
	}
	if dim <= 0 {
		dim = 1024
	}
	return &openaiEmbedder{
		url:    url,
		apiKey: apiKey,
		model:  model,
		dim:    dim,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

type openaiEmbedRequest struct {
	Input      any    `json:"input"`
	Model      string `json:"model"`
	Dimensions int    `json:"dimensions,omitempty"`
}

type openaiEmbedResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

func (e *openaiEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body, _ := json.Marshal(openaiEmbedRequest{
		Input:      texts,
		Model:      e.model,
		Dimensions: e.dim,
	})
	req, err := http.NewRequestWithContext(ctx, "POST", e.url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai_embedder: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai_embedder: call: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai_embedder: read body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai_embedder: HTTP %d: %.200s", resp.StatusCode, raw)
	}

	var api openaiEmbedResponse
	if err := json.Unmarshal(raw, &api); err != nil {
		return nil, fmt.Errorf("openai_embedder: decode: %w", err)
	}

	out := make([][]float32, len(api.Data))
	for i, d := range api.Data {
		v := make([]float32, len(d.Embedding))
		for j, f := range d.Embedding {
			v[j] = float32(f)
		}
		out[i] = v
	}
	return out, nil
}

func (e *openaiEmbedder) Model() string     { return e.model }
func (e *openaiEmbedder) Dimension() int { return e.dim }
