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

// nvidiaEmbedder sends input_type (required by NVIDIA) and omits dimensions
// (unsupported by NVIDIA). Implements fp.Embedder.
type nvidiaEmbedder struct {
	url    string
	apiKey string
	model  string
	dim    int
	client *http.Client
}

// NewNvidiaEmbedder creates an NVIDIA NIM embedder.
// dim should be 1024 to match public.embeddings schema.
func NewNvidiaEmbedder(url, apiKey, model string, dim int) fp.Embedder {
	if model == "" {
		model = "nvidia/nv-embed-v1"
	}
	if dim <= 0 {
		dim = 1024
	}
	return &nvidiaEmbedder{
		url:    url,
		apiKey: apiKey,
		model:  model,
		dim:    dim,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

type nvidiaRequest struct {
	Input     string `json:"input"`
	Model     string `json:"model"`
	InputType string `json:"input_type"`
}

type nvidiaResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

func (e *nvidiaEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	// NVIDIA embed models only support single-text requests.
	results := make([][]float32, len(texts))
	for i, t := range texts {
		vec, err := e.embedOne(ctx, t)
		if err != nil {
			return nil, fmt.Errorf("nvidia_embedder: text %d: %w", i, err)
		}
		results[i] = vec
	}
	return results, nil
}

func (e *nvidiaEmbedder) embedOne(ctx context.Context, text string) ([]float32, error) {
	body, _ := json.Marshal(nvidiaRequest{
		Input:     text,
		Model:     e.model,
		InputType: "passage",
	})
	req, err := http.NewRequestWithContext(ctx, "POST", e.url+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("nvidia_embedder: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("nvidia_embedder: call: %w", err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("nvidia_embedder: HTTP %d: %.200s", resp.StatusCode, raw)
	}

	var api nvidiaResponse
	if err := json.Unmarshal(raw, &api); err != nil {
		return nil, fmt.Errorf("nvidia_embedder: decode: %w", err)
	}
	if len(api.Data) == 0 {
		return nil, fmt.Errorf("nvidia_embedder: empty response")
	}
	vec := make([]float32, len(api.Data[0].Embedding))
	for i, f := range api.Data[0].Embedding {
		vec[i] = float32(f)
	}
	return vec, nil
}

func (e *nvidiaEmbedder) Model() string  { return e.model }
func (e *nvidiaEmbedder) Dimension() int { return e.dim }