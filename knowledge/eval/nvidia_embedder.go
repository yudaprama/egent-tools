package eval

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

// nvidiaEmbedder is a minimal OpenAI-compatible embedder for NVIDIA NIM.
// Unlike fp.OpenAIEmbedder, it does NOT send "dimensions" (unsupported by
// NVIDIA) and sends "input_type" (required by NVIDIA).
type nvidiaEmbedder struct {
	url    string
	apiKey string
	model  string
	dim    int
	client *http.Client
}

// newNVIDIAEmbedder creates an embedder targeting https://integrate.api.nvidia.com/v1/embeddings.
// Pass inputType as "query" for search queries or "passage" for indexed documents.
func newNVIDIAEmbedder(apiKey, model, inputType string) fp.Embedder {
	if model == "" {
		model = "nvidia/nv-embedqa-e5-v5"
	}
	if inputType == "" {
		inputType = "passage"
	}
	return &nvidiaEmbedder{
		url:    "https://integrate.api.nvidia.com/v1/embeddings",
		apiKey: apiKey,
		model:  model,
		dim:    1024,
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
	// NVIDIA embed-qa models only support single-text requests.
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
	req, err := http.NewRequestWithContext(ctx, "POST", e.url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("nvidia %s HTTP %d: %s", e.url, resp.StatusCode, string(raw))
	}

	var api nvidiaResponse
	if err := json.Unmarshal(raw, &api); err != nil {
		return nil, err
	}
	if len(api.Data) == 0 {
		return nil, fmt.Errorf("nvidia: empty response")
	}
	vec := make([]float32, len(api.Data[0].Embedding))
	for i, f := range api.Data[0].Embedding {
		vec[i] = float32(f)
	}
	return vec, nil
}

func (e *nvidiaEmbedder) Dimension() int {
	return e.dim
}
