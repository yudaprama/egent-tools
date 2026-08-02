package rerank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"
)

const defaultNvidiaRerankBaseURL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"

type NvidiaReranker struct {
	modelName     string
	modelID       string
	apiKey        string
	baseURL       string
	client        *http.Client
	customHeaders map[string]string
}

func (r *NvidiaReranker) SetCustomHeaders(headers map[string]string) {
	r.customHeaders = headers
}

type NvidiaRerankDocument struct {
	Text string `json:"text"`
}

type NvidiaRerankRequest struct {
	Model     string                 `json:"model"`
	Query     NvidiaRerankDocument   `json:"query"`
	Documents []NvidiaRerankDocument `json:"passages"`
}

type NvidiaRankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"logit"`
}

type NvidiaRerankResponse struct {
	Model   string             `json:"model"`
	Results []NvidiaRankResult `json:"rankings"`
}

func NewNvidiaReranker(config *RerankerConfig) (*NvidiaReranker, error) {
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = resolveNvidiaAPIKey()
	}
	baseURL := defaultNvidiaRerankBaseURL
	if config.BaseURL != "" {
		baseURL = config.BaseURL
	}
	if err := validateRerankBaseURL(baseURL); err != nil {
		return nil, err
	}
	modelName := config.ModelName
	if modelName == "" {
		modelName = "llama-nemotron-rerank-vl-1b-v2"
	}
	return &NvidiaReranker{
		modelName: modelName,
		modelID:   config.ModelID,
		apiKey:    apiKey,
		baseURL:   baseURL,
		client:    newRerankHTTPClient(30 * time.Second),
	}, nil
}

func (r *NvidiaReranker) Rerank(ctx context.Context, query string, documents []string) ([]RankResult, error) {
	if len(documents) == 0 {
		return []RankResult{}, nil
	}
	requestBody := &NvidiaRerankRequest{
		Model:     r.modelName,
		Query:     NvidiaRerankDocument{Text: query},
		Documents: make([]NvidiaRerankDocument, len(documents)),
	}
	for i := range requestBody.Documents {
		requestBody.Documents[i].Text = documents[i]
	}
	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request body: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.baseURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", r.apiKey))
	for k, v := range r.customHeaders {
		req.Header.Set(k, v)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("rerank API error: status %d: %s", resp.StatusCode, string(body))
	}
	var response NvidiaRerankResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}
	ret := make([]RankResult, len(response.Results))
	for i, result := range response.Results {
		ret[i] = RankResult{
			Index:          result.Index,
			Document:       DocumentInfo{Text: documents[result.Index]},
			RelevanceScore: result.RelevanceScore,
		}
	}
	return ret, nil
}

func (r *NvidiaReranker) GetModelName() string {
	return r.modelName
}

func (r *NvidiaReranker) GetModelID() string {
	return r.modelID
}

func resolveNvidiaAPIKey() string {
	v := os.Getenv("NVIDIA_API_KEYS")
	if v == "" {
		return ""
	}
	keys := strings.Split(v, ",")
	for i, k := range keys {
		keys[i] = strings.TrimSpace(k)
	}
	if len(keys) == 1 {
		return keys[0]
	}
	return keys[rand.Intn(len(keys))]
}