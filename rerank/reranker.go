package rerank

import (
	"context"
	"encoding/json"
	"fmt"
)

type Reranker interface {
	Rerank(ctx context.Context, query string, documents []string) ([]RankResult, error)
	GetModelName() string
	GetModelID() string
}

type RankResult struct {
	Index          int          `json:"index"`
	Document       DocumentInfo `json:"document"`
	RelevanceScore float64      `json:"relevance_score"`
}

func (r *RankResult) UnmarshalJSON(data []byte) error {
	var temp struct {
		Index          int          `json:"index"`
		Document       DocumentInfo `json:"document"`
		RelevanceScore *float64     `json:"relevance_score"`
		Score          *float64     `json:"score"`
	}
	if err := json.Unmarshal(data, &temp); err != nil {
		return fmt.Errorf("failed to unmarshal rank result: %w", err)
	}
	r.Index = temp.Index
	r.Document = temp.Document
	if temp.RelevanceScore != nil {
		r.RelevanceScore = *temp.RelevanceScore
	} else if temp.Score != nil {
		r.RelevanceScore = *temp.Score
	}
	return nil
}

type DocumentInfo struct {
	Text string `json:"text"`
}

func (d *DocumentInfo) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		d.Text = text
		return nil
	}
	var temp struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(data, &temp); err != nil {
		return fmt.Errorf("failed to unmarshal DocumentInfo: %w", err)
	}
	d.Text = temp.Text
	return nil
}

type RerankerConfig struct {
	APIKey      string
	BaseURL     string
	ModelName   string
	ModelID     string
	Provider    string
	ExtraConfig map[string]string
	CustomHeaders map[string]string
}

func ConfigFromModelID(modelID, apiKey, baseURL, modelName string) *RerankerConfig {
	if modelID == "" {
		return nil
	}
	return &RerankerConfig{
		ModelID:   modelID,
		APIKey:    apiKey,
		BaseURL:   baseURL,
		ModelName: modelName,
		Provider:  string(ProviderNvidia),
	}
}

func NewReranker(config *RerankerConfig) (Reranker, error) {
	providerName := ResolveProvider(config.Provider)
	if providerName == "" {
		providerName = DetectProvider(config.BaseURL)
	}
	var reranker Reranker
	var err error
	switch providerName {
	case ProviderNvidia:
		reranker, err = NewNvidiaReranker(config)
	default:
		reranker, err = NewNvidiaReranker(config)
	}
	if err != nil {
		return nil, err
	}
	return reranker, nil
}