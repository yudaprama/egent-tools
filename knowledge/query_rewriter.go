package knowledge

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// QueryRewriter expands a single user query into alternative phrasings
// (architecture review R6). The original query is always retained by the
// caller; this returns only the variants. Implementations must be safe for
// concurrent use. A nil rewriter (or one that returns no variants) leaves the
// single-query retrieval path unchanged — query rewriting is opt-in and should
// be measured against recall@K before being enabled in production.
type QueryRewriter interface {
	Rewrite(ctx context.Context, query string) ([]string, error)
}

// LLMQueryRewriter calls an OpenAI-compatible chat completions endpoint to
// produce up to MaxRewrites alternative phrasings of the query. It reuses the
// same MODEL_BASE_URL / MODEL_NAME / MODEL_API_KEY conventions as the host
// egent, configured by the caller.
type LLMQueryRewriter struct {
	baseURL     string
	apiKey      string
	model       string
	maxRewrites int
	client      *http.Client
}

// LLMQueryRewriterConfig configures an LLMQueryRewriter.
type LLMQueryRewriterConfig struct {
	BaseURL     string        // e.g. "https://gateway/v1"
	APIKey      string        // bearer token; empty = no auth header
	Model       string        // chat model id
	MaxRewrites int           // default 2
	Timeout     time.Duration // default 8s (rewriting must not dominate search latency)
}

// NewLLMQueryRewriter constructs a rewriter. Empty baseURL or Model yields nil
// so callers can treat "not configured" uniformly.
func NewLLMQueryRewriter(cfg LLMQueryRewriterConfig) *LLMQueryRewriter {
	if cfg.BaseURL == "" || cfg.Model == "" {
		return nil
	}
	if cfg.MaxRewrites <= 0 {
		cfg.MaxRewrites = 2
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 8 * time.Second
	}
	return &LLMQueryRewriter{
		baseURL:     strings.TrimRight(cfg.BaseURL, "/"),
		apiKey:      cfg.APIKey,
		model:       cfg.Model,
		maxRewrites: cfg.MaxRewrites,
		client:      &http.Client{Timeout: cfg.Timeout},
	}
}

const rewriteSystemPrompt = `You rewrite a search query into alternative phrasings to improve retrieval recall.
Output at most %d alternative queries, one per line. No numbering, no bullets, no commentary, no quotes.
Each line must be a self-contained search query. Do not include the original query.
If the query is already specific (e.g. an exact identifier), output nothing.`

// Rewrite calls the chat model and returns the parsed alternative queries.
func (r *LLMQueryRewriter) Rewrite(ctx context.Context, query string) ([]string, error) {
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	body := map[string]any{
		"model": r.model,
		"messages": []map[string]string{
			{"role": "system", "content": fmt.Sprintf(rewriteSystemPrompt, r.maxRewrites)},
			{"role": "user", "content": query},
		},
		"temperature": 0,
		"max_tokens":  96,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("query_rewriter: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.baseURL+"/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("query_rewriter: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if r.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.apiKey)
	}

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("query_rewriter: call: %w", err)
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("query_rewriter: read body: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("query_rewriter: HTTP %d: %s", resp.StatusCode, truncate(string(respBody), 300))
	}

	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("query_rewriter: decode: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return nil, nil
	}
	return parseRewrites(parsed.Choices[0].Message.Content, r.maxRewrites), nil
}

// parseRewrites extracts non-empty, de-duplicated, length-bounded query lines.
func parseRewrites(content string, max int) []string {
	out := make([]string, 0, max)
	seen := make(map[string]bool)
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 4096), 4096)
	for sc.Scan() && len(out) < max {
		line := strings.TrimSpace(sc.Text())
		line = strings.Trim(line, `"-*123456789.` + "\t")
		if line == "" || len(line) > 200 || seen[line] {
			continue
		}
		seen[line] = true
		out = append(out, line)
	}
	return out
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
