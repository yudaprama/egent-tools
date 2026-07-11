package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

// Extractor extracts key/value facts from conversation text.
// Implementations range from heuristic pattern-matching to LLM-based extraction.
type Extractor interface {
	Extract(ctx context.Context, text string) (map[string]string, error)
}

// HeuristicExtractor uses string patterns ("my name is", "i like", etc.)
// to extract facts. Fast, zero-cost, but limited to a handful of patterns.
type HeuristicExtractor struct{}

func NewHeuristicExtractor() *HeuristicExtractor {
	return &HeuristicExtractor{}
}

func (e *HeuristicExtractor) Extract(_ context.Context, text string) (map[string]string, error) {
	return extractHeuristic(text), nil
}

// LLMExtractor uses a ChatModel to extract facts from conversation text.
// It prompts the model for JSON key/value output, which gives it natural
// language understanding far beyond what heuristic patterns can match.
//
// The model should be a fast, cheap one — extraction doesn't need reasoning.
type LLMExtractor struct {
	model model.ChatModel
}

func NewLLMExtractor(m model.ChatModel) *LLMExtractor {
	return &LLMExtractor{model: m}
}

// extractPrompt instructs the LLM to produce structured facts from user text.
const extractPrompt = `You are a memory extraction system. Extract personal facts from the user's message.

Output ONLY a JSON object where each key is a dot-separated fact name and each value is the fact value.
Use these key conventions when applicable:
- "user.name" — the user's name or preferred name
- "user.location" — where the user lives or is from
- "preferences.<topic>" — things the user likes, dislikes, prefers, or uses
- "user.<attribute>" — any other personal fact about the user
- "preferences.<topic>" — use "false" suffix or negative value for dislikes

Examples:
Input: "Hi I'm Bob, I live in Tokyo and I hate mushrooms"
Output: {"user.name":"Bob","user.location":"Tokyo","preferences.mushrooms":"false"}

Input: "My favorite color is blue and I use VS Code"
Output: {"preferences.favorite_color":"blue","preferences.vs_code":"VS Code"}

Input: "I work at Acme Corp as an engineer"
Output: {"user.employer":"Acme Corp","user.job_title":"engineer"}

Input: "Call me Alice"
Output: {"user.name":"Alice"}

If no facts are found, output: {}

Rules:
- Keys MUST use lowercase with dots (e.g. "user.name", not "UserName" or "name")
- Values are plain strings (not nested objects)
- Omit any information the user doesn't explicitly share
- Never fabricate or assume facts`

func (e *LLMExtractor) Extract(ctx context.Context, text string) (map[string]string, error) {
	msgs := []*schema.Message{
		schema.SystemMessage(extractPrompt),
		schema.UserMessage(text),
	}

	resp, err := e.model.Generate(ctx, msgs)
	if err != nil {
		return nil, fmt.Errorf("llm extract: %w", err)
	}

	content := strings.TrimSpace(resp.Content)
	// Strip markdown code fences if present
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	if content == "" || content == "{}" {
		return map[string]string{}, nil
	}

	var facts map[string]string
	if err := json.Unmarshal([]byte(content), &facts); err != nil {
		slog.Warn("llm extract: failed to parse JSON, falling back to heuristic", "response", content, "error", err)
		return extractHeuristic(text), nil
	}

	// Normalise boolean-ish preference values
	for k, v := range facts {
		lower := strings.ToLower(v)
		if lower == "true" || lower == "yes" {
			facts[k] = "true"
		} else if lower == "false" || lower == "no" {
			facts[k] = "false"
		}
	}

	return facts, nil
}