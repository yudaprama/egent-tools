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

// extractHeuristic returns key/value pairs derived from text.
func extractHeuristic(text string) map[string]string {
	lower := strings.ToLower(text)
	facts := map[string]string{}

	if idx := strings.Index(lower, "my name is "); idx >= 0 {
		rest := text[idx+len("my name is "):]
		name := firstWord(rest)
		if name != "" {
			facts["user.name"] = name
		}
	}
	if idx := strings.Index(lower, "i'm "); idx >= 0 {
		rest := text[idx+len("i'm "):]
		name := firstWord(rest)
		if isValidName(name) {
			facts["user.name"] = name
		}
	}
	if idx := strings.Index(lower, "i live in "); idx >= 0 {
		rest := text[idx+len("i live in "):]
		loc := firstPhrase(rest)
		if loc != "" {
			facts["user.location"] = loc
		}
	}
	if idx := strings.Index(lower, "i'm from "); idx >= 0 {
		rest := text[idx+len("i'm from "):]
		loc := firstPhrase(rest)
		if loc != "" {
			facts["user.location"] = loc
		}
	}
	for _, prefix := range []string{"i like ", "i prefer ", "i love ", "i use "} {
		if idx := strings.Index(lower, prefix); idx >= 0 {
			rest := text[idx+len(prefix):]
			item := firstPhrase(rest)
			if item != "" {
				key := "preferences." + strings.ReplaceAll(strings.ToLower(item), " ", "_")
				facts[key] = item
			}
			break
		}
	}

	return facts
}

func firstWord(s string) string {
	s = strings.TrimSpace(s)
	for i, ch := range s {
		if ch == ' ' || ch == ',' || ch == '.' || ch == '!' || ch == '\n' {
			return s[:i]
		}
	}
	return s
}

func firstPhrase(s string) string {
	s = strings.TrimSpace(s)
	for i, ch := range s {
		if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
			return s[:i]
		}
	}
	if idx := strings.Index(s, ", "); idx > 0 {
		return s[:idx]
	}
	return s
}

func isValidName(s string) bool {
	if len(s) < 2 || len(s) > 30 {
		return false
	}
	switch strings.ToLower(s) {
	case "not", "sure", "sorry", "happy", "sad", "tired", "hungry", "going":
		return false
	}
	if !isAlpha(s[0]) {
		return false
	}
	return true
}

func isAlpha(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}