package embedding

import (
	"log/slog"

	fp "github.com/kawai-network/fileprocessor"

	"github.com/yudaprama/egent-common/envutil"
)

const (
	defaultDim = 1024

	openrouterEmbedURL = "https://openrouter.ai/api/v1/embeddings"
	openrouterModel    = "openai/text-embedding-3-small"

	nvidiaEmbedURL = "https://integrate.api.nvidia.com/v1"
	nvidiaModel    = "nvidia/nv-embed-v1"

	geminiModel = "gemini-embedding-001"
)

// ProviderConfig describes a single embedding provider.
type ProviderConfig struct {
	Name  string
	URL   string
	Key   string
	Model string
	Dim   int
}

// BuildProvidersFromEnvWithKeys constructs embedding providers from .env API keys.
// Env vars (comma-separated): OPENROUTER_API_KEYS, NVIDIA_API_KEYS, GEMINI_API_KEYS.
// Returns providers in priority order: OpenAI (OpenRouter) → NVIDIA → Gemini.
// Each provider is wrapped with billingEmbedder for usage tracking.
// A random key is selected per provider at startup.
func BuildProvidersFromEnvWithKeys() []fp.Embedder {
	var providers []fp.Embedder

	// 1. OpenAI via OpenRouter
	if key := envutil.PickRandomKey("OPENROUTER_API_KEYS"); key != "" {
		p := NewOpenAICompatibleEmbedder(openrouterEmbedURL, key, openrouterModel, defaultDim)
		providers = append(providers, newBillingEmbedder(p, openrouterModel))
		slog.Info("embedding: provider registered", "name", "openai", "model", openrouterModel)
	}

	// 2. NVIDIA NIM
	if key := envutil.PickRandomKey("NVIDIA_API_KEYS"); key != "" {
		p := NewNvidiaEmbedder(nvidiaEmbedURL, key, nvidiaModel, defaultDim)
		providers = append(providers, newBillingEmbedder(p, nvidiaModel))
		slog.Info("embedding: provider registered", "name", "nvidia", "model", nvidiaModel)
	}

	// 3. Google Gemini
	if key := envutil.PickRandomKey("GEMINI_API_KEYS"); key != "" {
		p := NewGeminiEmbedder(key, geminiModel, defaultDim)
		providers = append(providers, newBillingEmbedder(p, geminiModel))
		slog.Info("embedding: provider registered", "name", "gemini", "model", geminiModel)
	}

	return providers
}
