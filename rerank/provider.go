package rerank

import "strings"

type ProviderName string

const (
	ProviderNvidia ProviderName = "nvidia"
)

func ResolveProvider(name string) ProviderName {
	switch name {
	case "nvidia":
		return ProviderNvidia
	default:
		return ""
	}
}

func DetectProvider(baseURL string) ProviderName {
	if baseURL == "" {
		return ProviderNvidia
	}
	if strings.Contains(baseURL, "nvidia.com") {
		return ProviderNvidia
	}
	return ProviderNvidia
}