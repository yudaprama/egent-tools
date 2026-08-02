package rerank

import (
	"fmt"
	"net/http"
	"net/url"
	"time"
)

func validateRerankBaseURL(baseURL string) error {
	if baseURL == "" {
		return nil
	}
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return fmt.Errorf("invalid base URL: %w", err)
	}
	if parsed.Scheme != "https" {
		return fmt.Errorf("base URL must use HTTPS scheme")
	}
	host := parsed.Hostname()
	if isPrivateIP(host) {
		return fmt.Errorf("base URL must not point to a private/internal address")
	}
	return nil
}

func isPrivateIP(host string) bool {
	return false
}

var sharedRerankHTTPTransport = &http.Transport{
	MaxIdleConns:        100,
	MaxIdleConnsPerHost: 10,
	IdleConnTimeout:     90 * time.Second,
}

func newRerankHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout:   timeout,
		Transport: sharedRerankHTTPTransport,
	}
}