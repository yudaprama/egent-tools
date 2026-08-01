package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/scrypster/muninndb/sdk/go/muninn"
)

type muninnFixture struct {
	server         *httptest.Server
	writes         []muninn.WriteRequest
	batchWrites    []muninn.WriteRequest
	reads          []string
	forgets        []string
	activates      []struct {
		Vault   string
		Req     muninn.ActivateRequest
	}
	listRequests   []url.Values
	healthRequests int
}

func newMuninnFixture(t *testing.T) *muninnFixture {
	t.Helper()
	fixture := &muninnFixture{}
	mux := http.NewServeMux()

	mux.HandleFunc("/api/health", func(w http.ResponseWriter, r *http.Request) {
		fixture.healthRequests++
		writeJSON(t, w, muninn.HealthResponse{Status: "ok"})
	})

	mux.HandleFunc("/api/engrams/batch", func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Engrams []muninn.WriteRequest `json:"engrams"`
		}
		readJSON(t, r, &payload)
		fixture.batchWrites = append(fixture.batchWrites, payload.Engrams...)
		resp := muninn.BatchWriteResponse{Results: make([]muninn.BatchWriteResult, len(payload.Engrams))}
		for i, req := range payload.Engrams {
			resp.Results[i] = muninn.BatchWriteResult{
				Index:  i,
				ID:     "id-" + strings.ReplaceAll(req.Concept, ".", "-"),
				Status: "created",
			}
		}
		writeJSON(t, w, resp)
	})

	mux.HandleFunc("/api/engrams", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			var req muninn.WriteRequest
			readJSON(t, r, &req)
			fixture.writes = append(fixture.writes, req)
			writeJSON(t, w, muninn.WriteResponse{
				ID:        "id-" + strings.ReplaceAll(req.Concept, ".", "-"),
				CreatedAt: 1700000000,
			})
		case http.MethodGet:
			fixture.listRequests = append(fixture.listRequests, r.URL.Query())
			writeJSON(t, w, muninn.ListEngramsResponse{
				Engrams: []muninn.EngramItem{
					{
						ID:        "id-user-name",
						Concept:   "user.name",
						Content:   "Alice",
						Tags:      []string{"memory", "user:u1", "session:s1"},
						Vault:     r.URL.Query().Get("vault"),
						CreatedAt: 1700000001,
					},
				},
				Total:  1,
				Limit:  100,
				Offset: 0,
			})
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/api/engrams/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/api/engrams/")
		switch r.Method {
		case http.MethodGet:
			fixture.reads = append(fixture.reads, id)
			writeJSON(t, w, muninn.Engram{
				ID:        id,
				Concept:   "user.name",
				Content:   "Alice",
				Tags:      []string{"memory", "user:u1", "session:s1"},
				CreatedAt: 1700000001,
				UpdatedAt: 1700000002,
			})
		case http.MethodDelete:
			fixture.forgets = append(fixture.forgets, id)
			w.WriteHeader(http.StatusNoContent)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/api/activate", func(w http.ResponseWriter, r *http.Request) {
		var req muninn.ActivateRequest
		readJSON(t, r, &req)
		fixture.activates = append(fixture.activates, struct {
			Vault   string
			Req     muninn.ActivateRequest
		}{Vault: r.URL.Query().Get("vault"), Req: req})
		why := "exact concept match"
		writeJSON(t, w, muninn.ActivateResponse{
			Activations: []muninn.ActivationItem{
				{
					ID:         "id-user-name",
					Concept:    "user.name",
					Content:    "Alice",
					Tags:       []string{"memory", "user:u1", "session:s1"},
					Score:      0.91,
					Confidence: 0.9,
					Why:        &why,
				},
			},
		})
	})

	fixture.server = httptest.NewServer(mux)
	t.Cleanup(fixture.server.Close)
	return fixture
}

func TestMuninnStore_SetGetUsesCachedIDAndPreservesTimestamps(t *testing.T) {
	fixture := newMuninnFixture(t)
	store := NewMuninnStore(fixture.server.URL, "")
	ctx := context.Background()

	if err := store.Set(ctx, "t1", "u1", "s1", "user.name", "Alice"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	entry, err := store.Get(ctx, "t1", "u1", "s1", "user.name")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if entry == nil {
		t.Fatal("expected entry")
	}
	if entry.Key != "user.name" || entry.Value != "Alice" {
		t.Fatalf("unexpected entry: %+v", entry)
	}
	if entry.CreatedAt.Unix() != 1700000001 || entry.UpdatedAt.Unix() != 1700000002 {
		t.Fatalf("timestamps not preserved: %+v", entry)
	}
	if len(fixture.reads) != 1 || fixture.reads[0] != "id-user-name" {
		t.Fatalf("expected cached ID read, got %#v", fixture.reads)
	}
	if len(fixture.activates) != 0 {
		t.Fatalf("expected no activation for cached Get, got %#v", fixture.activates)
	}
}

func TestMuninnStore_DeleteUsesCachedID(t *testing.T) {
	fixture := newMuninnFixture(t)
	store := NewMuninnStore(fixture.server.URL, "")
	ctx := context.Background()

	if err := store.Set(ctx, "t1", "u1", "s1", "user.name", "Alice"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if err := store.Delete(ctx, "t1", "u1", "s1", "user.name"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if len(fixture.forgets) != 1 || fixture.forgets[0] != "id-user-name" {
		t.Fatalf("expected cached ID forget, got %#v", fixture.forgets)
	}
}

func TestMuninnStore_DeleteMissingReturnsErrMemoryNotFound(t *testing.T) {
	fixture := newMuninnFixture(t)
	store := NewMuninnStore(fixture.server.URL, "")
	ctx := context.Background()

	err := store.Delete(ctx, "t1", "u1", "s1", "missing.key")
	if !errors.Is(err, ErrMemoryNotFound) {
		t.Fatalf("expected ErrMemoryNotFound, got %v", err)
	}
}

func TestMuninnStore_SetBatchCachesIDs(t *testing.T) {
	fixture := newMuninnFixture(t)
	store := NewMuninnStore(fixture.server.URL, "")
	ctx := context.Background()

	if err := store.SetBatch(ctx, "t1", "u1", "s1", map[string]string{"user.name": "Alice", "user.location": "Paris"}); err != nil {
		t.Fatalf("SetBatch: %v", err)
	}
	if len(fixture.batchWrites) != 2 {
		t.Fatalf("expected 2 batch writes, got %d", len(fixture.batchWrites))
	}
	if id := store.lookupIDCache("t1", "u1", "s1", "user.name"); id != "id-user-name" {
		t.Fatalf("expected cached id-user-name, got %q", id)
	}
	if id := store.lookupIDCache("t1", "u1", "s1", "user.location"); id != "id-user-location" {
		t.Fatalf("expected cached id-user-location, got %q", id)
	}
}

func TestMuninnStore_ListCachesIDsAndPreservesCreatedAt(t *testing.T) {
	fixture := newMuninnFixture(t)
	store := NewMuninnStore(fixture.server.URL, "")

	entries, err := store.List(context.Background(), "t1", ByUserID("u1"), BySessionID("s1"))
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}
	if entries[0].CreatedAt.Unix() != 1700000001 {
		t.Fatalf("created timestamp not preserved: %+v", entries[0])
	}
	if id := store.lookupIDCache("t1", "u1", "s1", "user.name"); id != "id-user-name" {
		t.Fatalf("expected cached id-user-name, got %q", id)
	}
}

func TestMuninnStore_SearchPassesTagFilters(t *testing.T) {
	fixture := newMuninnFixture(t)
	store := NewMuninnStore(fixture.server.URL, "")
	ctx := context.Background()

	results, err := store.Search(ctx, "t1", "name", 5, ByUserID("u1"), BySessionID("s1"))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if len(fixture.activates) != 1 {
		t.Fatalf("expected 1 activate call, got %d", len(fixture.activates))
	}
	filters := fixture.activates[0].Req.Filters
	if len(filters) == 0 {
		t.Fatal("expected tag filters in activate request")
	}
	if filters[0].Field != "tags_all" {
		t.Fatalf("expected tags_all filter, got %s", filters[0].Field)
	}
}

func TestMuninnStore_SearchByTagCombinesFilters(t *testing.T) {
	fixture := newMuninnFixture(t)
	store := NewMuninnStore(fixture.server.URL, "")
	ctx := context.Background()

	results, err := store.Search(ctx, "t1", "name", 5, ByUserID("u1"), ByTag("memory"))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	filters := fixture.activates[0].Req.Filters
	if len(filters) != 1 {
		t.Fatalf("expected 1 combined filter, got %d", len(filters))
	}
	// JSON unmarshal produces []interface{}, so check via string conversion
	values, ok := filters[0].Value.([]interface{})
	if !ok {
		t.Fatalf("expected []interface{} value, got %T", filters[0].Value)
	}
	got := make(map[string]bool, len(values))
	for _, v := range values {
		got[fmt.Sprint(v)] = true
	}
	for _, want := range []string{"user:u1", "memory"} {
		if !got[want] {
			t.Fatalf("expected tags_all to contain %q, got %v", want, values)
		}
	}
}

func TestMuninnStore_ListFiltersByTag(t *testing.T) {
	fixture := newMuninnFixture(t)
	store := NewMuninnStore(fixture.server.URL, "")

	// The fixture returns one engram with tags ["memory", "user:u1", "session:s1"].
	// Searching with a matching tag should return the entry.
	entries, err := store.List(context.Background(), "t1", ByTag("memory"))
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry with matching tag, got %d", len(entries))
	}

	// Searching with a non-matching tag should return nothing.
	entries, err = store.List(context.Background(), "t1", ByTag("nonexistent"))
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected 0 entries with non-matching tag, got %d", len(entries))
	}
}

func TestMuninnStore_ActivateFormatsScores(t *testing.T) {
	fixture := newMuninnFixture(t)
	store := NewMuninnStore(fixture.server.URL, "")

	activated, err := store.Activate(context.Background(), "t1", "u1", "s1", []string{"name"}, 10)
	if err != nil {
		t.Fatalf("Activate: %v", err)
	}
	formatted := FormatActivatedMemories(activated)
	if !strings.Contains(formatted, "score=0.910") || !strings.Contains(formatted, "conf=0.90") {
		t.Fatalf("expected score and confidence in formatted output, got %q", formatted)
	}
}

func TestMuninnStore_Health(t *testing.T) {
	fixture := newMuninnFixture(t)
	store := NewMuninnStore(fixture.server.URL, "")

	if !store.Health(context.Background()) {
		t.Fatal("expected healthy store")
	}
	if fixture.healthRequests != 1 {
		t.Fatalf("expected one health request, got %d", fixture.healthRequests)
	}
}

func readJSON(t *testing.T, r *http.Request, v any) {
	t.Helper()
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		t.Fatalf("decode request: %v", err)
	}
}

func writeJSON(t *testing.T, w http.ResponseWriter, v any) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		t.Fatalf("encode response: %v", err)
	}
}
