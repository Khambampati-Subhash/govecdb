package httpapi

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A full pool refuses at once, with the standard error shape and a
// Retry-After, and only for its own kind of request.
func TestFullPoolRefusesRatherThanQueues(t *testing.T) {
	a := newAPI(t, Config{MaxInFlightReads: 1, MaxInFlightWrites: 1})
	a.createCollection("docs", 4)
	search := map[string]any{"query": []float32{1, 0, 0, 0}, "k": 1}

	// Occupy the read pool's one slot, as a long search would.
	if !a.server.reads.tryAcquire() {
		t.Fatal("could not take the read slot")
	}
	for _, req := range []struct{ method, path string }{
		{"POST", "/v1/collections/docs/search"},
		{"GET", "/v1/collections/docs/vectors"},
		{"GET", "/v1/collections"},
	} {
		var body any
		if req.method == "POST" {
			body = search
		}
		rec := a.do(req.method, req.path, body)
		a.expectError(rec, http.StatusServiceUnavailable, codeOverloaded)
		if rec.Header().Get("Retry-After") == "" {
			t.Errorf("%s %s: 503 overloaded without Retry-After", req.method, req.path)
		}
	}

	// Writes have their own pool, and the probes and the scrape none.
	a.expect(a.do("POST", "/v1/collections/docs/vectors", map[string]any{
		"vectors": []any{map[string]any{"id": "a", "values": []float32{1, 0, 0, 0}}},
	}), http.StatusOK)
	a.expect(a.do("GET", "/healthz", nil), http.StatusOK)
	a.expect(a.do("GET", "/readyz", nil), http.StatusOK)

	rec := a.do("GET", "/metrics", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /metrics = %d while the read pool is full", rec.Code)
	}
	for _, want := range []string{
		`govecdb_http_inflight{pool="read"} 1`,
		`govecdb_http_inflight{pool="write"} 0`,
		`govecdb_http_rejected_total{pool="read"} 3`,
		`govecdb_http_rejected_total{pool="write"} 0`,
	} {
		if !strings.Contains(rec.Body.String(), want) {
			t.Errorf("metrics missing %q", want)
		}
	}

	a.server.reads.release()
	a.expect(a.do("POST", "/v1/collections/docs/search", search), http.StatusOK)
}

// A request whose client has already gone is not served: the library takes
// no context, so not starting is the only way to not finish.
func TestAGoneClientIsNotServed(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req := httptest.NewRequestWithContext(ctx, "POST", "/v1/collections/docs/vectors",
		strings.NewReader(`{"vectors":[{"id":"a","values":[1,0,0,0]}]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	a.server.ServeHTTP(rec, req)
	if rec.Code != statusClientGone {
		t.Fatalf("status = %d, want %d", rec.Code, statusClientGone)
	}
	a.expectError(a.do("GET", "/v1/collections/docs/vectors/a", nil), http.StatusNotFound, codeNotFound)
	if a.server.writes.inflight.Load() != 0 {
		t.Fatal("the write slot was not given back")
	}
}

// Negative means no limit, for an embedder that sheds load itself.
func TestNegativePoolSizeIsUnlimited(t *testing.T) {
	a := newAPI(t, Config{MaxInFlightReads: -1})
	for range 1000 {
		if !a.server.reads.tryAcquire() {
			t.Fatal("an unlimited pool refused")
		}
	}
}
