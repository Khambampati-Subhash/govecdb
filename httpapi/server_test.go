package httpapi

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/khambampati-subhash/govecdb/service"
)

// api drives the handler directly rather than over a socket. What is under test
// is routing, decoding and status mapping, none of which a real listener would
// exercise differently — and cmd/govecdbd has the end-to-end test that does go
// over TCP.
type api struct {
	t      *testing.T
	server *Server
	token  string
}

func newAPI(t *testing.T, cfg Config) *api {
	t.Helper()

	if cfg.Manager == nil {
		mgr, err := service.NewManager(t.TempDir(), service.Options{})
		if err != nil {
			t.Fatalf("NewManager: %v", err)
		}
		t.Cleanup(func() { mgr.Close() })
		cfg.Manager = mgr
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return &api{t: t, server: s, token: cfg.AuthToken}
}

// do sends a request. body may be nil, a string (sent verbatim, so a test can
// send JSON no Go type would produce), or a value to marshal.
func (a *api) do(method, path string, body any) *httptest.ResponseRecorder {
	a.t.Helper()

	var reader io.Reader
	switch b := body.(type) {
	case nil:
	case string:
		reader = strings.NewReader(b)
	default:
		raw, err := json.Marshal(b)
		if err != nil {
			a.t.Fatalf("marshal request: %v", err)
		}
		reader = strings.NewReader(string(raw))
	}

	req := httptest.NewRequest(method, path, reader)
	if reader != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if a.token != "" {
		req.Header.Set("Authorization", "Bearer "+a.token)
	}
	rec := httptest.NewRecorder()
	a.server.ServeHTTP(rec, req)
	return rec
}

// expect asserts the status and decodes the body into a map.
func (a *api) expect(rec *httptest.ResponseRecorder, status int) map[string]any {
	a.t.Helper()

	if rec.Code != status {
		a.t.Fatalf("status = %d, want %d; body: %s", rec.Code, status, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		a.t.Fatalf("Content-Type = %q, want application/json", ct)
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		a.t.Fatalf("decode body %q: %v", rec.Body.String(), err)
	}
	return out
}

// expectError asserts a failure carries the documented shape and code.
func (a *api) expectError(rec *httptest.ResponseRecorder, status int, code string) {
	a.t.Helper()

	body := a.expect(rec, status)
	detail, ok := body["error"].(map[string]any)
	if !ok {
		a.t.Fatalf("body %v has no error object", body)
	}
	if detail["code"] != code {
		a.t.Errorf("code = %v, want %q (message: %v)", detail["code"], code, detail["message"])
	}
	if msg, _ := detail["message"].(string); msg == "" {
		a.t.Error("error message is empty")
	}
}

func (a *api) createCollection(name string, dim int) {
	a.t.Helper()
	rec := a.do("POST", "/v1/collections", map[string]any{
		"name": name, "dimension": dim, "sync_policy": "never",
	})
	a.expect(rec, http.StatusCreated)
}

func TestCollectionLifecycle(t *testing.T) {
	a := newAPI(t, Config{})

	body := a.expect(a.do("POST", "/v1/collections", map[string]any{
		"name": "docs", "dimension": 4, "metric": "euclidean",
		"m": 8, "sync_policy": "never", "snapshot_interval": "5m",
	}), http.StatusCreated)

	if body["name"] != "docs" || body["metric"] != "euclidean" || body["m"] != 8.0 {
		t.Fatalf("create returned %v", body)
	}
	// Defaults are resolved and reported, not left as zeros for a client to
	// guess at.
	if body["ef_construction"] != 200.0 || body["snapshots_kept"] != 2.0 {
		t.Errorf("defaults not reported: %v", body)
	}
	if body["snapshot_interval"] != "5m0s" {
		t.Errorf("snapshot_interval = %v, want 5m0s", body["snapshot_interval"])
	}

	list := a.expect(a.do("GET", "/v1/collections", nil), http.StatusOK)
	cols, ok := list["collections"].([]any)
	if !ok || len(cols) != 1 {
		t.Fatalf("list = %v, want one collection", list)
	}

	got := a.expect(a.do("GET", "/v1/collections/docs", nil), http.StatusOK)
	if got["name"] != "docs" {
		t.Fatalf("get = %v", got)
	}

	dropped := a.expect(a.do("DELETE", "/v1/collections/docs", nil), http.StatusOK)
	if dropped["dropped"] != "docs" {
		t.Errorf("drop = %v", dropped)
	}
	a.expectError(a.do("GET", "/v1/collections/docs", nil), http.StatusNotFound, codeNotFound)
}

// A silently ignored "dimensions" is a collection built at the wrong width,
// succeeding now and discovered weeks later with data in it.
func TestCreateRejectsAnUnknownField(t *testing.T) {
	a := newAPI(t, Config{})
	rec := a.do("POST", "/v1/collections", `{"name":"docs","dimensions":4}`)
	a.expectError(rec, http.StatusBadRequest, codeInvalidRequest)
}

func TestCreateRejectsABadName(t *testing.T) {
	a := newAPI(t, Config{})
	for _, name := range []string{"../escape", "has space", ""} {
		rec := a.do("POST", "/v1/collections", map[string]any{"name": name, "dimension": 4})
		a.expectError(rec, http.StatusBadRequest, codeInvalidName)
	}
}

func TestCreateReportsADuplicate(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)
	rec := a.do("POST", "/v1/collections", map[string]any{"name": "docs", "dimension": 4})
	a.expectError(rec, http.StatusConflict, codeAlreadyExists)
}

func TestCreateRejectsAnUnknownMetric(t *testing.T) {
	a := newAPI(t, Config{})
	rec := a.do("POST", "/v1/collections", map[string]any{
		"name": "docs", "dimension": 4, "metric": "manhattan",
	})
	a.expectError(rec, http.StatusBadRequest, codeInvalidSpec)
}

func TestAddAndSearch(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	added := a.expect(a.do("POST", "/v1/collections/docs/vectors", map[string]any{
		"vectors": []any{
			map[string]any{"id": "a", "values": []float32{1, 0, 0, 0}, "metadata": map[string]any{"page": 1}},
			map[string]any{"id": "b", "values": []float32{0, 1, 0, 0}},
			map[string]any{"id": "c", "values": []float32{0, 0, 1, 0}},
		},
	}), http.StatusOK)
	if added["added"] != 3.0 {
		t.Fatalf("added = %v, want 3", added["added"])
	}

	body := a.expect(a.do("POST", "/v1/collections/docs/search", map[string]any{
		"query": []float32{1, 0, 0, 0}, "k": 2,
	}), http.StatusOK)

	matches, ok := body["matches"].([]any)
	if !ok || len(matches) != 2 {
		t.Fatalf("matches = %v, want 2", body["matches"])
	}
	first := matches[0].(map[string]any)
	if first["id"] != "a" {
		t.Errorf("nearest = %v, want a", first["id"])
	}
	md, _ := first["metadata"].(map[string]any)
	if md["page"] != 1.0 {
		t.Errorf("metadata = %v, want page=1", md)
	}
	// A vector with no metadata omits the field rather than sending null.
	if _, present := matches[1].(map[string]any)["metadata"]; present {
		t.Errorf("second match carries a metadata field: %v", matches[1])
	}
}

func TestSearchRejectsABadRequest(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	for _, tc := range []struct {
		why  string
		body map[string]any
		code string
	}{
		{"wrong dimension", map[string]any{"query": []float32{1, 0}, "k": 1}, codeInvalidVector},
		{"k is zero", map[string]any{"query": []float32{1, 0, 0, 0}, "k": 0}, codeInvalidRequest},
		{"k is negative", map[string]any{"query": []float32{1, 0, 0, 0}, "k": -1}, codeInvalidRequest},
		{"no query", map[string]any{"k": 1}, codeInvalidVector},
	} {
		rec := a.do("POST", "/v1/collections/docs/search", tc.body)
		a.expectError(rec, http.StatusBadRequest, tc.code)
		if t.Failed() {
			t.Fatalf("failed on: %s", tc.why)
		}
	}
}

func TestGetAndDeleteVector(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)
	a.expect(a.do("POST", "/v1/collections/docs/vectors", map[string]any{
		"vectors": []any{map[string]any{
			"id": "a", "values": []float32{1, 0, 0, 0},
			"metadata": map[string]any{"source": "handbook"},
		}},
	}), http.StatusOK)

	got := a.expect(a.do("GET", "/v1/collections/docs/vectors/a", nil), http.StatusOK)
	if got["id"] != "a" {
		t.Fatalf("get = %v", got)
	}
	if md := got["metadata"].(map[string]any); md["source"] != "handbook" {
		t.Errorf("metadata = %v", md)
	}

	a.expectError(a.do("GET", "/v1/collections/docs/vectors/nope", nil),
		http.StatusNotFound, codeNotFound)

	a.expect(a.do("DELETE", "/v1/collections/docs/vectors/a", nil), http.StatusOK)
	a.expectError(a.do("GET", "/v1/collections/docs/vectors/a", nil),
		http.StatusNotFound, codeNotFound)

	// Deleting what is not there succeeds, matching the database: replay applies
	// records more than once across a snapshot boundary, so a delete that failed
	// the second time would make recovery order-sensitive.
	a.expect(a.do("DELETE", "/v1/collections/docs/vectors/a", nil), http.StatusOK)
}

// Ids are arbitrary UTF-8 up to 512 bytes and collection names are not, so the
// path has to survive things a name never contains. Percent-encoding is the
// client's job; this pins that the server decodes it rather than routing on the
// escaped form.
func TestVectorIDsSurviveThePath(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	for _, id := range []string{"doc/1", "a b", "café", "a#b", "a?b", "100%"} {
		a.expect(a.do("POST", "/v1/collections/docs/vectors", map[string]any{
			"vectors": []any{map[string]any{"id": id, "values": []float32{1, 0, 0, 0}}},
		}), http.StatusOK)

		path := "/v1/collections/docs/vectors/" + escapePathSegment(id)
		got := a.expect(a.do("GET", path, nil), http.StatusOK)
		if got["id"] != id {
			t.Errorf("GET %s returned id %v, want %q", path, got["id"], id)
		}
	}
}

// escapePathSegment percent-encodes every byte that is not unreserved, which is
// what a client library has to do. url.PathEscape leaves '/' and '+' alone in
// some positions, so the test does it exhaustively rather than relying on it.
func escapePathSegment(s string) string {
	const unreserved = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~"
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if strings.IndexByte(unreserved, s[i]) >= 0 {
			b.WriteByte(s[i])
			continue
		}
		fmt.Fprintf(&b, "%%%02X", s[i])
	}
	return b.String()
}

func TestUnknownCollection(t *testing.T) {
	a := newAPI(t, Config{})
	for _, tc := range []struct{ method, path string }{
		{"GET", "/v1/collections/nope"},
		{"DELETE", "/v1/collections/nope"},
		{"GET", "/v1/collections/nope/vectors/a"},
		{"DELETE", "/v1/collections/nope/vectors/a"},
	} {
		a.expectError(a.do(tc.method, tc.path, nil), http.StatusNotFound, codeNotFound)
	}
	a.expectError(a.do("POST", "/v1/collections/nope/search",
		map[string]any{"query": []float32{1, 0, 0, 0}, "k": 1}),
		http.StatusNotFound, codeNotFound)
}

func TestSnapshotAndCompact(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)
	a.expect(a.do("POST", "/v1/collections/docs/vectors", map[string]any{
		"vectors": []any{
			map[string]any{"id": "a", "values": []float32{1, 0, 0, 0}},
			map[string]any{"id": "b", "values": []float32{0, 1, 0, 0}},
		},
	}), http.StatusOK)
	a.expect(a.do("DELETE", "/v1/collections/docs/vectors/a", nil), http.StatusOK)

	snap := a.expect(a.do("POST", "/v1/collections/docs/snapshot", nil), http.StatusOK)
	if seq, _ := snap["snapshot_sequence"].(float64); seq == 0 {
		t.Errorf("snapshot_sequence = %v, want the sequence it covered", snap["snapshot_sequence"])
	}

	compact := a.expect(a.do("POST", "/v1/collections/docs/compact", nil), http.StatusOK)
	if compact["reclaimed"] != 1.0 {
		t.Errorf("reclaimed = %v, want the one tombstone", compact["reclaimed"])
	}

	// And the statistics a client would poll to decide to do that.
	info := a.expect(a.do("GET", "/v1/collections/docs", nil), http.StatusOK)
	stats, ok := info["stats"].(map[string]any)
	if !ok {
		t.Fatalf("no stats on a loaded collection: %v", info)
	}
	if stats["live"] != 1.0 || stats["deleted"] != 0.0 {
		t.Errorf("stats = %v, want one live and no tombstones after compaction", stats)
	}
}

// A collection that is not loaded has an unknown size, so the field is absent
// rather than zero: "0 live vectors" is a different claim from "not loaded".
func TestStatsAreOmittedForAnUnloadedCollection(t *testing.T) {
	root := t.TempDir()

	mgr, err := service.NewManager(root, service.Options{})
	if err != nil {
		t.Fatal(err)
	}
	first := &api{t: t, server: mustServer(t, Config{Manager: mgr})}
	first.createCollection("docs", 4)
	if err := mgr.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := service.NewManager(root, service.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reopened.Close() })
	a := &api{t: t, server: mustServer(t, Config{Manager: reopened})}

	info := a.expect(a.do("GET", "/v1/collections/docs", nil), http.StatusOK)
	if info["loaded"] != false {
		t.Errorf("loaded = %v, want false", info["loaded"])
	}
	if _, present := info["stats"]; present {
		t.Errorf("stats present for an unloaded collection: %v", info)
	}
}

func mustServer(t *testing.T, cfg Config) *Server {
	t.Helper()
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func TestAuth(t *testing.T) {
	a := newAPI(t, Config{AuthToken: "s3cret"})
	a.createCollection("docs", 4)

	// Correct token: the helper attaches it.
	a.expect(a.do("GET", "/v1/collections", nil), http.StatusOK)

	unauth := &api{t: t, server: a.server}
	rec := unauth.do("GET", "/v1/collections", nil)
	unauth.expectError(rec, http.StatusUnauthorized, codeUnauthorized)
	if h := rec.Header().Get("WWW-Authenticate"); !strings.Contains(h, "Bearer") {
		t.Errorf("WWW-Authenticate = %q, want a Bearer challenge", h)
	}

	wrong := &api{t: t, server: a.server, token: "wrong"}
	wrong.expectError(wrong.do("GET", "/v1/collections", nil),
		http.StatusUnauthorized, codeUnauthorized)

	// Health probes come from a load balancer that has no token, and a probe
	// that can fail for an authentication reason reports the wrong thing.
	unauth.expect(unauth.do("GET", "/healthz", nil), http.StatusOK)
	unauth.expect(unauth.do("GET", "/readyz", nil), http.StatusOK)

	// Metrics deliberately are not exempt: they name every collection and its
	// size.
	unauth.expectError(unauth.do("GET", "/metrics", nil),
		http.StatusUnauthorized, codeUnauthorized)
}

func TestNoAuthConfiguredMeansNoAuth(t *testing.T) {
	a := newAPI(t, Config{})
	a.expect(a.do("GET", "/v1/collections", nil), http.StatusOK)
	// Metrics answer in the Prometheus text format, so only the status is
	// checked here; TestMetricsExposition covers the body.
	if rec := a.do("GET", "/metrics", nil); rec.Code != http.StatusOK {
		t.Fatalf("GET /metrics = %d, want 200", rec.Code)
	}
}

func TestReadiness(t *testing.T) {
	a := newAPI(t, Config{})
	a.expect(a.do("GET", "/readyz", nil), http.StatusOK)

	// What a daemon does at the top of a graceful shutdown, so a load balancer
	// stops sending new work while requests in flight finish.
	a.server.SetReady(false)
	a.expect(a.do("GET", "/readyz", nil), http.StatusServiceUnavailable)
	// Liveness is a different question and keeps answering it.
	a.expect(a.do("GET", "/healthz", nil), http.StatusOK)
}

func TestBodyTooLarge(t *testing.T) {
	a := newAPI(t, Config{MaxBodyBytes: 64})
	a.createCollection("docs", 4)

	big := strings.Repeat("x", 512)
	rec := a.do("POST", "/v1/collections/docs/vectors",
		fmt.Sprintf(`{"vectors":[{"id":%q,"values":[1,0,0,0]}]}`, big))
	a.expectError(rec, http.StatusRequestEntityTooLarge, codePayloadTooLarge)
}

func TestUnsupportedMediaType(t *testing.T) {
	a := newAPI(t, Config{})

	req := httptest.NewRequest("POST", "/v1/collections", strings.NewReader(`{"name":"x","dimension":4}`))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	a.server.ServeHTTP(rec, req)
	a.expectError(rec, http.StatusUnsupportedMediaType, codeUnsupportedType)
}

func TestEmptyAndTrailingBodies(t *testing.T) {
	a := newAPI(t, Config{})

	a.expectError(a.do("POST", "/v1/collections", ``),
		http.StatusBadRequest, codeInvalidRequest)
	a.expectError(a.do("POST", "/v1/collections", `{"name":"a","dimension":4}{"name":"b"}`),
		http.StatusBadRequest, codeInvalidRequest)
	a.expectError(a.do("POST", "/v1/collections", `{"name":`),
		http.StatusBadRequest, codeInvalidRequest)
}

// The mux answers an unrouted path and a wrong method in plain text. A client
// parsing two error formats would discover the second one in production.
func TestRoutingErrorsAreJSON(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	a.expectError(a.do("GET", "/v1/nope", nil), http.StatusNotFound, codeNotFound)
	a.expectError(a.do("GET", "/", nil), http.StatusNotFound, codeNotFound)

	rec := a.do("DELETE", "/v1/collections", nil)
	a.expectError(rec, http.StatusMethodNotAllowed, codeInvalidRequest)
	// The useful part of the mux's own answer is kept.
	if allow := rec.Header().Get("Allow"); allow == "" {
		t.Error("405 without an Allow header")
	}
}

func TestMetricsExposition(t *testing.T) {
	a := newAPI(t, Config{Version: "1.2.3"})
	a.createCollection("docs", 4)
	a.expect(a.do("POST", "/v1/collections/docs/vectors", map[string]any{
		"vectors": []any{map[string]any{"id": "a", "values": []float32{1, 0, 0, 0}}},
	}), http.StatusOK)
	a.expectError(a.do("GET", "/v1/collections/nope", nil), http.StatusNotFound, codeNotFound)

	rec := a.do("GET", "/metrics", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Fatalf("Content-Type = %q, want the Prometheus text format", ct)
	}

	body := rec.Body.String()
	for _, want := range []string{
		`govecdb_build_info{version="1.2.3",`,
		"# TYPE govecdb_http_requests_total counter",
		`govecdb_http_requests_total{class="2xx"}`,
		`govecdb_http_requests_total{class="4xx"}`,
		"govecdb_collections 1",
		"govecdb_collections_loaded 1",
		`govecdb_collection_loaded{collection="docs"} 1`,
		`govecdb_collection_live_vectors{collection="docs"} 1`,
		`govecdb_collection_dead_ratio{collection="docs"}`,
		"govecdb_uptime_seconds",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics missing %q:\n%s", want, body)
		}
	}

	// Every HELP is followed by a TYPE, which is what a scraper needs and what
	// hand-written exposition gets wrong.
	var help, kind int
	for line := range strings.SplitSeq(body, "\n") {
		switch {
		case strings.HasPrefix(line, "# HELP"):
			help++
		case strings.HasPrefix(line, "# TYPE"):
			kind++
		}
	}
	if help != kind || help == 0 {
		t.Errorf("%d HELP lines against %d TYPE lines", help, kind)
	}
}

// A panic is a bug that has already happened; what this stops is the bug taking
// the connection with it and leaving no record.
func TestPanicBecomesA500(t *testing.T) {
	a := newAPI(t, Config{})
	handler := a.server.recoverer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		panic("boom")
	}))

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest("GET", "/v1/collections", nil))

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	if strings.Contains(rec.Body.String(), "boom") {
		t.Errorf("the panic value reached the client: %s", rec.Body.String())
	}
}

// A 5xx must not hand out internal detail; the operator gets it from the log.
func TestServerErrorsDoNotLeakDetail(t *testing.T) {
	a := newAPI(t, Config{})
	_, code, message := classify(fmt.Errorf("open /var/lib/govecdb/docs/wal: permission denied"))
	if code != codeInternal {
		t.Fatalf("code = %q, want %q", code, codeInternal)
	}
	if strings.Contains(message, "/var/lib") {
		t.Errorf("message leaks a path: %q", message)
	}
	_ = a
}
