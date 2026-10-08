package httpapi

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// Width is bounded as depth is. A filter runs once per node a search visits,
// so a wide one that matches nothing is a graph walk multiplied by its width:
// an `in` of 100,000 values was 4.8 s of CPU from a 592 KB body.
func TestFilterWidthIsBounded(t *testing.T) {
	a := newAPI(t, Config{MaxFilterClauses: 10, MaxFilterValues: 20})
	a.createCollection("docs", 4)
	a.expect(a.do("POST", "/v1/collections/docs/vectors", map[string]any{
		"vectors": []any{map[string]any{"id": "a", "values": []float32{1, 0, 0, 0},
			"metadata": map[string]any{"n": 3}}},
	}), http.StatusOK)

	values := func(n int) []any {
		vs := make([]any, n)
		for i := range vs {
			vs[i] = i
		}
		return vs
	}
	in := func(n int) map[string]any { return map[string]any{"op": "in", "key": "n", "values": values(n)} }
	and := func(fs ...any) map[string]any { return map[string]any{"op": "and", "filters": fs} }
	search := func(a *api, filter any) *httptest.ResponseRecorder {
		return a.do("POST", "/v1/collections/docs/search", map[string]any{
			"query": []float32{1, 0, 0, 0}, "k": 1, "filter": filter,
		})
	}

	a.expect(search(a, in(20)), http.StatusOK)
	a.expectError(search(a, in(21)), http.StatusBadRequest, codeInvalidFilter)
	// The value budget is per request, not per clause.
	a.expectError(search(a, and(in(10), in(11))), http.StatusBadRequest, codeInvalidFilter)

	// Ten clauses: the `and` and nine children.
	nine := make([]any, 9)
	for i := range nine {
		nine[i] = map[string]any{"op": "exists", "key": "n"}
	}
	a.expect(search(a, and(nine...)), http.StatusOK)
	a.expectError(search(a, and(append(nine, map[string]any{"op": "exists", "key": "n"})...)),
		http.StatusBadRequest, codeInvalidFilter)

	// The budget is a copy per request: the refusals above spent nothing.
	a.expect(search(a, in(20)), http.StatusOK)

	// And the defaults apply when the config says nothing.
	d := filterFixture(t)
	d.expect(search(d, map[string]any{"op": "in", "key": "page", "values": values(DefaultMaxFilterValues)}),
		http.StatusOK)
	d.expectError(search(d, map[string]any{"op": "in", "key": "page", "values": values(DefaultMaxFilterValues + 1)}),
		http.StatusBadRequest, codeInvalidFilter)
}
