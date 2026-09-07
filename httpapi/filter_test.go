package httpapi

import (
	"math/rand/v2"
	"net/http"
	"slices"
	"testing"
)

// filterFixture builds a small collection whose metadata covers every value
// type, so one corpus serves every operator.
func filterFixture(t *testing.T) *api {
	t.Helper()

	a := newAPI(t, Config{})
	a.createCollection("docs", 4)
	a.expect(a.do("POST", "/v1/collections/docs/vectors", map[string]any{
		"vectors": []any{
			map[string]any{"id": "a", "values": []float32{1, 0, 0, 0}, "metadata": map[string]any{
				"source": "handbook", "page": 1, "score": 0.5, "draft": true,
			}},
			map[string]any{"id": "b", "values": []float32{1, 0, 0, 0}, "metadata": map[string]any{
				"source": "handbook", "page": 10, "score": 1.5, "draft": false,
			}},
			map[string]any{"id": "c", "values": []float32{1, 0, 0, 0}, "metadata": map[string]any{
				"source": "notes", "page": 20, "retracted": true,
			}},
			map[string]any{"id": "d", "values": []float32{1, 0, 0, 0}},
		},
	}), http.StatusOK)
	return a
}

// searchIDs runs a filtered search wide enough to see the whole corpus and
// returns the ids it admitted, sorted so the assertion is about membership.
func (a *api) searchIDs(filter any) []string {
	a.t.Helper()

	body := map[string]any{"query": []float32{1, 0, 0, 0}, "k": 10}
	if filter != nil {
		body["filter"] = filter
	}
	out := a.expect(a.do("POST", "/v1/collections/docs/search", body), http.StatusOK)

	matches, _ := out["matches"].([]any)
	ids := make([]string, 0, len(matches))
	for _, m := range matches {
		ids = append(ids, m.(map[string]any)["id"].(string))
	}
	slices.Sort(ids)
	return ids
}

func TestFilterOperators(t *testing.T) {
	a := filterFixture(t)

	for _, tc := range []struct {
		name   string
		filter map[string]any
		want   []string
	}{
		{"eq", map[string]any{"op": "eq", "key": "source", "value": "handbook"}, []string{"a", "b"}},
		{"eq on a bool", map[string]any{"op": "eq", "key": "draft", "value": true}, []string{"a"}},
		{"ne", map[string]any{"op": "ne", "key": "source", "value": "handbook"}, []string{"c"}},
		{"lt", map[string]any{"op": "lt", "key": "page", "value": 10}, []string{"a"}},
		{"lte", map[string]any{"op": "lte", "key": "page", "value": 10}, []string{"a", "b"}},
		{"gt", map[string]any{"op": "gt", "key": "page", "value": 10}, []string{"c"}},
		{"gte", map[string]any{"op": "gte", "key": "page", "value": 10}, []string{"b", "c"}},
		{"in", map[string]any{"op": "in", "key": "page", "values": []any{1, 20}}, []string{"a", "c"}},
		{"exists", map[string]any{"op": "exists", "key": "retracted"}, []string{"c"}},

		// In with no values matches nothing: filtering by a set that turned out
		// to be empty should return nothing rather than everything.
		{"empty in", map[string]any{"op": "in", "key": "page", "values": []any{}}, []string{}},

		{"and", map[string]any{"op": "and", "filters": []any{
			map[string]any{"op": "eq", "key": "source", "value": "handbook"},
			map[string]any{"op": "gte", "key": "page", "value": 10},
		}}, []string{"b"}},

		{"or", map[string]any{"op": "or", "filters": []any{
			map[string]any{"op": "eq", "key": "source", "value": "notes"},
			map[string]any{"op": "eq", "key": "page", "value": 1},
		}}, []string{"a", "c"}},

		{"not", map[string]any{"op": "not",
			"filter": map[string]any{"op": "exists", "key": "source"},
		}, []string{"d"}},

		// And() with no filters matches everything, Or() with none matches
		// nothing — the identity elements, and what makes a filter built in a
		// loop behave when the loop runs zero times.
		{"empty and", map[string]any{"op": "and", "filters": []any{}}, []string{"a", "b", "c", "d"}},
		{"empty or", map[string]any{"op": "or", "filters": []any{}}, []string{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := a.searchIDs(tc.filter)
			if !slices.Equal(got, tc.want) {
				t.Errorf("ids = %v, want %v", got, tc.want)
			}
		})
	}
}

// Every comparison is false on an absent key, Ne included: Ne("k", x) means "k
// is there and is not x". Not(Eq(...)) is how to also match vectors lacking it.
// This is the rule most likely to be "fixed" into SQL's three-valued logic.
func TestFilterOnAnAbsentKeyIsFalse(t *testing.T) {
	a := filterFixture(t)

	if got := a.searchIDs(map[string]any{"op": "ne", "key": "source", "value": "handbook"}); slices.Contains(got, "d") {
		t.Errorf("ne matched %v, which includes the vector with no source at all", got)
	}
	want := []string{"c", "d"}
	got := a.searchIDs(map[string]any{"op": "not",
		"filter": map[string]any{"op": "eq", "key": "source", "value": "handbook"},
	})
	if !slices.Equal(got, want) {
		t.Errorf("not(eq) = %v, want %v — including the vector with no source", got, want)
	}
}

// A number written without a decimal point is stored as an integer and one
// written with a point as a float, and the two still compare exactly. So a
// client that serializes 10 and a client that serializes 10.0 agree.
func TestFilterNumbersCompareAcrossTypes(t *testing.T) {
	a := filterFixture(t)

	for _, tc := range []struct {
		name   string
		filter map[string]any
		want   []string
	}{
		{"float operand against a stored int", map[string]any{"op": "eq", "key": "page", "value": 10.0}, []string{"b"}},
		{"int operand against a stored int", map[string]any{"op": "eq", "key": "page", "value": 10}, []string{"b"}},
		{"int operand against a stored float", map[string]any{"op": "lt", "key": "score", "value": 1}, []string{"a"}},
		{"float operand against a stored float", map[string]any{"op": "gte", "key": "score", "value": 1.5}, []string{"b"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := a.searchIDs(tc.filter); !slices.Equal(got, tc.want) {
				t.Errorf("ids = %v, want %v", got, tc.want)
			}
		})
	}
}

// The property that makes filtering worth doing inside the traversal: a
// selective filter still returns K results. Post-filtering the nearest K would
// return almost none of these.
func TestSelectiveFilterStillReturnsK(t *testing.T) {
	const (
		total   = 500
		oneIn   = 50
		want    = total / oneIn // 10 vectors carry the key
		dim     = 16
		wantedK = 10
	)

	a := newAPI(t, Config{})
	a.expect(a.do("POST", "/v1/collections", map[string]any{
		"name": "docs", "dimension": dim, "sync_policy": "never",
	}), http.StatusCreated)

	rng := rand.New(rand.NewPCG(1, 2))
	vectors := make([]any, 0, total)
	for i := range total {
		values := make([]float32, dim)
		for d := range values {
			values[d] = float32(rng.NormFloat64())
		}
		v := map[string]any{"id": string(rune('a'+i%26)) + itoa(i), "values": values}
		if i%oneIn == 0 {
			v["metadata"] = map[string]any{"keep": true}
		}
		vectors = append(vectors, v)
	}
	a.expect(a.do("POST", "/v1/collections/docs/vectors",
		map[string]any{"vectors": vectors}), http.StatusOK)

	query := make([]float32, dim)
	for d := range query {
		query[d] = float32(rng.NormFloat64())
	}
	body := a.expect(a.do("POST", "/v1/collections/docs/search", map[string]any{
		"query": query, "k": wantedK,
		"filter": map[string]any{"op": "eq", "key": "keep", "value": true},
	}), http.StatusOK)

	matches, _ := body["matches"].([]any)
	if len(matches) != want {
		t.Fatalf("got %d matches, want %d — a filter applied to the results rather "+
			"than inside the traversal is what this looks like", len(matches), want)
	}
	for _, m := range matches {
		md, _ := m.(map[string]any)["metadata"].(map[string]any)
		if md["keep"] != true {
			t.Fatalf("match %v does not satisfy the filter", m)
		}
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

func TestFilterRejectsMalformedTrees(t *testing.T) {
	a := filterFixture(t)

	for _, tc := range []struct {
		name   string
		filter any
	}{
		{"unknown op", map[string]any{"op": "matches", "key": "source", "value": "x"}},
		{"no op", map[string]any{"key": "source", "value": "x"}},
		{"no key", map[string]any{"op": "eq", "value": "x"}},
		{"no value", map[string]any{"op": "eq", "key": "source"}},
		{"null value", map[string]any{"op": "eq", "key": "source", "value": nil}},
		{"object as a value", map[string]any{"op": "eq", "key": "source", "value": map[string]any{"a": 1}}},
		{"array as a value", map[string]any{"op": "eq", "key": "source", "value": []any{1}}},
		{"not with no filter", map[string]any{"op": "not"}},
		{"in with no key", map[string]any{"op": "in", "values": []any{1}}},
		{"null inside and", map[string]any{"op": "and", "filters": []any{nil}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := a.do("POST", "/v1/collections/docs/search", map[string]any{
				"query": []float32{1, 0, 0, 0}, "k": 1, "filter": tc.filter,
			})
			a.expectError(rec, http.StatusBadRequest, codeInvalidFilter)
		})
	}
}

// A filter is recursion driven by a request body, and a stack overflow in Go is
// not a recovered panic — it is the process.
func TestFilterDepthIsBounded(t *testing.T) {
	a := filterFixture(t)

	deep := map[string]any{"op": "exists", "key": "source"}
	for range maxFilterDepth + 2 {
		deep = map[string]any{"op": "not", "filter": deep}
	}
	rec := a.do("POST", "/v1/collections/docs/search", map[string]any{
		"query": []float32{1, 0, 0, 0}, "k": 1, "filter": deep,
	})
	a.expectError(rec, http.StatusBadRequest, codeInvalidFilter)

	// And a tree just inside the limit still works, so the bound is a bound and
	// not an off-by-one that refuses ordinary queries.
	ok := map[string]any{"op": "exists", "key": "source"}
	for range maxFilterDepth - 2 {
		ok = map[string]any{"op": "not", "filter": ok}
	}
	a.expect(a.do("POST", "/v1/collections/docs/search", map[string]any{
		"query": []float32{1, 0, 0, 0}, "k": 1, "filter": ok,
	}), http.StatusOK)
}
