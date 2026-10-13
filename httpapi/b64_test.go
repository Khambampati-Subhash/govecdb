package httpapi

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand/v2"
	"net/http"
	"slices"
	"strings"
	"testing"
)

// refB64 is the binary format written the obvious way: every float32
// little-endian, then one base64 encoding of the lot.
func refB64(vs []float32) string {
	raw := make([]byte, 0, 4*len(vs))
	for _, v := range vs {
		raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
	}
	return base64.StdEncoding.EncodeToString(raw)
}

// The chunked encoder must equal one encoding of the whole vector — the
// lengths straddle its 192-float chunk, where concatenation would show a seam.
func TestB64EncodesAsOneBlock(t *testing.T) {
	r := rand.New(rand.NewPCG(4, 4))
	for _, n := range []int{0, 1, 2, 3, 4, 191, 192, 193, 383, 384, 385, 512, 1536} {
		vs := make([]float32, n)
		for i := range vs {
			vs[i] = randFloat32(r)
		}
		got := string(appendB64Float32s(nil, vs))
		if want := `"` + refB64(vs) + `"`; got != want {
			t.Fatalf("n=%d: chunked encoding differs from one block\n got %.60s…\nwant %.60s…", n, got, want)
		}

		var back b64Float32s
		if err := json.Unmarshal([]byte(got), &back); err != nil {
			t.Fatalf("n=%d: decode: %v", n, err)
		}
		// Bits, not values: the format is a copy, so -0 must stay -0.
		if !slices.EqualFunc(vs, back, func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }) {
			t.Fatalf("n=%d: round trip changed the values", n)
		}
	}
	if got := string(appendB64Float32s(nil, nil)); got != "null" {
		t.Errorf("nil = %s, want null", got)
	}
}

func TestB64DecodeRefuses(t *testing.T) {
	one := refB64([]float32{1})
	for _, tc := range []struct {
		why, raw string
		ok       bool
	}{
		{"null leaves the slice", `null`, true},
		{"empty", `""`, true},
		{"one float", `"` + one + `"`, true},
		{`"\/" is a legal spelling of "/"`, `"` + strings.ReplaceAll(refB64([]float32{-1, 2.5e-38, 7}), "/", `\/`) + `"`, true},
		{"numbers", `[1,2]`, false},
		{"not base64", `"!!!!"`, false},
		{"unpadded", `"` + strings.TrimRight(one, "=") + `"`, false},
		{"URL alphabet", `"` + strings.NewReplacer("+", "-", "/", "_").Replace(refB64([]float32{-1, 2.5e-38, 7})) + `"`, false},
		{"not whole float32s", `"` + base64.StdEncoding.EncodeToString([]byte{1, 2, 3}) + `"`, false},
	} {
		var got b64Float32s
		err := json.Unmarshal([]byte(tc.raw), &got)
		if (err == nil) != tc.ok {
			t.Errorf("%s: %s: err = %v, want ok=%v", tc.why, tc.raw, err, tc.ok)
		}
	}
	// The URL case needs '/' or '+' in the reference, or it proves nothing.
	if s := refB64([]float32{-1, 2.5e-38, 7}); !strings.ContainsAny(s, "+/") {
		t.Fatalf("%s has neither + nor /; pick other values", s)
	}
}

// TestB64EndToEnd writes in binary, reads back both ways, and searches in
// binary: the two spellings are one vector.
func TestB64EndToEnd(t *testing.T) {
	a := newAPI(t, Config{})
	// Euclidean, so the index stores what was sent rather than its unit vector.
	a.expect(a.do("POST", "/v1/collections", map[string]any{
		"name": "docs", "dimension": 4, "metric": "euclidean", "sync_policy": "never",
	}), http.StatusCreated)

	vecs := map[string][]float32{
		"a": {1, 0, 0, 0},
		"b": {0, 1, 0, 0},
		"c": {-0.1, 2.5e-38, 3.4e38, 7}, // a subnormal-adjacent, a near-max, a negative
	}
	var in []any
	for id, v := range vecs {
		in = append(in, map[string]any{"id": id, "values_b64": refB64(v)})
	}
	a.expect(a.do("POST", "/v1/collections/docs/vectors", map[string]any{"vectors": in}), http.StatusOK)

	// One record, both formats.
	asNumbers := a.expect(a.do("GET", "/v1/collections/docs/vectors/c", nil), http.StatusOK)
	asB64 := a.expect(a.do("GET", "/v1/collections/docs/vectors/c?encoding=b64", nil), http.StatusOK)
	if asB64["values_b64"] != refB64(vecs["c"]) {
		t.Errorf("values_b64 = %v, want %s", asB64["values_b64"], refB64(vecs["c"]))
	}
	if _, ok := asB64["values"]; ok {
		t.Error("a b64 record also carries values")
	}
	if _, ok := asNumbers["values_b64"]; ok {
		t.Error("a JSON record carries values_b64")
	}
	if a.do("GET", "/v1/collections/docs/vectors/c?encoding=json", nil).Body.String() !=
		a.do("GET", "/v1/collections/docs/vectors/c", nil).Body.String() {
		t.Error("encoding=json differs from the default")
	}

	// A page and a batch carry the format through.
	page := a.expect(a.do("GET", "/v1/collections/docs/vectors?encoding=b64", nil), http.StatusOK)
	batch := a.expect(a.do("POST", "/v1/collections/docs/vectors/get?encoding=b64",
		map[string]any{"ids": []string{"c", "zz"}}), http.StatusOK)
	for name, body := range map[string]map[string]any{"page": page, "batch": batch} {
		for _, rec := range body["vectors"].([]any) {
			rec := rec.(map[string]any)
			if want := refB64(vecs[rec["id"].(string)]); rec["values_b64"] != want {
				t.Errorf("%s: %v values_b64 = %v, want %s", name, rec["id"], rec["values_b64"], want)
			}
		}
	}

	body := a.expect(a.do("POST", "/v1/collections/docs/search", map[string]any{
		"query_b64": refB64([]float32{1, 0.1, 0, 0}), "k": 1,
	}), http.StatusOK)
	if got := body["matches"].([]any)[0].(map[string]any)["id"]; got != "a" {
		t.Errorf("nearest = %v, want a", got)
	}
}

func TestB64RequestsThatAreRefused(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)
	unit := refB64([]float32{1, 0, 0, 0})
	nan := refB64([]float32{1, float32(math.NaN()), 0, 0})

	for _, tc := range []struct {
		why, method, path string
		body              any
		code              string
	}{
		{"both spellings of a vector", "POST", "/v1/collections/docs/vectors",
			`{"vectors":[{"id":"a","values":[1,0,0,0],"values_b64":"` + unit + `"}]}`, codeInvalidRequest},
		{"both spellings of a query", "POST", "/v1/collections/docs/search",
			`{"query":[1,0,0,0],"query_b64":"` + unit + `","k":1}`, codeInvalidRequest},
		// Binary can spell what JSON cannot, and it is refused the same way.
		{"a NaN", "POST", "/v1/collections/docs/vectors",
			`{"vectors":[{"id":"a","values_b64":"` + nan + `"}]}`, codeInvalidVector},
		{"a NaN query", "POST", "/v1/collections/docs/search",
			`{"query_b64":"` + nan + `","k":1}`, codeInvalidVector},
		{"wrong dimension", "POST", "/v1/collections/docs/vectors",
			`{"vectors":[{"id":"a","values_b64":"` + refB64([]float32{1, 0}) + `"}]}`, codeInvalidVector},
		{"bad base64", "POST", "/v1/collections/docs/vectors",
			`{"vectors":[{"id":"a","values_b64":"AAAA*AAA"}]}`, codeInvalidRequest},
		{"unknown encoding", "GET", "/v1/collections/docs/vectors/a?encoding=base64", nil, codeInvalidRequest},
		{"unknown encoding on a page", "GET", "/v1/collections/docs/vectors?encoding=hex", nil, codeInvalidRequest},
		{"unknown encoding on a batch", "POST", "/v1/collections/docs/vectors/get?encoding=B64",
			`{"ids":["a"]}`, codeInvalidRequest},
	} {
		a.expectError(a.do(tc.method, tc.path, tc.body), http.StatusBadRequest, tc.code)
		if t.Failed() {
			t.Fatalf("failed on: %s", tc.why)
		}
	}
	// None of them wrote anything.
	if body := a.expect(a.do("GET", "/v1/collections/docs/vectors", nil), http.StatusOK); len(body["vectors"].([]any)) != 0 {
		t.Errorf("a refused request left vectors behind: %v", body["vectors"])
	}
}
