package httpapi

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

// JSON has one number type and this database has two, so the rule is the syntax
// a client wrote. Both directions of getting it wrong are silent, which is why
// there is a test rather than a comment.
func TestMetadataNumberTyping(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	a.expect(a.do("POST", "/v1/collections/docs/vectors", `{"vectors":[{
		"id": "a",
		"values": [1,0,0,0],
		"metadata": {
			"page": 10,
			"score": 0.5,
			"whole_float": 10.0,
			"nanos": 1700000000000000001,
			"negative": -3,
			"exponent": 1e3
		}
	}]}`), http.StatusOK)

	raw := a.do("GET", "/v1/collections/docs/vectors/a", nil).Body.String()
	for _, want := range []string{
		`"page":10`,
		`"score":0.5`,
		// Written with a decimal point, so it is a float — and a float that
		// happens to be whole marshals back without one. The point is that it
		// did not become an integer.
		`"whole_float":10`,
		// The one that would round if every number became a float64: 1.7e18 needs
		// more than the 53 bits of mantissa a float64 has, and this is the
		// magnitude of an ordinary nanosecond timestamp.
		`"nanos":1700000000000000001`,
		`"negative":-3`,
	} {
		if !strings.Contains(raw, want) {
			t.Errorf("response is missing %s:\n%s", want, raw)
		}
	}

	// An exponent means float, whatever the value looks like.
	if !strings.Contains(raw, `"exponent":1000`) {
		t.Errorf("exponent not stored as a number: %s", raw)
	}
}

func TestMetadataRejectsWhatTheDatabaseCannotHold(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	for _, tc := range []struct{ name, metadata string }{
		{"a nested object", `{"nested": {"a": 1}}`},
		{"an array", `{"tags": ["a", "b"]}`},
		{"null", `{"missing": null}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := a.do("POST", "/v1/collections/docs/vectors",
				`{"vectors":[{"id":"a","values":[1,0,0,0],"metadata":`+tc.metadata+`}]}`)
			a.expectError(rec, http.StatusBadRequest, codeInvalidMetadata)
		})
	}
}

func TestAddRejectsBadVectors(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	for _, tc := range []struct {
		name, body, code string
	}{
		{"no vectors at all", `{"vectors":[]}`, codeInvalidRequest},
		{"wrong dimension", `{"vectors":[{"id":"a","values":[1,0]}]}`, codeInvalidVector},
		{"no values", `{"vectors":[{"id":"a"}]}`, codeInvalidVector},
		{"empty id", `{"vectors":[{"id":"","values":[1,0,0,0]}]}`, codeInvalidVector},
		{"unknown field", `{"vectors":[{"id":"a","values":[1,0,0,0],"embedding":[1]}]}`, codeInvalidRequest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a.expectError(a.do("POST", "/v1/collections/docs/vectors", tc.body),
				http.StatusBadRequest, tc.code)
		})
	}
}

// AddBatch validates every vector before writing any, so one bad record leaves
// the collection untouched. The guarantee is about validation, not durability.
func TestABadVectorLeavesTheBatchUnapplied(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	a.expectError(a.do("POST", "/v1/collections/docs/vectors", `{"vectors":[
		{"id":"good","values":[1,0,0,0]},
		{"id":"bad","values":[1,0]}
	]}`), http.StatusBadRequest, codeInvalidVector)

	a.expectError(a.do("GET", "/v1/collections/docs/vectors/good", nil),
		http.StatusNotFound, codeNotFound)
}

func TestScalarTyping(t *testing.T) {
	decodeValue := func(t *testing.T, raw string) any {
		t.Helper()
		dec := json.NewDecoder(strings.NewReader(raw))
		dec.UseNumber()
		var v any
		if err := dec.Decode(&v); err != nil {
			t.Fatalf("decode %s: %v", raw, err)
		}
		return v
	}

	for _, tc := range []struct {
		raw  string
		want any
	}{
		{`"text"`, "text"},
		{`true`, true},
		{`0`, int64(0)},
		{`-7`, int64(-7)},
		{`9223372036854775807`, int64(9223372036854775807)},
		{`0.5`, 0.5},
		{`1.0`, float64(1)},
		{`1e3`, float64(1000)},
		// Past int64, so it can only be a float — and saying so is better than
		// silently wrapping to a negative number.
		{`9223372036854775808`, float64(9223372036854775808)},
	} {
		got, err := scalar(decodeValue(t, tc.raw))
		if err != nil {
			t.Errorf("scalar(%s) = %v", tc.raw, err)
			continue
		}
		if got != tc.want {
			t.Errorf("scalar(%s) = %v (%T), want %v (%T)", tc.raw, got, got, tc.want, tc.want)
		}
	}
}
