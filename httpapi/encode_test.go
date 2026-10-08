package httpapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/khambampati-subhash/govecdb"
)

// The hand-written encoder must produce exactly what encoding/json produced
// with the types it replaced. These are those types; refMetadata, the old
// Marshaler, is at the bottom of the file.
type refMatch struct {
	ID       string      `json:"id"`
	Distance float32     `json:"distance"`
	Metadata refMetadata `json:"metadata,omitempty"`
}

type refVector struct {
	ID       string      `json:"id"`
	Values   []float32   `json:"values"`
	Metadata refMetadata `json:"metadata,omitempty"`
}

func refVectors(vs []govecdb.Vector) []refVector {
	out := make([]refVector, len(vs))
	for i, v := range vs {
		out[i] = refVector{ID: v.ID, Values: v.Values, Metadata: refMetadata(v.Metadata)}
	}
	return out
}

// stringPieces are what a string encoder gets wrong: the two JSON escapes, the
// three HTML ones, every short-form control escape and a long-form one, DEL,
// U+2028/9, multi-byte runes, a rune at the 4-byte limit, and invalid UTF-8 in
// several shapes (a stray continuation byte, a truncated sequence, an
// overlong encoding, a surrogate).
var stringPieces = []string{
	"a", "plain text", `"`, `\`, "<", ">", "&", "\b", "\f", "\n", "\r", "\t",
	"\x00", "\x1f", "\x7f", " ", "\u2028", "\u2029", "é", "漢字", "🙂",
	"\xff", "\x80", "\xe2\x82", "\xc0\xaf", "\xed\xa0\x80", "</script>",
}

func randString(r *rand.Rand) string {
	var b strings.Builder
	for range r.IntN(6) {
		if r.IntN(3) == 0 {
			b.WriteByte(byte(r.IntN(256))) // any byte at all
		} else {
			b.WriteString(stringPieces[r.IntN(len(stringPieces))])
		}
	}
	return b.String()
}

// floatPieces are where float formatting has edges: zero both ways, the 1e-6
// and 1e21 notation cutoffs either side, whole numbers that must gain ".0",
// the float64 mantissa limit, and the extremes.
var floatPieces = []float64{
	0, math.Copysign(0, -1), 1, -1, 10, 0.5, 1e-6, 9.99999e-7, 1e-7, 1e20, 1e21,
	9.99999e20, 1e22, 123456789, 1 << 53, 1<<53 + 2, 1e300, -1e-300,
	math.MaxFloat64, math.SmallestNonzeroFloat64, 0.1, 1.0 / 3,
}

func randFloat64(r *rand.Rand, finite bool) float64 {
	switch r.IntN(4) {
	case 0:
		return floatPieces[r.IntN(len(floatPieces))]
	case 1:
		if !finite {
			return []float64{math.NaN(), math.Inf(1), math.Inf(-1)}[r.IntN(3)]
		}
	case 2:
		return float64(r.IntN(2000) - 1000)
	}
	for {
		f := math.Float64frombits(r.Uint64())
		if !math.IsNaN(f) && !math.IsInf(f, 0) {
			return f
		}
	}
}

func randFloat32(r *rand.Rand) float32 {
	for {
		var f float32
		if r.IntN(3) == 0 {
			f = float32(floatPieces[r.IntN(len(floatPieces))]) // may round to Inf
		} else {
			f = math.Float32frombits(r.Uint32())
		}
		if !math.IsNaN(float64(f)) && !math.IsInf(float64(f), 0) {
			return f
		}
	}
}

func randMetadata(r *rand.Rand) govecdb.Metadata {
	n := r.IntN(5)
	if n == 0 {
		if r.IntN(2) == 0 {
			return nil
		}
		return govecdb.Metadata{}
	}
	md := make(govecdb.Metadata, n)
	for range n {
		k := randString(r)
		switch r.IntN(4) {
		case 0:
			md[k] = randString(r)
		case 1:
			md[k] = []int64{0, 1, -1, math.MaxInt64, math.MinInt64, 1700000000000000001}[r.IntN(6)]
		case 2:
			md[k] = r.IntN(2) == 0
		default:
			md[k] = randFloat64(r, false) // NaN must become null, not fail
		}
	}
	return md
}

func randVector(r *rand.Rand) govecdb.Vector {
	var vs []float32
	if r.IntN(8) != 0 { // sometimes nil, which encodes as null
		vs = make([]float32, r.IntN(6))
		for i := range vs {
			vs[i] = randFloat32(r)
		}
	}
	return govecdb.Vector{ID: randString(r), Values: vs, Metadata: randMetadata(r)}
}

func mustMarshal(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("reference encoder: %v", err)
	}
	return string(b)
}

func TestEncoderMatchesEncodingJSON(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 5))
	for trial := range 3000 {
		// Search responses.
		ms := make([]govecdb.Match, r.IntN(4))
		ref := make([]refMatch, len(ms))
		for i := range ms {
			ms[i] = govecdb.Match{ID: randString(r), Distance: randFloat32(r), Metadata: randMetadata(r)}
			ref[i] = refMatch{ID: ms[i].ID, Distance: ms[i].Distance, Metadata: refMetadata(ms[i].Metadata)}
		}
		got, err := encodeMatches(nil, ms)
		if err != nil {
			t.Fatal(err)
		}
		if want := mustMarshal(t, map[string]any{"matches": ref}); string(got) != want {
			t.Fatalf("trial %d: matches\n got %s\nwant %s", trial, got, want)
		}

		// Vectors, alone and in both page shapes.
		vs := make([]govecdb.Vector, r.IntN(4))
		for i := range vs {
			vs[i] = randVector(r)
		}
		for _, v := range vs {
			got, err := appendVector(nil, v)
			if err != nil {
				t.Fatal(err)
			}
			if want := mustMarshal(t, refVectors([]govecdb.Vector{v})[0]); string(got) != want {
				t.Fatalf("trial %d: vector\n got %s\nwant %s", trial, got, want)
			}
		}
		next := randString(r)
		for _, more := range []bool{false, true} {
			got, err := encodePage(nil, vs, next, more)
			if err != nil {
				t.Fatal(err)
			}
			body := map[string]any{"vectors": refVectors(vs)}
			if more {
				body["next"] = next
			}
			if want := mustMarshal(t, body); string(got) != want {
				t.Fatalf("trial %d: page\n got %s\nwant %s", trial, got, want)
			}
		}
		missing := []string{}
		for range r.IntN(3) {
			missing = append(missing, randString(r))
		}
		got, err = encodeBatch(nil, vs, missing)
		if err != nil {
			t.Fatal(err)
		}
		if want := mustMarshal(t, map[string]any{"vectors": refVectors(vs), "missing": missing}); string(got) != want {
			t.Fatalf("trial %d: batch\n got %s\nwant %s", trial, got, want)
		}
	}
}

// A NaN where the old encoder failed the response must still fail it: the
// distance and the vector values have no null fallback.
func TestEncoderRefusesWhatEncodingJSONRefused(t *testing.T) {
	for _, f := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := encodeMatches(nil, []govecdb.Match{{ID: "a", Distance: f}}); err == nil {
			t.Errorf("distance %v encoded", f)
		}
		if _, err := appendVector(nil, govecdb.Vector{ID: "a", Values: []float32{1, f}}); err == nil {
			t.Errorf("value %v encoded", f)
		}
	}
}

func FuzzAppendString(f *testing.F) {
	for _, s := range stringPieces {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		if got, want := string(appendString(nil, s)), mustMarshal(t, s); got != want {
			t.Fatalf("appendString(%q) = %s, want %s", s, got, want)
		}
	})
}

func FuzzAppendFloat(f *testing.F) {
	for _, x := range floatPieces {
		f.Add(math.Float64bits(x))
	}
	f.Fuzz(func(t *testing.T, bits uint64) {
		x := math.Float64frombits(bits)
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return
		}
		got, _ := appendFloat(nil, x, 64)
		if want := mustMarshal(t, x); string(got) != want {
			t.Fatalf("float64 %v = %s, want %s", x, got, want)
		}
		x32 := float32(x)
		if math.IsInf(float64(x32), 0) {
			return
		}
		got, _ = appendFloat(nil, float64(x32), 32)
		if want := mustMarshal(t, x32); string(got) != want {
			t.Fatalf("float32 %v = %s, want %s", x32, got, want)
		}
	})
}

// float32s must decode every array encoding/json decodes into a []float32 to
// the same values, and refuse what it refused.
func TestFloat32sDecodesAsEncodingJSONDoes(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 9))
	sep := []string{",", " , ", ",\n\t", ", "}
	for trial := range 3000 {
		var b strings.Builder
		b.WriteString([]string{"[", " [ ", "[\n"}[r.IntN(3)])
		n := r.IntN(8)
		for i := range n {
			if i > 0 {
				b.WriteString(sep[r.IntN(len(sep))])
			}
			switch r.IntN(4) {
			case 0:
				fmt.Fprintf(&b, "%d", r.IntN(2000)-1000)
			case 1:
				b.WriteString([]string{"1e38", "3.4028235e38", "-0", "0.0", "1E-45", "1e-50", "12.5e+3", "-7.000001"}[r.IntN(8)])
			default:
				b.WriteString(strconvG(randFloat64(r, true)))
			}
		}
		b.WriteString([]string{"]", " ] ", "\n]"}[r.IntN(3)])
		raw := `{"v":` + b.String() + `}`

		var want struct{ V []float32 }
		wantErr := json.Unmarshal([]byte(raw), &want)
		var got struct{ V float32s }
		gotErr := json.Unmarshal([]byte(raw), &got)
		if (wantErr == nil) != (gotErr == nil) {
			t.Fatalf("trial %d: %s: encoding/json err %v, float32s err %v", trial, raw, wantErr, gotErr)
		}
		if wantErr == nil && !slices.Equal(want.V, []float32(got.V)) {
			t.Fatalf("trial %d: %s: got %v, want %v", trial, raw, got.V, want.V)
		}
	}

	for _, tc := range []struct {
		raw string
		ok  bool
	}{
		{`null`, true}, {`[]`, true}, {`[ ]`, true},
		{`[1e39]`, false}, {`[-1e39]`, false},
		{`[1,null,3]`, false}, // encoding/json silently decoded a 0 here
		{`[1,"2"]`, false}, {`[[1]]`, false}, {`[true]`, false}, {`{"a":1}`, false}, {`"1,2"`, false},
	} {
		var got struct{ V float32s }
		err := json.Unmarshal([]byte(`{"v":`+tc.raw+`}`), &got)
		if (err == nil) != tc.ok {
			t.Errorf("%s: err = %v, want ok=%v", tc.raw, err, tc.ok)
		}
	}
	var empty struct{ V float32s }
	if err := json.Unmarshal([]byte(`{"v":[]}`), &empty); err != nil || empty.V == nil || len(empty.V) != 0 {
		t.Errorf("[] decoded to %#v, %v; want an empty non-nil slice, as encoding/json gives", empty.V, err)
	}
}

func strconvG(f float64) string { return fmt.Sprint(f) }

// Benchmarks: each new path beside the encoding/json path it replaced, on the
// shapes the review measured.

func benchMatches(k int) ([]govecdb.Match, []refMatch) {
	ms := make([]govecdb.Match, k)
	ref := make([]refMatch, k)
	for i := range ms {
		ms[i] = govecdb.Match{
			ID:       fmt.Sprintf("doc-%06d#chunk-%03d", i*7919, i%50),
			Distance: 0.1 + float32(i)/1000,
			Metadata: govecdb.Metadata{"source": "handbook.pdf", "page": int64(i), "score": 0.5 + float64(i)},
		}
		ref[i] = refMatch{ID: ms[i].ID, Distance: ms[i].Distance, Metadata: refMetadata(ms[i].Metadata)}
	}
	return ms, ref
}

func benchVectors(n, dim int) []govecdb.Vector {
	r := rand.New(rand.NewPCG(1, 1))
	vs := make([]govecdb.Vector, n)
	for i := range vs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = r.Float32()*2 - 1
		}
		vs[i] = govecdb.Vector{ID: fmt.Sprintf("v%06d", i), Values: v,
			Metadata: govecdb.Metadata{"source": "a.pdf", "page": int64(i)}}
	}
	return vs
}

func BenchmarkEncodeSearchK100(b *testing.B) {
	ms, ref := benchMatches(100)
	b.Run("encoding-json", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			json.Marshal(map[string]any{"matches": ref})
		}
	})
	b.Run("hand-written", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			bp := bufPool.Get().(*[]byte)
			out, _ := encodeMatches((*bp)[:0], ms)
			*bp = out[:0]
			bufPool.Put(bp)
		}
	})
}

func BenchmarkEncodePage1000x512(b *testing.B) {
	vs := benchVectors(1000, 512)
	b.Run("encoding-json", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			json.Marshal(map[string]any{"vectors": refVectors(vs)})
		}
	})
	b.Run("hand-written", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			encodePage(nil, vs, "", false)
		}
	})
}

func benchDecode[T any](b *testing.B, body []byte) {
	b.ReportAllocs()
	b.SetBytes(int64(len(body)))
	for b.Loop() {
		var dst T
		dec := json.NewDecoder(bytes.NewReader(body))
		dec.DisallowUnknownFields()
		dec.UseNumber()
		if err := dec.Decode(&dst); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeSearch512(b *testing.B) {
	q := benchVectors(1, 512)[0].Values
	body, _ := json.Marshal(map[string]any{"query": q, "k": 10})
	type oldRequest struct {
		Query []float32 `json:"query"`
		K     int       `json:"k"`
	}
	b.Run("encoding-json", func(b *testing.B) { benchDecode[oldRequest](b, body) })
	b.Run("float32s", func(b *testing.B) { benchDecode[searchRequest](b, body) })
}

func BenchmarkDecodeAdd1000x512(b *testing.B) {
	vs := benchVectors(1000, 512)
	in := make([]map[string]any, len(vs))
	for i, v := range vs {
		in[i] = map[string]any{"id": v.ID, "values": v.Values, "metadata": map[string]any(v.Metadata)}
	}
	body, _ := json.Marshal(map[string]any{"vectors": in})
	type oldVector struct {
		ID       string         `json:"id"`
		Values   []float32      `json:"values"`
		Metadata map[string]any `json:"metadata,omitempty"`
	}
	type oldRequest struct {
		Vectors []oldVector `json:"vectors"`
	}
	b.Run("encoding-json", func(b *testing.B) { benchDecode[oldRequest](b, body) })
	b.Run("float32s", func(b *testing.B) { benchDecode[addRequest](b, body) })
}

// keep maps imported for refMetadata below.
var _ = maps.Keys[map[string]int]

// refMetadata is the encoder the hand-written one replaced, kept as its
// reference: encoding/json plus the ".0" rule. It is metadata on its way back to
// a client, encoded so that scalar's rule reads it back as the type it was
// stored as.
//
// # Why not just let encoding/json do it
//
// The rule on the way in is syntactic: a number with a decimal point or an
// exponent is a float64, one without is an int64. encoding/json writes the
// float64 1.0 as `1`, which that rule then reads back as an int64 — so a client
// that round-trips a record changed the type of every integral float in it, and
// had to keep its own list of which keys were floats to undo that. Appending
// ".0" to an integral float closes the loop without inventing a typed wire
// format: every JSON parser still reads `1.0` as the number one, and this one
// reads it as the float it was.
type refMetadata govecdb.Metadata

func (m refMetadata) MarshalJSON() ([]byte, error) {
	if m == nil {
		return []byte("null"), nil
	}
	// Sorted, as encoding/json sorts map keys, so a response is reproducible.
	keys := slices.Sorted(maps.Keys(m))
	b := append(make([]byte, 0, 32*len(keys)), '{')
	for i, k := range keys {
		if i > 0 {
			b = append(b, ',')
		}
		kb, err := json.Marshal(k)
		if err != nil {
			return nil, err
		}
		b = append(append(b, kb...), ':')

		f, isFloat := m[k].(float64)
		switch {
		case isFloat && (math.IsNaN(f) || math.IsInf(f, 0)):
			// Refused on the way in now, but one stored before that check can
			// still be on disk, and JSON has no spelling for it. null is the
			// honest answer; failing the whole response over one value of one
			// record would make the record unreadable through this API at all.
			b = append(b, "null"...)
		case isFloat:
			fb, err := json.Marshal(f)
			if err != nil {
				return nil, err
			}
			b = append(b, fb...)
			if !bytes.ContainsAny(fb, ".eE") {
				b = append(b, ".0"...)
			}
		default:
			vb, err := json.Marshal(m[k])
			if err != nil {
				return nil, err
			}
			b = append(b, vb...)
		}
	}
	return append(b, '}'), nil
}
