package httpapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"slices"
	"strconv"
	"sync"
	"unicode/utf8"

	"github.com/khambampati-subhash/govecdb"
)

// The hot responses — search matches and vector records — are written by hand.
//
// # Why
//
// Measured, encoding a k=100 search response with three metadata keys through
// encoding/json was 78 µs and 1,707 allocations, for a search that itself took
// about 66 µs: the HTTP layer cost more than the database. Almost all of it was
// the old metadata Marshaler calling json.Marshal per key and per value, and then
// encoding/json re-validating and compacting every Marshaler's output. Writing
// the whole document with appends removes both.
//
// # What it must not change
//
// The bytes. A client cannot tell this encoder from encoding/json: strings are
// escaped exactly as json.Marshal escapes them (including <, >, & and
// U+2028/U+2029, and invalid UTF-8 as U+FFFD), floats are formatted with its
// algorithm, keys are sorted, and the metadata number rule holds — an integral
// float64 is written `1.0`, an int64 `1`, a NaN `null`. TestEncoderMatches…
// compares the two over random and adversarial input; a difference there is a
// bug here, never a reason to update the test.

// appendString appends s as a JSON string, byte-for-byte as json.Marshal
// writes it (HTML escaping on, which is json.Marshal's default).
func appendString(dst []byte, s string) []byte {
	const hex = "0123456789abcdef"
	dst = append(dst, '"')
	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < utf8.RuneSelf {
			if htmlSafe(b) {
				i++
				continue
			}
			dst = append(dst, s[start:i]...)
			switch b {
			case '\\', '"':
				dst = append(dst, '\\', b)
			case '\b':
				dst = append(dst, '\\', 'b')
			case '\f':
				dst = append(dst, '\\', 'f')
			case '\n':
				dst = append(dst, '\\', 'n')
			case '\r':
				dst = append(dst, '\\', 'r')
			case '\t':
				dst = append(dst, '\\', 't')
			default:
				// Other control bytes, and <, > and &, which a browser could
				// otherwise be talked into reading as markup.
				dst = append(dst, '\\', 'u', '0', '0', hex[b>>4], hex[b&0xF])
			}
			i++
			start = i
			continue
		}
		c, size := utf8.DecodeRuneInString(s[i:])
		if c == utf8.RuneError && size == 1 {
			dst = append(dst, s[start:i]...)
			dst = append(dst, `\ufffd`...)
			i += size
			start = i
			continue
		}
		// Valid JSON unescaped, but not valid JavaScript, which matters to
		// anything that evaluates a response rather than parsing it.
		if c == '\u2028' || c == '\u2029' {
			dst = append(dst, s[start:i]...)
			dst = append(dst, '\\', 'u', '2', '0', '2', hex[c&0xF])
			i += size
			start = i
			continue
		}
		i += size
	}
	dst = append(dst, s[start:]...)
	return append(dst, '"')
}

// htmlSafe is encoding/json's htmlSafeSet: printable ASCII (and DEL) other
// than the two characters JSON escapes and the three HTML ones.
func htmlSafe(b byte) bool {
	return b >= 0x20 && b != '"' && b != '\\' && b != '<' && b != '>' && b != '&'
}

// errUnsupportedFloat is a NaN or infinity where JSON has no spelling. It
// fails the response, as encoding/json would, everywhere except metadata —
// see appendMetadata.
var errUnsupportedFloat = errors.New("httpapi: cannot encode a NaN or infinite number")

// appendFloat appends f as encoding/json formats a float of the given width:
// shortest round-trip digits, plain notation between 1e-6 and 1e21 and an
// exponent outside it, with no zero-padded exponent.
func appendFloat(dst []byte, f float64, bits int) ([]byte, error) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return dst, errUnsupportedFloat
	}
	abs := math.Abs(f)
	format := byte('f')
	// float32 comparisons for a float32: its own cutoffs, not float64's.
	if abs != 0 {
		if bits == 64 && (abs < 1e-6 || abs >= 1e21) || bits == 32 && (float32(abs) < 1e-6 || float32(abs) >= 1e21) {
			format = 'e'
		}
	}
	dst = strconv.AppendFloat(dst, f, format, -1, bits)
	if format == 'e' {
		// e-09 → e-9
		n := len(dst)
		if n >= 4 && dst[n-4] == 'e' && dst[n-3] == '-' && dst[n-2] == '0' {
			dst[n-2] = dst[n-1]
			dst = dst[:n-1]
		}
	}
	return dst, nil
}

// appendFloat32s appends a vector as a JSON array; nil is null, as
// encoding/json writes a nil slice.
func appendFloat32s(dst []byte, vs []float32) ([]byte, error) {
	if vs == nil {
		return append(dst, "null"...), nil
	}
	dst = append(dst, '[')
	for i, v := range vs {
		if i > 0 {
			dst = append(dst, ',')
		}
		var err error
		if dst, err = appendFloat(dst, float64(v), 32); err != nil {
			return dst, err
		}
	}
	return append(dst, ']'), nil
}

// appendMetadata appends metadata as an object with sorted keys, applying the
// number rule that makes it round-trip through scalar: an integral float64
// gains ".0", so it reads back as a float rather than an int64.
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
//
// A NaN or infinite float64 is written null rather than failing the response.
// It is refused on the way in now, but one stored before that check can still
// be on disk, and failing the whole response over one value of one record
// would make the record unreadable through this API at all.
func appendMetadata(dst []byte, md govecdb.Metadata) ([]byte, error) {
	// Sorted, as encoding/json sorts map keys, so a response is reproducible.
	// The array keeps the usual handful of keys off the heap.
	var buf [16]string
	keys := buf[:0]
	for k := range md {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	dst = append(dst, '{')
	for i, k := range keys {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = appendString(dst, k)
		dst = append(dst, ':')
		switch v := md[k].(type) {
		case string:
			dst = appendString(dst, v)
		case int64:
			dst = strconv.AppendInt(dst, v, 10)
		case bool:
			dst = strconv.AppendBool(dst, v)
		case float64:
			if math.IsNaN(v) || math.IsInf(v, 0) {
				dst = append(dst, "null"...)
				continue
			}
			start := len(dst)
			dst, _ = appendFloat(dst, v, 64)
			if !bytes.ContainsAny(dst[start:], ".eE") {
				dst = append(dst, ".0"...)
			}
		default:
			// Not one of the four types metadata may hold, so not reachable
			// through the database's own validation. Encoded rather than
			// dropped, so a bug elsewhere shows up as an odd value instead of
			// a missing one.
			b, err := json.Marshal(v)
			if err != nil {
				return dst, err
			}
			dst = append(dst, b...)
		}
	}
	return append(dst, '}'), nil
}

// appendMatch appends one search result: {"id","distance","metadata"}, the
// metadata omitted when empty, as `omitempty` did.
func appendMatch(dst []byte, m govecdb.Match) ([]byte, error) {
	dst = append(dst, `{"id":`...)
	dst = appendString(dst, m.ID)
	dst = append(dst, `,"distance":`...)
	var err error
	if dst, err = appendFloat(dst, float64(m.Distance), 32); err != nil {
		return dst, err
	}
	if len(m.Metadata) > 0 {
		dst = append(dst, `,"metadata":`...)
		if dst, err = appendMetadata(dst, m.Metadata); err != nil {
			return dst, err
		}
	}
	return append(dst, '}'), nil
}

// appendVector appends one record: {"id","values","metadata"}.
func appendVector(dst []byte, v govecdb.Vector) ([]byte, error) {
	dst = append(dst, `{"id":`...)
	dst = appendString(dst, v.ID)
	dst = append(dst, `,"values":`...)
	var err error
	if dst, err = appendFloat32s(dst, v.Values); err != nil {
		return dst, err
	}
	if len(v.Metadata) > 0 {
		dst = append(dst, `,"metadata":`...)
		if dst, err = appendMetadata(dst, v.Metadata); err != nil {
			return dst, err
		}
	}
	return append(dst, '}'), nil
}

func appendVectors(dst []byte, vs []govecdb.Vector) ([]byte, error) {
	// Sized up front: a page is megabytes, too big for the pool, and growing
	// it by doubling allocated nearly three times its final size.
	dst = slices.Grow(dst, encodedSize(vs))
	dst = append(dst, '[')
	for i, v := range vs {
		if i > 0 {
			dst = append(dst, ',')
		}
		var err error
		if dst, err = appendVector(dst, v); err != nil {
			return dst, err
		}
	}
	return append(dst, ']'), nil
}

// encodedSize estimates vectors' encoded length: a float32 is at most about
// 15 bytes and usually 10-12, and an id and a little metadata fit in 96.
func encodedSize(vs []govecdb.Vector) int {
	n := 2
	for _, v := range vs {
		n += 12*len(v.Values) + len(v.ID) + 96
	}
	return n
}

func appendStrings(dst []byte, ss []string) []byte {
	dst = append(dst, '[')
	for i, s := range ss {
		if i > 0 {
			dst = append(dst, ',')
		}
		dst = appendString(dst, s)
	}
	return append(dst, ']')
}

// The response documents. Keys are in sorted order, which is the order
// encoding/json wrote the map[string]any these replace.

// encodeMatches writes {"matches":[...]}.
func encodeMatches(dst []byte, ms []govecdb.Match) ([]byte, error) {
	dst = append(dst, `{"matches":[`...)
	for i, m := range ms {
		if i > 0 {
			dst = append(dst, ',')
		}
		var err error
		if dst, err = appendMatch(dst, m); err != nil {
			return dst, err
		}
	}
	return append(dst, "]}"...), nil
}

// encodePage writes {"next":...,"vectors":[...]}, next omitted on the last page.
func encodePage(dst []byte, vs []govecdb.Vector, next string, more bool) ([]byte, error) {
	dst = append(dst, '{')
	if more {
		dst = append(dst, `"next":`...)
		dst = appendString(dst, next)
		dst = append(dst, ',')
	}
	dst = append(dst, `"vectors":`...)
	dst, err := appendVectors(dst, vs)
	if err != nil {
		return dst, err
	}
	return append(dst, '}'), nil
}

// encodeBatch writes {"missing":[...],"vectors":[...]}.
func encodeBatch(dst []byte, vs []govecdb.Vector, missing []string) ([]byte, error) {
	dst = append(dst, `{"missing":`...)
	dst = appendStrings(dst, missing)
	dst = append(dst, `,"vectors":`...)
	dst, err := appendVectors(dst, vs)
	if err != nil {
		return dst, err
	}
	return append(dst, '}'), nil
}

// bufPool recycles response buffers. Buffers past maxPooledBuffer are left to
// the collector: one 15 MB page should not pin 15 MB per pool slot forever.
var bufPool = sync.Pool{New: func() any {
	b := make([]byte, 0, 4<<10)
	return &b
}}

const maxPooledBuffer = 1 << 20

// writeEncoded runs enc into a pooled buffer and sends the result. An encoding
// error becomes a 500 before any status is written, the same invariant write
// keeps: a status line is never a lie.
func (s *Server) writeEncoded(w http.ResponseWriter, r *http.Request, status int,
	enc func([]byte) ([]byte, error)) {
	bp := bufPool.Get().(*[]byte)
	b, err := enc((*bp)[:0])
	if err != nil {
		s.log.Error("encoding response", "path", r.URL.Path, "error", err)
		http.Error(w, `{"error":{"code":"internal","message":"internal error"}}`,
			http.StatusInternalServerError)
	} else {
		s.send(w, r, status, append(b, '\n'))
	}
	if cap(b) <= maxPooledBuffer {
		*bp = b[:0]
		bufPool.Put(bp)
	}
}

// float32s is a []float32 that decodes without reflection.
//
// encoding/json visits every element of a []float32 through reflection, which
// for a 512-dimension query was half the cost of decoding the request. By the
// time UnmarshalJSON runs the decoder has already checked the value is
// well-formed JSON, so splitting on commas and parsing each token is safe: a
// number cannot contain a comma, and anything that is not a number is refused
// at its first byte, before a comma inside a string could matter.
//
// Parsed with ParseFloat(tok, 32), which is what encoding/json calls, so every
// value rounds to the same float32 and an overflow is refused the same way.
// One difference, deliberately stricter: a null element is refused, where
// encoding/json silently left a 0 in its place — a zero component the client
// never sent.
type float32s []float32

func (f *float32s) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if string(b) == "null" {
		return nil // as encoding/json leaves a slice on null
	}
	if len(b) < 2 || b[0] != '[' || b[len(b)-1] != ']' {
		return fmt.Errorf("want an array of numbers, got %.20s", b)
	}
	body := bytes.TrimSpace(b[1 : len(b)-1])
	if len(body) == 0 {
		*f = float32s{}
		return nil
	}
	out := make(float32s, 0, 1+bytes.Count(body, []byte{','}))
	for i := 0; len(body) > 0; i++ {
		tok := body
		if j := bytes.IndexByte(body, ','); j >= 0 {
			tok, body = body[:j], body[j+1:]
		} else {
			body = nil
		}
		tok = bytes.TrimSpace(tok)
		if len(tok) == 0 || (tok[0] != '-' && (tok[0] < '0' || tok[0] > '9')) {
			return fmt.Errorf("element %d: want a number, got %.20s", i, tok)
		}
		v, err := strconv.ParseFloat(string(tok), 32)
		if err != nil {
			return fmt.Errorf("element %d: number %s overflows float32", i, tok)
		}
		out = append(out, float32(v))
	}
	*f = out
	return nil
}
