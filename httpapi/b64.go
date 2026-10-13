package httpapi

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/url"
	"slices"

	"github.com/khambampati-subhash/govecdb"
)

// Binary vectors: a float32 vector as base64 of its little-endian bytes.
//
// # Why
//
// Decoding a 1,000 × 512 add batch from JSON numbers took 42 ms — on the order
// of applying it to the index — because every component is a decimal string to
// parse and round. The same vector in base64 is a third smaller and decoding it
// is a copy, not a parse. JSON numbers stay the default, unchanged.
//
// # The format
//
// Standard base64 (RFC 4648 §4, padded), of 4 × dimension bytes, each float32
// little-endian — what numpy's `astype('<f4').tobytes()` and Go's
// `binary.LittleEndian` produce. Padded, and not the URL alphabet: one
// spelling, so a client that gets it wrong is told rather than half-understood.
// The values are validated exactly as JSON numbers are, after decoding — a NaN
// costs nothing to spell in binary, and validate.go refuses it the same way.

// b64Float32s is a []float32 spelled as a base64 string on the wire.
type b64Float32s []float32

var errB64Float32s = errors.New("want base64 of little-endian float32s")

func (f *b64Float32s) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if string(b) == "null" {
		return nil // as a slice is left on null, matching float32s
	}
	if len(b) < 2 || b[0] != '"' || b[len(b)-1] != '"' {
		return fmt.Errorf("%w, got %.20s", errB64Float32s, b)
	}
	s := b[1 : len(b)-1]
	// The base64 alphabet needs no JSON escape, but '/' may legally be written
	// "\/" and some encoders do. Unescaping is the decoder's job; the fast path
	// is only for the string it would have produced unchanged.
	if bytes.IndexByte(s, '\\') >= 0 {
		var str string
		if err := json.Unmarshal(b, &str); err != nil {
			return err
		}
		s = []byte(str)
	}
	v, err := decodeB64Float32s(s)
	if err != nil {
		return err
	}
	*f = v
	return nil
}

// decodeB64Float32s decodes base64 of little-endian float32s.
func decodeB64Float32s(s []byte) ([]float32, error) {
	enc := base64.StdEncoding
	// DecodedLen is an upper bound; the exact length is known only after
	// decoding, and it must be whole float32s.
	buf := make([]byte, enc.DecodedLen(len(s)))
	n, err := enc.Decode(buf, s)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errB64Float32s, err)
	}
	if n%4 != 0 {
		return nil, fmt.Errorf("%w: %d bytes is not a whole number of float32s", errB64Float32s, n)
	}
	out := make([]float32, n/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
	}
	return out, nil
}

// appendB64Float32s appends vs as a JSON string of base64 of its
// little-endian bytes; nil is null, as appendFloat32s writes it.
//
// Encoded through a fixed chunk on the stack rather than a copy of the whole
// vector: base64 of a whole number of 3-byte groups ends without padding, so
// chunks of a multiple of 3 bytes concatenate into exactly the encoding of the
// whole. 768 bytes is 192 float32s and 256 groups.
func appendB64Float32s(dst []byte, vs []float32) []byte {
	if vs == nil {
		return append(dst, "null"...)
	}
	var chunk [768]byte
	dst = slices.Grow(dst, base64.StdEncoding.EncodedLen(4*len(vs))+2)
	dst = append(dst, '"')
	for len(vs) > 0 {
		n := min(len(vs), len(chunk)/4)
		for i, v := range vs[:n] {
			binary.LittleEndian.PutUint32(chunk[4*i:], math.Float32bits(v))
		}
		dst = base64.StdEncoding.AppendEncode(dst, chunk[:4*n])
		vs = vs[n:]
	}
	return append(dst, '"')
}

// oneOf returns whichever spelling of a vector a request used. Both at once is
// refused rather than one preferred: two spellings that can disagree have no
// right answer, and picking one would hide the client bug that sent both.
// Neither is left to the database, which refuses a vector of the wrong
// dimension — zero included — with the same error it always has.
func oneOf(field string, nums float32s, b64 b64Float32s) ([]float32, error) {
	if nums != nil && b64 != nil {
		return nil, fmt.Errorf("%w: set %s or %s_b64, not both", govecdb.ErrInvalidRequest, field, field)
	}
	if b64 != nil {
		return b64, nil
	}
	return nums, nil
}

// vectorFormat is how a response spells the values of the records in it.
type vectorFormat uint8

const (
	formatNumbers vectorFormat = iota // "values": [0.25, ...], the default
	formatB64                         // "values_b64": "AACAPg..."
)

// parseVectorFormat reads ?encoding=. An unknown encoding is refused rather
// than ignored, for the reason decode refuses unknown fields: a client that
// asked for something else and got JSON numbers would find out by parsing them
// as base64.
func parseVectorFormat(q url.Values) (vectorFormat, error) {
	switch e := q.Get("encoding"); e {
	case "", "json":
		return formatNumbers, nil
	case "b64":
		return formatB64, nil
	default:
		return 0, fmt.Errorf("%w: encoding %q, want json or b64", govecdb.ErrInvalidRequest, e)
	}
}
