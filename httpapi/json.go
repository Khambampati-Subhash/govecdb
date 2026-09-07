package httpapi

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"net/http"

	"github.com/khambampati-subhash/govecdb"
)

// decode reads a JSON request body into dst.
//
// # Strict on purpose
//
// Unknown fields are rejected. The usual argument for ignoring them is forward
// compatibility, and it does not survive contact with this API: a request that
// says "dimensions" instead of "dimension" would otherwise create a collection
// at the wrong width, succeed, and be discovered weeks later with data in it.
// Every field this package accepts is one a client meant to set.
//
// Numbers are decoded with UseNumber so that metadata and filter operands can
// keep the distinction JSON makes and Go's default decoder throws away — see
// scalar.
func decode(w http.ResponseWriter, r *http.Request, maxBody int64, dst any) error {
	if err := checkContentType(r); err != nil {
		return err
	}

	// MaxBytesReader rather than a check on Content-Length: the header is a claim
	// by the client, and a chunked body does not carry one at all.
	r.Body = http.MaxBytesReader(w, r.Body, maxBody)

	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	dec.UseNumber()

	if err := dec.Decode(dst); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return fmt.Errorf("%w: over %d bytes", errPayloadTooLarge, maxBody)
		}
		if errors.Is(err, io.EOF) {
			return fmt.Errorf("%w: the request body is empty", govecdb.ErrInvalidRequest)
		}
		return fmt.Errorf("%w: %s", govecdb.ErrInvalidRequest, err)
	}

	// A second JSON document after the first is not "extra whitespace"; it is a
	// client sending something this endpoint did not agree to read.
	if dec.More() {
		return fmt.Errorf("%w: unexpected data after the JSON body", govecdb.ErrInvalidRequest)
	}
	return nil
}

// checkContentType accepts application/json and nothing else, allowing the
// header to be absent because a body-less request has no type to declare.
//
// It is worth having beyond tidiness: refusing anything a browser can send from
// a form — text/plain, the urlencoded types — is what keeps a cross-site request
// from reaching a write endpoint without a preflight the browser will not skip.
func checkContentType(r *http.Request) error {
	ct := r.Header.Get("Content-Type")
	if ct == "" {
		return nil
	}
	mt, _, err := mime.ParseMediaType(ct)
	if err != nil {
		return fmt.Errorf("%w: %s", errUnsupportedMediaType, err)
	}
	if mt != "application/json" {
		return fmt.Errorf("%w: got %q", errUnsupportedMediaType, mt)
	}
	return nil
}

// scalar converts a decoded JSON value into one of the four types metadata is
// allowed to hold.
//
// # How a number is decided
//
// JSON has one number type and this database has two, so the rule is the syntax:
// a number written without a decimal point or exponent becomes an int64, and
// anything else becomes a float64. So {"page": 10} stores an integer and
// {"score": 10.0} stores a float, which is what each was written to mean.
//
// Getting this wrong in either direction is quiet rather than loud, which is why
// it is a rule and not a heuristic. Storing everything as float64 would make a
// nanosecond timestamp — about 1.7e18, well past the 2^53 where float64 stops
// counting — arrive rounded. Storing everything as int64 would turn a score of
// 0.5 into 0. The database's own comparisons already cross the two exactly, at
// any magnitude, so a filter written as {"gte": 10} still matches a stored 10.0.
func scalar(v any) (any, error) {
	switch t := v.(type) {
	case string:
		return t, nil
	case bool:
		return t, nil
	case json.Number:
		if i, err := t.Int64(); err == nil {
			return i, nil
		}
		f, err := t.Float64()
		if err != nil {
			return nil, fmt.Errorf("%w: %s is not a number this database can hold", govecdb.ErrInvalidMetadata, t)
		}
		// JSON cannot spell NaN or Infinity, so this is unreachable through a
		// conforming parser. It is here because the database rests on values being
		// finite — one NaN compares false against everything — and a check that
		// costs nothing is worth more than an argument about reachability.
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, fmt.Errorf("%w: %v is not finite", govecdb.ErrInvalidMetadata, f)
		}
		return f, nil
	case nil:
		return nil, fmt.Errorf("%w: null is not a value; omit the key instead", govecdb.ErrInvalidMetadata)
	default:
		return nil, fmt.Errorf("%w: %T is not a string, bool or number", govecdb.ErrInvalidMetadata, v)
	}
}

// metadata converts a decoded JSON object into the database's Metadata.
//
// Nested objects and arrays are refused rather than flattened or encoded. The
// closed set of value types is what lets the on-disk decoder be a total function
// over four tags, and a layer that quietly widened it here would be handing the
// storage format a shape it cannot round-trip.
func metadata(in map[string]any) (govecdb.Metadata, error) {
	if len(in) == 0 {
		return nil, nil
	}
	out := make(govecdb.Metadata, len(in))
	for k, v := range in {
		s, err := scalar(v)
		if err != nil {
			return nil, fmt.Errorf("metadata %q: %w", k, err)
		}
		out[k] = s
	}
	return out, nil
}
