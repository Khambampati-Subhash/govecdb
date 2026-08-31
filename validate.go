package govecdb

import (
	"fmt"
	"math"
	"unicode/utf8"

	"github.com/khambampati-subhash/govecdb/internal/store"
)

// Everything a caller hands in is checked here, once, before it reaches the
// index or the log. Two different reasons, and it is worth keeping them apart.
//
// The first is ordinary correctness: a wrong-dimension vector is a caller's bug
// and should be told so at the call rather than at replay.
//
// The second is that a library does not know where its arguments came from. An
// id, a K, a metadata map — any of them may be relaying input from somewhere
// this process does not trust, and each one multiplies an allocation. So the
// ceilings below exist to bound what a *single call* can make the process
// allocate, not to express an opinion about what the index can handle.

// Hard ceilings on the configurable limits. These bound what an option can be
// set to, so a caller cannot disable a bound by raising it to the moon.
const (
	maxDimension     = 1 << 16 // 65,536; the largest embeddings in use are ~16k
	maxM             = 512
	maxEfCeiling     = 1 << 20
	maxIDLimit       = 1 << 16
	maxKLimit        = 1 << 20
	maxBatchLimit    = 1 << 20
	maxMetadataLimit = store.MaxKeys
)

// validateID checks an identifier.
//
// Ids never become file names — segments and snapshots are numbered by the
// database, not named by the caller — so there is no path traversal to defend
// against here. What is left is real all the same: an unbounded id is a memory
// cost repeated in the index, the store, the log and every snapshot, and invalid
// UTF-8 propagates into anything that later formats an id into a log line or a
// JSON document.
func validateID(id string, maxBytes int) error {
	switch {
	case id == "":
		return fmt.Errorf("%w: id is empty", ErrInvalidVector)
	case len(id) > maxBytes:
		return fmt.Errorf("%w: id is %d bytes, max %d", ErrInvalidVector, len(id), maxBytes)
	case !utf8.ValidString(id):
		return fmt.Errorf("%w: id is not valid UTF-8", ErrInvalidVector)
	}
	return nil
}

// validateValues checks an embedding.
//
// The finiteness check is the one that earns its place. A NaN compares false
// against everything, so a single one poisons the ordering the whole index rests
// on: heap invariants stop holding, neighbor selection makes arbitrary choices,
// and searches return wrong answers with no error anywhere. An infinity is
// nearly as bad — it saturates every distance it touches. Both are cheap to
// admit and impossible to find afterwards, so they are refused at the door.
func validateValues(v []float32, dimension int) error {
	if len(v) == 0 {
		return fmt.Errorf("%w: values are empty", ErrInvalidVector)
	}
	if len(v) != dimension {
		return fmt.Errorf("%w: %d values, want %d", ErrInvalidVector, len(v), dimension)
	}
	for i, f := range v {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return fmt.Errorf("%w: value at index %d is %v, want a finite number", ErrInvalidVector, i, f)
		}
	}
	return nil
}

// validateMetadata checks the key/value data, translating the store's errors
// into this package's sentinel so callers match on one thing.
func validateMetadata(md Metadata, maxKeys int) error {
	if len(md) > maxKeys {
		return fmt.Errorf("%w: %d keys, max %d", ErrInvalidMetadata, len(md), maxKeys)
	}
	if err := store.Validate(md); err != nil {
		return fmt.Errorf("%w: %s", ErrInvalidMetadata, err)
	}
	return nil
}

// validateVector checks a whole record.
func (o *options) validateVector(v Vector) error {
	if err := validateID(v.ID, o.maxIDBytes); err != nil {
		return err
	}
	if err := validateValues(v.Values, o.dimension); err != nil {
		return fmt.Errorf("%s: %w", v.ID, err)
	}
	if err := validateMetadata(v.Metadata, o.maxMetadata); err != nil {
		return fmt.Errorf("%s: %w", v.ID, err)
	}
	return nil
}

// validateSearch checks a query and returns the effective search width.
//
// Ef is resolved here rather than inside Search so there is one place that knows
// what a zero means. Zero asks for the width to be chosen from the corpus size,
// which is the right default: recall at a fixed Ef falls as a corpus grows, so a
// constant that works at ten thousand vectors quietly stops working at a million.
func (o *options) validateSearch(req SearchRequest, suggest func(k int, target float64) int) (ef int, err error) {
	if err := validateValues(req.Query, o.dimension); err != nil {
		return 0, fmt.Errorf("query: %w", err)
	}
	if req.K <= 0 {
		return 0, fmt.Errorf("%w: K is %d, want at least 1", ErrInvalidRequest, req.K)
	}
	if req.K > o.maxK {
		return 0, fmt.Errorf("%w: K is %d, max %d", ErrInvalidRequest, req.K, o.maxK)
	}
	if req.Ef < 0 {
		return 0, fmt.Errorf("%w: Ef is %d, want 0 (auto) or positive", ErrInvalidRequest, req.Ef)
	}
	if req.Ef > o.maxEf {
		return 0, fmt.Errorf("%w: Ef is %d, max %d", ErrInvalidRequest, req.Ef, o.maxEf)
	}
	if req.TargetRecall < 0 || req.TargetRecall >= 1 {
		return 0, fmt.Errorf("%w: TargetRecall is %v, want 0 (default) or 0 < r < 1",
			ErrInvalidRequest, req.TargetRecall)
	}

	ef = req.Ef
	if ef == 0 {
		target := req.TargetRecall
		if target == 0 {
			target = o.targetRecall
		}
		ef = suggest(req.K, target)
		// A suggestion is a fitted curve, not a promise about this process's
		// memory. Clamping keeps the auto path inside the same bound an explicit
		// Ef has to satisfy.
		ef = min(ef, o.maxEf)
	}
	return max(ef, req.K), nil
}
