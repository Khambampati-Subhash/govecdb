package filter

import (
	"fmt"
	"math"
	"strings"
)

// normalize converts a caller's comparison operand into one of the four types a
// metadata value can actually have.
//
// Stored values get no such courtesy — Add refuses an int instead of widening it
// — because their width is a platform property and what one machine writes
// another has to read. An operand is never written down, so the only question is
// whether the conversion loses information, and for every Go integer type into
// int64 it does not. uint64 above MaxInt64 is the single exception and is
// refused rather than wrapped into a negative number.
//
// What this buys is the trap the Metadata doc warns about: Eq("page", 12) makes
// an untyped constant an int, and a strict rule would turn that into a filter
// matching nothing at all, with no error anywhere.
func normalize(v any) (any, error) {
	switch n := v.(type) {
	case string, bool, int64, float64:
		return n, nil
	case int:
		return int64(n), nil
	case int8:
		return int64(n), nil
	case int16:
		return int64(n), nil
	case int32:
		return int64(n), nil
	case uint:
		if uint64(n) > math.MaxInt64 {
			return nil, fmt.Errorf("%w: uint %d exceeds int64", ErrOperand, n)
		}
		return int64(n), nil
	case uint8:
		return int64(n), nil
	case uint16:
		return int64(n), nil
	case uint32:
		return int64(n), nil
	case uint64:
		if n > math.MaxInt64 {
			return nil, fmt.Errorf("%w: uint64 %d exceeds int64", ErrOperand, n)
		}
		return int64(n), nil
	case float32:
		// Widening float32 to float64 is exact, so this comparison is against the
		// number the caller actually holds. It will rarely equal a float64 that
		// was stored from a different computation, but that is float equality
		// being what it is, not a conversion losing anything.
		return float64(n), nil
	default:
		return nil, fmt.Errorf("%w: %T, want string, bool, int64 or float64", ErrOperand, v)
	}
}

// equal reports whether a stored metadata value equals a normalized operand.
//
// Booleans are handled here rather than in order because they have no ordering:
// true is not "greater than" false in any sense a filter should encourage, so
// order refuses them and only equality is offered.
func equal(stored, operand any) bool {
	if b, ok := stored.(bool); ok {
		ob, ok := operand.(bool)
		return ok && b == ob
	}
	c, ok := order(stored, operand)
	return ok && c == 0
}

// order compares a stored value against a normalized operand, reporting the
// usual -1/0/+1 and whether the two are ordered at all.
//
// Not ordered means: different kinds (a string against a number), a bool on
// either side, or a NaN. All three make every comparison false rather than
// picking an arbitrary winner — the same reason the index refuses a NaN vector
// value outright, one comparison that answers false to everything is impossible
// to find afterwards.
func order(stored, operand any) (int, bool) {
	switch s := stored.(type) {
	case string:
		o, ok := operand.(string)
		if !ok {
			return 0, false
		}
		return strings.Compare(s, o), true

	case int64:
		switch o := operand.(type) {
		case int64:
			switch {
			case s < o:
				return -1, true
			case s > o:
				return 1, true
			}
			return 0, true
		case float64:
			return compareIntFloat(s, o)
		}

	case float64:
		switch o := operand.(type) {
		case float64:
			switch {
			case math.IsNaN(s) || math.IsNaN(o):
				return 0, false
			case s < o:
				return -1, true
			case s > o:
				return 1, true
			}
			return 0, true
		case int64:
			// Same comparison with the arguments swapped, so the exact path is
			// written once.
			c, ok := compareIntFloat(o, s)
			return -c, ok
		}
	}
	return 0, false
}

// compareIntFloat compares an int64 against a float64 without losing precision.
//
// The obvious spelling — float64(i) < f — rounds i once its magnitude passes
// 2^53. That range is reached by ordinary data rather than by adversarial data:
// time.Now().UnixNano() is about 1.7e18, so a filter over an ingestion timestamp
// is exactly where the rounding lands, and it fails by silently comparing two
// numbers that are not the ones stored.
//
// Splitting f at its decimal point avoids the conversion entirely. The integer
// part is compared as an int64, and the fraction only decides ties.
func compareIntFloat(i int64, f float64) (int, bool) {
	if math.IsNaN(f) {
		return 0, false
	}

	// Outside int64's range the answer needs no arithmetic — and it has to be
	// taken first, because int64(math.Trunc(f)) below is undefined when f does not
	// fit. The bound is written as 2^63 rather than MaxInt64 because 2^63 is
	// exactly representable in a float64 and MaxInt64 is not: float64(MaxInt64)
	// rounds *up* to 2^63, so comparing against it would misjudge the boundary.
	// This also disposes of both infinities.
	const twoPow63 = 9223372036854775808.0
	if f >= twoPow63 {
		return -1, true
	}
	if f < -twoPow63 {
		return 1, true
	}

	// f is now in [-2^63, 2^63), so its truncation converts to int64 exactly.
	t := math.Trunc(f)
	ti := int64(t)
	if i != ti {
		if i < ti {
			return -1, true
		}
		return 1, true
	}

	// Equal integer parts: the fraction breaks the tie. It is zero only when f is
	// a whole number, which is the one case where the two are genuinely equal.
	// Negative f works without a special case — Trunc rounds toward zero, so
	// -2.5 truncates to -2.0 and the fraction is negative, making the int the
	// larger of the two.
	switch {
	case f > t:
		return -1, true
	case f < t:
		return 1, true
	}
	return 0, true
}
