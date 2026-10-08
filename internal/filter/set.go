package filter

import "math"

// setThreshold is the operand count above which In builds a valueSet instead
// of scanning its operands.
//
// Below it the scan is competitive: a type switch and a compare per operand
// (about 2.8 ns each, measured) against one hash (about 9 ns). At 16 operands
// a miss already costs 45 ns scanned against 9 hashed, so the line is drawn
// low. Above it the scan is the cost of the whole search — In is
// evaluated once per node the traversal visits, so a thousand operands against
// a graph walk is a thousand compares per node, and a hundred thousand of them
// turned one request into seconds of CPU with the collection's read lock held.
const setThreshold = 8

// valueSet answers "does a stored value equal any of these operands" in one
// lookup per stored value, with exactly the semantics of equal.
//
// # How the numeric rule survives hashing
//
// equal crosses int64 and float64: a stored int64 7 equals an operand 7.0, and
// a stored float64 7.0 equals an operand int64 7 — exactly, at any magnitude,
// via compareIntFloat. A map keyed by `any` would not, since int64(7) and
// float64(7) are different keys. So operands are filed by what they can equal:
//
//   - ints holds every int64 operand, and every float64 operand that is a
//     whole number inside int64's range — the floats an int64 can equal. Such
//     a float converts to int64 exactly, so the key is the same number.
//   - floats holds every float64 operand. NaN is dropped: it equals nothing,
//     and a map cannot find it anyway.
//
// A stored int64 is then looked up in ints, and nowhere else: a fractional or
// out-of-range float cannot equal it. A stored float64 is looked up in floats
// (float operands; Go map keys compare with ==, so -0 finds +0), and, when it is
// a whole number in range, in ints — that is where the int64 operands it can
// equal are. A whole-number float operand is in both maps, which costs a
// redundant hit and nothing else.
//
// Strings and bools have no cross-type rule and are filed as themselves.
type valueSet struct {
	strs     map[string]struct{}
	ints     map[int64]struct{}
	floats   map[float64]struct{}
	hasTrue  bool
	hasFalse bool
}

// newValueSet files normalized operands. vals must already be normalized,
// which In guarantees: only string, bool, int64 and float64 reach here.
func newValueSet(vals []any) *valueSet {
	s := &valueSet{}
	for _, v := range vals {
		switch t := v.(type) {
		case string:
			if s.strs == nil {
				s.strs = make(map[string]struct{})
			}
			s.strs[t] = struct{}{}
		case bool:
			if t {
				s.hasTrue = true
			} else {
				s.hasFalse = true
			}
		case int64:
			s.addInt(t)
		case float64:
			if math.IsNaN(t) {
				continue
			}
			if s.floats == nil {
				s.floats = make(map[float64]struct{})
			}
			s.floats[t] = struct{}{}
			if i, ok := wholeInt(t); ok {
				s.addInt(i)
			}
		}
	}
	return s
}

func (s *valueSet) addInt(i int64) {
	if s.ints == nil {
		s.ints = make(map[int64]struct{})
	}
	s.ints[i] = struct{}{}
}

// contains reports whether stored equals any operand. It allocates nothing:
// map lookups with a concrete key do not, and the type switch only unboxes.
func (s *valueSet) contains(stored any) bool {
	switch t := stored.(type) {
	case string:
		_, ok := s.strs[t]
		return ok
	case bool:
		if t {
			return s.hasTrue
		}
		return s.hasFalse
	case int64:
		_, ok := s.ints[t]
		return ok
	case float64:
		if _, ok := s.floats[t]; ok {
			return true
		}
		if i, ok := wholeInt(t); ok {
			_, ok := s.ints[i]
			return ok
		}
	}
	return false
}

// wholeInt reports f as an int64 when f is a whole number inside int64's
// range, which is exactly when some int64 compares equal to it under
// compareIntFloat. The range is [-2^63, 2^63) for the reason compareIntFloat
// gives: 2^63 is representable and MaxInt64 is not. NaN and the infinities
// fail the range check.
func wholeInt(f float64) (int64, bool) {
	const twoPow63 = 9223372036854775808.0
	if !(f >= -twoPow63 && f < twoPow63) || f != math.Trunc(f) {
		return 0, false
	}
	return int64(f), true
}
