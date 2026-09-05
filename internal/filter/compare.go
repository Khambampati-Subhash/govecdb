package filter

import (
	"fmt"

	"github.com/khambampati-subhash/govecdb/internal/store"
)

// op is which comparison a compare node performs.
//
// One node type with an operator, rather than six types, because the six differ
// only in the last line: they share the absent-key rule, the normalization, and
// the decision about what is comparable with what. Splitting them would mean six
// copies of that agreement, which is six chances for one of them to drift.
type op uint8

const (
	opEq op = iota
	opNe
	opLt
	opLte
	opGt
	opGte
)

func (o op) String() string {
	switch o {
	case opEq:
		return "Eq"
	case opNe:
		return "Ne"
	case opLt:
		return "Lt"
	case opLte:
		return "Lte"
	case opGt:
		return "Gt"
	default:
		return "Gte"
	}
}

// Eq matches when the key is present and its value equals v.
//
// Numbers compare across int64 and float64, so this matches whichever of the two
// the value happens to have been stored as.
func Eq(key string, v any) Filter { return newCompare(key, v, opEq) }

// Ne matches when the key is present and its value differs from v.
//
// Present is the part worth reading twice: a vector with no such key does not
// match, because every predicate here is false on a key that is not there. Use
// Not(Eq(...)) for "absent, or different".
func Ne(key string, v any) Filter { return newCompare(key, v, opNe) }

// Lt matches when the key is present and its value is less than v.
//
// Ordering is defined for strings (lexicographic, by byte) and for numbers. A
// bool has no ordering and never matches; neither does a comparison between two
// different kinds, such as a string against a number.
func Lt(key string, v any) Filter { return newCompare(key, v, opLt) }

// Lte matches when the key is present and its value is less than or equal to v.
func Lte(key string, v any) Filter { return newCompare(key, v, opLte) }

// Gt matches when the key is present and its value is greater than v.
func Gt(key string, v any) Filter { return newCompare(key, v, opGt) }

// Gte matches when the key is present and its value is greater than or equal to v.
func Gte(key string, v any) Filter { return newCompare(key, v, opGte) }

type compare struct {
	key string
	val any
	op  op
	err error
}

func newCompare(key string, v any, o op) Filter {
	n, err := normalize(v)
	if err != nil {
		return &compare{key: key, op: o, err: fmt.Errorf("%s(%q): %w", o, key, err)}
	}
	return &compare{key: key, val: n, op: o}
}

func (c *compare) Match(md store.Metadata) bool {
	stored, ok := md[c.key]
	if !ok {
		return false
	}

	switch c.op {
	case opEq:
		return equal(stored, c.val)
	case opNe:
		// Deliberately not "not equal or absent": absence was already ruled out
		// above, so this is "present and different". A value of another kind
		// counts as different, which is the honest answer — a string is not the
		// number it is being compared against.
		return !equal(stored, c.val)
	}

	cmp, ok := order(stored, c.val)
	if !ok {
		return false
	}
	switch c.op {
	case opLt:
		return cmp < 0
	case opLte:
		return cmp <= 0
	case opGt:
		return cmp > 0
	default:
		return cmp >= 0
	}
}

func (c *compare) Validate() error { return c.err }

// In matches when the key is present and its value equals any of vs.
//
// In() with no values matches nothing, which is the identity for a disjunction
// and what makes filtering by an empty set of ids return nothing rather than
// everything — the direction that fails safe.
func In(key string, vs ...any) Filter {
	f := &in{key: key, vals: make([]any, 0, len(vs))}
	for i, v := range vs {
		n, err := normalize(v)
		if err != nil {
			f.err = fmt.Errorf("In(%q) value %d: %w", key, i, err)
			return f
		}
		f.vals = append(f.vals, n)
	}
	return f
}

type in struct {
	key  string
	vals []any
	err  error
}

func (i *in) Match(md store.Metadata) bool {
	stored, ok := md[i.key]
	if !ok {
		return false
	}
	// Linear, because the value set is a handful of literals in every use this
	// package was built for, and a map keyed by `any` would hash every candidate
	// value to save comparisons that cost a type switch each. If a caller ever
	// passes thousands, that is the point to measure rather than to guess.
	for _, v := range i.vals {
		if equal(stored, v) {
			return true
		}
	}
	return false
}

func (i *in) Validate() error { return i.err }
