package filter

import (
	"fmt"

	"github.com/khambampati-subhash/govecdb/internal/store"
)

// Filter reports whether a vector's metadata matches.
//
// Validate is on the interface rather than done at construction because the
// constructors return a Filter and not (Filter, error): a query built by chaining
// calls reads well, and a filter that has to be checked for an error at every
// nesting level does not. The error is carried by the node that found it and
// reported once, by the layer that is about to run a search and has somewhere to
// report it to.
type Filter interface {
	// Match reports whether md satisfies the filter. It may panic on a filter
	// Validate would reject; callers validate first.
	Match(md store.Metadata) bool

	// Validate reports the first construction error in this filter or its
	// children.
	Validate() error
}

// And matches when every filter matches.
//
// And() with no filters matches everything, which is the identity for
// conjunction and the answer that makes a filter built up in a loop behave: an
// empty set of conditions constrains nothing. Or() is the mirror image and
// matches nothing, for the same reason.
func And(fs ...Filter) Filter { return and(fs) }

// Or matches when any filter matches. Or() with no filters matches nothing.
func Or(fs ...Filter) Filter { return or(fs) }

// Not inverts a filter.
//
// It is also the only way to express "the key is absent, or does not match",
// since every direct predicate is false on a key that is not there.
func Not(f Filter) Filter { return not{f} }

// Exists matches when the key is present, whatever its value.
func Exists(key string) Filter { return exists(key) }

type and []Filter

func (a and) Match(md store.Metadata) bool {
	for _, f := range a {
		if !f.Match(md) {
			return false
		}
	}
	return true
}

func (a and) Validate() error { return validateAll(a) }

type or []Filter

func (o or) Match(md store.Metadata) bool {
	for _, f := range o {
		if f.Match(md) {
			return true
		}
	}
	return false
}

func (o or) Validate() error { return validateAll(o) }

type not struct{ f Filter }

func (n not) Match(md store.Metadata) bool { return !n.f.Match(md) }

func (n not) Validate() error {
	if n.f == nil {
		return fmt.Errorf("%w: Not", ErrNilFilter)
	}
	return n.f.Validate()
}

type exists string

func (e exists) Match(md store.Metadata) bool {
	_, ok := md[string(e)]
	return ok
}

// Validate has nothing to check: a key is a string and any string is a legal
// key to ask about, including one no vector has.
func (e exists) Validate() error { return nil }

// validateAll reports the first problem among children, naming the position so a
// deeply nested tree points at the clause that is wrong rather than at itself.
func validateAll(fs []Filter) error {
	for i, f := range fs {
		if f == nil {
			return fmt.Errorf("%w: at index %d", ErrNilFilter, i)
		}
		if err := f.Validate(); err != nil {
			return fmt.Errorf("index %d: %w", i, err)
		}
	}
	return nil
}
