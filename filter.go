package govecdb

import "github.com/khambampati-subhash/govecdb/internal/filter"

// Filter restricts a search to vectors whose metadata matches.
//
// Build one with the constructors below and set it on SearchRequest.Filter:
//
//	db.Search(govecdb.SearchRequest{
//	    Query:  q,
//	    K:      10,
//	    Filter: govecdb.And(
//	        govecdb.Eq("source", "handbook.pdf"),
//	        govecdb.Gte("page", 10),
//	        govecdb.Not(govecdb.Exists("retracted")),
//	    ),
//	})
//
// # It is applied during the search, not to its results
//
// A filter narrows what the traversal will *accept*, not what it returns: a
// rejected vector still routes the search toward its neighbors, it just never
// becomes an answer. That is what makes a filtered search return K results
// rather than "however many of the nearest K happened to match", which is what
// filtering the result slice afterwards would give.
//
// The cost is worth knowing. A filter that matches a small share of the corpus
// makes the search explore further to find K of them, so latency rises as
// selectivity falls. At the extreme — a filter matching a handful of vectors out
// of millions — scanning is the better tool, and this index is the wrong one to
// ask.
//
// # Absent keys
//
// Every comparison is false when the key is not present, Ne included: Ne("k", x)
// means "k is there and is not x". Not(Eq("k", x)) is how to also match vectors
// that have no k at all.
//
// # The interface is open
//
// Implementing Filter is supported — a predicate this package has no constructor
// for is a legitimate thing to want. Match is called once per candidate node,
// under the index's read lock, and is handed a map it must not retain or mutate;
// Validate is called once per search, before any of that, and should return nil
// when there is nothing to report.
type Filter interface {
	// Match reports whether a vector's metadata satisfies the filter. The map is
	// nil when the vector carries no metadata.
	Match(md Metadata) bool

	// Validate reports a construction error, or nil. Search calls it once and
	// fails with ErrInvalidFilter if it returns anything.
	Validate() error
}

// Comparison values may be any Go string, bool, floating-point or integer type.
// Integers are converted to int64 and float32 to float64, both losslessly, so
// Eq("page", 12) does the obvious thing despite an untyped constant being an int
// rather than the int64 metadata actually holds. A uint64 above MaxInt64, or any
// other type, is reported by Validate as ErrInvalidFilter.
//
// Numbers compare across int64 and float64 exactly, at any magnitude — including
// past 2^53, where converting an integer to a float would start rounding it.

// Eq matches vectors whose key is present and equal to v.
func Eq(key string, v any) Filter { return filter.Eq(key, v) }

// Ne matches vectors whose key is present and not equal to v. A vector without
// the key does not match; use Not(Eq(...)) if it should.
func Ne(key string, v any) Filter { return filter.Ne(key, v) }

// Lt matches vectors whose key is present and less than v.
//
// Strings order lexicographically by byte and numbers numerically. Bools have no
// ordering and never match, and neither does a comparison between two different
// kinds — a string against a number, say.
func Lt(key string, v any) Filter { return filter.Lt(key, v) }

// Lte matches vectors whose key is present and less than or equal to v.
func Lte(key string, v any) Filter { return filter.Lte(key, v) }

// Gt matches vectors whose key is present and greater than v.
func Gt(key string, v any) Filter { return filter.Gt(key, v) }

// Gte matches vectors whose key is present and greater than or equal to v.
func Gte(key string, v any) Filter { return filter.Gte(key, v) }

// In matches vectors whose key is present and equal to any of vs.
//
// In(key) with no values matches nothing, which is the direction that fails
// safe: filtering by a set that turned out to be empty returns nothing rather
// than everything.
func In(key string, vs ...any) Filter { return filter.In(key, vs...) }

// Exists matches vectors that carry the key at all, whatever its value.
func Exists(key string) Filter { return filter.Exists(key) }

// And matches vectors that every filter matches.
//
// And() with no filters matches everything and Or() with none matches nothing —
// the identities for the two, and what makes a filter accumulated in a loop
// behave when the loop runs zero times.
func And(fs ...Filter) Filter { return filter.And(internalFilters(fs)...) }

// Or matches vectors that any filter matches.
func Or(fs ...Filter) Filter { return filter.Or(internalFilters(fs)...) }

// Not inverts a filter. It is also how to match vectors that lack a key
// entirely, since every direct comparison is false on an absent key.
func Not(f Filter) Filter {
	if f == nil {
		// Passed through rather than short-circuited, so the nil is reported by
		// Validate as ErrInvalidFilter instead of panicking at search time.
		return filter.Not(nil)
	}
	return filter.Not(f)
}

// internalFilters retypes a slice of this package's Filter as the internal one.
//
// The two interfaces have identical method sets, so each element converts
// implicitly; it is only the *slices* that Go will not assign across named
// types. The loop is the whole cost of keeping Filter declared here rather than
// aliased to an internal type a caller could not usefully name.
//
// A nil element is passed through for Validate to report, for the same reason
// Not does.
func internalFilters(fs []Filter) []filter.Filter {
	out := make([]filter.Filter, len(fs))
	for i, f := range fs {
		if f == nil {
			continue
		}
		out[i] = f
	}
	return out
}
