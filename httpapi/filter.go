package httpapi

import (
	"fmt"

	"github.com/khambampati-subhash/govecdb"
)

// maxFilterDepth caps how deeply a filter may nest.
//
// A filter is recursion driven by a request body, which is the shape of problem
// that ends in a stack overflow — and a stack overflow in Go is not a recovered
// panic, it is the process. Thirty-two is far past any query a person writes and
// far short of anything that hurts.
const maxFilterDepth = 32

// filterJSON is the wire form of a metadata filter.
//
// # Why the format lives here
//
// internal/filter deliberately has no wire format, on the grounds that nothing
// consumed one. Something does now, and it is this layer rather than that one:
// a serialization format is a compatibility promise, and the package that makes
// the promise should be the package a client can see. The filter package stays
// free to change its representation.
//
// # The shape
//
//	{"op": "and", "filters": [
//	  {"op": "eq",  "key": "source", "value": "handbook.pdf"},
//	  {"op": "gte", "key": "page",   "value": 10},
//	  {"op": "not", "filter": {"op": "exists", "key": "retracted"}}
//	]}
//
// One tagged object rather than the shorter {"eq": {...}} spelling, because a
// single "op" field is unambiguous to read, to validate and to produce from a
// client library that does not have Go's type switch.
type filterJSON struct {
	Op      string       `json:"op"`
	Key     string       `json:"key,omitempty"`
	Value   any          `json:"value,omitempty"`
	Values  []any        `json:"values,omitempty"`
	Filters []filterJSON `json:"filters,omitempty"`
	Filter  *filterJSON  `json:"filter,omitempty"`
}

// build turns the decoded form into a database filter.
//
// Errors are wrapped in govecdb.ErrInvalidFilter so that a filter refused here
// and a filter refused by the database's own Validate reach the client as the
// same code. The two failures are the same thing to whoever wrote the query.
func (f *filterJSON) build(depth int) (govecdb.Filter, error) {
	if f == nil {
		return nil, fmt.Errorf("%w: a filter is null", govecdb.ErrInvalidFilter)
	}
	if depth > maxFilterDepth {
		return nil, fmt.Errorf("%w: nested deeper than %d", govecdb.ErrInvalidFilter, maxFilterDepth)
	}

	switch f.Op {
	case "eq", "ne", "lt", "lte", "gt", "gte":
		v, err := f.comparison()
		if err != nil {
			return nil, err
		}
		switch f.Op {
		case "eq":
			return govecdb.Eq(f.Key, v), nil
		case "ne":
			return govecdb.Ne(f.Key, v), nil
		case "lt":
			return govecdb.Lt(f.Key, v), nil
		case "lte":
			return govecdb.Lte(f.Key, v), nil
		case "gt":
			return govecdb.Gt(f.Key, v), nil
		default:
			return govecdb.Gte(f.Key, v), nil
		}

	case "in":
		if err := f.requireKey(); err != nil {
			return nil, err
		}
		vs := make([]any, len(f.Values))
		for i, raw := range f.Values {
			v, err := scalar(raw)
			if err != nil {
				return nil, fmt.Errorf("%w: in[%d]: %w", govecdb.ErrInvalidFilter, i, err)
			}
			vs[i] = v
		}
		// An empty list is allowed and matches nothing, which is what the
		// database's In does and the direction that fails safe: a set that turned
		// out to be empty returns nothing rather than everything.
		return govecdb.In(f.Key, vs...), nil

	case "exists":
		if err := f.requireKey(); err != nil {
			return nil, err
		}
		return govecdb.Exists(f.Key), nil

	case "and", "or":
		subs := make([]govecdb.Filter, len(f.Filters))
		for i := range f.Filters {
			sub, err := f.Filters[i].build(depth + 1)
			if err != nil {
				return nil, err
			}
			subs[i] = sub
		}
		if f.Op == "and" {
			return govecdb.And(subs...), nil
		}
		return govecdb.Or(subs...), nil

	case "not":
		sub, err := f.Filter.build(depth + 1)
		if err != nil {
			return nil, err
		}
		return govecdb.Not(sub), nil

	case "":
		return nil, fmt.Errorf("%w: a filter has no op", govecdb.ErrInvalidFilter)
	default:
		return nil, fmt.Errorf("%w: unknown op %q", govecdb.ErrInvalidFilter, f.Op)
	}
}

func (f *filterJSON) requireKey() error {
	if f.Key == "" {
		return fmt.Errorf("%w: %q needs a key", govecdb.ErrInvalidFilter, f.Op)
	}
	return nil
}

// comparison validates the key and operand shared by the six ordered operators.
func (f *filterJSON) comparison() (any, error) {
	if err := f.requireKey(); err != nil {
		return nil, err
	}
	if f.Value == nil {
		return nil, fmt.Errorf("%w: %q on %q has no value", govecdb.ErrInvalidFilter, f.Op, f.Key)
	}
	v, err := scalar(f.Value)
	if err != nil {
		return nil, fmt.Errorf("%w: %q on %q: %w", govecdb.ErrInvalidFilter, f.Op, f.Key, err)
	}
	return v, nil
}
