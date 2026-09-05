// Package filter is the metadata query engine: predicates over the key/value
// data attached to a vector, combined into a tree that a search evaluates per
// candidate.
//
// # A tree, not a closure
//
// The whole package could be one type — func(Metadata) bool — and callers would
// write their own predicates. That is smaller and strictly worse. A closure
// cannot be examined, so nothing can report which clause rejected everything, no
// layer can reorder cheap comparisons ahead of expensive ones, and a filter
// arriving from outside the process could not be reconstructed without handing a
// stranger the ability to run code. An expression tree keeps the door open for
// all three and costs one interface.
//
// The interface stays at two methods, though — Match and Validate — because an
// optimizer and a wire format are things this package does not have yet, and the
// WAL taught the lesson already: Replay sat on an interface as a promise until
// writing it showed it did not belong there.
//
// # Absent keys are false, uniformly
//
// Every predicate about a key is false when that key is not there. Eq, Ne, Lt,
// In — all of them. That is worth stating because Ne is the one it surprises:
// "status != draft" does not match a vector with no status at all.
//
// The alternative is SQL's three-valued logic, where a comparison against a
// missing key is neither true nor false and propagates through AND and OR by its
// own rules. It is defensible in a query language people write by hand and badly
// out of place in a Go library, where the result of Match has to be a bool and
// every caller would have to learn a third state to use it. So there is one rule,
// and Not is how a caller asks the other question:
//
//	Ne("status", "draft")        // has a status, and it is not draft
//	Not(Eq("status", "draft"))   // has no status, OR it is not draft
//
// # Numbers compare across int64 and float64, exactly
//
// Metadata holds int64 and float64 as separate types, but a filter should not
// make a caller track which one a value was stored as: Lt("score", 0.5) means
// the same thing whether score was written as an integer or not. So the two
// compare numerically.
//
// Exactly, and not through float64(i) < f. That conversion rounds once i exceeds
// 2^53, which is not a hypothetical range — a Unix timestamp in nanoseconds is
// about 1.7e18, so a filter over an ingestion time is precisely where it would
// break, and it would break by quietly including the wrong records rather than
// by failing. See compareIntFloat.
//
// # Operands are normalized, values are not
//
// Metadata values must be exactly string, bool, int64 or float64; Add refuses an
// int rather than widening it, because the width of int is a platform property
// and a value stored on one machine would be read on another.
//
// A comparison operand is not stored, so that argument does not reach it, and the
// trap it leaves behind is real: Eq("page", 12) makes an untyped constant an int,
// which under a strict rule would match nothing, forever, silently. Every Go
// integer type converts to int64 without loss — uint64 above MaxInt64 excepted,
// and refused rather than wrapped — so operands are normalized on the way in and
// only genuinely unrepresentable ones are rejected.
//
// # Match assumes a validated filter
//
// Validate walks the tree and reports a bad operand or a nil child once, before
// a search runs; Match then evaluates without re-checking, because it runs once
// per candidate node and a search visits thousands. A filter that has not been
// validated may panic in Match rather than return a wrong answer, which is the
// right direction for a bug that is entirely inside this process.
package filter
