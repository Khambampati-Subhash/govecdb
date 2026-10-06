package govecdb

import (
	"errors"
	"fmt"
)

// Sentinel errors callers match with errors.Is. Every one of them wraps a
// message naming the field and the value that caused it, so matching gives you
// the category and printing gives you the detail.
var (
	// ErrClosed is returned by every operation on a closed database.
	ErrClosed = errors.New("govecdb: database is closed")

	// ErrReadOnly is returned by writes after the write-ahead log has failed, and
	// by every write to a database opened WithReadOnly. The wrapped message says
	// which.
	//
	// It is the deliberate half of failing closed. Once a write could not be
	// made durable, continuing to accept writes would produce a log with a hole
	// in it — and replay stops at the hole, silently discarding everything
	// after. Reads keep working, because the in-memory index is still correct;
	// what is lost is the ability to promise anything about the next write.
	//
	// It is not recoverable in place. Fix the disk, then reopen.
	ErrReadOnly = errors.New("govecdb: database is read-only")

	// ErrAlreadyOpen means the directory is already open, in this process or —
	// on Unix, through an flock — in another one.
	//
	// A log directory and a snapshot directory each belong to one writer: two
	// would interleave segment numbering and delete each other's temporary
	// files, and two processes doing it corrupt the database with nothing
	// refusing either. Read-only opens share the directory with each other but
	// not with a writer.
	ErrAlreadyOpen = errors.New("govecdb: directory is already open")

	// ErrInvalidConfig means Open was given options it cannot build from.
	ErrInvalidConfig = errors.New("govecdb: invalid configuration")

	// ErrInvalidVector means a vector was rejected: an empty or oversized id,
	// the wrong dimension, or a value that is not a finite number.
	ErrInvalidVector = errors.New("govecdb: invalid vector")

	// ErrInvalidRequest means a search request was rejected — a non-positive K,
	// or one of the limits exceeded.
	ErrInvalidRequest = errors.New("govecdb: invalid request")

	// ErrInvalidMetadata means metadata was rejected: an unsupported value type,
	// an empty or non-UTF-8 key, or a size over one of the caps.
	ErrInvalidMetadata = errors.New("govecdb: invalid metadata")

	// ErrInvalidFilter means a search filter could not be run: a comparison
	// operand that is not one of the metadata value types, or a nil filter passed
	// to And, Or or Not.
	//
	// It surfaces at Search rather than where the filter was built, because the
	// constructors return a Filter and not (Filter, error) — a query assembled by
	// nesting calls reads well only if the error is collected and reported once.
	ErrInvalidFilter = errors.New("govecdb: invalid filter")

	// ErrNotFound is returned when an id is not in the database.
	ErrNotFound = errors.New("govecdb: id not found")

	// ErrCorrupt means data on disk could not be decoded. The wrapped error says
	// which layer refused it and why.
	ErrCorrupt = errors.New("govecdb: corrupt data on disk")
)

// errOpenedReadOnly is what a write to a WithReadOnly database returns. It is
// ErrReadOnly to errors.Is, so a caller handles both causes with one check.
var errOpenedReadOnly = fmt.Errorf("%w: opened with WithReadOnly", ErrReadOnly)
