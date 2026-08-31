package govecdb

import "errors"

// Sentinel errors callers match with errors.Is. Every one of them wraps a
// message naming the field and the value that caused it, so matching gives you
// the category and printing gives you the detail.
var (
	// ErrClosed is returned by every operation on a closed database.
	ErrClosed = errors.New("govecdb: database is closed")

	// ErrReadOnly is returned by writes after the write-ahead log has failed.
	//
	// It is the deliberate half of failing closed. Once a write could not be
	// made durable, continuing to accept writes would produce a log with a hole
	// in it — and replay stops at the hole, silently discarding everything
	// after. Reads keep working, because the in-memory index is still correct;
	// what is lost is the ability to promise anything about the next write.
	//
	// It is not recoverable in place. Fix the disk, then reopen.
	ErrReadOnly = errors.New("govecdb: database is read-only after a durability failure")

	// ErrAlreadyOpen means this process already has the directory open.
	//
	// A log directory and a snapshot directory each belong to one writer: two
	// would interleave segment numbering and delete each other's temporary
	// files. This catches the case worth catching — the same process opening the
	// same path twice — and does not attempt to police other processes.
	ErrAlreadyOpen = errors.New("govecdb: directory is already open in this process")

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

	// ErrNotFound is returned when an id is not in the database.
	ErrNotFound = errors.New("govecdb: id not found")

	// ErrCorrupt means data on disk could not be decoded. The wrapped error says
	// which layer refused it and why.
	ErrCorrupt = errors.New("govecdb: corrupt data on disk")
)
