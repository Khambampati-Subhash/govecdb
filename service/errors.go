package service

import "errors"

// Sentinel errors, matched with errors.Is. Each is wrapped with the name that
// caused it, so matching gives the category and printing gives the detail — the
// same contract the root package's errors keep.
var (
	// ErrInvalidName means a collection name is empty, too long, or contains
	// something [ValidateName] refuses. It is a caller error, not a lookup miss:
	// a name that could never exist is worth distinguishing from one that simply
	// does not.
	ErrInvalidName = errors.New("service: invalid collection name")

	// ErrExists means a collection of that name is already there. Create never
	// adopts an existing directory — a spec that disagreed with the index inside
	// it would fail at open, far from the call that caused it.
	ErrExists = errors.New("service: collection already exists")

	// ErrNotFound means no collection of that name exists in the root directory.
	ErrNotFound = errors.New("service: collection not found")

	// ErrClosed is returned by every operation on a closed Manager.
	ErrClosed = errors.New("service: manager is closed")

	// ErrTooManyOpen means MaxOpen collections are loaded and every one of them
	// is in use, so there is nothing to evict to make room.
	//
	// It is deliberately not a queue. Waiting for a slot would turn a capacity
	// problem into a latency problem that shows up as timeouts somewhere else,
	// and the caller — which is usually a request handler — has a better answer
	// available: say so, now, with a status a client can act on.
	ErrTooManyOpen = errors.New("service: too many collections open")

	// ErrInvalidSpec means a collection's configuration was rejected. Open
	// returns it for a spec file that does not describe a database this build can
	// construct; Create returns it before anything reaches the disk.
	ErrInvalidSpec = errors.New("service: invalid collection spec")

	// ErrCorruptSpec means a spec file exists but could not be read or decoded.
	//
	// Kept apart from ErrInvalidSpec because the two have different fixes: an
	// invalid spec is a request to correct, a corrupt one is a disk to
	// investigate. Reporting both as "bad config" would send an operator looking
	// in the wrong place.
	ErrCorruptSpec = errors.New("service: corrupt collection spec")
)
