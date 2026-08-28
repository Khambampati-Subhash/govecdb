package wal

// WAL is what the layers above depend on, so the index and the database can be
// built against a log rather than against files. Injecting Nop is what lets the
// benchmarks measure the graph without measuring a disk.
//
// Replay is deliberately absent. The design note lists it as part of this
// interface, and it is not here yet for a reason worth recording: an interface
// method with no implementation is a promise, not a design. Recovery also may
// not belong on the live log at all — replaying is something done *before* a
// writer exists, to rebuild state, which reads more naturally as a package-level
// function over a directory than as a method on the thing currently appending.
// That gets settled when the reader lands, with an implementation in hand.
type WAL interface {
	// Append writes one record and returns the sequence number assigned to it.
	Append(typ RecordType, payload []byte) (uint64, error)

	// Sync forces buffered records to stable storage.
	Sync() error

	// Close flushes, syncs, and releases the log. It is idempotent.
	Close() error
}

// Writer implements WAL. Asserted at compile time so a signature change here
// surfaces at the definition rather than at some distant call site.
var _ WAL = (*Writer)(nil)

// Nop is a WAL that discards everything, for tests and benchmarks that need an
// index without a disk. It still hands out monotonic sequence numbers, so code
// paths that depend on ordering behave the same as they would against a real log
// — a stub that returned zero for every sequence would quietly hide exactly the
// bugs that only appear when ordering matters.
//
// Not safe for concurrent use, unlike Writer. Making it so would mean a mutex on
// the thing whose entire purpose is to cost nothing; tests that need concurrency
// should use a real Writer against a temp dir.
type Nop struct {
	seq uint64
}

var _ WAL = (*Nop)(nil)

func (n *Nop) Append(typ RecordType, payload []byte) (uint64, error) {
	n.seq++
	return n.seq, nil
}

func (n *Nop) Sync() error  { return nil }
func (n *Nop) Close() error { return nil }
