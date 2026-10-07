package govecdb

import (
	"fmt"
	"time"
)

// Events: the observability seam.
//
// # What it reports
//
// Two kinds of thing. What the database did on its own — recovered from a crash,
// took a snapshot on a timer, recalibrated in the background — and what it chose
// not to say through a return value, because the choice was right and the caller
// should still hear about it: a torn log tail repaired, a corrupt snapshot passed
// over, a truncation skipped. Before this, every one of those reached a `_`.
//
// # Events, not a Logger and a Metrics interface
//
// A Logger interface would make this package decide what a message says and what
// level it is; a Metrics interface would make it decide metric names and a label
// model. Both are opinions belonging to whoever runs the process. A typed event
// carries the facts, and a consumer can log it, count it, or alert on it with
// whatever it already uses — the daemon does all three from one callback.
//
// Each event has a String for the common case of logging it as it is.
//
// # Not on the hot path
//
// Nothing here fires per Search or per Add. An event is boxed into an interface,
// which allocates, and the search path is held at one allocation by a locked
// baseline. Per-operation latency is measured by timing the call, which the
// caller can do at no cost to anyone else.
//
// # Where they come from
//
// All from this package. The internal packages report what they found as values
// — wal.Replay's tears, snapshot.Load's rejections — and this is the layer that
// turns them into events, so internal/ stays free of any notion of an observer.

// Event is something the database reports to the observer installed with
// WithObserver. The concrete types below are the complete set; a type switch
// should keep a default case, because later versions add to it.
type Event interface {
	fmt.Stringer

	// event seals the set: callers consume events, they do not produce them.
	event()
}

// WithObserver installs fn to receive events. Nil, the default, discards them.
//
// fn is called synchronously, on whichever goroutine hit the event — Open's, a
// writer's, the snapshot timer's or the calibrator's — so it must be safe for
// concurrent use and should return promptly. It may be called while the
// database holds its write lock, and so must not call back into the same DB.
func WithObserver(fn func(Event)) Option {
	return func(o *options) error {
		o.observer = fn
		return nil
	}
}

// emit hands e to the observer, if there is one.
func (db *DB) emit(e Event) {
	if db.opts.observer != nil {
		db.opts.observer(e)
	}
}

// Recovered reports what Open rebuilt the database from. It fires once per
// open, after the snapshot is loaded and the log replayed, and Took is the
// number that answers "why did start-up take twenty minutes": a large Replayed
// and no snapshot is an index rebuilt from the log.
type Recovered struct {
	// SnapshotSeq is the sequence of the snapshot loaded, 0 for none.
	SnapshotSeq uint64

	// Replayed is how many log records were applied on top of it.
	Replayed int

	// Segments is how many log segment files were read.
	Segments int

	// Took is the whole of recovery: snapshot load and replay.
	Took time.Duration
}

// TornLog is a log segment that ended in a damaged record during recovery. The
// segment was read up to Offset and the rest discarded.
//
// One after a crash is the expected shape — it is a write that was never
// acknowledged. A tear on a clean shutdown, or a Discarded running to megabytes
// rather than a few hundred bytes, means something other than power loss
// happened to the disk.
type TornLog struct {
	// Segment is the file name within the log directory.
	Segment string

	// Offset is where the damaged record began.
	Offset int64

	// Discarded is how many bytes were dropped, from Offset to the end of the
	// file.
	Discarded int64

	// Cause is why the record was rejected.
	Cause error
}

// SnapshotRejected is a snapshot that failed verification at Open, so an older
// one — or the log alone — was used instead. Recovery still succeeds; the cost
// is a longer replay. On a healthy machine this should never fire.
type SnapshotRejected struct {
	Path  string
	Seq   uint64
	Cause error
}

// SnapshotTaken is a snapshot written, whether Snapshot was called or the
// interval timer fired.
type SnapshotTaken struct {
	// Seq is the log sequence the snapshot covers through.
	Seq uint64

	// Bytes is the size of the file.
	Bytes int64

	// Took covers writing, pruning and log truncation.
	Took time.Duration

	// Pruned is how many older snapshots retention removed.
	Pruned int

	// SegmentsRemoved is how many log segments truncation deleted.
	SegmentsRemoved int
}

// SnapshotFailed is a snapshot that could not be completed. A caller of
// Snapshot also gets the error; this is how a failure on the interval timer is
// seen at all. It costs a longer replay on the next start and nothing else, so
// the database stays writable.
type SnapshotFailed struct {
	Cause error
}

// TruncationSkipped is a log truncation declined because the oldest retained
// snapshot did not verify. The log keeps growing until a later snapshot
// replaces it — a disk problem, chosen over deleting records only an unreadable
// snapshot could stand in for, which would be a data problem.
type TruncationSkipped struct {
	// Path and Seq name the snapshot that failed.
	Path string
	Seq  uint64

	Cause error
}

// DurabilityFailure is the database going permanently read-only because a write
// could not be made durable. It fires once, on the first failure; every later
// write returns the same ErrReadOnly. Reads keep working.
type DurabilityFailure struct {
	Cause error
}

// Calibrated is a search-width calibration that completed, in the background
// or by Calibrate.
type Calibrated struct {
	// Live is the vector count it measured against.
	Live int

	// Scale is the factor now applied to suggested widths (Stats.EfScale).
	Scale float64

	Took time.Duration
}

// CalibrationFailed is a calibration that did not complete. The previous scale
// stays in force, so searches keep the width they had; repeated failures mean
// the automatic width has stopped tracking the data.
type CalibrationFailed struct {
	Cause error
}

func (Recovered) event()         {}
func (TornLog) event()           {}
func (SnapshotRejected) event()  {}
func (SnapshotTaken) event()     {}
func (SnapshotFailed) event()    {}
func (TruncationSkipped) event() {}
func (DurabilityFailure) event() {}
func (Calibrated) event()        {}
func (CalibrationFailed) event() {}

func (e Recovered) String() string {
	return fmt.Sprintf("recovered from snapshot seq %d and %d log records in %d segments, in %v",
		e.SnapshotSeq, e.Replayed, e.Segments, e.Took)
}

func (e TornLog) String() string {
	return fmt.Sprintf("torn log segment %s: %v at offset %d, %d bytes discarded",
		e.Segment, e.Cause, e.Offset, e.Discarded)
}

func (e SnapshotRejected) String() string {
	return fmt.Sprintf("snapshot %s (seq %d) rejected: %v", e.Path, e.Seq, e.Cause)
}

func (e SnapshotTaken) String() string {
	return fmt.Sprintf("snapshot at seq %d, %d bytes, in %v; pruned %d snapshots and %d log segments",
		e.Seq, e.Bytes, e.Took, e.Pruned, e.SegmentsRemoved)
}

func (e SnapshotFailed) String() string {
	return fmt.Sprintf("snapshot failed: %v", e.Cause)
}

func (e TruncationSkipped) String() string {
	return fmt.Sprintf("log truncation skipped: oldest snapshot %s (seq %d) did not verify: %v",
		e.Path, e.Seq, e.Cause)
}

func (e DurabilityFailure) String() string {
	return fmt.Sprintf("database is read-only: %v", e.Cause)
}

func (e Calibrated) String() string {
	return fmt.Sprintf("calibrated search width at %d vectors: scale %.3g, in %v",
		e.Live, e.Scale, e.Took)
}

func (e CalibrationFailed) String() string {
	return fmt.Sprintf("calibration failed: %v", e.Cause)
}
