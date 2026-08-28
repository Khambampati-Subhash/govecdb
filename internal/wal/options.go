package wal

import "time"

// SyncPolicy decides when the log is pushed to stable storage. It is the single
// knob that trades durability against throughput, and it is a knob rather than a
// constant because the right answer genuinely differs: a test wants speed, a
// primary datastore wants safety, and most services want the middle.
type SyncPolicy int

const (
	// SyncAlways fsyncs before every Append returns. A write that has been
	// acknowledged has survived power loss, full stop. Costs roughly 1-10 ms per
	// write on consumer NVMe, which is the price of that sentence being true.
	//
	// It is the zero value on purpose: a caller who configures nothing gets the
	// safe behaviour, not the fast one. Silent data loss should never be the
	// default anyone falls into by omission.
	SyncAlways SyncPolicy = iota

	// SyncInterval fsyncs on a timer, so writes are batched into group commits.
	// A power loss can lose up to one interval of acknowledged writes. That is a
	// documented product decision, not an accident — Redis ships the same trade.
	SyncInterval

	// SyncNever leaves flushing to the operating system. The process crashing is
	// survivable, since the data has reached the OS; the machine losing power is
	// not. For benchmarks and tests, and it should stay there.
	SyncNever
)

func (p SyncPolicy) String() string {
	switch p {
	case SyncAlways:
		return "always"
	case SyncInterval:
		return "interval"
	case SyncNever:
		return "never"
	default:
		return "unknown"
	}
}

// Options configures a Writer. The zero value is usable and safe: SyncAlways,
// with the size limits below filled in by their defaults.
type Options struct {
	// SyncPolicy decides when data reaches stable storage.
	SyncPolicy SyncPolicy

	// SyncInterval is how often to fsync under SyncInterval. Default 50ms.
	//
	// Worth keeping separate in your head from the checkpoint interval: this one
	// bounds how much acknowledged data a power loss destroys, which is
	// milliseconds of work. A checkpoint interval bounds recovery time and log
	// size on disk, which is minutes. 50ms is a sensible fsync interval and a
	// wildly wrong checkpoint interval.
	SyncInterval time.Duration

	// MaxSegmentBytes is the size at which the writer rotates to a new segment.
	// Default 64 MiB.
	//
	// It sets the granularity of truncation: after a checkpoint, whole segments
	// below it are deleted, so smaller segments reclaim space sooner and larger
	// ones mean fewer files. A single record larger than this still gets written
	// — a segment may exceed the limit rather than a write being impossible.
	MaxSegmentBytes int64

	// MaxRecordBytes caps one payload. Default 16 MiB.
	//
	// This is a corruption guard as much as a limit. On the way back in, a record
	// announces its own length, and a flipped bit there could ask for gigabytes;
	// the reader refuses anything above this cap. That only works if the writer
	// enforces the same bound, or a legitimate record would be unreadable.
	MaxRecordBytes int

	// FirstSeq is the sequence number the next Append will use. Default 1.
	//
	// Recovery sets it to one past the highest sequence it replayed, so numbering
	// continues across restarts rather than colliding.
	FirstSeq uint64
}

// Default sizes. Chosen so a segment holds a useful amount of work without the
// directory filling with files, and so one record can be large without being
// able to swallow a whole segment.
const (
	defaultSyncInterval    = 50 * time.Millisecond
	defaultMaxSegmentBytes = 64 << 20 // 64 MiB
	defaultMaxRecordBytes  = 16 << 20 // 16 MiB
	defaultFirstSeq        = 1
)

// withDefaults fills in the unset fields. Options is passed by value, so this
// cannot surprise a caller by mutating the struct they kept.
func (o Options) withDefaults() Options {
	if o.SyncInterval <= 0 {
		o.SyncInterval = defaultSyncInterval
	}
	if o.MaxSegmentBytes <= 0 {
		o.MaxSegmentBytes = defaultMaxSegmentBytes
	}
	if o.MaxRecordBytes <= 0 {
		o.MaxRecordBytes = defaultMaxRecordBytes
	}
	if o.FirstSeq == 0 {
		o.FirstSeq = defaultFirstSeq
	}
	return o
}
