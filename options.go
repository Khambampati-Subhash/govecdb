package govecdb

import (
	"fmt"
	"time"

	"github.com/khambampati-subhash/govecdb/internal/hnsw"
	"github.com/khambampati-subhash/govecdb/internal/wal"
)

// Metric selects how distance between two vectors is measured.
//
// Re-declared here rather than aliased from the index package so the public API
// does not depend on an internal type. That is not ceremony: internal packages
// cannot be named from outside the module, so an alias would produce a public
// surface callers can use but not write down.
type Metric int

const (
	// Cosine distance, 1 - cosine similarity. Vectors are unit-normalized on
	// insert, so only direction is stored and magnitude is discarded. The right
	// default for text embeddings.
	Cosine Metric = iota

	// Euclidean distance, returned squared. Monotonic with true distance, so
	// ranking is identical and the square root is skipped.
	Euclidean

	// DotProduct, negated so that larger dot products are smaller distances.
	// Vectors are stored as given.
	DotProduct
)

func (m Metric) String() string {
	switch m {
	case Cosine:
		return "cosine"
	case Euclidean:
		return "euclidean"
	case DotProduct:
		return "dotproduct"
	default:
		return "unknown"
	}
}

func (m Metric) valid() bool {
	return m == Cosine || m == Euclidean || m == DotProduct
}

// internal maps a public metric onto the index's own. The switch is exhaustive
// over valid() so an unhandled value cannot silently become Cosine.
func (m Metric) internal() hnsw.Metric {
	switch m {
	case Euclidean:
		return hnsw.Euclidean
	case DotProduct:
		return hnsw.DotProduct
	default:
		return hnsw.Cosine
	}
}

// SyncPolicy decides when writes reach stable storage. It is the single knob
// that trades durability for throughput.
type SyncPolicy int

const (
	// SyncAlways fsyncs before every write returns: an acknowledged write has
	// survived power loss, full stop. Costs about 4 ms per write.
	//
	// It is the zero value on purpose. A caller who configures nothing gets the
	// safe answer, not the fast one — silent data loss should never be the
	// default anybody falls into by omission.
	SyncAlways SyncPolicy = iota

	// SyncInterval fsyncs on a timer, batching writes into group commits. Up to
	// one interval of acknowledged writes is lost to power loss *and* to a
	// process crash: between ticks those records are still in this process's
	// buffer rather than the kernel's.
	SyncInterval

	// SyncNever leaves flushing to the operating system. Writes sit in a 64 KiB
	// buffer until it fills or Close is called, so a process crash loses up to a
	// buffer's worth and power loss loses whatever the OS had not written back.
	// For tests and bulk loads that will be re-run on failure.
	SyncNever
)

// String names the policy, so a layer that has to write one down — a config
// file, a log line, an API response — does not invent its own spelling. Metric
// carries one for the same reason.
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

func (p SyncPolicy) valid() bool {
	return p == SyncAlways || p == SyncInterval || p == SyncNever
}

func (p SyncPolicy) internal() wal.SyncPolicy {
	switch p {
	case SyncInterval:
		return wal.SyncInterval
	case SyncNever:
		return wal.SyncNever
	default:
		return wal.SyncAlways
	}
}

// Options is the full configuration of a database. It is built by Open from the
// functional options, and is not part of the public surface — callers set fields
// through With* so that adding one later is not a breaking change and so every
// value passes through validation in one place.
type options struct {
	dimension int
	metric    Metric

	m              int
	efConstruction int
	seed           int64

	syncPolicy   SyncPolicy
	syncInterval time.Duration

	maxSegmentBytes int64
	snapshotEvery   time.Duration
	snapshotsKept   int

	targetRecall float64

	maxIDBytes  int
	maxK        int
	maxEf       int
	maxBatch    int
	maxMetadata int
}

// Defaults. The limits exist to bound what a single call can make the process
// allocate; they are far above any legitimate use and are not a statement about
// what the index can handle.
const (
	defaultM              = 16
	defaultEfConstruction = 200
	defaultTargetRecall   = 0.95
	defaultSyncInterval   = 50 * time.Millisecond
	defaultSnapshotsKept  = 2
	defaultMaxIDBytes     = 512
	defaultMaxK           = 10_000
	defaultMaxEf          = 100_000
	defaultMaxBatch       = 10_000
	defaultMaxMetadata    = 256
)

func defaultOptions() options {
	return options{
		metric:         Cosine,
		m:              defaultM,
		efConstruction: defaultEfConstruction,
		seed:           1,
		syncPolicy:     SyncAlways,
		syncInterval:   defaultSyncInterval,
		snapshotsKept:  defaultSnapshotsKept,
		targetRecall:   defaultTargetRecall,
		maxIDBytes:     defaultMaxIDBytes,
		maxK:           defaultMaxK,
		maxEf:          defaultMaxEf,
		maxBatch:       defaultMaxBatch,
		maxMetadata:    defaultMaxMetadata,
	}
}

// Option configures a database at Open time.
//
// Functional options rather than an exported config struct: a struct makes every
// added field a potential breaking change for anyone using positional literals,
// and gives no place to put per-field validation. These carry their own errors,
// so a bad value is reported by Open with the field named.
type Option func(*options) error

// WithDimension sets the vector dimension. Required — there is no sensible
// default, and guessing one from the first vector would make the second vector's
// rejection look arbitrary.
func WithDimension(d int) Option {
	return func(o *options) error {
		if d <= 0 || d > maxDimension {
			return fmt.Errorf("%w: dimension %d, want 1..%d", ErrInvalidConfig, d, maxDimension)
		}
		o.dimension = d
		return nil
	}
}

// WithMetric sets the distance metric. Defaults to Cosine.
//
// It is structural: an index built for one metric cannot answer queries under
// another, and reopening a database with a different metric is refused.
func WithMetric(m Metric) Option {
	return func(o *options) error {
		if !m.valid() {
			return fmt.Errorf("%w: unknown metric %d", ErrInvalidConfig, m)
		}
		o.metric = m
		return nil
	}
}

// WithM sets the index's neighbors-per-node. Defaults to 16.
//
// Structural — changing it requires rebuilding — and worth less than it looks:
// compared at equal recall, M=32 beats M=16 by about 10% latency for six times
// the build time and twice the memory. 16 is the right answer for almost
// everybody.
//
// Must be at least 2. M=1 is not a thin graph but an undefined one: the level
// distribution is scaled by 1/ln(M), which is a division by zero.
func WithM(m int) Option {
	return func(o *options) error {
		if m < 2 || m > maxM {
			return fmt.Errorf("%w: M %d, want 2..%d", ErrInvalidConfig, m, maxM)
		}
		o.m = m
		return nil
	}
}

// WithEfConstruction sets the build-time search width. Defaults to 200.
// Higher means a better graph and slower inserts; 100-200 is the useful band.
func WithEfConstruction(ef int) Option {
	return func(o *options) error {
		if ef <= 0 || ef > maxEfCeiling {
			return fmt.Errorf("%w: EfConstruction %d, want 1..%d", ErrInvalidConfig, ef, maxEfCeiling)
		}
		o.efConstruction = ef
		return nil
	}
}

// WithSeed makes index construction reproducible. Defaults to 1.
func WithSeed(seed int64) Option {
	return func(o *options) error {
		o.seed = seed
		return nil
	}
}

// WithSyncPolicy chooses when writes reach stable storage. Defaults to
// SyncAlways — see SyncPolicy for what each one costs and what it promises.
func WithSyncPolicy(p SyncPolicy) Option {
	return func(o *options) error {
		if !p.valid() {
			return fmt.Errorf("%w: unknown sync policy %d", ErrInvalidConfig, p)
		}
		o.syncPolicy = p
		return nil
	}
}

// WithSyncInterval sets how often SyncInterval fsyncs. Defaults to 50ms.
//
// Worth keeping separate in your head from WithSnapshotInterval: this one bounds
// how much acknowledged data a crash destroys, in milliseconds. That one bounds
// recovery time and disk usage, in minutes.
func WithSyncInterval(d time.Duration) Option {
	return func(o *options) error {
		if d <= 0 {
			return fmt.Errorf("%w: sync interval %v must be positive", ErrInvalidConfig, d)
		}
		o.syncInterval = d
		return nil
	}
}

// WithMaxSegmentBytes sets the log's segment size. Defaults to 64 MiB.
//
// It is a truncation granularity, not a cap on what can be written: a record
// larger than a segment still gets written, into a segment of its own.
func WithMaxSegmentBytes(n int64) Option {
	return func(o *options) error {
		if n <= 0 {
			return fmt.Errorf("%w: segment size %d must be positive", ErrInvalidConfig, n)
		}
		o.maxSegmentBytes = n
		return nil
	}
}

// WithSnapshotInterval takes a snapshot automatically on a timer. Off by default.
//
// A snapshot is what bounds recovery time: without one, starting up replays the
// whole log and rebuilds the index, which costs about 700 µs per vector. With
// one, it loads a graph instead. Minutes is the right order of magnitude — every
// snapshot costs a ~10 ms fsync floor plus the time to write the index out, and
// it takes a read lock for the duration, so writers wait.
func WithSnapshotInterval(d time.Duration) Option {
	return func(o *options) error {
		if d <= 0 {
			return fmt.Errorf("%w: snapshot interval %v must be positive", ErrInvalidConfig, d)
		}
		o.snapshotEvery = d
		return nil
	}
}

// WithSnapshotsKept sets how many snapshots to retain. Defaults to 2.
//
// More than one on purpose. One snapshot is one copy, and a copy that fails its
// checksum is no copy at all; a second means a corrupt newest snapshot costs a
// longer log replay rather than the database. Setting it to 1 is supported and
// should be a deliberate choice.
func WithSnapshotsKept(n int) Option {
	return func(o *options) error {
		if n < 1 {
			return fmt.Errorf("%w: must keep at least one snapshot, got %d", ErrInvalidConfig, n)
		}
		o.snapshotsKept = n
		return nil
	}
}

// WithSearchTargetRecall sets the recall a search aims for when SearchRequest.Ef
// is left zero. Defaults to 0.95, treated as a floor.
func WithSearchTargetRecall(r float64) Option {
	return func(o *options) error {
		if r <= 0 || r >= 1 {
			return fmt.Errorf("%w: target recall %v, want 0 < r < 1", ErrInvalidConfig, r)
		}
		o.targetRecall = r
		return nil
	}
}

// WithLimits overrides the per-call bounds: the longest id, the largest K and
// Ef a search may ask for, the largest batch, and the most metadata keys.
//
// These are not tuning knobs. They are what stops one malformed call from
// deciding how much this process allocates — an id length and a K both come
// straight from a caller who may be relaying somebody else's input. Raise them
// deliberately; a zero leaves that limit at its default.
func WithLimits(maxIDBytes, maxK, maxEf, maxBatch, maxMetadataKeys int) Option {
	return func(o *options) error {
		for _, f := range []struct {
			name string
			v    *int
			in   int
			max  int
		}{
			{"max id bytes", &o.maxIDBytes, maxIDBytes, maxIDLimit},
			{"max k", &o.maxK, maxK, maxKLimit},
			{"max ef", &o.maxEf, maxEf, maxEfCeiling},
			{"max batch", &o.maxBatch, maxBatch, maxBatchLimit},
			{"max metadata keys", &o.maxMetadata, maxMetadataKeys, maxMetadataLimit},
		} {
			if f.in == 0 {
				continue
			}
			if f.in < 0 || f.in > f.max {
				return fmt.Errorf("%w: %s %d, want 1..%d", ErrInvalidConfig, f.name, f.in, f.max)
			}
			*f.v = f.in
		}
		return nil
	}
}

// validate checks the combination, after every individual option has checked
// itself. Dimension is the only field with no usable default.
func (o *options) validate() error {
	if o.dimension == 0 {
		return fmt.Errorf("%w: dimension is required, set WithDimension", ErrInvalidConfig)
	}
	if o.maxEf < o.maxK {
		return fmt.Errorf("%w: max ef %d is below max k %d, which would refuse a legal search",
			ErrInvalidConfig, o.maxEf, o.maxK)
	}
	return nil
}
