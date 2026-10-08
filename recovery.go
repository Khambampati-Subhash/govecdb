package govecdb

import (
	"fmt"
	"io"
	"path/filepath"
	"time"

	"github.com/khambampati-subhash/govecdb/internal/snapshot"
	"github.com/khambampati-subhash/govecdb/internal/store"
	"github.com/khambampati-subhash/govecdb/internal/wal"
)

// Recovery: rebuilding the in-memory state from what is on disk.
//
// The order is snapshot then log, and it is the only order that works. A
// snapshot is the state as of some sequence N; the log holds every record ever
// written, including the ones already inside that snapshot. So the snapshot goes
// down first and the log is replayed on top, skipping anything at or below N.
//
// Skipping rather than re-applying is not just an optimisation. Re-applying a
// PUT whose vector is unchanged is an early return in the index and costs
// nothing, but re-applying one whose *metadata* changed and then changed back
// would be work for no reason, and re-applying a PUT that replaced a vector
// creates a tombstone every single time the database starts. A restart should
// not make the index worse.

// nextSeq is where the log resumes. It lives on DB because Open needs it between
// restore and opening the writer.
func (db *DB) restore() error {
	start := time.Now()
	snapDir := filepath.Join(db.dir, snapshotSubdir)
	walDir := filepath.Join(db.dir, walSubdir)

	if err := db.loadSnapshot(snapDir); err != nil {
		return err
	}
	replayed, segments, err := db.replayLog(walDir)
	if err != nil {
		return err
	}
	db.emit(Recovered{
		SnapshotSeq: db.snapSeq.Load(),
		Replayed:    replayed,
		Segments:    segments,
		Took:        time.Since(start),
	})
	return nil
}

// loadSnapshot installs the newest usable snapshot, or an empty database if
// there is none.
//
// A corrupt newest snapshot is not fatal: internal/snapshot falls back to an
// older one, and if every one of them fails the result is simply no snapshot and
// a longer replay. Refusing to start over a bad *cache* would be the wrong call
// — the log is the source of truth — as long as the log is still all there.
// Once truncation has run it is not, and replayLog refuses then (see
// checkLogReachesBack).
func (db *DB) loadSnapshot(dir string) error {
	var (
		idx *hnswIndex
		st  *store.Map
	)
	res, err := snapshot.Load(dir, func(r io.Reader) error {
		var err error
		// The hard ceiling, not the configured limit, as for log records
		// (decodeRecord): lowering WithLimits must not make old data corrupt.
		idx, st, err = readSnapshot(r, maxIDLimit)
		return err
	})
	// Reported before the error check: a rejection followed by an I/O error on
	// the next file is two facts, and the second should not hide the first.
	for _, r := range res.Rejected {
		db.emit(SnapshotRejected{Path: r.Snapshot.Path, Seq: r.Snapshot.Seq, Cause: r.Cause})
	}
	if err != nil {
		return fmt.Errorf("govecdb: load snapshot: %w", err)
	}

	if !res.Found {
		fresh, err := newHNSWIndex(db.opts)
		if err != nil {
			return fmt.Errorf("govecdb: create index: %w", err)
		}
		db.index, db.store = fresh, store.New()
		db.snapSeq.Store(0)
		db.nextSeq = 1
		return nil
	}

	// The snapshot carries the config the index was built with, and the caller
	// has passed their own. A disagreement on anything structural means the
	// graph on disk was built under rules this process is not using — its edges
	// were chosen for a different metric, or a different M — and searching it
	// would return quietly wrong answers rather than failing.
	if err := db.checkConfig(idx); err != nil {
		return err
	}

	db.index, db.store = idx, st
	db.snapSeq.Store(res.Snapshot.Seq)
	db.nextSeq = res.Snapshot.Seq + 1
	return nil
}

// checkConfig refuses a snapshot whose structural settings differ from the ones
// this process was opened with.
//
// Only the structural ones. EfConstruction and the seed affect how a graph gets
// built but not how an existing one is read, so changing them between runs is
// legitimate — new inserts simply use the new value.
func (db *DB) checkConfig(idx *hnswIndex) error {
	dim, metric, m := idx.config()
	switch {
	case dim != db.opts.dimension:
		return fmt.Errorf("%w: snapshot has dimension %d, opened with %d",
			ErrInvalidConfig, dim, db.opts.dimension)
	case metric != db.opts.metric:
		return fmt.Errorf("%w: snapshot uses metric %s, opened with %s",
			ErrInvalidConfig, metric, db.opts.metric)
	case m != db.opts.m:
		return fmt.Errorf("%w: snapshot was built with M=%d, opened with M=%d",
			ErrInvalidConfig, m, db.opts.m)
	}
	return nil
}

// replayLog applies every record written after the snapshot, and reports how
// many it applied and how many segments it read.
//
// Consecutive PUTs are gathered and applied as one batch, which an index that
// batches links in parallel: replay without a snapshot is a full index build,
// and serially that was ~23 minutes for 250,000 vectors of dimension 512. A
// DELETE flushes what is pending first, so every operation still lands in log
// order and the state is the one record-by-record replay would leave.
func (db *DB) replayLog(dir string) (applied, segments int, err error) {
	r := replayer{db: db}
	snapSeq := db.snapSeq.Load()
	first := true
	res, err := wal.Replay(dir, wal.Options{}, func(rec wal.Record) error {
		if first {
			first = false
			if err := checkLogReachesBack(rec.Seq, snapSeq); err != nil {
				return err
			}
		}
		if rec.Seq <= snapSeq {
			return nil // already inside the snapshot
		}
		applied++
		return r.apply(rec)
	})
	if err == nil {
		err = r.flush()
	}
	for _, t := range res.Tears {
		db.emit(TornLog{Segment: t.File(), Offset: t.Offset, Discarded: t.Discarded, Cause: t.Cause})
	}
	if err != nil {
		return applied, res.Segments, fmt.Errorf("govecdb: replay log: %w", err)
	}

	// The log can be *behind* the snapshot — a snapshot taken and the log
	// truncated, or a log directory removed — so the resume point is whichever
	// is further along. Resuming below a sequence a snapshot already covers
	// would hand out numbers a second time, and replay refuses a log whose
	// sequences do not strictly increase.
	db.nextSeq = max(db.nextSeq, res.NextSeq())

	// Tears are the expected shape after a crash, not a reason to refuse to
	// start: a torn tail is a write that was never acknowledged. They are
	// reported as events above rather than returned.
	return applied, res.Segments, nil
}

// checkLogReachesBack refuses a log whose first record leaves a gap after the
// snapshot that was loaded — snapSeq 0 when none was.
//
// # Why this is an error and not a slower start
//
// Truncation deletes the segments a retained snapshot stands in for. If every
// snapshot is then unusable — rotted, or a snapshots directory deleted as "just
// a cache" — the log no longer reaches back to the start, and replaying what is
// left onto an empty index opens a database with most of its data missing and
// nothing saying so. Measured: 551 of 2,000 vectors, opened writable, and the
// next snapshot made the loss permanent by pruning the damaged files and
// truncating the rest of the log. A database that cannot be opened is
// recoverable — restore the snapshots and reopen; one that opens with its data
// silently gone is not.
//
// It cannot misfire on a healthy database. A log never truncated starts at 1.
// Truncation keeps every record above the oldest retained snapshot, and the
// snapshot loaded is at least that new, so the first record is at most one
// past it. A crash tears the end of a segment, never the start of the log —
// rot in the very first record would trip it, and should, because those
// records are gone. And a log *behind* its snapshot — a crash under SyncNever — starts
// at or below it, which passes.
func checkLogReachesBack(first, snapSeq uint64) error {
	if first <= snapSeq+1 {
		return nil
	}
	if snapSeq == 0 {
		return fmt.Errorf("%w: the log starts at seq %d and no usable snapshot covers seqs 1-%d: "+
			"they were truncated behind snapshots that are now missing or corrupt; "+
			"restore the snapshots directory to open this database",
			ErrCorrupt, first, first-1)
	}
	return fmt.Errorf("%w: the log starts at seq %d but the newest usable snapshot covers only through seq %d: "+
		"seqs %d-%d are missing; restore the newer snapshots to open this database",
		ErrCorrupt, first, snapSeq, snapSeq+1, first-1)
}

// replayBatch is how many consecutive PUTs replay gathers before applying
// them: enough to keep every worker busy across several chunks, and at
// dimension 512 about 8 MiB of vectors held at once.
const replayBatch = 4096

// replayer applies log records, holding back runs of PUTs to apply together.
// Decoded records own their memory, so holding them past the callback that
// produced them is safe; the raw payload is never kept.
type replayer struct {
	db      *DB
	pending []Vector
	lastSeq uint64
}

// flush applies the PUTs held back so far.
func (r *replayer) flush() error {
	if len(r.pending) == 0 {
		return nil
	}
	err := r.db.applyPuts(r.pending)
	if err != nil {
		err = fmt.Errorf("PUTs through seq %d: %w", r.lastSeq, err)
	}
	clear(r.pending) // drop the references, so applied vectors are not pinned
	r.pending = r.pending[:0]
	return err
}

// apply replays one record into the in-memory state.
//
// It does not write to the log. These records are already in it, and appending
// them again would grow the log by its own contents on every start.
func (r *replayer) apply(rec wal.Record) error {
	db := r.db
	switch rec.Type {
	case wal.TypePut:
		v, isPut, err := decodeRecord(rec.Type, rec.Payload, &db.opts)
		if err != nil {
			return fmt.Errorf("seq %d: %w", rec.Seq, err)
		}
		if !isPut {
			return fmt.Errorf("%w: seq %d decoded as a delete under a PUT", ErrCorrupt, rec.Seq)
		}
		r.pending = append(r.pending, v)
		r.lastSeq = rec.Seq
		if len(r.pending) >= replayBatch {
			return r.flush()
		}
		return nil

	case wal.TypeDelete:
		// Everything before this record lands before it does.
		if err := r.flush(); err != nil {
			return err
		}
		v, _, err := decodeRecord(rec.Type, rec.Payload, &db.opts)
		if err != nil {
			return fmt.Errorf("seq %d: %w", rec.Seq, err)
		}
		db.applyDelete(v.ID)
		return nil

	case wal.TypeCheckpoint:
		// Reserved in the log's format and not written by this version. Ignored
		// rather than refused, so a log written by a build that does emit them
		// still replays here.
		return nil

	default:
		return fmt.Errorf("%w: seq %d has record type %d", ErrCorrupt, rec.Seq, rec.Type)
	}
}
