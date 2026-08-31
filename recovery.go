package govecdb

import (
	"fmt"
	"io"
	"path/filepath"

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
	snapDir := filepath.Join(db.dir, snapshotSubdir)
	walDir := filepath.Join(db.dir, walSubdir)

	if err := db.loadSnapshot(snapDir); err != nil {
		return err
	}
	return db.replayLog(walDir)
}

// loadSnapshot installs the newest usable snapshot, or an empty database if
// there is none.
//
// A corrupt newest snapshot is not fatal: internal/snapshot falls back to an
// older one, and if every one of them fails the result is simply no snapshot and
// a longer replay. Refusing to start over a bad *cache* would be the wrong call
// — the log is the source of truth and it is still there.
func (db *DB) loadSnapshot(dir string) error {
	var (
		idx *hnswIndex
		st  *store.Map
	)
	res, err := snapshot.Load(dir, func(r io.Reader) error {
		var err error
		idx, st, err = readSnapshot(r, db.opts.maxIDBytes)
		return err
	})
	if err != nil {
		return fmt.Errorf("govecdb: load snapshot: %w", err)
	}

	if !res.Found {
		fresh, err := newHNSWIndex(db.opts)
		if err != nil {
			return fmt.Errorf("govecdb: create index: %w", err)
		}
		db.index, db.store = fresh, store.New()
		db.snapSeq, db.nextSeq = 0, 1
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
	db.snapSeq = res.Snapshot.Seq
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

// replayLog applies every record written after the snapshot.
func (db *DB) replayLog(dir string) error {
	res, err := wal.Replay(dir, wal.Options{}, func(rec wal.Record) error {
		if rec.Seq <= db.snapSeq {
			return nil // already inside the snapshot
		}
		return db.apply(rec)
	})
	if err != nil {
		return fmt.Errorf("govecdb: replay log: %w", err)
	}

	// The log can be *behind* the snapshot — a snapshot taken and the log
	// truncated, or a log directory removed — so the resume point is whichever
	// is further along. Resuming below a sequence a snapshot already covers
	// would hand out numbers a second time, and replay refuses a log whose
	// sequences do not strictly increase.
	db.nextSeq = max(db.nextSeq, res.NextSeq())

	// Tears are the expected shape after a crash, not a reason to refuse to
	// start: a torn tail is a write that was never acknowledged. They are worth
	// surfacing, and there is nowhere yet to surface them to — see the note on
	// observability in doc.go.
	return nil
}

// apply replays one record into the in-memory state.
//
// It does not write to the log. These records are already in it, and appending
// them again would grow the log by its own contents on every start.
func (db *DB) apply(rec wal.Record) error {
	switch rec.Type {
	case wal.TypePut:
		v, isPut, err := decodeRecord(rec.Type, rec.Payload, &db.opts)
		if err != nil {
			return fmt.Errorf("seq %d: %w", rec.Seq, err)
		}
		if !isPut {
			return fmt.Errorf("%w: seq %d decoded as a delete under a PUT", ErrCorrupt, rec.Seq)
		}
		if err := db.index.Insert(v.ID, v.Values); err != nil {
			return fmt.Errorf("seq %d: index insert %q: %w", rec.Seq, v.ID, err)
		}
		db.store.Put(v.ID, v.Metadata)
		return nil

	case wal.TypeDelete:
		v, _, err := decodeRecord(rec.Type, rec.Payload, &db.opts)
		if err != nil {
			return fmt.Errorf("seq %d: %w", rec.Seq, err)
		}
		db.index.Delete(v.ID)
		db.store.Delete(v.ID)
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
