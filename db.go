package govecdb

import (
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/khambampati-subhash/govecdb/internal/dirlock"
	"github.com/khambampati-subhash/govecdb/internal/snapshot"
	"github.com/khambampati-subhash/govecdb/internal/store"
	"github.com/khambampati-subhash/govecdb/internal/wal"
)

// DB is an embeddable vector database: an index, a write-ahead log, and
// point-in-time snapshots over one directory.
//
// It is safe for concurrent use. Searches run in parallel with each other;
// writes serialize against each other and against the index.
//
// # The ordering rule
//
// Every write is appended to the log first and only then applied to the index.
// Reverse the two and a crash between them acknowledges a write that no longer
// exists. The index is derived state — it can always be rebuilt from the log,
// and it is never the source of truth.
//
// # No context parameter
//
// Deliberate. Every operation here is local and bounded: a search is
// microseconds of in-memory work, a write is one append. The only genuinely
// long operation is Snapshot, and it cannot be abandoned halfway without
// leaving the index locked. A ctx that no method could honour would be a
// promise of cancellation that is never kept, which is worse than not offering
// one.
type DB struct {
	// mu guards the fields below and serializes writes. It is not held during a
	// search: the index has its own lock and readers run under that, which is
	// what lets queries scale across cores.
	mu   sync.RWMutex
	opts options
	dir  string

	index Index
	store *store.Map
	log   *wal.Writer

	// snapSeq is the log sequence covered by the newest snapshot this process
	// has written or loaded.
	snapSeq uint64

	// nextSeq is where the log resumes after recovery. Written by restore and
	// read once by Open; the log owns the sequence from then on.
	nextSeq uint64

	closed bool

	// lock is the cross-process half of one writer per directory; openDirs is
	// the in-process half. Held from Open to Close.
	lock *dirlock.Handle

	// failed is sticky and is the whole of failing closed. Once an append could
	// not be made durable, the log has a hole in it and appending past that hole
	// produces records replay will silently discard. Reads keep working because
	// the in-memory index is still correct; writes stop, permanently, because
	// nothing about the next one can be promised.
	failed error

	// buf is the reused encode buffer for the write path.
	buf []byte

	stopSnap chan struct{}
	doneSnap chan struct{}

	// The background calibrator (calibrate.go). calKick holds at most one
	// pending request; calAt is the live count the last calibration measured,
	// 0 for never. All nil/zero under WithEfCalibration(false).
	calKick chan struct{}
	stopCal chan struct{}
	doneCal chan struct{}
	calAt   atomic.Int64
}

// Subdirectories. Separate rather than interleaved so retention on one is not a
// filter over the other's files: deleting old log segments and pruning old
// snapshots are different policies on different schedules.
const (
	walSubdir      = "wal"
	snapshotSubdir = "snapshots"

	// dirPerm is 0700, not 0755. Embeddings are derived from whatever the user
	// embedded and are frequently reversible enough to matter, so the database
	// directory is not world-readable. It is also what protects the files
	// inside: another user cannot traverse into a directory they cannot read,
	// whatever the individual file modes are.
	dirPerm os.FileMode = 0o700
)

// openDirs is how ErrAlreadyOpen is detected: one writer per directory, which
// is what the log and the snapshot store both assume. Two DBs on one path would
// interleave segment numbering and delete each other's temporary files.
//
// It covers this process only; other processes are kept out by an flock on the
// directory (internal/dirlock), which the kernel releases when a process dies
// and so cannot outlive a crash the way a lock file would. Both are needed:
// flock is per open file description, so it would also refuse a second Open in
// this process, but with an error that blames another process.
var openDirs sync.Map

// Open opens or creates a database in dir.
//
// # What it does on the way up
//
//  1. Loads the newest snapshot that passes its checksum, falling back to an
//     older one if it does not.
//  2. Replays log records written after that snapshot.
//  3. Opens the log for appending at the next free sequence.
//
// A directory that does not exist becomes an empty database. A directory whose
// index was built with a different dimension, metric, or M is refused: those are
// structural, and reopening under different ones would search a graph whose
// edges were chosen under different rules.
//
// A directory held by another process is refused with ErrAlreadyOpen. A
// read-only open (WithReadOnly) takes a shared lock instead, so several readers
// may coexist, but never alongside a writer.
func Open(dir string, opts ...Option) (db *DB, err error) {
	o := defaultOptions()
	for _, opt := range opts {
		if opt == nil {
			return nil, fmt.Errorf("%w: nil option", ErrInvalidConfig)
		}
		if err := opt(&o); err != nil {
			return nil, err
		}
	}
	if err := o.validate(); err != nil {
		return nil, err
	}
	if dir == "" {
		return nil, fmt.Errorf("%w: directory is empty", ErrInvalidConfig)
	}
	dir = filepath.Clean(dir)

	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("%w: resolving %q: %w", ErrInvalidConfig, dir, err)
	}
	if _, loaded := openDirs.LoadOrStore(abs, struct{}{}); loaded {
		return nil, fmt.Errorf("%w: %s", ErrAlreadyOpen, abs)
	}
	// Every failure past this point has to release the claim, or a failed Open
	// would make the directory permanently unopenable by this process.
	defer func() {
		if err != nil {
			openDirs.Delete(abs)
		}
	}()

	if o.readOnly {
		// Creating a database is a write. A read-only open of a path that is not
		// there is a typo, and answering it with an empty database would hide it.
		if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
			return nil, fmt.Errorf("%w: read-only open of %q, which is not an existing directory", ErrInvalidConfig, dir)
		}
	} else if err := makeDirs(dir); err != nil {
		return nil, err
	}

	lock, err := dirlock.Lock(dir, o.readOnly)
	if errors.Is(err, dirlock.ErrLocked) {
		return nil, fmt.Errorf("%w: %s is in use by another process", ErrAlreadyOpen, abs)
	}
	if err != nil {
		return nil, fmt.Errorf("govecdb: %w", err)
	}
	defer func() {
		if err != nil {
			lock.Release()
		}
	}()

	d := &DB{opts: o, dir: dir, lock: lock}
	if err := d.restore(); err != nil {
		return nil, err
	}
	// Calibration writes nothing, so a read-only database gets it too: a
	// static collection is exactly the one whose width is worth measuring once.
	if o.calibrate {
		d.calKick = make(chan struct{}, 1)
		d.stopCal = make(chan struct{})
		d.doneCal = make(chan struct{})
		go d.calibrationLoop()
		d.kickCalibration(d.index.Len())
	}
	defer func() {
		if err != nil && d.stopCal != nil {
			close(d.stopCal)
			<-d.doneCal
		}
	}()

	if o.readOnly {
		// No writer at all, rather than a writer nobody calls: wal.Open starts a
		// fresh segment, which is a file created by an open that promised not to
		// write.
		return d, nil
	}

	d.log, err = wal.Open(filepath.Join(dir, walSubdir), wal.Options{
		SyncPolicy:      o.syncPolicy.internal(),
		SyncInterval:    o.syncInterval,
		MaxSegmentBytes: o.maxSegmentBytes,
		FirstSeq:        d.nextSeq,
	})
	if err != nil {
		return nil, fmt.Errorf("govecdb: open log: %w", err)
	}

	if o.snapshotEvery > 0 {
		d.stopSnap = make(chan struct{})
		d.doneSnap = make(chan struct{})
		go d.snapshotLoop()
	}
	return d, nil
}

// Add stores a vector, replacing any vector already under its id.
//
// The log is written first, then the index, so a crash between them costs a
// replayed record rather than an acknowledged write that vanished.
func (db *DB) Add(v Vector) error {
	if err := db.opts.validateVector(v); err != nil {
		return err
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	if err := db.writable(); err != nil {
		return err
	}
	if err := db.putLocked(v); err != nil {
		return err
	}
	db.kickCalibration(db.index.Len())
	return nil
}

// AddBatch stores many vectors. It validates all of them before writing any, so
// a batch with one bad record leaves the database untouched rather than half
// applied — the only all-or-nothing guarantee here, and it is about *validation*
// rather than durability: a batch whose write fails may have made a prefix
// durable, which the next start replays. The database is read-only by then.
//
// The whole batch is logged before any of it reaches the index, and under
// SyncAlways that is one fsync rather than one per vector. It used to be one per
// vector: 100 vectors took 0.56–0.70 s against 47 ms now, nearly all of which is
// the index (BenchmarkWritePath). Every REST upsert comes through here.
func (db *DB) AddBatch(vs []Vector) error {
	if len(vs) == 0 {
		return nil
	}
	if len(vs) > db.opts.maxBatch {
		return fmt.Errorf("%w: batch of %d, max %d", ErrInvalidRequest, len(vs), db.opts.maxBatch)
	}
	for i := range vs {
		if err := db.opts.validateVector(vs[i]); err != nil {
			return fmt.Errorf("vector %d: %w", i, err)
		}
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	if err := db.writable(); err != nil {
		return err
	}
	_, err := db.log.AppendBatch(wal.TypePut, len(vs), func(i int) []byte {
		db.buf = encodePut(db.buf[:0], vs[i])
		return db.buf
	})
	if err != nil {
		return db.fail(err)
	}
	if err := db.applyPuts(vs); err != nil {
		return err
	}
	db.kickCalibration(db.index.Len())
	return nil
}

// applyPuts applies PUTs that are already in the log, in order, so a batch
// naming one id twice ends as replaying it would: with the last vector and the
// last metadata. Callers hold the write lock.
//
// An index that batches gets them in one call and links them in parallel; one
// that does not gets them one Insert at a time.
func (db *DB) applyPuts(vs []Vector) error {
	b, ok := db.index.(indexBatcher)
	if !ok || len(vs) < 2 {
		for i := range vs {
			if err := db.applyPut(vs[i]); err != nil {
				return fmt.Errorf("vector %d: %w", i, err)
			}
		}
		return nil
	}

	ids := make([]string, len(vs))
	values := make([][]float32, len(vs))
	for i := range vs {
		ids[i], values[i] = vs[i].ID, vs[i].Values
	}
	if err := b.InsertBatch(ids, values, db.opts.insertWorkers); err != nil {
		// As in applyPut: the records are durable and the next start replays
		// them, so failing the call is right and the gap closes on restart.
		return fmt.Errorf("govecdb: index insert batch: %w", err)
	}
	for i := range vs {
		db.store.Put(vs[i].ID, vs[i].Metadata)
	}
	return nil
}

// putLocked appends a PUT and applies it. Callers hold the write lock.
func (db *DB) putLocked(v Vector) error {
	db.buf = encodePut(db.buf[:0], v)
	if _, err := db.log.Append(wal.TypePut, db.buf); err != nil {
		return db.fail(err)
	}
	return db.applyPut(v)
}

// applyPut applies a PUT that is already in the log.
func (db *DB) applyPut(v Vector) error {
	if err := db.index.Insert(v.ID, v.Values); err != nil {
		// The record is already durable, so replay will apply it on the next
		// start. Failing the call rather than pretending otherwise is right, but
		// the log and the index now disagree until then — which is exactly the
		// direction the ordering rule chooses on purpose.
		return fmt.Errorf("govecdb: index insert %q: %w", v.ID, err)
	}
	db.store.Put(v.ID, v.Metadata)
	return nil
}

// Delete removes a vector. Deleting an id that is not there is not an error:
// replay applies records more than once across a snapshot boundary, and an
// operation that failed the second time would make recovery order-sensitive.
func (db *DB) Delete(id string) error {
	if err := validateID(id, db.opts.maxIDBytes); err != nil {
		return err
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	if err := db.writable(); err != nil {
		return err
	}

	db.buf = encodeDelete(db.buf[:0], id)
	if _, err := db.log.Append(wal.TypeDelete, db.buf); err != nil {
		return db.fail(err)
	}
	db.index.Delete(id)
	db.store.Delete(id)
	db.kickCalibration(db.index.Len())
	return nil
}

// Get returns the vector stored under id.
//
// Values come back in the form the index holds them, which for Cosine is the
// unit vector: that metric is a statement that magnitude carries no meaning, and
// keeping a second copy of every embedding to hand back a number nothing uses
// would double the memory of the largest thing in the process.
func (db *DB) Get(id string) (Vector, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return Vector{}, ErrClosed
	}
	values, ok := db.index.Lookup(id)
	if !ok {
		return Vector{}, fmt.Errorf("%w: %q", ErrNotFound, id)
	}
	md, _ := db.store.Get(id)
	return Vector{ID: id, Values: values, Metadata: md}, nil
}

// Search returns the nearest vectors to a query, nearest first.
//
// It holds only a read lock here and none at all inside the index's traversal,
// so searches run in parallel with one another. Writers are excluded for the
// moment it takes to read the index handle, not for the search.
func (db *DB) Search(req SearchRequest) ([]Match, error) {
	db.mu.RLock()
	if db.closed {
		db.mu.RUnlock()
		return nil, ErrClosed
	}
	idx, st, opts := db.index, db.store, db.opts
	db.mu.RUnlock()

	ef, err := opts.validateSearch(req, idx.SuggestedEf)
	if err != nil {
		return nil, err
	}

	// The index is handed a predicate over ids, and the metadata it stands for
	// stays on this side of the boundary. st.Match evaluates without copying,
	// because this runs once per candidate node rather than once per result.
	var allow func(id string) bool
	if req.Filter != nil {
		f := req.Filter
		allow = func(id string) bool { return st.Match(id, f.Match) }
	}

	matches, err := idx.Search(req.Query, req.K, ef, allow)
	if err != nil {
		return nil, err
	}
	for i := range matches {
		if md, ok := st.Get(matches[i].ID); ok {
			matches[i].Metadata = md
		}
	}
	return matches, nil
}

// Snapshot writes the current state to disk and prunes older snapshots.
//
// It is what bounds recovery time: without one, starting up replays the whole
// log and rebuilds the index at roughly 700 µs of CPU per vector, spread across
// cores; with one, it loads a
// graph instead. The cost is a read lock held for the write — searches continue,
// writers wait — plus a fixed ~10 ms of fsync whatever the size.
//
// The sequence it records is read before the state is serialized, so it can only
// ever under-claim: a write landing during the snapshot is replayed on the next
// start rather than assumed to be inside it. Claiming the other way would skip a
// record that never made it in.
func (db *DB) Snapshot() error {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return ErrClosed
	}
	if db.opts.readOnly {
		db.mu.Unlock()
		return errOpenedReadOnly
	}
	seq := db.log.LastSeq()
	idx, st := db.index, db.store
	db.mu.Unlock()

	// Every attempt past this point reports, success or failure, because the
	// interval timer discards the error and an event is the only way its
	// failures are seen.
	start := time.Now()
	taken, err := db.snapshotAt(seq, idx, st)
	if err != nil {
		db.emit(SnapshotFailed{Cause: err})
		return err
	}
	taken.Took = time.Since(start)
	db.emit(taken)

	db.mu.Lock()
	db.snapSeq = max(db.snapSeq, seq)
	db.mu.Unlock()
	return nil
}

// snapshotAt writes a snapshot of idx and st as of seq, prunes, and truncates
// the log behind it.
func (db *DB) snapshotAt(seq uint64, idx Index, st *store.Map) (SnapshotTaken, error) {
	ser, ok := idx.(indexSerializer)
	if !ok {
		return SnapshotTaken{}, fmt.Errorf("govecdb: index of type %T cannot be snapshotted", idx)
	}

	dir := filepath.Join(db.dir, snapshotSubdir)
	snap, err := snapshot.Create(dir, seq, func(w io.Writer) error {
		return writeSnapshot(w, ser, st)
	})
	if err != nil {
		return SnapshotTaken{}, fmt.Errorf("govecdb: snapshot: %w", err)
	}

	// Pruned only after the new one is durable, so the number of usable copies
	// never dips below the retention on the way through.
	pruned, err := snapshot.Prune(dir, db.opts.snapshotsKept)
	if err != nil {
		return SnapshotTaken{}, fmt.Errorf("govecdb: prune snapshots: %w", err)
	}
	removed, err := db.truncateLog(dir)
	if err != nil {
		return SnapshotTaken{}, err
	}
	return SnapshotTaken{Seq: seq, Bytes: snap.Bytes, Pruned: pruned, SegmentsRemoved: removed}, nil
}

// truncateLog deletes log segments that the retained snapshots make redundant.
//
// # Against the oldest snapshot, never the newest
//
// Retaining more than one snapshot is what makes a corrupt one survivable, and
// that only works if the log still reaches back far enough for the older one to
// be usable. Truncating to the newest would delete exactly the records the older
// copy needs — leaving a second snapshot that is paid for and cannot be used.
//
// # And only after checking that snapshot is readable
//
// The oldest snapshot is verified before anything is deleted, because the
// question being asked is "may I delete the records this snapshot stands in
// for?", and a snapshot nobody has checked cannot stand in for anything. A
// failure here is deliberately not an error: a log that keeps growing is a disk
// problem, while deleting records only an unreadable snapshot could replace is a
// data problem, and the two are not close enough to trade.
//
// The skip is reported as a TruncationSkipped event instead. It reports how
// many segments it removed.
func (db *DB) truncateLog(snapDir string) (int, error) {
	all, err := snapshot.List(snapDir)
	if err != nil {
		return 0, fmt.Errorf("govecdb: list snapshots: %w", err)
	}
	if len(all) == 0 {
		return 0, nil
	}

	// List is newest first.
	oldest := all[len(all)-1]
	if err := snapshot.Verify(oldest); err != nil {
		db.emit(TruncationSkipped{Path: oldest.Path, Seq: oldest.Seq, Cause: err})
		return 0, nil
	}

	// Records at or below the snapshot's sequence are reconstructible from it;
	// everything above has to stay.
	removed, err := wal.Truncate(filepath.Join(db.dir, walSubdir), oldest.Seq+1, wal.Options{})
	if err != nil {
		return removed, fmt.Errorf("govecdb: truncate log: %w", err)
	}
	return removed, nil
}

// Compact rebuilds the index over its live vectors and reports how many slots
// were reclaimed. It stops the world: no search runs while it does.
//
// The database does not schedule this. Stats().DeadRatio() is the signal, and
// only the caller knows which moment can afford the pause — around 0.5 is where
// it pays, because the pause tracks survivors rather than garbage.
func (db *DB) Compact() (int, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return 0, ErrClosed
	}
	return db.index.Compact(), nil
}

// Len is how many vectors a search can return.
func (db *DB) Len() int {
	db.mu.RLock()
	defer db.mu.RUnlock()
	if db.closed {
		return 0
	}
	return db.index.Len()
}

// Stats reports what the database is holding.
func (db *DB) Stats() Stats {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return Stats{}
	}
	live, deleted, slots := db.index.Stats()
	return Stats{
		Live:         live,
		Deleted:      deleted,
		Slots:        slots,
		WithMetadata: db.store.Len(),
		LastSeq:      db.lastSeqLocked(),
		SnapshotSeq:  db.snapSeq,
		EfScale:      efScaleOf(db.index),
	}
}

// Sync forces buffered writes to stable storage. Under SyncAlways every write is
// already durable and this is a no-op in effect; under the other policies it is
// how a caller draws a line before doing something that must not outlive the
// data behind it.
func (db *DB) Sync() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return ErrClosed
	}
	if db.failed != nil {
		return db.failed
	}
	if db.log == nil {
		return nil // read-only: nothing was written, so nothing is waiting
	}
	if err := db.log.Sync(); err != nil {
		return db.fail(err)
	}
	return nil
}

// Close flushes, syncs and releases the database. It is idempotent, because
// Close belongs in a defer and a shutdown path should not have to track whether
// it already ran.
//
// It does not take a final snapshot. Doing so would make shutdown take seconds
// on a large index and would fail in exactly the situations — a full or failing
// disk — where shutting down cleanly matters most. Call Snapshot first if the
// next start should be fast.
func (db *DB) Close() error {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return nil
	}
	db.closed = true
	stop, done := db.stopSnap, db.doneSnap
	db.mu.Unlock()

	// Stop the snapshotter before taking the lock back, so a snapshot already in
	// flight finishes instead of deadlocking against us.
	if stop != nil {
		close(stop)
		<-done
	}
	// The calibrator too; a calibration in flight gives up between queries.
	if db.stopCal != nil {
		close(db.stopCal)
		<-db.doneCal
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	if abs, err := filepath.Abs(db.dir); err == nil {
		openDirs.Delete(abs)
	}
	var err error
	if db.log != nil {
		err = db.log.Close()
	}
	// Released after the log is closed, so a process that opens the directory
	// the moment the lock is free cannot find a segment still being written.
	if lerr := db.lock.Release(); err == nil && lerr != nil {
		err = lerr
	}
	// A sticky failure is the more useful thing to report: Close's own error is
	// usually a downstream symptom of it.
	if db.failed != nil {
		return db.failed
	}
	if err != nil {
		return fmt.Errorf("govecdb: close log: %w", err)
	}
	return nil
}

// makeDirs creates the database directory and its two subdirectories.
//
// # Why the permissions are handled explicitly
//
// os.MkdirAll applies its mode only to directories it actually creates, so
// calling it on a path that already exists silently leaves whatever mode is
// there. That is the right behaviour for the database directory: if an operator
// made it themselves, its mode is their decision and quietly tightening it would
// be presumptuous — a group that was granted read access would lose it with no
// message anywhere.
//
// The two subdirectories are different. Nothing but this package creates them,
// so they are created here at 0700 rather than left to the log and the snapshot
// store, which default to 0755. That is what keeps the data unreadable by other
// users even when the parent directory is permissive: the files inside carry
// ordinary modes, and it is the directory they cannot traverse that protects
// them.
func makeDirs(dir string) error {
	if err := os.MkdirAll(dir, dirPerm); err != nil {
		return fmt.Errorf("govecdb: create %q: %w", dir, err)
	}
	for _, sub := range []string{walSubdir, snapshotSubdir} {
		p := filepath.Join(dir, sub)
		if err := os.MkdirAll(p, dirPerm); err != nil {
			return fmt.Errorf("govecdb: create %q: %w", p, err)
		}
	}
	return nil
}

// writable reports whether a write may proceed. Callers hold the write lock.
func (db *DB) writable() error {
	if db.closed {
		return ErrClosed
	}
	if db.opts.readOnly {
		return errOpenedReadOnly
	}
	if db.failed != nil {
		return db.failed
	}
	return nil
}

// lastSeqLocked is the sequence of the newest record in the log. A read-only
// database has no writer to ask, so it reports where replay stopped.
func (db *DB) lastSeqLocked() uint64 {
	if db.log == nil {
		return max(db.nextSeq, 1) - 1
	}
	return db.log.LastSeq()
}

// fail records the first durability failure and returns it wrapped in
// ErrReadOnly, so a caller can match the category and still print the cause.
//
// The event fires here, under the write lock, which is the reason WithObserver
// forbids calling back into the DB. Deferring it past the unlock would mean
// threading a pending event through all four callers for an event that fires
// at most once in a database's life.
func (db *DB) fail(err error) error {
	if db.failed == nil {
		db.failed = fmt.Errorf("%w after a durability failure: %w", ErrReadOnly, err)
		db.emit(DurabilityFailure{Cause: err})
	}
	return db.failed
}

// snapshotLoop drives WithSnapshotInterval.
//
// A timer rather than a counter of writes, because what a snapshot bounds is
// recovery *time*, and time is what passes whether or not anybody is writing.
func (db *DB) snapshotLoop() {
	defer close(db.doneSnap)

	// The first snapshot lands somewhere in the second half of the interval
	// rather than at exactly one interval. A service opens many databases at
	// once — at start-up, or a whole generation of a rebuild — and identical
	// timers would snapshot them all in the same instant, every interval, each
	// holding its writers off while it serializes. A random phase spreads them
	// out once, and the shared period keeps them apart.
	d := db.opts.snapshotEvery
	first := time.NewTimer(d/2 + rand.N(d/2+1))
	defer first.Stop()
	select {
	case <-db.stopSnap:
		return
	case <-first.C:
		_ = db.Snapshot()
	}

	t := time.NewTicker(d)
	defer t.Stop()

	for {
		select {
		case <-db.stopSnap:
			return
		case <-t.C:
			// Errors are dropped rather than escalated. A failed snapshot costs
			// a longer replay on the next start, which is a performance problem;
			// turning it into a durability failure would take a working database
			// read-only over one that is still entirely correct. Snapshot
			// reports the failure as a SnapshotFailed event, which is how an
			// operator sees it.
			_ = db.Snapshot()
		}
	}
}
