package wal

import (
	"bufio"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// Writer is an append-only log across a directory of segments. Safe for
// concurrent use: appends serialize on a mutex, which is not a bottleneck
// because the thing an append is waiting for is the disk, not the lock.
type Writer struct {
	mu   sync.Mutex
	opts Options
	dir  string

	file     *os.File
	buf      *bufio.Writer
	segIndex uint32
	segBytes int64

	nextSeq uint64

	// lastSeq mirrors nextSeq-1 for LastSeq, which must not take mu: mu is held
	// across a SyncAlways fsync, and a caller asking where the log has got to —
	// a stats endpoint, say — should not wait milliseconds for the answer.
	lastSeq atomic.Uint64

	// header is reused across appends so the hot path allocates nothing beyond
	// what bufio already holds.
	header [recordHeaderSize]byte

	// failed is sticky. A log that could not be written to cannot be trusted for
	// the writes that follow: if append N never reached the disk, appending N+1
	// on top produces a log with a hole in it, and replay would stop at the hole
	// and silently discard everything after. So the first failure ends the
	// Writer's useful life and every later call returns it.
	//
	// What the *database* does about that -- refuse writes, go read-only, fail
	// closed -- is the wider policy decision, and it belongs at the layer above.
	// This is the half of it that has to live here.
	failed error

	closed bool

	// syncMu is held across every fsync of file, and across closing it. It is
	// what lets Sync and the interval flusher fsync *outside* mu: an fsync is
	// milliseconds, and holding mu for it stalled every Append behind every
	// flush tick. Rotation and Close take it before closing the file, so an
	// fsync in flight never finds its file closed underneath it. Taken after
	// mu, never before.
	syncMu sync.Mutex

	// syncErr is the first fsync failure, guarded by syncMu. Every later fsync
	// reports it rather than trying again: after a failed fsync the kernel may
	// have dropped the dirty pages, and a second fsync that "succeeds" would
	// acknowledge records sitting behind a hole. It is mirrored into failed as
	// soon as mu can be taken, but this copy is what closes the gap between the
	// two locks.
	syncErr error

	// Background flusher, only running under SyncInterval.
	stop chan struct{}
	done chan struct{}
}

// Open opens (or creates) a log in dir and returns a Writer positioned to append.
//
// # It always starts a new segment
//
// Even when segments already exist, Open never appends to the last one. The
// reason is torn writes: a segment whose tail was cut off by power loss ends in
// a partial record, and recovery stops at the first record that fails its
// checksum. Appending after that partial record would put perfectly good writes
// *behind* a permanent stopping point, where replay can never reach them —
// silent data loss produced by the recovery mechanism itself.
//
// Starting a fresh segment costs one mostly-empty file per restart and makes
// that impossible. Repairing the torn tail is the reader's job, not a
// precondition for accepting new writes.
func Open(dir string, opts Options) (*Writer, error) {
	opts = opts.withDefaults()

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("wal: create dir: %w", err)
	}

	existing, err := listSegments(dir)
	if err != nil {
		return nil, fmt.Errorf("wal: list segments: %w", err)
	}
	next := uint32(1)
	if n := len(existing); n > 0 {
		next = existing[n-1] + 1
	}

	w := &Writer{
		opts:    opts,
		dir:     dir,
		nextSeq: opts.FirstSeq,
	}
	w.lastSeq.Store(opts.FirstSeq - 1)
	if err := w.openSegment(next); err != nil {
		return nil, err
	}

	if opts.SyncPolicy == SyncInterval {
		w.stop = make(chan struct{})
		w.done = make(chan struct{})
		go w.flushLoop()
	}
	return w, nil
}

// Append writes one record and returns the sequence number it was given.
//
// The sequence is assigned here rather than by the caller because ordering is
// the log's invariant to hold: two goroutines appending concurrently must come
// out numbered in the order they were written, and only the thing holding the
// lock knows that order.
func (w *Writer) Append(typ RecordType, payload []byte) (uint64, error) {
	if !typ.valid() {
		return 0, fmt.Errorf("%w: %d", ErrInvalidType, typ)
	}
	if len(payload) > w.opts.MaxRecordBytes {
		return 0, fmt.Errorf("%w: %d bytes, limit %d", ErrRecordTooLarge, len(payload), w.opts.MaxRecordBytes)
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.usable(); err != nil {
		return 0, err
	}
	seq, err := w.writeLocked(typ, payload)
	if err != nil {
		return 0, err
	}
	if w.opts.SyncPolicy == SyncAlways {
		if err := w.syncLocked(); err != nil {
			return 0, w.fail(err)
		}
	}
	return seq, nil
}

// AppendBatch writes n records of one type and returns the sequence of the
// first. payload(i) produces record i; the slice it returns may alias a buffer
// it reuses, because each record is copied into the log before the next call.
//
// It exists for one reason: under SyncAlways, n calls to Append are n fsyncs,
// and an fsync is the whole cost of a write — 4 ms against 0.7 µs for the
// append itself. A batch pays one, at the end, and is acknowledged only after
// it, so "returned nil means durable" holds for every record in it.
//
// The records are streamed through the same buffer Append uses rather than
// encoded up front: a 10,000-vector batch at dimension 768 is 30 MB, and
// holding it twice to save nothing would be the wrong trade.
//
// A record refused partway through fails the writer, unlike a refused Append.
// Its predecessors are already in the buffer, and a later Sync would make
// durable a prefix the caller was told had failed.
//
// payload runs under the writer's lock and must not call back into it.
func (w *Writer) AppendBatch(typ RecordType, n int, payload func(i int) []byte) (uint64, error) {
	if !typ.valid() {
		return 0, fmt.Errorf("%w: %d", ErrInvalidType, typ)
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.usable(); err != nil {
		return 0, err
	}
	first := w.nextSeq
	for i := range n {
		p := payload(i)
		if len(p) > w.opts.MaxRecordBytes {
			return 0, w.fail(fmt.Errorf("%w: record %d of batch, %d bytes, limit %d", ErrRecordTooLarge, i, len(p), w.opts.MaxRecordBytes))
		}
		if _, err := w.writeLocked(typ, p); err != nil {
			return 0, err
		}
	}
	if w.opts.SyncPolicy == SyncAlways {
		if err := w.syncLocked(); err != nil {
			return 0, w.fail(err)
		}
	}
	return first, nil
}

// usable reports why the writer cannot take a record, if it cannot.
func (w *Writer) usable() error {
	if w.closed {
		return ErrClosed
	}
	return w.failed
}

// writeLocked frames one record into the buffer, rotating first if it would
// overflow the segment. It does not sync; that is the caller's policy. Any error
// has already failed the writer.
func (w *Writer) writeLocked(typ RecordType, payload []byte) (uint64, error) {
	size := int64(recordHeaderSize + len(payload))
	if err := w.maybeRotate(size); err != nil {
		return 0, w.fail(err)
	}

	seq := w.nextSeq
	encodeRecordHeader(w.header[:], typ, seq, payload)

	if _, err := w.buf.Write(w.header[:]); err != nil {
		return 0, w.fail(fmt.Errorf("wal: write header: %w", err))
	}
	if _, err := w.buf.Write(payload); err != nil {
		return 0, w.fail(fmt.Errorf("wal: write payload: %w", err))
	}

	w.segBytes += size
	w.nextSeq++
	w.lastSeq.Store(seq)
	return seq, nil
}

// Sync flushes buffered data and fsyncs the current segment. It is what
// SyncAlways calls on every append and what a caller reaches for before doing
// something that must not outlive the log, such as acknowledging a batch.
//
// It returns only once every record appended before it is durable: the buffer
// is moved into the current segment under the lock, every earlier segment was
// fsynced by the rotation that closed it, and the fsync of the current one
// runs after. That fsync runs outside the lock, so appends carry on into the
// buffer while it does.
func (w *Writer) Sync() error {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return ErrClosed
	}
	if w.failed != nil {
		w.mu.Unlock()
		return w.failed
	}
	return w.syncUnlocking()
}

// Close flushes, fsyncs, and releases the current segment. It is idempotent —
// closing twice is a no-op rather than an error, because Close lands in defers
// and a shutdown path should not care whether it ran already.
func (w *Writer) Close() error {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return nil
	}
	w.closed = true
	stop, done := w.stop, w.done
	w.mu.Unlock()

	// Stop the flusher before taking the lock back, so a tick already in flight
	// finishes rather than deadlocking against us.
	if stop != nil {
		close(stop)
		<-done
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	// A previous failure means the buffer's contents are already untrustworthy;
	// flushing on top of a hole would write good bytes after missing ones. Shut
	// the file and report the original cause.
	if w.failed != nil {
		w.syncMu.Lock()
		w.file.Close()
		w.syncMu.Unlock()
		return w.failed
	}
	syncErr := w.syncLocked()
	// A Sync still fsyncing outside mu holds syncMu; wait it out.
	w.syncMu.Lock()
	closeErr := w.file.Close()
	w.syncMu.Unlock()
	if syncErr != nil {
		return syncErr
	}
	return closeErr
}

// LastSeq reports the sequence number most recently assigned, or FirstSeq-1 when
// nothing has been appended. Recovery and the checkpointer both need to know
// where the log has got to.
//
// It takes no lock, so it never waits behind an append's fsync. A record it
// reports has been framed into the log but is durable only per the sync policy.
func (w *Writer) LastSeq() uint64 {
	return w.lastSeq.Load()
}

// Segment reports the index of the segment currently being written, which is
// what a truncation policy compares against: everything strictly below the
// segment holding the last checkpoint is deletable.
func (w *Writer) Segment() uint32 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.segIndex
}

// syncLocked flushes the buffer into the file and then the file onto the disk.
// Both halves matter and they are different things: Flush moves bytes from this
// process into the kernel, Sync moves them from the kernel onto stable storage.
// Doing only the first is what makes people believe they have durability.
//
// Callers hold mu, and it stays held across the fsync: this is the path for
// SyncAlways, where the append is not acknowledged until the fsync returns
// anyway, and for rotation and Close.
func (w *Writer) syncLocked() error {
	if err := w.buf.Flush(); err != nil {
		return fmt.Errorf("wal: flush: %w", err)
	}
	w.syncMu.Lock()
	defer w.syncMu.Unlock()
	return w.fsyncHeld(w.file)
}

// syncUnlocking flushes the buffer under mu, then fsyncs with mu released, so
// appends into the buffer overlap the disk. Callers hold mu; it is released on
// return. Any failure fails the writer.
func (w *Writer) syncUnlocking() error {
	if err := w.buf.Flush(); err != nil {
		err = w.fail(fmt.Errorf("wal: flush: %w", err))
		w.mu.Unlock()
		return err
	}
	// Taken before mu is released, so the file cannot be rotated away between
	// reading it here and fsyncing it below.
	w.syncMu.Lock()
	f := w.file
	w.mu.Unlock()

	err := w.fsyncHeld(f)
	w.syncMu.Unlock()
	if err != nil {
		w.mu.Lock()
		err = w.fail(err)
		w.mu.Unlock()
	}
	return err
}

// fsyncHeld fsyncs f, or reports the fsync failure that came before. Callers
// hold syncMu.
func (w *Writer) fsyncHeld(f *os.File) error {
	if w.syncErr != nil {
		return w.syncErr
	}
	if err := f.Sync(); err != nil {
		w.syncErr = fmt.Errorf("wal: fsync: %w", err)
		return w.syncErr
	}
	return nil
}

// maybeRotate starts a new segment when the incoming record would push the
// current one past its limit.
//
// A record larger than MaxSegmentBytes is written anyway, into a segment of its
// own. Refusing it would make the segment size a limit on what can be stored,
// which is not what it is for — it is a truncation granularity.
func (w *Writer) maybeRotate(size int64) error {
	if w.segBytes+size <= w.opts.MaxSegmentBytes {
		return nil
	}
	if w.segBytes <= fileHeaderSize {
		return nil // nothing but the header here; rotating would leave an empty file
	}
	return w.rotate()
}

// rotate closes the current segment durably and opens the next one.
//
// The old segment is fsynced before the new one is created. If it were not, a
// crash could leave the new segment on disk while the tail of the old one was
// still only in the page cache — a hole in the middle of the log rather than at
// its end, which is the one shape recovery cannot repair.
//
// The fsync and the close share one hold of syncMu, so a Sync fsyncing outside
// mu finishes before the file it holds is closed, and none can start on it
// after: a new one reads the file under mu, which rotation holds throughout.
// A failed fsync from anywhere stops rotation here, so no segment is ever
// created behind a hole.
func (w *Writer) rotate() error {
	if err := w.buf.Flush(); err != nil {
		return fmt.Errorf("wal: flush: %w", err)
	}
	w.syncMu.Lock()
	err := w.fsyncHeld(w.file)
	if err == nil {
		if cerr := w.file.Close(); cerr != nil {
			err = fmt.Errorf("wal: close segment: %w", cerr)
		}
	}
	w.syncMu.Unlock()
	if err != nil {
		return err
	}
	return w.openSegment(w.segIndex + 1)
}

// openSegment creates a segment file, writes its header, and makes the file's
// existence durable.
//
// # Why the directory is fsynced
//
// fsync on a file makes its *contents* durable. It says nothing about the
// directory entry that gives the file a name, and a file with no name is a file
// that is not there after a crash.
//
// That is the difference between an inconvenience and a broken promise. Every
// SyncAlways append fsyncs this file, so its records are on the platter — but if
// the directory entry linking it was never made durable, power loss can take the
// whole segment away and those acknowledged writes with it. The window is small
// and it is exactly the shape of failure this package exists to prevent.
//
// One extra fsync per segment, which is once per Open and once per rotation —
// every 64 MiB of log at the default. Rotation already costs an fsync and a
// create, so this is not the expensive part of a rare operation.
func (w *Writer) openSegment(index uint32) error {
	path := segmentPath(w.dir, index)

	// O_EXCL: a segment index that already exists means either a bug in the
	// rotation arithmetic or another process writing this directory. Both are
	// worth failing loudly over rather than overwriting somebody's log.
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("wal: create segment %s: %w", segmentName(index), err)
	}

	var hdr [fileHeaderSize]byte
	encodeFileHeader(hdr[:])
	if _, err := f.Write(hdr[:]); err != nil {
		f.Close()
		return fmt.Errorf("wal: write segment header: %w", err)
	}

	if err := syncDir(w.dir); err != nil {
		f.Close()
		return err
	}

	w.file = f
	w.buf = bufio.NewWriterSize(f, 64<<10)
	w.segIndex = index
	w.segBytes = fileHeaderSize
	return nil
}

// syncDir fsyncs a directory, which is what makes a file's creation durable.
//
// The error is returned rather than swallowed. Some filesystems refuse to sync a
// directory, and the temptation is to ignore that so those platforms keep
// working — but ignoring it silently downgrades the guarantee the call exists to
// provide. A loud failure on an unusual filesystem beats a quiet loss of
// durability on every one.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return fmt.Errorf("wal: open dir for fsync: %w", err)
	}
	defer d.Close()
	if err := d.Sync(); err != nil {
		return fmt.Errorf("wal: fsync dir: %w", err)
	}
	return nil
}

// fail records the first error and returns it, so every later call reports the
// same cause rather than a confusing downstream symptom.
func (w *Writer) fail(err error) error {
	if w.failed == nil {
		w.failed = err
	}
	return w.failed
}

// flushLoop drives SyncInterval. A timer rather than a check inside Append,
// because the write that most needs flushing is the last one before the traffic
// stops — exactly the one no subsequent Append would ever come along to trigger.
func (w *Writer) flushLoop() {
	defer close(w.done)

	t := time.NewTicker(w.opts.SyncInterval)
	defer t.Stop()

	for {
		select {
		case <-w.stop:
			return
		case <-t.C:
			w.mu.Lock()
			if w.closed || w.failed != nil {
				w.mu.Unlock()
				continue
			}
			// Flushed under the lock, fsynced outside it: appends used to wait
			// out every tick's fsync, milliseconds each. Nobody is waiting on
			// this tick to report to, so an error is held — failing the writer —
			// for the next Append, Sync or Close to return.
			_ = w.syncUnlocking()
		}
	}
}
