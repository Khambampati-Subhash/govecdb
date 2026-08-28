package wal

import (
	"bufio"
	"fmt"
	"os"
	"sync"
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

	if w.closed {
		return 0, ErrClosed
	}
	if w.failed != nil {
		return 0, w.failed
	}

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

	if w.opts.SyncPolicy == SyncAlways {
		if err := w.syncLocked(); err != nil {
			return 0, w.fail(err)
		}
	}
	return seq, nil
}

// Sync flushes buffered data and fsyncs the current segment. It is what
// SyncAlways calls on every append and what a caller reaches for before doing
// something that must not outlive the log, such as acknowledging a batch.
func (w *Writer) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return ErrClosed
	}
	if w.failed != nil {
		return w.failed
	}
	if err := w.syncLocked(); err != nil {
		return w.fail(err)
	}
	return nil
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
		w.file.Close()
		return w.failed
	}
	syncErr := w.syncLocked()
	closeErr := w.file.Close()
	if syncErr != nil {
		return syncErr
	}
	return closeErr
}

// LastSeq reports the sequence number most recently assigned, or FirstSeq-1 when
// nothing has been appended. Recovery and the checkpointer both need to know
// where the log has got to.
func (w *Writer) LastSeq() uint64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.nextSeq - 1
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
func (w *Writer) syncLocked() error {
	if err := w.buf.Flush(); err != nil {
		return fmt.Errorf("wal: flush: %w", err)
	}
	if err := w.file.Sync(); err != nil {
		return fmt.Errorf("wal: fsync: %w", err)
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
func (w *Writer) rotate() error {
	if err := w.syncLocked(); err != nil {
		return err
	}
	if err := w.file.Close(); err != nil {
		return fmt.Errorf("wal: close segment: %w", err)
	}
	return w.openSegment(w.segIndex + 1)
}

// openSegment creates a segment file and writes its header.
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

	w.file = f
	w.buf = bufio.NewWriterSize(f, 64<<10)
	w.segIndex = index
	w.segBytes = fileHeaderSize
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
			if !w.closed && w.failed == nil {
				if err := w.syncLocked(); err != nil {
					// Nobody is waiting on this tick to report to, so the error
					// is held for the next Append, Sync or Close to return.
					w.fail(err)
				}
			}
			w.mu.Unlock()
		}
	}
}
