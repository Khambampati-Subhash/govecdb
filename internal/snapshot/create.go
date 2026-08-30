package snapshot

import (
	"bufio"
	"fmt"
	"hash/crc32"
	"io"
	"os"
)

// Create writes a snapshot covering WAL sequence seq, calling write to produce
// the payload.
//
// # It is atomic, and that is what the checksum is not for
//
// The payload goes to a temporary file, which is fsynced, closed, renamed into
// place, and then the *directory* is fsynced so the rename itself survives a
// power loss. A crash at any point leaves either no snapshot or a complete one,
// never a half-written file wearing a finished name.
//
// That ordering is what lets the checksum mean something narrower and more
// useful: it guards against bit rot and against a file being tampered with after
// the fact, not against interrupted writes. If atomicity were left to the
// checksum instead, every crash would produce a plausible-looking snapshot that
// only fails at the moment recovery needs it.
//
// The directory fsync is the step that is easy to leave out and impossible to
// notice missing. Without it the rename can still be in the filesystem's journal
// and not on the platter, so a crash leaves a snapshot that exists under neither
// name — the temporary file already unlinked, the final one not yet durable.
//
// # write is called exactly once
//
// It receives a buffered writer and may write as much as it likes. An error from
// it aborts the snapshot and removes the temporary file, so a payload that could
// not be produced never becomes a snapshot that cannot be trusted.
//
// The caller is responsible for the payload being a *consistent* point in time —
// this package writes what it is handed. What "consistent" costs is the caller's
// problem too: holding a read lock for the whole write, or building the payload
// under a lock and streaming it after.
func Create(dir string, seq uint64, write func(io.Writer) error) (Snapshot, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: create dir: %w", err)
	}

	f, err := os.CreateTemp(dir, tempPattern)
	if err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: create temp: %w", err)
	}
	tmp := f.Name()

	// Any failure past this point must leave nothing behind. An orphaned
	// temporary is not dangerous — it can never be mistaken for a snapshot — but
	// it is litter that only a later Prune would clear.
	committed := false
	defer func() {
		if !committed {
			f.Close()
			os.Remove(tmp)
		}
	}()

	bw := bufio.NewWriterSize(f, 64<<10)
	h := crc32.New(crcTable)

	var hdr [headerSize]byte
	encodeHeader(hdr[:], seq)
	if _, err := bw.Write(hdr[:]); err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: write header: %w", err)
	}
	h.Write(hdr[:]) // hash.Hash never returns an error

	// counted tees the payload into the hash and tracks its length, so the
	// snapshot is hashed as it is written rather than by reading it back.
	c := &counted{w: bw, h: h}
	if err := write(c); err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: write payload: %w", err)
	}

	var trailer [trailerSize]byte
	encodeTrailer(trailer[:], h.Sum32(), c.n)
	if _, err := bw.Write(trailer[:]); err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: write trailer: %w", err)
	}

	if err := bw.Flush(); err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: flush: %w", err)
	}
	// Flush moved the bytes into the kernel; Sync moves them onto the disk.
	// Renaming after only the first is how a snapshot becomes a file full of
	// zeros that passes every name check and no checksum.
	if err := f.Sync(); err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: fsync: %w", err)
	}
	if err := f.Close(); err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: close: %w", err)
	}

	path := filePath(dir, seq)
	if err := os.Rename(tmp, path); err != nil {
		return Snapshot{}, fmt.Errorf("snapshot: rename: %w", err)
	}
	committed = true

	if err := syncDir(dir); err != nil {
		return Snapshot{}, err
	}

	return Snapshot{
		Seq:   seq,
		Path:  path,
		Bytes: int64(headerSize) + int64(c.n) + int64(trailerSize),
	}, nil
}

// counted writes through to the file and the hash at once, counting the bytes.
//
// It exists so the payload is hashed on the way out. The alternative — writing
// the file, then reading it back to checksum it — would double the I/O of every
// snapshot to learn something the writer already knew.
type counted struct {
	w io.Writer
	h io.Writer
	n uint64
}

func (c *counted) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	if n > 0 {
		c.h.Write(p[:n])
		c.n += uint64(n)
	}
	return n, err
}

// syncDir fsyncs a directory, which is what makes a rename durable.
//
// The error is returned rather than swallowed. Some filesystems refuse to sync a
// directory, and the temptation is to ignore that so those platforms keep
// working — but ignoring it silently downgrades the guarantee this whole
// function exists to provide. A loud failure on an unusual filesystem is better
// than a quiet loss of atomicity on every one.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return fmt.Errorf("snapshot: open dir for fsync: %w", err)
	}
	defer d.Close()
	if err := d.Sync(); err != nil {
		return fmt.Errorf("snapshot: fsync dir: %w", err)
	}
	return nil
}
