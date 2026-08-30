package wal

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
)

// segmentReader scans one segment file, record by record.
//
// It is deliberately not exported. Reading a *segment* is not a thing callers
// have any reason to do — recovery is about a directory, and that is what Replay
// takes. Keeping this unexported means the tolerance rules below live in exactly
// one place instead of becoming an API somebody depends on the details of.
//
// # Nothing is read on the strength of an unverified length
//
// A record announces its own payload length, and that length arrives *before*
// the checksum that would prove it. A flipped bit there is the failure that
// actually hurts: it turns a bad read into an arbitrary allocation. So the file
// size is taken up front and every length is refused unless it fits both what
// remains of the file and the configured cap — before a byte of payload is read
// and before a byte of memory is reserved for it.
type segmentReader struct {
	f  *os.File
	br *bufio.Reader

	// size is the file's length, read once at open. It is what turns "this
	// record claims 3 GB" into a cheap refusal rather than a 3 GB allocation.
	size int64
	// off is where the next record starts — and, once the scan stops, the
	// offset everything after is discarded from.
	off int64

	maxRecord int

	hdr [recordHeaderSize]byte
	// buf is reused across records, which is what keeps replay allocation-free
	// per record. It is also why a payload handed to a caller is only valid
	// until the next record is read. See Replay.
	buf []byte
	rec Record

	// damage is why the scan stopped, or nil if it reached the end of the file
	// cleanly. It is not an error return: a damaged tail is the *expected*
	// outcome of power loss, and whether it is tolerable is Replay's call.
	damage error
}

// openSegmentReader opens a segment and validates its file header.
//
// A returned error is fatal — the file is not one of ours, or is a version this
// build must not guess at. Damage *within* the file is not an error here.
func openSegmentReader(path string, maxRecord int) (*segmentReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}

	r := &segmentReader{
		f: f,
		// Matching the writer's buffer size: the same bytes, read back the same
		// way they were laid down.
		br:        bufio.NewReaderSize(f, 64<<10),
		size:      info.Size(),
		off:       fileHeaderSize,
		maxRecord: maxRecord,
	}

	// A file too short to hold even a header is a crash between creating the
	// segment and writing into it — os.OpenFile and the header Write are two
	// syscalls, and power can be lost between them. There is nothing in such a
	// file to lose, so it is damage at offset zero rather than a reason to
	// refuse the whole log.
	if r.size < fileHeaderSize {
		r.damage = ErrShortRecord
		r.off = 0
		return r, nil
	}

	var hdr [fileHeaderSize]byte
	if _, err := io.ReadFull(r.br, hdr[:]); err != nil {
		f.Close()
		return nil, err
	}
	if _, err := decodeFileHeader(hdr[:]); err != nil {
		f.Close()
		return nil, err
	}
	return r, nil
}

// next advances to the next record, leaving it in r.rec. It returns false when
// the scan ends, for either reason: cleanly at the end of the file, or on
// damage, which is left in r.damage.
//
// A returned error is an I/O failure — the disk, not the log. That distinction
// is the point of the signature: a torn record means "this segment ends here",
// while a read error means the machine cannot be trusted to answer questions
// about any of it.
func (r *segmentReader) next() (bool, error) {
	if r.damage != nil || r.off >= r.size {
		return false, nil
	}

	if r.size-r.off < recordHeaderSize {
		return r.tear(ErrShortRecord), nil
	}
	torn, err := r.readInto(r.hdr[:])
	if err != nil {
		return false, err
	}
	if torn {
		return r.tear(ErrShortRecord), nil
	}

	_, typ, seq, length, err := decodeRecordHeader(r.hdr[:])
	if err != nil {
		// Unreachable: the slice is exactly recordHeaderSize. Handled anyway
		// rather than ignored, because "unreachable" is a claim about today.
		return false, fmt.Errorf("wal: %s: %w", r.f.Name(), err)
	}

	// Both guards run before anything is read or allocated, and they are ordered
	// most-informative-first: a length that overruns the file is what a torn tail
	// looks like, which is the common case, while one that overruns the cap is a
	// bit-flip in a file that is otherwise the right size.
	body := r.off + recordHeaderSize
	if int64(length) > r.size-body {
		return r.tear(ErrShortRecord), nil
	}
	if int64(length) > int64(r.maxRecord) {
		return r.tear(ErrRecordTooLarge), nil
	}

	if cap(r.buf) < int(length) {
		r.buf = make([]byte, length)
	}
	payload := r.buf[:length]
	torn, err = r.readInto(payload)
	if err != nil {
		return false, err
	}
	if torn {
		return r.tear(ErrShortRecord), nil
	}

	if !verifyChecksum(r.hdr[:], payload) {
		return r.tear(ErrChecksum), nil
	}
	// Checked after the checksum on purpose. Zero-filled space decodes as type
	// zero, and reporting that as an unknown type would describe the symptom
	// while hiding the cause; the checksum names it correctly as corruption.
	// Getting here means a record that is *intact* and carries a type this build
	// does not know — a format problem, not a damaged disk.
	if !typ.valid() {
		return r.tear(ErrInvalidType), nil
	}

	r.rec = Record{Type: typ, Seq: seq, Payload: payload}
	r.off = body + int64(length)
	return true, nil
}

// readInto fills p, separating the two ways that can fail. Running out of file
// is damage — the segment is smaller than its own records claim. Anything else
// is an I/O error, and no amount of log structure repairs a disk.
func (r *segmentReader) readInto(p []byte) (torn bool, err error) {
	if _, err := io.ReadFull(r.br, p); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return true, nil
		}
		return false, fmt.Errorf("wal: read %s: %w", r.f.Name(), err)
	}
	return false, nil
}

// tear ends the scan and records the cause, leaving r.off at the start of the
// damaged record — the point everything after is discarded from. It returns
// false so the scan can `return r.tear(...), nil` in one line.
func (r *segmentReader) tear(cause error) bool {
	r.damage = cause
	return false
}

func (r *segmentReader) close() error { return r.f.Close() }
