package wal

import "errors"

var (
	// ErrClosed is returned by any operation on a closed WAL. It is not a
	// warning: a caller that keeps appending after Close is losing writes.
	ErrClosed = errors.New("wal: closed")

	// ErrRecordTooLarge is returned when a payload exceeds Options.MaxRecordBytes.
	// The cap exists so that a corrupt length field on the way back in cannot ask
	// for an arbitrary allocation, which means it has to be enforced on the way
	// out too — otherwise a legitimately huge record would be unreadable.
	ErrRecordTooLarge = errors.New("wal: record exceeds the configured maximum")

	// ErrInvalidType is returned for a zero or unknown record type. Zero is
	// deliberately not a valid type: a run of zero bytes is what unwritten or
	// zero-filled space looks like, and it should never decode as a real record.
	ErrInvalidType = errors.New("wal: invalid record type")

	// ErrBadMagic means the file does not begin with this log's magic bytes —
	// almost always a path pointing somewhere unintended rather than corruption.
	ErrBadMagic = errors.New("wal: not a wal segment")

	// ErrUnsupportedVersion means the segment was written by a format version
	// this build does not know how to read. Refusing is the point of the version
	// field; guessing would be worse than failing.
	ErrUnsupportedVersion = errors.New("wal: unsupported format version")

	// ErrShortRecord means the file ended in the middle of a record. At the tail
	// of the last segment this is the *expected* outcome of a power loss, and
	// recovery truncates rather than failing; anywhere else it is corruption.
	ErrShortRecord = errors.New("wal: truncated record")

	// ErrChecksum means a record's contents do not match its checksum.
	ErrChecksum = errors.New("wal: checksum mismatch")
)
