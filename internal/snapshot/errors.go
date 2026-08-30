package snapshot

import "errors"

var (
	// ErrBadMagic means the file does not begin with this format's magic bytes —
	// almost always a path pointing somewhere unintended rather than corruption.
	ErrBadMagic = errors.New("snapshot: not a snapshot file")

	// ErrUnsupportedVersion means the snapshot was written by a format version
	// this build does not know how to read. Refusing is the point of carrying
	// the version field; guessing at an unknown layout would be worse.
	ErrUnsupportedVersion = errors.New("snapshot: unsupported format version")

	// ErrShortSnapshot means the file is smaller than its own framing requires.
	// Because writes are atomic, this is not a half-written snapshot — it is a
	// file that was damaged or replaced after it was renamed into place.
	ErrShortSnapshot = errors.New("snapshot: truncated file")

	// ErrChecksum means the contents do not match the checksum in the trailer.
	ErrChecksum = errors.New("snapshot: checksum mismatch")

	// ErrLengthMismatch means the trailer's payload length disagrees with the
	// size of the file holding it. It is the cheap check that catches a file
	// which grew or shrank before the checksum has to read the whole thing.
	ErrLengthMismatch = errors.New("snapshot: payload length does not match file size")

	// ErrSeqMismatch means the sequence in the file name disagrees with the one
	// inside the file. The name is not covered by the checksum and the header is,
	// so the header wins and the file is refused: a snapshot that lies about
	// which sequence it represents would replay the log from the wrong place.
	ErrSeqMismatch = errors.New("snapshot: sequence in the name does not match the header")

	// ErrKeepTooFew is returned by Prune when asked to keep fewer than one
	// snapshot. Deleting the last one is not retention, it is deletion, and it
	// would leave recovery with nothing but a log nobody can truncate.
	ErrKeepTooFew = errors.New("snapshot: retention must keep at least one snapshot")
)
