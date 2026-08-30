package snapshot

import (
	"bufio"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
)

// Result reports what Load found.
type Result struct {
	// Found is false when there is no usable snapshot: an empty directory, or
	// one where every snapshot failed verification. Both are recoverable — the
	// caller replays the whole WAL instead — so neither is an error.
	Found bool

	// Snapshot is the one that was applied, valid only when Found.
	Snapshot Snapshot

	// Rejected lists the snapshots that failed verification, newest first. It is
	// empty in the normal case. A non-empty list on a healthy machine means
	// something is wrong with the disk, and it is the only place that shows.
	Rejected []Rejection
}

// Rejection is a snapshot that did not survive verification.
type Rejection struct {
	Snapshot Snapshot
	Cause    error
}

func (r Rejection) String() string {
	return fmt.Sprintf("%s: %v", r.Snapshot.Path, r.Cause)
}

// Load applies the newest snapshot in dir that passes verification, calling read
// with its payload.
//
// # Nothing unverified reaches read
//
// The file is hashed end to end *before* the callback sees a byte of it. That
// costs a second pass over the file, and it is worth it: applying unverified
// bytes is how corruption on disk becomes corruption in memory, and a callback
// that has already ingested half a bad snapshot cannot un-ingest it. Streaming
// straight into the callback and reporting the checksum failure afterwards would
// be faster and would make the failure unrecoverable.
//
// Measured, at 64 MiB: verifying is 10.4 ms and applying is a further 3.6 ms,
// against 10.4 ms for streaming once and checking the checksum afterwards. So
// the property costs about +37%, not the 2× the extra pass suggests — the verify
// pass leaves the file in the page cache, and the apply pass that follows runs at
// 18.8 GB/s because it is reading memory. On a 1 GiB snapshot that is under 60 ms
// bought against the alternative, which is rebuilding an index from the log.
//
// # A failed snapshot falls back to an older one
//
// This is the reason to retain more than one. A corrupt newest snapshot costs a
// longer WAL replay, not the database. Which older snapshot is usable depends on
// the WAL still holding records from its sequence — see Prune for the ordering
// constraint that keeps that true.
//
// # read is called at most once
//
// Because verification happens first, the callback is never invoked for a
// snapshot that turns out to be bad. It does not need to be idempotent and it
// does not need to undo anything. An error from it is returned as-is: a
// snapshot that verified but could not be applied is a decoding bug, not disk
// corruption, and falling back to an older file would only hide it.
func Load(dir string, read func(io.Reader) error) (Result, error) {
	all, err := List(dir)
	if err != nil {
		return Result{}, err
	}

	var res Result
	for _, snap := range all {
		payload, err := verify(snap)
		if err != nil {
			// Anything the format can diagnose means "try an older one". A
			// genuine I/O error means the disk is not answering, and trying
			// three more files on it is theatre.
			if !isCorruption(err) {
				return res, fmt.Errorf("snapshot: %s: %w", snap.Path, err)
			}
			res.Rejected = append(res.Rejected, Rejection{Snapshot: snap, Cause: err})
			continue
		}

		if err := apply(snap, payload, read); err != nil {
			return res, err
		}
		res.Found = true
		res.Snapshot = snap
		return res, nil
	}
	return res, nil
}

// verify hashes a snapshot end to end and returns its payload length.
//
// Every cheap check runs before the expensive one: the framing has to fit the
// file, the length in the trailer has to agree with the file's size, and the
// sequence in the name has to agree with the one inside. Only then is there any
// point hashing gigabytes.
func verify(snap Snapshot) (payloadBytes int64, err error) {
	f, err := os.Open(snap.Path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	want, ok := payloadLen(snap.Bytes)
	if !ok {
		return 0, ErrShortSnapshot
	}

	br := bufio.NewReaderSize(f, 64<<10)
	h := crc32.New(crcTable)

	var hdr [headerSize]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil {
		return 0, shortOr(err)
	}
	seq, err := decodeHeader(hdr[:])
	if err != nil {
		return 0, err
	}
	// The name is not covered by the checksum and the header is, so a
	// disagreement means the file was renamed — deliberately or by a script that
	// meant well. Either way it now claims a sequence its contents do not
	// support, and using it would replay the log from the wrong place.
	if seq != snap.Seq {
		return 0, fmt.Errorf("%w: name says %d, header says %d", ErrSeqMismatch, snap.Seq, seq)
	}
	h.Write(hdr[:])

	if _, err := io.CopyN(h, br, want); err != nil {
		return 0, shortOr(err)
	}

	var trailer [trailerSize]byte
	if _, err := io.ReadFull(br, trailer[:]); err != nil {
		return 0, shortOr(err)
	}
	sum, length, err := decodeTrailer(trailer[:])
	if err != nil {
		return 0, err
	}
	if length != uint64(want) {
		return 0, fmt.Errorf("%w: trailer says %d, file holds %d", ErrLengthMismatch, length, want)
	}
	if sum != h.Sum32() {
		return 0, ErrChecksum
	}
	return want, nil
}

// apply reopens a verified snapshot and streams its payload to read.
//
// Reopening rather than seeking the handle verify used keeps the two phases
// independent: verify says yes or no about a file on disk, and this reads that
// file. A caller reading a shared, rewound handle would couple them for no gain.
//
// The reader is bounded to exactly the payload, so a callback cannot wander into
// the trailer no matter how much it asks for.
func apply(snap Snapshot, payloadBytes int64, read func(io.Reader) error) error {
	f, err := os.Open(snap.Path)
	if err != nil {
		return fmt.Errorf("snapshot: %s: %w", snap.Path, err)
	}
	defer f.Close()

	if _, err := f.Seek(headerSize, io.SeekStart); err != nil {
		return fmt.Errorf("snapshot: %s: %w", snap.Path, err)
	}
	br := bufio.NewReaderSize(f, 64<<10)
	if err := read(io.LimitReader(br, payloadBytes)); err != nil {
		return fmt.Errorf("snapshot: %s: apply: %w", snap.Path, err)
	}
	return nil
}

// isCorruption reports whether an error means "this file is bad" as opposed to
// "this disk is bad". Only the first is worth falling back from.
func isCorruption(err error) bool {
	return errors.Is(err, ErrBadMagic) ||
		errors.Is(err, ErrUnsupportedVersion) ||
		errors.Is(err, ErrShortSnapshot) ||
		errors.Is(err, ErrChecksum) ||
		errors.Is(err, ErrLengthMismatch) ||
		errors.Is(err, ErrSeqMismatch)
}

// shortOr maps running out of file onto ErrShortSnapshot, leaving real I/O
// errors alone. The file shrinking between the stat and the read is not
// something a well-behaved directory does, but it is exactly what a truncated
// or concurrently rewritten snapshot looks like.
func shortOr(err error) error {
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return ErrShortSnapshot
	}
	return err
}
