// Package snapshot stores point-in-time state durably, so recovery does not
// have to start from the beginning of the write-ahead log.
//
// # What a snapshot is for
//
// The WAL alone is a complete recovery story, and a slow one: a log that has
// been appended to for a week takes a week's worth of replaying. A snapshot is
// the fixed point that makes the log finite. It records the state as of WAL
// sequence N, and recovery becomes "load the snapshot, then replay the log from
// N+1" — which also makes every WAL segment below N deletable, which is the only
// thing that stops a log growing forever.
//
// So the sequence number is not metadata attached to a snapshot. It is the
// entire reason the file exists, and it is why it is checksummed as carefully as
// the payload: a snapshot whose sequence is wrong by one replays the log from
// the wrong place and silently loses or duplicates a write.
//
// # The format
//
//	header:   magic "GVSS" (4) | version (2) | reserved (2) | seq (8)
//	payload:  opaque, streamed
//	trailer:  crc32c (4) | payload length (8)
//
// The checksum lives in a trailer rather than the header because the payload is
// *streamed*: a snapshot is gigabytes where a WAL record is kilobytes, so
// nothing here ever holds the whole thing in memory, and a header checksum would
// mean either buffering it all or seeking back to patch it.
//
// The payload is opaque, exactly as it is in the WAL. This package moves bytes
// durably and knows nothing about vectors or graphs, which keeps it testable
// without the index and honest about where the format boundary is.
//
// # Two properties worth stating plainly
//
// Writes are atomic. A snapshot is written to a temporary file, fsynced, then
// renamed into place, and the directory is fsynced so the rename itself
// survives. A crash therefore leaves either no snapshot or a complete one —
// never a half-written file that looks finished. That is what makes the
// checksum a guard against bit rot rather than against partial writes.
//
// Nothing unverified reaches a caller. Load hashes a snapshot end to end
// *before* handing a byte of it to the callback, and falls back to an older
// snapshot if that fails. Applying unverified bytes is how corruption on disk
// becomes corruption in memory; see Load for what the extra pass costs.
//
// The package is split one responsibility per file:
//
//	format.go  The wire format: header, trailer, and the checksum.
//	file.go    Snapshot naming, discovery, and Latest.
//	create.go  Writing one atomically.
//	load.go    Verifying and applying the newest that survives verification.
//	prune.go   Retention: how many to keep, and why more than one.
//	errors.go  Sentinel errors callers match on.
package snapshot
