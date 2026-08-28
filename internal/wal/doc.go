// Package wal is an append-only write-ahead log: the thing that makes GoVecDB
// survive a crash.
//
// The ordering rule is the whole point. A write is appended here FIRST, and only
// then applied to the in-memory graph. Reverse those two and a crash between
// them means acknowledging a write that no longer exists. The graph is derived
// state — it can always be rebuilt by replaying this log, and it is never the
// source of truth.
//
// # The format
//
// A segment file opens with an 8-byte header and then holds records back to
// back, each self-describing:
//
//	file:    magic "GVWL" (4) | version (2) | reserved (2)
//	record:  crc32c (4) | type (1) | seq (8) | length (4) | payload
//
// Three properties of that layout carry their weight:
//
//   - **Versioned from the first byte written.** A log format without a version
//     field cannot be migrated later, only abandoned.
//   - **A checksum per record, covering everything after it** — type, seq,
//     length and payload. Power loss mid-write leaves a torn record at the tail,
//     and recovery has to be able to tell that from a good one. Checksumming the
//     length field matters as much as the payload: a corrupt length is what turns
//     a bad read into a huge allocation.
//   - **The payload is opaque here.** This package moves bytes durably and knows
//     nothing about vectors, so it stays testable without the index and reusable
//     for whatever else needs logging.
//
// # Segments
//
// The log is a directory of numbered segments (`wal-000001.log`), not one file,
// and that is load-bearing rather than tidy: once a snapshot at sequence N is
// durable, every segment below N can be deleted outright. Truncating a single
// growing file is not something a filesystem does well, so segment rotation is
// here from the first commit rather than retrofitted.
//
// Opening an existing directory always starts a NEW segment. See Open for why.
//
// # Durability is a knob
//
// SyncAlways / SyncInterval / SyncNever trade durability against throughput, and
// the zero value is SyncAlways — a caller who says nothing gets the safe answer,
// not the fast one.
//
// The package is split one responsibility per file:
//
//	record.go   The wire format: encode, decode, and the checksum.
//	segment.go  Segment file naming and discovery.
//	options.go  Options, the sync policy, and their defaults.
//	writer.go   The append-only writer, rotation, and the sync policies.
//	wal.go      The WAL interface callers depend on, plus a no-op implementation.
//	errors.go   Sentinel errors callers match on.
package wal
