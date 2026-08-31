// Package govecdb is an embeddable vector database in pure Go: it stores
// embeddings and answers "what is most similar to this?".
//
//	db, err := govecdb.Open("data", govecdb.WithDimension(768))
//	defer db.Close()
//
//	db.Add(govecdb.Vector{ID: "doc-1", Values: embedding, Metadata: govecdb.Metadata{
//		"source": "handbook.pdf",
//		"page":   int64(12),
//	}})
//
//	matches, err := db.Search(govecdb.SearchRequest{Query: query, K: 10})
//
// No CGO, no third-party dependencies, and no server: it runs inside your
// process and stores everything in one directory.
//
// # The ordering rule
//
// Every write is appended to a write-ahead log first and only then applied to
// the in-memory index. Reverse those two and a crash between them acknowledges a
// write that no longer exists. The index is derived state — it can always be
// rebuilt by replaying the log — and it is never the source of truth.
//
// Durability is a knob, not a constant. The zero value is SyncAlways, so a
// caller who configures nothing gets the safe answer rather than the fast one;
// see SyncPolicy for what each setting costs and, more importantly, what each
// one actually promises.
//
// # Recovery
//
// Open loads the newest snapshot that passes its checksum, then replays the log
// records written after it. A snapshot is what bounds startup time: without one,
// recovery rebuilds the index at roughly 700 µs per vector, and with one it
// loads a graph at gigabytes per second. Snapshots are taken by calling Snapshot
// or by setting WithSnapshotInterval; nothing is written automatically by
// default, because how long a restart may take is the caller's decision.
//
// Each snapshot also deletes the log segments it has made redundant, so the log
// does not grow forever. That is measured against the *oldest retained* snapshot
// rather than the newest, and only after verifying it reads: retaining more than
// one snapshot is what makes a corrupt one survivable, and it only survives if
// the log still reaches back far enough to replay on top of it. WithSnapshotsKept
// is therefore also the knob deciding how much log is kept.
//
// # What this package validates, and why it bothers
//
// A library does not know where its arguments came from. An id, a K, a metadata
// map may all be relaying input from somewhere the process does not trust, and
// each one multiplies an allocation — so every one of them is bounded, and the
// bounds are configurable through WithLimits without being removable.
//
// Two checks are worth calling out because their absence would be silent rather
// than loud. Vector values must be finite: a single NaN compares false against
// everything and poisons the ordering the whole index rests on, so searches
// would return wrong answers with no error anywhere. And metadata values are a
// closed set of four types, because decoding metadata is where bytes from a disk
// become live objects, and a decoder that reconstructs arbitrary types from
// names on the wire is a far larger surface than filtering needs.
//
// # Concurrency
//
// A DB is safe for concurrent use. Searches run in parallel with each other and
// with everything except Compact; writes serialize. One process should hold one
// directory — Open refuses a second handle on the same path from the same
// process, and does not attempt to police other processes.
//
// # Not implemented yet
//
// Metadata filtering: metadata is stored, returned with results, and survives
// restarts, but there is no query language over it yet.
//
// Observability: there is no logger or metrics seam, so a torn log tail found
// during recovery is repaired correctly and reported to nobody, and a skipped
// truncation says nothing about why.
//
// Collections: one database is one index. Multiple named collections would sit
// above this rather than change it.
//
// The package is split one responsibility per file:
//
//	vector.go    Vector, SearchRequest, Match, Stats — what callers pass and get.
//	options.go   Open's functional options and their defaults.
//	validate.go  The input boundary: what is checked, and the limits.
//	index.go     The Index interface and the HNSW adapter behind it.
//	codec.go     Domain encoding: log payloads and the snapshot payload.
//	db.go        The DB facade and its lifecycle.
//	recovery.go  Rebuilding state on Open: snapshot first, then the log.
//	errors.go    Sentinel errors callers match on.
package govecdb
