# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet. See [the v2 scope](docs/MIGRATION.md#v2-scope) for what is planned
and in what order.

## [1.0.0] - 2026-09-06

First release. GoVecDB is an embeddable vector database in pure Go — no CGO, no
third-party dependencies, no server.

This is the result of a ground-up rebuild. The previous ~45,700-line
implementation (clustering, gRPC, REST, several competing index variants) was
removed and rewritten one subsystem at a time; it remains in git history. See
[docs/MIGRATION.md](docs/MIGRATION.md) for the record of what was built and why.

### Added

**The index** — HNSW, written from scratch and verified against brute force.

- Concurrent reads: `Search` holds a read lock and per-traversal scratch comes
  from a pool, so queries scale across cores. Measured ~13× on 16 threads.
- `Delete` is a tombstone; dead slots keep routing but never answer.
- `Insert` is an upsert — a second insert under a live id replaces the vector.
- `Compact()` rebuilds over the live vectors; `Stats().DeadRatio()` is the signal.
- `SuggestedEf` fits the measured `ef ∝ n^0.78` curve, because recall at a fixed
  search width *falls* as a corpus grows.

**Durability** — write to the log first, then apply to the index.

- Write-ahead log with a versioned record format, CRC32C over the type, sequence
  and length as well as the payload, segment rotation, and three fsync policies.
- `Replay` validates every checksum and truncates a torn tail rather than
  failing recovery, because a partial final record is the normal outcome of
  power loss.
- Snapshots: atomic (temp → fsync → rename → fsync the directory), checksummed,
  and keyed by the log sequence they cover, with fallback to an older copy when
  the newest fails verification.
- Log truncation after every snapshot, against the *oldest retained* snapshot —
  so the log stops growing without making the fallback copy unusable.
- Recovery on `Open`: newest valid snapshot, then replay the records above it.

**The public API** — `Open`, `Add`, `AddBatch`, `Get`, `Delete`, `Search`,
`Snapshot`, `Compact`, `Len`, `Stats`, `Sync`, `Close`.

- Metadata attached to vectors, restricted to `string`, `bool`, `int64` and
  `float64` — a closed set, because decoding is where disk bytes become live
  objects.
- Metadata filtering: `Eq`, `Ne`, `Lt`, `Lte`, `Gt`, `Gte`, `In`, `Exists`,
  `And`, `Or`, `Not`. Filters are applied *during* the graph traversal, so a
  filtered search returns `K` results rather than however many of the nearest
  `K` happened to match.
- Fail-closed: a durability failure makes the database permanently `ErrReadOnly`.
  Reads keep working; writes stop, because nothing about the next one can be
  promised.
- Input validation as a security boundary. Vector values must be finite — one
  NaN compares false against everything and poisons the ordering the index rests
  on — and every per-call limit is bounded and configurable through `WithLimits`.

### Known limitations

Stated here rather than discovered later:

- **Linux and macOS only.** The WAL and snapshot store fsync the containing
  directory, which is not a portable operation on Windows. Untested there.
- **`Compact()` stops the world.** It holds the write lock for a full rebuild.
  The caller chooses the moment; the database never triggers it.
- **Concurrent writers serialize.** Reads scale, writes do not.
- **One database is one index.** Collections are v2.
- **No observability seam.** A torn log tail found during recovery is repaired
  correctly and reported to nobody.
- **One writer per directory, enforced in-process only.** Cross-process locking
  is a deliberate gap: a lock file left by a crash blocks a restart that should
  have succeeded.
- **A highly selective filter approaches a full scan.** Past roughly one vector
  in a hundred, a scan over the metadata is the better tool.

[Unreleased]: https://github.com/khambampati-subhash/govecdb/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/khambampati-subhash/govecdb/releases/tag/v1.0.0
