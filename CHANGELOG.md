# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Driven by a field report from the first application to run GoVecDB end to end at
scale — millions of 512-dimension vectors across sixteen collections. Each item
below names what it fixes.

### Added

- **`DB.GetBatch`, `DB.Scan` and `DB.Range` — a database can be enumerated.**
  `Get` needed an id the caller already had, so anything rebuild-shaped —
  resharding, re-embedding, copying a collection under a new spec — needed a
  ledger kept outside the database. `GetBatch(ids)` reads many under one lock;
  `Scan(after, limit)` pages in id order with a cursor; `Range(fn)` walks
  everything, sorted once and read a page at a time with no lock held across
  the walk, so writers are never stalled behind it. The `Index` interface gains
  `IDs`, and `hnsw.Graph` gains `IDs`.

### Changed

- **Non-finite float metadata is refused.** `Add` and `AddBatch` now return
  `ErrInvalidMetadata` for a NaN or ±Inf metadata value, as they already did for
  a vector component. A stored NaN was invisible to every range filter, and it is
  a value JSON cannot spell, so an embedded deployment could hold data the REST
  API could never return. Values already on disk still decode: a log written
  before this check replays unchanged.

## [1.1.1] - 2026-09-29

A bug fix and a performance release. No exported API changed, no on-disk format
changed, and every recall baseline holds at its locked value.

### Fixed

- **`AddBatch` fsynced once per vector under `SyncAlways`.** Every REST upsert
  goes through it, so a 1,000-vector POST paid 1,000 fsyncs — about four seconds
  on a laptop SSD — while looking like a bulk write. The batch is now logged
  through a new internal `wal.Writer.AppendBatch` and synced once, after the last
  record: 100 vectors cost 47 ms end to end, against 0.56–0.70 s before. Nothing
  is acknowledged before it is durable, exactly as before. A batch whose *write*
  fails now applies none of itself to the index (it used to apply a prefix); the
  database is read-only either way, and any prefix that reached the disk is
  replayed on the next start.

### Changed

- **Distance kernels use eight accumulators, not four.** Go emits scalar FMAs,
  and four dependency chains left the kernel waiting on FMA latency. Dot product
  is 27.5 → 18.4 ns at dimension 128 and 239 → 116 ns at 768 (Apple M4). Search
  is 23% faster at 10k × 128 and 44–46% faster at dimensions 768 and 1536.
- **Insert allocates almost nothing.** Pruning reuses pooled scratch and
  rewrites a neighbor list in place, `slices.SortFunc` replaces the reflection
  swapper of `sort.Slice`, and a list is sized once to `maxConn+1` instead of
  doubling on overflow. Insert: 666 → 479 µs, **208 → 6 allocs**, 50 KB → 1 KB.
- **Search is 1 alloc/op, down from 2**, filtered or not — the layer search now
  returns a slice owned by its pooled state. The locked baseline moves with it.

Measured on an Apple M4 Max; amd64 compiles to the same bounds-check-free loop
but has not been measured.

## [1.1.0] - 2026-09-07

GoVecDB can now be run as a service. Nothing about the library changed to make
that possible: the new packages sit strictly above `DB`, which does not know they
exist. Embedding it behaves exactly as it did in 1.0.0.

Still **zero third-party dependencies** — `go.mod` has no `require` block and
there is no `go.sum`. The server is `net/http` and `encoding/json`.

### Added

**Collections** (`service`) — many independent databases in one directory.

- Each collection has its own dimension, metric, index and durability policy,
  created at runtime and stored in a `collection.json` next to its data. The
  structural options are written down because reopening a database under
  different ones is refused; the *effective* values are recorded, never the zero
  that meant "default", so a later change to this module's defaults cannot
  silently reshape an existing index.
- Collections load on demand. Starting the process reads no indexes; the first
  request for one loads its snapshot and replays its log.
- `MaxOpen` and `IdleTimeout` close what is not being used. `Use(name, fn)`
  borrows a database for the length of a callback, and only a collection with no
  borrowers can be evicted or dropped — so a search in flight is never closed
  underneath.
- `ValidateName` is a security boundary: a collection name becomes a directory
  name, which vector ids never do.

**A REST API** (`httpapi`) — an `http.Handler` over a collection manager.

- Collection lifecycle, add/get/delete, search with filters, snapshot and
  compact, plus `/healthz`, `/readyz` and a Prometheus `/metrics`.
- A JSON filter format covering every operator the library has, with the same
  semantics — including that every comparison is false on an absent key.
- One error shape with a stable `code`, including for the 404 and 405 that
  `net/http` would otherwise answer in plain text.
- Bodies capped, unknown fields rejected, filter trees depth-limited, and 5xx
  responses that say "internal error" while the detail goes to the log.
- Optional bearer-token authentication, constant-time compared, exempting only
  the two health probes.

**A daemon** (`cmd/govecdbd`) and a `Dockerfile`.

- Binds `127.0.0.1` unless told otherwise, and reads its token from
  `GOVECDB_AUTH_TOKEN` rather than a flag — a flag lands in `ps` output.
- Graceful shutdown that fails `/readyz` first, optionally drains, then waits out
  requests in flight before closing any collection.
- Optional TLS, structured logging, and a `FROM scratch` image.

**In the root package**: `SyncPolicy.String()`, so a layer that writes a policy
into a config file or an API response does not invent its own spelling. `Metric`
already had one.

### Documentation

- [docs/SERVICE.md](docs/SERVICE.md) — the service guide: API reference,
  configuration, operating notes, security, and the module decision.
- READMEs for `service/` and `httpapi/`, matching the `internal/` convention.

### Decided

**The library and the HTTP API stay in one module; gRPC and Raft will be a
separate one.** `net/http` and `encoding/json` are standard library and cost
nothing to depend on. gRPC and Raft are not, and retrofitting a module split
after a server exists means moving every import path — so the decision was made
before the server was written, as the v2 plan asked. See
[the reasoning](docs/SERVICE.md#the-module-decision).

### Known limitations

Everything from 1.0.0 still applies. New with the service:

- **One process.** No clustering and no replication; that is v2 item 8.
- **No gRPC.** See above.
- **One shared bearer token**, no users and no roles. Anything richer belongs in
  a proxy in front, which is also where TLS termination, rate limiting and audit
  belong.
- **`POST .../compact` stops the world** for that collection, exactly as
  `Compact()` does. Online compaction is v2 item 2.
- **A collection's dimension, metric and M cannot be changed.** They are
  structural. Create a new collection and re-index.

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

[Unreleased]: https://github.com/khambampati-subhash/govecdb/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/khambampati-subhash/govecdb/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/khambampati-subhash/govecdb/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/khambampati-subhash/govecdb/releases/tag/v1.0.0
