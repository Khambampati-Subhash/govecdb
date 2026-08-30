# GoVecDB v1 — Clean Rebuild

> Branch: `v1-restructure` · Same module path · Embeddable library only.
> Gate: `go build ./... && go vet ./... && go test ./... -race` green after every step.

## What changed about the plan

The original plan on this branch was **salvage & port**: keep the proven legacy
core and clean it in place. That is no longer what we are doing.

Building the HNSW index from scratch (rather than untangling `index/`) produced a
better result than porting would have — 1,075 readable lines, recall 0.999 at dim
32 verified against brute force, and a search path down to 2 allocations. The
old index was ~8,100 lines across three competing graph implementations with
entangled helpers.

So the strategy is now **rebuild from scratch, one subsystem at a time**, using the
old code as a *reference in git history* rather than as a source to copy.

## Current state

```
govecdb/
├── internal/hnsw/     # the entire codebase — index engine, 1,075 lines
├── docs/
├── go.mod             # stdlib only; no require block, no go.sum
├── README.md  CLAUDE.md  CONTRIBUTING.md  LICENSE
```

All legacy packages were deleted in the clean-slate commit. **Nothing is lost** —
`main` still has every line. To consult the old implementation:

```bash
git show main:persist/wal.go
git show main:persist/snapshot.go
git log main --oneline -- persist/
```

### Deleted (recoverable from `main`)

| Category | Packages |
|---|---|
| Superseded by the new HNSW | `index/`, `store/`, `collection/`, `api/`, `filter/`, `persist/` |
| Out of v1 scope (v2) | `cluster/`, `api/rest/`, `proto/`, `client/`, `segment/` |
| Dead / never wired | `utils/`, `diskann/`, `quantization/`, `batch/`, `streaming/`, `accuracy/`, `internal/{benchmark,errors,health,logging,metrics,monitoring}` |
| Tooling for the above | `cmd/`, `demo/`, `examples/`, `deployments/`, `Makefile`, `run_*.sh` |

Deleting these dropped every third-party dependency: raft, gRPC, protobuf, bolt,
chroma-go, onnxruntime, uuid. The module is now pure stdlib.

## Target layout

```
govecdb/
├── vector.go            # public API: Vector, SearchRequest, SearchResult
├── db.go                # DB + Collection interfaces + facade
├── options.go           # functional-options construction
├── errors.go            # exported sentinel errors
├── internal/
│   ├── hnsw/            # index engine — concurrent reads       ✅ done
│   ├── wal/             # write-ahead log — writer + replay     ✅ done
│   ├── snapshot/        # durable store ✅, graph codec next
│   ├── store/           # in-memory vector store
│   ├── filter/          # metadata query engine
│   └── obs/             # Logger + Metrics interfaces, no-op defaults
└── docs/
```

## Ordered execution (each = one green-gated commit)

1. ~~**`internal/hnsw`**~~ — from-scratch index, brute-force recall tests, benchmarks. **Done.**
2. ~~**Concurrent reads**~~ — scratch pooled into `searchState`, graph guarded by an
   `RWMutex`, parallel `Search`. **Done.** Fine-grained write locking is deferred.
3. ~~**Tombstone `Delete`**~~ — dead slots keep routing, results filter them,
   entry re-election. **Done.** Upsert and compaction are the remaining index work.
4. ~~**Upsert**~~ — `Insert` replaces an existing id by tombstoning its slot and
   building a new one; an unchanged vector is an early return. **Done.**
5. ~~**Compaction**~~ — `Compact()` rebuilds over the live vectors and swaps the
   graph in whole; `Stats().DeadRatio()` is the signal, the policy stays outside
   the index. **Done.** Online (non-blocking) compaction is deferred — see below.
6. ~~**`internal/wal`**~~ — record format, append-only writer with segment
   rotation, and `Replay`: a CRC-validating scan that truncates a torn tail and
   carries the sequence forward. **Done.** Checkpointing and segment truncation
   wait for step 7, which is what makes a checkpoint mean anything.
7. **`internal/snapshot`** — the durable **store** is done: atomic writes,
   versioned and checksummed framing keyed by WAL sequence, discovery, corruption
   fallback, and retention. The payload is opaque, so what remains is the **graph
   codec** (`hnsw.Graph` ↔ bytes) and the restore orchestration on top of it.
   **The codec is next.**
8. **`internal/store`** — vector + metadata storage behind a `Store` interface.
9. **`internal/filter`** — metadata query engine, with tests from day one.
10. **Public API** — `vector.go` / `db.go` / `options.go` facade; this is what users import.
11. **Examples + README** for the real API.

### Deferred on purpose, not overlooked

- **Online compaction.** `Compact()` holds the write lock for a full index build:
  2.6 s per 5k×128 vectors at a 25% dead ratio. Building the replacement outside
  the lock means writes landing in the old graph while the new one is built, and
  reconciling them wants a change log and a double-buffered swap — both of which
  the WAL should shape first, since it will already be recording those writes.
- **Fine-grained write locking**, for the same reason: the WAL's ordering
  constraint decides what a finer lock is allowed to do.

## Phase — WAL (done)

Design constraints, all decided and now implemented:

- **Write to the WAL first, then apply to the in-memory graph.** On recovery,
  replay the log to rebuild the graph. The graph is *derived state* — it is never
  the source of truth.
- **The WAL is an interface** (`Append`, `Sync`, `Close`) so the index can be
  constructed with a no-op WAL in tests and benchmarks. `Replay` is deliberately
  **not** on it: recovery runs before a writer exists, so it is a package-level
  function over a directory. Putting it on the interface would have meant opening
  a writer in order to read — creating a segment as a side effect of recovery.
- **Records are versioned** from the first commit. A log format without a version
  byte cannot be migrated later.
- **Checksums per record**, covering type, seq and length as well as the payload.
  Recovery truncates to the last intact record rather than failing outright.
- **`fsync` policy is a knob**, not a hardcode: always / interval / never trade
  durability against throughput. Measured at ~4,000× between the extremes.

Settled while building the reader, and worth carrying into the next phase:

- **A tear is tolerated at the end of *any* segment, not only the last.** `Open`
  always starts a new segment, so a second crash leaves a torn tail in the middle
  of the directory — refusing to read past it would make recovery impossible from
  the shape the writer is designed to produce.
- **Truncation is logical.** Recovery is a read; the damaged bytes stay on disk.
- **Nothing is read on an unverified length.** The file size is taken up front
  and a length is refused against both it and `MaxRecordBytes` before any read.

Still open: writing `TypeCheckpoint` records and deleting segments below one.
No longer blocked — `internal/snapshot` now provides the sequence a checkpoint
points at. The constraint to respect when it lands is in the snapshot phase below:
truncate against the **oldest retained** snapshot, not the newest.

Same bar as the index: small single-responsibility files, comments that explain
*why*, and tests that verify crash recovery rather than assuming it. The tear
tests damage a log by truncating and rewriting it, which reproduces the shapes
power loss leaves behind; the harness that `SIGKILL`s a child mid-write to
reproduce the *timing* is still outstanding.

## Phase — Snapshot (store done, codec next)

The step splits in two, and only the first half is built.

**Done — the durable store.** Atomic writes (temp → fsync → rename → fsync the
directory), versioned and checksummed framing keyed by the WAL sequence it
covers, discovery, fallback to an older snapshot when the newest fails
verification, and retention.

Decided while building it:

- **The payload is opaque**, as it is in the WAL. The package moves bytes durably
  and knows nothing about vectors, which keeps it testable without the index and
  puts the format boundary somewhere defensible.
- **The checksum lives in a trailer**, because the payload is streamed — a
  snapshot is gigabytes where a WAL record is kilobytes, so nothing may hold it
  all in memory.
- **Atomicity comes from the rename, not the checksum.** That lets the checksum
  mean the narrower and more useful thing: bit rot, not interrupted writes.
- **Nothing unverified reaches the caller.** Verification is a separate pass
  before the callback sees a byte; measured at +37% over streaming once, because
  the second pass reads the page cache.
- **WAL truncation must follow the oldest *retained* snapshot**, never the
  newest, or the fallback copy is unusable while still being paid for.
- **No `Snapshotter` interface yet.** There is an implementation but no consumer,
  and the WAL taught the lesson directly: `Replay` was on the interface as a
  promise until writing it showed it did not belong there. It gets defined when
  `db.go` exists and its needs are known.

**Next — the graph codec.** Turning a `hnsw.Graph` into bytes and back. It is a
real format decision, not plumbing: writing neighbour lists to disk freezes
HNSW's internal representation the way `TestLayoutIsFrozen` freezes the WAL's, so
it gets its own slice rather than being smuggled in behind a blob store.

The open question — store the **graph** or just the **live vectors** — is now
**answered by measurement**, in `docs/DURABILITY.md` §6. Replay reads a log at
368 ns/record, but *applying* a record costs 703 µs, so reading is 0.05% of
recovery and the rebuild is all of it. For 1M × 128 that is ~703 s of rebuild
against ~0.2 s to load a serialized graph: three orders of magnitude. A
vectors-only snapshot would bound log *size* while leaving recovery *time*
essentially unimproved, which is half a snapshot. **Serialize the graph**, and
accept freezing the representation as the price.

Restore orchestration — load the snapshot, replay the WAL from `Seq+1` — is a
dozen lines once the codec exists, and lands with the public API that has both a
graph and a log to hand.

## SOLID / patterns

- **SRP** — one package = one responsibility.
- **DIP** — `DB`/`Collection` depend on `Index`, `Store`, `WAL`, `Snapshotter`,
  `DistanceFunc`, `Logger` interfaces; concrete impls injected.
- **OCP / Strategy** — distance metric, index type, persistence pluggable.
- **Factory + Functional Options** — `govecdb.Open(cfg, WithWAL(dir), WithMetric(...))`.
- **ISP / Liskov** — small interfaces so a flat index and HNSW are interchangeable.

## Risks

- **Recall regression** — the baseline is locked in `graph_test.go` (0.999 @ dim 32,
  0.972 @ dim 768). Any index change must keep those numbers.
- **Allocation regression** — search is 2 allocs/op; `-benchmem` guards it. The
  scratch pool must keep it there: allocating a `searchState` per search would
  undo the whole zero-allocation path.
- Every step is a separate commit and reverts cleanly.
