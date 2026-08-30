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
│   ├── snapshot/        # snapshots + recovery
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
7. **`internal/snapshot`** — point-in-time graph snapshot + recovery that replays
   the WAL. **Next.**
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

Still open, and now blocked on snapshots rather than on the log: writing
`TypeCheckpoint` records and deleting segments below one. A checkpoint has no
meaning until there is a snapshot for it to point at.

Same bar as the index: small single-responsibility files, comments that explain
*why*, and tests that verify crash recovery rather than assuming it. The tear
tests damage a log by truncating and rewriting it, which reproduces the shapes
power loss leaves behind; the harness that `SIGKILL`s a child mid-write to
reproduce the *timing* is still outstanding.

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
