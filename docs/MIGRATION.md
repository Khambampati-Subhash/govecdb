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
│   ├── wal/             # write-ahead log
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
4. **Upsert + compaction** — `Insert` must replace a duplicate id, and tombstones
   need a rebuild pass to reclaim memory. **Next.**
5. **`internal/wal`** — append-only log behind a `WAL` interface, with segment rotation.
6. **`internal/snapshot`** — point-in-time graph snapshot + recovery that replays the WAL.
7. **`internal/store`** — vector + metadata storage behind a `Store` interface.
8. **`internal/filter`** — metadata query engine, with tests from day one.
9. **Public API** — `vector.go` / `db.go` / `options.go` facade; this is what users import.
10. **Examples + README** for the real API.

## Phase — WAL

Design constraints, decided:

- **Write to the WAL first, then apply to the in-memory graph.** On recovery,
  replay the log to rebuild the graph. The graph is *derived state* — it is never
  the source of truth.
- **The WAL is an interface** (`Append`, `Replay`, `Sync`, `Close`) so the index can
  be constructed with a no-op WAL in tests and benchmarks.
- **Records are versioned** from the first commit. A log format without a version
  byte cannot be migrated later.
- **Checksums per record.** A torn write at the tail must be detectable, and
  recovery must truncate to the last intact record rather than failing outright.
- **`fsync` policy is a knob**, not a hardcode: always / interval / never trade
  durability against throughput.

Same bar as the index: small single-responsibility files, comments that explain
*why*, and tests that verify crash recovery rather than assuming it.

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
