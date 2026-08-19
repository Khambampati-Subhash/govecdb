# v1 Progress

Execution log for [`Plan.md`](Plan.md). One entry per task: what shipped, why it
was built that way, and what it measured. Tasks are only marked done when
`go build ./... && go vet ./... && go test ./... -race` is green.

**Legend:** ✅ done · 🔨 in progress · ⬜ not started

---

## A · Finish the graph

| # | Task | Status |
|---|------|--------|
| 1 | Pool the search scratch, then guard the graph with an `RWMutex` | ✅ |
| 2 | Tombstone-based `Delete` with stable slot indices and entry re-election | ⬜ |
| 3 | Real upsert semantics for `Insert` | ⬜ |
| 4 | Compaction pass once tombstones cross a threshold | ⬜ |

## B · Durability

| # | Task | Status |
|---|------|--------|
| 5 | Versioned record format + append-only writer with segment rotation | ⬜ |
| 6 | Reader that validates every CRC and truncates the torn tail | ⬜ |
| 7 | Write-failure policy decided once, at the WAL boundary | ⬜ |
| 8 | Checkpoint serializer: temp file → fsync → atomic rename | ⬜ |
| 9 | Recovery: newest valid snapshot → replay above its seq → open for writes | ⬜ |
| 10 | Background flusher with an fsync policy knob + checkpoint scheduler | ⬜ |
| 11 | Crash harness: `SIGKILL` mid-write, reopen, assert the surviving prefix | ⬜ |

## C · Make it a library

| # | Task | Status |
|---|------|--------|
| 12 | Payload/metadata store so an id carries more than a vector | ⬜ |
| 13 | Collections/namespaces with load-on-demand and idle eviction | ⬜ |
| 14 | `Index` / `Store` / `WAL` interfaces + the public facade | ⬜ (stub early — see the sequencing warning in `Plan.md`) |
| 15 | `Close()` — flush, final fsync, optional checkpoint | ⬜ |
| 16 | Examples + README rewritten against the real API | ⬜ |

---

## Task 1 — Concurrent reads ✅

*Plan.md A.1: move the scratch off `Graph` into a pooled `searchState`, then
guard the graph with an `RWMutex` and re-baseline the benchmarks.*

### The problem

`Search` was not a reader. Every call wrote five `Graph` fields — `visited`,
`scratchCands`, `scratchRes`, `scratchSel`, and `queryBuf`. Dropping an
`RWMutex` on top of that would have been decorative: two goroutines holding
`RLock` would each have reset the same visited array and rewritten the same heap
slices mid-traversal, corrupting each other's results while the race detector
stayed quiet about the *lock* being correctly held.

So the ordering mattered. Pool the scratch first; only then does the lock mean
anything.

### What was built

**`internal/hnsw/state.go` (new).** A `searchState` holds everything one
traversal mutates:

```go
type searchState struct {
    visited  visitedList
    cands    []candidate // min-heap: the frontier still worth expanding
    results  []candidate // max-heap: the best ef found so far, worst at [0]
    rejected []candidate // selectNeighbors' backfill list
    queryBuf []float32   // normalized copy of the caller's query
}
```

It is handed out by a `sync.Pool` **on the Graph**, not a package global, so a
graph's scratch — sized to *its* node count — dies with the graph instead of
pinning the largest index's 40 KB visited array for the process lifetime.

Pooling rather than allocating per call is not an optimization detail here: a
per-search `searchState` would allocate the visited array plus three heaps every
query, which is precisely the cost the 2-allocations-per-search path exists to
avoid.

**The lock.** `Graph.mu sync.RWMutex`. `Search` and `Len` take `RLock`; `Insert`
takes `Lock`. The scratch-consuming internals — `searchLayer`,
`selectNeighbors`, `pruneConnections` — now take an explicit `*searchState`,
which makes "who owns this buffer" a signature-level fact rather than a
convention.

Two things deliberately happen *outside* the critical section, because they only
touch caller data and construction-time config:

- `Insert` copies and normalizes the vector before `Lock`.
- `Search` normalizes the query into pooled scratch before `RLock`.

**Why the lock is coarse.** One `RWMutex` over the whole graph. An HNSW insert
rewrites neighbor lists several hops away from the new node — `connect` and
`pruneConnections` both reach into arbitrary existing nodes — so there is no
small region to lock instead. Reads scale; concurrent writers serialize.
Fine-grained write locking is a later step, and is recorded as deferred rather
than overlooked in `docs/MIGRATION.md`.

### The `visitedList` invariant survives pooling

Worth spelling out, because it looks fragile and is not. A pooled state can be
reused across graphs of different sizes, and `reset(n)` may keep marks written
during a previous, unrelated search. That is safe: `gen` only ever increases
within a state, and a stamp is only ever written with the *current* `gen`, so a
stale stamp can never equal a future generation. Correct regardless of which
graph the state visited last.

### Tests added — `internal/hnsw/concurrent_test.go`

- **`TestConcurrentSearchMatchesSerial`** — the load-bearing one. The same 64
  queries run from 16 goroutines must return *exactly* what they return
  serially, rank for rank. This fails the moment scratch is shared again, which
  a race detector run alone would not reliably catch.
- **`TestConcurrentInsertAndSearch`** — 4 writers and 8 readers against one
  graph; every id a reader gets back must be one that was actually inserted, so
  a torn read is an assertion failure and not just a flake.
- **`TestSearchDoesNotMutateQuery`** — the read-only contract at the API edge:
  cosine normalization must land in pooled scratch, never in the caller's slice.
- **`BenchmarkSearchParallel`** — `RunParallel`, to measure what the lock bought.

### Measured — Apple M4 Max, 10k × 128 dim, k=10, ef=64

| | Before | After | |
|---|---|---|---|
| Recall@10, dim 32 | 0.999 | **0.999** | baseline held |
| Recall@10, dim 768 | 0.972 | **0.972** | baseline held |
| Search allocations | 2 | **2** | baseline held |
| Search | 109–116 µs/op | 105–112 µs/op | unchanged within noise |
| Insert | ~700 µs, 208 allocs | ~700 µs, 208 allocs | unchanged |
| Search, 16 goroutines | — | **8.3 µs/op** | **12.7× throughput** |

Search bytes moved 1264 → 1269 B/op on the serial benchmark — pool refills
amortized across GC cycles. The parallel benchmark reports 1264 flat.

**None of the locked baselines were renegotiated.** Concurrency cost nothing on
the serial path.

### Files touched

| File | Change |
|---|---|
| `internal/hnsw/state.go` | **New.** `searchState` + pool accessors. |
| `internal/hnsw/concurrent_test.go` | **New.** The four tests above. |
| `internal/hnsw/graph.go` | Scratch fields → `mu sync.RWMutex` + `pool sync.Pool`; `Len` locks. |
| `internal/hnsw/search.go` | `Search` takes `RLock`; `searchLayer` takes `*searchState`. |
| `internal/hnsw/insert.go` | `Insert` takes `Lock`; prepare-before-lock. |
| `internal/hnsw/neighbors.go` | `selectNeighbors` / `pruneConnections` take `*searchState`. |
| `doc.go`, both `README.md`s, `CLAUDE.md`, `docs/MIGRATION.md`, `docs/diagrams/` | Concurrency claims corrected. |

### Follow-ups this opened

- Concurrent **writers** still serialize. Fine-grained locking is out of scope
  until the durability layer settles, since a WAL write ordering constraint will
  shape it.
- `Insert` now copies the vector before the duplicate check, so a duplicate id
  costs one wasted allocation. Task 3 (upsert) removes the dead path entirely.

---

## Next: Task 2 — tombstone `Delete`

Ordering note: `Plan.md` puts the whole of section A ahead of the WAL on
purpose — the record format should not freeze around an operation set that is
still missing `Delete` and a real upsert. `docs/MIGRATION.md` has been reordered
to match.
