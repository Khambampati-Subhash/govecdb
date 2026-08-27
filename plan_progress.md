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
| 2 | Tombstone-based `Delete` with stable slot indices and entry re-election | ✅ |
| 3 | Real upsert semantics for `Insert` | ✅ |
| 4 | Compaction pass once tombstones cross a threshold | ✅ |

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

## Task 2 — Tombstone `Delete` ✅

*Plan.md A.2: tombstone-based `Delete` that keeps slot indices stable, keeps
traversing through dead nodes while filtering them out of results, and re-elects
the entry point when the entry node itself is deleted.*

### Why a tombstone is the only option

Two independent reasons, and the second is the one that actually forces it:

1. **Slots are addressed by index.** Every neighbor list in the graph is a list
   of `int` indices into `g.nodes`. Removing one slot shifts every index above
   it, invalidating every neighbor list — an O(N·M) repair for a single delete.
2. **A node is a bridge.** HNSW reaches a region by hopping through whatever
   lies between, and those hops do not care whether the waypoint is still
   wanted. Cut one out and a whole neighborhood can become unreachable —
   vectors nobody deleted, gone.

So `Delete` marks the slot and changes nothing else.

### The core distinction

A tombstone is **dead as an answer, alive as a route**. That maps onto the two
structures `searchLayer` already runs on, and this split *is* the design:

| structure | admits |
|---|---|
| `cands` — the frontier | everything; a dead node is still a bridge |
| `results` | live nodes only; a dead node is not an answer |

The tempting simplification — let `searchLayer` return everything and filter the
final slice in `Search` — is wrong, and quietly so. It returns fewer than `k`
hits as tombstones accumulate, instead of searching wider to find `k` live ones.
Recall would rot silently as a function of delete volume.

`greedyClosest` is left completely untouched: it is pure routing on the upper
layers and never produces an answer, so it stays blind to tombstones by design.

### Entry-point re-election

Deleting the entry point breaks two invariants at once. `reelectEntry` restores
both with one choice — the live node with the highest top level:

- **the entry must be live**, or every search starts from a corpse;
- **the entry must sit at exactly `maxLevel`**, because `Insert` descends from
  `maxLevel` indexing the entry's own neighbor slice, and a shorter entry would
  index past the end of it.

Layers above the new entry are left populated but unreachable — harmless, since
only tombstones live up there now. If every slot is dead, the graph resets to
`entry = -1`, so the next `Insert` takes the first-node path and starts fresh.

The scan is O(N), which is affordable precisely because it is rare: it costs a
pass only when the one specific node that happens to be the entry is deleted.
A level-indexed structure to avoid it would add permanent write-path cost to
make a rare path cheap.

### The bug this turned up

The first implementation passed every semantic test — and then a stress test
found **43 of 145 live vectors unreachable** after a delete-heavy workload.

Measuring the same workload across `M`, against an insert-only control:

| M | insert-only unreachable | delete-heavy, first cut |
|---|---|---|
| 2 | 14/145 | 43/145 |
| 4 | **0**/145 | 12/145 |
| 8 | **0**/145 | 1/145 |
| 16 (default) | 0/145 | 0/145 |

The control column matters: M=2 strands vectors with *no deletes at all*, so
that row is a pre-existing property of an `M` so low that pruning drops reverse
edges — not something deletes introduced. But M=4 and M=8 go from perfect to
broken, and that is a genuine tombstone defect.

**Root cause:** `pruneConnections` is the one place tombstones and live nodes
compete for a scarce resource — edge slots. Ranked purely by distance, a dead
node could evict the edge `Insert` had just created, destroying the new node's
only *inbound* link. Nothing can rescue a node in that state: its own outgoing
edges never help anyone find it.

**Fix:** rank live nodes ahead of tombstones, then by distance. Demoting rather
than dropping is what keeps it safe — `selectNeighbors` still backfills to the
cap, so a node with few live candidates keeps its tombstone edges and the
bridges they carry. Tombstones only lose slots where live alternatives exist.

| M | before fix | after fix |
|---|---|---|
| 2 | 43/145 | **2**/145 |
| 4 | 12/145 | **0**/145 |
| 8 | 1/145 | **0**/145 |

With no tombstones present the comparator falls straight through to distance, so
the delete-free path is bit-identical — confirmed by the recall baselines not
moving.

A second, narrower guard also went in: `searchLayer` returns live nodes only, so
inserting into a region whose every member is tombstoned finds nothing to attach
to. `Insert` falls back to linking the node it searched from — a dead neighbor
still routes, and edges are bidirectional, so linking to a tombstone beats
isolation. After the pruning fix this only triggers at pathologically low `M`.

### API decisions

- **`Delete(id) bool`, not `error`.** There is no failure mode, and inventing an
  always-nil error is worse than an honest bool. Deleting an unknown id is a
  no-op, which makes `Delete` idempotent — deliberate, not lenient: WAL recovery
  replays records, and an operation that failed on its second application would
  make replay order-sensitive.
- **`Delete` unbinds the id but keeps the slot.** Re-inserting the same id
  allocates a *new* slot rather than resurrecting the old one, whose edges were
  chosen for the old vector and would be wrong for a new one.
- **`Len` counts live vectors only.** A `Len` that silently included tombstones
  would be a trap. `Stats{Live, Deleted, Slots}` exposes what is underneath, and
  exists to answer "is it time to compact?" without leaking graph internals to
  whatever ends up deciding that.

### Measured

Locked baselines all held — the tombstone check is a bool test on a cache line
already loaded for the vector:

| | Before | After |
|---|---|---|
| Recall@10, dim 32 | 0.999 | **0.999** |
| Recall@10, dim 768 | 0.972 | **0.972** |
| Search allocations | 2 | **2** |
| Search | 105–112 µs/op | 107–113 µs/op |
| Search, 16 goroutines | 8.3 µs/op | 8.3–8.6 µs/op |

Recall *under* deletes is the new number that matters: with half the graph
tombstoned, recall@10 against brute force over the survivors is **1.000**.

`BenchmarkSearchTombstones` measures what deferred deletion costs — dead slots
keep `results` under-filled, which loosens the pruning bound and widens the
search. This is the curve task 4's threshold should be set against:

| Tombstones | Search | vs. clean |
|---|---|---|
| 0% | 108 µs | — |
| 25% | 127 µs | 1.2× |
| 50% | 165 µs | 1.6× |
| 75% | 262 µs | 2.5× |

Allocations stay at 2/op throughout: tombstones cost time, not memory churn.

### Tests added — `internal/hnsw/delete_test.go`

- **`TestTombstonesDoNotStrandVectors`** — the one that found the bug. Asserts a
  *comparative* invariant across M ∈ {2,4,8,16}: deleting must not strand more
  vectors than a delete-free graph of the same config, and must strand zero
  wherever the delete-free graph is itself perfect. Comparative because M=2 is
  pathological on its own, so an absolute threshold would either be arbitrary or
  encode a pre-existing weakness as acceptable.
- **`TestDeletedNodesStillRoute`** — deletes 70% and requires every survivor to
  remain retrievable; the direct test of "dead nodes keep bridging".
- **`TestRecallAfterDeletes`** — recall vs brute force over the survivors, so
  deletion is held to the same bar as the rest of the index.
- **`TestDeleteEntryPointReelects`** — deletes the entry 20 times in a row,
  asserting both invariants after each.
- **`TestDeleteAllThenReinsert`** — the collapse-to-empty path.
- **`TestReinsertAfterDeleteUsesNewSlot`** — same id, new vector, and the old
  tombstoned vector must never resurface.
- **`TestDeleteIsIdempotent`**, **`TestLenAndStatsTrackTombstones`**,
  **`TestConcurrentDeleteAndSearch`** (deletes against 8 live readers, `-race`).

### Files touched

| File | Change |
|---|---|
| `internal/hnsw/delete.go` | **New.** `Delete` + `reelectEntry`. |
| `internal/hnsw/delete_test.go` | **New.** The nine tests above. |
| `internal/hnsw/node.go` | `deleted bool` tombstone flag. |
| `internal/hnsw/graph.go` | `numDeleted`; `Len` is live-only; `Stats` type + method. |
| `internal/hnsw/search.go` | Frontier/results split in `searchLayer`. |
| `internal/hnsw/neighbors.go` | `pruneConnections` demotes tombstones. |
| `internal/hnsw/insert.go` | Isolation fallback for fully tombstoned regions. |
| `internal/hnsw/bench_test.go` | `BenchmarkSearchTombstones`. |
| `doc.go`, READMEs, `CLAUDE.md`, `docs/MIGRATION.md`, `docs/diagrams/` | Delete semantics documented. |

### Follow-ups this opened

- **Compaction is now load-bearing, not optional** (task 4). Tombstones never
  release memory, delete+reinsert of one id grows `nodes` without bound, and the
  cost curve above is the argument for a threshold somewhere near 25–30%.
- `Insert` still no-ops on a duplicate id (task 3). `Delete` + `Insert` is a
  working upsert in the meantime, which is exactly why task 3 should land before
  the WAL record format freezes.

---

## Task 3 — Upsert semantics ✅

*Plan.md A.3: give `Insert` real upsert semantics so a duplicate id replaces
instead of silently no-opping, fixing the operation set before the WAL format
freezes.*

### Why this had to come before the WAL

`Insert` used to return `nil` and do nothing when the id already existed. That is
not a smaller feature set, it is a *different* one: a caller who re-embeds a
document and stores it gets a silent no-op and a stale vector, with no error to
notice.

The sequencing argument is the stronger one. The WAL record format freezes around
whatever operations exist. A separate UPDATE record would have to mean "insert if
absent" anyway — replay runs against a snapshot that may or may not already hold
the id — so it would be a second record type describing one state transition,
differing only in what it assumes about the past. Upsert collapses that into a
single PUT whose meaning does not depend on history.

### The decision: replace by tombstone + new slot, not in place

Editing the vector in place is the obvious implementation and it is wrong.

The slot keeps its index, and with it **every neighbor list in the graph that
references that index**. Those edges were chosen for the *old* vector: they
encode "these two are close", a claim the new vector does not make. Searches
would keep being routed into the slot from a region it no longer belongs to, and
never from the region it now does. Recall would rot in a way no test of the
updated id alone would catch — the damage is to the neighborhood, not the node.

Repairing the edges is not available either. `pruneConnections` makes adjacency
asymmetric, so a node's own neighbor list is *not* the list of nodes pointing at
it; finding every inbound edge means scanning the whole graph, O(N·M) per update.

So a replacement tombstones the old slot and builds a fresh one, which gets
correct edges by construction. The price is one dead slot per update, and it is
the same debt `Delete` takes on — payable to the same compaction pass.

### The tombstone happens inside `Insert`'s lock

`Delete` was refactored into a locking shell over an unlocked `tombstone(id)`,
which `Insert` calls directly. Calling `Delete` instead would have released the
write lock between the unbind and the re-add, and in that window the id belongs
to nobody: a concurrent reader would observe an *update* as a *disappearance*.

`TestConcurrentUpsertAndSearch` pins this with `Len`, which must hold constant at
n while eight readers watch every id being replaced. It dips the moment the
upsert stops being atomic.

### Unchanged vectors return early

`slices.Equal` against the **stored** form, before anything is tombstoned:

```go
if prev, exists := g.ids[id]; exists {
    if slices.Equal(g.nodes[prev].vector, vec) {
        return nil
    }
    g.tombstone(id)
}
```

This is the WAL-replay path, not a micro-optimization. Recovery replays records
across a snapshot boundary that is deliberately conservative, so re-applying
records the graph already holds is the *normal* case. Without the early return,
every recovery would inflate the graph with tombstones for vectors that never
changed — 332 ns against 776 µs, and no slot consumed.

Comparing against the stored form has a second effect worth naming: under Cosine
the graph stores direction only, so a rescaled vector is correctly recognized as
unchanged. Doubling is exact in binary floating point, so that case is
bit-identical rather than approximately equal, and the test asserts it.

### Validation stays ahead of mutation

`ErrEmptyVector` / `ErrDimensionMismatch` are checked before the lock, and the
tombstone happens after. The reverse order — unbind the id, then discover the
vector is malformed — turns a caller's bug into data loss.
`TestUpsertRejectsBadVectorWithoutDestroying` asserts `Stats` is byte-identical
after a rejected update and that the old vector is still retrievable.

### Measured — Apple M4 Max, 128 dim

| Operation | Cost | |
|---|---|---|
| `Insert`, new id | 713 µs, 208 allocs | unchanged |
| `Insert`, replacing an id | 776 µs, 210 allocs | insert + tombstone, **+9%** |
| `Insert`, vector unchanged | **332 ns**, 2 allocs | the replay path |

Locked baselines all held — the upsert branch is a map lookup on a path that
already does one:

| | Before | After |
|---|---|---|
| Recall@10, dim 32 | 0.999 | **0.999** |
| Recall@10, dim 768 | 0.972 | **0.972** |
| Search allocations | 2 | **2** |
| Search | 107–113 µs/op | 110 µs/op |
| Search, 16 goroutines | 8.3–8.6 µs/op | 8.2 µs/op |

New number that matters: **recall@10 after replacing half the graph is 0.999** —
identical to a graph that was built that way from the start.

### Tests added — `internal/hnsw/upsert_test.go`

- **`TestInsertReplacesVector`** — the new vector answers, the old one does not at
  any rank, and asking for more results than there are live vectors proves the id
  did not become two entries.
- **`TestRepeatedUpsertsResolveToLatest`** — the workload upsert exists for:
  three full re-embeddings of a 300-vector corpus. Every id must resolve to its
  *latest* vector after every round, which by round three means staying reachable
  in a graph that is three-quarters tombstones.
- **`TestUpsertUnchangedVectorIsFree`** — `Stats` must be unchanged after
  re-inserting every vector, both identically and rescaled (Cosine).
- **`TestUpsertRejectsBadVectorWithoutDestroying`** — validation before mutation.
- **`TestUpsertEntryPointReelects`** — 20 consecutive updates *of the entry
  point*, asserting both entry invariants each time; the tombstone and the insert
  have to agree about the entry within one lock.
- **`TestUpsertCountsTheOldSlotAsATombstone`** — `Stats{Live, Deleted, Slots}`
  accounting: an update adds no live vectors and exactly one dead slot.
- **`TestRecallAfterUpserts`**, **`TestConcurrentUpsertAndSearch`** (`-race`).
- **`BenchmarkUpsert`**, **`BenchmarkUpsertUnchanged`**.

### Files touched

| File | Change |
|---|---|
| `internal/hnsw/insert.go` | Upsert branch: unchanged-vector early return, then `tombstone` + fresh slot. |
| `internal/hnsw/delete.go` | `Delete` split into a locking shell over an unlocked `tombstone`. |
| `internal/hnsw/upsert_test.go` | **New.** The eight tests above. |
| `internal/hnsw/bench_test.go` | `BenchmarkUpsert`, `BenchmarkUpsertUnchanged`. |
| `doc.go`, both `README.md`s, `CLAUDE.md`, `docs/MIGRATION.md`, `docs/diagrams/02` | Upsert semantics documented; the "duplicate id is a no-op" claim removed. |

### Follow-ups this opened

- **Compaction is now the only thing standing between this index and unbounded
  memory growth** (task 4). Deletes leak slots for vectors a caller asked to
  remove; updates leak slots for vectors that are *still live*, at whatever rate
  the corpus is re-embedded. `Stats` already reports the ratio a policy needs.
- The graph has no way to report *what* was replaced. Once a payload store exists
  (task 12), an upsert will need to evict the old payload too, and that is where
  "did this create or replace?" stops being an internal detail.

---

## Task 4 — Compaction ✅

*Plan.md A.4: add a compaction pass that rebuilds the graph once tombstones cross
a threshold, since tombstoned slots never release memory on their own.*

### What it does

`Compact() int` rebuilds the graph over its live vectors and reports the slots
reclaimed. On a graph with no tombstones it returns 0 without touching anything,
so it is cheap to poll.

### Rebuild, never renumber

Removing dead slots from the existing graph is not available, for the reason that
has now shaped three tasks in a row: **every neighbor list is a list of indices
into `nodes`**, so dropping one slot shifts every index above it and invalidates
every list in the graph at once. Repairing in place is the same O(N·M) scan that
ruled out in-place updates, except it runs per dead slot instead of once.

So compaction builds a *new* graph from the live vectors and swaps `nodes`,
`ids`, `entry` and `maxLevel` across in one assignment under the write lock. The
slot-index invariant is never violated — it is retired along with the graph that
held it. That is now written down as invariant 5 in `docs/diagrams/README.md`,
because compaction is precisely the change that *looks* like it breaks it.

### The rebuild is equal to a fresh build, and that is testable

The replacement seeds a fresh RNG from the same config and re-inserts the live
vectors in slot order, so it produces **exactly the graph you would have built if
the dead vectors had never existed** — not merely an equally good one.

`TestCompactMatchesAFreshBuild` asserts that against a directly-built reference:
same `Stats`, and 50 queries agreeing rank for rank on both id *and* distance.
Equality rather than sampled recall, which is a much sharper instrument.

It is also what guards the quiet half of the change. `Insert` was split into the
public entry point and `insertPrepared`, which takes a vector already in stored
form. Compaction has to use it: sending stored vectors back through `prepare`
would copy every vector for nothing, and re-normalizing an already-unit vector
drifts it by an ulp — the compacted graph would hold *almost* the same numbers,
every recall test would still pass, and only an equality test would notice.

### Compaction is a repair, not just a reclaim

Edges that pointed at tombstones become edges between live nodes, and `results`
fills at full speed again so the pruning bound in `searchLayer` tightens. The
rebuilt graph is strictly better than the one it replaced; the memory is only the
headline.

### Stop-the-world, and why the policy lives outside

`Compact` holds the write lock for a full index build. Building the replacement
outside the lock means writes landing in the old graph while the new one is
built, and reconciling them wants a change log plus a double-buffered swap — both
of which the WAL should shape first, since it will already be recording those
writes. Recorded as deferred in `docs/MIGRATION.md`, not overlooked.

Because the pause is real, the index does not decide when to take it. There is no
`CompactionThreshold` config and no background goroutine: an automatic trigger
inside `Delete` would mean an innocuous call occasionally blocking for seconds.
`Stats().DeadRatio()` reports the number and the caller picks the moment.

### Measured — the threshold guess was wrong

Task 2 recorded that the search-cost curve "is the argument for a threshold
somewhere near 25–30%". Measuring the *other* side of the trade inverts that:

| Dead slots | Search | `Compact()` pause, 5k × 128 | Reclaims |
|---|---|---|---|
| 25% | 1.2× | **2.61 s** | 25% of slots |
| 50% | 1.6× | **1.69 s** | 50% of slots |
| 75% | 2.5× | **0.79 s** | 75% of slots |

A compaction is a full build over the **survivors**, so the pause tracks how many
vectors live, not how many are reclaimed. Compacting early is therefore the worst
of both: a longer pause, more often, giving back less memory. Read together the
curves argue for **~0.5**, where the standing cost is 1.6× search and a graph
carrying twice the slots it needs. The documented threshold hint moved from 0.25
to 0.5 on the strength of that.

Locked baselines held — compaction adds no work to any hot path:

| | Before | After |
|---|---|---|
| Recall@10, dim 32 | 0.999 | **0.999** |
| Recall@10, dim 768 | 0.972 | **0.972** |
| Search allocations | 2 | **2** |
| Search | 110 µs/op | 105 µs/op |

Recall after compacting a 50%-tombstoned graph is 1.000, unchanged from before
the rebuild — at that ratio tombstones were costing time, not accuracy.

### Tests added — `internal/hnsw/compact_test.go`

- **`TestCompactMatchesAFreshBuild`** — the load-bearing one, described above.
- **`TestCompactKeepsEveryLiveVector`** — compaction rebuilds every neighbor list
  in the graph, so a bug here does not corrupt a vector, it silently drops one.
  Every survivor must return itself at distance ~0, and 100 queries must never
  surface a compacted-away id.
- **`TestCompactIsANoOpWithoutTombstones`** — the early return, pinned by
  comparing search results across the call: `DeadRatio` invites polling, and
  rebuilding a clean graph would burn a full build to reclaim nothing.
- **`TestCompactAfterUpserts`** — three re-embedding rounds, then compaction must
  keep each id's *latest* vector, not an earlier generation of the same id.
- **`TestCompactRecall`** — recall before and after, asserting the rebuild never
  costs recall.
- **`TestCompactAllDeleted`**, **`TestCompactEmptyGraph`** (also pins
  `DeadRatio` returning 0 rather than NaN on an empty graph),
  **`TestCompactReclaimsSlots`**, **`TestConcurrentCompactAndSearch`** (`-race`;
  `Len` must hold constant across five compactions, since the live population is
  identical either side of the swap).
- **`BenchmarkCompact`** at 25/50/75% dead, rebuilding the fixture outside the
  timer because a compacted graph is clean and the second call measures nothing.

### Files touched

| File | Change |
|---|---|
| `internal/hnsw/compact.go` | **New.** `Compact`. |
| `internal/hnsw/compact_test.go` | **New.** The nine tests above. |
| `internal/hnsw/graph.go` | `New` split over `newGraph`; `Stats.DeadRatio()`. |
| `internal/hnsw/insert.go` | `insertPrepared` extracted for the rebuild path. |
| `internal/hnsw/bench_test.go` | `BenchmarkCompact`. |
| `doc.go`, both `README.md`s, `CLAUDE.md`, `docs/MIGRATION.md`, `docs/diagrams/04` | Compaction documented; the 25% threshold hint corrected to 50%. |

### Follow-ups this opened

- **Online compaction** — the pause is the price of a coarse write lock, and the
  fix (change log + double-buffered swap) wants the WAL to exist first.
- **Nothing calls `Compact` yet.** The policy belongs to the collection/DB layer
  (task 13), which is also what will own the goroutine that runs it.
- A compacted graph diverges from what WAL replay would rebuild. Both are correct
  — the graph is derived state and compaction changes no logical state — but
  recovery-by-equality tests must compare against an *uncompacted* replay.

---

## A · Finish the graph — complete

All four tasks are done: pooled scratch and concurrent reads, tombstone deletes,
upsert, compaction. The operation set is now closed — PUT and DELETE, with
compaction as a physical-layout operation that logs nothing — which is exactly
the precondition Plan.md set before freezing a record format.

## Interlude — test depth and a measurement harness

*Not a numbered task. Before freezing a record format around this index, its
behaviour should be measured rather than assumed, and the parts no recall test
can see should be tested directly.*

### The gaps that existed

Three files had no direct test at all, and each hid a class of bug that recall
numbers would have dented rather than broken:

- **The distance kernels** are unrolled four-wide with a scalar tail — and every
  dimension used anywhere else in the package is divisible by four. A wrong tail
  loop was invisible. They are now checked against a float64 reference across 28
  dimensions (1, 2, 3, 5, … 129, 769, 1536), plus a case where the tail element is
  the *only* difference between two inputs.
- **The heaps** are hand-written to dodge `container/heap`'s boxing, so the sift
  loops are ours. The invariant is now asserted after every push and pop under
  randomized interleaving, with ties, and with payload integrity checked
  separately from ordering.
- **`visitedList`'s wraparound.** When the generation counter laps `MaxUint32`,
  every stale stamp matches the new generation and a search would treat the whole
  graph as visited — returning almost nothing, silently. Reaching it honestly
  takes 2³² searches; the test drives the counter to the edge directly.

Recall coverage was also narrower than it looked: every recall test used Cosine,
so Euclidean and DotProduct had no accuracy coverage above the kernel level, and
every one pinned a single seed.

### The harness

`recall_test.go` is one piece of code with two jobs, selected by a `-results`
flag. Without it: a small grid, threshold assertions, ~20 s inside the normal
test run. With it: a wider grid over 5,000-vector corpora, every cell written to
CSV. Because it is the same code, a number printed in a README and a threshold
enforced by CI cannot drift apart.

`docs/benchmarks/plot.go` renders that CSV to SVG. It is `//go:build ignore` and
stdlib-only — hand-written SVG, because a charting library would have been the
first crack in "no third-party dependencies", and axes plus polylines are a
couple hundred lines.

### What the measurements changed

Three claims in this repo were wrong, and would have stayed wrong:

1. **The compaction threshold.** Task 4 already corrected 25% → 50% by measuring
   the pause instead of only the search cost.
2. **"Euclidean is weak."** At 5,000 vectors Euclidean scored 0.820 against
   Cosine's 0.830 at `ef=64`, and the first explanation drafted for that gap was a
   story about normalization putting Cosine on the unit sphere. Measuring all
   three metrics at a wide `ef` killed it: 0.998 / 0.980 / 0.999. No metric is
   weak; the whole grid was simply sitting at a corpus size where `ef=64` is
   narrow.
3. **"The all-positive test corpus depresses recall."** Plausible — every vector
   shares an orthant, so any two are ~0.75 similar before you look at the data. A
   centred corpus was introduced on that theory. Measured side by side, positive
   scores *higher* (0.865 vs 0.852). The real driver was corpus size at fixed
   `ef`, which the scale sweep isolates. The corpus stays centred for being
   representative, and the positive case stays in the sweep so the correction
   remains checkable.

A fourth thing surfaced that was not wrong, just unknown: **compaction slightly
lowers recall** (0.968 → 0.949 at 50% dead). Tombstones keep `results`
under-filled, which loosens the pruning bound and widens the search past what
`ef` asked for — recall nobody requested, at 184 µs against 88 µs. The test
tolerance now documents that as a withdrawn subsidy rather than a regression.

### The headline numbers

5,000 vectors, dim 128, k=10, Apple M4 Max:

| Sweep | Range measured |
|---|---|
| `ef` 10 → 512 | recall 0.310 → 0.999, latency 27 µs → 438 µs |
| `N` 500 → 20,000 at ef=64 | recall 0.997 → 0.652, latency 44 µs → 141 µs (**3.2× for 40× data**) |
| dim 8 → 1536 at ef=64 | recall 1.000 → 0.558, latency 17 µs → 1,090 µs |
| `M` 4 → 48 | recall 0.294 → 0.994, build 0.6 s → 73 s |
| tombstones 0 → 75% | latency 104 µs → 243 µs, and 69 µs after `Compact()` |

The most useful of these for anyone using the library: **`ef` must grow with `N`.**
A fixed `ef=64` is a starting point, not a setting.

### The follow-up: measuring the surface, not two slices

The sweeps above vary one knob at a time with the others at their defaults, which
is enough to show a slope and not enough to choose a configuration. Two 2-D grids
were added afterwards, and both changed an answer:

**`M` × `ef` at 5,000 vectors.** Reading the `M` chart and the `ef` chart together
had suggested M=32 was roughly twice as fast as M=16 at matched recall. It is not
— those charts each hold the *other* knob at its default, so the points being
compared sat at different recall levels. Measured on the grid, at equal recall:

| Target | via `ef` (M=16) | via `M` | Build |
|---|---|---|---|
| ~0.96 | ef=128 → 0.964, 180 µs | M=32/ef=64 → 0.968, **162 µs** | 3.8 s → 23 s |
| ~0.997 | ef=256 → 0.998, 282 µs | M=32/ef=128 → 0.997, **251 µs** | 3.8 s → 23 s |

**~10% latency for 6× the build and double the memory.** That turns "raise M"
from general advice into a narrow special case, and confirms M=16 as the default.

**`N` × `ef` across a 20× range.** Holding a recall target needs `ef ∝ n^0.78` —
sub-linear, but far steeper than the `log N` the hop count follows. Fitted into
`SuggestedEf(n, k, target)` and `g.SuggestedEf(k, target)`, so the guidance lives
in the API rather than only in a README.

Calibrated exactly on the sweep, that function **undershot on three of four
verification corpora** (0.921 against a 0.95 target) — recall on one corpus does
not transfer precisely to another. The anchors now carry margin and the target is
documented as a floor to clear: 0.95 measures 0.969 at 1,000 vectors and 0.972 at
5,000. `TestSuggestedEfAchievesTarget` builds real graphs and fails if a
suggestion misses, so the constants cannot rot quietly when neighbour selection
or the default `M` changes.

The grids also surfaced the variance figure that governs how precise any of this
can be: **the same configuration on three different corpora returned 0.595, 0.646
and 0.677** — an eight-point spread at mid-range recall, compressing near
saturation. That number is why the suggestion carries margin rather than aiming
at a median.

### Files touched

| File | Change |
|---|---|
| `internal/hnsw/suggest.go` | **New.** `SuggestedEf` + the `Graph` method, fitted to the N x ef grid. |
| `internal/hnsw/suggest_test.go` | **New.** Verifies suggestions against real graphs; shape and clamping. |
| `internal/hnsw/distance_test.go` | **New.** Kernels vs float64 reference across 28 dims, tails, wiring. |
| `internal/hnsw/pq_test.go` | **New.** Heap invariants, interleaving, ties, payload. |
| `internal/hnsw/visited_test.go` | **New.** Stamps, resize reuse, the 2³² wraparound. |
| `internal/hnsw/recall_test.go` | **New.** Seven sweeps, dual-mode harness, seed stability. |
| `internal/hnsw/bench_test.go` | `BenchmarkSearchByDimension`, `BenchmarkSearchByScale`. |
| `docs/benchmarks/` | **New.** `plot.go`, `results.csv`, seven SVGs, and a README. |
| Root `README.md`, `internal/hnsw/README.md`, `CLAUDE.md` | Charts embedded; `ef` guidance corrected. |

---

## Next: Task 5 — the WAL record format and writer

Design constraints are already decided in `docs/MIGRATION.md`: versioned records
from the first commit, CRC per record, truncate the torn tail rather than failing
recovery, segment rotation built in from the start (not bolted on), and fsync as
a policy knob.
