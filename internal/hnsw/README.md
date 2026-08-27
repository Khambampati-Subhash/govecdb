# internal/hnsw

A small, readable HNSW (Hierarchical Navigable Small World) index — the core of
GoVecDB's approximate nearest-neighbor search. Safe for concurrent use, written
to be understood first and then made fast with changes that are each measured
rather than assumed.

## Why HNSW

Comparing a query against every stored vector is `O(N)` and collapses at scale.
HNSW builds a layered graph — sparse "express lanes" on top, every vector on the
bottom — so a search takes big jumps to the right region, then smaller steps
down to the true neighbors, visiting only a tiny fraction of nodes (`~O(log N)`).

## Files (one responsibility each)

| File | Responsibility |
|------|----------------|
| `doc.go` | Package overview and this file map. |
| `config.go` | `Config` knobs, `DefaultConfig`, and the sentinel errors callers match on. |
| `graph.go` | The `Graph` type: state, `New`, `Len`, and the shared helpers (`randomLevel`, `prepare`). |
| `insert.go` | `Insert` — building the graph, and replacing an id that is already in it. |
| `delete.go` | `Delete` — tombstoning a slot, and re-electing the entry point when it is the one deleted. |
| `compact.go` | `Compact` — rebuilding the graph over its live vectors to reclaim tombstoned slots. |
| `suggest.go` | `SuggestedEf` — the measured `ef ∝ n^0.78` curve, fitted so callers need not guess. |
| `search.go` | `Result`, `Search`, and the primitives it rides on: `greedyClosest`, `searchLayer`. |
| `neighbors.go` | Edge management: alpha-pruned `selectNeighbors`, `pruneConnections`, `connect`, adjacency lookups. |
| `node.go` | A single vector: `id`, `vector`, per-layer neighbor lists. |
| `distance.go` | `Metric` (Cosine / Euclidean / DotProduct) + unrolled kernels. Everything returns **smaller = closer**, so the graph never branches on the metric. |
| `pq.go` | Hand-written min/max heaps over `[]candidate` — no `container/heap`, no interface boxing. |
| `visited.go` | Generation-stamped visited set, reused across searches. |
| `state.go` | `searchState`: the pooled per-traversal scratch that makes `Search` read-only. |
| `graph_test.go` | Correctness + recall-vs-brute-force at 32 and 768 dimensions. |
| `concurrent_test.go` | Parallel-vs-serial equivalence, mixed reader/writer race coverage. |
| `delete_test.go` | Tombstone semantics, recall under deletes, entry re-election, stranding. |
| `upsert_test.go` | Replacement semantics, replay no-ops, recall under updates, atomicity. |
| `compact_test.go` | Slot reclamation, equality against a fresh build, recall after rebuild. |
| `distance_test.go` | Kernels vs a float64 reference across 28 dimensions, tail handling, metric wiring. |
| `pq_test.go` | Heap invariants under interleaved push/pop, ties, payload integrity. |
| `visited_test.go` | Generation stamps, reuse across graph sizes, the 2³²-search wraparound. |
| `recall_test.go` | The sweep harness: dimension / scale / ef / M / metric / distribution / tombstones, plus the M x ef and N x ef grids. |
| `suggest_test.go` | Builds real graphs and fails if a suggested `ef` misses its recall target. |
| `bench_test.go` | Insert / upsert / search / compaction / distance benchmarks. |

## The knobs

| Knob | Where | Adaptable? |
|------|-------|-----------|
| `M` — neighbors per node (layers > 0; layer 0 uses `2*M`) | `Config`, set once | **No** — structural; changing it means rebuilding. |
| `EfConstruction` — search width during inserts | `Config` | Kept fixed (100–200). |
| `Alpha` — pruning relaxation (see below) | `Config` | Fixed per graph; 1.0–1.4 useful, default 1.2. |
| `ef` — search width at query time | `Search(query, k, ef)` | **Yes** — per query; auto-clamped to `>= k`. |

## Design decisions that matter

### 1. Normalize on insert, so cosine becomes a dot product
Cosine similarity needs `dot(a,b) / (|a|·|b|)` — three accumulators and two
square roots per comparison. Since only *direction* matters, we unit-normalize
each vector once at insert time; from then on `|a| = |b| = 1` and cosine
distance is just `1 - dot(a,b)`. **2.5× faster per comparison**, and the ranking
is provably unchanged (`TestNormalizationPreservesRanking`).

Only Cosine normalizes — doing it for Euclidean or DotProduct would silently
change what those metrics mean.

### 2. Alpha-pruned neighbor selection (DiskANN/Vamana), not "keep the M closest"
Keeping simply the M nearest neighbors produces edges that all point into the
same cluster, which strands searches in local minima. Instead a candidate `c` is
dropped when it already sits closer to an already-chosen neighbor `s` than to the
node itself — you could reach `c` by hopping through `s`, so that edge buys
nothing.

`Alpha` scales that test: `alpha * d(c,s) <= d(c,q)` rejects. `Alpha = 1.0` is
the classic HNSW heuristic; `> 1.0` prunes harder and keeps more long-range
shortcut edges, making the graph more navigable.

### 3. Zero-allocation search path
Two changes took search from **795 allocations to 2**:
- The per-search `map[int]struct{}` visited set became a reusable array of
  generation stamps — clearing is a counter bump, not an allocation.
- `container/heap` passes values as `any`, boxing every candidate. The heaps here
  operate on `[]candidate` directly.

### 4. Unrolled distance kernels
Four independent accumulators break the floating-point dependency chain so the
CPU can overlap additions, and slices are re-sliced to a common length to hoist
bounds checks. Euclidean went 74.5 ns → 26.6 ns.

### 5. Copy on insert
The graph stores its own copy of every vector. Beyond enabling normalization,
this stops the graph from aliasing (and being corrupted by) a caller's reused
buffer — see `TestInsertDoesNotAliasCaller`.

### 6. Pooled scratch is what makes concurrent search possible
The scratch state used to live on `Graph`, so every `Search` *wrote* to the
index: it stamped the visited array and rewrote the heap slices. Putting an
`RWMutex` on top of that would have been theatre — two readers holding `RLock`
would still have corrupted each other's traversal.

So the scratch moved first, into a pooled `searchState` (`state.go`), and only
then did the `RWMutex` go on. Now `Search` genuinely only reads the graph:
readers run in parallel, `Insert` excludes them, and the pool keeps the
2-allocations-per-search property intact because it hands back the same 40 KB
visited array and heaps instead of reallocating them.

The lock is coarse — one `RWMutex` over the whole graph. HNSW inserts rewrite
neighbor lists several hops from the new node, so there is no small region to
lock instead; fine-grained writes are a later step. Reads scale, writers
serialize.

`TestConcurrentSearchMatchesSerial` pins the guarantee: the same queries run from
16 goroutines must return exactly what they return serially.

### 7. Delete is a tombstone, and dead nodes keep routing
Removing a node outright is not an option twice over. Slots are addressed by
index, so removing one shifts every index above it and invalidates every
neighbor list in the graph. Worse, a node is a *bridge* — HNSW reaches a region
by hopping through whatever lies between — so cutting one out can strand a whole
neighborhood of vectors nobody deleted.

So `Delete` marks the slot and leaves everything else alone. The distinction
that makes it work lives in `searchLayer`, across its two structures:

| | admits |
|---|---|
| `cands` — the frontier | **everything.** A dead node is still a bridge. |
| `results` | **live nodes only.** A dead node is not an answer. |

Filtering at the end instead would be simpler and wrong: it would return fewer
than `k` hits as tombstones pile up, rather than searching wider to find `k`
live ones.

Two consequences worth knowing:

- **`pruneConnections` demotes tombstones.** It is the one place dead and live
  nodes compete for a scarce resource — edge slots. Ranked purely by distance, a
  tombstone can evict the edge `Insert` just created and leave the new vector
  with no inbound link, which is silent data loss. Demoting (not dropping) is
  the fix: `selectNeighbors` still backfills to the cap, so a node with few live
  candidates keeps its tombstone edges and the bridges they carry.
- **`Delete` is idempotent** and returns `bool`, not `error`. Deleting an unknown
  id is a no-op because WAL recovery replays records, and an operation that
  failed on its second application would make replay order-sensitive.

`Len` counts live vectors only; `Stats` exposes the tombstones behind them.

### 8. `Insert` is an upsert, and a replacement builds a new slot
There is no `Update`. A second `Insert` under a live id replaces the vector, so
one operation covers create and replace.

That is a narrowing of the operation set rather than a convenience. The WAL
record format freezes around whatever operations exist, and a separate UPDATE
record would have to mean "insert if absent" anyway to survive replay against a
snapshot that may or may not already hold the id — two record types describing
one state transition, differing only in what they assume about the past.

A replacement **tombstones the old slot and builds a fresh one**; it does not
overwrite the vector in place. Overwriting would keep the slot's index, and with
it every neighbor list in the graph pointing at that index — edges chosen for the
*old* vector, encoding "these two are close" about a pair that no longer is.
Searches would keep being routed into the slot from a region it left. Repairing
those edges is not an option either: pruning makes edges asymmetric, so a node's
own neighbor list is not the list of nodes pointing at it, and finding every
inbound edge means scanning the whole graph — O(N·M) per update.

Two consequences:

- **Updates pay into the same tombstone debt as deletes.** Re-embedding a corpus
  of 300 vectors three times leaves 300 live vectors carried by 1,200 slots.
  That workload — not deletion — is what makes compaction load-bearing.
- **Re-applying an unchanged vector is free.** The comparison runs against the
  *stored* form, so it costs one `slices.Equal` and changes nothing (331 ns
  against 776 µs for a real replacement). That is the shape of WAL replay across
  a snapshot boundary; without it every recovery would inflate the graph with
  tombstones for vectors that never changed. Under Cosine the graph stores
  direction only, so a rescaled vector is recognized as unchanged too.

The whole upsert commits under **one** write lock — `Insert` tombstones directly
rather than calling `Delete` — so a concurrent reader never observes the moment
where the id belongs to nobody. An update is never visible as a disappearance.

### 9. Compaction rebuilds; it never renumbers
`Compact()` is the only thing that gives tombstoned memory back. It builds a
**new** graph from the live vectors and swaps it in whole, rather than removing
dead slots from the existing one.

It has to. Every neighbor list in the graph is a list of indices into `nodes`,
so dropping one slot shifts every index above it and invalidates every list at
once. Repairing them in place is the same O(N·M) scan that ruled out in-place
updates, except now it runs per dead slot instead of once.

What comes out is **exactly the graph you would have built if the dead vectors
had never existed** — the replacement seeds a fresh RNG from the same config and
re-inserts in slot order, so `TestCompactMatchesAFreshBuild` can assert equality
against a directly-built reference, rank for rank, rather than sampling recall
and hoping. That test is also what guards `insertPrepared`: stored vectors move
across as-is, because re-normalizing an already-unit vector drifts it by an ulp
and the compacted graph would quietly stop holding the same numbers.

`Compact` **stops the world** — it holds the write lock for a full index build.
That is deliberate for v1: building the replacement outside the lock means
writes landing in the old graph while the new one is built, and reconciling them
needs a change log and a double-buffered swap that the durability layer should
shape first. So the index does not decide *when*; `Stats().DeadRatio()` reports
the ratio and the caller picks a moment that tolerates the pause.

## Measured results

Apple M4 Max, 10k vectors × 128 dim, k=10, ef=64:

| | Before | After | Change |
|---|---|---|---|
| Search | 250,878 ns/op | 105,117 ns/op | **2.4× faster** |
| Search allocations | 795 | 2 | **~400× fewer** |
| Search bytes | 95,579 B/op | 1,269 B/op | **75× less** |
| Search, 16 goroutines | — | 8,288 ns/op | **12.7× throughput** |
| Insert allocations | 1,680 | 208 | **8× fewer** |
| Cosine distance | 73.9 ns | 30.1 ns | **2.5× faster** |
| Euclidean distance | 74.5 ns | 26.6 ns | **2.8× faster** |
| Recall@10 (dim 32) | 0.994 | 0.999 | more accurate |
| Recall@10 (dim 768) | — | 0.972 | — |

Concurrency cost nothing on the serial path: search stayed at 2 allocs/op and the
recall figures are unchanged to three decimals.

### The price of tombstones

Dead slots stay on the frontier and keep `results` under-filled, which loosens
the pruning bound and widens the search. `BenchmarkSearchTombstones` measures
the curve a compaction threshold should be set against:

| Tombstones | Search | vs. clean |
|---|---|---|
| 0% | 108 µs | — |
| 25% | 127 µs | 1.2× |
| 50% | 165 µs | 1.6× |
| 75% | 262 µs | 2.5× |

Allocations stay at 2/op throughout — tombstones cost time, not memory churn.
Recall does not degrade: with half the graph deleted, recall@10 against brute
force over the survivors is 1.000.

### The price of an update

| Operation | Cost | |
|---|---|---|
| `Insert`, new id | 713 µs, 208 allocs | — |
| `Insert`, replacing an id | 776 µs, 210 allocs | insert + tombstone, +9% |
| `Insert`, vector unchanged | **332 ns**, 2 allocs | the WAL-replay path |

Recall@10 after replacing half the graph is 0.999 — updates are held to the same
bar as inserts.

### When to compact — read both curves, not one

| Dead slots | Search | `Compact()` pause, 5k × 128 | Reclaims |
|---|---|---|---|
| 25% | 1.2× | 2.61 s | 25% of slots |
| 50% | 1.6× | 1.69 s | 50% of slots |
| 75% | 2.5× | 0.79 s | 75% of slots |

A compaction is a full index build over the **survivors**, so its pause tracks
how many vectors live, not how many get reclaimed — which makes compacting early
the worst of both: a longer stop-the-world pause, more often, handing back less
memory.

The search curve on its own argues for a 25% threshold. Both curves together
argue for **~50%**, where the standing cost is 1.6× on search and a graph
carrying 2× the slots it needs. That is the trade to tune; `Stats().DeadRatio()`
is the number to tune it on.

## How this package is tested

Recall numbers are only worth as much as what produced them, so the suite is
built in three layers that check different things.

**Fixed-configuration tests pin absolute numbers.** `TestRecallVsBruteForce`
(0.999 at dim 32), `TestRecallHighDimension` (0.972 at dim 768) and
`TestRecallIsStableAcrossSeeds` vary nothing, so a regression moves them
immediately. The last one exists because every other recall test pins a single
seed, which measures that seed as much as the index — it runs five and fails if
the spread exceeds 0.05.

**Sweeps assert shape.** [`recall_test.go`](recall_test.go) varies dimension,
corpus size, `ef`, `M`, metric, data distribution and tombstone ratio. Absolute
floors there are loose on purpose — recall at a fixed `ef` genuinely falls as
either corpus or dimension grows, so one threshold across the whole grid could
only ever be its hardest cell's. What the sweeps actually defend are the
*relationships*: recall must not fall as `ef` rises, must not fall as `M` rises,
must recover under a wide search whatever the metric, and must not drop when a
graph is compacted.

**Component tests cover what recall can only measure indirectly.** These were
the real gaps:

- [`distance_test.go`](distance_test.go) — the kernels are unrolled four-wide
  with a scalar tail, and every other test in the package uses a dimension
  divisible by four. They are now checked against a float64 reference at 28
  dimensions including 1, 2, 3, 5, 129 and 769, plus a case where the *only*
  difference between two inputs is the tail element.
- [`pq_test.go`](pq_test.go) — the hand-written heaps exist to avoid
  `container/heap`'s boxing, which makes the sift loops ours to get right. Their
  invariant is now checked after every push and pop under randomized interleaving,
  not just on a clean drain.
- [`visited_test.go`](visited_test.go) — including the wraparound branch. When
  the generation counter laps `MaxUint32`, every stale stamp would suddenly match
  the current generation and a search would treat the whole graph as already
  visited, returning almost nothing. Real traffic reaches that once every 2³²
  searches; the test drives the counter there directly.

The sweep harness doubles as the benchmark harness — `-results <path>` widens
the grid and writes every cell to CSV for [`docs/benchmarks`](../../docs/benchmarks)
to plot. Same code, so a number in a README and a threshold in CI cannot disagree.

### What measuring changed

Two things in this README were wrong before the sweeps existed, which is the
argument for having them:

- **A guessed compaction threshold.** The tombstone cost curve alone suggested
  compacting at ~25% dead. Measuring the *other* side showed the pause tracks
  survivors, so compacting early costs more and reclaims less — the guidance
  moved to ~50%.
- **A plausible explanation that wasn't true.** Recall dropping at larger corpora
  looked like it might be the all-positive test corpus putting every vector in
  one orthant. Measured side by side, the centered and positive corpora score
  within 1.5 points of each other. The cause was corpus size at a fixed `ef`, and
  nothing to do with the data's shape.
- **An overstated case for raising `M`.** Reading the `M` and `ef` charts
  together suggested M=32 was roughly twice as fast as M=16 at matched recall.
  It is not — those two charts each hold the *other* knob at its default, so the
  points being compared sat at different recall levels. The 2-D grid puts the
  real figure at **~10% latency for 6× the build time**, which turns "raise M"
  from advice into a narrow special case.

## Usage

```go
g, _ := hnsw.New(hnsw.DefaultConfig(128, hnsw.Cosine))
_ = g.Insert("doc1", vec1)

results, _ := g.Search(query, 10 /*k*/, 64 /*ef*/)
for _, r := range results {
    fmt.Println(r.ID, r.Distance) // ascending; smaller = closer
}
```

## Algorithm at a glance

**Insert(id, vec):** copy+normalize → if the id is live, return early when the
vector is unchanged, otherwise tombstone its slot → draw a random top level →
greedily descend to `level+1` → for each layer down to 0, `searchLayer` with
`EfConstruction`, alpha-select neighbors, connect both ways, prune → promote to
entry point if it reached a new top level.

**Search(query, k, ef):** normalize query → greedily descend upper layers → one
wide `searchLayer` on layer 0 → return the `k` closest.

**Compact():** nothing to do if no slot is dead → otherwise build a replacement
graph, re-inserting every live vector in slot order with its stored vector →
swap `nodes` / `ids` / `entry` / `maxLevel` across and zero the tombstone count.

## Not implemented yet (deliberately)

Persistence (WAL + snapshots), fine-grained write locking, **online** compaction
— `Compact` exists but stops the world — and SIMD assembly. Each is a separate
upcoming slice — see `docs/MIGRATION.md`.

```bash
go test ./internal/hnsw/ -v                      # correctness + recall
go test ./internal/hnsw/ -race                   # concurrency
go test ./internal/hnsw/ -run='^$' -bench=. -benchmem
```
