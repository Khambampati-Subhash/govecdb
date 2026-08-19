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
| `insert.go` | `Insert` — building the graph. |
| `search.go` | `Result`, `Search`, and the primitives it rides on: `greedyClosest`, `searchLayer`. |
| `neighbors.go` | Edge management: alpha-pruned `selectNeighbors`, `pruneConnections`, `connect`, adjacency lookups. |
| `node.go` | A single vector: `id`, `vector`, per-layer neighbor lists. |
| `distance.go` | `Metric` (Cosine / Euclidean / DotProduct) + unrolled kernels. Everything returns **smaller = closer**, so the graph never branches on the metric. |
| `pq.go` | Hand-written min/max heaps over `[]candidate` — no `container/heap`, no interface boxing. |
| `visited.go` | Generation-stamped visited set, reused across searches. |
| `state.go` | `searchState`: the pooled per-traversal scratch that makes `Search` read-only. |
| `graph_test.go` | Correctness + recall-vs-brute-force at 32 and 768 dimensions. |
| `concurrent_test.go` | Parallel-vs-serial equivalence, mixed reader/writer race coverage. |
| `bench_test.go` | Insert / search / distance benchmarks. |

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

**Insert(id, vec):** copy+normalize → draw a random top level → greedily descend
to `level+1` → for each layer down to 0, `searchLayer` with `EfConstruction`,
alpha-select neighbors, connect both ways, prune → promote to entry point if it
reached a new top level.

**Search(query, k, ef):** normalize query → greedily descend upper layers → one
wide `searchLayer` on layer 0 → return the `k` closest.

## Not implemented yet (deliberately)

Delete / update, persistence (WAL + snapshots), fine-grained write locking, and
SIMD assembly. Each is a separate upcoming slice — see `docs/MIGRATION.md`.

```bash
go test ./internal/hnsw/ -v                      # correctness + recall
go test ./internal/hnsw/ -race                   # concurrency
go test ./internal/hnsw/ -run='^$' -bench=. -benchmem
```
