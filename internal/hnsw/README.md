# internal/hnsw

A small, readable HNSW (Hierarchical Navigable Small World) index — the core of
GoVecDB's approximate nearest-neighbor search. This is the v1 "learn it by
building it" implementation: **single-threaded, no memory pools or SIMD,
optimized for clarity over raw speed.** Optimizations arrive as later, separately
scoped commits.

## Why HNSW

Comparing a query against every stored vector is `O(N)` and collapses at scale.
HNSW builds a layered graph — sparse "express lanes" on top, every vector on the
bottom — so a search takes big jumps to the right region, then smaller steps
down to the true neighbors, visiting only a tiny fraction of nodes (`~O(log N)`).

## Files (one responsibility each)

| File | Responsibility |
|------|----------------|
| `distance.go` | `Metric` (Cosine / Euclidean / DotProduct). Every function returns **smaller = closer**, so the graph never branches on the metric. |
| `node.go` | A single vector: `id`, `vector`, and per-layer neighbor lists. |
| `pq.go` | Two heaps: a min-heap (explore closest first) and a max-heap (drop the farthest result when over `ef`). |
| `graph.go` | The algorithm: `New`, `Insert`, `Search`, plus internals (`searchLayer`, `greedyClosest`, neighbor selection & pruning). |
| `graph_test.go` | Empty-graph, validation, exact-match, and recall-vs-brute-force tests. |

## The three knobs

| Knob | Where | Adaptable? |
|------|-------|-----------|
| `M` — neighbors per node (layers > 0; layer 0 uses `2*M`) | `Config`, set once | **No** — structural; changing it means rebuilding. |
| `EfConstruction` — search width during inserts | `Config` | Kept fixed in practice. |
| `ef` — search width at query time | `Search(query, k, ef)` | **Yes** — per query; auto-clamped to `>= k`. Bigger `ef` = better recall, slower. |

## Usage

```go
g, _ := hnsw.New(hnsw.DefaultConfig(128, hnsw.Cosine)) // dim=128
_ = g.Insert("doc1", vec1)
_ = g.Insert("doc2", vec2)

results, _ := g.Search(query, 10 /*k*/, 64 /*ef*/)
for _, r := range results {
    fmt.Println(r.ID, r.Distance) // ascending distance; smaller = closer
}
```

## Algorithm at a glance

**Insert(id, vec):**
1. Draw a random top level (exponential decay — most nodes land on layer 0).
2. Greedily descend from the current entry point down to `level+1` (`ef=1`) to
   reach a good starting region.
3. From `min(maxLevel, level)` down to 0: `searchLayer` with `EfConstruction`,
   select the closest `M`, connect both directions, prune neighbors back to the
   layer cap.
4. If the new node reached a higher level than any existing node, it becomes the
   entry point.

**Search(query, k, ef):**
1. Greedily descend the upper layers to the right region.
2. One wide `searchLayer` on layer 0 with `ef`.
3. Return the `k` closest, sorted ascending by distance.

## How state maps to the concepts

- **Empty graph = empty container.** `entry = -1`, zero nodes, no memory spent on
  the graph until the first `Insert`.
- **No thread-safety yet.** Callers serialize access; a concurrent version is a
  later step.

## Not implemented yet (deliberately)

Delete / update, persistence (WAL + snapshots), the diverse-neighbor selection
heuristic, concurrency, and SIMD/pooling. Each is a separate upcoming slice — see
`docs/MIGRATION.md`.

## Test

```bash
go test ./internal/hnsw/ -v
```

Current recall@10 vs brute force on the test set: **~0.99**.
