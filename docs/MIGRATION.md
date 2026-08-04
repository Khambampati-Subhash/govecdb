# GoVecDB v1 — Clean Restructure (Salvage & Rebuild)

> Branch: `v1-restructure` · Same module path · History preserved · Embeddable library only.
> Strategy: fresh SOLID layout, **port the proven & tested core**, drop dead + out-of-scope code.
> Gate: `go build ./... && go vet ./... && go test ./... -race` green after every step.

## Baseline (verified before starting)
Core packages pass tests today: `api`, `index`, `store`, `persist`, `collection`, `segment`.
`filter` builds but has **no tests** (we add them). This is the correctness we preserve.

## Target layout

```
govecdb/
├── vector.go            # public API: Vector, SearchRequest, SearchResult, DistanceMetric
├── filter.go            # public filter expressions (FieldFilter, LogicalFilter, helpers)
├── db.go                # DB + Collection interfaces + facade
├── options.go           # functional-options construction
├── errors.go            # exported sentinel errors
├── internal/
│   ├── distance/        # Cosine/Euclidean/Dot/Manhattan kernels        (Strategy)
│   ├── hnsw/            # HNSW index engine (live subset, cleaned)       (SRP)
│   ├── store/          # in-memory vector store                         (Repository)
│   ├── wal/           # write-ahead log                                 (SRP)
│   ├── snapshot/     # snapshots + recovery
│   ├── filter/      # metadata query engine (inverted + numeric index)
│   └── obs/        # Logger + Metrics interfaces, no-op defaults         (DIP)
├── cmd/govecdb-bench/  # benchmark CLI (trimmed)
├── examples/
├── docs/
└── README.md
```

## Source → destination map (salvage)

| New location | Ported from | Notes |
|---|---|---|
| `vector.go`, `filter.go`, `db.go`, `errors.go` | `api/types.go` | Split the 573-line types file by concern; keep interfaces |
| `internal/distance/` | `index/simd_distance.go` + live helpers from `index/optimized_metrics.go` | One canonical kernel set; drop the 3-way duplication |
| `internal/hnsw/` | `index/{hnsw,graph,node,heap,types,context,metrics,advanced_memory_pool}.go` + needed helpers from `optimized_graph.go` (`connWithDist`, `max`, `ConnectionSet`, `OptimizedHNSWNode`) | Untangle helpers into this pkg; behind an `Index` interface |
| `internal/store/` | `store/{store,mem_store}.go` | Repository behind `Store` interface |
| `internal/wal/` | `persist/wal.go` (+ `persist/types.go` records) | Single WAL; drop `store/wal.go` + `persist/optimized_persistence.go` |
| `internal/snapshot/` | `persist/snapshot.go` | |
| `internal/filter/` | `filter/{hybrid_engine,inverted_index,numeric_index,interfaces}.go` | + new tests |
| `db.go` facade | `collection/{collection,persistent,manifest}.go` | Merge in-memory + persistent into one type; persistence via injected WAL/snapshot (Strategy), constructed with options |

## Dropped for v1 (history keeps them; recoverable from `main`)

**Dead / orphaned (unused by anything):**
`utils/`, `diskann/`, `quantization/`, `batch/`, `streaming/`, `internal/` (old),
`accuracy/`.

**Superseded duplicates:**
`index/{concurrent_index,optimized_graph,optimized_metrics,optimized_structures,simd_impl,multi_index}.go`,
`collection/enhanced_collection.go`, `store/wal.go`, `persist/optimized_persistence.go`,
`api/streaming_api.go`.

**Out of v1 scope (embeddable-first) — revisit in v2:**
`cluster/`, `api/rest/`, `proto/`, `segment/` (only used by dropped enhanced collection),
`client/`, `cmd/{server,benchmark_chroma,benchmark_suite,quality_check}`.

## SOLID / patterns applied
- **SRP** — one package = one responsibility; split god-files (`api/types.go`, `collection`).
- **DIP** — `DB`/`Collection` depend on `Index`, `Store`, `WAL`, `Snapshotter`, `DistanceFunc`,
  `Logger`, `Metrics` interfaces; concrete impls injected.
- **OCP / Strategy** — distance metric, index type, persistence pluggable.
- **Factory + Functional Options** — `govecdb.Open(cfg, WithWAL(dir), WithMetric(...), WithLogger(...))`.
- **ISP / Liskov** — small interfaces so `FlatIndex` and `HNSWIndex` are interchangeable.

## Ordered execution (each = one green-gated commit)
1. **Scaffold + public types** — `vector.go`/`filter.go`/`db.go`/`errors.go` from `api/types.go`; define core interfaces. *(no behavior change)*
2. **`internal/distance`** — port canonical kernels + tests.
3. **`internal/hnsw`** — port live index, untangle helpers, put behind `Index` interface + factory.
4. **`internal/store`** — port mem store behind `Store` interface.
5. **`internal/wal` + `internal/snapshot`** — single WAL + snapshot behind interfaces.
6. **`internal/filter`** — port + **add tests** (currently zero).
7. **`db.go` facade** — unify in-memory + persistent collection; functional options.
8. **Delete dropped packages**; `go mod tidy`.
9. **Examples + README + docs** refreshed to the new API.
10. **Full `go test ./... -race`**, recall regression test, benchmark vs baseline.

## Risks
- **HNSW untangle (step 3)** is the delicate one — isolated commit, tests green before/after.
- **Recall regression** — lock baseline recall numbers before touching the index.
- Every step reverts cleanly (separate commits on a throwaway branch).
