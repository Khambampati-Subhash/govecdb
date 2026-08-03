# GoVecDB — Enhancement & Restructuring Plan

> Status: **proposed** (review before executing). Nothing here changes code until each phase's gate passes.
> Baseline: `go build ./...` clean, `go vet ./...` clean, tests compile. ~45,700 LOC across ~90 Go files.

## 1. Executive summary

The **live request path is small and reasonably well-designed**:

```
api (interfaces) → collection.VectorCollection → index.HNSWIndex
                                                → store.MemoryStore
                                                → persist.WAL
                                                → filter / segment
   (wired via api.VectorStore / api.VectorIndex interfaces + an index adapter)
```

It already uses dependency injection, interface abstractions, and an adapter. The
problem is **not** the live code — it's that roughly **40% of the repo is dead or
duplicate scaffolding** that never connects to anything. Removing it is the
highest-ROI, lowest-risk work and eliminates most *apparent* SOLID violations.

Decisions taken:
- **Delete** the orphaned experimental packages.
- Save this plan first; review before any code change.

## 2. Verified findings (import-graph + symbol analysis)

### 2a. Orphaned packages — imported by nobody (clean deletes)
| Package | LOC (approx) | Notes |
|---|---|---|
| `utils/` | ~1,300 | `dist.go` + `pool.go`; `index/` has its own SIMD + pools |
| `diskann/` | ~700 | Standalone index, never wired |
| `quantization/` | ~1,000 | `advanced_` + `vector_` quantization, never wired |
| `batch/` | ~1,000 | `optimized_batch.go`, never wired |
| `streaming/` | ~600 | `optimized_streaming.go`, never wired |

### 2b. `internal/` — imported by nobody (**decision required**, not auto-delete)
`internal/{logging,metrics,health,errors,monitoring,benchmark}` (~2,500 LOC) is
observability/plumbing that is currently **unwired**. Options: (a) wire it into the
collection via injected `Logger`/`MetricsRecorder` interfaces, or (b) delete it.
Do not leave it dead. Recommendation: **wire logging + metrics + health**, delete
`monitoring` + `benchmark` if redundant with `cmd/benchmark*`.

### 2c. `index/` internal duplication (NOT a clean delete — entangled)
Three graph implementations, three distance impls, two metrics files. The live path
uses `HNSWIndex`/`HNSWGraph`. But the "optimized" variants are **entangled** — live
files borrow helpers from them:

| Variant file | Disposition | Why |
|---|---|---|
| `concurrent_index.go` | **clean delete** | self-contained, nothing else references it |
| `multi_index.go` (IVF/LSH/PQ) | **clean delete** | self-contained; delete unless promoting as a real feature |
| `optimized_graph.go` | **untangle then delete** | live `graph.go`/`heap.go`/`hnsw.go`/`advanced_memory_pool.go` use `connWithDist`, `max`, `ConnectionSet`, `OptimizedHNSWNode`, `candidateWithDist` from here |
| `optimized_metrics.go` | **untangle then delete** | live `simd_distance.go` calls `getOptimizedDistanceFunc` |
| `optimized_structures.go` | **untangle then delete** | verify remaining helpers, migrate live ones |
| `simd_impl.go` | **clean delete** | `DotProductAVX`/`EuclideanAVX` referenced nowhere |

> Untangle = move the still-needed helpers into a live file (e.g. `graph.go` /
> `distance.go`), then delete the rest. This is a small refactor, not `rm`.

### 2d. `collection/` variants
`NewVectorCollection` is the **only** constructor used externally (7 call sites).
`enhanced_collection.go` (~1,000 LOC), `persistent.go`, and `api/streaming_api.go`
are referenced only within their own package/tests. Decide: fold persistence into
`VectorCollection` as an injected strategy, or delete the unused variant.

### 2e. WAL duplication
Two WAL implementations: `persist/wal.go` and `store/wal.go` (plus
`persist/optimized_persistence.go`). Unify to one, injected where needed.

### 2f. Documentation accuracy bugs (fix regardless of phase)
1. README claims *"Hand-written Assembly (AVX2 for AMD64, NEON for ARM64)"* — there
   are **zero `.s` files**. The "SIMD" is pure Go with manual loop unrolling. Reword
   to "vectorized pure-Go with loop unrolling," or actually add asm.
2. README Quick Start calls `collection.NewPersistentCollection(config, "./data")`
   — **that function does not exist** (real: `NewPersistentVectorCollection(*PersistentCollectionConfig)`).
   The headline sample does not compile.
3. Recall degrades badly at high dim (0.66 → 0.35 @ 16384). Worth a real fix, not
   just a doc note.

### 2g. Hygiene
- All 25 deps in `go.mod` are marked `// indirect` (grpc, raft, chroma-go are
  direct) — `go mod tidy` never run properly.
- `filter/` has ~2,600 LOC and **zero tests**.
- 5 near-identical benchmark shell scripts at repo root.

## 3. Target architecture (post-cleanup)

```
                +------------------+
                |   api  (types,   |   public contract: interfaces + DTOs only
                |   interfaces)    |
                +--------+---------+
                         |
        +----------------+-----------------+
        |                                  |
+-------v-------+                 +--------v---------+
| collection    |  injects        | transport/       |
| VectorColl.   +---------------> | rest, grpc       |
+---+---+---+---+  interfaces     +------------------+
    |   |   |   \
    v   v   v    v
 index store persist filter        each behind an api.* interface
   |
  factory: index.New(cfg) -> HNSW | Flat | (IVF...)   (Strategy + Factory)
```

Cross-cutting (`internal/`): `Logger`, `MetricsRecorder`, `HealthChecker`
interfaces injected into `collection`, not imported concretely.

## 4. Phased execution plan

### Phase 0 — Safety net  *(gate for everything)*
- [ ] Makefile/CI target: `go build ./... && go vet ./... && go test ./... -race`.
- [ ] Run `go mod tidy`; verify direct vs indirect deps are correct.
- [ ] Capture current benchmark numbers as `docs/BASELINE.md` (per-dim recall + QPS).
- **Gate:** all green + baseline recorded.

### Phase 1 — Remove dead weight  *(≈ −10k LOC)*
- [ ] Delete clean-delete packages: `utils/`, `diskann/`, `quantization/`,
      `batch/`, `streaming/`.
- [ ] Delete `index/concurrent_index.go`, `index/multi_index.go`, `index/simd_impl.go`.
- [ ] Untangle-then-delete `index/optimized_graph.go`, `optimized_metrics.go`,
      `optimized_structures.go`: migrate `connWithDist`, `max`, `ConnectionSet`,
      `OptimizedHNSWNode`, `candidateWithDist`, `getOptimizedDistanceFunc` into a
      live file, then remove the variant files.
- [ ] Collapse `collection/enhanced_collection.go` (+ decide `persistent.go`).
- [ ] `internal/` decision: wire `logging`/`metrics`/`health`, or delete.
- **Gate after each deletion:** build + vet + `go test -race` still green.

### Phase 2 — Public API & duplication cleanup
- [ ] Fix README (asm claim, `NewPersistentCollection` sample, verify all samples compile).
- [ ] Unify WAL to one implementation.
- [ ] Clarify `api/` (contract) vs a `transport/` layer holding `rest` (+ future `grpc`).

### Phase 3 — SOLID & design-pattern pass on surviving code
- [ ] **SRP:** extract background optimization out of `VectorCollection` (currently
      20 methods / 16 fields) into an injected `Optimizer`; extract stats/metadata.
- [ ] **Factory:** `index.New(cfg)` returning the right `api.VectorIndex` impl.
- [ ] **Strategy:** distance metric, (optional) quantization, optimizer as pluggable.
- [ ] **Options pattern:** `NewVectorCollection(cfg, ...Option)` — retire multi-ctor sprawl.
- [ ] **DIP:** inject `persist`, `filter`, `segment` via interfaces, not concrete types.
- [ ] Canonicalize a single `distance.go`.

### Phase 4 — Testing & correctness
- [ ] Add `filter/` tests (currently zero).
- [ ] Recall regression test; investigate high-dim recall collapse (0.35 @ 16384).
- [ ] `go test -race ./...` in CI.

### Phase 5 — Structure & docs
- [ ] Standardize layout (private → `internal/`, public at top level).
- [ ] Regenerate `docs/ARCHITECTURE.md` from the real post-cleanup design.
- [ ] Consolidate the 5 root benchmark scripts into one parameterized script / Make targets.

## 5. Sequencing rationale & risks
- **Phase 1 before Phase 3**: never polish (SOLID-ify) code that's about to be deleted.
- **Risk — index untangling:** the optimized files share helpers with the live path;
  do it as an isolated commit with build/test gate, not bundled with deletions.
- **Risk — recall regression:** lock in baseline recall (Phase 0) so any index change
  is measured, not guessed.
- **Every phase is independently revertable** (separate commits, green gate each step).

## 6. Suggested commit sequence
1. `chore: go mod tidy + CI verify target + baseline benchmarks`
2. `refactor: remove orphaned packages (utils, diskann, quantization, batch, streaming)`
3. `refactor: remove unused index variants (concurrent, multi_index, simd_impl)`
4. `refactor: untangle and remove optimized_* index files`
5. `refactor: consolidate collection constructors`
6. `docs: fix README inaccuracies (asm claim, NewPersistentCollection sample)`
7. `refactor: unify WAL`
8. `refactor: index factory + options pattern + optimizer extraction`
9. `test: add filter tests + recall regression`
10. `docs: regenerate architecture; consolidate benchmark scripts`
