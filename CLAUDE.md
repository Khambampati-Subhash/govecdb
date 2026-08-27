# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

**GoVecDB** — a high-performance, embeddable **vector database in pure Go** (no
CGO). It stores embeddings and answers "what is most similar to this?" using an
**HNSW** approximate-nearest-neighbor index. WAL persistence is the next phase,
not yet built.

Module path: `github.com/khambampati-subhash/govecdb` · Go 1.24+ (built with 1.25).
**Zero third-party dependencies** — `go.mod` has no `require` block and there is no
`go.sum`. Do not add a dependency without asking; stdlib-only is a design goal.

## Current effort: v1 rebuild (active)

Branch **`v1-restructure`**. Strategy: **rebuild from scratch, one subsystem at a
time**, using the old implementation as a reference in git history rather than as a
source to copy. Read `docs/MIGRATION.md` before making structural changes — it has
the current state, the ordered steps, and the WAL design constraints.

Scope for v1: **embeddable library only** (no cluster / REST server / gRPC — those
stay in `main` history and return in v2).

### The codebase is `internal/hnsw/` — that's all of it
The from-scratch HNSW index (see its `README.md`) is currently the entire tree, and
it is the reference for style: small single-responsibility files, comments that
explain *why*, tested against brute-force recall.

**Next phase: `internal/wal/`.** Design constraints are in `docs/MIGRATION.md`.

### The legacy code is gone
Every previous package (`index/`, `store/`, `persist/`, `api/`, `collection/`,
`filter/`, `cluster/`, `segment/`, `proto/`, and the dead experiments) was deleted
in the clean-slate commit. **Nothing is lost** — `main` has it all. To consult the
old implementation rather than resurrect it:

```bash
git show main:persist/wal.go
git log main --oneline -- persist/
```

Do not restore these packages into the tree. If something there is worth having,
rewrite it to the current bar.

## Commands

```bash
go build ./...                 # build everything
go vet ./...                   # static checks
go test ./...                  # all tests
go test ./internal/hnsw/ -v    # the new HNSW package
go test ./... -race            # race detector (run before merging)
```

Measurement (the sweeps double as the benchmark harness — same code, so a README
number and a CI threshold cannot disagree):

```bash
go test ./internal/hnsw/ -run TestSweep -results docs/benchmarks/results.csv -timeout 40m
go run docs/benchmarks/plot.go   # regenerates the SVGs the READMEs embed
```

Without `-results` the same sweeps run a small grid and assert thresholds, inside
the normal `go test ./...`. `docs/benchmarks/plot.go` is `//go:build ignore` and
stdlib-only — do not add a charting dependency.

The sweeps **skip themselves under `-race`** (`skipUnderRace`, via a `race`
build-tag constant): they are single-goroutine, so the detector observes nothing
while costing ~10× and pushing the package past the default 10-minute timeout.
Race coverage lives in the five `TestConcurrent*` tests. Do not "fix" the skip by
raising the timeout.

If `go` is not on PATH: `export PATH=$PATH:/usr/local/go/bin`.

## Conventions

- **SOLID first.** One package = one responsibility. Depend on interfaces
  (`Index`, `Store`, `WAL`, `DistanceFunc`, ...), inject concretes. Prefer
  Factory + functional-options construction over many constructors.
- **Distances return "smaller = closer"** everywhere, so callers never branch on
  the metric.
- **Green-gated steps.** Each change keeps `go build`, `go vet`, and `go test`
  passing; land it as its own focused commit.
- **Comments explain *why*,** not the obvious *what*. Match the density in
  `internal/hnsw/`.

### Git
- Work on the `v1-restructure` branch; do not commit to `main` directly.
- **Commit messages must NOT include a `Co-Authored-By` trailer** (repo owner
  preference).
- Commit/push only when asked.

## HNSW quick reference

- `M` (neighbors/node) is **structural** — set once, changing it needs a rebuild.
- `EfConstruction` is the build-time search width (kept fixed, ~100–200).
- `ef` is the **query-time** knob in `Search(query, k, ef)`; must be `>= k`, bigger
  = higher recall + slower. **Recall at a fixed `ef` falls as `N` or dimension
  grows** — 0.997 at 500 vectors down to 0.652 at 20,000, all at `ef=64`. That is
  not degradation, it is a fixed-width beam covering less of a bigger space, and
  it means `ef=64` is a starting point rather than a default that holds. Measured
  in `docs/benchmarks/`.
- Empty graph = empty container: no graph memory until the first insert.
- The graph is **safe for concurrent use**: `Search` holds `RLock` and runs in
  parallel, `Insert` holds the write lock. Per-traversal scratch comes from a
  pooled `searchState` (`state.go`) — that is *why* `Search` can be a reader, so
  never move scratch back onto `Graph`. Concurrent writers still serialize.
- Insert **copies** the caller's vector (and normalizes it for Cosine), so the graph
  never aliases a reused caller buffer.
- `Insert` is an **upsert** — there is no `Update`. A second Insert under a live id
  tombstones the old slot and builds a new one, because that slot's *inbound* edges
  were chosen for the old vector and pruning makes them impossible to find without
  an O(N·M) scan. So updates create tombstones exactly like deletes do. Re-inserting
  an unchanged vector is an early return (`slices.Equal` against the stored form) —
  that is the WAL-replay path and it must not cost a slot. The tombstone happens
  *inside* Insert's write lock, never via `Delete`: an update must not be observable
  as a disappearance.
- `Delete` is a **tombstone**, never a real removal: the slot keeps its index and
  its edges. In `searchLayer` the frontier (`cands`) admits dead nodes — they are
  still bridges — while `results` admits only live ones. Do not "simplify" this
  into filtering the final result slice; that silently returns fewer than `k`.
  `pruneConnections` **demotes** tombstones so they cannot evict a fresh live
  edge and strand a vector. `Len` is live-only; `Stats` shows the tombstones.
- `Compact()` is the only thing that reclaims dead slots. It **rebuilds** — a new
  graph over the live vectors, swapped in whole — because neighbor lists are slot
  indices, so nothing may ever be renumbered in place. It re-inserts *stored*
  vectors via `insertPrepared` (no re-copy, and no re-normalize: that drifts a
  unit vector by an ulp and the rebuild would stop being bit-equal to a fresh
  build, which `TestCompactMatchesAFreshBuild` asserts). It **stops the world**;
  the index never self-triggers, callers poll `Stats().DeadRatio()`. Threshold
  ~0.5, not 0.25: the pause tracks *survivors*, so compacting early costs more
  and reclaims less.
- Durability model *(phase 2, not yet built)*: **write to WAL first, then apply to
  the in-memory graph**; on recovery, replay the WAL to rebuild the graph — the
  graph is derived state, never the source of truth.

## Locked baselines — do not regress

Any index change must hold these; they are enforced by tests and `-benchmem`:

| Baseline | Value | Guarded by |
|---|---|---|
| Recall@10, dim 32 | 0.999 | `TestRecallVsBruteForce` |
| Recall@10, dim 768 | 0.972 | `TestRecallHighDimension` |
| Search allocations | 2 allocs/op | `BenchmarkSearch -benchmem` |
| Recall spread across seeds | ≤ 0.05 | `TestRecallIsStableAcrossSeeds` |

The sweep tests in `recall_test.go` defend **shape**, not absolute values: recall
must not fall as `ef` rises, nor as `M` rises, must recover under a wide search
for every metric, and must not drop when a graph is compacted. Their absolute
floors are deliberately loose — see the note at the top of that file before
"tightening" one.
