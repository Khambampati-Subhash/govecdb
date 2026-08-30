# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

**GoVecDB** — a high-performance, embeddable **vector database in pure Go** (no
CGO). It stores embeddings and answers "what is most similar to this?" using an
**HNSW** approximate-nearest-neighbor index, with a write-ahead log for
durability (writer built; reader/recovery next).

Module path: `github.com/khambampati-subhash/govecdb` · Go 1.24+ (built with 1.25).
**Zero third-party dependencies** — `go.mod` has no `require` block and there is no
`go.sum`. Do not add a dependency without asking; stdlib-only is a design goal.

## Current effort: v1 rebuild (active)

Work lands on **`main`**. Strategy: **rebuild from scratch, one subsystem at a
time**, using the old implementation as a reference in git history rather than as a
source to copy. Read `docs/MIGRATION.md` before making structural changes — it has
the current state, the ordered steps, and the WAL design constraints.

Scope for v1: **embeddable library only** (no cluster / REST server / gRPC — those
stay in `main` history and return in v2).

### The codebase is `internal/hnsw/` and `internal/wal/`
Both have their own `README.md`, and `internal/hnsw/` is the reference for style:
small single-responsibility files, comments that explain *why*, measured rather
than assumed.

- `internal/hnsw/` — the index. Complete: concurrent reads, tombstone delete,
  upsert, compaction, and a measurement harness behind `-results`.
- `internal/wal/` — durability. Complete: record format, append-only writer with
  segment rotation and sync policies, and `Replay` — a CRC-validating scan that
  truncates torn tails and carries the sequence forward. **`internal/snapshot`
  is the next phase**; checkpointing and segment truncation land with it. Design
  constraints are in `docs/MIGRATION.md`.

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
- **Commit and push to `main` directly.** No feature branch, no PR — the repo
  owner asked for this explicitly (2026-08-28), superseding the earlier
  `v1-restructure` rule. That branch still exists and is level with `main`.
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
  in `docs/benchmarks/`. Use `SuggestedEf(n, k, target)` / `g.SuggestedEf(k, target)`
  rather than a hardcoded number: it fits the measured `ef ∝ n^0.78` curve, treats
  the target as a **floor** (0.95 measures ~0.97), and is guarded by
  `TestSuggestedEfAchievesTarget`, which builds real graphs and fails if a
  suggestion misses. Its constants carry deliberate margin — calibrated exactly on
  the sweep it undershot on 3 of 4 corpora.
- **Raising `M` is worth less than the M-vs-recall chart implies.** Compared at
  *equal recall* on the M×ef grid, M=32 beats M=16 by only ~10% latency (162µs vs
  180µs at ~0.96) for 6x the build time and ~2x the graph memory. M=16 is the right
  default; the two 1-D charts overstate the case because they compare points at
  different recall levels.
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
- Durability model: **write to WAL first, then apply to the in-memory graph**; on
  recovery, replay the WAL to rebuild the graph — the graph is derived state,
  never the source of truth.

## WAL quick reference (`internal/wal`) — writer + replay done

- Format: `magic "GVWL" | version | reserved` (8B file header), then
  `crc32c | type | seq | len | payload` (17B record header). Little-endian.
  `TestLayoutIsFrozen` pins those sizes — changing one is a migration, not an edit.
- **The checksum covers type, seq AND length**, not just the payload. A corrupt
  length is the dangerous one: it turns a bad read into an arbitrary allocation.
  Never allocate on a length that has not passed its checksum.
- **Type 0 is invalid on purpose** — zero-filled space must never decode as a record.
- The payload is **opaque** to this package; it knows nothing about vectors, which
  keeps it testable without the index. Domain encoding belongs a layer up.
- **`Open` always starts a new segment**, even when segments exist. A torn tail
  from power loss stops replay, so appending after it would bury good records
  behind a permanent stopping point. Do not "optimize" this into reopening the
  last segment.
- Rotation **fsyncs the old segment before creating the new one** regardless of
  sync policy — otherwise a crash leaves a hole in the *middle* of the log, which
  is the one shape recovery cannot repair.
- `MaxSegmentBytes` is a **truncation granularity, not a size cap**: an oversized
  record gets its own segment rather than being refused.
- **Failure is sticky.** The first write error ends the Writer; every later call
  returns it. Appending over a hole is how a durability bug becomes data loss.
- Sync policy zero value is **`SyncAlways`** — safe by omission. It costs
  **4.06 ms/append vs 1.01 µs** for interval: ~4,000×. Append is 0 allocs.
- **Recovery is `Replay(dir, opts, fn)`, a function — not a method on `WAL`.**
  It runs before a writer exists; a method would mean opening a writer in order
  to read, which creates a segment as a side effect of recovery. Feed
  `res.NextSeq()` into `Options.FirstSeq` when reopening.
- **A tear ends a segment, not the replay.** A damaged record truncates that
  segment and the scan continues with the next one. This is required, not
  lenient: `Open` always starts a new segment, so a second crash puts a torn tail
  in the *middle* of the directory. Do not "harden" this into tolerating damage
  only in the last segment — that makes a twice-crashed database unrecoverable.
  It is safe because failure is sticky and rotation fsyncs before the next
  segment exists, so the writer never put records behind a tear.
- **Truncation is logical.** The damaged bytes stay on disk; nothing will append
  to them. Recovery is a read, and rewriting the file would destroy the only
  evidence a crash happened.
- **Never read on an unverified length** — the length arrives before the checksum
  that would prove it. The reader takes the file size up front and refuses a
  length against both it and `MaxRecordBytes` before allocating or reading.
- Sequence numbers must **strictly increase**. Gaps are fine (that is what a tear
  leaves); a repeat is `ErrOutOfOrder`, and almost always means a `Writer` was
  opened without carrying `FirstSeq` forward.
- **Replayed payloads alias a reused buffer** — valid only during the callback,
  `Record.Clone()` to keep one. That contract is what makes replay 0 allocs/record
  (358 ns/record, 5.8 GB/s); the ~16 allocs are per *segment*, not per record.

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
