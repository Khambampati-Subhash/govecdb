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
`docs/DURABILITY.md` is the companion: what survives which failure, what each
guarantee costs, and every latency number in one place. **Update it when you
change a durability guarantee or move a benchmark** — it is the document a user
would be misled by if it went stale.

Scope for v1: **embeddable library only** (no cluster / REST server / gRPC — those
stay in `main` history and return in v2).

### The codebase is `internal/hnsw/`, `internal/wal/` and `internal/snapshot/`
Each has its own `README.md`, and `internal/hnsw/` is the reference for style:
small single-responsibility files, comments that explain *why*, measured rather
than assumed.

- `internal/hnsw/` — the index. Complete: concurrent reads, tombstone delete,
  upsert, compaction, and a measurement harness behind `-results`.
- `internal/wal/` — durability. Complete: record format, append-only writer with
  segment rotation and sync policies, and `Replay` — a CRC-validating scan that
  truncates torn tails and carries the sequence forward. Checkpointing and
  segment truncation are still open, and no longer blocked.
- `internal/snapshot/` — point-in-time state. The durable **store** is done:
  atomic writes, checksummed framing keyed by WAL sequence, discovery, fallback
  and retention, with an **opaque payload**. **The graph codec (`hnsw.Graph` ↔
  bytes) is the next phase**, and it is a format decision rather than plumbing.
  Design constraints are in `docs/MIGRATION.md`.

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
go build ./...                   # build everything
go vet ./...                     # static checks
go test ./...                    # all tests
go test ./internal/hnsw/ -v      # the index
go test ./internal/wal/ -v       # the write-ahead log
go test ./internal/snapshot/ -v  # point-in-time state
go test ./... -race              # race detector (run before merging)
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
- **`codec.go` serializes the graph** (`(*Graph).WriteTo` / `hnsw.Read`), which
  freezes the internal representation on disk — neighbor lists are slot indices,
  so changing `node` is a format migration. `TestCodecLayoutIsFrozen` pins it.
  Store only what cannot be recomputed: nodes, `entry`, `maxLevel`. `ids` and
  `numDeleted` are **derived on load**, so a file cannot contradict itself.
  `Read` validates *structure* (neighbor indices in range, counts within
  `maxConn`, entry live and at `maxLevel`) because a checksum proves the bytes
  are what was written, not that they describe a walkable graph. It carries **no
  checksum of its own** — `internal/snapshot` verifies the payload first, and
  `WriteTo` is 5 allocs regardless of graph size; keep it that way.

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
  **4.04 ms/append vs 692 ns** for never: ~5,800×. Append is 0 allocs.
- **The fast policies do not survive a process crash either.** Records sit in a
  64 KiB *user-space* bufio buffer, so under `SyncInterval`/`SyncNever` an
  acknowledged write may not have reached the kernel at all. Do not restore the
  older claim that `SyncNever` "reached the OS" — it hasn't, until the buffer
  fills. Full table in `docs/DURABILITY.md`.
- **`openSegment` fsyncs the directory.** `fsync` on a file makes its contents
  durable and says nothing about the directory entry naming it, so without this a
  crash can take a freshly created segment away along with `SyncAlways` writes
  already inside it. Do not remove it to make rotation faster: it is why rotation
  is 8.6 ms rather than 4.8, which amortizes to **0.27 µs/record** at 64 MiB
  segments — below even `SyncNever`'s per-append cost.
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

## Snapshot quick reference (`internal/snapshot`) — store done, codec next

- Format: `magic "GVSS" | version | reserved | seq` (16B header), payload, then
  `crc32c | length` (12B trailer). Little-endian. `TestLayoutIsFrozen` pins the
  sizes — changing one is a migration, not an edit.
- **The checksum is in a trailer, not the header,** because the payload is
  *streamed*. A snapshot is gigabytes where a WAL record is kilobytes, so nothing
  may hold it all in memory. It covers the header and payload; the length field
  is cross-checked against the file size instead, which is stronger.
- **The seq is the point of the file, not metadata on it.** A snapshot whose
  sequence is wrong by one replays the log from the wrong place. It is inside the
  file *and* in the name, and a disagreement is `ErrSeqMismatch` — the header
  wins, because only the header is checksummed.
- **Atomicity comes from the rename, not the checksum**: temp → fsync → rename →
  **fsync the directory**. Do not drop that last fsync; without it a crash can
  leave the snapshot under neither name. The error is propagated on purpose —
  swallowing it silently downgrades the guarantee.
- **Nothing unverified reaches the caller.** `Load` hashes end to end *before* the
  callback sees a byte, so the callback runs at most once and never needs to undo.
  Costs +38% over streaming once (14.5 ms vs 10.5 at 64 MiB) because the apply
  pass reads the page cache at 18.2 GB/s. Do not "optimize" this into one pass.
- **A corrupt snapshot falls back to an older one; a callback error does not.**
  The first is disk rot, the second is a decoder bug, and falling back would hide
  it behind a slow startup.
- `Create` has a **~10 ms floor at any size** — two fsyncs. That is why snapshots
  ride a checkpoint interval in minutes, not the WAL's fsync interval in ms.
- **WAL truncation follows the *oldest retained* snapshot, never the newest**, and
  runs after `Prune` — otherwise the fallback copy is unusable but still stored.
- Payload is **opaque** to this package; `hnsw.(*Graph).WriteTo` produces it.
  The graph-vs-live-vectors question was settled by measurement and the answer is
  **serialize the graph**: for 1M × 128 that is ~0.37 s to verify and decode
  against ~703 s to rebuild, about 1,900×. See `docs/DURABILITY.md` §6.
- **No `Snapshotter` interface yet** — an implementation without a consumer. The
  WAL's `Replay` is the precedent: it sat on the interface as a promise until
  writing it showed it did not belong.

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
