# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

**GoVecDB** — a high-performance, embeddable **vector database in pure Go** (no
CGO). It stores embeddings and answers "what is most similar to this?" using an
**HNSW** approximate-nearest-neighbor index, with a write-ahead log and
checksummed snapshots for durability, and a metadata query engine for filtering.

Module path: `github.com/khambampati-subhash/govecdb` · Go 1.24+ (built with 1.25).
**Zero third-party dependencies** — `go.mod` has no `require` block and there is no
`go.sum`. Do not add a dependency without asking; stdlib-only is a design goal.

## Where the project is: v1 complete, v2 started at the server end

Work lands on **`main`**. v1 was a **rebuild from scratch, one subsystem at a
time**, using the old implementation as a reference in git history rather than as
a source to copy — and it is **done**: an embeddable library with durability,
recovery and metadata filtering.

**v1.1.0 added the service**: collections (`service/`), a REST API (`httpapi/`)
and a daemon (`cmd/govecdbd`) — v2 items 4 and 7, taken out of order because
neither needed the observability seam and together they are what makes this
operable. They sit strictly above `DB`; embedding the library is unchanged.

Read `docs/MIGRATION.md` before making structural changes. It now holds both the
v1 record (why each subsystem is shaped the way it is) and **the v2 scope**, in
dependency order: observability seam → online compaction → fine-grained write
locking → ~~collections~~ → selectivity estimation → quantized index →
~~REST~~/gRPC → clustering. Do not start one of those without reading what blocks
it; several look independent and are not. `docs/SERVICE.md` is the service
manual and holds the module decision.

`docs/DURABILITY.md` is the companion: what survives which failure, what each
guarantee costs, and every latency number in one place. **Update it when you
change a durability guarantee or move a benchmark** — it is the document a user
would be misled by if it went stale. `docs/PLAN.md` and `docs/PLAN_PROGRESS.md`
are the original plan and its execution log.

**Two v2 constraints worth knowing before writing any of it.** The first is now
**decided rather than pending**: the library and the HTTP API are one module
because `net/http` is stdlib, and **gRPC and Raft go in a separate module** that
imports this one. Do not add a `require` block to `go.mod`; CI fails the build if
one appears. The second is unchanged: replication should be built on the WAL's
existing `Replay` and record format, and if it needs a format change, making it
before v2 ships is far cheaper than after.

### The codebase is the root package, `internal/`, and the service on top
The root package (`db.go`, `vector.go`, `filter.go`, `options.go`, `validate.go`,
`index.go`, `codec.go`, `recovery.go`, `errors.go`) is the public API. Under it:
`internal/hnsw`, `internal/wal`, `internal/snapshot`, `internal/store`,
`internal/filter`. Each internal package has its own `README.md`, and
`internal/hnsw/` is the reference for style: small single-responsibility files,
comments that explain *why*, measured rather than assumed.

**All 11 rebuild steps are done.** What remains is deferred work, not gaps:
online compaction, finer write locking, an observability seam, and a crash
harness — see `docs/MIGRATION.md`.

- `internal/hnsw/` — the index. Complete: concurrent reads, tombstone delete,
  upsert, compaction, serialization, filtered search, and a measurement harness
  behind `-results`.
- `internal/wal/` — durability. Complete: record format, append-only writer with
  segment rotation and sync policies, `Replay` — a CRC-validating scan that
  truncates torn tails and carries the sequence forward — and `Truncate`.
- `internal/snapshot/` — point-in-time state. Complete: atomic writes, checksummed
  framing keyed by WAL sequence, discovery, fallback and retention, with an
  **opaque payload**. The graph codec lives in `internal/hnsw/codec.go`.
- `internal/store/` — metadata storage, and `Match` for the filtered search path.
- `internal/filter/` — the metadata query engine.

**v2 items 4 and 7 shipped in v1.1.0**, as three packages strictly *above* `DB`
— nothing in the root package or `internal/` knows they exist, and adding them
changed no behaviour for anyone embedding the library:

- `service/` — collections: many independent databases in one directory, with
  on-disk specs, load-on-demand and idle eviction.
- `httpapi/` — a REST/JSON `http.Handler` over a `service.Manager`. `net/http`
  and `encoding/json` only.
- `cmd/govecdbd/` — the daemon: flags, TLS, signals, graceful shutdown.

See the service quick reference below, `docs/SERVICE.md`, and the two package
READMEs.

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
go test . -v                     # the public API
go test ./internal/hnsw/ -v      # the index
go test ./internal/wal/ -v       # the write-ahead log
go test ./internal/snapshot/ -v  # point-in-time state
go test ./internal/store/ -v     # metadata
go test ./internal/filter/ -v    # the metadata query engine
go test ./service/ -v            # collections
go test ./httpapi/ -v            # the REST layer
go test ./cmd/... -v             # the daemon, over a real socket
go test ./... -race              # race detector (run before merging)
```

Running the service locally:

```bash
go run ./cmd/govecdbd -dir /tmp/govecdb        # 127.0.0.1:8080 by default
GOVECDB_AUTH_TOKEN=x go run ./cmd/govecdbd -dir /tmp/govecdb -addr 0.0.0.0:8080
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
  Must be **0 (use the default 16) or >= 2**; `New` returns `ErrInvalidConfig`
  otherwise. `M=1` is undefined, not merely thin: `ml = 1/ln(M)` is `+Inf`, so
  `randomLevel` returns `MaxInt64` and `make([][]int, level+1)` panics on the
  *first* `Insert`. Negative M is refused rather than defaulted, so a caller's
  mistake does not become a silently different graph. `validateHeader` enforces
  `M >= 2` too — a stored config is the *effective* one, so a loaded graph would
  otherwise bypass `New` entirely.
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

## WAL quick reference (`internal/wal`) — complete

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
- **`Truncate` judges a segment from the *next* one's first sequence**, never by
  scanning for its own last. Sequences increase across the log, so a later
  segment starting at or below the line proves this one ends below it. That first
  record is read through the **checksummed reader** — the sequence authorises
  deleting files, so an unverified one would let a flipped bit destroy a segment.
  It never touches the newest segment, nor one whose successor it cannot read.
- **Truncate against the *oldest retained* snapshot, never the newest**, and only
  after `snapshot.Verify` passes on it. Retaining two snapshots is what makes a
  corrupt one survivable; truncating to the newest deletes the records the older
  one needs. A failed verify skips truncation on purpose — a growing log is a
  disk problem, deleting records only an unreadable snapshot could replace is a
  data problem.
- **`TypeCheckpoint` stays reserved and unwritten.** Truncation reads the
  snapshot directory, which is the authority on what is recoverable; a log record
  duplicating that could disagree with it. Do not "finish" it by emitting one.

## Snapshot quick reference (`internal/snapshot`) — complete

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

## Filter quick reference (`internal/filter`) — complete

- **The filter is applied inside the traversal, never to the results.**
  `hnsw.SearchFilter` takes a `func(id string) bool` and `admits()` gates entry to
  `results` only — the frontier still admits rejected nodes, because a
  non-matching vector is very often the bridge to a matching one. Post-filtering
  returns fewer than `k`; measured, **10 results against 2** at one-in-fifty
  (`TestSearchFilterFindsKWherePostFilteringWouldNot`). Do not "simplify" this
  into filtering the output slice — it is the same mistake as filtering
  tombstones at the end, and `admits()` deliberately holds both checks so an edit
  cannot fix one and forget the other.
- **The predicate is over ids, not metadata.** `internal/hnsw` must stay ignorant
  of metadata: giving it a second data model would make every future `Index`
  implementation responsible for one. The root package closes over the store.
- **`store.Map.Match` does not copy**, which is why it exists next to `Get`. It
  runs once per candidate node, and `Get`'s per-call map copy would become the
  dominant cost of a search. `TestMatchDoesNotAllocate` pins it at 0 allocs, and
  the search's **2 allocs/op baseline holds under filtering** — the cost of a
  filter is travel (85 µs → 965 µs from unfiltered to one-in-fifty), not garbage.
- **Insert passes `nil`.** A build must never see a query's filter, or the graph's
  shape would depend on whichever query ran first.
- **A predicate on an absent key is false — `Ne` included.** One uniform rule;
  `Not(Eq(...))` is how to also match vectors lacking the key. Do not "fix" `Ne`
  into meaning "differs or absent": that reintroduces SQL's three-valued logic in
  a function that returns a `bool`.
- **`int64` and `float64` compare exactly, via `compareIntFloat`** — never
  `float64(i) < f`, which rounds past 2^53. A nanosecond timestamp is ~1.7e18, so
  that is ordinary data, and it fails by silently matching the wrong records. The
  range check is written against **2^63, not `MaxInt64`**: `float64(MaxInt64)`
  rounds *up* to 2^63, so comparing against it misjudges the boundary.
- **Operands are normalized (`int`→`int64`, `float32`→`float64`); stored values
  are not.** `Add` refuses an `int` because its width is a platform property and
  it gets written down; an operand never does, so `Eq("page", 12)` is accepted
  rather than silently matching nothing forever. `uint64` above `MaxInt64` is
  refused, not wrapped.
- **Constructors return `Filter`, not `(Filter, error)`** — the error rides in the
  node that found it and `Validate` reports it once, surfaced at `Search` as
  `ErrInvalidFilter`. `Match` assumes a validated filter and may panic otherwise;
  that is deliberate, since it runs per node.
- `And()` matches **everything**, `Or()` and `In(key)` match **nothing** — the
  identity elements, and what makes a filter built in a loop behave at zero
  iterations.
- **No `String()`, no wire format, no clause reordering.** All three are one
  method away and none has a consumer; the same rule that kept `Snapshotter`
  undefined.

## Public API quick reference (root package)

- **No internal type may appear in an exported signature.** `internal/` cannot be
  named from outside the module, so an alias would give callers a type they can
  use but not write down. `Metric`, `SyncPolicy`, `Match`, `Stats`, `Metadata`,
  `Filter` are the root package's own, with adapters in `index.go`.
- **`Filter` is declared here, not aliased from `internal/filter`**, so callers can
  implement one. The two interfaces have identical method sets so elements convert
  implicitly; only the *slices* need the retyping loop in `internalFilters`. That
  loop is the price of the extension point — do not "simplify" it into an alias.
- **`Index` is an interface in this package's types** so a flat or quantized
  index is a different implementation, not a different database. Serialization is
  deliberately *off* it (`indexSerializer`, checked at snapshot time): an index
  that cannot write itself out is still a usable index.
- **Fail closed** — a WAL failure makes the DB permanently `ErrReadOnly`; reads
  keep working. This is the policy `internal/wal` explicitly deferred upward.
- **No `context.Context`**, on purpose. Everything is local and bounded, and
  `Snapshot` cannot be abandoned halfway without leaving the index locked. Do not
  add a ctx no method can honour.
- **`SearchRequest.Ef == 0` means "choose it"** via `SuggestedEf`. Do not
  substitute a constant: recall at a fixed width falls as the corpus grows.
- **`SearchRequest.Filter` is validated once in `validateSearch`**, never per
  node. See the filter quick reference above for the semantics it commits to.
- **Validation is the security boundary** (`validate.go`). Two checks matter most
  because their absence is silent: **values must be finite** (one NaN compares
  false against everything and poisons the ordering the index rests on), and
  **metadata is a closed set of string/bool/int64/float64** — no `gob`, no
  reflection, because decoding is where disk bytes become live objects. Limits
  (`WithLimits`) are configurable but not removable.
- **Recovery order is snapshot then log**, skipping records at or below the
  snapshot's sequence. Do not "simplify" into replaying everything: re-applying a
  PUT that replaced a vector tombstones a slot on *every* start.
- **The snapshot payload is two self-delimiting sections sharing one
  `*bufio.Reader`.** That works because `bufio.NewReaderSize` returns the reader
  it is given when already large enough — so `hnsw.Read` reuses it instead of
  swallowing the metadata section. `TestSnapshotPayloadSectionsDoNotOverread` is
  the guard; getting it wrong corrupts a restore rather than failing to compile.
- **One writer per directory**, enforced in-process via `openDirs`. Cross-process
  locking is a deliberate gap — a lock file left by a crash blocks a restart that
  should have succeeded.
- **Directories this package creates are 0700**; an existing directory's mode is
  left alone (`MkdirAll` only applies its mode on creation, and silently
  tightening an operator's choice would revoke access granted on purpose).
- Reopening with a different **dimension, metric or M** is refused — structural.

## Service quick reference (`service`, `httpapi`, `cmd/govecdbd`)

- **The module decision is made and written down.** The library and the HTTP API
  are **one module** — `net/http` and `encoding/json` are stdlib, so there is no
  dependency to isolate. **gRPC and Raft go in a separate module** that imports
  this one. Do not add a `require` block to `go.mod` for a server feature; CI
  fails the build if one appears. Reasoning in `docs/SERVICE.md`.
- **These three packages sit above `DB` and never inside it.** If a change needs
  the root package to know about collections or HTTP, it is the wrong change.
- **`service.ValidateName` is a security boundary**, not style. A collection name
  becomes a *directory* name, which vector ids never do (`validate.go` says so).
  ASCII alphanumerics, `-` and `_`, alphanumeric first, ≤ 64 bytes. **No dot at
  all**, so `.` and `..` are refused by construction rather than by a special case
  someone later deletes. **ASCII only**, because macOS normalizes to NFD and Linux
  does not — a name holding `é` stops comparing equal to itself when the volume
  moves. The narrowness is also why collection names need no escaping in a URL
  path or a Prometheus label; `httpapi/metrics.go` depends on that.
- **A collection's spec is written down, with defaults resolved.** Dimension,
  metric and `M` are structural, so `collection.json` records the *effective*
  values, never the zero that meant "default" — otherwise changing a library
  default silently rebuilds an existing index under a different `M`. Same rule the
  HNSW header follows. It is written temp → fsync → rename → fsync the directory.
- **The manager releases its lock while opening a collection**, registering a
  placeholder that concurrent callers wait on. Not just an optimisation:
  `govecdb` refuses a second writer on one directory, so two goroutines opening
  the same collection surface `ErrAlreadyOpen` on an ordinary request.
  `TestConcurrentUseOpensOnce` guards it.
- **`Manager.Use(name, fn)` is a callback, not `Acquire`/`Release`.** A borrow
  leaked by an early return is a collection that is never evicted again. Only a
  collection with no borrowers can be evicted or dropped, which is what stops a
  search being closed underneath. `Drop` **waits** rather than failing.
- **`ErrTooManyOpen` does not queue.** Waiting for a slot turns a capacity problem
  into a timeout somewhere else; the handler answers 503 with `Retry-After`.
- **Decoding is the security boundary, one layer out.** Bodies capped with
  `MaxBytesReader` (never by trusting `Content-Length`), **unknown fields
  rejected** (a silent `dimensions` builds a collection at the wrong width),
  filter trees depth-limited (recursion driven by a request body), `Content-Type`
  must be JSON. **5xx bodies say "internal error"**; the detail goes to the log.
  Do not add a second validator for `K`, `Ef`, dimension or metadata size — those
  bounds live in `validate.go` and are tested there.
- **A JSON number without a decimal point or exponent is an `int64`; anything
  else is a `float64`.** Storing everything as float rounds a nanosecond timestamp
  (1.7e18 is past 2^53); storing everything as int turns 0.5 into 0. It is free at
  query time because `int64` and `float64` compare exactly.
- **The filter wire format lives in `httpapi`, not `internal/filter`.** A
  serialization format is a compatibility promise and the package making it should
  be the one a client can see. `internal/filter` still has no wire format.
- **One error shape, always.** `net/http` answers an unrouted path and a wrong
  method in plain text; a middleware rewrites those two into the JSON shape and
  keeps the `Allow` header. Note `classify` checks `ErrInvalidFilter` **before**
  `ErrInvalidMetadata`: one function decodes both a metadata value and a filter
  operand, so a bad operand wraps the second inside the first.
- **Auth is one shared bearer token**, constant-time compared, exempting only
  `/healthz` and `/readyz` — a probe that can fail for an authentication reason
  reports the wrong thing. **`/metrics` is not exempt**: it names every collection
  and its size. No users, no roles; that would imply an authorization story this
  does not have.
- **No latency histogram.** Bucket boundaries chosen without a dependency are
  chosen badly. Count plus total duration is an honest mean; percentiles wait for
  the observability seam (v2 item 1).
- **The daemon binds `127.0.0.1` by default** and reads its token from
  `GOVECDB_AUTH_TOKEN`, never a flag — a flag lands in `ps` output and shell
  history. Non-loopback without a token **warns, does not refuse**: binding
  `0.0.0.0` in a container is ordinary.
- **Shutdown order matters**: fail `/readyz` → drain → `srv.Shutdown` → close the
  manager. Never close the manager first; a handler holding a borrowed collection
  must finish against a live database.

## Locked baselines — do not regress

Any index change must hold these; they are enforced by tests and `-benchmem`:

| Baseline | Value | Guarded by |
|---|---|---|
| Recall@10, dim 32 | 0.999 | `TestRecallVsBruteForce` |
| Recall@10, dim 768 | 0.972 | `TestRecallHighDimension` |
| Search allocations | 2 allocs/op | `BenchmarkSearch -benchmem` |
| Filtered search allocations | 2 allocs/op | `BenchmarkSearchFilter -benchmem` |
| Metadata predicate allocations | 0 allocs/op | `TestMatchDoesNotAllocate` |
| Recall spread across seeds | ≤ 0.05 | `TestRecallIsStableAcrossSeeds` |

The sweep tests in `recall_test.go` defend **shape**, not absolute values: recall
must not fall as `ef` rises, nor as `M` rises, must recover under a wide search
for every metric, and must not drop when a graph is compacted. Their absolute
floors are deliberately loose — see the note at the top of that file before
"tightening" one.
