# GoVecDB — the v1 rebuild, and what v2 is scoped to

> Branch: `main` · Same module path.
> Gate: `go build ./... && go vet ./... && go test ./... -race` green after every step.
>
> **v1 is complete** — an embeddable library, no server, no cluster. The record of
> how it was built is below, because the reasoning is the part worth keeping.
> **[v2 scope starts here](#v2-scope).**

## What changed about the plan

The original plan on this branch was **salvage & port**: keep the proven legacy
core and clean it in place. That is no longer what we are doing.

Building the HNSW index from scratch (rather than untangling `index/`) produced a
better result than porting would have — 1,075 readable lines, recall 0.999 at dim
32 verified against brute force, and a search path down to 2 allocations. The
old index was ~8,100 lines across three competing graph implementations with
entangled helpers.

So the strategy is now **rebuild from scratch, one subsystem at a time**, using the
old code as a *reference in git history* rather than as a source to copy.

## Current state

```
govecdb/
├── *.go               # the public API: Open/Add/Get/Search/Snapshot/Compact
├── internal/hnsw/     # index engine, with serialization
├── internal/wal/      # write-ahead log: writer, replay, truncation
├── internal/snapshot/ # atomic checksummed state keyed by log sequence
├── internal/store/    # metadata storage
├── internal/filter/   # metadata query engine
├── docs/
├── go.mod             # stdlib only; no require block, no go.sum
├── README.md  CLAUDE.md  CONTRIBUTING.md  LICENSE
```

All legacy packages were deleted in the clean-slate commit. **Nothing is lost** —
`main` still has every line. To consult the old implementation:

```bash
git show main:persist/wal.go
git show main:persist/snapshot.go
git log main --oneline -- persist/
```

### Deleted (recoverable from `main`)

| Category | Packages |
|---|---|
| Superseded by the new HNSW | `index/`, `store/`, `collection/`, `api/`, `filter/`, `persist/` |
| Out of v1 scope (v2) | `cluster/`, `api/rest/`, `proto/`, `client/`, `segment/` |
| Dead / never wired | `utils/`, `diskann/`, `quantization/`, `batch/`, `streaming/`, `accuracy/`, `internal/{benchmark,errors,health,logging,metrics,monitoring}` |
| Tooling for the above | `cmd/`, `demo/`, `examples/`, `deployments/`, `Makefile`, `run_*.sh` |

Deleting these dropped every third-party dependency: raft, gRPC, protobuf, bolt,
chroma-go, onnxruntime, uuid. The module is now pure stdlib.

## Target layout

```
govecdb/
├── vector.go            # public API: Vector, SearchRequest, Match, Stats
├── filter.go            # Filter and its constructors                 ✅ done
├── db.go                # the DB facade                               ✅ done
├── options.go           # functional-options construction             ✅ done
├── errors.go            # exported sentinel errors                    ✅ done
├── internal/
│   ├── hnsw/            # index engine — concurrent reads             ✅ done
│   ├── wal/             # write-ahead log — writer + replay + truncate ✅ done
│   ├── snapshot/        # store + graph codec (in hnsw/)              ✅ done
│   ├── store/           # metadata storage                            ✅ done
│   ├── filter/          # metadata query engine                       ✅ done
│   └── obs/             # Logger + Metrics interfaces, no-op defaults  ⬜ v2
└── docs/
```

`internal/obs` is the only box still empty, and it is [v2 item 1](#1-the-observability-seam).
Note what is *not* here: no `collection/`, no `api/`, no `cluster/`. Those are
v2, and they are layers above this rather than changes to it — which is what made
cutting them from v1 clean rather than costly.

## Where this stands

**All 11 steps done — there is a working, importable, filterable database.**

| | Step | State |
|---|---|---|
| 1–5 | `internal/hnsw` — index, concurrent reads, delete, upsert, compaction | ✅ |
| 6 | `internal/wal` — writer, rotation, sync policies, replay | ✅ |
| 7 | `internal/snapshot` + graph codec | ✅ |
| 8 | `internal/store` — metadata storage | ✅ |
| 9 | `internal/filter` — metadata queries | ✅ |
| 10 | **Public API** — `govecdb.Open/Add/Get/Search/Snapshot/Compact` | ✅ |
| 11 | Examples + README | ✅ |

All three things that were blocked on step 10 landed with it:

- **Restore orchestration** — `Open` loads the newest snapshot that passes its
  checksum, falls back to an older one if it does not, and replays the log
  records written after it. ✅
- **Snapshot scheduling** — `Snapshot()` and `WithSnapshotInterval`, with
  retention through `WithSnapshotsKept`. ✅
- **WAL truncation** — `wal.Truncate`, run after every snapshot against the
  oldest retained snapshot's sequence. The log no longer grows forever. ✅

**The durability story is complete, and so is the feature scope for v1.** The
crash harness landed too — `internal/wal/crash_test.go` `SIGKILL`s a child
mid-write and asserts every acknowledged record survives — so what remains is not
a gap in what the database does or in what proves it. It is [v2](#v2-scope).

`TypeCheckpoint` stays reserved and unwritten, and that is now a decision rather
than a gap: truncation reads the snapshot directory, which is the authority on
what is actually recoverable, and a log record duplicating that could disagree
with it. The constant remains so the numbering is not rearranged later.

---

## v2 scope

v1 deliberately shipped one thing: a database you import. Everything below either
was cut to get there, or became worth doing only once there was something to
operate.

The order is a dependency order, not a priority list. Each item says what blocks
it, because several of these look independent and are not.

| | Item | Blocked on |
|---|---|---|
| 1 | `internal/obs` — Logger + Metrics seam | nothing |
| 2 | Online (non-blocking) compaction | 1, for the same reason everything wants 1 |
| 3 | Fine-grained write locking | 2 |
| 4 | Collections / namespaces | nothing, but wants 1 |
| 5 | Filter selectivity estimation | 4 is unrelated; needs store statistics |
| 6 | Quantized index | nothing — the `Index` interface was built for it |
| 7 | REST / gRPC server | 4 |
| 8 | Clustering and replication | 7 |

### 1. The observability seam

The first thing, because it is the thing every other item needs and the one gap
that is currently *silent*. Today a torn log tail found during recovery is
repaired correctly and reported to nobody; a truncation skipped because a
snapshot failed verification says nothing about why. Both are exactly the events
an operator needs to see, and both currently reach a `_`.

`Logger` and `Metrics` as interfaces with no-op defaults, injected through the
existing functional options. Deliberately **not** `log/slog` in the signatures:
the module has no third-party dependencies and should not acquire an opinion
about logging, and a no-op default keeps the zero value useful.

The design constraint that makes this non-trivial: the events worth reporting
happen inside `internal/`, which cannot name a root-package type. So the seam is
defined in `internal/obs` and adapted at the root, the same shape `Index` already
uses.

### 2. Online compaction

`Compact()` stops the world — it holds the write lock for a full index rebuild,
2.6 s per 5k×128 vectors at a 25% dead ratio. That is acceptable for a library
where the caller picks the moment, and not acceptable for a server where nobody
can.

Building the replacement outside the lock means writes landing in the old graph
while the new one is built, and reconciling them wants a change log and a
double-buffered swap. The WAL should shape that, since it is already recording
exactly those writes — which is why this was deferred rather than attempted in v1.

### 3. Fine-grained write locking

Concurrent writers currently serialize: one `RWMutex` over the whole graph. HNSW
inserts mutate neighbour lists several hops from the new node, so there is no
obvious small region to lock instead.

It follows online compaction rather than leading it, because the WAL's ordering
constraint — write to the log, then apply — decides what a finer lock is allowed
to do, and the double-buffered swap decides what it must not break.

### 4. Collections / namespaces

One database is one index. Many workloads want many independent indexes in one
process, with load-on-demand and idle eviction so an idle collection costs no
memory.

This sits *above* `DB` rather than changing it, which is why it was cut cleanly
from v1 and why it is not a rewrite. The parts that need thought are lifecycle,
not indexing: one directory per collection, the one-writer-per-directory rule
already enforced by `openDirs`, and eviction that cannot race a search.

### 5. Filter selectivity estimation

Measured, a filter admitting fewer than about one vector in a hundred makes the
graph the wrong tool — a scan over the metadata, distance-checking only what
matches, beats a traversal that is visiting most of the graph anyway. Today that
is documented and left to the caller.

Deciding it automatically needs statistics over the metadata store: per-key
cardinality, and enough of a distribution to estimate a predicate's selectivity
before running it. That is a bigger thing than the filter itself, and it is the
point at which this stops being a filter and starts being a query planner.

### 6. Quantized index

`Index` is an interface stated in the root package's own types precisely so that
a flat brute-force index for small corpora, or a scalar/product-quantized one for
large ones, is a *different implementation* rather than a different database.
Nothing about this needs the interface to change.

Product quantization trades recall for memory — the graph currently holds full
float32 vectors, which is the dominant cost at scale. The measurement harness
already reports recall against brute-force ground truth, so the trade is
measurable on day one rather than argued about.

### 7. REST / gRPC server

Returns from git history rather than from scratch: `api/`, `proto/` and `client/`
all exist on `main` before the clean-slate commit. They should be rewritten to
the current bar rather than restored — the note in *Deleted (recoverable)* above
applies to them as much as to the index.

It follows collections because a server with exactly one index is not a useful
server, and because the collection lifecycle decides what the API's resource
model looks like.

**This is where the dependency rule gets its first real test.** A server needs
HTTP and probably gRPC, and the module has no third-party dependencies today.
`net/http` is stdlib; gRPC is not. The likely answer is that the server lives in
a *separate module* so the library stays stdlib-only, and that decision should be
made before code is written rather than discovered afterwards.

### 8. Clustering and replication

The largest item and the last, for the obvious reason. Raft was in the legacy
tree and is not stdlib, so item 7's module decision governs this one too.

The useful observation is that the WAL is already the replication stream: it is
an ordered, checksummed, sequence-numbered record of every state change, which is
exactly what a follower needs to apply. Replication should be built on `Replay`
and the existing record format rather than beside them — and if that turns out to
require changing the record format, doing it *before* v2 ships is much cheaper
than after.

### Out of scope for v2 as well

- **Cross-process locking.** Still deliberate. A lock file left behind by a crash
  blocks a restart that should have succeeded, and the in-process `openDirs`
  check already catches the case worth catching.
- **Point-in-time recovery and audit.** Time-based log retention serves those,
  and truncation-to-snapshot serves recovery. They are different features and
  only the second is needed.

## Ordered execution (each = one green-gated commit)

1. ~~**`internal/hnsw`**~~ — from-scratch index, brute-force recall tests, benchmarks. **Done.**
2. ~~**Concurrent reads**~~ — scratch pooled into `searchState`, graph guarded by an
   `RWMutex`, parallel `Search`. **Done.** Fine-grained write locking is deferred.
3. ~~**Tombstone `Delete`**~~ — dead slots keep routing, results filter them,
   entry re-election. **Done.** Upsert and compaction are the remaining index work.
4. ~~**Upsert**~~ — `Insert` replaces an existing id by tombstoning its slot and
   building a new one; an unchanged vector is an early return. **Done.**
5. ~~**Compaction**~~ — `Compact()` rebuilds over the live vectors and swaps the
   graph in whole; `Stats().DeadRatio()` is the signal, the policy stays outside
   the index. **Done.** Online (non-blocking) compaction is deferred — see below.
6. ~~**`internal/wal`**~~ — record format, append-only writer with segment
   rotation, and `Replay`: a CRC-validating scan that truncates a torn tail and
   carries the sequence forward. **Done.** Checkpointing and segment truncation
   wait for step 7, which is what makes a checkpoint mean anything.
7. ~~**`internal/snapshot` + graph codec**~~ — atomic, versioned, checksummed
   snapshots keyed by WAL sequence, with discovery, corruption fallback and
   retention; plus `(*Graph).WriteTo` / `hnsw.Read` so a snapshot holds a graph
   rather than a pile of vectors. **Done.** Wiring them into startup is policy
   and lands with the public API.
8. ~~**`internal/store`**~~ — metadata storage behind a `Store` interface. **Done.**
   Metadata *only*: values live once, in the index. A store that owned whole
   records would mean two copies of every vector in memory, and vectors are the
   largest thing the process holds.
9. ~~**`internal/filter`**~~ — metadata query engine: comparisons, set membership,
   existence and boolean composition, applied *inside* the traversal rather than
   to the results. **Done.**
10. ~~**Public API**~~ — `vector.go` / `db.go` / `options.go` / `codec.go` /
    `recovery.go` / `validate.go` / `filter.go`. **Done.** Owns restore-on-open,
    snapshot scheduling and WAL truncation.
11. ~~**Examples + README**~~ for the real API. **Done.** Godoc examples exist and
    run as tests, and the root README documents the importable database.

### Deferred on purpose, not overlooked

Both of these were decided during v1 and are now [v2 items 2 and 3](#2-online-compaction):
**online compaction**, because `Compact()` holds the write lock for a full
rebuild and doing it outside the lock wants a change log the WAL should shape
first; and **fine-grained write locking**, for the same reason — the WAL's
ordering constraint decides what a finer lock is allowed to do.

They are recorded here as well because the reasoning is a v1 finding: neither was
an oversight, and neither should be attempted without the WAL work in front of it.

## Phase — WAL (done)

Design constraints, all decided and now implemented:

- **Write to the WAL first, then apply to the in-memory graph.** On recovery,
  replay the log to rebuild the graph. The graph is *derived state* — it is never
  the source of truth.
- **The WAL is an interface** (`Append`, `Sync`, `Close`) so the index can be
  constructed with a no-op WAL in tests and benchmarks. `Replay` is deliberately
  **not** on it: recovery runs before a writer exists, so it is a package-level
  function over a directory. Putting it on the interface would have meant opening
  a writer in order to read — creating a segment as a side effect of recovery.
- **Records are versioned** from the first commit. A log format without a version
  byte cannot be migrated later.
- **Checksums per record**, covering type, seq and length as well as the payload.
  Recovery truncates to the last intact record rather than failing outright.
- **`fsync` policy is a knob**, not a hardcode: always / interval / never trade
  durability against throughput. Measured at ~4,000× between the extremes.

Settled while building the reader, and worth carrying into the next phase:

- **A tear is tolerated at the end of *any* segment, not only the last.** `Open`
  always starts a new segment, so a second crash leaves a torn tail in the middle
  of the directory — refusing to read past it would make recovery impossible from
  the shape the writer is designed to produce.
- **Truncation is logical.** Recovery is a read; the damaged bytes stay on disk.
- **Nothing is read on an unverified length.** The file size is taken up front
  and a length is refused against both it and `MaxRecordBytes` before any read.

Truncation landed with the public API: `Truncate(dir, keepFromSeq, opts)` deletes
segments holding no record at or after the line. Two things decided while
building it:

- **A segment is judged from the *next* one's first sequence**, not by scanning
  to find its own last. Sequences increase across the log, so if a later segment
  starts at or below the line, this one ends below it. One record read per
  segment instead of a full scan, and conservative in the safe direction.
- **That record goes through the checksummed reader.** The sequence authorises
  deleting files, and an unverified one would let a flipped bit destroy a
  segment.

`TypeCheckpoint` stays reserved and unwritten — see the note above.

Same bar as the index: small single-responsibility files, comments that explain
*why*, and tests that verify crash recovery rather than assuming it. The tear
tests damage a log by truncating and rewriting it, which reproduces the shapes
power loss leaves behind. The harness that reproduces the *timing* now exists
too: `crash_test.go` re-executes the test binary, lets the child acknowledge
appends under `SyncAlways`, `SIGKILL`s it, and asserts every acknowledged record
survives. It reports zero torn segments reliably — under `SyncAlways` the bytes
reach the file before fsync, so the wide window is one where the log already ends
at a record boundary. The two approaches cover different things and neither
replaces the other.

## Phase — Snapshot (done)

The step split in two, and both halves are built.

**Done — the durable store.** Atomic writes (temp → fsync → rename → fsync the
directory), versioned and checksummed framing keyed by the WAL sequence it
covers, discovery, fallback to an older snapshot when the newest fails
verification, and retention.

Decided while building it:

- **The payload is opaque**, as it is in the WAL. The package moves bytes durably
  and knows nothing about vectors, which keeps it testable without the index and
  puts the format boundary somewhere defensible.
- **The checksum lives in a trailer**, because the payload is streamed — a
  snapshot is gigabytes where a WAL record is kilobytes, so nothing may hold it
  all in memory.
- **Atomicity comes from the rename, not the checksum.** That lets the checksum
  mean the narrower and more useful thing: bit rot, not interrupted writes.
- **Nothing unverified reaches the caller.** Verification is a separate pass
  before the callback sees a byte; measured at +37% over streaming once, because
  the second pass reads the page cache.
- **WAL truncation must follow the oldest *retained* snapshot**, never the
  newest, or the fallback copy is unusable while still being paid for.
- **No `Snapshotter` interface yet.** There is an implementation but no consumer,
  and the WAL taught the lesson directly: `Replay` was on the interface as a
  promise until writing it showed it did not belong there. It gets defined when
  `db.go` exists and its needs are known.

**Done — the graph codec** (`internal/hnsw/codec.go`). `(*Graph).WriteTo` and
`hnsw.Read` turn the index into bytes and back, so a snapshot holds a graph
rather than a pile of vectors.

The open question — store the **graph** or just the **live vectors** — was
answered by measurement before a line of it was written. Replay reads a log at
368 ns/record, but *applying* a record costs 703 µs, so reading is 0.05% of
recovery and the rebuild is all of it. A graph of dim 128 at M=16 encodes to
662 bytes per vector, so 1M × 128 is ~0.37 s to verify and decode against ~703 s
to rebuild: about **1,900×**. A vectors-only snapshot would bound log *size*
while leaving recovery *time* essentially unimproved, which is half a snapshot.

Decided while building it:

- **Store only what cannot be recomputed.** The node array, `entry`, `maxLevel`.
  The id index and the tombstone count are pure functions of the nodes, so they
  are derived on load — writing them down creates a second source of truth a
  corrupt file could put in disagreement with the first.
- **Validate structure on load, not just integrity.** A checksum proves the bytes
  are the bytes that were written, not that they describe a graph a search can
  walk. Neighbour indices in range, counts within `maxConn`, and the entry point
  live and at `maxLevel` — the last one is a panic in the next `Insert` if it is
  wrong, a long way from the file that caused it.
- **No checksum in this format.** `internal/snapshot` already hashes the whole
  payload before returning a byte of it; hashing twice would cost a second pass
  over gigabytes to learn the same thing. The contract is that `Read` trusts its
  input to have been verified, and that is written down where it can be read.
- **The RNG is not restored.** `math/rand`'s source cannot be marshaled. Levels
  stay correctly distributed so nothing about recall changes; what differs is
  that build → save → load → insert is no longer byte-identical to building
  straight through. A draw counter would not survive `Compact`, which replaces
  the node array but not the RNG, so it would be subtly wrong rather than absent.

**Next — restore orchestration.** Load the newest snapshot, replay the WAL from
`Seq+1`, and take snapshots on a schedule. It is a small amount of code and it
has nowhere to live yet: it is policy, and policy belongs to the public API
(step 10), which is the first thing to own both a graph and a log. WAL
checkpointing and segment truncation land in the same place, for the same reason.

## Phase — Public API (done)

The facade at the module root, and the first thing a user actually imports.

Decided while building it:

- **No internal type appears in an exported signature.** `internal/` cannot be
  named from outside the module, so aliasing `hnsw.Metric` would have produced a
  public surface callers can use but not write down. `Metric`, `SyncPolicy`,
  `Match` and `Stats` are this package's own, with adapters underneath.
- **`Index` is an interface stated in this package's types**, so a flat or
  quantized index is a different implementation rather than a different database.
  Serialization is deliberately *not* on it: an index that cannot write itself
  out is still a usable index, so `Snapshot` asks for the capability and says so
  plainly when it is absent.
- **Fail closed.** The WAL's sticky failure is surfaced as `ErrReadOnly`: once a
  write could not be made durable, writes stop permanently and reads keep
  working. This is the policy the WAL explicitly deferred to "a layer above" —
  this is that layer.
- **No `context.Context`.** Every operation is local and bounded; the only long
  one is `Snapshot`, and it cannot be abandoned halfway without leaving the index
  locked. A ctx no method could honour is a promise of cancellation never kept.
- **`SearchRequest.Ef` defaults to zero, meaning "choose it"**, because recall at
  a fixed width *falls* as a corpus grows — any constant a caller picks today is
  wrong later.
- **One writer per directory**, enforced within the process. The log and the
  snapshot store both assume it; two would interleave segment numbering and
  delete each other's temporary files. Cross-process locking is a deliberate gap:
  a lock file left by a crash blocks a restart that should have succeeded.
- **Reopening with a different dimension, metric or M is refused.** Those are
  structural — a graph's edges were chosen under one set of rules, and searching
  it under another returns quietly wrong answers rather than failing.

### Input validation, and why a library bothers

A library does not know where its arguments came from. Every id, `K`, and
metadata map may be relaying input from somewhere the process does not trust, and
each one multiplies an allocation — so all of them are bounded, configurably
through `WithLimits` but not removably.

Two checks are worth naming because their absence would be silent rather than
loud:

- **Vector values must be finite.** A NaN compares false against everything, so
  one of them poisons the ordering the whole index rests on: heap invariants stop
  holding and searches return wrong answers with no error anywhere.
- **Metadata values are a closed set of four types.** Decoding metadata is where
  bytes from a disk become live objects, and a decoder that reconstructs
  arbitrary types from names on the wire is a far larger surface than filtering
  needs. Nothing here uses `encoding/gob` or reflection.

Directories this package creates are `0700`. An *existing* directory is left
alone — if an operator set its mode, quietly tightening it would revoke access
somebody granted on purpose — but `wal/` and `snapshots/` are created here rather
than left to those packages' `0755` default, and a directory that cannot be
traversed is what protects the ordinary-mode files inside it.

## Phase — Filter (done)

The last feature gap, and it turned out to be two decisions rather than a pile of
predicates.

**Where it is applied.** Inside the traversal, not over the results. Filtering the
returned slice is one line and wrong for exactly the reason the tombstone design
was already written down: a selective filter would leave far fewer than `k` hits
rather than making the search look wider for `k` matching ones. So the filter
gates entry to the *result set* while the frontier still admits everything —
a rejected vector is very often the bridge to one that matches. Measured, a
one-in-fifty filter returns 10 results this way against 2 by post-filtering
(`TestSearchFilterFindsKWherePostFilteringWouldNot`).

The honest cost is the same curve tombstones produce, and for the same mechanism:
`results` fills slowly, the pruning bound stays loose, the search widens. 85 µs
unfiltered to 965 µs at one in fifty, with **the 2 allocs/op search baseline
intact** — the cost is travel, not garbage. Past roughly one in a hundred a scan
is the better tool, and that is written down rather than hidden.

**What crosses the boundary.** The index takes a `func(id string) bool`, not a
filter and not metadata. `internal/hnsw` stores vectors, and giving it a second
data model would make every future index implementation responsible for one too.
The layer that owns metadata closes over it, and the index pays one parameter.
`internal/store` grew `Match` for that path — it evaluates a predicate against
the stored map *without copying it*, because it is called per candidate node and
`Get`'s copy would have become the dominant cost of a search.

Decided while building it:

- **Absent keys are false, uniformly** — `Ne` included, which is the one that
  surprises. The alternative is SQL's three-valued logic, which is out of place
  where `Match` returns a `bool`; `Not(Eq(...))` asks the other question
  explicitly.
- **Numbers compare across `int64` and `float64` exactly.** `float64(i) < f`
  rounds past 2^53, and a nanosecond timestamp is ~1.7e18 — so the naive spelling
  breaks on ordinary data, by silently comparing numbers that are not the ones
  stored.
- **Operands are normalized, stored values are not.** `Add` refuses an `int`
  because the width of `int` is a platform property and the value gets written
  down. An operand never does, so `Eq("page", 12)` is accepted rather than
  becoming a filter that matches nothing forever.
- **Constructors return `Filter`, not `(Filter, error)`.** A query built by
  nesting calls only reads well if errors are collected and reported once, so a
  bad operand rides in the node that found it and surfaces at `Search` as
  `ErrInvalidFilter`.
- **`Filter` is declared in the root package, not aliased from `internal/`**, so
  a caller can implement one. That costs a slice-retyping loop at the boundary
  and buys an open extension point.

## SOLID / patterns

As built, rather than as planned — two of these ended up narrower than the
original sketch, and the narrowing is the interesting part.

- **SRP** — one package = one responsibility.
- **DIP** — `DB` depends on `Index`, `Store`, `WAL` and `Filter` as interfaces,
  with concretes injected. `Snapshotter` and `Logger` were in the sketch and are
  **not** here: the first had an implementation and no consumer, the second is
  [v2 item 1](#1-the-observability-seam). An interface written before its caller
  is a guess.
- **OCP / Strategy** — distance metric, index type and persistence are
  pluggable. `Index` is stated in the root package's own types precisely so a
  quantized index is a different implementation and not a different database.
- **ISP** — serialization is deliberately *off* `Index`: an index that cannot
  write itself out is still a usable index, so `Snapshot` asks for the capability
  with a type assertion instead of widening the interface for everyone.
- **Factory + Functional Options** — `govecdb.Open(dir, WithDimension(768),
  WithMetric(...))`. Note there is no `WithWAL(dir)`: the log lives in a
  subdirectory of the database directory, because two directories to configure is
  two directories to get out of sync.

## Risks

**Carried from v1, still guarded:**

- **Recall regression** — the baselines are locked (0.999 @ dim 32, 0.972 @ dim
  768) and asserted against brute force. Any index change must keep them.
- **Allocation regression** — search is 2 allocs/op, including under a filter;
  `-benchmem` guards both. The scratch pool must keep it there, and
  `store.Match` must keep not copying.
- Every step is a separate commit and reverts cleanly.

**New with v2:**

- **The dependency rule is the one most likely to break.** The module has no
  third-party dependencies and that is a design goal, not an accident. gRPC and
  Raft are not stdlib. Decide the multi-module split *before* writing item 7,
  because retrofitting one after a server exists means moving every import path.
- **The record format is now load-bearing for two things.** Replication (item 8)
  wants to be built on `Replay` and the existing records. If it turns out to need
  a format change, that change is far cheaper before v2 ships than after — the
  frozen-layout tests will say so loudly, which is what they are for.
- **Online compaction is the one item that can corrupt data**, rather than merely
  fail. It swaps a live graph while writes are landing; everything else on the v2
  list is additive.
