# GoVecDB

An embeddable **vector database in pure Go** — no CGO, no dependencies. Stores
embeddings and answers *"what is most similar to this?"* using an HNSW
approximate-nearest-neighbor index.

[![Go Version](https://img.shields.io/badge/go-1.24+-blue.svg)](https://golang.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> ## Status: the v1 rebuild is complete
>
> GoVecDB was rewritten from the ground up. The previous implementation — ~45,700
> lines covering clustering, gRPC, REST, segments and several competing index
> variants — was removed from the working tree. It remains in git history and is
> recoverable at any time.
>
> **What exists today:** an importable database. `govecdb.Open` gives you add,
> get, delete, search, **filter**, snapshot and compact over one directory,
> durable through a write-ahead log and recoverable from snapshots. Underneath:
> `internal/hnsw` (the index, with serialization), `internal/wal` (append-only log
> with replay that truncates a torn tail), `internal/snapshot` (atomic checksummed
> state keyed by log sequence), `internal/store` (metadata) and `internal/filter`
> (the query engine over it).
>
> **What is deliberately out of scope for v1:** clustering, REST/gRPC servers,
> collections and quantization. Those are
> [v2](docs/MIGRATION.md#v2-scope), along with online compaction and an
> observability seam. See [durability and latency](docs/DURABILITY.md) for what is
> guaranteed today, what it costs, and what is not guaranteed yet.

## Install

Requires **Go 1.24+**.

```bash
go get github.com/khambampati-subhash/govecdb
```

That is the whole installation. There is no server to run, no CGO toolchain, and
no third-party packages come with it — GoVecDB is a library that stores its data
in a directory you choose.

## Quick start

A complete program. Save it as `main.go`, `go mod tidy`, `go run .`:

```go
package main

import (
	"fmt"
	"log"

	"github.com/khambampati-subhash/govecdb"
)

func main() {
	// Creates ./data if it does not exist. Dimension is required — it is
	// structural, and reopening with a different one is refused.
	db, err := govecdb.Open("data", govecdb.WithDimension(4))
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// Metadata values must be string, bool, int64 or float64.
	// Note int64(12), not 12 — an untyped constant is an int, and Add says so
	// rather than guessing.
	err = db.AddBatch([]govecdb.Vector{
		{ID: "cat", Values: []float32{1, 0, 0, 0}, Metadata: govecdb.Metadata{"kind": "animal"}},
		{ID: "dog", Values: []float32{0.9, 0.1, 0, 0}, Metadata: govecdb.Metadata{"kind": "animal"}},
		{ID: "car", Values: []float32{0, 0, 1, 0}, Metadata: govecdb.Metadata{"kind": "vehicle"}},
	})
	if err != nil {
		log.Fatal(err)
	}

	matches, err := db.Search(govecdb.SearchRequest{
		Query: []float32{1, 0, 0, 0},
		K:     2,
	})
	if err != nil {
		log.Fatal(err)
	}
	for _, m := range matches {
		// Distance is "smaller = closer" whatever the metric.
		fmt.Printf("%-4s  %.4f  %v\n", m.ID, m.Distance, m.Metadata)
	}

	// Bounds how long the next start takes. Without one, reopening replays the
	// whole log; with one, it loads a graph.
	if err := db.Snapshot(); err != nil {
		log.Fatal(err)
	}
}
```

Run it twice — the second run finds the data still there, because every write
went to a write-ahead log before it reached the index.

Searches can be restricted by metadata:

```go
matches, err := db.Search(govecdb.SearchRequest{
    Query:  query,
    K:      10,
    Filter: govecdb.And(
        govecdb.Eq("source", "handbook.pdf"),
        govecdb.Gte("page", 10),
        govecdb.Not(govecdb.Exists("retracted")),
    ),
})
```

The filter runs *during* the graph traversal, not over the results — so you get
10 matching vectors rather than however many of the nearest 10 happened to match.
[See what that costs.](#filtering-costs-search-width-not-allocations)

Leaving `Ef` zero lets the search width be chosen from the corpus size, which is
what keeps recall steady as the database grows — recall at a *fixed* width falls
as a corpus gets bigger, so any constant you pick today is wrong later.

Durability is a knob and the zero value is the safe one: `SyncAlways` means an
acknowledged write has survived power loss, at roughly 4 ms each.
`WithSyncPolicy(govecdb.SyncInterval)` is about a thousand times faster and loses
up to one interval to a crash. See [durability and latency](docs/DURABILITY.md).

Nothing snapshots automatically. Call `db.Snapshot()`, or set
`WithSnapshotInterval` — without one, a restart replays the whole log and rebuilds
the index at ~700 µs per vector; with one, it loads a graph at gigabytes per
second.

## What GoVecDB is for

AI models turn text, images, and audio into **embeddings** — lists of numbers where
*things that mean similar things sit close together* in number-space. That reframes
"find me content similar to this" into a geometry problem: **find the stored vectors
nearest to my query vector** (nearest-neighbor search).

The naive approach — compare the query against *every* stored vector — is `O(N)` and
collapses at millions of vectors. GoVecDB's purpose is to make that search fast and
durable, in pure Go so it stays embeddable:

> Store millions of embeddings and answer *"what's most similar to this?"* in
> sub-millisecond time, and survive crashes.

This powers **semantic search**, **RAG** for LLMs, **recommendations**, and
**anomaly detection**.

## How it works

### The core trick: HNSW (skip brute force)

Instead of scanning every vector, GoVecDB builds a **hierarchical graph** (HNSW —
Hierarchical Navigable Small World). Think of finding a house in a country: you don't
knock on every door — you take highways to the right region, then regional roads, then
local streets. HNSW searches the same way, turning `O(N)` into roughly `O(log N)`.

```mermaid
flowchart TB
    Q([Query vector]) --> TOP

    subgraph TOP [Layer 2 · few nodes · long jumps]
        direction LR
        n1(( )) --- n2(( )) --- n3(( ))
    end
    subgraph MID [Layer 1 · more nodes · medium hops]
        direction LR
        m1(( )) --- m2(( )) --- m3(( )) --- m4(( ))
    end
    subgraph BOT [Layer 0 · every vector · short hops]
        direction LR
        b1(( )) --- b2(( )) --- b3(( )) --- b4(( )) --- b5(( ))
    end

    TOP -->|zoom into region| MID
    MID -->|refine| BOT
    BOT --> R([Nearest K results])
```

A search **enters at the sparse top layer, takes big jumps toward the right region,
then drops layer by layer taking smaller steps** until it lands among the true nearest
neighbors — visiting only a tiny fraction of all vectors. Two knobs trade speed for
accuracy: `M` (connections per node) and `EfConstruction`/`ef` (how wide the search
explores).

For the design decisions behind the implementation — why vectors are normalized on
insert, why neighbor selection is alpha-pruned, how the search path reaches two
allocations — see [`internal/hnsw/README.md`](internal/hnsw/README.md).

## The API

```go
import "github.com/khambampati-subhash/govecdb"

db, _ := govecdb.Open("data", govecdb.WithDimension(128))
defer db.Close()

_ = db.Add(govecdb.Vector{ID: "doc1", Values: v1, Metadata: govecdb.Metadata{
    "source": "handbook.pdf",
    "page":   int64(12),
}})
_ = db.Add(govecdb.Vector{ID: "doc1", Values: v2}) // upsert: same id, new vector

// Ef defaults to a width chosen from the corpus size — see SuggestedEf below.
matches, _ := db.Search(govecdb.SearchRequest{Query: query, K: 10})
for _, m := range matches {
    fmt.Println(m.ID, m.Distance, m.Metadata) // ascending; smaller = closer
}

_ = db.Delete("doc1") // tombstone; the slot keeps routing, never answers

_ = db.Snapshot() // bounds restart time, and truncates the log behind it

if db.Stats().DeadRatio() > 0.5 {
    _, _ = db.Compact() // rebuild over the live vectors; stop-the-world
}
```

### Filters

| Constructor | Matches |
|---|---|
| `Eq` / `Ne` | present and equal / present and different |
| `Lt` / `Lte` / `Gt` / `Gte` | ordered comparisons — strings lexicographic, numbers numeric |
| `In(key, v...)` | present and equal to any listed value |
| `Exists(key)` | the key is present, whatever its value |
| `And` / `Or` / `Not` | boolean composition |

Metadata values are `string`, `bool`, `int64` or `float64`. Two rules are worth
knowing up front:

- **A predicate on an absent key is false — including `Ne`.** `Ne("status",
  "draft")` means *has a status, and it is not draft*. Use `Not(Eq(...))` to also
  match vectors with no `status` at all.
- **`int64` and `float64` compare numerically and exactly**, at any magnitude.
  `float64(i) < f` would round past 2^53, and a nanosecond timestamp is ~1.7e18 —
  well inside the range where that silently returns the wrong records.

`Filter` is an interface, so a predicate with no constructor here is a legitimate
thing to implement yourself.

| Knob | Where | Adaptable? |
|------|-------|-----------|
| `M` — neighbors per node (layers > 0; layer 0 uses `2*M`) | `Config`, set once | **No** — structural; changing it means rebuilding |
| `EfConstruction` — search width during inserts | `Config` | Kept fixed (100–200) |
| `Alpha` — pruning relaxation | `Config` | Fixed per graph; 1.0–1.4 useful, default 1.2 |
| `ef` — search width at query time | `SearchRequest.Ef` | **Yes** — per query; auto-clamped to `>= k`. Grows with `N`, so leave it zero and it is chosen for you, [see the charts](#measured-behaviour) |

### Configuration

Everything is a functional option on `Open`. Only `WithDimension` is required.

| Option | Default | Notes |
|---|---|---|
| `WithDimension(int)` | — | **Required.** Structural; reopening with a different one is refused. |
| `WithMetric(Metric)` | `Cosine` | `Cosine`, `Euclidean`, `DotProduct`. Structural. |
| `WithM(int)` | `16` | Neighbours per node. **Structural** — changing it needs a rebuild. |
| `WithEfConstruction(int)` | `200` | Build-time search width. |
| `WithSeed(int64)` | `1` | Makes index construction reproducible. |
| `WithSyncPolicy(SyncPolicy)` | `SyncAlways` | `SyncAlways`, `SyncInterval`, `SyncNever`. |
| `WithSyncInterval(time.Duration)` | `50ms` | Only meaningful under `SyncInterval`. |
| `WithMaxSegmentBytes(int64)` | `64 MiB` | Log truncation granularity, not a size cap. |
| `WithSnapshotInterval(time.Duration)` | off | Snapshot on a timer. Off means you call `Snapshot()`. |
| `WithSnapshotsKept(int)` | `2` | Also decides how much log is kept — see below. |
| `WithSearchTargetRecall(float64)` | `0.95` | What a zero `Ef` aims for. Treated as a floor. |
| `WithLimits(id, k, ef, batch, mdKeys)` | `512, 10k, 100k, 10k, 256` | Per-call bounds. Configurable, not removable. |

Three of these interact in a way worth stating plainly:

- **`WithSnapshotsKept` is also the log-retention knob.** Truncation runs against
  the *oldest retained* snapshot, never the newest, because keeping more than one
  is what makes a corrupt one survivable — and that only works if the log still
  reaches back far enough to replay on top of the older copy.
- **`SyncAlways` is the zero value on purpose**, so a caller who configures
  nothing gets the safe answer rather than the fast one. It costs ~4 ms per write
  against ~692 ns for `SyncNever`.
- **The fast policies do not survive a process crash either.** Records sit in a
  64 KiB user-space buffer, so under `SyncInterval` or `SyncNever` an
  acknowledged write may not have reached the kernel at all. See
  [durability](docs/DURABILITY.md).

## Project layout

```
govecdb/
├── *.go                  ← the public API. This IS package govecdb.
├── internal/
│   ├── hnsw/             the index: graph, search, delete, compaction, codec
│   ├── wal/              write-ahead log: writer, replay, truncation
│   ├── snapshot/         atomic checksummed state keyed by log sequence
│   ├── store/            metadata storage
│   └── filter/           the metadata query engine
└── docs/                 durability, roadmap, benchmarks, diagrams
```

**The `.go` files at the repository root are not loose files — they are the
package you import.** Go resolves `github.com/khambampati-subhash/govecdb` to the
module root, so moving them into a subdirectory would change the import path to
`.../govecdb/pkg/govecdb`. A flat root package is the standard layout for a Go
library whose main import path is the module path.

One responsibility per file:

| File | What it holds |
|---|---|
| `doc.go` | Package documentation — start here. |
| `vector.go` | `Vector`, `SearchRequest`, `Match`, `Stats`. |
| `filter.go` | `Filter` and its constructors. |
| `db.go` | The `DB` facade and its lifecycle. |
| `options.go` | `Open`'s functional options and defaults. |
| `index.go` | The `Index` interface and the HNSW adapter. |
| `validate.go` | The input boundary: what is checked, and the limits. |
| `codec.go` | Domain encoding: log payloads and the snapshot payload. |
| `recovery.go` | Rebuilding state on `Open`: snapshot first, then the log. |
| `errors.go` | Sentinel errors to match with `errors.Is`. |

Each `internal/` package has its own README explaining the decisions behind it —
[`hnsw`](internal/hnsw/README.md), [`wal`](internal/wal/README.md),
[`snapshot`](internal/snapshot/README.md), [`store`](internal/store/README.md),
[`filter`](internal/filter/README.md).

## Measured performance

Apple M4 Max, 10k vectors × 128 dim, k=10, ef=64:

| Metric | Value |
|---|---|
| Search | 105,141 ns/op |
| Search allocations | **2 allocs/op**, 1,264 B/op |
| Cosine distance (normalized) | 30.1 ns, 0 allocs |
| Euclidean distance | 25.9 ns, 0 allocs |
| Recall@10 (dim 32) | **0.999** |
| Recall@10 (dim 768) | **0.972** |

Recall is measured against brute-force ground truth in `graph_test.go`, not estimated.

## Measured behaviour

Every chart below is generated from [`docs/benchmarks/results.csv`](docs/benchmarks/results.csv),
which is written by the same tests CI runs — there is no separate benchmarking
script whose numbers can drift from the ones the suite defends. Reproduce with:

```bash
go test ./internal/hnsw/ -run TestSweep -results docs/benchmarks/results.csv -timeout 40m && go run docs/benchmarks/plot.go
```

Apple M4 Max · 5,000 vectors · k=10 · recall against brute-force ground truth.

### `ef` is the knob, and its default is a starting point — not a setting

![Recall and latency against ef](docs/benchmarks/recall-vs-ef.svg)

`ef` is the search width, the one parameter you can change per query. At 5,000
vectors of 128 dimensions it spans **0.310 recall at 27 µs** to **0.999 at
438 µs**. The suite asserts the shape as well as the numbers: a wider search may
cost more, but it must never find *less*.

### Recall at a fixed `ef` falls as the corpus grows

![Search latency against corpus size](docs/benchmarks/latency-vs-corpus-size.svg)

This is the most practically useful thing in this README. Hold `ef` at 64 and
recall slides from 0.997 at 500 vectors to **0.652 at 20,000** — not because the
index degrades, but because a fixed-width beam covers a shrinking share of a
growing space. **`ef` has to grow with `N`.** That it is a `Search` argument
rather than a build-time constant is the whole point.

Latency, meanwhile, grows **3.2× for a 40× corpus** — the sub-linear behaviour
the index exists for. (Even that overstates it: past ~4 MB of vectors the
distance kernels start paying for memory rather than arithmetic, so the measured
curve is nearer `sqrt(N)` than the `log(N)` the algorithm implies.)

### Dimension costs recall and latency at both ends

![Recall and latency across dimensions](docs/benchmarks/recall-vs-dimension.svg)

At `ef=64`, recall runs from 1.000 at 8 dimensions to **0.558 at 1536** while a
query goes from 17 µs to 1,090 µs. High-dimensional embeddings need a wider `ef`,
and the build cost rises with them: the same corpus takes 0.65 s to index at 8
dimensions and 56 s at 1536.

### `M` is structural — read this chart before you build

![Recall and build time against M](docs/benchmarks/recall-vs-m.svg)

`M` cannot be changed without rebuilding, and it buys recall at a steep build
price: **M=4 gives 0.294 recall for a 0.6 s build; M=48 gives 0.994 for 73 s.**
The default of 16 sits where the curve turns.

### Choosing `M` and `ef` together

![Recall against latency for every M and ef](docs/benchmarks/recall-vs-latency-pareto.svg)

The two charts above each vary one knob with the other at its default, which
shows a slope but cannot answer *"which pair?"* — "M=16 gives 0.823" really means
"M=16 *at a narrow ef*". This is the surface: one line per `M`, one point per
`ef`, ringed where nothing beats that point on both axes at once.

Compared at **equal recall**, raising `M` is worth less than the M-chart alone
suggests:

| Target | via `ef` (M=16) | via `M` (ef=64/128) | Build cost |
|---|---|---|---|
| ~0.96 | ef=128 → 0.964, **180 µs** | M=32/ef=64 → 0.968, **162 µs** | 3.8 s → 23 s |
| ~0.997 | ef=256 → 0.998, **282 µs** | M=32/ef=128 → 0.997, **251 µs** | 3.8 s → 23 s |

**About 10% latency, for 6× the build time and roughly double the graph memory.**
That is a much weaker case for raising `M` than comparing the two 1-D charts
implies — which is exactly why the 2-D grid is the one to read. `M=16` is a good
default; reach for `M=24`–`32` only when query latency is the binding constraint
and you can afford the build.

### Picking `ef` without a benchmark: `SuggestedEf`

![Recall against ef at three corpus sizes](docs/benchmarks/recall-vs-ef-by-corpus-size.svg)

Holding a recall target needs a wider search as the corpus grows — measured,
`ef` scales as roughly **n^0.78**. That relationship is fitted into the API so
the guidance lives where it is used:

```go
ef := g.SuggestedEf(10 /*k*/, 0.95 /*target recall*/)
results, _ := g.Search(query, 10, ef)
```

`targetRecall` is a **floor to clear, not a point to hit** — the calibration
carries margin, so 0.95 measures 0.969 at 1,000 vectors and 0.972 at 5,000.
Calibrated exactly on the sweep it undershot on three of four verification
corpora, because recall on one corpus does not transfer precisely to another.

It is a starting point, not a guarantee: calibrated on uniform random 128-dim
vectors at M=16, which is the pessimistic case. `TestSuggestedEfAchievesTarget`
builds real graphs and fails if a suggestion misses, so the constants cannot rot
quietly.

### Tombstones, and what `Compact()` gives back

![Latency with tombstones and after compaction](docs/benchmarks/tombstones-vs-compaction.svg)

Dead slots ride the search frontier, so latency climbs with them — 104 µs clean,
**243 µs at 75% tombstoned**, back to **69 µs** after `Compact()`.

There is a subtlety worth knowing, because it looks like a regression and isn't:
compaction *slightly lowers* recall (0.968 → 0.949 at 50% dead). Tombstones keep
the result set under-filled, which loosens the pruning bound and makes the search
explore wider than `ef` asked for. That bought recall nobody requested at a
latency nobody wanted — 184 µs against 88 µs. A marginally larger `ef` on the
compacted graph recovers the recall and is still twice as fast.

### Filtering costs search width, not allocations

A metadata filter is applied while the graph is being walked, not to the results.
A rejected vector still rides the search frontier — it is often the bridge to one
that matches — but never enters the result set. That is what makes a filtered
search return `K` results instead of however many of the nearest `K` happened to
match: measured on a one-in-fifty filter, **10 results against 2** for the same
query post-filtered.

What it costs is travel. 10,000 × 128, `k=10`, `ef=64`:

| Admitted | none (unfiltered) | 1 in 2 | 1 in 10 | 1 in 50 |
|---|---|---|---|---|
| Latency | 85 µs | 165 µs | 385 µs | 965 µs |
| Allocs | **2** | **2** | **2** | **2** |

This is the same curve tombstones produce and the same mechanism: with fewer
admissible nodes the result set fills slowly, which loosens the pruning bound and
makes the search explore wider until it has `k`.

**The allocation count does not move**, which is the part worth defending — the
metadata lookup borrows the stored map rather than copying it, so the 2 allocs/op
search baseline survives filtering intact.

Past roughly one in a hundred the graph stops being the right tool: a scan over
the metadata, distance-checking only what matches, beats a traversal that is
visiting most of the graph anyway.

### Metric and data shape

| Metric | recall @ ef=64 | @ ef=256 | | Corpus | recall | latency |
|---|---|---|---|---|---|---|
| Cosine | 0.830 | 0.998 | | centered | 0.852 | 120 µs |
| Euclidean | 0.820 | 0.980 | | positive orthant | 0.865 | 90 µs |
| DotProduct | 0.848 | 0.999 | | clustered | 0.884 | 38 µs |

All three metrics behave alike — no metric is weak here, and a wide search
recovers every one of them, which is what rules out the graph rather than the
data being at fault. Clustered data, which is what real embeddings look like, is
**3× faster** to search than uniform noise.

## Roadmap

1. ~~**HNSW index**~~ — done, from scratch, tested against brute force
2. ~~**Concurrent reads**~~ — done; pooled search scratch + `RWMutex`, parallel `Search`
3. ~~**Delete**~~ — done; tombstones that keep routing, filtered out of results
4. ~~**Upsert**~~ — done; `Insert` replaces an existing id, tombstoning the old slot
5. ~~**Compaction**~~ — done; `Compact()` rebuilds over the live vectors and swaps in
6. ~~**WAL**~~ — done; record format, append-only writer with segment rotation, and
   `Replay` — a CRC-validating scan that truncates torn tails
7. ~~**Snapshots**~~ — done; atomic checksummed store keyed by WAL sequence, plus
   a graph codec so recovery loads an index (~0.37 s/1M vectors) instead of
   rebuilding one (~703 s)
8. ~~**Public API**~~ — done; `Open` / `Add` / `Get` / `Search` / `Snapshot` /
   `Compact`, with restore-on-open and snapshot scheduling
9. ~~**WAL truncation**~~ — done; segments a snapshot has made redundant are
   deleted after each snapshot, against the *oldest retained* one
10. ~~**Metadata filtering**~~ — done; a query engine applied inside the traversal,
    so a filtered search still returns `K`

**v1 is complete.** [v2 is scoped](docs/MIGRATION.md#v2-scope) in dependency
order — an observability seam, online compaction, fine-grained write locking,
collections, filter selectivity estimation, a quantized index, then REST/gRPC and
clustering. The last two almost certainly land in a separate module: gRPC and
Raft are not stdlib, and this one keeps its zero-dependency guarantee.

## Running the tests and benchmarks

```bash
go build ./... && go vet ./... && go test ./...
```

The race detector is the gate before merging anything:

```bash
go test ./... -race
```

Per package, when you want the detail:

```bash
go test . -v                     # the public API
go test ./internal/hnsw/ -v      # the index
go test ./internal/wal/ -v       # the write-ahead log
go test ./internal/snapshot/ -v  # point-in-time state
go test ./internal/store/ -v     # metadata
go test ./internal/filter/ -v    # the metadata query engine
```

Benchmarks, including the filter-selectivity and durability numbers quoted above:

```bash
go test ./internal/hnsw/ -run='^$' -bench=. -benchmem
```

Regenerating the charts. The sweeps *are* the benchmark harness — the same code
runs in CI asserting thresholds and, with `-results`, writes the CSV the charts
are drawn from, so a README number and a CI threshold cannot disagree:

```bash
go test ./internal/hnsw/ -run TestSweep -results docs/benchmarks/results.csv -timeout 40m
```

```bash
go run docs/benchmarks/plot.go
```

Two things that look like problems and are not. The sweeps **skip themselves
under `-race`** — they are single-goroutine, so the detector observes nothing
while costing ~10× and pushing the package past the default timeout; race
coverage lives in the `TestConcurrent*` tests instead. And the crash harness
(`internal/wal/crash_test.go`) re-executes the test binary as a child process and
`SIGKILL`s it, so seeing a child appear and die during `go test ./internal/wal/`
is the test working.

Requires Go 1.24+. The module has **zero third-party dependencies** — `go.mod` has
no `require` block, and there is no `go.sum`. Keep it that way.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Note that during the v1 rebuild the codebase
is changing shape quickly; open an issue before starting substantial work.

## License

MIT — see [LICENSE](LICENSE).
