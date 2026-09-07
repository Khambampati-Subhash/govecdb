# Running GoVecDB as a service

GoVecDB is an embeddable library first. This document is the other way to run it:
one process, many collections, an HTTP API in front.

Nothing about the library changed to make this possible. `service` sits above
`govecdb.DB`, `httpapi` sits above `service`, and `cmd/govecdbd` is flags and a
shutdown sequence around the two. The database does not know any of them exist.

| Layer | Package | Responsibility |
|---|---|---|
| Daemon | `cmd/govecdbd` | Flags, listener, signals, TLS. |
| API | [`httpapi`](../httpapi/README.md) | Routing, JSON, errors, auth, metrics. |
| Collections | [`service`](../service/README.md) | Lifecycle, on-disk specs, eviction. |
| Database | root package | One index, one log, one directory. |

## Contents

- [Should you?](#should-you)
- [Quick start](#quick-start)
- [Collections](#collections)
- [API reference](#api-reference)
- [Filters](#filters)
- [Configuration](#configuration)
- [Operating it](#operating-it)
- [Security](#security)
- [The module decision](#the-module-decision)
- [What is deliberately not here](#what-is-deliberately-not-here)

## Should you?

**Embed the library** when the thing doing the searching is a Go program you
control. It is faster — a search is microseconds of in-memory work, and a network
hop is not — and there is no second process to deploy, monitor or restart.

**Run the service** when any of these is true:

- The clients are not Go, or not one program.
- You want many independent indexes in one process, created at runtime.
- The index should outlive the process that queries it, or be shared by several.
- You would rather operate a database than link one.

The cost is honest: a network hop, JSON encoding on both sides, and one process
that is now a single point of failure — there is no clustering yet (see [what is
deliberately not here](#what-is-deliberately-not-here)).

## Quick start

### From source

```bash
go install github.com/khambampati-subhash/govecdb/cmd/govecdbd@latest
govecdbd -dir ./data
```

It binds `127.0.0.1:8080` by default. That is deliberate: a database should not
become reachable from the network because somebody omitted a flag.

### With Docker

```bash
docker build -t govecdb .
docker run -p 8080:8080 \
  -v govecdb-data:/data \
  -e GOVECDB_AUTH_TOKEN="$(openssl rand -hex 32)" \
  govecdb
```

The image is `FROM scratch` and holds one static binary — with no third-party
dependencies and no CGO there is nothing else to ship.

### The first five requests

```bash
# 1. Create a collection.
curl -sX POST localhost:8080/v1/collections \
  -H 'Content-Type: application/json' \
  -d '{"name": "docs", "dimension": 4, "metric": "cosine"}'

# 2. Add vectors.
curl -sX POST localhost:8080/v1/collections/docs/vectors \
  -H 'Content-Type: application/json' \
  -d '{"vectors": [
        {"id": "a", "values": [1,0,0,0], "metadata": {"source": "handbook", "page": 1}},
        {"id": "b", "values": [0,1,0,0], "metadata": {"source": "handbook", "page": 12}},
        {"id": "c", "values": [0,0,1,0], "metadata": {"source": "notes"}}
      ]}'

# 3. Search.
curl -sX POST localhost:8080/v1/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{"query": [1,0,0,0], "k": 2}'

# 4. Search with a filter.
curl -sX POST localhost:8080/v1/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{"query": [1,0,0,0], "k": 2,
       "filter": {"op": "and", "filters": [
         {"op": "eq",  "key": "source", "value": "handbook"},
         {"op": "gte", "key": "page",   "value": 10}
       ]}}'

# 5. See what it is holding.
curl -s localhost:8080/v1/collections/docs
```

## Collections

A collection is one database: one index, one write-ahead log, one directory.
Collections are independent — different dimensions, different metrics, separate
durability settings — and nothing is shared between them but the process.

```
data/                       ← -dir
  docs/
    collection.json         ← the spec: dimension, metric, M, and the rest
    data/
      wal/                  ← the write-ahead log
      snapshots/
  images/
    collection.json
    data/
```

Three things follow from that layout and are worth knowing:

**Dimension, metric and `M` are structural.** An index built for one cannot
answer under another, so they are fixed when the collection is created and stored
in `collection.json`. There is no endpoint that changes them; there is no
in-place migration. Create a new collection and re-index.

**Collections load on demand.** Starting the daemon reads no indexes. The first
request for a collection loads its snapshot and replays its log; after that it is
in memory. With `-max-open` and `-idle-timeout`, a quiet collection is closed
again and costs nothing until it is next asked for.

**Names become directory names.** They are restricted to ASCII letters, digits,
`-` and `_`, must start with a letter or digit, and are at most 64 bytes. This is
a security boundary rather than a style rule — see the [service
README](../service/README.md#names-are-a-security-boundary) for what each rule
stops.

## API reference

Every response is JSON. Every failure has the same shape:

```json
{"error": {"code": "not_found", "message": "service: collection not found: \"docs\""}}
```

Branch on `code`; read `message`. The full table of codes is in the [httpapi
README](../httpapi/README.md#errors-are-one-shape).

### `POST /v1/collections` — create

```json
{
  "name": "docs",
  "dimension": 768,

  "metric": "cosine",              // cosine (default) | euclidean | dotproduct
  "m": 16,                         // neighbours per node; structural
  "ef_construction": 200,          // build-time search width
  "seed": 1,                       // reproducible construction

  "sync_policy": "always",         // always (default) | interval | never
  "sync_interval": "50ms",         // used by sync_policy: interval
  "snapshot_interval": "5m",       // automatic snapshots; omit to disable
  "snapshots_kept": 2,
  "target_recall": 0.95
}
```

`201 Created` with the collection, every default resolved to the value it
actually got. `409` if the name is taken, `400` if the name or spec is refused.

Unknown fields are **rejected**, not ignored. `dimensions` instead of `dimension`
would otherwise build a collection at the wrong width and be found weeks later.

### `GET /v1/collections` — list

```json
{"collections": [{"name": "docs", "dimension": 768, "...": "...", "loaded": true,
                  "stats": {"live": 1200, "deleted": 3, "slots": 1203,
                            "with_metadata": 1200, "dead_ratio": 0.0025,
                            "last_sequence": 1205, "snapshot_sequence": 1100}}]}
```

Listing loads nothing. `stats` is **absent** for a collection that is not loaded,
because its size is unknown and `0` would be a different claim.

### `GET /v1/collections/{name}` — one collection

Same object. `404` if there is none.

### `DELETE /v1/collections/{name}` — drop

`200` with `{"dropped": "docs"}`. The data is gone when this returns: no
tombstone, no recycle bin. It waits for requests already in flight rather than
failing while one overlaps.

### `POST /v1/collections/{name}/vectors` — add or replace

```json
{"vectors": [
  {"id": "doc-1", "values": [0.1, 0.2, "..."], "metadata": {"source": "a.pdf", "page": 3}}
]}
```

`200` with `{"added": 1}`. Adding an existing id **replaces** it.

Every vector is validated before any is written, so a batch with one bad record
leaves the collection untouched. That is a guarantee about *validation*, not
durability: a batch that fails partway through writing has durably applied its
prefix.

Metadata values must be a string, a bool, or a number — no nested objects, no
arrays, no `null`. A number written without a decimal point becomes an integer
and one written with a point becomes a float; the two still compare exactly, so
`{"gte": 10}` matches a stored `10.0`. The reasoning is in the [httpapi
README](../httpapi/README.md#how-a-json-number-becomes-a-metadata-value).

### `GET /v1/collections/{name}/vectors/{id}` — fetch one

Ids are arbitrary UTF-8 and must be percent-encoded in the path:
`/v1/collections/docs/vectors/doc%2F1` fetches `doc/1`.

The values come back in the form the index holds them, which for `cosine` is the
**unit vector** rather than what you sent. That metric is a statement that
magnitude carries no meaning, and storing a second copy of every embedding to
hand back a number nothing uses would double the memory of the largest thing in
the process.

### `DELETE /v1/collections/{name}/vectors/{id}` — delete one

`200`. Deleting an id that is not there also succeeds, matching the database:
replay applies records more than once across a snapshot boundary, and an
operation that failed the second time would make recovery order-sensitive.

A delete is a **tombstone**. The slot keeps routing searches and stops answering
them; only compaction reclaims it.

### `POST /v1/collections/{name}/search` — nearest neighbours

```json
{
  "query": [0.1, 0.2, "..."],
  "k": 10,
  "ef": 0,                  // omit: chosen from the corpus size
  "target_recall": 0.95,    // what an omitted ef aims for, as a floor
  "filter": {"op": "exists", "key": "source"}
}
```

```json
{"matches": [{"id": "doc-1", "distance": 0.0123, "metadata": {"source": "a.pdf"}}]}
```

`distance` is smaller-is-closer for every metric, on that metric's own scale —
squared for `euclidean`, a negated dot product for `dotproduct`. It is comparable
within one query and meaningless across metrics.

**Leave `ef` out.** Recall at a fixed search width *falls* as a collection grows:
0.997 at 500 vectors down to 0.652 at 20,000, both at `ef=64`. Any constant that
works today is wrong later. Omitted, the width is fitted from the corpus size and
`target_recall`.

### `POST /v1/collections/{name}/snapshot`

`200` with `{"snapshot_sequence": 1205}`. See [operating
it](#snapshots-are-what-bound-restart-time).

### `POST /v1/collections/{name}/compact`

`200` with `{"reclaimed": 41}`. **Stops the world for that collection** — no
search runs while it does. Poll `stats.dead_ratio` and call it when you can
afford the pause; around 0.5 is where it pays, because the pause tracks survivors
rather than garbage.

### `GET /healthz`, `GET /readyz`, `GET /metrics`

Health probes need no credentials. `/metrics` does, when a token is configured:
it names every collection and reports its size.

## Filters

A filter is a tagged object, nested freely:

```json
{"op": "and", "filters": [
  {"op": "eq",  "key": "source", "value": "handbook.pdf"},
  {"op": "gte", "key": "page",   "value": 10},
  {"op": "not", "filter": {"op": "exists", "key": "retracted"}}
]}
```

| Op | Fields | Matches |
|---|---|---|
| `eq` `ne` `lt` `lte` `gt` `gte` | `key`, `value` | The key is present and compares that way. |
| `in` | `key`, `values` | The key is present and equals one of them. |
| `exists` | `key` | The key is present, whatever its value. |
| `and` `or` | `filters` | Every / any of them. |
| `not` | `filter` | The inverse. |

Three semantics that are easy to assume wrong:

**Filters are applied during the traversal, not to the results.** A filtered
search returns `k` matches, rather than however many of the nearest `k` happened
to match. Measured at one-in-fifty selectivity: 10 results against 2.

**Every comparison is false on an absent key — `ne` included.** `ne` means "the
key is there and is not that". Use `{"op":"not","filter":{"op":"eq",...}}` to
also match vectors that lack the key entirely.

**`and` with no filters matches everything; `or` and `in` with none match
nothing.** These are the identity elements, and they are what make a filter built
in a loop behave when the loop runs zero times.

Selectivity costs latency, not memory. Unfiltered to one-in-fifty is 85 µs → 965
µs on the measured corpus, at the same 2 allocations per search. Past roughly one
vector in a hundred, a scan over your own metadata is the better tool and this
index is the wrong thing to ask.

## Configuration

```
govecdbd -dir <directory> [flags]
```

| Flag | Default | What it does |
|---|---|---|
| `-dir` | *required* | Directory holding the collections. |
| `-addr` | `127.0.0.1:8080` | Listen address. |
| `-max-open` | `0` | Most collections loaded at once. 0 is no limit. |
| `-idle-timeout` | `0` | Close a collection nothing has used for this long. 0 never does. |
| `-max-body` | `32 MiB` | Largest request body. |
| `-timeout` | `2m` | Per-request read and write timeout. |
| `-drain` | `0` | Keep serving this long after `/readyz` starts failing. |
| `-shutdown-timeout` | `30s` | How long to wait for requests in flight. |
| `-tls-cert`, `-tls-key` | — | PEM pair; enables TLS. Both or neither. |
| `-log-level` | `info` | `debug`, `info`, `warn`, `error`. |
| `-log-json` | `false` | Structured JSON logs. |
| `-version` | | Print the version and exit. |

| Environment | What it does |
|---|---|
| `GOVECDB_AUTH_TOKEN` | Requires `Authorization: Bearer <token>` on everything but the health probes. |

There is **no flag for the token** on purpose. A flag lands in `ps` output, in
shell history, and in whatever collects a container's command line — all places a
credential outlives the process that used it.

Two sizing notes:

- `-max-body` at 32 MiB does not admit the largest batch the database accepts
  (10,000 × 768 floats is roughly 90 MB of JSON). That is deliberate: the default
  should not let one request decide how much memory the process spends. Raise it
  knowingly, or send smaller batches.
- `-timeout` bounds a whole request. Compaction rebuilds an index and can
  legitimately outlast two minutes on a large collection; raise it or run
  compaction against a collection small enough not to need it.

## Operating it

### Durability is the library's, unchanged

Every write goes to the write-ahead log before it reaches the index, so a crash
between the two costs a replayed record rather than an acknowledged write that
vanished. `sync_policy` is the one knob that trades durability for throughput,
and it is per collection:

| `sync_policy` | An acknowledged write survives | Cost |
|---|---|---|
| `always` *(default)* | power loss | ~4 ms per append |
| `interval` | neither a process crash nor power loss, up to one interval | ~692 ns per append |
| `never` | neither, up to a 64 KiB buffer | ~692 ns per append |

The fast policies do **not** survive a process crash: records sit in a user-space
buffer until it fills. Full numbers and what each guarantee costs are in
[DURABILITY.md](DURABILITY.md).

A durability failure is **fail-closed and permanent**: the collection refuses
writes with `503 read_only` and keeps serving reads, because the in-memory index
is still correct but nothing about the next write can be promised. Fix the disk
and restart.

### Snapshots are what bound restart time

Without a snapshot, starting a collection replays its whole log and rebuilds the
index at roughly 700 µs per vector. With one, it loads a graph — about 1,900×
faster for a million vectors.

Set `snapshot_interval` on collections you care about restarting quickly. Minutes
is the right order of magnitude: every snapshot costs a ~10 ms fsync floor plus
the time to write the index out, and it takes a read lock for the duration, so
writers wait and searches do not.

Snapshotting also truncates the log, which is the only thing that stops it
growing forever. It truncates against the *oldest retained* snapshot rather than
the newest, so the second copy stays usable — which is the entire reason
`snapshots_kept` defaults to 2.

The daemon does **not** snapshot on shutdown. A shutdown that takes seconds per
collection and fails on a full disk is the one an operator cannot afford. Call
`POST .../snapshot` before a planned restart if you want the next start to be
fast.

### What to watch

`/metrics` is Prometheus text. The four that matter:

| Metric | Watch for |
|---|---|
| `govecdb_collection_dead_ratio` | Approaching 0.5 — time to compact. |
| `govecdb_collection_wal_sequence` − `..._snapshot_sequence` | A widening gap is a slow restart waiting to happen. |
| `govecdb_http_requests_total{class="5xx"}` | Anything above zero. |
| `govecdb_collections_loaded` vs `-max-open` | At the cap, with `503 too_many_open` appearing. |

There is no latency histogram, and the [reason](../httpapi/README.md#metrics-counters-and-gauges-no-histogram)
is that bucket boundaries chosen without a dependency are boundaries chosen badly.
`govecdb_http_request_duration_seconds_total` over the request count is an honest
mean; percentiles wait for the observability seam.

### Shutting down

`SIGINT` or `SIGTERM` starts a graceful shutdown, in this order:

1. `/readyz` starts failing. `/healthz` keeps succeeding — the process is fine,
   it is leaving.
2. The listener stays open for `-drain`, so a load balancer notices before
   connections are cut.
3. Requests in flight get `-shutdown-timeout` to finish.
4. Collections close, never before, so a handler holding one finishes against a
   live database.

A second signal terminates outright.

In Kubernetes: point `readinessProbe` at `/readyz`, `livenessProbe` at
`/healthz`, and set `-drain` to a couple of endpoint-propagation intervals.

## Security

What this is: a database process, with one shared bearer token, meant for a
private network or a sidecar.

What it is not: a multi-tenant service. There are no users, no roles, and no
per-collection permissions — inventing them would imply an authorization story
this does not have.

- **Set `GOVECDB_AUTH_TOKEN`** if the address is reachable by anything you do not
  control. The daemon warns loudly when it serves a non-loopback address without
  one; it does not refuse, because binding `0.0.0.0` inside a container is
  ordinary.
- **Terminate TLS in front, or pass `-tls-cert`/`-tls-key`.** The token is a
  bearer credential and travels in a header.
- **Everything a client sends is bounded**: body size, filter depth, id length,
  `k`, `ef`, batch size, metadata size. The last five are the library's own
  limits, tested there.
- **`/metrics` requires the token.** It enumerates collections and their sizes.
- **5xx responses say "internal error".** The detail — filesystem paths, internal
  state — goes to the log.
- **The database directory is created 0700.** An existing directory's mode is
  left alone, because silently tightening an operator's choice would revoke
  access somebody granted on purpose.

Report a vulnerability through [SECURITY.md](../SECURITY.md).

## The module decision

The v2 plan named the server as the point where the zero-dependency rule gets its
first real test, and said the module split should be decided **before** the code
was written rather than discovered afterwards. It was:

> The library and the HTTP API stay in **one module**. `net/http` and
> `encoding/json` are standard library and cost nothing, so there is no
> dependency to isolate. gRPC and Raft are not standard library, so when they
> arrive they arrive as a **separate module** that imports this one.

What that buys: one tag, one CI pipeline, no `replace` directives, and `go install
.../cmd/govecdbd@latest` working with no ceremony. `go.mod` still has no `require`
block and there is still no `go.sum`; CI fails the build if either appears.

What it costs: when a gRPC server does arrive it will live at a different import
path from the HTTP one. That is a smaller price than moving every existing import
path later, which is what deciding this *after* writing the server would have
meant.

## What is deliberately not here

| | Why |
|---|---|
| **Clustering and replication** | v2 item 8. The WAL is already an ordered, checksummed, sequence-numbered record of every state change — which is exactly what a follower needs — so replication should be built on `Replay` rather than beside it. Raft is not stdlib, so it is a separate module. |
| **gRPC** | Same reason. See [the module decision](#the-module-decision). |
| **Cross-process locking** | Unchanged from the library. One daemon per directory; a lock file left behind by a crash blocks a restart that should have succeeded. |
| **Online compaction** | v2 item 2. `POST .../compact` stops the world, so you choose the moment. |
| **Per-user auth, rate limiting, audit** | A proxy in front does these properly, and this process would do them badly. |
| **Changing a collection's dimension, metric or M** | Structural. Create a new collection and re-index. |
| **Pagination** | No endpoint returns an unbounded list: collections are few, and a search returns `k`. The list response is an object rather than a bare array so a cursor can be added without breaking clients. |
