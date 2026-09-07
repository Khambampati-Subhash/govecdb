# httpapi

A REST/JSON API over a `service.Manager`, written against the standard library
and nothing else.

It is an `http.Handler`, not a program. `cmd/govecdbd` is flags and a signal
handler around this; anything already running a Go server can mount it on a path
of its own instead.

## The module decision, made before the code

The v2 plan flagged the server as the point where the zero-dependency rule gets
its first real test, and said the decision should be made **before** code was
written rather than discovered afterwards. It was:

> The library and the HTTP API stay in **one module**, because `net/http` and
> `encoding/json` are standard library and cost nothing. gRPC and Raft are not,
> so when they arrive they arrive as a **separate module** that imports this one.

Nothing here has to move for that to happen — which is the whole reason for
settling it first. `go.mod` still has no `require` block, and CI still fails the
build if one appears.

## Files (one responsibility each)

| File | Responsibility |
|------|----------------|
| `doc.go` | Package overview, the route table, and this file map. |
| `server.go` | Routing, the middleware chain, and response writing. |
| `collections.go` | Collection lifecycle, snapshot and compact. |
| `vectors.go` | Add, get and delete. |
| `search.go` | Nearest neighbours. |
| `filter.go` | The filter wire format. |
| `json.go` | Decoding, and the rules for turning JSON into metadata. |
| `metrics.go` | Prometheus text exposition. |
| `errors.go` | One error shape, and the mapping onto status codes. |

## Routes

```
GET    /healthz                              liveness, never authenticated
GET    /readyz                               readiness, never authenticated
GET    /metrics                              Prometheus text

GET    /v1/collections                       list
POST   /v1/collections                       create
GET    /v1/collections/{name}                spec and statistics
DELETE /v1/collections/{name}                drop

POST   /v1/collections/{name}/vectors        add or replace
GET    /v1/collections/{name}/vectors/{id}   fetch one
DELETE /v1/collections/{name}/vectors/{id}   delete one
POST   /v1/collections/{name}/search         nearest neighbours

POST   /v1/collections/{name}/snapshot       snapshot now
POST   /v1/collections/{name}/compact        reclaim tombstoned slots
```

Full request and response shapes are in [docs/SERVICE.md](../docs/SERVICE.md).

## Decoding is the security boundary, again

The root package says validation is where *disk* bytes become live objects. Here
they are *network* bytes, which is the same problem with a shorter fuse.

- **Bodies are capped** by `http.MaxBytesReader` — not by trusting
  `Content-Length`, which is a claim the client makes and a chunked body does not
  make at all. The cap is configurable because a batch of 10,000 embeddings is
  legitimately large; the default of 32 MiB deliberately does *not* admit the
  largest batch the database accepts, so raising it is a decision somebody makes.
- **Unknown fields are rejected.** The usual argument for ignoring them is
  forward compatibility and it does not survive contact with this API: a request
  saying `dimensions` instead of `dimension` would otherwise create a collection
  at the wrong width, succeed, and be found weeks later with data in it.
- **Filter trees are depth-limited.** A filter is recursion driven by a request
  body, and a stack overflow in Go is not a recovered panic — it is the process.
- **`Content-Type` must be `application/json`** when there is a body. Refusing
  the types a browser can send from a plain form is what keeps a cross-site
  request from reaching a write endpoint without a preflight.
- **5xx bodies say "internal error".** The detail is full of filesystem paths,
  helps nobody on the other end, and goes to the log where the operator is.

What it does *not* do is invent a second validator. Dimension, `K`, `Ef`, id
length and metadata size are bounded and tested in the root package; this layer's
job is to hand it well-formed Go values and translate what comes back.

## Two decisions worth knowing

### How a JSON number becomes a metadata value

JSON has one number type and this database has two, so the rule is the syntax: a
number written **without** a decimal point or exponent becomes an `int64`, and
anything else becomes a `float64`. `{"page": 10}` stores an integer; `{"score":
10.0}` stores a float.

Getting this wrong is quiet in both directions, which is why it is a rule and not
a heuristic. Storing everything as `float64` would round a nanosecond timestamp —
about 1.7e18, well past the 2^53 where `float64` stops counting. Storing
everything as `int64` would turn a score of 0.5 into 0.

It costs nothing at query time: the database compares `int64` and `float64`
exactly at any magnitude, so a filter written as `{"gte": 10}` still matches a
stored `10.0`.

### The filter format lives here

`internal/filter` deliberately has no wire format, on the grounds that nothing
consumed one. Something does now, and it is this layer rather than that one: a
serialization format is a compatibility promise, and the package making the
promise should be the one a client can see. `internal/filter` stays free to
change its representation.

```json
{"op": "and", "filters": [
  {"op": "eq",  "key": "source", "value": "handbook.pdf"},
  {"op": "gte", "key": "page",   "value": 10},
  {"op": "not", "filter": {"op": "exists", "key": "retracted"}}
]}
```

Ops: `eq` `ne` `lt` `lte` `gt` `gte` `in` `exists` `and` `or` `not`. The
semantics are the library's, unchanged — including the one most likely to be
"fixed" by a well-meaning edit: **every comparison is false on an absent key,
`ne` included.** `not(eq(...))` is how to also match vectors that lack the key.

## Errors are one shape

```json
{"error": {"code": "invalid_vector", "message": "govecdb: invalid vector: 2 values, want 4"}}
```

`code` is stable and is what a client should branch on. `message` is for a human.

| Code | Status | Means |
|---|---|---|
| `invalid_request` | 400 | Malformed body, unknown field, bad duration, no vectors. |
| `invalid_vector` | 400 | Wrong dimension, empty id, a non-finite value. |
| `invalid_metadata` | 400 | A value that is not a string, bool or number. |
| `invalid_filter` | 400 | An unknown op, a missing key, a tree too deep. |
| `invalid_spec` | 400 | A collection configuration the database refuses. |
| `invalid_name` | 400 | A collection name outside the allowed set. |
| `not_found` | 404 | No such collection, vector, or route. |
| `already_exists` | 409 | A collection of that name is already there. |
| `payload_too_large` | 413 | Over `MaxBodyBytes`. |
| `unsupported_media_type` | 415 | A `Content-Type` other than `application/json`. |
| `unauthorized` | 401 | Missing or wrong bearer token. |
| `read_only` | 503 | A durability failure; the database refuses writes until restarted. |
| `too_many_open` | 503 | Every collection slot is busy. Retry — `Retry-After` is set. |
| `unavailable` | 503 | Shutting down. |
| `internal` | 500 | Anything else. The cause is in the log. |

`net/http` answers an unrouted path and a wrong method in plain text, which would
leave a client parsing two formats and finding the second in production. A
middleware rewrites those two into the shape above, keeping the `Allow` header
the mux attaches to a 405.

## Authentication is one shared token

`Config.AuthToken` requires `Authorization: Bearer <token>` on every route except
`/healthz` and `/readyz`. The comparison is constant-time.

One token, no users and no roles. That is the honest shape of what this is — a
database process on a private network — and inventing an identity model would
imply an authorization story it does not have. Anything richer belongs in a proxy
in front, which is also where TLS termination, rate limiting and audit belong.

`/metrics` is deliberately **not** exempt: it names every collection and reports
its size. Health probes are, because a probe that can fail for an authentication
reason reports the wrong thing about the process.

## Metrics: counters and gauges, no histogram

A latency histogram is the thing you actually want and it is not worth what it
costs here — bucket boundaries chosen without a dependency are boundaries chosen
badly, and a hand-rolled one produces a number that looks like a quantile and is
not. So there is a request count and a total duration, whose ratio is an honest
mean.

Percentile latency belongs to the observability seam in the library (v2 item 1),
where it can be measured at the operation rather than at the socket.

Collection names are interpolated into labels without escaping, which is safe
only because `service.ValidateName` allows nothing a label parser reacts to. That
is a real dependency between two packages, so it is written down in both.
