# gRPC and clustering — do we need them, and what instead

Decided 2026-10-13, after v1.5.0. [CLUSTER.md](CLUSTER.md) is the design for
*how* to replicate and shard; this is the decision on *whether*, *when*, and
what to build first. It overrides CLUSTER.md's stage order where they differ.

## The verdict

| Question | Answer | Do instead, now | Revisit when |
|---|---|---|---|
| gRPC (v2 item 7b) | **No.** | Binary vectors on the REST API, HTTP/2 without TLS, an official Go client — all stdlib, all in this module. | A user needs a gRPC-native client (polyglot codegen, a service mesh that speaks only gRPC), **or** after binary vectors the HTTP layer is still > 15% of a request, measured. |
| Sharding (item 8) | **No.** | Nothing — one node holds the known workload ~5× over. | One collection needs > 50% of a node's RAM, **or** sustained ingest exceeds what one node's `AddBatch` sustains (~19K vec/s at dim 256). |
| Replication (item 8) | **Not yet as a cluster; yes to the groundwork.** | Online backup (export and install a snapshot), which is also the first replication primitive. | Someone states an availability target (failover in minutes, not a restore) or a recovery point tighter than the snapshot interval. |

## Why

### gRPC buys four things, and three are available without it

What gRPC would give this server, and what each costs here:

| gRPC gives | Measured need | Stdlib answer |
|---|---|---|
| Binary vectors instead of JSON floats | **Real.** Decoding a 1,000 × 512 `AddBatch` body takes **42.5 ms** (`BenchmarkDecodeAdd1000x512`, the hand-written path) — on the order of the index apply itself. A 512-dim search body is **40 µs** to decode against ~380 µs to search 100K × 512, ~10%. | A base64 little-endian float32 field next to `values`. Decoding is a copy, not a parse. |
| One multiplexed connection for a fan-out | **Real** for the downstream client: a 16-shard search sends 16 requests at once. | `http.Server.Protocols.SetUnencryptedHTTP2` (Go 1.24+, already our floor). |
| Streaming bulk ingest | Not measured as a need: `AddBatch` takes 10,000 per call, and concurrent calls now share a commit. | Chunked NDJSON, if ever. |
| Generated typed clients | **The one real bug so far was a client's:** the downstream app's hand-written client treated `503 overloaded` as permanent and dropped shard results (FEEDBACK.md). | An official Go client in this module that owns retry and `Retry-After`. |

Against that, gRPC costs a separate module (grpc-go and protobuf are not
stdlib, and `go.mod` has no `require` block by design), a second API surface
duplicating every bound the REST layer already enforces — auth, body limits,
filter width, the two backpressure pools, error codes — and a `.proto` that is
a compatibility promise of its own. The REST layer took a review per release to
get those right. Doing it twice for an encoding we can get in one field is the
wrong trade.

### Sharding answers a capacity problem nobody has

Per vector at dim 512: 2,048 B of vector, ~370–395 B of graph (measured after
the flat layer-0 layout), plus id and metadata — call it ~3.2 KB. One million
is ~3.2 GB; a 64 GB node holds the order of ten million with room for a
compaction's second graph. The only workload at scale reported so far is
"millions of 512-dim vectors across sixteen collections" — well inside one
node — and that application already shards client-side across daemons over
REST. Search scales across cores (9.3× at 16 goroutines), and since v1.5
concurrent writes share a commit.

A cluster's router, partition map, scatter-gather and split would be the most
complex code in the project, guarding against a limit no workload has reached.
The levers that raise the single-node ceiling come first, and they are smaller:
a flat vector arena (if compaction's memory sharing can be kept), a
sub-quadratic prune for the single-writer insert, and — when the owner lifts
its block — the quantized index, which is the biggest one: vectors are two
thirds of memory at dim 512.

### Replication answers a failure problem, and that one is real

Replication is not about the ceiling. A node is a single point of failure for
two different things:

- **The process or host.** A restart from a snapshot is seconds — 1M × 128
  loads in ~0.36 s, then replays the log since the last snapshot. A supervisor
  restarting the daemon is adequate availability for most users today.
- **The disk.** Lose it and everything since the last copy taken *off the
  machine* is gone. There is no supported way to take that copy while the
  database is running: snapshot files can be copied by hand, but nothing says
  which set is consistent or how to restore one.

The second is the honest gap, and closing it is the first half of replication
anyway — CLUSTER.md §6's `ExportSnapshot` / `InstallSnapshot` are exactly a
consistent online backup and its restore. So that ships first, standing on its
own value, and nothing built for it is thrown away if replication follows.

### The epoch is not urgent

CLUSTER.md §5 says the WAL epoch must land "before v2 ships". It need not. The
segment format is versioned *per segment* and the reader would accept a v1
segment as epoch 0, so adding the field later costs exactly what it costs now —
there is no installed base of followers to stay compatible with. Its real
deadline is **before the first replica exists**, and it lands as the first step
of Phase 3, not speculatively.

## The plan

Each phase is green-gated commits, measured before and after, in the order
below. Phases 1 and 2 are scheduled; 3–5 start only on their trigger.

### Phase 1 — a faster, safer REST surface (this module, stdlib)

| # | Deliverable | Exit gate |
|---|---|---|
| 1.1 | **Binary vectors.** `values_b64` (base64 of little-endian float32) accepted wherever `values` / `query` are; `?encoding=b64` returns vectors that way from get, `GetBatch` and scan. Exactly one of the two per vector, else `400`. Same validation — dimension, finiteness — after decoding. | `BenchmarkDecodeAdd1000x512` under 5 ms on the b64 path (from 42.5); JSON remains the default and is unchanged byte for byte. |
| 1.2 | **HTTP/2 without TLS.** `-h2c` on `govecdbd` sets `Protocols` with unencrypted HTTP/2 beside HTTP/1.1. Off by default; TLS deployments already negotiate HTTP/2. | A daemon test drives 16 concurrent searches over one h2c connection. |
| 1.3 | **`client/` — an official Go client.** Typed calls over `net/http`, `context.Context` at this layer (the library still takes none), b64 vectors by default, and the retry rules written once: `overloaded` and `unavailable` retried honouring `Retry-After` with jittered backoff and a ceiling, `too_many_open` likewise, everything else returned. Retrying is safe because a PUT is an upsert and a delete is idempotent. | The FEEDBACK.md scenario as a test — a front layer refusing each first add and search with the daemon's own error body — passes against the real handler, and fails with retries off. |

### Phase 2 — online backup (this module)

| # | Deliverable | Exit gate |
|---|---|---|
| 2.1 | `(*DB).ExportSnapshot(w io.Writer) (seq uint64, err error)`: streams a consistent snapshot — the same payload and checksummed trailer as `Snapshot` — to any writer, holding `writeMu` only as `Snapshot` does. | Exported bytes pass `snapshot` verification; a database restored from them has the same `Range` as the source at `seq`. |
| 2.2 | `InstallSnapshot(dir string, r io.Reader) (seq uint64, err error)` — a function, like `wal.Replay`: verifies end to end before anything is renamed into place, refuses a non-empty directory. | A torn or corrupted stream installs nothing; a good one opens and replays. |
| 2.3 | `GET /v1/collections/{name}/export` (streamed, auth required, lifts its write deadline like `snapshot`) and `govecdbd -restore <file> -collection <name>`. | Daemon test: export under concurrent writes, restore into a fresh root, compare. |
| 2.4 | `docs/OPERATIONS.md`: how to back up, what the recovery point is (the export's `seq`), how to restore. | Reviewed against the tests above. |

### Phase 3 — a warm standby (trigger: an availability or RPO target)

CLUSTER.md stage 2, minus everything that needs consensus:

1. The WAL epoch and the snapshot `lastEpoch` (CLUSTER.md §5), frozen-layout
   tests and DURABILITY.md in the same commit; a v1 log still replays.
2. `DurableSeq`, `ReadLog`, `Apply`, `WithReplica`, `Promote`, the retention
   floor (CLUSTER.md §6).
3. A follower mode in `govecdbd` that bootstraps from Phase 2's export and
   tails the leader's log over HTTP — stdlib, so it stays in this module.
   Promotion is manual.

Exit: CLUSTER.md stage 2's property test (identical `Range` under random
writes, restarts and torn tails; a reissued sequence is `ErrDiverged`), and
`kill -9` of the leader loses nothing at or below the follower's applied
sequence. The recovery point is replication lag, stated as such.

### Phase 4 — automatic failover (trigger: zero RPO or unattended failover)

CLUSTER.md stages 3–4: quorum acks and a Raft control plane. This is the first
phase that needs a third-party dependency, so it is the first that lives in the
separate `govecdb-cluster` module.

### Phase 5 — sharding (trigger: capacity, as in the verdict table)

CLUSTER.md stages 1, 5 and 6. Start client-side: the Phase 1 client can own
hash routing and the exact k-way merge — what the downstream application
already does by hand — before any server-side router exists.

## What changes elsewhere

- `MIGRATION.md`'s v2 table points items 7b and 8 here.
- `CLUSTER.md` keeps the design; its stage order and its "before v2" deadline
  for the epoch are superseded by this document.
