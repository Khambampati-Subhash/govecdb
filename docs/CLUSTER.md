# GoVecDB — horizontal scaling design (v2 item 8)

Status: **design, not built.** This is the plan for sharding and replication,
grounded in the code as of v1.3.1. Only the engine primitives in
[§6](#6-what-lands-in-this-module) land in this module; everything else lives in
a separate module, `govecdb-cluster`, which imports this one. `go.mod` here
keeps no `require` block.

## Decisions

| Question | Decision |
|---|---|
| Shard key | FNV-1a 64 (`hash/fnv`) of the vector id, split into contiguous hash ranges. Never by vector space. |
| Search | Full `k` to every shard, `Ef = 0`, exact k-way merge on `(Distance, ID)`, dedupe by id. |
| Replication | Leader + followers per shard, by **shipping WAL records**; followers apply through the replay path. |
| Raft | **Control plane only** — shard map, epochs, leaders, in-sync set. Not the data log. |
| Format changes | An `epoch` in the WAL record header and in the snapshot payload header — **before v2 ships**. |
| Minimum production cluster | 3 nodes, RF 3, quorum acks, 3-voter Raft controller co-located on the nodes, stateless routers over the existing REST API. |

## 1. Facts the design rests on

1. **A sequence number is not a record identity across a leader restart.**
   Recovery resumes at `max(nextSeq, res.NextSeq())` (`recovery.go`), so the
   sequence numbers of records lost in a torn tail are reissued to different
   content. Locally that is correct; for a follower that already received the
   lost records it is silent divergence. This is the entire case for an epoch.
2. **Applied state cannot be rolled back.** Upsert and delete both tombstone.
   A replica that applied a record outside the leader's history must be
   re-bootstrapped, never rewound — so followers apply only records the leader
   has made durable.
3. **The writer assigns sequence numbers itself** (`internal/wal/writer.go`);
   there is no way to append under a given sequence today.
4. **Under `SyncInterval`/`SyncNever` acknowledged records sit in a 64 KiB
   user-space buffer**, and a record can straddle two flushes. Shipping is
   bounded by a *durable* sequence, never `LastSeq`.
5. **Truncation ignores followers** — it deletes below the oldest retained
   snapshot only.
6. **`Match.Distance` is comparable across shards of one spec**: one kernel per
   metric, cosine normalized on insert. arm64 (NEON, FMA) and amd64 can differ
   by ulps, which can change a tie-break but never correctness — hence the
   `(Distance, ID)` merge order.
7. **Search width per shard follows the shard's own size**: `SuggestedEf` and
   `Calibrate` already use the shard's `Len`.

## 2. Sharding

- **Hash on id, not on vector space.** Partitioning by space makes an upsert
  that moves a vector a cross-shard delete+put, makes hot shards out of
  clustered data, drifts with the data and loses recall at the boundaries. By
  hash, `Get`/`Add`/`Delete` go to exactly one shard with no lookup.
- **P partitions, each a contiguous `[lo, hi)` of the 64-bit hash space.** The
  hash function and the range table are structural and recorded in the cluster
  spec. Each partition replica is an ordinary `service.Manager` collection
  `<coll>-<pid>` (base name ≤ 56 bytes to fit `ValidateName`).
- **Choosing P.** Total search work is `S·(n/S)^0.78 = S^0.22 ×` the
  single-shard cost (1.84× at 16, 2.5× at 64); wall-clock latency falls roughly
  as `S^-0.78`. Size P so a partition holds ~2–10M vectors at dim 512.
- **Rebalance by moving partitions** (add replica → hand over leadership → drop
  old), no rebuild. **Split** is the rare operation: read `S0 = LastSeq`, `Range`
  the source into the two children via `AddBatch`, tail the log from `S0+1`
  routing each record by hash, fence writes briefly at the router, drain, swap
  the range table at a new map epoch, drop the source. Converges because PUT is
  idempotent and deleting an absent id is not an error.

## 3. Scatter-gather search

- **Every shard gets the full k.** ef ∝ k^0.2, so a smaller per-shard k saves
  almost nothing, and each global top-k item is in its own shard's top-k —
  global recall ≈ per-shard recall at the same target.
- **`Ef = 0` to every shard.** A client-supplied ef is per shard; one sized for
  the whole collection overshoots every shard by ~`S^0.78`.
- **Merge** with a k-way heap on `(Distance, ID)`. Cross-shard `Scan`: same
  `after` cursor to every shard, merge by id — exact.
- **Tail latency**: per-shard timeouts, a hedged request to an in-sync replica
  after ~p95, and opt-in `allow_partial` (response marked `partial: true`).
  Default fails the query.

## 4. Replication

**Shipped**: the leader's WAL records as-is (type, epoch, seq, payload), bounded
by the leader's durable sequence, over HTTP/2 as octet-stream so the CRC travels
end to end.

**Applied**: `DB.Apply` reuses the replayer, so runs of PUTs take the parallel
`InsertBatch` path. Under the writer lock it validates epoch and sequence,
appends the records to the follower's own WAL **under the leader's seq and
epoch**, syncs per policy, then applies. Log-first holds, replica snapshots are
keyed by leader seq, a replica restart is ordinary recovery, and a promoted
replica can serve its log to others.

| Apply rule | Result |
|---|---|
| epoch < replica's epoch | `ErrStaleEpoch` (fences a deposed leader) |
| within one epoch, seq not contiguous | refused |
| seq ≤ last applied, under a new epoch | `ErrDiverged` → re-bootstrap |
| gap | legal only at an epoch boundary |

**Bootstrap**: the leader registers a retention hold; `ExportSnapshot` streams
the newest verified snapshot file raw (header + trailer, so the same checksum
verifies it); the follower `InstallSnapshot`s it (temp → fsync → verify → rename
→ fsync dir), opens `WithReplica`, and pulls `ReadLog(seq+1)`.

**Retention**: `truncateLog` deletes below `min(oldest snapshot, retention
floor)`; the floor callback belongs to the cluster. Giving up on a laggard is
the cluster's decision: raise the floor, the laggard gets `ErrLogTruncated` and
re-bootstraps.

**Commit and consistency**
- Writes: durable locally, then wait for a quorum of in-sync replicas to ack a
  durable seq ≥ the write's. `acks=leader` is opt-in async.
- Leader reads may observe a write not yet replicated (documented; avoiding it
  would split `Add`).
- Follower reads have bounded staleness, observable as seq lag. Read-your-writes:
  writes return `seq`; a search carrying `min_seq` goes to a replica at or past
  it (bounded wait), else the leader.

**Failover**: every open as writer takes a fresh epoch from the controller. The
controller promotes the in-sync member with the highest `(epoch, seq)` and bumps
the epoch. A rejoining node presents `(lastEpoch, lastSeq)`; past where that
epoch ended in the new leader's log ⇒ wipe and re-bootstrap. A partitioned old
leader cannot reach quorum (followers refuse its epoch), so its writes fail
loudly rather than vanish.

**Why Raft is not the data log**: a Raft log in front of the WAL doubles the
fsync, which *is* the cost of a write (4.24 ms vs 780 ns); using the WAL as the
Raft log needs suffix truncation, contradicting append-only logical truncation;
commit-then-apply would split `Add`; one Raft group per shard means multi-Raft
machinery. Precedent: PacificA. Raft (e.g. hashicorp/raft) lives only in the
cluster module's controller.

## 5. Format changes — before v2 ships

1. **WAL record header gains `epoch uint64`, covered by the CRC**:
   `crc32c | type | epoch | seq | len` — 25 bytes instead of 17 (~0.4% of a
   512-dim record). Segment format version 1 → 2; the reader accepts both, a v1
   segment reads as epoch 0, and the version is per segment so a mixed directory
   replays. Replay adds one check: epochs never decrease. In the record rather
   than a side file: a side epoch file must be fsynced in lockstep with the log
   (Kafka KIP-101 / KIP-279 are that bug class); in-record, `(epoch, seq)` cannot
   disagree with itself. `TestLayoutIsFrozen` and `DURABILITY.md` change in the
   same commit.
2. **Snapshot payload header gains `lastEpoch`** (payload version 1 → 2), so a
   follower restarted right after `InstallSnapshot` can still run the divergence
   check.

The record payload encoding becomes a wire contract between node versions:
upgrade followers first.

## 6. What lands in this module

Root package:

```go
type LogOp uint8 // LogPut | LogDelete
type LogRecord struct {
	Epoch, Seq uint64
	Op         LogOp
	Payload    []byte // aliases a reused buffer; Clone to keep
}
func (r LogRecord) Clone() LogRecord

// Leader side
func (db *DB) ReadLog(from uint64, fn func(LogRecord) error) (next uint64, err error) // bounded by DurableSeq
func (db *DB) DurableSeq() uint64
func (db *DB) Epoch() uint64
func (db *DB) ExportSnapshot(w io.Writer) (seq uint64, err error)
func WithLogRetention(floor func() uint64) Option
func WithLogNotify(fn func(durableSeq uint64)) Option // synchronous, no alloc; WithObserver's rules

// Follower side
func WithReplica() Option // client writes -> ErrReplica; Apply allowed
func (db *DB) Apply(recs []LogRecord) (lastSeq uint64, err error)
func (db *DB) Promote(epoch uint64) error
func InstallSnapshot(dir string, r io.Reader) (seq uint64, err error) // a function, like wal.Replay

// Resharding
func DecodeLogRecord(r LogRecord) (v Vector, isPut bool, err error)

var ErrReplica, ErrDiverged, ErrStaleEpoch, ErrLogTruncated error
```

New events: `Promoted`, `Diverged`, `LogRetained`. Nothing fires per record.

Internal: `wal.Options.Epoch`; `Writer.AppendRecords` taking caller-assigned,
strictly increasing `(epoch, seq)`; `Writer.DurableSeq`; `wal.ReadFrom(dir,
from, upTo, …)` that never reads past the durable point; `snapshot.Install`.
`service`: `Options.OpenOptions(name)`, `Manager.CreateFromSnapshot`; cluster
nodes run with eviction off. `httpapi`: `ErrReplica` → `421 not_leader` with a
leader hint; write responses carry `seq`.

Never in this module: Raft, gRPC, the router, the shipper transport.

## 7. Router and minimum cluster

- **Router**: stateless, N instances behind a load balancer, stdlib only,
  caching the shard map from the controller and speaking the existing REST API.
  JSON is fine for queries (~5 KB at dim 512); move bulk ingest to a binary
  framing only once measured.
- **Node**: `service.Manager` + `httpapi.Server`, plus replication endpoints.
  gRPC (7b) is a later client option and blocks nothing.
- **Minimum production-reliable**: 3 nodes, RF 3, `acks=quorum`, leaders on
  `SyncInterval`, co-located 3-voter controller, fresh epoch per writer open,
  snapshot bootstrap, retention floor, automatic failover, fixed P with
  partition moves. Static sharding alone (Stage 1) scales but loses data with a
  disk.

## 8. Blockers found in the current code

| # | Blocker | State |
|---|---|---|
| 1 | Any writer (fsync, `AddBatch`, a snapshot with a queued write) stalls every search through `db.mu` and the graph's `RWMutex` | being fixed: writer-only lock, snapshot and compact under it |
| 2 | Overlapping snapshots fail each other; `Close` does not wait for a manual one | being fixed with 1 |
| 3 | Sequence numbers are reissued after a torn tail | needs the epoch (§5) |
| 4 | The writer cannot take an external sequence; no durable-sequence bound | Stage 2 |
| 5 | Truncation ignores followers | Stage 2 (retention floor) |
| 6 | `Compact` held the write lock for a full rebuild — a failover trigger on a leader | being fixed: rebuild under the read lock, swap under the write lock |
| 7 | Writes return no sequence | Stage 2 (`seq` in responses) |
| 8 | `Range` sorts every live id once (~80 MB at 5M ids) | budget for it in a split |
| 9 | `service.Manager` eviction and `Drop` (`RemoveAll`) | configure off / fence on cluster nodes |

## 9. Stages

| # | Deliverable | Module | Exit gate |
|---|---|---|---|
| 0 | Search off the writer path; WAL v2 with epoch; snapshot payload v2; `DurableSeq` | this | frozen-layout tests updated; a v1 log replays; no search stall under a looping writer or a snapshot |
| 1 | Static sharding: router, hash ranges, exact merge, cross-shard `Scan` | cluster | merged recall within 0.02 of single-shard at equal per-shard target |
| 2 | `ReadLog`, `Apply`, `WithReplica`, `Promote`, retention floor, notify, export/install snapshot, `DecodeLogRecord`; service + httpapi hooks | this | leader+follower property test under random writes, restarts and torn tails gives identical `Range`; reissued seq ⇒ `ErrDiverged`; truncate respects the floor |
| 3 | Shipper, follower loop, bootstrap, quorum acks, `min_seq` reads, manual promote | cluster | `kill -9` / power-cut the leader: no quorum-acked write lost |
| 4 | Raft controller: map, epochs, in-sync set, automatic failover | cluster | partition tests: no acked write lost, no split-brain write accepted |
| 5 | Partition move | cluster | zero write errors during a move under load |
| 6 | Split | cluster | final id set equals the source's |

Online compaction (v2 item 2) should land before Stage 4. The quantized index
(item 6) is independent and is the largest lever on vectors per node.

## 10. Risks

- Leader reads can observe unreplicated writes (accepted, documented).
- Re-bootstrap is a full snapshot transfer — GBs per partition.
- Fan-out tail latency grows with S: keep S ≲ 64 per collection, and hedge.
- The payload format becomes a wire contract; upgrade order matters.
- A stuck retention floor means unbounded log growth; the cluster must cap lag
  and force re-bootstrap.
