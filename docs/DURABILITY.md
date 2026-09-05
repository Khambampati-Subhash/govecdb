# Durability and Latency

What GoVecDB promises about surviving a crash, what each promise costs, and the
measurements behind both.

Every number here was measured in one session on one machine — **Apple M4 Max,
darwin/arm64, Go 1.25, APFS on internal NVMe** — so the figures can be compared
against each other. Reproduce them with the commands at the bottom.

> **Status.** The durability path is complete and wired together behind
> `govecdb.Open`: writes go to the log before the index, startup restores from
> the newest usable snapshot and replays what came after, snapshots can run on a
> schedule, and log segments a snapshot has made redundant are deleted. [What is
> not yet guaranteed](#what-is-not-yet-guaranteed) is explicit about what remains.

---

## 1. The ordering rule

Everything below follows from one rule:

> **A write is appended to the WAL first, and only then applied to the in-memory
> graph.**

Reverse those two and a crash between them acknowledges a write that no longer
exists — the one failure a database is not allowed to have. The graph is
*derived state*: it can always be rebuilt by replaying the log, and it is never
the source of truth.

That is also why **reads never touch the disk**. Search runs entirely against the
in-memory graph, so query latency is completely independent of the sync policy.
Durability is a write-side cost, in full.

---

## 2. How durable, exactly

Durability is a knob, and this is what each setting actually promises.

| Policy | Per append | Throughput | A crashed **process** loses | **Power loss** loses |
|---|---|---|---|---|
| `SyncAlways` *(default)* | **4.04 ms** | 248 writes/s | nothing | nothing |
| `SyncInterval` | 897 ns | ~1.1M writes/s | ≤ one interval (50 ms) | ≤ one interval (50 ms) |
| `SyncNever` | 692 ns | ~1.4M writes/s | ≤ 64 KiB | everything not yet written back |

**Durability costs about 5,800×.** That gap is why this is a knob and not a
constant — and why the zero value is `SyncAlways`: a caller who configures
nothing gets the safe answer, not the fast one.

### The part that is easy to get wrong

Records land in a **64 KiB user-space buffer** first. Under `SyncInterval` and
`SyncNever`, an acknowledged write has not necessarily reached the *kernel*, let
alone the platter — so **a process crash loses it just as a power cut does**.
Only `SyncAlways`, an explicit `Sync()`, or a clean `Close()` closes that window.

Three separate places bytes can be sitting, and only the last is durable:

```
Append() ─→ 64 KiB bufio buffer ─→ OS page cache ─→ platter
            ^ lost to a process   ^ lost to power   ^ safe
              crash                 loss
```

`SyncAlways` moves a record through all three before `Append` returns.
`SyncInterval` moves it through all three on a 50 ms tick. `SyncNever` moves it
out of the first only when it fills.

### What is protected, and against what

| Failure | Protected by | Outcome |
|---|---|---|
| Process killed mid-write | WAL-first ordering | Log truncated at the last complete record; graph rebuilt from it |
| Power loss mid-write | Per-record checksum | Torn tail detected and dropped; the prefix replays intact |
| Power loss mid-**rotation** | Old segment fsynced before the new one is created | No hole in the middle of the log — the one shape recovery cannot repair |
| Power loss just after a segment is created | **Directory fsync** on segment creation | The segment cannot vanish along with the acknowledged writes inside it |
| Power loss mid-**snapshot** | Temp file → fsync → rename → directory fsync | Either no snapshot or a complete one; never a half-written file wearing a finished name |
| Power loss mid-**truncation** | Oldest segment deleted first, directory fsynced | A contiguous run of the newest segments survives — never a hole in the middle |
| Truncation against a snapshot that is silently corrupt | The oldest retained snapshot is **verified** before anything is deleted | Truncation is skipped; the log keeps growing rather than losing records |
| Bit rot in a WAL record | crc32c over type, seq, length **and** payload | Replay stops there; that segment is truncated |
| Bit rot in a snapshot | crc32c over header and payload, verified before use | Snapshot rejected; falls back to an older one |
| A corrupt record **length** | Length refused against file size and cap *before* any read | No wild allocation — the failure that actually hurts |
| A flipped bit in the sequence truncation reads | The segment's first record is read through the checksummed reader | An unverified sequence never authorises deleting a file |
| A snapshot renamed to the wrong sequence | Sequence checksummed in the header, cross-checked against the name | Rejected — replaying the log from the wrong place is silent data loss |
| Sequence numbers rewound | Strict-increase check during replay | `ErrOutOfOrder` rather than an ambiguous apply order |
| A failed write followed by more writes | Sticky failure — the Writer refuses everything after the first error | No hole for replay to stop at |

### What "acknowledged" means at each layer

| Layer | Durable when |
|---|---|
| `wal.Writer.Append` returns, `SyncAlways` | Immediately — data *and* the segment's directory entry are on the platter |
| `wal.Writer.Sync()` returns | Immediately, any policy |
| `wal.Writer.Close()` returns | Immediately, any policy |
| `snapshot.Create` returns | Immediately — file fsynced, renamed, directory fsynced |
| A vector is visible to `Search` | Says nothing about durability — that is the in-memory graph |

---

## 3. Write latency

End to end, one vector (dim 128, `M=16`, `EfConstruction=200`, ~2 KB WAL record):

| Stage | Cost | Notes |
|---|---|---|
| WAL append, `SyncAlways` | 4.04 ms | 0 allocs; the fsync dominates everything |
| WAL append, `SyncInterval` | 897 ns | 0 allocs |
| WAL append, `SyncNever` | 692 ns | 0 allocs |
| WAL append, small (12 B) | 19.0 ns | Per-record framing cost with the payload removed |
| HNSW insert | **703 µs** | 50.9 KB, 208 allocs |
| HNSW upsert (replace an id) | 798 µs | A tombstone plus a full insert |
| HNSW upsert, unchanged vector | **311 ns** | The replay path — an early return, not an insert |
| Segment rotation, amortized | 0.27 µs | 8.65 ms once per 64 MiB ≈ 32,000 records |

**Which half is the bottleneck depends entirely on the policy:**

| Policy | WAL | Index | Total | Rate | Bound by |
|---|---|---|---|---|---|
| `SyncAlways` | 4.04 ms | 0.70 ms | **4.74 ms** | ~211 writes/s | the **disk** (85%) |
| `SyncInterval` | 0.90 µs | 0.70 ms | **0.70 ms** | ~1,420 writes/s | the **index** (99.9%) |
| `SyncNever` | 0.69 µs | 0.70 ms | **0.70 ms** | ~1,420 writes/s | the **index** (99.9%) |

So: turning off durability buys **6.7×** on the write path, not the 5,800× the
append benchmark alone suggests — because once the fsync is gone, building the
graph is all that is left. Anyone tempted by `SyncNever` for write throughput
should know they are trading all durability for less than an order of magnitude.

That `311 ns` row matters more than it looks. Re-inserting an unchanged vector is
the WAL-replay path, and it costs one comparison rather than one insert — which
is what stops recovery across a snapshot boundary from paying full price for
changing nothing.

---

## 4. Read latency

Reads are served from memory and are **unaffected by the sync policy**. Corpus of
10,000 × 128 dims, `k=10`, `ef=64`, cosine.

| | Latency | Allocs |
|---|---|---|
| `Search`, single-threaded | **105 µs** | **2** |
| `Search`, 16 threads | 7.9 µs/op wall | 2 |

Parallel search scales ~13× on 16 threads: `Search` holds only a read lock, and
per-traversal scratch comes from a pool, so readers genuinely run concurrently.
That is ~126,000 queries/s.

**By dimension** (n=2,000) — every hop computes a distance, so dimension
multiplies the whole traversal rather than just the final comparison:

| Dimension | 32 | 128 | 384 | 768 | 1536 |
|---|---|---|---|---|---|
| Latency | 34 µs | 68 µs | 147 µs | 298 µs | 589 µs |

**By corpus size** (dim 128) — sublinear, which is the entire point of HNSW:

| Vectors | 1,000 | 5,000 | 20,000 |
|---|---|---|---|
| Latency | 52 µs | 90 µs | 117 µs |

20× the data costs 2.3× the time. Note that recall at a *fixed* `ef` falls as
`n` grows — use `SuggestedEf` rather than a hardcoded number.

**By tombstone load** — deleted vectors keep their edges and are still traversed:

| Dead slots | 0% | 25% | 50% | 75% |
|---|---|---|---|---|
| Latency | 104 µs | 128 µs | 169 µs | 270 µs |

**By filter selectivity** (`BenchmarkSearchFilter`, 10,000 × 128, `k=10`,
`ef=64`) — a metadata filter is applied *during* the traversal, so a search that
can only accept one vector in fifty has to travel further to find ten of them:

| Admitted | none (unfiltered) | 1 in 2 | 1 in 10 | 1 in 50 |
|---|---|---|---|---|
| Latency | 85 µs | 165 µs | 385 µs | 965 µs |
| Allocs | 2 | 2 | 2 | 2 |

This is the same curve tombstones produce and for the same reason: a node the
filter rejects still rides the search frontier, but never enters the result set,
so `results` fills slowly, the pruning bound stays loose, and the traversal
widens until it has `k`. That is what makes a filtered search return `k` results
rather than "however many of the nearest `k` happened to match" — measured at 10
against 2 for a one-in-fifty filter in `TestSearchFilterFindsKWherePostFiltering-
WouldNot`.

**The allocation count does not move.** The filter costs traversal width, not
garbage: the predicate is a lookup against the metadata store that borrows the
map instead of copying it (`store.Map.Match`, asserted at 0 allocs), and the
search's own 2 allocs/op baseline is untouched.

Past roughly one in a hundred, the graph stops being the right tool — a scan over
the metadata, distance-checking only what matches, beats a traversal that is
visiting most of the graph anyway.

---

## 5. Snapshots

| | Latency | Throughput | Allocs |
|---|---|---|---|
| `Create`, 1 MiB | 10.1 ms | 104 MB/s | 24 |
| `Create`, 64 MiB | 34.2 ms | 1.96 GB/s | 24 |
| `Load`, 1 MiB | 452 µs | 2.3 GB/s | 30 |
| `Load`, 64 MiB | 14.5 ms | 4.6 GB/s | 30 |
| — verify pass | 10.5 ms | 6.4 GB/s | 10 |
| — apply pass | 3.7 ms | 18.2 GB/s | 6 |

`Create` carries a **~10 ms floor at any size** — two fsyncs, one for the file
and one for the directory that names it. It does not shrink with the payload,
which is why snapshots ride a checkpoint interval measured in **minutes**, not
the WAL's fsync interval measured in milliseconds.

`Load` reads the file twice on purpose: hash it end to end, *then* hand it to the
caller. Applying unverified bytes is how corruption on disk becomes corruption in
memory, and a caller cannot un-ingest half a bad snapshot. The cost is **+38%**
over streaming once, not the 2× an extra pass suggests, because the verify pass
leaves the file in the page cache and the apply pass then runs at 18.2 GB/s.

Two clocks, often confused:

| Clock | Cadence | What it bounds |
|---|---|---|
| WAL fsync | ~50 ms | How much acknowledged data power loss destroys |
| Checkpoint / snapshot | seconds to minutes | Recovery time, and log size on disk |

50 ms is a reasonable fsync interval and a wildly wrong checkpoint interval.

---

## 6. Recovery

Startup replays the log and rebuilds the graph. The two halves are not close in
cost:

| Stage | Per record | 1M records |
|---|---|---|
| Reading the log (`wal.Replay`) | 368 ns (5.7 GB/s, **0 allocs/record**) | 0.37 s |
| Applying it (`hnsw.Insert`) | 703 µs | **703 s** (11.7 min) |

**Reading the log is 0.05% of recovery.** Rebuilding the index is everything.

That single fact settled the design question the snapshot phase left open —
whether a snapshot should store the **graph** or just the **live vectors**. The
graph codec now exists, so this is measured rather than projected. A graph of
dim 128 at `M=16` encodes to **662 bytes per vector**, so 1M vectors is ~662 MB:

| Snapshot holds | 1M × 128 recovery | Cost |
|---|---|---|
| Live vectors | Rebuild every vector at 703 µs | **~703 s** |
| The graph itself | verify 662 MB at 6.4 GB/s, decode at 2.46 GB/s | **~0.37 s** |

About **1,900×** — three orders of magnitude. A vectors-only snapshot would bound
log *size* while leaving recovery *time* essentially unimproved, which is half a
snapshot. The price of the choice is that the codec freezes HNSW's internal
representation on disk, the way `TestLayoutIsFrozen` freezes the WAL's.

### The graph codec

| | Latency | Throughput | Allocs |
|---|---|---|---|
| `WriteTo`, 10k × 128 | 1.36 ms | 4.9 GB/s | **5** |
| `Read`, 10k × 128 | 2.69 ms | 2.46 GB/s | 50,676 |

`WriteTo` allocates a **constant** five times regardless of graph size — the
fixed-field scratch and the payload buffers are reused, so a snapshot of a
million vectors allocates the same five times as one of ten. `Read` allocates
~5 per node because those allocations *are* the graph: the node, its vector, its
neighbour slices, its id.

Reading is slower than writing because it builds a data structure rather than
copying bytes. It is still ~2,600× faster than rebuilding the index from the
same vectors.

For reference, `Compact()` is the same rebuild operation and confirms the shape —
its pause tracks *survivors*, not garbage:

| Dead ratio | 25% | 50% | 75% |
|---|---|---|---|
| Compact (5,000 × 128) | 2.65 s | 1.69 s | 0.80 s |

### Log truncation, and the rule that keeps the fallback real

After each snapshot, segments holding no record the retained snapshots still need
are deleted. Two decisions make that safe rather than merely tidy:

**Against the oldest retained snapshot, never the newest.** Keeping two snapshots
is what makes a corrupt one survivable, and that only works if the log still
reaches back far enough for the *older* one to be replayed on top of. Truncating
to the newest would delete exactly those records, leaving a second copy that is
paid for and cannot be used. `TestTruncationLeavesTheOlderSnapshotUsable`
destroys the newest snapshot after truncation and requires full recovery from the
older one.

**And only after verifying that snapshot reads.** The question truncation asks is
"may I delete the records this snapshot stands in for?", and a snapshot nobody
has checked cannot stand in for anything. If it fails verification, truncation is
skipped — a log that keeps growing is a disk problem, while deleting records only
an unreadable snapshot could replace is a data problem, and the two are not close
enough to trade.

`WithSnapshotsKept` is therefore the knob that decides how much log survives:
more retained snapshots means an older oldest, and an older oldest means less is
deleted.

### What recovery tolerates

- **A torn tail** in any segment — truncated there, replay continues with the
  next segment. Tolerating it *anywhere* rather than only in the last segment is
  required, not lenient: `Open` always starts a new segment, so a second crash
  leaves a torn tail in the middle of the directory.
- **A corrupt record** — the segment is truncated at that point.
- **A zero-filled block** — rejected by the checksum before the type check;
  record type 0 is invalid precisely so zeroed space never decodes as data.
- **A missing or empty log** — a database that has never been written to.
- **A corrupt newest snapshot** — falls back to an older one, at the cost of a
  longer replay. This is why retention keeps more than one.
- **Every snapshot corrupt** — not an error; replay the whole WAL, which is slow
  and correct.

---

## 7. What is not yet guaranteed

Stated plainly, because a durability document that only lists strengths is
marketing.

- **`TypeCheckpoint` is reserved and unwritten.** Truncation reads the snapshot
  directory directly, which is the authority on what is recoverable; a log record
  duplicating that could disagree with it. The constant stays so the numbering is
  not rearranged later.
- **Torn *writes* within a record are detected, never repaired.** The crash
  harness (below) shows that under `SyncAlways` a killed process reliably leaves
  the log at a clean record boundary, because the bytes reach the file before
  fsync is called. A tear needs the kill to land inside the `write` itself — a
  narrow window, covered deterministically in `replay_test.go` by damaging a log
  directly rather than by hoping to hit it.
- **Single writer, assumed rather than enforced.** `O_EXCL` on segment creation
  catches two processes starting together, but not one joining later. There is no
  lock file. A snapshot directory and a log directory each belong to one database
  instance.
- **No redundancy, only detection.** Checksums say *that* something is corrupt,
  never what it was. A damaged WAL record costs everything after it in that
  segment; a damaged snapshot costs a fallback to an older one.
- **`fsync` is trusted.** If the drive or the virtualization layer lies about
  flushing — consumer SSDs with volatile write caches, some cloud disks — no
  amount of ordering here helps. That caveat is not specific to this database.
- **The log directory's own parent is not fsynced.** Creating the directory is
  durable only once its parent is, and this stops at the log directory.

---

## 8. Reproducing these numbers

```bash
go test ./internal/hnsw/ -run='^$' -bench='Insert|Upsert|Search|Compact' -benchmem -timeout 30m
```

```bash
go test ./internal/wal/ ./internal/snapshot/ -run='^$' -bench=. -benchmem -timeout 20m
```

The correctness claims are tests rather than prose — `go test ./... -race` runs
them all. The ones that carry this document:

| Claim | Test |
|---|---|
| Torn tail truncated, prefix intact | `TestReplayTornTail` |
| Damage in a middle segment is survivable | `TestReplayContinuesPastATearInAnEarlierSegment` |
| Three crashes in a row stay contiguous | `TestReplayToleratesRepeatedTears` |
| A corrupt length never causes a wild read | `TestReplayRefusesAnUnverifiedLength` |
| Zeroed space never decodes as a record | `TestReplayStopsAtZeroFilledSpace` |
| A rewound sequence is refused | `TestReplayDetectsRewoundSequences` |
| `SyncAlways` is durable when `Append` returns | `TestSyncAlwaysIsDurableOnReturn` |
| A failed snapshot leaves nothing behind | `TestCreateIsAtomic` |
| Corrupt snapshots never reach the caller | `TestLoadRejects` |
| A corrupt snapshot falls back to an older one | `TestLoadFallsBackToAnOlderSnapshot` |
| The log stops growing | `TestLogDoesNotGrowForever` |
| Truncation keeps the older snapshot usable | `TestTruncationLeavesTheOlderSnapshotUsable` |
| Truncation is skipped when the oldest snapshot is unreadable | `TestTruncationIsSkippedWhenTheOldestSnapshotIsUnreadable` |
| An unverified sequence cannot delete a segment | `TestTruncateRefusesAnUnverifiedSequence` |
| On-disk layouts cannot drift | `TestLayoutIsFrozen` (wal, snapshot) |
