# internal/wal

The append-only write-ahead log: what makes GoVecDB survive a crash.

## The ordering rule is the whole point

A write is appended here **first**, and only then applied to the in-memory graph.

Reverse those two and a crash between them means acknowledging a write that no
longer exists — the one failure a database is not allowed to have. The graph is
*derived state*: it can always be rebuilt by replaying this log, and it is never
the source of truth.

## Files (one responsibility each)

| File | Responsibility |
|------|----------------|
| `doc.go` | Package overview, the format, and this file map. |
| `record.go` | The wire format: encode, decode, and the checksum. |
| `segment.go` | Segment file naming and discovery. |
| `options.go` | `Options`, `SyncPolicy`, and their defaults. |
| `writer.go` | The append-only writer, rotation, and the sync policies. |
| `reader.go` | The validating scan of a single segment. |
| `replay.go` | Recovery across a directory, and what it reports. |
| `truncate.go` | Deleting segments a snapshot has made redundant. |
| `wal.go` | The `WAL` interface callers depend on, plus `Nop`. |
| `errors.go` | Sentinel errors callers match on. |

## The format

```
segment file:  magic "GVWL" (4) | version (2) | reserved (2)
record:        crc32c (4) | type (1) | seq (8) | length (4) | payload
```

Eight-byte file header, seventeen-byte record header. `TestLayoutIsFrozen` pins
both numbers, because an on-disk format that changes because a Go type changed is
a format that eats its own data.

### Versioned from the first byte written
A log format without a version field cannot be migrated later, only abandoned. A
segment written by a newer format is **refused**, not guessed at — refusing is
the entire point of carrying the field.

### The checksum covers everything after it
Not just the payload: the type, the sequence number and the **length** as well.

A flipped bit in a payload gives a wrong answer. A flipped bit in a *length*
gives a read of arbitrary size at an arbitrary offset — that is the failure that
actually hurts, and it is why nothing may be allocated on the strength of a
length until its checksum has passed.

Castagnoli rather than the IEEE default, because that polynomial has SSE4.2 and
ARMv8 hardware support: **11.9 GB/s** measured, so the check costs a few cycles
per cache line rather than a table walk.

### Zero is not a valid record type
Unwritten space, a filesystem hole, and a zero-filled block all read as zeros.
None of them is a record, and none of them should ever decode as one.

## Segments, and why they are here from the first commit

The log is a **directory of numbered segments**, not one growing file.

That is load-bearing rather than tidy. Once a snapshot at sequence N is durable,
every record below N is dead weight — and deleting whole files is something every
filesystem does instantly, while punching a hole in the front of a large file is
not. Segment rotation is not an addition to a WAL writer; truncation is
impossible without it, so building a single-file log first would have been rework.

`MaxSegmentBytes` is therefore a **truncation granularity, not a size limit**: a
record larger than a segment is still written, into a segment of its own.
Refusing it would make the two settings secretly coupled.

### Opening always starts a new segment

Even when segments already exist, `Open` never appends to the last one.

A segment whose tail was cut off by power loss ends in a partial record, and
recovery stops at the first record that fails its checksum. Appending after that
partial record would put perfectly good writes *behind* a permanent stopping
point, where replay can never reach them — silent data loss produced by the
recovery mechanism itself. A fresh segment costs one mostly-empty file per
restart and makes that impossible.

Rotation fsyncs the old segment **before** creating the new one, for the same
reason at a smaller scale: otherwise a crash could leave the new segment on disk
while the tail of the old one was still in the page cache. That is a hole in the
*middle* of the log rather than at its end, and it is the one shape recovery
cannot repair.

## Durability is a knob, and here is its price

| Policy | Per append | Throughput | A crashed process loses | Power loss loses |
|---|---|---|---|---|
| `SyncAlways` | **4.04 ms** | 248 writes/s | nothing | nothing |
| `SyncInterval` | 897 ns | ~1.1M writes/s | ≤ one interval | ≤ one interval |
| `SyncNever` | 692 ns | ~1.4M writes/s | ≤ 64 KiB (the buffer) | everything not written back |

Note what the fast policies do **not** promise. Records live in a 64 KiB
user-space buffer until it fills or something flushes it, so under `SyncInterval`
and `SyncNever` an acknowledged write has not necessarily reached the *kernel*,
let alone the disk — a process crash loses it just as a power cut does. Only a
clean `Close` (or an explicit `Sync`) makes that window zero.

**Durability costs about 4,000×.** That gap is why this is a knob and not a
constant — and why the zero value is `SyncAlways`: a caller who configures
nothing gets the safe answer, not the fast one. Silent data loss should never be
the default anyone falls into by omission.

`SyncInterval` uses a timer rather than a check inside `Append`, because the
write that most needs flushing is the last one before the traffic stops —
precisely the one no later `Append` is coming to trigger.

### Two clocks, often confused

| Clock | Cadence | What it bounds |
|---|---|---|
| WAL **fsync** | ~50 ms | How much acknowledged data power loss destroys |
| **Checkpoint** | seconds to minutes | Recovery time, and log size on disk |

50 ms is a reasonable fsync interval and a wildly wrong checkpoint interval.

## Failure is sticky

The first write error ends the `Writer`'s useful life; every later call returns
it.

If append N never reached the disk, appending N+1 on top produces a log with a
hole in it — and replay stops at the hole, silently discarding everything after.
Continuing after a failed write is how a durability bug becomes a data-loss bug.

What the *database* does about that — refuse writes, go read-only, fail closed —
is the wider policy decision and belongs a layer up. This is the half that has to
live here.

## Recovery

```go
res, err := wal.Replay(dir, opts, func(r wal.Record) error {
    return apply(r) // rebuild the graph from the log
})
if err != nil {
    return err
}
w, err := wal.Open(dir, wal.Options{FirstSeq: res.NextSeq()})
```

`Replay` is a **function, not a method on `WAL`** — recovery runs *before* a
writer exists. Making it a method would have meant either opening a writer in
order to read (creating a segment as a side effect of recovery) or a second
constructor handing back a `WAL` that cannot write. Rebuilding state is done to a
directory, so it takes a path.

### Nothing is read on the strength of an unverified length

A record announces its own payload length, and that length arrives **before** the
checksum that would prove it. So the file size is taken up front and every length
is refused unless it fits both what remains of the file *and* `MaxRecordBytes` —
before a byte of payload is read and before a byte of memory is reserved for it.

That guard is the difference between a flipped bit and a 3 GB allocation, and it
is why the writer has to enforce the same cap: a record it could write but the
reader would refuse is a record that can never be read back.

### A tear ends a segment, not the replay

A record that fails to validate — torn, corrupt, oversized, or an unknown type —
stops that segment's scan. Everything from there to the end of the file is
dropped, the stop is reported as a `Tear`, and **replay continues with the next
segment**.

Continuing is required rather than lenient. `Open` always starts a *new* segment,
so the shape a second crash leaves behind is: segment K torn by the first crash,
segment K+1 full of good records written after the restart. A reader that only
forgave damage in the *last* segment would make a database unrecoverable from
precisely the situation the writer is designed to produce.

It is safe for the same reason it is necessary. A tear can only ever be at the
end of a segment's written region: the writer's failure is sticky, so it never
writes past a point it failed at, and rotation fsyncs the old segment before the
new one exists. Records cannot hide behind a tear, because the writer never put
any there.

### Truncation is logical, not physical

Recovery is a read. The damaged bytes stay on disk — nothing will ever append to
them, so rewriting the file would buy nothing and cost the one copy of the
evidence that a crash happened.

### Sequence numbers must strictly increase

A **gap** is fine and expected; it is what a tear leaves behind. A **repeat** is
`ErrOutOfOrder`, because two records claiming one identity make the order the log
specifies unrecoverable. Almost always the cause is a `Writer` opened without
carrying `FirstSeq` forward from `res.NextSeq()`.

### The payload is only valid during the callback

`fn` receives bytes pointing into a buffer the reader reuses — that is what keeps
replay at **zero allocations per record**. A callback that keeps a record past
its return must `Clone()` it first. Same contract as `bufio.Scanner.Bytes`, and
the same footgun, so it is stated here rather than left to be discovered.

### What is fatal, and what is not

| Condition | Outcome |
|---|---|
| Torn / corrupt / oversized record | Tear — segment truncated there, replay continues |
| Segment file too short to hold a header | Tear at offset 0 (a crash between `create` and the header write) |
| Zero-filled space | Tear: the checksum rejects it before the type does |
| Callback returns an error | Fatal — the state built is a prefix, not the log |
| Sequence rewind | Fatal (`ErrOutOfOrder`) |
| Bad magic / unknown format version | Fatal — guessing at an unknown layout is worse than failing |

## Truncation

`Truncate(dir, keepFromSeq, opts)` deletes segments holding no record at or after
`keepFromSeq`. It is what stops the log growing forever: once a snapshot covers
sequence N, every record at or below N can be rebuilt from it.

### Judging a segment without a sequence range

A segment does not record which sequences it holds, and finding its **last** one
means scanning it to the end. So the decision runs the other way round.
Sequences increase across the whole log, so every record in segment *i* comes
before every record in any later segment *j*:

> If some later segment starts **at or below** `keepFromSeq`, then segment *i*
> ends below it too, and segment *i* is disposable.

One record read per segment instead of a full scan. It is deliberately
conservative — a segment whose last record sits just under the line may survive
an extra round — and conservative is the right direction for a deletion.

That one record is read through the **ordinary reader**, checksum and all, not by
pulling the header apart. The sequence is covered by the record's checksum, and
this number authorises deleting files: trusting an unverified one would let a
single flipped bit destroy a segment. `TestTruncateRefusesAnUnverifiedSequence`
is the guard.

### What it refuses to reason about

- **The newest segment**, always. Nothing follows it, so nothing can vouch for
  where it ends — and it is the one a writer is appending to.
- **A segment whose successor cannot be read**: an empty one left by a restart, a
  torn first record, a file that is not a segment. Truncation is an optimization,
  and declining costs disk while guessing costs data.

Oldest is deleted first, so an interrupted truncation leaves a contiguous run of
the newest segments rather than holes in the middle. The directory is fsynced
afterwards, for the same reason creating a segment is.

### The caller's constraint

**`keepFromSeq` must come from the oldest *retained* snapshot, not the newest.**

Retaining two snapshots is what makes a corrupt one survivable, and that only
works if the log still reaches back far enough for the older one to be replayed
on top of. Truncating to the newest would delete exactly those records, leaving a
second copy that is paid for and cannot be used.

Safe to call while a `Writer` is appending: it only ever removes segments below
the one being written.

## Measured

Apple M4 Max, ~2 KB payloads:

| | ns/op | allocs |
|---|---|---|
| Append, `SyncAlways` | 4,036,239 | 0 |
| Append, `SyncInterval` | 897 | 0 |
| Append, `SyncNever` | 692 | 0 |
| Append, small (12 B) | 19.0 | 0 |
| Checksum, 16 KB | 1,542 (10.6 GB/s) | 0 |
| Segment rotation | 8,649,214 | 10 |
| **Replay, per record** | **368** (5.7 GB/s) | **0** |

The append path allocates nothing: the record header is reused across calls and
the payload is written straight through without a copy.

Rotation costs **two** fsyncs, a close and a create: one for the outgoing
segment's data, one for the directory that gives the incoming segment its name.
The second is what stops a crash from taking a freshly created segment away
along with the acknowledged writes inside it — `fsync` on a file makes its
contents durable and says nothing about the directory entry naming it.

It is also why rotation is 8.6 ms rather than 4.8. That sounds expensive until it
is amortized: at the default 64 MiB it happens once per ~32,000 records of 2 KB,
which is **0.27 µs per record** — below even `SyncNever`'s per-append cost. It is
the right place to spend an fsync, and the reason `MaxSegmentBytes` defaults to
64 MiB rather than something that would make rotation frequent.

Replay allocates nothing per record either; the ~16 allocations it does make are
**per segment** — a 64 KiB read buffer and the file handle. Recovering a
1 GB log is therefore a few seconds of streaming, not a few seconds of GC. At
358 ns/record, replaying a million records costs about 0.36 s.

## Not implemented yet (deliberately)

**`TypeCheckpoint` is reserved and stays unwritten.** It was meant to mark the
point a snapshot had made durable, so segments below it could be dropped — but
truncation reads the snapshot directory directly, which is the authority on what
is actually recoverable. A record duplicating that could disagree with it, and a
log record claiming a snapshot exists is worth less than the snapshot. The
constant stays so the numbering is not rearranged later; replay ignores the type
rather than refusing it, so a log written by a build that does emit them still
loads.

**The crash harness** that `SIGKILL`s a child mid-write and asserts the surviving
prefix is exactly consistent. The tear tests here damage a log by truncating and
rewriting it, which reproduces the *shapes* power loss leaves behind but not the
timing that produces them. See `docs/MIGRATION.md`.

```bash
go test ./internal/wal/ -v
go test ./internal/wal/ -race
go test ./internal/wal/ -run='^$' -bench=. -benchmem
```
