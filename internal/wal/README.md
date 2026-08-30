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

| Policy | Per append | Throughput | What an acknowledged write means |
|---|---|---|---|
| `SyncAlways` | **4.06 ms** | 295 writes/s | It survived power loss. |
| `SyncInterval` | 1.01 µs | ~1M writes/s | It survived the process dying; up to one interval is lost to power loss. |
| `SyncNever` | 0.81 µs | ~1.2M writes/s | It reached the OS. A clean shutdown keeps it; power loss may not. |

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

## Measured

Apple M4 Max, ~2 KB payloads:

| | ns/op | allocs |
|---|---|---|
| Append, `SyncAlways` | 4,058,089 | 0 |
| Append, `SyncInterval` | 1,013 | 0 |
| Append, `SyncNever` | 809 | 0 |
| Append, small (12 B) | 18.9 | 0 |
| Checksum, 16 KB | 1,373 (11.9 GB/s) | 0 |
| Segment rotation | 5,344,281 | 7 |
| **Replay, per record** | **358** (5.8 GB/s) | **0** |

The append path allocates nothing: the record header is reused across calls and
the payload is written straight through without a copy.

Rotation costs an fsync, a close and a create — which is why `MaxSegmentBytes`
defaults to 64 MiB rather than something that would make it frequent.

Replay allocates nothing per record either; the ~16 allocations it does make are
**per segment** — a 64 KiB read buffer and the file handle. Recovering a
1 GB log is therefore a few seconds of streaming, not a few seconds of GC. At
358 ns/record, replaying a million records costs about 0.36 s.

## Not implemented yet (deliberately)

Checkpointing and segment truncation. `TypeCheckpoint` is reserved in the format
so the numbering is not rearranged later, but nothing writes it yet and nothing
deletes segments below one — that arrives with `internal/snapshot`, which is what
makes a checkpoint mean anything.

Also pending: the crash harness that `SIGKILL`s a child mid-write and asserts the
surviving prefix is exactly consistent. The tear tests here damage a log by
truncating and rewriting it, which reproduces the *shapes* power loss leaves
behind but not the timing that produces them. See `docs/MIGRATION.md`.

```bash
go test ./internal/wal/ -v
go test ./internal/wal/ -race
go test ./internal/wal/ -run='^$' -bench=. -benchmem
```
