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

The append path allocates nothing: the record header is reused across calls and
the payload is written straight through without a copy.

Rotation costs an fsync, a close and a create — which is why `MaxSegmentBytes`
defaults to 64 MiB rather than something that would make it frequent.

## Not implemented yet (deliberately)

Reading. There is no reader, no CRC-validating scan, and no torn-tail
truncation — that is the next slice, and it is why `WAL` has no `Replay` method
yet: an interface method with no implementation is a promise, not a design.

Recovery may not even belong on the live log. Replaying is something done
*before* a writer exists, to rebuild state, which reads more naturally as a
package-level function over a directory than as a method on the thing currently
appending. That gets settled with an implementation in hand.

Also pending: checkpointing, segment truncation, and the crash harness that
`SIGKILL`s a child mid-write and asserts the surviving prefix is exactly
consistent. See `docs/MIGRATION.md`.

```bash
go test ./internal/wal/ -v
go test ./internal/wal/ -race
go test ./internal/wal/ -run='^$' -bench=. -benchmem
```
