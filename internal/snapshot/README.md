# internal/snapshot

Point-in-time state on disk, so recovery does not have to start at the beginning
of the write-ahead log.

## Why the sequence number is the whole point

The WAL alone is a complete recovery story, and a slow one: a log appended to for
a week takes a week's worth of replaying. A snapshot is the fixed point that
makes the log finite. It records the state as of WAL sequence N, and recovery
becomes *load the snapshot, then replay from N+1* — which is also what makes
every WAL segment below N deletable, the only thing that stops a log growing
forever.

So the sequence is not metadata attached to a snapshot. It is the reason the file
exists, and it is why it is checksummed as carefully as the payload: **a snapshot
whose sequence is wrong by one replays the log from the wrong place and silently
loses or duplicates a write.**

## Files (one responsibility each)

| File | Responsibility |
|------|----------------|
| `doc.go` | Package overview, the format, and this file map. |
| `format.go` | The wire format: header, trailer, and the checksum. |
| `file.go` | Snapshot naming, discovery, `List`, `Latest`. |
| `create.go` | Writing one atomically. |
| `load.go` | Verifying, and applying the newest that survives verification. |
| `prune.go` | Retention: how many to keep, and why more than one. |
| `errors.go` | Sentinel errors callers match on. |

## The format

```
header:   magic "GVSS" (4) | version (2) | reserved (2) | seq (8)
payload:  opaque, streamed
trailer:  crc32c (4) | payload length (8)
```

Sixteen-byte header, twelve-byte trailer. `TestLayoutIsFrozen` pins both, because
an on-disk format that changes when a Go type changes is a format that eats its
own data.

**The checksum is in a trailer, not the header**, because the payload is
*streamed*. A snapshot is gigabytes where a WAL record is kilobytes, so nothing
here ever holds the whole thing in memory — and a header checksum would mean
either buffering it all or seeking back to patch it.

It covers the **header and the payload**. The trailer's length field is not
covered and does not need to be: it is cross-checked against the size of the file
containing it, which is a stronger statement than a hash of the field alone.

**The payload is opaque**, exactly as it is in the WAL. This package moves bytes
durably and knows nothing about vectors or graphs, which keeps it testable
without the index and honest about where the format boundary sits.

Snapshots are **named for the sequence they cover** (`snap-00000000000000000042.snap`)
rather than numbered separately, because that number is what a caller needs: it
says where to resume the log and which segments are safe to delete. Twenty digits
is every value a `uint64` can take, fixed width so lexical order matches numeric
order.

## Writes are atomic — and that is what the checksum is *not* for

Temp file → fsync → rename → **fsync the directory**.

A crash leaves either no snapshot or a complete one, never a half-written file
wearing a finished name. That ordering lets the checksum mean something narrower
and more useful: it guards against bit rot and against tampering after the fact,
not against interrupted writes. Leave atomicity to the checksum instead and every
crash produces a plausible-looking snapshot that only fails at the moment
recovery needs it.

The **directory fsync** is the step that is easy to omit and impossible to notice
missing. Without it the rename can be in the filesystem's journal and not on the
platter, so a crash leaves a snapshot under neither name — temp already unlinked,
final not yet durable. The error is propagated rather than swallowed: some
filesystems refuse to sync a directory, and ignoring that would silently
downgrade the guarantee the whole function exists to provide.

## Nothing unverified reaches the caller

`Load` hashes the file end to end **before** the callback sees a byte.

Applying unverified bytes is how corruption on disk becomes corruption in memory,
and a callback that has already ingested half a bad snapshot cannot un-ingest it.
Streaming straight into the callback and reporting the checksum failure
afterwards would be faster and would make the failure unrecoverable.

Because verification comes first, **the callback is called at most once**, never
for a snapshot that turns out to be bad. It needs no idempotence and no undo.

### What that costs, measured

| 64 MiB snapshot | |
|---|---|
| Verify pass (checksum) | 10.5 ms |
| Apply pass | + 3.7 ms |
| **`Load` total** | **14.5 ms** |
| Streaming once, checksum after | 10.5 ms |

So the property costs about **+38%**, not the 2× an extra pass suggests: the
verify pass leaves the file in the page cache, so the apply pass runs at
**18.2 GB/s** because it is reading memory. On a 1 GiB snapshot that is under
60 ms — bought against the alternative, rebuilding an index from the log, which
is three orders of magnitude more. See [`docs/DURABILITY.md`](../../docs/DURABILITY.md).

### A failed snapshot falls back to an older one

This is the reason to retain more than one. A corrupt newest snapshot costs a
longer WAL replay, not the database. Every rejection is reported in
`Result.Rejected`, which on a healthy machine is empty — and is the only place a
failing disk shows up.

A **callback** error is *not* a reason to fall back. A snapshot that verified but
could not be decoded is a bug in the decoder, and quietly reaching for an older
file would hide it behind a slow startup.

### What is fatal, and what is a rejection

| Condition | Outcome |
|---|---|
| Bad magic / unknown version / short / checksum / length / sequence mismatch | Rejected — try the next-oldest |
| Every snapshot rejected | `Found: false` — replay the whole WAL, which is slow but correct |
| No snapshots at all | `Found: false` — the normal first startup |
| I/O error | Fatal — trying three more files on a disk that is not answering is theatre |
| Callback error | Fatal — a decoder bug, not disk corruption |

## Retention, and the ordering that makes fallback real

`Prune(dir, keep)` deletes all but the newest `keep`, plus temporaries a crash
left behind. `keep < 1` is refused: deleting the last snapshot is not retention,
it is deletion, and it leaves recovery with nothing but a log nobody may truncate.

**WAL truncation must be driven by the *oldest retained* snapshot, never the
newest, and must run after `Prune`.** A caller that prunes to the newest snapshot
and then truncates the log below it has just made the second copy unusable while
still paying to store it. This package does not enforce that — the WAL is not its
business — but it is the part that is easy to get backwards.

Oldest is deleted first, so an interrupted prune leaves a contiguous run of the
newest snapshots rather than holes in the middle of the history.

**Single writer.** `Prune` removes every temporary it finds, so it must not run
while a `Create` is in flight. A snapshot directory belongs to one database
instance, the same way a WAL directory does — the WAL refuses a second writer
outright with `O_EXCL`.

## Measured

Apple M4 Max:

| | ns/op | throughput | allocs |
|---|---|---|---|
| `Create`, 1 MiB | 10,103,512 | 104 MB/s | 24 |
| `Create`, 64 MiB | 34,205,718 | 1.96 GB/s | 24 |
| `Load`, 1 MiB | 451,833 | 2.3 GB/s | 30 |
| `Load`, 64 MiB | 14,521,713 | 4.6 GB/s | 30 |
| verify only, 64 MiB | 10,453,687 | 6.4 GB/s | 10 |
| apply only, 64 MiB | 3,690,253 | 18.2 GB/s | 6 |

`Create` carries a **~10 ms floor** at any size — two fsyncs, one for the file and
one for the directory. That is the cost of the atomicity guarantee, it does not
shrink with the payload, and it is why snapshots are taken on a checkpoint
interval measured in minutes rather than on a WAL fsync interval measured in
milliseconds. Allocation counts are per snapshot, not per byte.

## Not implemented yet (deliberately)

**The graph codec.** Nothing here knows how to turn a `hnsw.Graph` into bytes;
the payload is opaque and there is no producer for it yet. That is a real format
decision — writing neighbour lists to disk freezes HNSW's internal representation
the way `TestLayoutIsFrozen` freezes this one — and it deserves its own slice
rather than being smuggled in behind a blob store.

**Restore orchestration** — *load the snapshot, then replay the WAL from
`Seq+1`* — is a dozen lines once the codec exists, and lands with the public API
that has both a graph and a log to hand.

**WAL checkpointing and segment truncation.** Unblocked by this package now: a
snapshot at sequence N is what a `TypeCheckpoint` record points at and what
authorises deleting segments below the oldest retained snapshot. See
`docs/MIGRATION.md`.

```bash
go test ./internal/snapshot/ -v
go test ./internal/snapshot/ -race
go test ./internal/snapshot/ -run='^$' -bench=. -benchmem
```
