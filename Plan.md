### Executable items

Original list, kept as written. Inline comments are the review verdict on each.

1. We need to make our grpah multi threaded, add locks or use sync.pool fr scratch buffers  // pool first, then lock
2. Delete function in hnsw graphtone not actual delete after everysome period we will rebuild the graph  // right call
3. Now hnsw as everythiing we need to have a wal writer, data format how we store it with crc32 checksum  // merge with 5
4. WAL Reader, checksum validator, wal truncator  // absorbs item 6
5. Then segment manager creaate segment files then to manage them  with offset tracker also  with global offset  // fold into 3
6. Retention go routine which deletes old files (need to verify is it needed or not)?  // drop it
6. We need checkpoint writer reader and also files creater.  // keep as-is
7. We need WAL periodic flusher to flush to disk  // make it a knob
// wal -> durable part is done upto here cojmpleltyi think mentioned everything
// durability yes, complete no — nothing here is importable

8. On startup reuild graph, inmemory  // part of 4 + checkpoint

---

### Why three items changed

**Item 1 — `Search` is not read-only.** It writes `queryBuf`, `visited`, and the
three scratch slices on every call, so plain `RLock` on `Search` is a data race,
not a fast path. Pool the scratch first; only then does `RWMutex` mean anything.

**Item 5 folds into 3.** Segment rotation isn't an addition to a WAL writer —
truncation is impossible without it. A single-file WAL built first is rework.

**Item 6 is not needed.** Age-based retention and checkpoint-based truncation are
different things and only the second is required: once a checkpoint at seq N is
durable, every segment below N is deletable immediately. No timer, no goroutine.
Time-based retention serves PITR and audit, which v1 does not have.

---

### Super plan — one sentence per task

#### A · Finish the graph (nothing durable should target a moving structure)

1. Move `visited`, `scratchCands`, `scratchRes`, `scratchSel`, and `queryBuf` off
   `Graph` into a pooled `searchState` so `Search` stops mutating shared state,
   then guard the graph with an `RWMutex` and re-baseline the benchmarks.
2. Add tombstone-based `Delete` that keeps slot indices stable, keeps traversing
   *through* dead nodes while filtering them out of results, and re-elects the
   entry point when the entry node itself is deleted.
3. Give `Insert` real upsert semantics so a duplicate id replaces instead of
   silently no-opping, fixing the operation set before the WAL format freezes.
4. Add a compaction pass that rebuilds the graph once tombstones cross a
   threshold, since tombstoned slots never release memory on their own.

#### B · Durability

5. Define the versioned record — `magic|version` file header, then
   `crc32 | type | seq | len | payload` per entry — and write the append-only
   writer with segment rotation built in from the first commit.
6. Write the reader that validates every CRC, stops at the first bad record, and
   truncates the torn tail rather than failing recovery, because a partial final
   record is the normal outcome of power loss.
7. Decide the write-failure policy once, at the WAL boundary: what a full disk or
   a failed `fsync` does to the DB (read-only, or errors forever).
8. Write the checkpoint serializer that dumps in-memory state to a temp file,
   fsyncs it, and atomically renames — never by re-reading the WAL.
9. Wire recovery as load-newest-valid-snapshot → replay records above its seq →
   open for writes, which is where item 8 of the original list actually lives.
10. Add the background flusher with an fsync policy knob (always / interval /
    never) plus the checkpoint scheduler that truncates segments below the last
    durable snapshot.
11. Build a crash harness that `SIGKILL`s a child mid-write, reopens, and asserts
    the surviving prefix is exactly consistent — the highest-value test here.

#### C · Make it a library (the third that is currently missing)

12. Add the payload/metadata store so an id carries its original text and fields,
    not just a vector.
13. Add collections/namespaces so one process serves many independent indexes,
    with load-on-demand and idle eviction.
14. Define the `Index`, `Store`, and `WAL` interfaces and build the public
    `db.go` / `vector.go` / `options.go` / `errors.go` facade on top of them.
15. Implement `Close()` — flush, final fsync, optional checkpoint — so a clean
    shutdown is not indistinguishable from a crash.
16. Write the examples and rewrite the README against the real API.

---

### One sequencing warning

Do **not** leave task 14 for the very end. Sketch `db.go` as a stub early, even
unimplemented, so the public API shape pressure-tests the internals while they
are still cheap to change. Discovering that eight internal packages do not
compose is much more expensive after all eight exist.

### Locked baselines

Task 1 puts these in play — renegotiate consciously with new numbers, do not let
them drift:

| Baseline | Value | Guarded by |
|---|---|---|
| Recall@10, dim 32 | 0.999 | `TestRecallVsBruteForce` |
| Recall@10, dim 768 | 0.972 | `TestRecallHighDimension` |
| Search allocations | 2 allocs/op | `BenchmarkSearch -benchmem` |
