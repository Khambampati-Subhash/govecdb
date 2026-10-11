# Architecture diagrams

End-to-end design for GoVecDB (current as of v1.4.1). Sources are [D2](https://d2lang.com); the
`.svg` files are generated and render directly on GitHub.

```bash
d2 --layout elk docs/diagrams/01-overview.d2 docs/diagrams/01-overview.svg
```

| Diagram | Covers |
|---|---|
| [01-overview](01-overview.svg) · [src](01-overview.d2) | The whole system around one `govecdb.DB` and its RAM/disk boundary — public API, the validation boundary, the three DB locks, WAL writer, in-RAM graph + metadata store, background snapshot / prune-truncate / calibration goroutines, on-disk layout and recovery. `service`/`httpapi`/`govecdbd` appear only as a separate layer above |
| [02-insert-path](02-insert-path.svg) · [src](02-insert-path.d2) | Write path for `Add`/`AddBatch`: validate → **WAL append under `writeMu` (the commit point)**, sync policies and sticky failure → apply under `applyMu` (the **upsert branch**, serial `link` or parallel `InsertBatch` chunks, `store.Put`) → calibration kick → ack, plus what a crash at each point means |
| [03-search-path](03-search-path.svg) · [src](03-search-path.d2) | Query path: `validateSearch` (filter validated once) → auto-ef (`SuggestedEf` × `efScale`) → under `applyMu.RLock`: normalize → greedy descent → layer-0 `searchLayer` (prefetch, `admits()` = live ∧ filter) → top-k → in-RAM metadata. No disk I/O anywhere; 1 alloc/op in the graph; the rejected disk-vector and post-filter designs |
| [04-hnsw-internals](04-hnsw-internals.svg) · [src](04-hnsw-internals.d2) | What the index actually stores — `Graph`, `node` and the pooled `searchState` field by field, the layered graph, level assignment, alpha-pruned neighbor selection (alpha **1.0**), the slot lifecycle (deletes *and* updates both leave tombstones; online `Compact`), memory budget |
| [05-persistence-recovery](05-persistence-recovery.svg) · [src](05-persistence-recovery.d2) | WAL file and record format, segment rotation and fsync ordering, sync policies, snapshot format and the Create → Prune → Verify-oldest → Truncate cycle, and recovery: snapshot fallback, replay of seq > N, a tear ends only its segment, `ErrCorrupt` when the log no longer reaches back |

## The five invariants these encode

Everything above follows from five decisions. If a future change contradicts one
of them, it is the change that is wrong.

1. **Vectors are RAM-resident.** HNSW computes a distance at *every hop* of
   traversal, so `node.vector` can never be a disk read. Metadata is in RAM
   too (`internal/store`), so a search does no I/O at all; disk holds only the
   WAL and snapshots. Making traversal disk-friendly is DiskANN — a different
   index, out of scope for v1.

2. **The WAL append is the commit point.** Append (and fsync, per policy) *then*
   apply to the graph, then ack. Reverse those and a crash between them means
   acknowledging a write that no longer exists.

3. **Snapshots are written from RAM, never by re-reading the WAL.** The log is
   read exactly once in the life of a process: at startup. Segments become
   deletable only below the **oldest retained** snapshot, and only once it
   verifies — truncating to the newest would strand the fallback copy.

4. **The graph is derived state.** It can always be rebuilt from snapshot + WAL
   replay. Replay links on several workers by default, so the rebuilt graph is
   equivalent, not identical; with `WithInsertWorkers(1)` and `WithSeed` it is
   bit-identical, which is what makes recovery testable by equality.

5. **A slot index is immutable for the life of a graph.** Slots are never
   removed, reused, or renumbered — every neighbor list in the graph is a list of
   indices, so shifting one would invalidate all of them. `Delete` flags a slot;
   an update flags the old one and appends a new one; neither ever edits a slot
   another node might be pointing at. Compaction does not break this invariant,
   it *retires* it: the rebuild produces a **new** graph that is swapped in
   whole, never a renumbering of the live one.

## Three clocks, often confused

| Clock | Cadence | What it bounds |
|---|---|---|
| WAL append | every write, synchronous | nothing on its own — under the fast policies it lands in a 64 KiB user-space buffer |
| WAL **fsync** | every write by default (`SyncAlways`, ~4.24 ms); 50 ms under `SyncInterval`; never under `SyncNever` | how much acknowledged data a crash can destroy |
| **Checkpoint** | off in the library; 10 min in `govecdbd` | recovery time and WAL size on disk |

50 ms is a reasonable *fsync* interval and a wildly wrong *checkpoint* interval.
A snapshot has a ~12 ms floor at any size (two fsyncs).

## Dark mode

All five render on a black canvas. Each source sets `theme-id: 200` (Dark Mauve)
so default label, markdown, and inline-code text renders light, plus a root
`style.fill: "#000000"` for the background.

Note that D2 applies explicit styles on top of any theme — a dark theme alone
would have left every hardcoded pastel fill in place. The palette below is
therefore set per shape, with hue preserved so the semantic coding survives:

| Meaning | Surface | Stroke |
|---|---|---|
| RAM / in-memory | `#0D2137` | `#4FC3F7` |
| WAL | `#33240E` | `#FFB74D` |
| Disk | `#0E2A16` | `#66BB6A` |
| Critical / ordering | `#3B1215` | `#FF6B6B` |
| API surface | `#241B33` | `#B39DDB` |
| Decision / caveat | `#332B0D` | `#FFD54F` |
| Container group | `#0C0C14` | `#44445A` |

Containers need an **explicit** fill. Without one they fall back to the theme's
pale slate, which reads as a light box wrapping dark children — hence the
`group` class.

## Notes on the D2 sources

Two D2 features are TALA-only and silently unavailable under dagre/ELK, which is
why the sources avoid them:

- **`near: <object>`** — anchoring a note beside another shape. ELK and dagre
  accept only constants (`top-center`, …). Annotations here are attached with
  faint dashed edges instead, which keeps the layout engine from giving each
  note its own column.
- **`direction:` on a nested container** — ignored by both engines. The layer
  bands in `04` use `grid-columns` instead, which is engine-independent and
  still renders intra-container edges.

One more, unrelated to layout engines: **a shape used as a container draws its
own label behind its children.** `01`'s background-workers group was a hexagon
and its title ended up underneath the Checkpointer node. Containers are plain
rectangles here; only leaf nodes get decorative shapes.
