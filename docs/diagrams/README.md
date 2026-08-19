# Architecture diagrams

End-to-end design for GoVecDB v1. Sources are [D2](https://d2lang.com); the
`.svg` files are generated and render directly on GitHub.

```bash
d2 --layout elk docs/diagrams/01-overview.d2 docs/diagrams/01-overview.svg
```

| Diagram | Covers |
|---|---|
| [01-overview](01-overview.svg) · [src](01-overview.d2) | The whole system and the RAM/disk boundary — API, validation, namespace registry, WAL engine, graph, background checkpointer, disk layout |
| [02-insert-path](02-insert-path.svg) · [src](02-insert-path.d2) | Write flow, step by step: validate → seq → **WAL append (commit point)** → apply to graph → ack, plus the HNSW insert internals and what a crash at each point means |
| [03-search-path](03-search-path.svg) · [src](03-search-path.d2) | Query flow: normalize → greedy descent → `searchLayer` on L0 → top-k → payload hydration. Includes why the path is 2 allocs/op and the rejected fetch-vectors-from-disk design |
| [04-hnsw-internals](04-hnsw-internals.svg) · [src](04-hnsw-internals.d2) | What the index actually stores — `Graph` and `node` field by field, the layered graph, level assignment, alpha-pruned neighbor selection, memory budget |
| [05-persistence-recovery](05-persistence-recovery.svg) · [src](05-persistence-recovery.d2) | Phase 2: WAL record format, segment rotation, checkpoint cycle, truncation, and the recovery path including torn-tail handling |

## The four invariants these encode

Everything above follows from four decisions. If a future change contradicts one
of them, it is the change that is wrong.

1. **Vectors are RAM-resident.** HNSW computes a distance at *every hop* of
   traversal, so `node.vector` can never be a disk read. Disk holds durability
   and cold payload only. Making traversal disk-friendly is DiskANN — a
   different index, out of scope for v1.

2. **The WAL append is the commit point.** Append (and fsync, per policy) *then*
   apply to the graph, then ack. Reverse those and a crash between them means
   acknowledging a write that no longer exists.

3. **Snapshots are written from RAM, never by re-reading the WAL.** The log is
   read exactly once in the life of a process: at startup. A snapshot at seq N
   is what makes segments below N deletable.

4. **The graph is derived state.** It can always be rebuilt from snapshot + WAL
   replay, and — because level assignment is seeded — replaying the same inserts
   in the same order reproduces an identical graph. That makes recovery testable
   by equality, not by sampling.

## Three clocks, often confused

| Clock | Cadence | What it bounds |
|---|---|---|
| WAL append | every write, synchronous | nothing — a buffered write |
| WAL **fsync** | ~50 ms (policy: always / interval / never) | how much acknowledged data a power loss can destroy |
| **Checkpoint** | seconds to minutes | recovery time and WAL size on disk |

50 ms is a reasonable *fsync* interval and a wildly wrong *checkpoint* interval.

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
