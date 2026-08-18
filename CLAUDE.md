# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

**GoVecDB** — a high-performance, embeddable **vector database in pure Go** (no
CGO). It stores embeddings and answers "what is most similar to this?" using an
**HNSW** approximate-nearest-neighbor index. WAL persistence is the next phase,
not yet built.

Module path: `github.com/khambampati-subhash/govecdb` · Go 1.24+ (built with 1.25).
**Zero third-party dependencies** — `go.mod` has no `require` block and there is no
`go.sum`. Do not add a dependency without asking; stdlib-only is a design goal.

## Current effort: v1 rebuild (active)

Branch **`v1-restructure`**. Strategy: **rebuild from scratch, one subsystem at a
time**, using the old implementation as a reference in git history rather than as a
source to copy. Read `docs/MIGRATION.md` before making structural changes — it has
the current state, the ordered steps, and the WAL design constraints.

Scope for v1: **embeddable library only** (no cluster / REST server / gRPC — those
stay in `main` history and return in v2).

### The codebase is `internal/hnsw/` — that's all of it
The from-scratch HNSW index (see its `README.md`) is currently the entire tree, and
it is the reference for style: small single-responsibility files, comments that
explain *why*, tested against brute-force recall.

**Next phase: `internal/wal/`.** Design constraints are in `docs/MIGRATION.md`.

### The legacy code is gone
Every previous package (`index/`, `store/`, `persist/`, `api/`, `collection/`,
`filter/`, `cluster/`, `segment/`, `proto/`, and the dead experiments) was deleted
in the clean-slate commit. **Nothing is lost** — `main` has it all. To consult the
old implementation rather than resurrect it:

```bash
git show main:persist/wal.go
git log main --oneline -- persist/
```

Do not restore these packages into the tree. If something there is worth having,
rewrite it to the current bar.

## Commands

```bash
go build ./...                 # build everything
go vet ./...                   # static checks
go test ./...                  # all tests
go test ./internal/hnsw/ -v    # the new HNSW package
go test ./... -race            # race detector (run before merging)
```

If `go` is not on PATH: `export PATH=$PATH:/usr/local/go/bin`.

## Conventions

- **SOLID first.** One package = one responsibility. Depend on interfaces
  (`Index`, `Store`, `WAL`, `DistanceFunc`, ...), inject concretes. Prefer
  Factory + functional-options construction over many constructors.
- **Distances return "smaller = closer"** everywhere, so callers never branch on
  the metric.
- **Green-gated steps.** Each change keeps `go build`, `go vet`, and `go test`
  passing; land it as its own focused commit.
- **Comments explain *why*,** not the obvious *what*. Match the density in
  `internal/hnsw/`.

### Git
- Work on the `v1-restructure` branch; do not commit to `main` directly.
- **Commit messages must NOT include a `Co-Authored-By` trailer** (repo owner
  preference).
- Commit/push only when asked.

## HNSW quick reference

- `M` (neighbors/node) is **structural** — set once, changing it needs a rebuild.
- `EfConstruction` is the build-time search width (kept fixed, ~100–200).
- `ef` is the **query-time** knob in `Search(query, k, ef)`; must be `>= k`, bigger
  = higher recall + slower.
- Empty graph = empty container: no graph memory until the first insert.
- The graph is **single-threaded** today — callers serialize access. Concurrency is
  a later phase, not an oversight.
- Insert **copies** the caller's vector (and normalizes it for Cosine), so the graph
  never aliases a reused caller buffer.
- Durability model *(phase 2, not yet built)*: **write to WAL first, then apply to
  the in-memory graph**; on recovery, replay the WAL to rebuild the graph — the
  graph is derived state, never the source of truth.

## Locked baselines — do not regress

Any index change must hold these; they are enforced by tests and `-benchmem`:

| Baseline | Value | Guarded by |
|---|---|---|
| Recall@10, dim 32 | 0.999 | `TestRecallVsBruteForce` |
| Recall@10, dim 768 | 0.972 | `TestRecallHighDimension` |
| Search allocations | 2 allocs/op | `BenchmarkSearch -benchmem` |
