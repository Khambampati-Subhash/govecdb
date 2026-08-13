# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

**GoVecDB** — a high-performance, embeddable **vector database in pure Go** (no
CGO). It stores embeddings and answers "what is most similar to this?" using an
**HNSW** approximate-nearest-neighbor index, with WAL persistence for durability.

Module path: `github.com/khambampati-subhash/govecdb` · Go 1.24+ (built with 1.25).

## Current effort: v1 restructure (active)

We are rebuilding into a clean, SOLID, embeddable-library-first layout on branch
**`v1-restructure`**. Strategy: **salvage & clean** — keep the proven, tested core
and drop dead / out-of-scope code. **Read these before making structural changes:**

- `docs/MIGRATION.md` — the source→destination map, drop list, and ordered steps.
- `docs/REFACTOR_PLAN.md` — the findings and phased rationale.

Scope for v1: **embeddable library only** (no cluster / REST server / gRPC — those
stay in `main` history and return in v2).

### New clean code lives here
- `internal/hnsw/` — from-scratch, readable HNSW (see its `README.md`). This is the
  reference for style: small single-responsibility files, heavily but purposefully
  commented, tested against brute-force recall.

### Legacy code still present (being migrated / dropped)
The top-level packages `index/`, `store/`, `persist/`, `filter/`, `collection/`,
`api/` are the **old** implementation. They still build and pass tests, and are the
salvage source — but new work should go into the new `internal/*` layout per
`docs/MIGRATION.md`, not extend the old packages. Packages slated for removal:
`utils`, `diskann`, `quantization`, `batch`, `streaming`, `internal` (old obs),
`accuracy`, `segment`, `cluster`, `api/rest`, `proto`, `client`, and the
`index/optimized_*`, `index/concurrent_index.go`, `index/multi_index.go`,
`collection/enhanced_collection.go` duplicates.

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
- Durability model: **write to WAL first, then apply to the in-memory graph**; on
  recovery, replay the WAL to rebuild the graph (the graph is derived state).
