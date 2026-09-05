# Contributing to GoVecDB

Thanks for your interest in GoVecDB.

> **Read this first:** GoVecDB is **released at v1.x**, which is a commitment:
> the public API is additive from here. A breaking change would force the module
> path to `.../govecdb/v2` and every user to edit their imports, so proposals
> that change an existing signature need a strong case and an issue first.
>
> `internal/` is *not* part of that promise — it cannot be imported from outside
> the module, and it is where the design still has room to move.
>
> Small fixes and test improvements are welcome without ceremony. For anything
> substantial, **open an issue first** — and check
> [the v2 scope](docs/MIGRATION.md#v2-scope), because collections, a server,
> clustering, quantization, online compaction and an observability seam are
> already planned in dependency order.

## Development setup

```bash
git clone https://github.com/khambampati-subhash/govecdb.git
cd govecdb
```

Requires **Go 1.24+**. There is nothing to install — the module has zero
third-party dependencies, no `require` block, and no `go.sum`.

```bash
go build ./...
go vet ./...
go test ./... -race
```

Benchmarks:

```bash
go test ./internal/hnsw/ -run='^$' -bench=. -benchmem
```

If `go` is not on your PATH: `export PATH=$PATH:/usr/local/go/bin`.

## Standards

**Every change must keep `go build`, `go vet`, and `go test` green.** Land work as
its own focused commit; a commit that leaves the tree broken will be sent back.

- **SOLID first.** One package = one responsibility. Depend on interfaces
  (`Index`, `Store`, `WAL`, `DistanceFunc`), inject concretes. Prefer factory +
  functional options over a pile of constructors.
- **Distances return "smaller = closer"** everywhere, so callers never branch on
  the metric.
- **Comments explain *why*, not the obvious *what*.** Match the density and tone of
  `internal/hnsw/` — that package is the style reference.
- **No new third-party dependencies** without discussing it in an issue first.
  Pure-stdlib, no-CGO is a design goal, not an accident.
- **Performance claims need numbers.** "Faster" means a `-benchmem` before/after in
  the PR description, not an assertion.

### Locked baselines

Index changes must not regress these — they are enforced by tests:

| Baseline | Value | Guarded by |
|---|---|---|
| Recall@10, dim 32 | 0.999 | `TestRecallVsBruteForce` |
| Recall@10, dim 768 | 0.972 | `TestRecallHighDimension` |
| Search allocations | 2 allocs/op | `BenchmarkSearch -benchmem` |
| Filtered search allocations | 2 allocs/op | `BenchmarkSearchFilter -benchmem` |
| Metadata predicate allocations | 0 allocs/op | `TestMatchDoesNotAllocate` |
| Recall spread across seeds | ≤ 0.05 | `TestRecallIsStableAcrossSeeds` |

**On-disk formats are frozen**, and the `TestLayoutIsFrozen` tests exist to make
that loud: changing a WAL record header, the snapshot framing, the graph codec or
the metadata encoding is a migration, not an edit.

## Tests

New behavior needs a test. For the index specifically, correctness means **recall
measured against brute-force ground truth**, not a hand-picked example that happens
to pass — see `graph_test.go` for the pattern.

Run the race detector before opening a PR: `go test ./... -race`.

## Commit messages

Conventional-commit style, scoped to the package:

```
perf(hnsw): normalize vectors, alpha-pruned selection, zero-alloc search
feat(wal): append-only record log with per-record checksums
```

Do **not** add `Co-Authored-By` trailers.

## Pull requests

1. Branch off `main`.
2. Keep the PR focused — one concern per PR.
3. State what you verified: build, vet, tests, race, and benchmark deltas if the
   change touches a hot path.
4. If you knowingly left something out of scope, say so in the description.

CI runs build, vet, tests and the race detector on Go 1.24 and 1.25 across Linux
and macOS, plus `gofmt`, `staticcheck`, CodeQL, and a job that extracts the
README's quick-start program and runs it. All of it is reproducible locally with
the commands above — nothing in CI is a tool you cannot run yourself.

## Reporting bugs

Use the [issue templates](https://github.com/khambampati-subhash/govecdb/issues/new/choose).
Include your Go version (`go version`), OS, and a reproducer — a Go test is ideal.
The interesting bugs here are about ordering, recall and crash timing, and those
are hard to guess at from prose. For recall or performance issues, include the
options you opened with and your `K`/`Ef`.

For security issues, see [SECURITY.md](SECURITY.md) — please do not open a public
issue.

## License

Contributions are licensed under the MIT License — see [LICENSE](LICENSE).
