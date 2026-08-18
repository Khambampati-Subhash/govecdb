# Contributing to GoVecDB

Thanks for your interest in GoVecDB.

> **Read this first:** GoVecDB is in a **ground-up v1 rebuild** on the
> `v1-restructure` branch. The previous ~45,700-line implementation has been
> removed from the working tree (it lives on in git history on `main`). Today the
> codebase is one package: `internal/hnsw`.
>
> The shape of the code is changing quickly and there is no public API yet, so
> **please open an issue before starting substantial work** — otherwise you risk
> building against something that is about to move. Small fixes and test
> improvements are always welcome without ceremony.

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

1. Branch off `v1-restructure` (not `main`).
2. Keep the PR focused — one concern per PR.
3. State what you verified: build, vet, tests, race, and benchmark deltas if the
   change touches a hot path.
4. If you knowingly left something out of scope, say so in the description.

## Reporting bugs

Include your Go version (`go version`), OS, steps to reproduce, expected vs actual
behavior, and a minimal code sample. For recall or performance issues, include the
`Config` you used and the `k`/`ef` values.

## License

Contributions are licensed under the MIT License — see [LICENSE](LICENSE).
