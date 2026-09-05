<!--
Thanks for contributing. Nothing here is ceremony — each box maps to something
that has actually gone wrong in this codebase before.
-->

## What this changes

<!-- One or two sentences. What is different afterwards? -->

## Why

<!--
The important half. This codebase's comments explain *why* rather than *what*,
and so should its pull requests: what did you consider and reject?
-->

## Checklist

- [ ] `go build ./... && go vet ./... && go test ./...` passes
- [ ] `go test ./... -race` passes — the gate before merging
- [ ] `gofmt -l .` is empty
- [ ] No new third-party dependency (`go.mod` still has no `require` block)

If this touches any of the following, please confirm:

- [ ] **Durability guarantee or a benchmark number** → `docs/DURABILITY.md` updated.
      It is the document a user would be misled by if it went stale.
- [ ] **An on-disk format** (WAL record, snapshot framing, graph codec, metadata
      encoding) → this is a migration, not an edit. The `TestLayoutIsFrozen`
      tests exist to make that decision loud.
- [ ] **A locked baseline** (recall, allocations per search) → renegotiated
      deliberately, with new numbers, rather than allowed to drift.
- [ ] **The public API** → we are on v1.x, so it is additive. A breaking change
      would force the module path to `/v2` and every user to edit their imports.
