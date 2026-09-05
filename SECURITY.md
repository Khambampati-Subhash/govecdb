# Security Policy

## Reporting a vulnerability

**Please do not open a public issue.** Use GitHub's private reporting:

**[Report a vulnerability →](https://github.com/khambampati-subhash/govecdb/security/advisories/new)**

Include a reproducer if you can — a Go test is ideal. You should get an initial
response within a week. If a fix is warranted, the advisory is published together
with the release that carries it, and you are credited unless you ask otherwise.

## Supported versions

| Version | Supported |
|---|---|
| 1.x | ✅ |
| < 1.0 | ❌ — pre-release, upgrade to 1.x |

## What this library's threat model actually is

GoVecDB is an **embeddable library**. It runs inside your process, with your
privileges, and has no network surface, no authentication and no multi-tenancy —
there is no server to attack. That makes the boundary narrow and worth stating
precisely, because it is not "nothing".

**Data crossing the boundary is data that came from somewhere you may not trust.**
Two places matter:

1. **Arguments to the API.** A library does not know where its arguments came
   from. An id, a `K`, a metadata map may all be relaying input from outside the
   process, and each one multiplies an allocation. Every one of them is bounded
   (see `WithLimits`), and the bounds are configurable but not removable.

2. **Bytes read back from disk.** This is the sharper one: decoding is where
   bytes become live objects. The metadata value set is deliberately closed to
   `string`, `bool`, `int64` and `float64` — there is no `gob`, no reflection,
   and nothing that reconstructs arbitrary types from names on the wire. Every
   decoder checks a length against both the file size and a cap **before**
   allocating on it, because a corrupt length turns a bad read into an arbitrary
   allocation. The WAL checksum covers the type, sequence and length fields as
   well as the payload, for exactly that reason.

### In scope

- Memory-safety or panic-on-malformed-input in any decode path — the WAL reader,
  the snapshot loader, the graph codec, or metadata decoding.
- Unbounded allocation driven by a value read from disk or passed to the API.
- Any way to make an acknowledged, durably-written record disappear, or to make
  recovery silently return incomplete data.
- Path traversal or unexpected file access from a caller-supplied value.

### Not in scope

- **A malicious database directory.** If an attacker can write to your data
  directory they already have your process's privileges. GoVecDB detects
  corruption because disks are unreliable, not because it is defending against
  an author of that corruption. It is not a sandbox for hostile files.
- **Resource exhaustion from your own workload.** A large `K` on a large corpus
  is slow because the work is real.
- **`SyncNever` losing data on a crash.** That is the documented trade, and it is
  documented in [`docs/DURABILITY.md`](docs/DURABILITY.md) precisely so it is a
  choice rather than a surprise.
- **Denial of service by filling the disk.**

## Dependencies

There are none. `go.mod` has no `require` block and there is no `go.sum` — the
module is pure standard library, and CI fails if that stops being true. So the
only supply chain here is the Go toolchain itself.
