# service

Many databases in one directory: collections, created and dropped at runtime,
loaded when something asks for one.

This is the layer a server needs and a library does not, which is why it sits
*above* `govecdb.DB` rather than inside it. Nothing here changes how a single
database behaves — `DB` does not know this package exists.

## Why this exists before the HTTP API

A server with exactly one index is not a useful server, and the resource model of
the API *is* the collection lifecycle. So the lifecycle is settled first, in a
package that can be tested without a socket.

## Files (one responsibility each)

| File | Responsibility |
|------|----------------|
| `doc.go` | Package overview and this file map. |
| `name.go` | `ValidateName` — the one place a caller's string becomes a directory. |
| `spec.go` | What a collection is, and how it is written down and read back. |
| `manager.go` | Lifecycle: create, borrow, drop, evict, close. |
| `errors.go` | Sentinel errors callers match on. |

## The three things that needed thought

Everything else is bookkeeping.

### Lifecycle: the spec file

A collection's dimension, metric and `M` are **structural** — reopening a
database under different ones is refused — so they cannot live in a command-line
flag that somebody edits between restarts. Each collection writes them to
`collection.json` when it is created, and every later open reads them back.

```
root/
  docs/
    collection.json     ← dimension, metric, M, and the rest
    data/               ← the database: wal/ and snapshots/
```

Two details that are not decoration:

- **Effective values are recorded, never the zero that meant "default".**
  `Spec.Defaults()` runs before anything is written, so a change to *this
  module's* defaults cannot silently rebuild an existing collection's index under
  a different `M`. It is the same rule the HNSW header follows: a stored config is
  the effective one.
- **It is written atomically** — temp, fsync, rename, fsync the directory. The
  file is small and written once, but it is also the only record of how to reopen
  the index sitting next to it. A half-written spec after a crash is a collection
  with its data intact and unreachable.

`collection.json` is also the **marker** that makes a directory a collection.
`List` looks for it rather than reporting every subdirectory, so an operator's
stray tarball is not announced as a database.

### Loading: the placeholder

Opening a collection loads a snapshot and replays a log, which takes as long as
it takes. Doing that under the manager's lock would stall every *other*
collection behind it.

So an opening collection is registered as a placeholder, the lock is released for
the duration, and concurrent callers wait on that one open. This is not only an
efficiency question: `govecdb` refuses a second writer on one directory, so two
goroutines opening the same collection would surface `ErrAlreadyOpen` on an
ordinary request. `TestConcurrentUseOpensOnce` is the guard.

### Eviction: reference counting

An idle collection should stop costing memory; a collection being used must not
be closed underneath a search in flight. Both fall out of one mechanism.

`Use(name, fn)` borrows a database for the length of a callback. It is a callback
rather than an `Acquire`/`Release` pair **because a borrow that can be leaked by
an early return is a collection that is never evicted again**. Only a collection
with no borrowers is a candidate for eviction or for `Drop`.

- `MaxOpen` closes the least recently used *idle* collection to make room. If
  every loaded collection is busy it returns `ErrTooManyOpen` rather than
  queueing — waiting would turn a capacity problem into a latency problem that
  surfaces as a timeout somewhere else, and a request handler has a better answer
  available: say so now.
- `IdleTimeout` closes what nothing has touched. Reopening costs a snapshot load
  and a replay, so it wants to be long relative to how bursty the traffic is —
  minutes, not seconds.
- `Drop` **waits** for in-flight borrows instead of refusing. A search takes
  microseconds and a drop is a deliberate administrative act; failing it because a
  request happened to overlap is a race the operator has no way to win. New
  borrowers are refused from the moment the drop is claimed, which is what lets
  the wait terminate.

## Names are a security boundary

A collection name becomes a directory name. Vector ids never do — that is why
`validate.go` in the root package says there is no path traversal to defend
against there, and why there is one here.

`ValidateName` allows ASCII letters, digits, `-` and `_`, first character
alphanumeric, at most 64 bytes. What each rule stops is in its doc comment; the
two least obvious:

- **No `.` at all**, so `.` and `..` are refused by construction rather than by a
  special case somebody could later delete.
- **ASCII only**, because macOS normalizes filenames to NFD and Linux does not. A
  name containing `é` would not compare equal to itself after the volume moved —
  a database that appears to lose a collection.

The payoff reaches past the filesystem: names this narrow need no escaping in a
URL path, in a Prometheus label, or in a log line, so nothing downstream has to
re-derive the rules and get them subtly wrong.

## What is deliberately not here

- **Cross-process locking.** Unchanged from the root package: one `Manager` per
  root directory, enforced by `govecdb`'s own in-process check when a collection
  is opened. A lock file left behind by a crash blocks a restart that should have
  succeeded.
- **A snapshot on shutdown.** `Close` does not take one, for the same reason
  `DB.Close` does not: a shutdown that takes seconds per collection and fails on a
  full disk is the shutdown an operator cannot afford.
- **Per-collection auth or quotas.** Those belong to whatever is in front of this;
  see `httpapi`.
