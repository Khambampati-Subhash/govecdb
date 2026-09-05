# internal/store

The metadata attached to each vector, and the encoding that puts it on disk.

## Why metadata only, and not the vectors

The obvious design is a store that owns whole records — id, values, metadata —
with the index as a derived structure over it. That would mean **two copies of
every vector in memory**, one here and one in the graph, and vectors are the
largest thing the process holds: a million 768-dimension embeddings is 3 GB per
copy. Paying that for a tidier ownership diagram is not a good trade.

So values live once, in the index, and this package holds only what the index has
no place for.

The cost of that choice is real and belongs in the open: for a normalizing metric
the index keeps the *unit* vector, so the magnitude a caller passed in is not
recoverable. A cosine index is already a statement that magnitude is not
meaningful, which is what makes the trade defensible rather than merely cheap.

## Files (one responsibility each)

| File | Responsibility |
|------|----------------|
| `store.go` | The `Store` interface and `Map`, the in-memory implementation. |
| `codec.go` | Validation, the wire format, and the limits enforced both ways. |

## The value types are a closed set

`string`, `bool`, `int64`, `float64`. Nothing else.

Decoding metadata is the one place in this database where **untrusted bytes
become live objects**. A reflection-based decoder — `gob`, or anything that
reconstructs arbitrary types from type names on the wire — turns that into a much
larger attack surface for no benefit anybody asked for. Four types cover what
filters need, and a decoder over four tags is a total function that can be read in
one sitting.

`int` is deliberately **not** among them, and that is the rule callers trip over:
`map[string]any{"n": 1}` is an `int`, not an `int64`. `Validate` says so, naming
the key, rather than widening it — the width of `int` is a platform property, and
a silent widening on one machine becomes a silent narrowing on another.

(Filter *operands* are normalized, precisely because they are never written down.
See `internal/filter`.)

## Limits are enforced on the way in and on the way out

| Cap | Value |
|---|---|
| `MaxKeys` | 256 |
| `MaxKeyBytes` | 256 |
| `MaxStringBytes` | 64 KiB |
| `MaxEncodedBytes` | 1 MiB |

`MaxEncodedBytes` is the bound that actually protects memory — the per-field caps
multiplied together are far larger.

They are checked when writing *and* when reading. A cap the writer does not
honour is a record that can be written and never read back, which is worse than
either limit alone.

## Copies, and the one method that does not make one

`Put` stores a copy and `Get` returns one. A shared map would be mutable state
escaping the lock: a caller who reuses the map they passed would be editing
stored state from outside, with no lock and no way for this package to notice. It
is the same reason the index copies vectors on insert.

**`Match` is the exception, and the reason it exists.** It evaluates a predicate
against the stored map *without* copying, because it is called once per candidate
node inside a filtered search — where `Get`'s per-call allocation would become the
dominant cost of the search rather than a rounding error. `TestMatchDoesNotAllocate`
pins it at 0 allocs/op.

The price is a narrower contract: the predicate runs under the read lock and
borrows a map it must not retain or mutate. That is a promise a predicate can
keep, unlike an arbitrary caller — which is why `Get` still copies.

An id with no metadata is passed a `nil` map rather than skipped. "Has no
metadata" is a thing a filter can legitimately ask about, and a nil map reads as
empty for every operation a predicate performs on one.

## One spelling for "no metadata"

`Put` with a nil or empty map **deletes** the entry rather than storing
emptiness. Otherwise "absent" and "present but empty" would be two spellings of
the same thing, and every caller downstream would have to handle both.

## Iteration holds the read lock

`All` iterates under `RLock`, so `fn` must not call back into the store — doing
so deadlocks. It is documented rather than defended against, because the
alternative is copying the whole map in order to iterate it.

## A failed read leaves the store as it was

`ReadFrom` builds the replacement map aside and swaps it in only on success. A
snapshot section that turns out to be corrupt half way through must not leave the
store half replaced — recovery falls back to an older snapshot, and that fallback
is only usable if the failed attempt changed nothing.
