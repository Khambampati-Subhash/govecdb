# internal/filter

Predicates over the metadata attached to a vector, combined into a tree that a
search evaluates once per candidate node.

## Why a tree and not a `func(Metadata) bool`

The whole package could be one function type, and callers would write their own
predicates. That is smaller and strictly worse:

- A closure cannot be **examined**, so nothing can report which clause rejected
  everything — and "my filtered search returns nothing" is the question this
  package will be asked most.
- Nothing can **reorder** it, so a cheap `Eq` on a string cannot be moved ahead
  of an expensive one.
- A filter arriving from **outside the process** could not be reconstructed
  without handing a stranger the ability to run code.

An expression tree keeps all three doors open and costs one interface.

The interface stays at two methods all the same — `Match` and `Validate` —
because the optimizer and the wire format are things this package does not have
yet. `internal/wal` taught that lesson directly: `Replay` sat on an interface as
a promise until writing it showed it did not belong there.

## Files (one responsibility each)

| File | Responsibility |
|------|----------------|
| `doc.go` | Package overview and the design decisions below. |
| `filter.go` | The `Filter` interface, `And` / `Or` / `Not` / `Exists`. |
| `compare.go` | `Eq` / `Ne` / `Lt` / `Lte` / `Gt` / `Gte` / `In`. |
| `value.go` | Operand normalization, and the comparison kernel. |
| `errors.go` | Sentinel errors callers match on. |

## Absent keys are false, uniformly

**Every predicate about a key is false when that key is not there.** `Eq`, `Ne`,
`Lt`, `In` — all of them.

`Ne` is the one this surprises: `Ne("status", "draft")` does *not* match a vector
with no status at all. That is deliberate, and `Not` is how to ask the other
question:

```go
Ne("status", "draft")        // has a status, and it is not draft
Not(Eq("status", "draft"))   // has no status, OR it is not draft
```

The alternative is SQL's three-valued logic, where a comparison against a missing
key is neither true nor false and propagates through `AND` and `OR` by its own
rules. That is defensible in a language people write by hand and badly out of
place in a Go library, where `Match` returns a `bool` and every caller would have
to learn a third state to use it. One rule, and an explicit `Not`.

## Numbers compare across `int64` and `float64` — exactly

Metadata holds `int64` and `float64` as separate types, but a filter should not
make a caller track which one a value was written as: `Lt("score", 0.5)` means
the same thing either way. So the two compare numerically.

**Exactly**, and not through `float64(i) < f`. That conversion rounds once `i`
passes 2^53, and the range where it starts is where ordinary data lives rather
than a corner:

```go
time.Now().UnixNano()  // ~1.7e18, far past 2^53
```

A filter over an ingestion timestamp is precisely where the naive comparison
breaks, and it breaks by quietly including the wrong records rather than by
failing. `compareIntFloat` splits the float at its decimal point instead:
integer parts compare as `int64`, and the fraction only breaks ties.

Two boundary details that are easy to get backwards, both pinned by tests:

- The range check is written against **2^63, not `MaxInt64`**, because 2^63 is
  exactly representable in a `float64` and `MaxInt64` is not — `float64(MaxInt64)`
  rounds *up* to 2^63. Comparing against the rounded value would misjudge the
  edge.
- `math.Trunc` rounds toward zero, so negative values need no special case:
  `-2.5` truncates to `-2.0` and the negative fraction makes the integer the
  larger of the two.

## Operands are normalized; stored values are not

`Add` refuses an `int` metadata value rather than widening it, because the width
of `int` is a platform property and what one machine writes another has to read.

A **comparison operand** is never written down, so that argument does not reach
it — and the trap it would otherwise leave is real:

```go
Eq("page", 12)   // an untyped constant is an int, not the int64 metadata holds
```

Under a strict rule that matches nothing, forever, silently. So every Go integer
type is normalized to `int64` and `float32` to `float64`, both lossless. The one
exception is a `uint64` above `MaxInt64`, which is **refused rather than
wrapped**: converting it would produce a negative number, and a filter that
silently means the opposite of what it says is worse than one that will not run.

## What has no ordering

`order` refuses, and every ordering comparison on it is false:

| | Reason |
|---|---|
| `bool` against anything | `true` is not "greater than" `false` in any sense worth encouraging |
| A string against a number | Different kinds; there is no answer, only a convention |
| Either side `NaN` | IEEE's rule. The same reason the index refuses a `NaN` *vector* value outright: one comparison that answers false to everything is impossible to find afterwards |

`NaN` is reachable — `store.Validate` checks vector values for finiteness but not
metadata ones — so it is answered for rather than assumed away. Note the
consequence: `Ne("score", NaN)` is **true** against a stored `NaN`, because `Ne`
is "present and not equal" and a `NaN` equals nothing, itself included.

## Validate runs once; Match runs per node

`Match` is called once per candidate node inside a search, and a search visits
thousands. So the checking is split:

- **`Validate`** walks the tree once, before the search, and reports a bad
  operand or a nil child. The constructors return a `Filter` rather than
  `(Filter, error)` — a query built by nesting calls only reads well if the error
  is collected and reported once, by the layer that has somewhere to report it
  to. `govecdb.Search` is that layer, and surfaces it as `ErrInvalidFilter`.
- **`Match`** assumes a validated filter and does not re-check. An unvalidated
  one may panic rather than return a wrong answer, which is the right direction
  for a bug entirely inside this process.

A nil child is caught rather than tolerated: treating it as "matches nothing"
would turn one typo into an empty result set that explains nothing.

## The identities are not an accident

`And()` with no filters matches **everything**; `Or()` with none matches
**nothing**. Those are the identity elements for conjunction and disjunction, and
they are what makes a filter accumulated in a loop behave when the loop runs zero
times. `In(key)` with no values matches nothing for the same reason, which is
also the direction that fails safe: filtering by a set that turned out to be
empty returns nothing rather than everything.

## Where it is applied

Not here. This package answers "does this metadata match?" and knows nothing
about the index.

The join happens in `govecdb.Search`, which closes a `func(id string) bool` over
this filter and the metadata store and hands it to the index. The index applies
it **during the traversal**, admitting rejected nodes to the search frontier but
never to the result set — the same split `internal/hnsw` already uses for
tombstones, and for the same reason: filtering the output instead would return
fewer than `k` results rather than searching wider for `k` matching ones.

That the predicate is over **ids** rather than metadata is what keeps
`internal/hnsw` free of a second data model.

## Measured

The cost of a filter is not the predicate call; it is how much wider the search
has to travel to fill `k`. 10,000 × 128, `k=10`, `ef=64`:

| Admitted | none (unfiltered) | 1 in 2 | 1 in 10 | 1 in 50 |
|---|---|---|---|---|
| Latency | 85 µs | 165 µs | 385 µs | 965 µs |
| Allocs | 2 | 2 | 2 | 2 |

The allocation count does not move, which is the property worth defending:
`store.Map.Match` borrows the metadata map instead of copying it, so the search's
2 allocs/op baseline survives filtering. `Get` copies — it hands the map out —
and using it per candidate would have made the copy the dominant cost of a
search.

Past roughly one in a hundred, the graph stops being the right tool: a scan over
the metadata, distance-checking only what matches, beats a traversal that is
visiting most of the graph anyway.

## Not implemented yet (deliberately)

- **No clause reordering.** The tree makes it possible; nothing measures which
  clause is cheap yet, and guessing would be a pessimization with extra steps.
- **No wire format.** Encoding a filter is what would let one arrive from outside
  the process, which is a feature with a security boundary attached and no
  consumer today.
- **No `String()`.** Printing a filter is the obvious next thing to want when a
  search returns nothing, and it is one method away — but the interface is the
  contract every caller-implemented `Filter` has to satisfy, so it is not widened
  before something needs it.
