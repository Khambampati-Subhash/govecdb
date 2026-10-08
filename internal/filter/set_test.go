package filter

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/khambampati-subhash/govecdb/internal/store"
)

// tricky are the values where a hashed In could disagree with the scan: the
// float64 mantissa edge, the int64 range edges (including 2^63, which float64
// holds and int64 does not), signed zero, fractions next to whole numbers,
// NaN and the infinities, and the types equality must not cross.
var tricky = []any{
	int64(0), 0.0, math.Copysign(0, -1),
	int64(7), 7.0, 7.5, -7.0, int64(-7),
	int64(1 << 53), int64(1<<53 + 1), float64(1 << 53), float64(1<<53) + 2,
	int64(math.MaxInt64), int64(math.MinInt64), int64(math.MaxInt64 - 1),
	9223372036854775808.0, -9223372036854775808.0, 9223372036854774784.0,
	math.NaN(), math.Inf(1), math.Inf(-1),
	math.SmallestNonzeroFloat64, 1e300,
	"7", "", "a", true, false,
}

// The set must answer exactly what the scan answers, for every stored value
// against every operand list. The scan is the reference: it is equal itself.
func TestValueSetMatchesTheScan(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 11))
	pick := func() any {
		if r.IntN(4) == 0 {
			// Random numbers too, so the property is not only about the list.
			if r.IntN(2) == 0 {
				return int64(r.IntN(41) - 20)
			}
			return float64(r.IntN(41)-20) / float64(1+r.IntN(2))
		}
		return tricky[r.IntN(len(tricky))]
	}

	for trial := range 2000 {
		ops := make([]any, setThreshold+1+r.IntN(40))
		for i := range ops {
			ops[i] = pick()
		}
		set := newValueSet(ops)
		stored := append([]any{pick(), pick(), pick()}, tricky...)
		for _, s := range stored {
			want := false
			for _, o := range ops {
				if equal(s, o) {
					want = true
					break
				}
			}
			if got := set.contains(s); got != want {
				t.Fatalf("trial %d: stored %#v against %#v: set says %v, scan says %v",
					trial, s, ops, got, want)
			}
		}
	}
}

// Through the public constructor, at sizes either side of the threshold, so a
// change to In that bypassed the set (or used it below the threshold) with
// different answers would show.
func TestInAnswersTheSameEitherSideOfTheThreshold(t *testing.T) {
	for _, n := range []int{setThreshold, setThreshold + 1, 200} {
		ops := make([]any, n)
		for i := range ops {
			ops[i] = int64(i * 3) // 0, 3, 6, ...
		}
		f := In("k", ops...)
		for v := range 3 * n {
			md := store.Metadata{"k": float64(v)}
			if got, want := f.Match(md), v%3 == 0; got != want {
				t.Fatalf("n=%d: In matched float64 %d = %v, want %v", n, v, got, want)
			}
		}
	}
}

func TestInWithASetDoesNotAllocate(t *testing.T) {
	ops := make([]any, 100)
	for i := range ops {
		ops[i] = fmt.Sprint(i)
	}
	ops = append(ops, int64(5), 2.5)
	f := In("k", ops...)
	for _, v := range []any{"42", "nope", int64(5), 5.0, 2.5, true} {
		md := store.Metadata{"k": v}
		if n := testing.AllocsPerRun(100, func() { f.Match(md) }); n != 0 {
			t.Fatalf("Match(%#v) allocates %v times, want 0", v, n)
		}
	}
}

// BenchmarkInMiss measures the case that made a wide In a denial of service:
// a value that matches none of the operands, so the scan reads every one.
func BenchmarkInMiss(b *testing.B) {
	for _, n := range []int{16, 1000, 100000} {
		ops := make([]any, n)
		for i := range ops {
			ops[i] = int64(i)
		}
		f := In("k", ops...)
		md := store.Metadata{"k": int64(-1)}
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			for b.Loop() {
				f.Match(md)
			}
		})
	}
}
