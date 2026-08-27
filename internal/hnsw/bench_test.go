package hnsw

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

const (
	benchN   = 10000
	benchDim = 128
	benchK   = 10
	benchEf  = 64
)

var (
	fixtureOnce sync.Once
	fixtureG    *Graph
	fixtureQ    [][]float32
)

// buildFixture builds one shared graph for the search benchmarks.
func buildFixture() (*Graph, [][]float32) {
	fixtureOnce.Do(func() {
		rng := rand.New(rand.NewSource(11))
		g, _ := New(DefaultConfig(benchDim, Cosine))
		for i := range benchN {
			_ = g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, benchDim))
		}
		qs := make([][]float32, 1000)
		for i := range qs {
			qs[i] = randomVector(rng, benchDim)
		}
		fixtureG, fixtureQ = g, qs
	})
	return fixtureG, fixtureQ
}

func BenchmarkInsert(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	vecs := make([][]float32, b.N)
	for i := range vecs {
		vecs[i] = randomVector(rng, benchDim)
	}
	g, _ := New(DefaultConfig(benchDim, Cosine))
	b.ReportAllocs()
	b.ResetTimer()
	for i := range b.N {
		_ = g.Insert(fmt.Sprintf("v%d", i), vecs[i])
	}
}

// BenchmarkUpsert measures replacing an id that is already in the graph: a
// tombstone plus a full insert. The graph accumulates one dead slot per
// iteration, which is not a benchmark artifact — it is the cost being reported,
// and the reason compaction is the next step rather than a later one.
func BenchmarkUpsert(b *testing.B) {
	const n = 2000
	rng := rand.New(rand.NewSource(3))
	g, _ := New(DefaultConfig(benchDim, Cosine))
	for i := range n {
		_ = g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, benchDim))
	}
	vecs := make([][]float32, b.N)
	for i := range vecs {
		vecs[i] = randomVector(rng, benchDim)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := range b.N {
		_ = g.Insert(fmt.Sprintf("v%d", i%n), vecs[i])
	}
}

// BenchmarkUpsertUnchanged is the WAL-replay path: a record the graph already
// holds. It has to cost one comparison rather than one insert, or recovery
// across a snapshot boundary would pay full price for changing nothing.
func BenchmarkUpsertUnchanged(b *testing.B) {
	const n = 2000
	rng := rand.New(rand.NewSource(3))
	g, _ := New(DefaultConfig(benchDim, Cosine))
	vecs := make([][]float32, n)
	for i := range n {
		vecs[i] = randomVector(rng, benchDim)
		_ = g.Insert(fmt.Sprintf("v%d", i), vecs[i])
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := range b.N {
		_ = g.Insert(fmt.Sprintf("v%d", i%n), vecs[i%n])
	}
}

func BenchmarkSearch(b *testing.B) {
	g, qs := buildFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := range b.N {
		_, _ = g.Search(qs[i%len(qs)], benchK, benchEf)
	}
}

// BenchmarkSearchByDimension is the cost curve that matters for choosing an
// embedding model: every hop of a traversal computes a distance, so dimension
// multiplies the whole search, not just the final comparison.
//
// The corpus is smaller than benchN so the 1536-dim fixture is affordable to
// build; the shape across dimensions is the point, not the absolute numbers.
func BenchmarkSearchByDimension(b *testing.B) {
	const n = 2000
	for _, dim := range []int{32, 128, 384, 768, 1536} {
		b.Run(fmt.Sprintf("dim=%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(int64(dim)))
			g, _ := New(DefaultConfig(dim, Cosine))
			for i := range n {
				_ = g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, dim))
			}
			qs := make([][]float32, 200)
			for i := range qs {
				qs[i] = randomVector(rng, dim)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := range b.N {
				_, _ = g.Search(qs[i%len(qs)], benchK, benchEf)
			}
		})
	}
}

// BenchmarkSearchByScale exercises the property the whole index exists for: a
// 40x corpus should not cost 40x the search. Fixtures are built outside the
// timer, which is why the largest size stays at 20k.
func BenchmarkSearchByScale(b *testing.B) {
	for _, n := range []int{1000, 5000, 20000} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			rng := rand.New(rand.NewSource(17))
			g, _ := New(DefaultConfig(benchDim, Cosine))
			for i := range n {
				_ = g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, benchDim))
			}
			qs := make([][]float32, 200)
			for i := range qs {
				qs[i] = randomVector(rng, benchDim)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := range b.N {
				_, _ = g.Search(qs[i%len(qs)], benchK, benchEf)
			}
		})
	}
}

// BenchmarkSearchTombstones measures the price of deferred deletion: dead slots
// stay on the frontier and keep results under-filled, which loosens the pruning
// bound and widens the search. This is the curve a compaction threshold should
// be set against, so it is worth having a number rather than an assertion.
func BenchmarkSearchTombstones(b *testing.B) {
	for _, pct := range []int{0, 25, 50, 75} {
		b.Run(fmt.Sprintf("dead=%d%%", pct), func(b *testing.B) {
			rng := rand.New(rand.NewSource(11))
			g, _ := New(DefaultConfig(benchDim, Cosine))
			for i := range benchN {
				_ = g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, benchDim))
			}
			for i := range benchN {
				if i%100 < pct {
					g.Delete(fmt.Sprintf("v%d", i))
				}
			}
			qs := make([][]float32, 200)
			for i := range qs {
				qs[i] = randomVector(rng, benchDim)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := range b.N {
				_, _ = g.Search(qs[i%len(qs)], benchK, benchEf)
			}
		})
	}
}

// BenchmarkCompact measures the stop-the-world pause. A compaction is a full
// index build over the *surviving* vectors, so its cost tracks how many live —
// reclaiming more is cheaper, not dearer.
//
// The fixture is rebuilt outside the timer on every iteration because a
// compacted graph is clean, and the second call would return immediately and
// report a pause of nothing.
func BenchmarkCompact(b *testing.B) {
	const n = 5000
	for _, pct := range []int{25, 50, 75} {
		b.Run(fmt.Sprintf("dead=%d%%", pct), func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				b.StopTimer()
				rng := rand.New(rand.NewSource(11))
				g, _ := New(DefaultConfig(benchDim, Cosine))
				for i := range n {
					_ = g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, benchDim))
				}
				for i := range n {
					if i%100 < pct {
						g.Delete(fmt.Sprintf("v%d", i))
					}
				}
				b.StartTimer()

				g.Compact()
			}
		})
	}
}

func BenchmarkDistance(b *testing.B) {
	rng := rand.New(rand.NewSource(5))
	a := randomVector(rng, benchDim)
	c := randomVector(rng, benchDim)
	for _, tc := range []struct {
		name string
		fn   DistanceFunc
	}{
		{"CosineGeneral", Cosine.Func()},
		// What the graph actually runs: vectors are unit length on insert, so
		// cosine collapses to a single dot product.
		{"CosineNormalized", Cosine.fastFunc(true)},
		{"Euclidean", Euclidean.Func()},
		{"DotProduct", DotProduct.Func()},
	} {
		b.Run(tc.name, func(b *testing.B) {
			var sink float32
			for range b.N {
				sink = tc.fn(a, c)
			}
			_ = sink
		})
	}
}
