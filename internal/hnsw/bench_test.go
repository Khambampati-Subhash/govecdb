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

func BenchmarkSearch(b *testing.B) {
	g, qs := buildFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := range b.N {
		_, _ = g.Search(qs[i%len(qs)], benchK, benchEf)
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
