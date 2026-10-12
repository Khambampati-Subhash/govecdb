//go:build amd64 && !purego

package hnsw

import (
	"math/rand"
	"testing"
)

// TestAMD64DispatchBothWays holds each side of the CPU-feature branch to the
// Go kernels: the assembly where this CPU has AVX2 and FMA, and the fallback
// by forcing the flag off, which is the path a CPU without them takes and
// otherwise nothing here would run. The lengths straddle every loop boundary
// at the dimensions embeddings actually come in, where TestKernelMatchesGeneric
// stops at 200.
func TestAMD64DispatchBothWays(t *testing.T) {
	t.Logf("AVX2+FMA kernels in use: %v", useAVX2)
	rng := rand.New(rand.NewSource(9))
	buf := signedVector(rng, 1600)
	other := signedVector(rng, 1600)
	dims := []int{0, 1, 7, 8, 9, 63, 64, 65, 71, 72, 73, 127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025, 1536}

	check := func(t *testing.T, exact bool) {
		for _, dim := range dims {
			for off := range 3 {
				a, b := buf[off:off+dim], other[2-off:2-off+dim]
				gd, wd := dot(a, b), dotGeneric(a, b)
				ge, we := squaredEuclidean(a, b), squaredEuclideanGeneric(a, b)
				if exact {
					if gd != wd || ge != we {
						t.Fatalf("dim %d off %d: fallback is not the Go kernel: dot %v vs %v, l2 %v vs %v", dim, off, gd, wd, ge, we)
					}
					continue
				}
				if !closeEnough(gd, float64(wd)) {
					t.Fatalf("dim %d off %d: dot = %v, generic %v", dim, off, gd, wd)
				}
				if !closeEnough(ge, float64(we)) {
					t.Fatalf("dim %d off %d: squaredEuclidean = %v, generic %v", dim, off, ge, we)
				}
			}
		}
	}

	t.Run("detected", func(t *testing.T) { check(t, !useAVX2) })
	t.Run("fallback", func(t *testing.T) {
		saved := useAVX2
		useAVX2 = false
		defer func() { useAVX2 = saved }()
		check(t, true)
	})
}
