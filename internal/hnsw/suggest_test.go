package hnsw

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestSuggestedEfAchievesTarget is what separates SuggestedEf from a rule of
// thumb someone wrote down once. It builds real graphs, searches at exactly the
// width the function suggests, and fails if the recall target is missed.
//
// Without this the constants would rot silently: they are fitted to a
// measurement, and any change to neighbour selection, pruning or the default M
// moves the surface underneath them.
func TestSuggestedEfAchievesTarget(t *testing.T) {
	skipUnderRace(t)

	const dim = 128
	sizes := []int{1000, 5000}
	if measuring() {
		sizes = append(sizes, 20000)
	}

	// k=100 and M=32 are here because both terms were wrong before anything
	// checked them: k was assumed linear and M was absent, and the first made
	// a 100-candidate search ten times wider than it needed to be.
	for _, n := range sizes {
		for _, tc := range []struct {
			k, m   int
			target float64
		}{
			{10, 16, 0.90}, {10, 16, 0.95}, {100, 16, 0.95}, {10, 32, 0.95}, {100, 32, 0.95},
		} {
			k := tc.k
			t.Run(fmt.Sprintf("n=%d/k=%d/M=%d/target=%.2f", n, k, tc.m, tc.target), func(t *testing.T) {
				target := tc.target
				rng := rand.New(rand.NewSource(int64(1300 + n)))
				c := uniformCorpus(rng, n, dim)
				cfg := DefaultConfig(dim, Cosine)
				cfg.M = tc.m
				g, _ := buildIndex(t, cfg, c)

				ef := g.SuggestedEf(k, target)
				recall, perQuery := evaluate(t, g, c, Cosine, makeQueries(rng, 100, dim), k, ef)
				t.Logf("n=%d k=%d M=%d target=%.2f -> ef=%d, measured recall %.3f at %v", n, k, tc.m, target, ef, recall, perQuery)

				// targetRecall is a floor, and the anchors carry margin so that
				// it holds on a corpus other than the one they were measured on.
				// The small tolerance that remains is corpus-to-corpus variance,
				// not slack for a bad fit: calibrated without margin this
				// undershot on three of four cases, which is what the margin
				// exists to fix.
				if recall < target-0.02 {
					t.Fatalf("SuggestedEf(n=%d, k=%d, M=%d, %.2f) = %d reached only %.3f", n, k, tc.m, target, ef, recall)
				}
			})
		}
	}
}

// TestSuggestedEfShape covers the arithmetic without building anything, so the
// monotonicity and clamping rules stay guarded even if the calibration moves.
func TestSuggestedEfShape(t *testing.T) {
	t.Run("grows with corpus size", func(t *testing.T) {
		prev := 0
		for _, n := range []int{100, 1000, 5000, 20000, 1_000_000} {
			ef := SuggestedEf(n, 10, 0.95)
			if ef <= prev {
				t.Fatalf("n=%d gave ef=%d, not greater than the previous %d", n, ef, prev)
			}
			prev = ef
		}
	})

	t.Run("grows with the recall target", func(t *testing.T) {
		prev := 0
		for _, target := range []float64{0.80, 0.90, 0.95, 0.99} {
			ef := SuggestedEf(5000, 10, target)
			if ef < prev {
				t.Fatalf("target=%.2f gave ef=%d, below the previous %d", target, ef, prev)
			}
			prev = ef
		}
	})

	t.Run("clamps outside the measured band", func(t *testing.T) {
		// Below and above the anchors the measurements say nothing, so the
		// answer must flatten rather than extrapolate into invented precision.
		if lo, floor := SuggestedEf(5000, 10, 0.10), SuggestedEf(5000, 10, 0.80); lo != floor {
			t.Fatalf("target 0.10 gave %d, want the 0.80 floor of %d", lo, floor)
		}
		if hi, ceiling := SuggestedEf(5000, 10, 1.00), SuggestedEf(5000, 10, 0.99); hi != ceiling {
			t.Fatalf("target 1.00 gave %d, want the 0.99 ceiling of %d", hi, ceiling)
		}
	})

	t.Run("grows sub-linearly with k", func(t *testing.T) {
		// Measured, the width that holds a target barely moves from k=10 to
		// k=100; a linear term made the larger pool ten times as expensive.
		k10, k100 := SuggestedEf(62500, 10, 0.95), SuggestedEf(62500, 100, 0.95)
		if k100 <= k10 || float64(k100) > 2*float64(k10) {
			t.Fatalf("k=10 -> %d, k=100 -> %d; want growth, but under 2x", k10, k100)
		}
	})

	t.Run("falls with M", func(t *testing.T) {
		m16 := suggestEf(20000, 10, 0.95, 16, 1)
		m32 := suggestEf(20000, 10, 0.95, 32, 1)
		if m32 >= m16 || float64(m32) < 0.4*float64(m16) {
			t.Fatalf("M=16 -> %d, M=32 -> %d; want M=32 a little over half", m16, m32)
		}
	})

	t.Run("never below k", func(t *testing.T) {
		// Search clamps ef up to k regardless; a suggestion that needed that
		// rescue would be misleading to print in a log line.
		for _, k := range []int{1, 10, 100, 1000} {
			if ef := SuggestedEf(1, k, 0.80); ef < k {
				t.Fatalf("k=%d gave ef=%d", k, ef)
			}
		}
	})

	t.Run("degenerate input", func(t *testing.T) {
		if ef := SuggestedEf(0, 10, 0.95); ef != 10 {
			t.Fatalf("empty corpus gave ef=%d, want k", ef)
		}
		if ef := SuggestedEf(1000, 0, 0.95); ef < 1 {
			t.Fatalf("k=0 gave ef=%d", ef)
		}
	})
}

// TestGraphSuggestedEfUsesLiveCount pins that the method reads the live
// population rather than the slot count — a graph full of tombstones is a
// smaller search problem than its memory suggests, and suggesting a width for
// vectors nobody can retrieve would be wrong in the expensive direction.
func TestGraphSuggestedEfUsesLiveCount(t *testing.T) {
	g, _ := buildGraph(t, 2000, 16, 99)

	full := g.SuggestedEf(10, 0.95)
	for i := range 1500 {
		if !g.Delete(fmt.Sprintf("v%d", i)) {
			t.Fatalf("Delete(v%d) removed nothing", i)
		}
	}
	afterDeletes := g.SuggestedEf(10, 0.95)

	if afterDeletes >= full {
		t.Fatalf("ef suggestion did not fall with the live count: %d -> %d", full, afterDeletes)
	}
	if want := SuggestedEf(500, 10, 0.95); afterDeletes != want {
		t.Fatalf("suggestion = %d, want %d (500 live vectors)", afterDeletes, want)
	}
}
