package hnsw

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// The heaps in pq.go are hand-written to avoid container/heap's interface
// boxing, which means the sift-up and sift-down loops are ours to get right.
// They had no direct test: every previous check of them was indirect, through
// recall numbers that a subtly wrong ordering would dent rather than break.

// heapInvariant verifies the shape rather than the output — a heap can drain in
// the right order once and still be malformed, which only shows up under a
// different interleaving of pushes and pops.
func heapInvariant(t *testing.T, h []candidate, isMin bool) {
	t.Helper()
	for i := 1; i < len(h); i++ {
		parent := (i - 1) / 2
		if isMin && h[parent].dist > h[i].dist {
			t.Fatalf("min-heap violated at %d: parent %v > child %v", i, h[parent].dist, h[i].dist)
		}
		if !isMin && h[parent].dist < h[i].dist {
			t.Fatalf("max-heap violated at %d: parent %v < child %v", i, h[parent].dist, h[i].dist)
		}
	}
}

func TestMinHeapDrainsInAscendingOrder(t *testing.T) {
	rng := rand.New(rand.NewSource(11))

	for _, n := range []int{1, 2, 3, 7, 8, 9, 64, 257} {
		var h []candidate
		want := make([]float32, 0, n)
		for i := range n {
			d := rng.Float32() * 100
			h = minPush(h, candidate{idx: i, dist: d})
			want = append(want, d)
			heapInvariant(t, h, true)
		}
		slices.Sort(want)

		for i := range n {
			var c candidate
			c, h = minPop(h)
			if c.dist != want[i] {
				t.Fatalf("n=%d pop %d: got %v, want %v", n, i, c.dist, want[i])
			}
			heapInvariant(t, h, true)
		}
		if len(h) != 0 {
			t.Fatalf("n=%d: heap has %d left after draining", n, len(h))
		}
	}
}

func TestMaxHeapDrainsInDescendingOrder(t *testing.T) {
	rng := rand.New(rand.NewSource(12))

	for _, n := range []int{1, 2, 3, 7, 8, 9, 64, 257} {
		var h []candidate
		want := make([]float32, 0, n)
		for i := range n {
			d := rng.Float32() * 100
			h = maxPush(h, candidate{idx: i, dist: d})
			want = append(want, d)
			heapInvariant(t, h, false)
		}
		slices.Sort(want)
		slices.Reverse(want)

		for i := range n {
			var c candidate
			c, h = maxPop(h)
			if c.dist != want[i] {
				t.Fatalf("n=%d pop %d: got %v, want %v", n, i, c.dist, want[i])
			}
			heapInvariant(t, h, false)
		}
	}
}

// TestHeapsUnderInterleavedOps is the case a drain-only test cannot reach:
// searchLayer pushes and pops in an unpredictable order, and a sift bug that
// only bites when a pop is followed by a push would survive the tests above.
func TestHeapsUnderInterleavedOps(t *testing.T) {
	rng := rand.New(rand.NewSource(13))

	for range 200 {
		// Both heaps receive identical pushes, but a pop step takes the smallest
		// from one and the largest from the other — so their contents diverge
		// immediately and each needs its own reference multiset.
		var (
			minH, maxH       []candidate
			minLive, maxLive []float32
		)
		for step := range 300 {
			// Bias towards pushes so the heaps actually grow.
			if len(minLive) == 0 || rng.Intn(100) < 60 {
				d := rng.Float32() * 100
				minH = minPush(minH, candidate{idx: step, dist: d})
				maxH = maxPush(maxH, candidate{idx: step, dist: d})
				minLive = append(minLive, d)
				maxLive = append(maxLive, d)
			} else {
				slices.Sort(minLive)
				var lo candidate
				lo, minH = minPop(minH)
				if lo.dist != minLive[0] {
					t.Fatalf("minPop gave %v, want the smallest %v", lo.dist, minLive[0])
				}
				minLive = minLive[1:]

				slices.Sort(maxLive)
				var hi candidate
				hi, maxH = maxPop(maxH)
				if hi.dist != maxLive[len(maxLive)-1] {
					t.Fatalf("maxPop gave %v, want the largest %v", hi.dist, maxLive[len(maxLive)-1])
				}
				maxLive = maxLive[:len(maxLive)-1]
			}
			heapInvariant(t, minH, true)
			heapInvariant(t, maxH, false)
		}
	}
}

// TestHeapsHandleDuplicateDistances covers the shape real data produces:
// duplicated vectors, or a corpus small enough that ties are common. The sift
// comparisons use <= / >= for exactly this reason, and a strict comparison there
// would loop or mis-order.
func TestHeapsHandleDuplicateDistances(t *testing.T) {
	var minH, maxH []candidate
	for i := range 50 {
		// Three distinct values across fifty entries: ties everywhere.
		d := float32(i % 3)
		minH = minPush(minH, candidate{idx: i, dist: d})
		maxH = maxPush(maxH, candidate{idx: i, dist: d})
	}
	heapInvariant(t, minH, true)
	heapInvariant(t, maxH, false)

	var last float32 = -1
	for range 50 {
		var c candidate
		c, minH = minPop(minH)
		if c.dist < last {
			t.Fatalf("min-heap went backwards: %v after %v", c.dist, last)
		}
		last = c.dist
	}

	last = math.MaxFloat32
	for range 50 {
		var c candidate
		c, maxH = maxPop(maxH)
		if c.dist > last {
			t.Fatalf("max-heap went backwards: %v after %v", c.dist, last)
		}
		last = c.dist
	}
}

// TestHeapsCarryTheirPayload guards the half of a candidate that ordering tests
// ignore. The heaps sort on dist, but what searchLayer actually needs back is
// idx — a swap that moved distances without their indexes would rank perfectly
// and return the wrong nodes.
func TestHeapsCarryTheirPayload(t *testing.T) {
	rng := rand.New(rand.NewSource(14))

	const n = 200
	distOf := make(map[int]float32, n)
	var minH []candidate
	for i := range n {
		d := rng.Float32() * 100
		distOf[i] = d
		minH = minPush(minH, candidate{idx: i, dist: d})
	}

	for range n {
		var c candidate
		c, minH = minPop(minH)
		if want, ok := distOf[c.idx]; !ok || want != c.dist {
			t.Fatalf("candidate %d came back with distance %v, want %v", c.idx, c.dist, want)
		}
	}
}

// TestHeapReusesBackingArray pins the property state.go depends on: the heaps
// hand back a slice sharing the caller's array, so a pooled searchState keeps
// its capacity across traversals instead of reallocating.
func TestHeapReusesBackingArray(t *testing.T) {
	h := make([]candidate, 0, 64)
	original := h[:cap(h)]

	for i := range 64 {
		h = minPush(h, candidate{idx: i, dist: float32(64 - i)})
	}
	if cap(h) != cap(original) {
		t.Fatalf("heap reallocated within its capacity: cap %d -> %d", cap(original), cap(h))
	}
	if &h[0] != &original[0] {
		t.Fatal("heap moved to a different backing array despite having capacity")
	}

	// Draining must not shrink capacity either — the next traversal inherits it.
	for range 64 {
		_, h = minPop(h)
	}
	if cap(h) != cap(original) {
		t.Fatalf("draining lost capacity: %d, want %d", cap(h), cap(original))
	}
}
