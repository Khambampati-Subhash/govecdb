package hnsw

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// exactFiltered is the brute-force truth: the k live vectors nearest q that
// allow accepts, by stored vector.
func exactFiltered(g *Graph, q []float32, k int, allow func(string) bool) []string {
	g.mu.RLock()
	defer g.mu.RUnlock()
	var all []candidate
	for i, n := range g.nodes {
		if !n.deleted && allow(n.id) {
			all = append(all, candidate{i, g.dist(n.vector, q)})
		}
	}
	slices.SortFunc(all, func(a, b candidate) int {
		switch {
		case a.dist < b.dist:
			return -1
		case a.dist > b.dist:
			return 1
		}
		return 0
	})
	out := make([]string, 0, k)
	for _, c := range all[:min(k, len(all))] {
		out = append(out, g.nodes[c.idx].id)
	}
	return out
}

// everyNth accepts ids v0, vN, v2N, ...
func everyNth(n, every int) func(string) bool {
	set := make(map[string]bool, n/every+1)
	for i := 0; i < n; i += every {
		set[fmt.Sprintf("v%d", i)] = true
	}
	return func(id string) bool { return set[id] }
}

// TestSelectiveFilterIsScannedExactly: a filter matching far fewer nodes than
// the walk would have to visit to find them is answered by a scan, and a scan
// is exact — every result, in order, is the brute-force answer. The walk at
// this selectivity visited the whole graph and could still miss.
func TestSelectiveFilterIsScannedExactly(t *testing.T) {
	const n, dim, k = 6000, 16, 10
	g := buildFilterGraph(t, n, dim)
	allow := everyNth(n, 300) // 20 matches

	g.mu.RLock()
	scan := g.shouldScan(allow, 64)
	g.mu.RUnlock()
	if !scan {
		t.Fatal("one in 300 of 6,000 at ef=64 was not scanned")
	}

	rng := rand.New(rand.NewSource(3))
	for range 20 {
		q := randomVector(rng, dim)
		Normalize(q)
		res, err := g.SearchFilter(q, k, 64, allow)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]string, len(res))
		for i, r := range res {
			got[i] = r.ID
		}
		if want := exactFiltered(g, q, k, allow); !slices.Equal(got, want) {
			t.Fatalf("scanned search returned %v, exact answer is %v", got, want)
		}
	}
}

// TestUnselectiveFilterWalksAfterFewProbes: a filter matching half the graph
// keeps the walk, and the probe that decided so stopped early — it is paid on
// every ordinary filtered search, so it must stay a few dozen calls.
func TestUnselectiveFilterWalksAfterFewProbes(t *testing.T) {
	const n = 6000
	g := buildFilterGraph(t, n, 16)
	half := everyNth(n, 2)
	calls := 0
	counting := func(id string) bool { calls++; return half(id) }

	g.mu.RLock()
	scan := g.shouldScan(counting, 64)
	g.mu.RUnlock()
	if scan {
		t.Fatal("a filter matching half the graph was scanned")
	}
	// 2·sampleMatches matches at rate 1/2 is about 64 probes.
	if calls > 4*2*sampleMatches {
		t.Fatalf("the probe made %d predicate calls to recognize a one-in-two filter", calls)
	}
}

// TestProbeSeesAContiguousRun: a filter matching one block of consecutive
// slots — one tenant's bulk import — must be estimated by its size, not by
// where it happens to sit. A probe that walked slots from zero would see all
// of it or none of it.
func TestProbeSeesAContiguousRun(t *testing.T) {
	const n = 8000
	g := buildFilterGraph(t, n, 16)
	block := func(lo, hi int) func(string) bool {
		set := make(map[string]bool, hi-lo)
		for i := lo; i < hi; i++ {
			set[fmt.Sprintf("v%d", i)] = true
		}
		return func(id string) bool { return set[id] }
	}
	g.mu.RLock()
	defer g.mu.RUnlock()
	// 40 of 8,000 in one run, at the front and at the back: scan both.
	for _, b := range [][2]int{{0, 40}, {n - 40, n}} {
		if !g.shouldScan(block(b[0], b[1]), 64) {
			t.Errorf("block %v was walked; it matches 40 of %d", b, n)
		}
	}
	// Half the graph in one run: walk.
	if g.shouldScan(block(0, n/2), 64) {
		t.Error("a block of half the graph was scanned")
	}
}

// TestScanSkipsTombstonesAndFillsShort: the scan returns live nodes only, and
// when fewer than k match it returns all of them rather than padding.
func TestScanSkipsTombstonesAndFillsShort(t *testing.T) {
	const n, dim = 3000, 16
	g := buildFilterGraph(t, n, dim)
	allow := everyNth(n, 500) // v0, v500, ..., v2500
	g.Delete("v500")
	g.Delete("v1500")

	rng := rand.New(rand.NewSource(4))
	q := randomVector(rng, dim)
	Normalize(q)
	res, err := g.SearchFilter(q, 10, 64, allow)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]string, len(res))
	for i, r := range res {
		got[i] = r.ID
	}
	if want := exactFiltered(g, q, 10, allow); len(got) != 4 || !slices.Equal(got, want) {
		t.Fatalf("got %v, want the 4 live matches %v", got, want)
	}
}

// TestProbeOnAMostlyDeadGraph: the probe skips tombstones, so on a graph that
// is mostly dead it may run out of tries before its budget — and must still
// judge from the live nodes it did see.
func TestProbeOnAMostlyDeadGraph(t *testing.T) {
	const n = 8000
	g := buildFilterGraph(t, n, 16)
	for i := range n {
		if i%20 != 0 {
			g.Delete(fmt.Sprintf("v%d", i))
		}
	}
	every20 := everyNth(n, 20) // every live node
	g.mu.RLock()
	defer g.mu.RUnlock()
	if g.shouldScan(every20, 4) {
		t.Error("a filter matching every live node was scanned")
	}
}
