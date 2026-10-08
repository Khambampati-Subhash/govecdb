package hnsw

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// searchLayerOnePass is searchLayer as it was before the gather/touch/score
// split: each neighbor is visited and scored in a single pass. It is kept
// here as the reference the split is held to, because the claim that makes
// the split safe — the nodes scored and their order are unchanged — is exactly
// the kind that an edit to either loop could quietly break.
func (g *Graph) searchLayerOnePass(st *searchState, query []float32, entryPoint, ef, lc int, allow func(id string) bool) []candidate {
	st.visited.reset(len(g.nodes))
	if st.hide > 0 {
		st.visited.visit(st.hide - 1)
	}
	cands, results := st.cands[:0], st.results[:0]
	d := g.dist(g.nodes[entryPoint].vector, query)
	st.visited.visit(entryPoint)
	cands = append(cands, candidate{entryPoint, d})
	if g.admits(entryPoint, allow) {
		results = append(results, candidate{entryPoint, d})
	}
	var c candidate
	for len(cands) > 0 {
		c, cands = minPop(cands)
		if len(results) >= ef && c.dist > results[0].dist {
			break
		}
		for _, nb := range g.readNeighbors(st, c.idx, lc) {
			if st.visited.visit(nb) {
				continue
			}
			nd := g.dist(g.nodes[nb].vector, query)
			if len(results) < ef || nd < results[0].dist {
				cands = minPush(cands, candidate{nb, nd})
				if g.admits(nb, allow) {
					results = maxPush(results, candidate{nb, nd})
					if len(results) > ef {
						_, results = maxPop(results)
					}
				}
			}
		}
	}
	out := slices.Grow(st.found[:0], len(results))[:len(results)]
	for i := len(out) - 1; i >= 0; i-- {
		out[i], results = maxPop(results)
	}
	st.found, st.cands, st.results = out, cands, results
	return out
}

// TestSearchLayerMatchesOnePass holds searchLayer to the one-pass loop, entry
// for entry and bit for bit, on every layer, with tombstones, with a filter
// and with a hidden slot — each of which changes what the frontier and the
// result set admit, and so each a way for the two loops to diverge.
func TestSearchLayerMatchesOnePass(t *testing.T) {
	const dim, n = 48, 3000
	rng := rand.New(rand.NewSource(5))
	g, err := New(DefaultConfig(dim, Cosine))
	if err != nil {
		t.Fatal(err)
	}
	for i := range n {
		if err := g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, dim)); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < n; i += 5 {
		g.Delete(fmt.Sprintf("v%d", i))
	}
	odd := func(id string) bool { return id[len(id)-1]%2 == 1 }

	st, ref := g.acquireState(), g.acquireState()
	defer g.releaseState(st)
	defer g.releaseState(ref)
	for q := range 60 {
		for _, ef := range []int{1, 10, 40} {
			query := randomVector(rng, dim)
			Normalize(query)
			for lc := 0; lc <= g.maxLevel; lc++ {
				for _, allow := range []func(string) bool{nil, odd} {
					hide := 0
					if q%3 == 0 {
						hide = 1 + rng.Intn(n)
					}
					st.hide, ref.hide = hide, hide
					got := g.searchLayer(st, query, g.entry, ef, lc, allow)
					want := g.searchLayerOnePass(ref, query, g.entry, ef, lc, allow)
					if !slices.Equal(got, want) {
						t.Fatalf("query %d ef %d layer %d filtered=%v: searchLayer diverged from the one-pass loop\n got %v\nwant %v",
							q, ef, lc, allow != nil, got, want)
					}
				}
			}
		}
	}
	st.hide, ref.hide = 0, 0
}

// TestTouchLinesReadsEveryLine pins the stride: one load per 128-byte line,
// starting at the first element, and none past the end.
func TestTouchLinesReadsEveryLine(t *testing.T) {
	for _, n := range []int{0, 1, 31, 32, 33, 512, 768} {
		v := make([]float32, n)
		want := float32(0)
		for i := 0; i < n; i += touchStride {
			v[i] = 1
			want++
		}
		if got := touchLines(v); got != want {
			t.Fatalf("len %d: touched %v lines, want %v", n, got, want)
		}
	}
}
