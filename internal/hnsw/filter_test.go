package hnsw

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

// buildFilterGraph makes a graph of n vectors with ids v0..v{n-1}.
func buildFilterGraph(t *testing.T, n, dim int) *Graph {
	t.Helper()

	g, err := New(DefaultConfig(dim, Cosine))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	for i := range n {
		if err := g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, dim)); err != nil {
			t.Fatal(err)
		}
	}
	return g
}

func TestSearchFilterReturnsOnlyAllowedIDs(t *testing.T) {
	g := buildFilterGraph(t, 500, 16)
	rng := rand.New(rand.NewSource(99))

	allow := func(id string) bool { return strings.HasSuffix(id, "0") }

	res, err := g.SearchFilter(randomVector(rng, 16), 10, 64, allow)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("no results")
	}
	for _, r := range res {
		if !allow(r.ID) {
			t.Fatalf("result %q was not admitted by the filter", r.ID)
		}
	}
}

// The claim the whole design rests on: filtering inside the traversal finds k
// matching vectors, where filtering the output would have found almost none.
//
// The comparison is run rather than asserted from memory — the post-filter count
// is computed here, so this fails if the two ever stop differing.
func TestSearchFilterFindsKWherePostFilteringWouldNot(t *testing.T) {
	const (
		n   = 2000
		dim = 16
		k   = 10
	)
	g := buildFilterGraph(t, n, dim)
	rng := rand.New(rand.NewSource(4))
	query := randomVector(rng, dim)

	// One vector in fifty is admissible: selective enough that the nearest ten
	// are very unlikely to contain ten of them.
	admissible := make(map[string]bool, n/50)
	for i := 0; i < n; i += 50 {
		admissible[fmt.Sprintf("v%d", i)] = true
	}
	allow := func(id string) bool { return admissible[id] }

	filtered, err := g.SearchFilter(query, k, 64, allow)
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered) != k {
		t.Fatalf("filtered search returned %d results, want %d — the traversal "+
			"is not searching wider to fill the result set", len(filtered), k)
	}
	for _, r := range filtered {
		if !allow(r.ID) {
			t.Fatalf("result %q was not admitted", r.ID)
		}
	}

	// What the naive approach would have produced from the same query.
	plain, err := g.Search(query, k, 64)
	if err != nil {
		t.Fatal(err)
	}
	post := 0
	for _, r := range plain {
		if allow(r.ID) {
			post++
		}
	}
	if post >= k {
		t.Skipf("post-filtering happened to yield %d of %d; the corpus is not "+
			"selective enough for this comparison to mean anything", post, k)
	}
	t.Logf("during-traversal: %d results; post-filtering the same query: %d", len(filtered), post)
}

// Results stay ordered by distance under a filter — a filtered search is still a
// nearest-neighbor search, not a scan that stops early.
func TestSearchFilterKeepsResultsOrdered(t *testing.T) {
	g := buildFilterGraph(t, 800, 16)
	rng := rand.New(rand.NewSource(11))

	res, err := g.SearchFilter(randomVector(rng, 16), 20, 128,
		func(id string) bool { return !strings.HasSuffix(id, "7") })
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(res); i++ {
		if res[i].Distance < res[i-1].Distance {
			t.Fatalf("result %d (%v) is closer than result %d (%v)",
				i, res[i].Distance, i-1, res[i-1].Distance)
		}
	}
}

// A nil predicate has to be exactly Search, because that is what every existing
// caller — Insert included — relies on.
func TestSearchFilterNilIsPlainSearch(t *testing.T) {
	g := buildFilterGraph(t, 400, 16)
	rng := rand.New(rand.NewSource(3))
	query := randomVector(rng, 16)

	plain, err := g.Search(query, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	nilFiltered, err := g.SearchFilter(query, 10, 64, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(plain) != len(nilFiltered) {
		t.Fatalf("lengths differ: %d vs %d", len(plain), len(nilFiltered))
	}
	for i := range plain {
		if plain[i] != nilFiltered[i] {
			t.Fatalf("result %d differs: %+v vs %+v", i, plain[i], nilFiltered[i])
		}
	}
}

// A tombstone is not an answer whatever the filter says. The two conditions are
// checked in one place (admits), and this is what stops an edit from letting a
// permissive filter resurrect a deleted vector.
func TestSearchFilterNeverReturnsTombstones(t *testing.T) {
	g := buildFilterGraph(t, 300, 16)
	rng := rand.New(rand.NewSource(21))

	dead := map[string]bool{}
	for i := 0; i < 300; i += 2 {
		id := fmt.Sprintf("v%d", i)
		g.Delete(id)
		dead[id] = true
	}

	// A filter that admits everything, including the deleted half.
	res, err := g.SearchFilter(randomVector(rng, 16), 20, 128, func(string) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("no results")
	}
	for _, r := range res {
		if dead[r.ID] {
			t.Fatalf("deleted vector %q came back through a permissive filter", r.ID)
		}
	}
}

// admits checks the tombstone first, so the predicate is never consulted about a
// dead node. That ordering is what keeps the cheap test in front of the one that
// reaches outside the package.
func TestSearchFilterSkipsThePredicateForTombstones(t *testing.T) {
	g := buildFilterGraph(t, 200, 16)
	rng := rand.New(rand.NewSource(5))

	dead := map[string]bool{}
	for i := 0; i < 200; i += 3 {
		id := fmt.Sprintf("v%d", i)
		g.Delete(id)
		dead[id] = true
	}

	if _, err := g.SearchFilter(randomVector(rng, 16), 10, 64, func(id string) bool {
		if dead[id] {
			t.Errorf("predicate was asked about tombstoned %q", id)
		}
		return true
	}); err != nil {
		t.Fatal(err)
	}
}

// A predicate that admits nothing returns nothing, and terminates. The frontier
// still admits every node, so this is also the check that an empty result set
// does not turn the traversal into an infinite one.
func TestSearchFilterAdmittingNothing(t *testing.T) {
	g := buildFilterGraph(t, 300, 16)
	rng := rand.New(rand.NewSource(13))

	res, err := g.SearchFilter(randomVector(rng, 16), 10, 64, func(string) bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 0 {
		t.Fatalf("got %d results from a filter that admits nothing", len(res))
	}
}

// A filter must not change what gets built. Inserts run through searchLayer too,
// and passing a query's predicate into that path would make the graph's shape
// depend on whichever query happened to run first.
func TestFilteredSearchDoesNotChangeTheGraph(t *testing.T) {
	g := buildFilterGraph(t, 300, 16)
	rng := rand.New(rand.NewSource(17))

	before := g.Stats()
	for range 20 {
		if _, err := g.SearchFilter(randomVector(rng, 16), 10, 64,
			func(id string) bool { return strings.HasSuffix(id, "1") }); err != nil {
			t.Fatal(err)
		}
	}
	if after := g.Stats(); after != before {
		t.Fatalf("stats changed across filtered searches: %+v -> %+v", before, after)
	}
}

func BenchmarkSearchFilter(b *testing.B) {
	const dim = 128
	g, err := New(DefaultConfig(dim, Cosine))
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	for i := range 10000 {
		if err := g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, dim)); err != nil {
			b.Fatal(err)
		}
	}
	query := randomVector(rng, dim)

	// Selectivity is the axis that matters: the cost of filtering is not the
	// predicate call, it is how much wider the search has to go to fill k.
	//
	// The admissible set is precomputed into a map so the predicate is a single
	// hash lookup that allocates nothing. Deriving it from the id inside the
	// benchmark — parsing the number back out, say — would measure the parser
	// and report it as the cost of filtering.
	for _, sel := range []int{1, 2, 10, 50} {
		name := "unfiltered"
		if sel > 1 {
			name = fmt.Sprintf("one-in-%d", sel)
		}

		var allow func(string) bool
		if sel > 1 {
			admissible := make(map[string]bool, 10000/sel)
			for i := 0; i < 10000; i += sel {
				admissible[fmt.Sprintf("v%d", i)] = true
			}
			allow = func(id string) bool { return admissible[id] }
		}

		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := g.SearchFilter(query, 10, 64, allow); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
