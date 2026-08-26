package hnsw

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

func TestCompactReclaimsSlots(t *testing.T) {
	const n = 1000
	g, _ := buildGraph(t, n, 16, 41)

	for i := range n {
		if i%10 < 4 {
			if !g.Delete(fmt.Sprintf("v%d", i)) {
				t.Fatalf("Delete(v%d) removed nothing", i)
			}
		}
	}
	before := g.Stats()
	if before != (Stats{Live: 600, Deleted: 400, Slots: 1000}) {
		t.Fatalf("before compaction: %+v", before)
	}
	if got := before.DeadRatio(); got != 0.4 {
		t.Fatalf("DeadRatio = %v, want 0.4", got)
	}

	if got := g.Compact(); got != 400 {
		t.Fatalf("Compact reclaimed %d slots, want 400", got)
	}

	// The whole point: slots now equal live vectors, and nothing was lost.
	after := g.Stats()
	if after != (Stats{Live: 600, Deleted: 0, Slots: 600}) {
		t.Fatalf("after compaction: %+v", after)
	}
	if got := after.DeadRatio(); got != 0 {
		t.Fatalf("DeadRatio = %v, want 0", got)
	}
}

// TestCompactKeepsEveryLiveVector is the safety property. Compaction rebuilds
// every neighbor list in the graph from scratch, so a bug here does not corrupt
// a vector — it silently drops one.
func TestCompactKeepsEveryLiveVector(t *testing.T) {
	const (
		n   = 1000
		dim = 32
	)
	g, data := buildGraph(t, n, dim, 43)

	survivors := make(map[string][]float32, n/2)
	deleted := make(map[string]struct{}, n/2)
	for i := range n {
		id := fmt.Sprintf("v%d", i)
		if i%2 == 0 {
			if !g.Delete(id) {
				t.Fatalf("Delete(%s) removed nothing", id)
			}
			deleted[id] = struct{}{}
			continue
		}
		survivors[id] = data[id]
	}
	g.Compact()

	if got := g.Len(); got != len(survivors) {
		t.Fatalf("Len = %d, want %d", got, len(survivors))
	}
	for id, vec := range survivors {
		res, err := g.Search(vec, 1, 64)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != id {
			t.Fatalf("%s did not survive compaction: %+v", id, res)
		}
		if res[0].Distance > 1e-6 {
			t.Fatalf("%s survived with the wrong vector: distance %v", id, res[0].Distance)
		}
	}

	// A compacted-away vector must not come back from the dead.
	rng := rand.New(rand.NewSource(44))
	for range 100 {
		res, err := g.Search(randomVector(rng, dim), 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range res {
			if _, dead := deleted[r.ID]; dead {
				t.Fatalf("compaction resurrected deleted id %s", r.ID)
			}
		}
	}
}

// TestCompactMatchesAFreshBuild pins what compaction *is*: the graph you would
// have built if the dead vectors had never existed. The replacement draws its
// levels from an RNG seeded off the same config and re-inserts in slot order, so
// the two graphs are identical structures, not merely equally good ones — which
// makes this testable by equality instead of by sampling recall.
//
// It also guards the subtler half of insertPrepared. Sending stored vectors back
// through prepare would re-normalize already-unit vectors, drifting them by an
// ulp; the distances below would then differ in their last digits and this test
// would fail while every recall test still passed.
func TestCompactMatchesAFreshBuild(t *testing.T) {
	const (
		n   = 800
		dim = 16
	)
	g, data := buildGraph(t, n, dim, 31)

	// Slot order is insertion order for an insert-only graph, so replaying the
	// survivors in this order is exactly what Compact does internally.
	var liveIDs []string
	for i := range n {
		id := fmt.Sprintf("v%d", i)
		if i%3 == 0 {
			if !g.Delete(id) {
				t.Fatalf("Delete(%s) removed nothing", id)
			}
			continue
		}
		liveIDs = append(liveIDs, id)
	}

	wantReclaimed := n - len(liveIDs)
	if got := g.Compact(); got != wantReclaimed {
		t.Fatalf("Compact reclaimed %d, want %d", got, wantReclaimed)
	}

	ref, err := New(DefaultConfig(dim, Cosine))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range liveIDs {
		if err := ref.Insert(id, data[id]); err != nil {
			t.Fatal(err)
		}
	}

	if g.Stats() != ref.Stats() {
		t.Fatalf("compacted %+v, fresh build %+v", g.Stats(), ref.Stats())
	}

	rng := rand.New(rand.NewSource(32))
	for q := range 50 {
		query := randomVector(rng, dim)
		got, err := g.Search(query, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		want, err := ref.Search(query, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(want) {
			t.Fatalf("query %d: compacted returned %d hits, fresh build %d", q, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("query %d rank %d: compacted %+v, fresh build %+v", q, i, got[i], want[i])
			}
		}
	}
}

// TestCompactIsANoOpWithoutTombstones guards the early return. Rebuilding a
// clean graph would throw away a perfectly good index and burn a full build to
// reclaim nothing, which matters because DeadRatio invites callers to poll
// Compact on a timer.
func TestCompactIsANoOpWithoutTombstones(t *testing.T) {
	const dim = 16
	g, _ := buildGraph(t, 300, dim, 45)

	rng := rand.New(rand.NewSource(46))
	queries := make([][]float32, 20)
	for i := range queries {
		queries[i] = randomVector(rng, dim)
	}
	before := make([][]Result, len(queries))
	for i, q := range queries {
		res, err := g.Search(q, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		before[i] = res
	}

	if got := g.Compact(); got != 0 {
		t.Fatalf("Compact on a clean graph reclaimed %d slots", got)
	}

	for i, q := range queries {
		res, err := g.Search(q, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		for r := range res {
			if res[r] != before[i][r] {
				t.Fatalf("query %d rank %d changed across a no-op compaction: %+v -> %+v",
					i, r, before[i][r], res[r])
			}
		}
	}
}

// TestCompactAfterUpserts is the workload compaction actually exists for. Deletes
// leak slots for vectors someone asked to remove; updates leak them for vectors
// that are still live, at whatever rate the corpus is re-embedded.
func TestCompactAfterUpserts(t *testing.T) {
	const (
		n      = 300
		dim    = 16
		rounds = 3
	)
	g, data := buildGraph(t, n, dim, 47)
	rng := rand.New(rand.NewSource(48))

	for range rounds {
		for i := range n {
			id := fmt.Sprintf("v%d", i)
			v := randomVector(rng, dim)
			data[id] = v
			if err := g.Insert(id, v); err != nil {
				t.Fatal(err)
			}
		}
	}
	if s := g.Stats(); s.Slots != n*(rounds+1) {
		t.Fatalf("before compaction: %+v", s)
	}

	if got := g.Compact(); got != n*rounds {
		t.Fatalf("Compact reclaimed %d slots, want %d", got, n*rounds)
	}
	if s := g.Stats(); s != (Stats{Live: n, Deleted: 0, Slots: n}) {
		t.Fatalf("after compaction: %+v", s)
	}

	// Every id must still resolve to its *latest* vector — compaction must keep
	// the surviving slot, not an earlier generation of the same id.
	for id, vec := range data {
		res, err := g.Search(vec, 1, 64)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != id || res[0].Distance > 1e-6 {
			t.Fatalf("%s lost its latest vector to compaction: %+v", id, res)
		}
	}
}

// TestCompactRecall holds the rebuilt graph to the project's bar, and records
// that compaction is not merely a memory win: edges that pointed at tombstones
// become edges between live nodes, and results fill at full speed again.
func TestCompactRecall(t *testing.T) {
	const (
		n   = 2000
		dim = 32
		k   = 10
		ef  = 64
	)
	g, data := buildGraph(t, n, dim, 49)

	survivors := make(map[string][]float32, n/2)
	for i := range n {
		id := fmt.Sprintf("v%d", i)
		if i%2 == 0 {
			if !g.Delete(id) {
				t.Fatalf("Delete(%s) removed nothing", id)
			}
			continue
		}
		survivors[id] = data[id]
	}

	measure := func(seed int64) float64 {
		rng := rand.New(rand.NewSource(seed))
		var hits, total int
		for range 100 {
			query := randomVector(rng, dim)
			got, err := g.Search(query, k, ef)
			if err != nil {
				t.Fatal(err)
			}
			want := bruteForceNearest(survivors, CosineDistance, query, k)
			wantSet := make(map[string]struct{}, len(want))
			for _, id := range want {
				wantSet[id] = struct{}{}
			}
			for _, r := range got {
				if _, ok := wantSet[r.ID]; ok {
					hits++
				}
			}
			total += len(want)
		}
		return float64(hits) / float64(total)
	}

	before := measure(50)
	g.Compact()
	after := measure(50)
	t.Logf("recall@%d: %.3f with 50%% tombstones, %.3f after compaction", k, before, after)

	if after < 0.90 {
		t.Fatalf("recall after compaction: %.3f (want >= 0.90)", after)
	}
	// Rebuilding removes dead ends; it must never cost recall.
	if after < before-0.01 {
		t.Fatalf("compaction cost recall: %.3f -> %.3f", before, after)
	}
}

// TestCompactAllDeleted covers the collapse-to-empty path: a graph whose every
// slot is a tombstone must compact to a genuinely fresh, working graph.
func TestCompactAllDeleted(t *testing.T) {
	const (
		n   = 200
		dim = 8
	)
	g, _ := buildGraph(t, n, dim, 51)
	for i := range n {
		if !g.Delete(fmt.Sprintf("v%d", i)) {
			t.Fatalf("Delete(v%d) removed nothing", i)
		}
	}

	if got := g.Compact(); got != n {
		t.Fatalf("Compact reclaimed %d slots, want %d", got, n)
	}
	if s := g.Stats(); s != (Stats{}) {
		t.Fatalf("a fully tombstoned graph should compact to empty, got %+v", s)
	}
	g.mu.RLock()
	entry := g.entry
	g.mu.RUnlock()
	if entry != -1 {
		t.Fatalf("empty graph has entry %d, want -1", entry)
	}

	rng := rand.New(rand.NewSource(52))
	fresh := make(map[string][]float32, 50)
	for i := range 50 {
		id := fmt.Sprintf("fresh%d", i)
		v := randomVector(rng, dim)
		fresh[id] = v
		if err := g.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}
	for id, vec := range fresh {
		res, err := g.Search(vec, 1, 32)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != id {
			t.Fatalf("%s unreachable after compacting an empty graph: %+v", id, res)
		}
	}
}

func TestCompactEmptyGraph(t *testing.T) {
	g, err := New(DefaultConfig(8, Cosine))
	if err != nil {
		t.Fatal(err)
	}
	if got := g.Compact(); got != 0 {
		t.Fatalf("Compact on a never-used graph reclaimed %d", got)
	}
	s := g.Stats()
	if s != (Stats{}) {
		t.Fatalf("%+v", s)
	}
	// Guards the zero check in DeadRatio: an empty graph is 0% dead, not NaN.
	if got := s.DeadRatio(); got != 0 {
		t.Fatalf("DeadRatio on an empty graph = %v, want 0 (not NaN)", got)
	}
}

// TestConcurrentCompactAndSearch runs a compaction against live readers. Compact
// takes the write lock for the whole rebuild, so the guarantee is not that
// readers keep running — it is that they never observe the swap: the live
// population is identical before and after, and no reader may see it change.
func TestConcurrentCompactAndSearch(t *testing.T) {
	const (
		n       = 800
		dim     = 16
		readers = 8
	)
	g, _ := buildGraph(t, n, dim, 53)
	for i := range n {
		if i%2 == 0 {
			if !g.Delete(fmt.Sprintf("v%d", i)) {
				t.Fatalf("Delete(v%d) removed nothing", i)
			}
		}
	}
	live := n / 2

	var wg sync.WaitGroup
	done := make(chan struct{})

	for r := range readers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(600 + r)))
			for {
				select {
				case <-done:
					return
				default:
				}
				if got := g.Len(); got != live {
					t.Errorf("Len = %d across a compaction, want a constant %d", got, live)
					return
				}
				if _, err := g.Search(randomVector(rng, dim), 10, 64); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}

	for range 5 {
		g.Compact()
	}
	close(done)
	wg.Wait()

	if s := g.Stats(); s != (Stats{Live: live, Deleted: 0, Slots: live}) {
		t.Fatalf("after compaction: %+v", s)
	}
}
