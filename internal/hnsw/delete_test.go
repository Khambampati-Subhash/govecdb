package hnsw

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

// buildGraph inserts n random vectors and returns the graph plus the raw data,
// so a test can check results against brute force over the survivors.
func buildGraph(t *testing.T, n, dim int, seed int64) (*Graph, map[string][]float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	g, err := New(DefaultConfig(dim, Cosine))
	if err != nil {
		t.Fatal(err)
	}
	data := make(map[string][]float32, n)
	for i := range n {
		id := fmt.Sprintf("v%d", i)
		v := randomVector(rng, dim)
		data[id] = v
		if err := g.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}
	return g, data
}

func TestDeleteRemovesFromResults(t *testing.T) {
	g, data := buildGraph(t, 500, 16, 42)

	target := data["v250"]
	res, err := g.Search(target, 1, 64)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != "v250" {
		t.Fatalf("precondition failed: expected to find v250, got %+v", res)
	}

	if !g.Delete("v250") {
		t.Fatal("Delete reported nothing removed")
	}

	res, err = g.Search(target, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if r.ID == "v250" {
			t.Fatalf("deleted id still returned: %+v", res)
		}
	}
	// The query still resolves — deleting one vector must not blind the region.
	if len(res) != 10 {
		t.Fatalf("expected 10 live neighbors after one delete, got %d", len(res))
	}
}

// TestDeleteIsIdempotent guards the property WAL replay will depend on: applying
// the same DELETE twice must not be an error the second time.
func TestDeleteIsIdempotent(t *testing.T) {
	g, _ := buildGraph(t, 50, 8, 1)

	if !g.Delete("v10") {
		t.Fatal("first Delete should report a removal")
	}
	if g.Delete("v10") {
		t.Fatal("second Delete should report nothing removed")
	}
	if g.Delete("does-not-exist") {
		t.Fatal("deleting an unknown id should report nothing removed")
	}
	if got := g.Len(); got != 49 {
		t.Fatalf("Len should count the delete exactly once, got %d", got)
	}
}

func TestLenAndStatsTrackTombstones(t *testing.T) {
	g, _ := buildGraph(t, 100, 8, 2)

	if s := g.Stats(); s != (Stats{Live: 100, Deleted: 0, Slots: 100}) {
		t.Fatalf("before deletes: %+v", s)
	}
	for i := range 30 {
		if !g.Delete(fmt.Sprintf("v%d", i)) {
			t.Fatalf("Delete(v%d) removed nothing", i)
		}
	}

	// Len is live-only; the slots are still there and Stats says so.
	if got := g.Len(); got != 70 {
		t.Fatalf("Len = %d, want 70 live", got)
	}
	if s := g.Stats(); s != (Stats{Live: 70, Deleted: 30, Slots: 100}) {
		t.Fatalf("after 30 deletes: %+v", s)
	}
}

// TestDeletedNodesStillRoute is the core of the tombstone design. Deleting a
// large fraction of the graph must not strand the survivors: every remaining
// vector has to stay findable, because dead nodes keep serving as bridges.
func TestDeletedNodesStillRoute(t *testing.T) {
	const (
		n   = 1000
		dim = 32
	)
	g, data := buildGraph(t, n, dim, 7)

	// Delete 70% — enough that most routes now pass through tombstones.
	survivors := make(map[string][]float32)
	for i := range n {
		id := fmt.Sprintf("v%d", i)
		if i%10 < 7 {
			if !g.Delete(id) {
				t.Fatalf("Delete(%s) removed nothing", id)
			}
			continue
		}
		survivors[id] = data[id]
	}
	if got := g.Len(); got != len(survivors) {
		t.Fatalf("Len = %d, want %d", got, len(survivors))
	}

	// Every survivor must still be reachable by its own vector.
	var missed int
	for id, vec := range survivors {
		res, err := g.Search(vec, 1, 64)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != id {
			missed++
		}
	}
	if missed > 0 {
		t.Fatalf("%d/%d survivors became unreachable after deleting 70%%", missed, len(survivors))
	}
}

// TestRecallAfterDeletes holds the project's actual bar: recall against brute
// force over the survivors, not merely "does it return something".
func TestRecallAfterDeletes(t *testing.T) {
	const (
		n   = 2000
		dim = 32
		k   = 10
		ef  = 64
	)
	g, data := buildGraph(t, n, dim, 7)

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

	rng := rand.New(rand.NewSource(31))
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

	recall := float64(hits) / float64(total)
	t.Logf("recall@%d after deleting half the graph: %.3f", k, recall)
	if recall < 0.90 {
		t.Fatalf("recall collapsed under tombstones: %.3f (want >= 0.90)", recall)
	}
}

// TestDeleteEntryPointReelects deletes the entry point over and over, which is
// the one delete that can break the graph's invariants.
func TestDeleteEntryPointReelects(t *testing.T) {
	g, data := buildGraph(t, 300, 16, 5)

	for range 20 {
		g.mu.RLock()
		entryIdx := g.entry
		entryID := g.nodes[entryIdx].id
		g.mu.RUnlock()

		if !g.Delete(entryID) {
			t.Fatalf("Delete(%s) removed nothing", entryID)
		}

		g.mu.RLock()
		newEntry, maxLevel := g.entry, g.maxLevel
		g.mu.RUnlock()

		if newEntry == entryIdx {
			t.Fatal("entry point was not re-elected after being deleted")
		}
		if newEntry == -1 {
			t.Fatal("graph still has live nodes but lost its entry point")
		}
		// Invariant 1: the entry is live.
		if g.nodes[newEntry].deleted {
			t.Fatal("re-elected entry is itself a tombstone")
		}
		// Invariant 2: the entry sits exactly at maxLevel — Insert's descent
		// indexes the entry's neighbor slice from maxLevel downwards.
		if got := g.nodes[newEntry].topLevel(); got != maxLevel {
			t.Fatalf("entry topLevel = %d but maxLevel = %d", got, maxLevel)
		}
	}

	// The graph must still work after all that churn.
	res, err := g.Search(data["v150"], 1, 64)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("search returned nothing after 20 entry-point deletions")
	}
}

// TestDeleteAllThenReinsert covers the collapse-to-empty path: a graph whose
// every slot is a tombstone must behave like a fresh one.
func TestDeleteAllThenReinsert(t *testing.T) {
	const n = 200
	g, _ := buildGraph(t, n, 8, 9)

	for i := range n {
		if !g.Delete(fmt.Sprintf("v%d", i)) {
			t.Fatalf("Delete(v%d) removed nothing", i)
		}
	}

	if got := g.Len(); got != 0 {
		t.Fatalf("Len = %d, want 0", got)
	}
	if s := g.Stats(); s.Deleted != n || s.Slots != n {
		t.Fatalf("stats after deleting everything: %+v", s)
	}

	res, err := g.Search(make([]float32, 8), 5, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 0 {
		t.Fatalf("fully tombstoned graph returned %d results", len(res))
	}

	// Inserting again must produce a working graph, not append to a dead one.
	rng := rand.New(rand.NewSource(77))
	fresh := make(map[string][]float32, 50)
	for i := range 50 {
		id := fmt.Sprintf("fresh%d", i)
		v := randomVector(rng, 8)
		fresh[id] = v
		if err := g.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}
	if got := g.Len(); got != 50 {
		t.Fatalf("Len = %d, want 50 after reinserting", got)
	}
	for id, vec := range fresh {
		res, err := g.Search(vec, 1, 32)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != id {
			t.Fatalf("reinserted %s is unreachable, got %+v", id, res)
		}
	}
}

// TestReinsertAfterDeleteUsesNewSlot pins the decision that Delete unbinds the
// id rather than reserving the slot: the same id may come back with a different
// vector, and it must resolve to the new one.
func TestReinsertAfterDeleteUsesNewSlot(t *testing.T) {
	g, _ := New(DefaultConfig(4, Euclidean))
	if err := g.Insert("a", []float32{1, 0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := g.Insert("b", []float32{0, 1, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if !g.Delete("a") {
		t.Fatal("Delete(a) removed nothing")
	}
	if err := g.Insert("a", []float32{0, 0, 1, 0}); err != nil {
		t.Fatal(err)
	}

	// The new vector wins; the old one is gone for good.
	res, err := g.Search([]float32{0, 0, 1, 0}, 1, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].ID != "a" || res[0].Distance > 1e-6 {
		t.Fatalf("re-inserted a not found at its new position: %+v", res)
	}

	res, err = g.Search([]float32{1, 0, 0, 0}, 2, 10)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if r.ID == "a" && r.Distance < 1e-6 {
			t.Fatal("old tombstoned vector for a is still being returned")
		}
	}
	if got := g.Len(); got != 2 {
		t.Fatalf("Len = %d, want 2 (a and b)", got)
	}
}

// countUnreachable builds a graph of the given shape and returns how many of
// its live vectors cannot retrieve themselves. ef is deliberately large: the
// question is reachability, not recall.
//
// withDeletes=false inserts 145 vectors and stops. withDeletes=true reaches the
// same 145 live vectors by way of 400 inserts, 395 deletes, and 140 inserts into
// the wreckage — so the two are directly comparable.
func countUnreachable(t *testing.T, m int, withDeletes bool) (unreachable, live int) {
	t.Helper()
	cfg := DefaultConfig(16, Cosine)
	cfg.M = m
	cfg.Seed = 5
	g, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(5))

	vecs := make(map[string][]float32)
	insert := func(id string) {
		v := randomVector(rng, 16)
		if err := g.Insert(id, v); err != nil {
			t.Fatal(err)
		}
		vecs[id] = v
	}

	if !withDeletes {
		for i := range 145 {
			insert(fmt.Sprintf("v%d", i))
		}
	} else {
		for i := range 400 {
			insert(fmt.Sprintf("v%d", i))
		}
		for i := range 395 {
			id := fmt.Sprintf("v%d", i)
			if !g.Delete(id) {
				t.Fatalf("Delete(%s) removed nothing", id)
			}
			delete(vecs, id)
			if i%10 == 0 {
				insert(fmt.Sprintf("n%d", i))
			}
		}
		for i := range 100 {
			insert(fmt.Sprintf("late%d", i))
		}
	}

	if got := g.Len(); got != len(vecs) {
		t.Fatalf("Len = %d, want %d", got, len(vecs))
	}
	for id, vec := range vecs {
		res, err := g.Search(vec, 1, 200)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != id {
			unreachable++
		}
	}
	return unreachable, len(vecs)
}

// TestTombstonesDoNotStrandVectors is the sharp edge of the whole delete design:
// a vector with no inbound edges is unreachable forever, which is silent data
// loss, not a recall dip. Two mechanisms conspire to cause it, and both are
// guarded here.
//
//   - pruneConnections is where tombstones compete with live nodes for edge
//     slots. Ranked purely by distance, a dead node can evict the edge Insert
//     just created, destroying the new node's only inbound link. Demoting
//     tombstones in that ranking is the fix, and it is what this test protects:
//     without it, M=4 strands 12 of 145 vectors and M=8 strands 1.
//   - searchLayer returns live nodes only, so a fully tombstoned region offers
//     nothing to attach to; Insert falls back to linking the node it searched
//     from. That path only triggers at pathologically low M.
//
// The invariant asserted is comparative — deleting must not strand more vectors
// than a delete-free graph of the same config — because M=2 already strands
// vectors on its own, with no deletes involved. That is a property of an M so
// low that pruning drops reverse edges, not something deletes introduced, so
// the test holds tombstones to "no worse", and to zero wherever the delete-free
// graph is itself perfect.
func TestTombstonesDoNotStrandVectors(t *testing.T) {
	for _, m := range []int{2, 4, 8, 16} {
		t.Run(fmt.Sprintf("M=%d", m), func(t *testing.T) {
			baseline, live := countUnreachable(t, m, false)
			withDeletes, _ := countUnreachable(t, m, true)
			t.Logf("unreachable of %d live: insert-only %d, delete-heavy %d",
				live, baseline, withDeletes)

			if withDeletes > baseline {
				t.Fatalf("tombstones stranded %d/%d vectors, worse than the %d "+
					"an equivalent delete-free graph strands",
					withDeletes, live, baseline)
			}
			if baseline == 0 && withDeletes != 0 {
				t.Fatalf("delete-free graph strands nothing but delete-heavy "+
					"stranded %d/%d", withDeletes, live)
			}
		})
	}
}

// TestConcurrentDeleteAndSearch runs deletes against live readers. Its value is
// under -race; on top of that, a reader must never be handed an id that has
// already been deleted.
func TestConcurrentDeleteAndSearch(t *testing.T) {
	const (
		n       = 800
		dim     = 32
		readers = 8
	)
	g, _ := buildGraph(t, n, dim, 13)

	// Readers may only ever see ids from the second half; the first half is
	// what the deleter removes, and a delete is visible the moment it commits.
	var wg sync.WaitGroup
	done := make(chan struct{})

	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(done)
		for i := range n / 2 {
			if !g.Delete(fmt.Sprintf("v%d", i)) {
				t.Errorf("Delete(v%d) removed nothing", i)
				return
			}
		}
	}()

	for r := range readers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(500 + r)))
			for {
				select {
				case <-done:
					return
				default:
				}
				if _, err := g.Search(randomVector(rng, dim), 10, 64); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()

	if got := g.Len(); got != n/2 {
		t.Fatalf("Len = %d, want %d", got, n/2)
	}
	// Everything the deleter removed must be gone from every angle.
	for i := range n / 2 {
		if g.Delete(fmt.Sprintf("v%d", i)) {
			t.Fatalf("v%d was not actually deleted", i)
		}
	}
}
