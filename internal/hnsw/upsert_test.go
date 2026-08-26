package hnsw

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

// TestInsertReplacesVector is the whole of upsert in one test: a second Insert
// under a live id resolves to the new vector, and the old one stops being an
// answer — without the id turning into two entries.
func TestInsertReplacesVector(t *testing.T) {
	g, _ := New(DefaultConfig(4, Euclidean))
	for _, kv := range []struct {
		id  string
		vec []float32
	}{
		{"a", []float32{1, 0, 0, 0}},
		{"b", []float32{0, 1, 0, 0}},
		{"c", []float32{0, 0, 1, 0}},
	} {
		if err := g.Insert(kv.id, kv.vec); err != nil {
			t.Fatal(err)
		}
	}

	if err := g.Insert("a", []float32{0, 0, 0, 1}); err != nil {
		t.Fatal(err)
	}

	res, err := g.Search([]float32{0, 0, 0, 1}, 1, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].ID != "a" || res[0].Distance > 1e-6 {
		t.Fatalf("a did not move to its new vector: %+v", res)
	}

	// The replaced vector must be gone as an answer at every rank, not merely
	// outranked by the new one.
	res, err = g.Search([]float32{1, 0, 0, 0}, 3, 10)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if r.ID == "a" && r.Distance < 1e-6 {
			t.Fatalf("the replaced vector for a is still being returned: %+v", res)
		}
	}

	// An id is one entry. Asking for more results than there are live vectors
	// is how a resurrected slot would show up: as a duplicate id.
	res, err = g.Search([]float32{0, 0, 0, 1}, 10, 32)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 3 {
		t.Fatalf("expected 3 live vectors, got %d: %+v", len(res), res)
	}
	seen := make(map[string]struct{}, len(res))
	for _, r := range res {
		if _, dup := seen[r.ID]; dup {
			t.Fatalf("id %s returned twice: %+v", r.ID, res)
		}
		seen[r.ID] = struct{}{}
	}
	if got := g.Len(); got != 3 {
		t.Fatalf("Len = %d, want 3", got)
	}
}

// TestUpsertCountsTheOldSlotAsATombstone pins the accounting: an update adds no
// live vectors and one dead slot. Stats has to say so, because "is it time to
// compact?" is a question about slots, and an update-heavy workload accumulates
// them just as fast as a delete-heavy one.
func TestUpsertCountsTheOldSlotAsATombstone(t *testing.T) {
	g, _ := buildGraph(t, 100, 8, 3)
	if s := g.Stats(); s != (Stats{Live: 100, Deleted: 0, Slots: 100}) {
		t.Fatalf("before updates: %+v", s)
	}

	rng := rand.New(rand.NewSource(4))
	for i := range 10 {
		if err := g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, 8)); err != nil {
			t.Fatal(err)
		}
	}

	if s := g.Stats(); s != (Stats{Live: 100, Deleted: 10, Slots: 110}) {
		t.Fatalf("after 10 updates: %+v", s)
	}
}

// TestUpsertUnchangedVectorIsFree pins the replay path. Re-applying a record the
// graph already holds is the normal shape of WAL recovery across a snapshot
// boundary; if it cost a slot, every recovery would inflate the graph with
// tombstones for vectors that never changed.
func TestUpsertUnchangedVectorIsFree(t *testing.T) {
	t.Run("identical", func(t *testing.T) {
		g, data := buildGraph(t, 50, 8, 6)
		before := g.Stats()

		for i := range 50 {
			id := fmt.Sprintf("v%d", i)
			// A distinct slice with the same contents: equality is by value, not
			// by whether the caller happened to hand back the same array.
			again := append([]float32(nil), data[id]...)
			if err := g.Insert(id, again); err != nil {
				t.Fatal(err)
			}
		}

		if got := g.Stats(); got != before {
			t.Fatalf("replaying every vector unchanged mutated the graph: %+v -> %+v", before, got)
		}
	})

	t.Run("rescaled under cosine", func(t *testing.T) {
		// buildGraph uses Cosine, which stores direction only, so a
		// magnitude-only change is not a change. The comparison runs against the
		// stored (normalized) form, which is what makes this fall out for free.
		// Doubling is exact in binary floating point, so the normalized result is
		// bit-identical rather than merely close.
		g, data := buildGraph(t, 50, 8, 6)
		before := g.Stats()

		for i := range 50 {
			id := fmt.Sprintf("v%d", i)
			scaled := make([]float32, 8)
			for j, x := range data[id] {
				scaled[j] = x * 2
			}
			if err := g.Insert(id, scaled); err != nil {
				t.Fatal(err)
			}
		}

		if got := g.Stats(); got != before {
			t.Fatalf("rescaling every vector mutated the graph: %+v -> %+v", before, got)
		}
	})
}

// TestUpsertRejectsBadVectorWithoutDestroying: validation runs before anything
// is tombstoned, so a rejected update leaves the stored vector intact. The
// alternative ordering — unbind the id, then discover the vector is malformed —
// turns a caller's bug into data loss.
func TestUpsertRejectsBadVectorWithoutDestroying(t *testing.T) {
	g, data := buildGraph(t, 20, 8, 19)
	before := g.Stats()

	if err := g.Insert("v5", nil); err != ErrEmptyVector {
		t.Fatalf("want ErrEmptyVector, got %v", err)
	}
	if err := g.Insert("v5", make([]float32, 9)); err != ErrDimensionMismatch {
		t.Fatalf("want ErrDimensionMismatch, got %v", err)
	}

	if got := g.Stats(); got != before {
		t.Fatalf("a rejected update mutated the graph: %+v -> %+v", before, got)
	}
	res, err := g.Search(data["v5"], 1, 32)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != "v5" {
		t.Fatalf("v5 lost its vector to a rejected update: %+v", res)
	}
}

// TestUpsertEntryPointReelects updates the entry point over and over. It is the
// same hazard Delete has — the entry must stay live and stay at maxLevel — but
// reached through a path that also inserts, so the two halves have to agree
// about the entry within a single write lock.
func TestUpsertEntryPointReelects(t *testing.T) {
	const (
		n   = 300
		dim = 16
	)
	g, data := buildGraph(t, n, dim, 5)
	rng := rand.New(rand.NewSource(8))

	for round := range 20 {
		g.mu.RLock()
		oldIdx := g.entry
		entryID := g.nodes[oldIdx].id
		g.mu.RUnlock()

		if err := g.Insert(entryID, randomVector(rng, dim)); err != nil {
			t.Fatal(err)
		}

		g.mu.RLock()
		entry, maxLevel := g.entry, g.maxLevel
		g.mu.RUnlock()

		if entry == oldIdx {
			t.Fatalf("round %d: entry still points at the replaced slot", round)
		}
		if entry == -1 {
			t.Fatalf("round %d: graph is full of live nodes but lost its entry", round)
		}
		// Invariant 1: the entry is live.
		if g.nodes[entry].deleted {
			t.Fatalf("round %d: entry is a tombstone", round)
		}
		// Invariant 2: the entry sits exactly at maxLevel — Insert's descent
		// indexes the entry's own neighbor slice from maxLevel downwards.
		if got := g.nodes[entry].topLevel(); got != maxLevel {
			t.Fatalf("round %d: entry topLevel = %d but maxLevel = %d", round, got, maxLevel)
		}
		if got := g.Len(); got != n {
			t.Fatalf("round %d: Len = %d, want %d — an update is not a delete", round, got, n)
		}
	}

	res, err := g.Search(data["v150"], 1, 64)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != "v150" {
		t.Fatalf("graph stopped working after 20 entry-point updates: %+v", res)
	}
}

// TestRepeatedUpsertsResolveToLatest is the workload upsert exists for:
// re-embedding the same corpus, repeatedly. Every id must resolve to its most
// recent vector after every round, and the live population must not drift.
//
// It is also the sharpest test of the tombstone side: by the last round three
// quarters of the slots are dead, and a live vector that lost its inbound edges
// to one of them would be silently unreachable.
func TestRepeatedUpsertsResolveToLatest(t *testing.T) {
	const (
		n      = 300
		dim    = 16
		rounds = 3
	)
	g, data := buildGraph(t, n, dim, 11)
	rng := rand.New(rand.NewSource(12))

	for round := 1; round <= rounds; round++ {
		for i := range n {
			id := fmt.Sprintf("v%d", i)
			v := randomVector(rng, dim)
			data[id] = v
			if err := g.Insert(id, v); err != nil {
				t.Fatal(err)
			}
		}
		if got := g.Len(); got != n {
			t.Fatalf("round %d: Len = %d, want %d", round, got, n)
		}

		// ef is deliberately wide: the question here is reachability, not recall.
		var missed int
		for id, vec := range data {
			res, err := g.Search(vec, 1, 200)
			if err != nil {
				t.Fatal(err)
			}
			if len(res) == 0 || res[0].ID != id {
				missed++
			}
		}
		if missed > 0 {
			t.Fatalf("round %d: %d/%d ids do not resolve to their latest vector", round, missed, n)
		}
	}

	// Three full rounds of replacement: 300 live vectors carried by 1200 slots.
	// Nothing reclaims them yet, and this ratio is the case for compaction.
	want := Stats{Live: n, Deleted: n * rounds, Slots: n * (rounds + 1)}
	if s := g.Stats(); s != want {
		t.Fatalf("after %d rounds: %+v, want %+v", rounds, s, want)
	}
}

// TestRecallAfterUpserts holds updates to the same bar as the rest of the index:
// recall against brute force over the current state of the data, not merely
// "the new vector comes back".
func TestRecallAfterUpserts(t *testing.T) {
	const (
		n   = 2000
		dim = 32
		k   = 10
		ef  = 64
	)
	g, data := buildGraph(t, n, dim, 17)

	rng := rand.New(rand.NewSource(18))
	for i := 0; i < n; i += 2 {
		id := fmt.Sprintf("v%d", i)
		v := randomVector(rng, dim)
		data[id] = v
		if err := g.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}

	var hits, total int
	for range 100 {
		query := randomVector(rng, dim)
		got, err := g.Search(query, k, ef)
		if err != nil {
			t.Fatal(err)
		}
		want := bruteForceNearest(data, CosineDistance, query, k)
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
	t.Logf("recall@%d after updating half the graph: %.3f", k, recall)
	if recall < 0.90 {
		t.Fatalf("recall collapsed under updates: %.3f (want >= 0.90)", recall)
	}
}

// TestConcurrentUpsertAndSearch covers the reason Insert tombstones under its
// own write lock instead of calling Delete: an update must never be observable
// as a disappearance. Len is the sharp instrument — while ids are only ever
// being replaced it must hold constant, and it would dip if the unbind and the
// re-add were two separately locked steps.
func TestConcurrentUpsertAndSearch(t *testing.T) {
	const (
		n       = 400
		dim     = 16
		readers = 8
	)
	g, _ := buildGraph(t, n, dim, 23)

	var wg sync.WaitGroup
	done := make(chan struct{})

	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(done)
		rng := rand.New(rand.NewSource(24))
		for i := range n {
			if err := g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, dim)); err != nil {
				t.Error(err)
				return
			}
		}
	}()

	for r := range readers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(900 + r)))
			for {
				select {
				case <-done:
					return
				default:
				}
				if got := g.Len(); got != n {
					t.Errorf("Len = %d mid-update, want a constant %d", got, n)
					return
				}
				if _, err := g.Search(randomVector(rng, dim), 10, 64); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()

	if got := g.Len(); got != n {
		t.Fatalf("Len = %d after updating every id, want %d", got, n)
	}
	if s := g.Stats(); s.Deleted != n || s.Slots != 2*n {
		t.Fatalf("every id was replaced exactly once: %+v", s)
	}
}
