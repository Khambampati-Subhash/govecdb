package hnsw

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

// TestConcurrentSearchMatchesSerial is the test that would have failed before
// the scratch moved off Graph: identical queries run in parallel must produce
// exactly what they produce serially. If two searches shared a visited set or a
// heap they would corrupt each other's traversal, and the results would drift.
func TestConcurrentSearchMatchesSerial(t *testing.T) {
	const (
		n       = 1000
		dim     = 32
		k       = 10
		ef      = 64
		queries = 64
	)
	rng := rand.New(rand.NewSource(1234))
	g, _ := New(DefaultConfig(dim, Cosine))
	for i := range n {
		if err := g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, dim)); err != nil {
			t.Fatal(err)
		}
	}

	qs := make([][]float32, queries)
	want := make([][]Result, queries)
	for i := range qs {
		qs[i] = randomVector(rng, dim)
		res, err := g.Search(qs[i], k, ef)
		if err != nil {
			t.Fatal(err)
		}
		want[i] = res
	}

	const goroutines = 16
	got := make([][][]Result, goroutines)
	var wg sync.WaitGroup
	for gi := range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			local := make([][]Result, queries)
			for i := range qs {
				res, err := g.Search(qs[i], k, ef)
				if err != nil {
					t.Error(err)
					return
				}
				local[i] = res
			}
			got[gi] = local
		}()
	}
	wg.Wait()

	for gi := range goroutines {
		for qi := range queries {
			if len(got[gi][qi]) != len(want[qi]) {
				t.Fatalf("goroutine %d query %d: got %d results, want %d",
					gi, qi, len(got[gi][qi]), len(want[qi]))
			}
			for i := range want[qi] {
				if got[gi][qi][i] != want[qi][i] {
					t.Fatalf("goroutine %d query %d rank %d: got %+v, want %+v",
						gi, qi, i, got[gi][qi][i], want[qi][i])
				}
			}
		}
	}
}

// TestConcurrentInsertAndSearch runs writers and readers against one graph. Its
// value is under -race; on top of that it asserts readers never observe a torn
// graph — every id handed back must be one that was actually inserted.
func TestConcurrentInsertAndSearch(t *testing.T) {
	const (
		dim         = 32
		writers     = 4
		perWriter   = 250
		readers     = 8
		readQueries = 200
	)
	g, _ := New(DefaultConfig(dim, Cosine))

	valid := make(map[string]struct{}, writers*perWriter)
	for w := range writers {
		for i := range perWriter {
			valid[fmt.Sprintf("w%d-%d", w, i)] = struct{}{}
		}
	}

	var wg sync.WaitGroup
	done := make(chan struct{})

	for w := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(w)))
			for i := range perWriter {
				if err := g.Insert(fmt.Sprintf("w%d-%d", w, i), randomVector(rng, dim)); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}

	for r := range readers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(1000 + r)))
			for range readQueries {
				select {
				case <-done:
					return
				default:
				}
				res, err := g.Search(randomVector(rng, dim), 10, 64)
				if err != nil {
					t.Error(err)
					return
				}
				for _, hit := range res {
					if _, ok := valid[hit.ID]; !ok {
						t.Errorf("search returned unknown id %q", hit.ID)
						return
					}
				}
			}
		}()
	}

	wg.Wait()
	close(done)

	if g.Len() != writers*perWriter {
		t.Fatalf("expected %d nodes, got %d", writers*perWriter, g.Len())
	}
}

// TestSearchDoesNotMutateQuery is the read-only contract at the API edge: with
// Cosine the query gets normalized, and that must happen in pooled scratch, not
// in the caller's slice.
func TestSearchDoesNotMutateQuery(t *testing.T) {
	g, _ := New(DefaultConfig(4, Cosine))
	if err := g.Insert("a", []float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	query := []float32{4, 3, 2, 1}
	before := append([]float32(nil), query...)
	if _, err := g.Search(query, 1, 10); err != nil {
		t.Fatal(err)
	}
	for i := range before {
		if query[i] != before[i] {
			t.Fatalf("Search mutated the caller's query: got %v, want %v", query, before)
		}
	}
}

func BenchmarkSearchParallel(b *testing.B) {
	g, qs := buildFixture()
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			_, _ = g.Search(qs[i%len(qs)], benchK, benchEf)
			i++
		}
	})
}
