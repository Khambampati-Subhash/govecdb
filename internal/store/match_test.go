package store

import (
	"fmt"
	"sync"
	"testing"
)

func TestMatchSeesTheStoredValues(t *testing.T) {
	s := New()
	s.Put("a", Metadata{"kind": "doc", "page": int64(3)})

	if !s.Match("a", func(md Metadata) bool {
		return md["kind"] == "doc" && md["page"] == int64(3)
	}) {
		t.Fatal("the predicate did not see the stored metadata")
	}
	if s.Match("a", func(md Metadata) bool { return md["kind"] == "other" }) {
		t.Fatal("the predicate matched a value that is not stored")
	}
}

// An id with no metadata is handed a nil map rather than skipped, because "has
// no metadata" is a thing a filter is entitled to ask about.
func TestMatchPassesNilForAnAbsentID(t *testing.T) {
	s := New()
	s.Put("a", Metadata{"k": "v"})

	called := false
	got := s.Match("missing", func(md Metadata) bool {
		called = true
		if md != nil {
			t.Errorf("predicate got %#v, want a nil map", md)
		}
		_, ok := md["anything"]
		return !ok
	})
	if !called {
		t.Fatal("the predicate was not called for an id with no metadata")
	}
	if !got {
		t.Fatal("Match did not return the predicate's answer")
	}
}

// The whole reason Match exists rather than callers using Get: it runs once per
// candidate node in a search, so it must not copy the map. Get's copy is one
// allocation per call, which at thousands of candidates is the search's dominant
// cost rather than a rounding error.
func TestMatchDoesNotAllocate(t *testing.T) {
	s := New()
	s.Put("a", Metadata{"one": "1", "two": int64(2), "three": 3.0, "four": true})

	pred := func(md Metadata) bool { return md["two"] == int64(2) }
	if n := testing.AllocsPerRun(100, func() { s.Match("a", pred) }); n != 0 {
		t.Fatalf("Match allocates %v times per call, want 0 — it is copying the map", n)
	}

	// Get, for contrast, is expected to allocate: it hands the map out.
	if n := testing.AllocsPerRun(100, func() { s.Get("a") }); n == 0 {
		t.Fatal("Get no longer copies; the reason Match exists needs rechecking")
	}
}

func TestMatchIsSafeAlongsideWriters(t *testing.T) {
	s := New()
	for i := range 100 {
		s.Put(fmt.Sprintf("v%d", i), Metadata{"n": int64(i)})
	}

	var wg sync.WaitGroup
	for w := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 200 {
				s.Put(fmt.Sprintf("v%d", i%100), Metadata{"n": int64(i), "w": int64(w)})
			}
		}()
	}
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 200 {
				s.Match(fmt.Sprintf("v%d", i%100), func(md Metadata) bool {
					_, ok := md["n"]
					return ok
				})
			}
		}()
	}
	wg.Wait()
}
