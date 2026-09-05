package govecdb

import (
	"errors"
	"fmt"
	"math/rand"
	"testing"
)

func TestSearchWithFilter(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 200, 1)
	rng := rand.New(rand.NewSource(9))
	query := vec(rng, testDim)

	cases := []struct {
		name string
		f    Filter
		ok   func(md Metadata) bool
	}{
		{"eq bool", Eq("even", true), func(md Metadata) bool { return md["even"] == true }},
		{"lt", Lt("n", 50), func(md Metadata) bool { return md["n"].(int64) < 50 }},
		{"gte", Gte("n", 150), func(md Metadata) bool { return md["n"].(int64) >= 150 }},
		{"in", In("n", 3, 7, 11, 19, 23, 31, 47), func(md Metadata) bool {
			switch md["n"].(int64) {
			case 3, 7, 11, 19, 23, 31, 47:
				return true
			}
			return false
		}},
		{"and", And(Eq("even", true), Lt("n", 40)), func(md Metadata) bool {
			return md["even"] == true && md["n"].(int64) < 40
		}},
		{"or", Or(Lt("n", 10), Gt("n", 190)), func(md Metadata) bool {
			n := md["n"].(int64)
			return n < 10 || n > 190
		}},
		{"not", Not(Eq("even", true)), func(md Metadata) bool { return md["even"] == false }},
		{"exists", Exists("n"), func(md Metadata) bool { _, ok := md["n"]; return ok }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := db.Search(SearchRequest{Query: query, K: 5, Filter: tc.f})
			if err != nil {
				t.Fatal(err)
			}
			if len(res) == 0 {
				t.Fatal("no results")
			}
			for _, m := range res {
				if m.Metadata == nil {
					t.Fatalf("%s: result %q came back without metadata", tc.name, m.ID)
				}
				if !tc.ok(m.Metadata) {
					t.Fatalf("%s: result %q has metadata %+v, which the filter should have excluded",
						tc.name, m.ID, m.Metadata)
				}
			}
		})
	}
}

// The end-to-end version of the index-level claim: a selective filter still
// returns K, because the filter narrows what the traversal accepts rather than
// what it hands back.
func TestFilteredSearchStillReturnsK(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 1000, 3)
	rng := rand.New(rand.NewSource(31))

	// One vector in twenty.
	res, err := db.Search(SearchRequest{
		Query:  vec(rng, testDim),
		K:      10,
		Filter: Eq("bucket", int64(0)),
	})
	if err != nil {
		t.Fatal(err)
	}
	// fill does not write a "bucket" key, so this filter matches nothing — the
	// check that an unsatisfiable filter returns empty rather than everything.
	if len(res) != 0 {
		t.Fatalf("a filter on an absent key returned %d results", len(res))
	}

	res, err = db.Search(SearchRequest{
		Query:  vec(rng, testDim),
		K:      10,
		Filter: Lt("n", 50), // 50 of 1000
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 10 {
		t.Fatalf("got %d results, want 10 — the search is not widening to fill K", len(res))
	}
	for _, m := range res {
		if m.Metadata["n"].(int64) >= 50 {
			t.Fatalf("result %q escaped the filter: %+v", m.ID, m.Metadata)
		}
	}
}

// A nil filter has to be exactly an unfiltered search, since that is what every
// call written before filters existed does.
func TestNilFilterIsUnfiltered(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 200, 5)
	rng := rand.New(rand.NewSource(2))
	query := vec(rng, testDim)

	plain, err := db.Search(SearchRequest{Query: query, K: 10})
	if err != nil {
		t.Fatal(err)
	}
	explicit, err := db.Search(SearchRequest{Query: query, K: 10, Filter: nil})
	if err != nil {
		t.Fatal(err)
	}
	if len(plain) != len(explicit) {
		t.Fatalf("lengths differ: %d vs %d", len(plain), len(explicit))
	}
	for i := range plain {
		if plain[i].ID != explicit[i].ID {
			t.Fatalf("result %d differs: %q vs %q", i, plain[i].ID, explicit[i].ID)
		}
	}
}

func TestSearchRejectsInvalidFilters(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 20, 7)
	rng := rand.New(rand.NewSource(4))
	query := vec(rng, testDim)

	for _, tc := range []struct {
		name string
		f    Filter
	}{
		{"unsupported operand type", Eq("n", []string{"a"})},
		{"nil operand", Lt("n", nil)},
		{"uint64 beyond int64", Gt("n", uint64(1)<<63)},
		{"nil inside And", And(Eq("n", 1), nil)},
		{"nil inside Or", Or(nil)},
		{"nil inside Not", Not(nil)},
		{"bad operand nested deep", And(Or(Not(Eq("n", struct{}{}))))},
		{"bad value in In", In("n", 1, 2, map[string]any{})},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := db.Search(SearchRequest{Query: query, K: 5, Filter: tc.f})
			if !errors.Is(err, ErrInvalidFilter) {
				t.Fatalf("error = %v, want ErrInvalidFilter", err)
			}
		})
	}
}

// A filter reads metadata, and metadata is restored from a snapshot and the log
// like everything else. This is the check that the two halves of recovery agree:
// an index that came back from a snapshot and a store that came back with it.
func TestFilteredSearchSurvivesRestart(t *testing.T) {
	db, dir := openDB(t)
	fill(t, db, 300, 11)

	// Snapshot part of the way, then write more, so recovery has to use both the
	// snapshot and the log tail.
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(13))
	for i := 300; i < 360; i++ {
		if err := db.Add(Vector{
			ID:       fmt.Sprintf("v%d", i),
			Values:   vec(rng, testDim),
			Metadata: Metadata{"n": int64(i), "even": i%2 == 0},
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	reopened := openDBAt(t, dir)
	query := vec(rng, testDim)

	// A vector from the snapshot and one from the log tail, both reachable
	// through a filter.
	for _, tc := range []struct {
		name string
		f    Filter
	}{
		{"from the snapshot", And(Gte("n", 100), Lt("n", 110))},
		{"from the log tail", And(Gte("n", 350), Lt("n", 360))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			res, err := reopened.Search(SearchRequest{Query: query, K: 5, Filter: tc.f})
			if err != nil {
				t.Fatal(err)
			}
			if len(res) == 0 {
				t.Fatal("no results: metadata did not survive the restart")
			}
			for _, m := range res {
				if !tc.f.Match(m.Metadata) {
					t.Fatalf("result %q has metadata %+v, which the filter excludes", m.ID, m.Metadata)
				}
			}
		})
	}
}

// Deleting a vector removes its metadata too, so a filter that used to match it
// must stop doing so.
func TestFilterDoesNotMatchDeletedVectors(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 100, 17)
	rng := rand.New(rand.NewSource(19))

	for i := range 50 {
		if err := db.Delete(fmt.Sprintf("v%d", i)); err != nil {
			t.Fatal(err)
		}
	}

	res, err := db.Search(SearchRequest{Query: vec(rng, testDim), K: 10, Filter: Lt("n", 50)})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 0 {
		t.Fatalf("got %d results for a filter matching only deleted vectors: %+v", len(res), res)
	}
}

// Vectors carrying no metadata at all are a legitimate thing to filter for, and
// Not is the only way to reach them.
func TestFilterOnVectorsWithoutMetadata(t *testing.T) {
	db, _ := openDB(t)
	rng := rand.New(rand.NewSource(23))

	for i := range 100 {
		v := Vector{ID: fmt.Sprintf("v%d", i), Values: vec(rng, testDim)}
		if i%2 == 0 {
			v.Metadata = Metadata{"tagged": true}
		}
		if err := db.Add(v); err != nil {
			t.Fatal(err)
		}
	}

	query := vec(rng, testDim)

	tagged, err := db.Search(SearchRequest{Query: query, K: 10, Filter: Exists("tagged")})
	if err != nil {
		t.Fatal(err)
	}
	if len(tagged) != 10 {
		t.Fatalf("got %d tagged results, want 10", len(tagged))
	}

	untagged, err := db.Search(SearchRequest{Query: query, K: 10, Filter: Not(Exists("tagged"))})
	if err != nil {
		t.Fatal(err)
	}
	if len(untagged) != 10 {
		t.Fatalf("got %d untagged results, want 10", len(untagged))
	}
	for _, m := range untagged {
		if m.Metadata != nil {
			t.Fatalf("%q carries metadata %+v but matched Not(Exists)", m.ID, m.Metadata)
		}
	}
}

// A caller's own predicate is a supported thing to pass, which is the point of
// Filter being an interface rather than a closed set of constructors.
type oddIDs struct{}

func (oddIDs) Match(md Metadata) bool {
	n, ok := md["n"].(int64)
	return ok && n%2 == 1
}

func (oddIDs) Validate() error { return nil }

func TestCallerImplementedFilter(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 200, 29)
	rng := rand.New(rand.NewSource(31))

	res, err := db.Search(SearchRequest{Query: vec(rng, testDim), K: 10, Filter: oddIDs{}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 10 {
		t.Fatalf("got %d results, want 10", len(res))
	}
	for _, m := range res {
		if m.Metadata["n"].(int64)%2 != 1 {
			t.Fatalf("%q is not odd: %+v", m.ID, m.Metadata)
		}
	}
}

// A caller's Validate is honoured, so a custom filter can refuse to run for the
// same reasons the built-in ones do.
type brokenFilter struct{}

func (brokenFilter) Match(Metadata) bool { return true }

func (brokenFilter) Validate() error { return errors.New("deliberately broken") }

func TestCallerFilterValidationIsHonoured(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 20, 37)
	rng := rand.New(rand.NewSource(41))

	_, err := db.Search(SearchRequest{Query: vec(rng, testDim), K: 5, Filter: brokenFilter{}})
	if !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("error = %v, want ErrInvalidFilter", err)
	}
}
