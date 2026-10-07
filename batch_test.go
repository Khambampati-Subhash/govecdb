package govecdb

import (
	"errors"
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// TestAddBatchInParallelHoldsTheSameData: a parallel build is a different
// graph, but the same database — every id, vector and metadata value as a
// serial one, with the last of a duplicated id winning both.
func TestAddBatchInParallelHoldsTheSameData(t *testing.T) {
	rng := rand.New(rand.NewSource(801))
	vs := make([]Vector, 600)
	for i := range vs {
		vs[i] = Vector{ID: fmt.Sprintf("v%d", i%500), Values: vec(rng, testDim), Metadata: Metadata{"i": int64(i)}}
	}

	serial, _ := openDB(t, WithInsertWorkers(1))
	parallel, _ := openDB(t, WithInsertWorkers(8))
	for _, db := range []*DB{serial, parallel} {
		if err := db.AddBatch(vs); err != nil {
			t.Fatal(err)
		}
	}
	if serial.Len() != 500 || parallel.Len() != 500 {
		t.Fatalf("Len serial %d parallel %d, want 500", serial.Len(), parallel.Len())
	}
	for i := 100; i < 600; i++ { // the last occurrence of every id
		want := vs[i]
		got := mustGet(t, parallel, want.ID)
		ref := mustGet(t, serial, want.ID)
		if !slices.Equal(got.Values, ref.Values) || got.Metadata["i"] != int64(i) {
			t.Fatalf("%s = %v / %v, want the last write (i=%d) as serial holds it", want.ID, got.Metadata, got.Values[:2], i)
		}
	}
	// Every vector finds itself.
	for _, v := range vs[100:] {
		m, err := parallel.Search(SearchRequest{Query: v.Values, K: 1})
		if err != nil || len(m) == 0 || m[0].ID != v.ID {
			t.Fatalf("searching for %s found %v (%v)", v.ID, m, err)
		}
	}
}

// TestReplayKeepsLogOrderAcrossBatches: replay gathers PUTs and flushes at a
// DELETE. An id written, deleted and written again — around the flush — must
// end as the log says, as must one deleted last.
func TestReplayKeepsLogOrderAcrossBatches(t *testing.T) {
	db, dir := openDB(t)
	rng := rand.New(rand.NewSource(802))
	vs := make([]Vector, 200)
	for i := range vs {
		vs[i] = Vector{ID: fmt.Sprintf("v%d", i), Values: vec(rng, testDim)}
	}
	if err := db.AddBatch(vs); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 50; i++ {
		if err := db.Delete(vs[i].ID); err != nil {
			t.Fatal(err)
		}
	}
	back := Vector{ID: "v0", Values: vec(rng, testDim), Metadata: Metadata{"back": true}}
	if err := db.Add(back); err != nil {
		t.Fatal(err)
	}
	moved := Vector{ID: "v60", Values: vec(rng, testDim)}
	if err := db.AddBatch([]Vector{moved, vs[61], moved}); err != nil {
		t.Fatal(err)
	}
	if err := db.Delete("v199"); err != nil {
		t.Fatal(err)
	}
	want := db.Len()
	movedStored := mustGet(t, db, "v60").Values

	db = reopen(t, db, dir)
	if db.Len() != want || want != 200-50+1-1 {
		t.Fatalf("Len after replay = %d, before %d, want 150", db.Len(), want)
	}
	if got := mustGet(t, db, "v0"); got.Metadata["back"] != true {
		t.Fatalf("v0 = %v, want the write after its delete", got.Metadata)
	}
	if got := mustGet(t, db, "v60"); !slices.Equal(got.Values, movedStored) {
		t.Fatal("v60 after replay is not the vector it held before the restart")
	}
	for _, id := range []string{"v1", "v49", "v199"} {
		if _, err := db.Get(id); !errors.Is(err, ErrNotFound) {
			t.Fatalf("Get(%s) after replay = %v, want ErrNotFound", id, err)
		}
	}
}

func TestInsertWorkersRejectsNegative(t *testing.T) {
	if _, err := Open(t.TempDir(), WithDimension(testDim), WithInsertWorkers(-1)); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("err = %v, want ErrInvalidConfig", err)
	}
}
