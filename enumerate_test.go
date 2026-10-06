package govecdb

import (
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"
)

func TestGetBatchKeepsOrderAndSkipsTheMissing(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 20, 1)

	got, err := db.GetBatch([]string{"v7", "nope", "v3", "v19", "also-nope"})
	if err != nil {
		t.Fatal(err)
	}
	if ids := vectorIDs(got); !slices.Equal(ids, []string{"v7", "v3", "v19"}) {
		t.Fatalf("GetBatch returned %v", ids)
	}
	// Values and metadata are the stored ones, not just the ids.
	if got[0].Metadata["n"] != int64(7) || len(got[0].Values) != testDim {
		t.Fatalf("v7 came back as %+v", got[0])
	}

	if got, err := db.GetBatch(nil); err != nil || len(got) != 0 {
		t.Fatalf("empty batch: %v, %v", got, err)
	}
}

func TestGetBatchRefusesAnOversizedBatch(t *testing.T) {
	db, _ := openDB(t, WithLimits(0, 0, 0, 4, 0))
	if _, err := db.GetBatch(make([]string, 5)); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("GetBatch over maxBatch = %v, want ErrInvalidRequest", err)
	}
}

// TestScanPagesThroughEverythingOnce walks the whole database by cursor, the way
// an HTTP client would, and checks it saw every live vector exactly once and in
// order — including after deletes, which leave tombstones the walk must skip.
func TestScanPagesThroughEverythingOnce(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 103, 2)
	for i := 0; i < 103; i += 10 {
		if err := db.Delete(fmt.Sprintf("v%d", i)); err != nil {
			t.Fatal(err)
		}
	}

	var seen []string
	after := ""
	for pages := 0; ; pages++ {
		if pages > 100 {
			t.Fatal("Scan never ran out")
		}
		page, err := db.Scan(after, 7)
		if err != nil {
			t.Fatal(err)
		}
		seen = append(seen, vectorIDs(page)...)
		if len(page) < 7 {
			break
		}
		after = page[len(page)-1].ID
	}

	if len(seen) != db.Len() {
		t.Fatalf("Scan saw %d vectors, the database holds %d", len(seen), db.Len())
	}
	if !slices.IsSorted(seen) {
		t.Fatal("Scan pages are not in id order")
	}
	if slices.Contains(seen, "v0") || slices.Contains(seen, "v100") {
		t.Fatal("Scan returned a deleted vector")
	}
	if len(slices.Compact(slices.Clone(seen))) != len(seen) {
		t.Fatal("Scan returned an id twice")
	}
}

func TestScanRejects(t *testing.T) {
	db, _ := openDB(t, WithLimits(0, 0, 0, 10, 0))
	for _, limit := range []int{0, -1, 11} {
		if _, err := db.Scan("", limit); !errors.Is(err, ErrInvalidRequest) {
			t.Fatalf("Scan(limit=%d) = %v, want ErrInvalidRequest", limit, err)
		}
	}
}

func TestRangeVisitsEveryVectorInOrder(t *testing.T) {
	db, _ := openDB(t)
	// More than one page, so the page boundary is exercised.
	fill(t, db, rangePage*2+5, 3)

	var seen []string
	if err := db.Range(func(v Vector) bool {
		if len(v.Values) != testDim {
			t.Fatalf("%s has %d values", v.ID, len(v.Values))
		}
		seen = append(seen, v.ID)
		return true
	}); err != nil {
		t.Fatal(err)
	}
	if len(seen) != db.Len() || !slices.IsSorted(seen) {
		t.Fatalf("Range saw %d (sorted=%v), want %d sorted", len(seen), slices.IsSorted(seen), db.Len())
	}

	calls := 0
	if err := db.Range(func(Vector) bool { calls++; return calls < 3 }); err != nil {
		t.Fatal(err)
	}
	if calls != 3 {
		t.Fatalf("Range kept going after fn returned false: %d calls", calls)
	}
}

// TestRangeCallbackMayWrite is the reason fn runs with no lock held: a rebuild
// copies vectors out and writes them somewhere, and "somewhere" is allowed to be
// this database.
func TestRangeCallbackMayWrite(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 50, 4)

	err := db.Range(func(v Vector) bool {
		v.Metadata = Metadata{"copied": true}
		if err := db.Add(v); err != nil {
			t.Error(err)
			return false
		}
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	if v := mustGet(t, db, "v49"); v.Metadata["copied"] != true {
		t.Fatalf("write inside Range did not land: %+v", v.Metadata)
	}
}

func TestEnumerationOnAClosedDatabase(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 5, 5)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := db.GetBatch([]string{"v1"}); !errors.Is(err, ErrClosed) {
		t.Fatalf("GetBatch = %v", err)
	}
	if _, err := db.Scan("", 1); !errors.Is(err, ErrClosed) {
		t.Fatalf("Scan = %v", err)
	}
	if err := db.Range(func(Vector) bool { return true }); !errors.Is(err, ErrClosed) {
		t.Fatalf("Range = %v", err)
	}
}

// TestRangeAgainstConcurrentWriters is for the race detector: pages are read
// under short locks while writers keep going, and every vector handed out must
// be whole.
func TestRangeAgainstConcurrentWriters(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 600, 6)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range 300 {
			_ = db.Delete(fmt.Sprintf("v%d", i))
		}
	}()
	err := db.Range(func(v Vector) bool {
		if len(v.Values) != testDim || v.Metadata == nil {
			t.Errorf("torn vector %s: %d values, metadata %v", v.ID, len(v.Values), v.Metadata)
		}
		return true
	})
	wg.Wait()
	if err != nil {
		t.Fatal(err)
	}
}

func vectorIDs(vs []Vector) []string {
	out := make([]string, len(vs))
	for i, v := range vs {
		out[i] = v.ID
	}
	return out
}
