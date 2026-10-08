package govecdb

import (
	"fmt"
	"math/rand"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Readers against writers. Every test here measures one stall that existed:
// a search waiting on a writer's fsync, on a whole batch, or on a snapshot a
// queued writer was waiting behind. They assert causality — searches finished
// while the writer was still busy — rather than absolute latencies, because a
// loaded CI machine makes every absolute number a flake.

// openConcurrent opens a database of dimension dim with calibration off, so no
// background goroutine competes with what is being measured.
func openConcurrent(t *testing.T, dim int, opts ...Option) (*DB, string) {
	t.Helper()
	all := append([]Option{WithDimension(dim), WithEfCalibration(false)}, opts...)
	return openDB(t, all...)
}

// loadRandom adds n random vectors of dimension dim in batches.
func loadRandom(t *testing.T, db *DB, n, dim int, seed int64) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	for lo := 0; lo < n; lo += 5000 {
		vs := make([]Vector, min(5000, n-lo))
		for i := range vs {
			vs[i] = Vector{ID: fmt.Sprintf("v%d", lo+i), Values: vec(rng, dim), Metadata: Metadata{"n": int64(lo + i)}}
		}
		if err := db.AddBatch(vs); err != nil {
			t.Fatal(err)
		}
	}
}

func searchOnce(t *testing.T, db *DB, q []float32) time.Duration {
	t.Helper()
	start := time.Now()
	if _, err := db.Search(SearchRequest{Query: q, K: 10, Ef: 64}); err != nil {
		t.Error(err)
	}
	return time.Since(start)
}

func searchLatencies(t *testing.T, db *DB, dim, n int, seed int64) []time.Duration {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	out := make([]time.Duration, n)
	for i := range out {
		out[i] = searchOnce(t, db, vec(rng, dim))
	}
	slices.Sort(out)
	return out
}

// searchFor searches back to back for d and returns the sorted latencies.
func searchFor(t *testing.T, db *DB, dim int, d time.Duration, seed int64) []time.Duration {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	var out []time.Duration
	for end := time.Now().Add(d); time.Now().Before(end); {
		out = append(out, searchOnce(t, db, vec(rng, dim)))
	}
	slices.Sort(out)
	return out
}

func pct(sorted []time.Duration, p float64) time.Duration {
	return sorted[min(len(sorted)-1, int(p*float64(len(sorted))))]
}

// TestSearchDoesNotWaitForASyncAlwaysWriter: a writer under SyncAlways spends
// nearly all of every Add in fsync. When the database lock was held across it,
// a search queued behind every one — p50 went from 40 µs to 4.9 ms, the fsync
// time. Searches now wait at most for the index insert, never for the disk.
func TestSearchDoesNotWaitForASyncAlwaysWriter(t *testing.T) {
	const dim = 32
	db, _ := openConcurrent(t, dim, WithSyncPolicy(SyncAlways))
	loadRandom(t, db, 2000, dim, 1)

	base := searchLatencies(t, db, dim, 300, 2)

	stop := make(chan struct{})
	writes := make(chan int)
	go func() {
		rng := rand.New(rand.NewSource(3))
		n := 0
		for {
			select {
			case <-stop:
				writes <- n
				return
			default:
			}
			if err := db.Add(Vector{ID: fmt.Sprintf("w%d", n), Values: vec(rng, dim)}); err != nil {
				t.Error(err)
			}
			n++
		}
	}()
	start := time.Now()
	time.Sleep(20 * time.Millisecond)
	under := searchFor(t, db, dim, 300*time.Millisecond, 4)
	close(stop)
	n := <-writes
	elapsed := time.Since(start)

	t.Logf("no writer: p50 %v p99 %v; SyncAlways writer (%d adds, ~%v each): %d searches, p50 %v p99 %v",
		pct(base, .5), pct(base, .99), n, elapsed/time.Duration(max(n, 1)), len(under), pct(under, .5), pct(under, .99))
	if n == 0 {
		t.Fatal("the writer never completed an Add")
	}
	// The stall this guards was the whole fsync on every search. Twice the
	// baseline plus a millisecond of scheduling slack is far below it on any
	// disk where fsync is slow enough to matter. The slack also covers the one
	// thing a search may still wait for, an index insert.
	if limit := 2*pct(base, .5) + time.Millisecond; pct(under, .5) > limit {
		t.Fatalf("search p50 under a SyncAlways writer is %v, over %v: searches are waiting on the writer's fsync",
			pct(under, .5), limit)
	}
	// Under -race an insert costs as much as the fsync, so the tail is the
	// index's write lock and says nothing about the log.
	if raceDetectorEnabled {
		return
	}
	if limit := 2*pct(base, .99) + time.Millisecond; pct(under, .99) > limit {
		t.Fatalf("search p99 under a SyncAlways writer is %v, over %v: searches are waiting on the writer's fsync",
			pct(under, .99), limit)
	}
}

// TestSearchDuringAddBatchDoesNotWaitForTheBatch: a search issued during a
// large batch used to wait for all of it (545 ms of a 565 ms batch). The index
// still takes its write lock per chunk, so a search can wait for one chunk —
// what it must not do is wait for the batch.
func TestSearchDuringAddBatchDoesNotWaitForTheBatch(t *testing.T) {
	const dim, preload, batch = 64, 5000, 10000
	db, _ := openConcurrent(t, dim)
	loadRandom(t, db, preload, dim, 5)

	rng := rand.New(rand.NewSource(6))
	vs := make([]Vector, batch)
	for i := range vs {
		vs[i] = Vector{ID: fmt.Sprintf("b%d", i), Values: vec(rng, dim)}
	}

	var finished atomic.Bool
	done := make(chan time.Duration)
	go func() {
		start := time.Now()
		if err := db.AddBatch(vs); err != nil {
			t.Error(err)
		}
		finished.Store(true)
		done <- time.Since(start)
	}()

	// Wait until the batch is visibly being applied.
	for db.Len() == preload && !finished.Load() {
		time.Sleep(50 * time.Microsecond)
	}
	during, worst := 0, time.Duration(0)
	for !finished.Load() {
		d := searchOnce(t, db, vec(rng, dim))
		if !finished.Load() {
			during++
			worst = max(worst, d)
		}
	}
	took := <-done

	t.Logf("AddBatch(%d) took %v; %d searches completed inside it, slowest %v", batch, took, during, worst)
	if during < 3 {
		t.Fatalf("only %d searches completed during a %v batch: searches are waiting for the whole batch", during, took)
	}
}

// TestReadersNeverSeeATornPut: a PUT replacing a vector updates the index and
// then the metadata store. A reader between the two would see the new vector
// with the old metadata — a pairing no write ever made. Readers never wait on
// a writer's fsync now, so this is held by applyMu alone, for single writes
// and for batches applied in groups.
//
// The real window is nanoseconds, so the index is wrapped to hold it open: it
// pauses after every insert, between the index update and the metadata one.
func TestReadersNeverSeeATornPut(t *testing.T) {
	const dim = 8
	db, _ := openConcurrent(t, dim)
	loadRandom(t, db, 500, dim, 11)
	db.index = slowInsertIndex{db.index.(*hnswIndex)}

	unit := func(i int) []float32 {
		v := make([]float32, dim)
		v[i] = 1
		return v
	}
	versions := []Vector{
		{ID: "x", Values: unit(0), Metadata: Metadata{"which": "a"}},
		{ID: "x", Values: unit(1), Metadata: Metadata{"which": "b"}},
	}
	check := func(v Vector) {
		want := "a"
		if v.Values[1] == 1 {
			want = "b"
		}
		if v.Metadata["which"] != want {
			t.Errorf("x has the vector of %q with the metadata of %v", want, v.Metadata["which"])
		}
	}
	if err := db.Add(versions[0]); err != nil {
		t.Fatal(err)
	}

	stop := make(chan struct{})
	var writes atomic.Int64
	reads := 0
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		rng := rand.New(rand.NewSource(12))
		for i := 0; ; i++ {
			writes.Store(int64(i))
			select {
			case <-stop:
				return
			default:
			}
			v := versions[i%2]
			var err error
			if i%3 == 0 {
				// x last in a batch large enough to be applied in groups, so the
				// metadata puts ahead of it make the window a torn read needs.
				batch := make([]Vector, 300)
				for j := range batch {
					batch[j] = Vector{ID: fmt.Sprintf("f%d", j), Values: vec(rng, dim)}
				}
				batch[len(batch)-1] = v
				err = db.AddBatch(batch)
			} else {
				err = db.Add(v)
			}
			if err != nil {
				t.Error(err)
				return
			}
		}
	}()

	for end := time.Now().Add(300 * time.Millisecond); time.Now().Before(end); {
		v, err := db.Get("x")
		if err != nil {
			t.Fatal(err)
		}
		check(v)
		reads++
		vs, err := db.GetBatch([]string{"f1", "x"})
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range vs {
			if v.ID == "x" {
				check(v)
			}
		}
	}
	close(stop)
	wg.Wait()
	t.Logf("%d reads against %d writes", reads, writes.Load())
}

// slowInsertIndex pauses after each insert returns, which is where a reader
// not excluded by applyMu would find the new vector beside the old metadata.
type slowInsertIndex struct{ *hnswIndex }

func (s slowInsertIndex) Insert(id string, values []float32) error {
	err := s.hnswIndex.Insert(id, values)
	time.Sleep(200 * time.Microsecond)
	return err
}

func (s slowInsertIndex) InsertBatch(ids []string, values [][]float32, workers int) error {
	err := s.hnswIndex.InsertBatch(ids, values, workers)
	time.Sleep(200 * time.Microsecond)
	return err
}
