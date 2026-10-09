package govecdb

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
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
//
// The race-safe check is differential. A reader may still wait for one apply,
// and under -race an apply costs ~10× (1.5 ms against 0.15 ms here, more on a
// CI runner) — enough to break any fixed slack while waiting on nothing but
// the index. So the same searches also run against a SyncNever writer, which
// applies at least as often and never fsyncs: if searches waited on the disk,
// the SyncAlways p50 would sit a whole fsync above that one; if they wait only
// on applies, the two agree on any machine.
func TestSearchDoesNotWaitForASyncAlwaysWriter(t *testing.T) {
	const dim = 32
	measure := func(policy SyncPolicy) (base, under []time.Duration, n int, each time.Duration) {
		db, _ := openConcurrent(t, dim, WithSyncPolicy(policy))
		loadRandom(t, db, 2000, dim, 1)
		base = searchLatencies(t, db, dim, 300, 2)

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
		under = searchFor(t, db, dim, 300*time.Millisecond, 4)
		close(stop)
		n = <-writes
		return base, under, n, time.Since(start) / time.Duration(max(n, 1))
	}

	base, under, n, each := measure(SyncAlways)
	_, neverUnder, neverN, neverEach := measure(SyncNever)
	t.Logf("no writer: p50 %v p99 %v; SyncAlways writer (%d adds, ~%v each): %d searches, p50 %v p99 %v; "+
		"SyncNever writer (%d adds, ~%v each): p50 %v",
		pct(base, .5), pct(base, .99), n, each, len(under), pct(under, .5), pct(under, .99),
		neverN, neverEach, pct(neverUnder, .5))
	if n == 0 {
		t.Fatal("the writer never completed an Add")
	}

	// Waiting on the fsync would put SyncAlways a full fsync above SyncNever.
	// Twice the SyncNever p50 plus a millisecond is well below that on any disk
	// where fsync is slow enough to matter, and holds under -race.
	if limit := 2*pct(neverUnder, .5) + time.Millisecond; pct(under, .5) > limit {
		t.Fatalf("search p50 under a SyncAlways writer is %v, over %v (twice the SyncNever-writer p50 plus 1ms): "+
			"searches are waiting on the writer's fsync", pct(under, .5), limit)
	}
	// Under -race an insert costs as much as the fsync, so absolute bounds
	// against the idle baseline say nothing about the log.
	if raceDetectorEnabled {
		return
	}
	// The stall this guards was the whole fsync on every search. Twice the
	// baseline plus a millisecond of scheduling slack is far below it on any
	// disk where fsync is slow enough to matter. The slack also covers the one
	// thing a search may still wait for, an index insert.
	if limit := 2*pct(base, .5) + time.Millisecond; pct(under, .5) > limit {
		t.Fatalf("search p50 under a SyncAlways writer is %v, over %v: searches are waiting on the writer's fsync",
			pct(under, .5), limit)
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

// tempSnapshotAppears waits until a snapshot is being written, which is when
// its temporary file exists. It reports false if done closes first.
func tempSnapshotAppears(dir string, done <-chan struct{}) bool {
	snaps := filepath.Join(dir, snapshotSubdir)
	for {
		select {
		case <-done:
			return false
		default:
		}
		entries, _ := os.ReadDir(snaps)
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".tmp") {
				return true
			}
		}
	}
}

// TestSearchDuringSnapshotIsNotStalledByAQueuedAdd: the snapshot holds the
// graph's read lock while it serializes. A writer used to wait for that lock
// while holding the database lock, so every search queued behind the writer
// for the rest of the snapshot. Writers now wait for a snapshot without holding
// anything a search needs.
func TestSearchDuringSnapshotIsNotStalledByAQueuedAdd(t *testing.T) {
	const dim = 128
	db, dir := openConcurrent(t, dim)
	loadRandom(t, db, 10000, dim, 7)
	rng := rand.New(rand.NewSource(8))

	for attempt := range 5 {
		// Something must change, or the snapshot is skipped as a no-op.
		if err := db.Add(Vector{ID: fmt.Sprintf("s%d", attempt), Values: vec(rng, dim)}); err != nil {
			t.Fatal(err)
		}
		snapDone := make(chan struct{})
		var snapTook time.Duration
		go func() {
			start := time.Now()
			if err := db.Snapshot(); err != nil {
				t.Error(err)
			}
			snapTook = time.Since(start)
			close(snapDone)
		}()
		if !tempSnapshotAppears(dir, snapDone) {
			continue // finished before it could be observed; try again
		}

		var added atomic.Bool
		addDone := make(chan struct{})
		go func() {
			if err := db.Add(Vector{ID: fmt.Sprintf("q%d", attempt), Values: vec(rand.New(rand.NewSource(int64(attempt))), dim)}); err != nil {
				t.Error(err)
			}
			added.Store(true)
			close(addDone)
		}()
		time.Sleep(200 * time.Microsecond) // let the Add queue

		during := 0
		snapFinished := func() bool {
			select {
			case <-snapDone:
				return true
			default:
				return false
			}
		}
		for !snapFinished() && !added.Load() {
			searchOnce(t, db, vec(rng, dim))
			if !added.Load() && !snapFinished() {
				during++
			}
		}
		<-snapDone
		<-addDone
		if added.Load() && during == 0 && !snapFinished() {
			continue
		}
		t.Logf("snapshot took %v; %d searches completed while an Add was queued behind it", snapTook, during)
		if during < 3 {
			t.Fatalf("only %d searches completed while an Add waited on a %v snapshot: searches are queued behind the writer", during, snapTook)
		}
		return
	}
	t.Skip("snapshot was never observed in progress")
}

// TestConcurrentSnapshotsAllSucceed: two snapshots overlapping used to make one
// fail, because the first one's prune deleted the second one's temporary file.
func TestConcurrentSnapshotsAllSucceed(t *testing.T) {
	const dim = 64
	db, dir := openConcurrent(t, dim)
	loadRandom(t, db, 3000, dim, 9)

	var wg sync.WaitGroup
	errs := make(chan error, 10)
	for i := range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(100 + i)))
			if err := db.Add(Vector{ID: fmt.Sprintf("c%d", i), Values: vec(rng, dim)}); err != nil {
				errs <- err
				return
			}
			errs <- db.Snapshot()
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Error(err)
		}
	}

	db = reopen(t, db, dir, WithDimension(dim), WithEfCalibration(false))
	if got := db.Len(); got != 3010 {
		t.Fatalf("Len after reopen = %d, want 3010", got)
	}
}

// TestCloseWaitsForAManualSnapshot: Close used to wait only for the interval
// snapshotter, so a Snapshot a caller had started kept pruning snapshots and
// truncating the log after Close had released the directory to the next Open.
func TestCloseWaitsForAManualSnapshot(t *testing.T) {
	const dim = 128
	var taken atomic.Int32
	db, dir := openConcurrent(t, dim, WithObserver(func(e Event) {
		if _, ok := e.(SnapshotTaken); ok {
			taken.Add(1)
		}
	}))
	loadRandom(t, db, 10000, dim, 10)

	snapErr := make(chan error, 1)
	snapDone := make(chan struct{})
	go func() {
		snapErr <- db.Snapshot()
		close(snapDone)
	}()
	if !tempSnapshotAppears(dir, snapDone) {
		t.Skip("snapshot finished before it could be observed in progress")
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	// Close has returned, so the snapshot must be over — complete, not merely
	// started — and nothing of it may be left in the directory.
	if taken.Load() != 1 {
		t.Fatal("Close returned while a Snapshot was still in flight")
	}
	if err := <-snapErr; err != nil && !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(filepath.Join(dir, snapshotSubdir))
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Fatalf("temporary snapshot %s left behind after Close", e.Name())
		}
	}
	db2 := openDBAt(t, dir, WithDimension(dim), WithEfCalibration(false))
	if s := db2.Stats(); s.SnapshotSeq == 0 || s.Live != 10000 {
		t.Fatalf("after reopen: %+v", s)
	}
}

// TestSnapshotSkipsWhenNothingChanged: an idle database on a timer rewrote
// and re-verified its whole index every interval. A snapshot with nothing new
// since the last one now writes nothing — unless a compaction changed the
// graph, which no log record stands for.
func TestSnapshotSkipsWhenNothingChanged(t *testing.T) {
	var taken atomic.Int32
	observe := func(e Event) {
		if _, ok := e.(SnapshotTaken); ok {
			taken.Add(1)
		}
	}
	db, dir := openDB(t, WithObserver(observe))
	fill(t, db, 50, 13)

	snap := func(want int32) {
		t.Helper()
		if err := db.Snapshot(); err != nil {
			t.Fatal(err)
		}
		if got := taken.Load(); got != want {
			t.Fatalf("%d snapshots taken, want %d", got, want)
		}
	}
	snap(1)
	snap(1) // nothing changed
	if err := db.Delete("v3"); err != nil {
		t.Fatal(err)
	}
	snap(2)
	if n, err := db.Compact(); err != nil || n == 0 {
		t.Fatalf("Compact = %d, %v", n, err)
	}
	snap(3) // compaction changed the graph without a log record
	snap(3)

	// A reopened database knows what its snapshot covers.
	db = reopen(t, db, dir, WithObserver(observe))
	snap(3)
	if s := db.Stats(); s.SnapshotSeq != s.LastSeq {
		t.Fatalf("SnapshotSeq %d, LastSeq %d", s.SnapshotSeq, s.LastSeq)
	}
}

// TestCompactKeepsSearchesRunningWhileWritersWait: DB.Compact holds writeMu,
// so no write reaches the graph during the rebuild — the index's fast path,
// building under its read lock, always holds, and searches run throughout.
// Writers wait for it instead.
func TestCompactKeepsSearchesRunningWhileWritersWait(t *testing.T) {
	const dim = 32
	db, _ := openConcurrent(t, dim)
	loadRandom(t, db, 6000, dim, 14)
	for i := 0; i < 6000; i += 2 {
		if err := db.Delete(fmt.Sprintf("v%d", i)); err != nil {
			t.Fatal(err)
		}
	}

	// The writer records when each Add returned; owned by its goroutine until
	// wg.Wait.
	stop := make(chan struct{})
	var finished []time.Time
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		rng := rand.New(rand.NewSource(15))
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			if err := db.Add(Vector{ID: fmt.Sprintf("n%d", i), Values: vec(rng, dim)}); err != nil {
				t.Error(err)
				return
			}
			finished = append(finished, time.Now())
		}
	}()
	time.Sleep(5 * time.Millisecond)

	var running atomic.Bool
	running.Store(true)
	type result struct {
		n    int
		took time.Duration
		err  error
	}
	done := make(chan result)
	go func() {
		start := time.Now()
		n, err := db.Compact()
		running.Store(false)
		done <- result{n, time.Since(start), err}
	}()

	rng := rand.New(rand.NewSource(16))
	during := 0
	for running.Load() {
		searchOnce(t, db, vec(rng, dim))
		if running.Load() {
			during++
		}
	}
	r := <-done
	close(stop)
	wg.Wait()

	var gap time.Duration
	for i := 1; i < len(finished); i++ {
		gap = max(gap, finished[i].Sub(finished[i-1]))
	}
	t.Logf("Compact reclaimed %d in %v; %d searches completed during it; longest writer wait %v",
		r.n, r.took, during, gap)
	if r.err != nil {
		t.Fatal(r.err)
	}
	if r.n != 3000 {
		t.Fatalf("Compact reclaimed %d, want 3000", r.n)
	}
	if during < 3 {
		t.Fatalf("only %d searches completed during Compact: it is excluding readers", during)
	}
	// Writers are excluded for the rebuild, so the writer has one wait about
	// as long as the rebuild itself. Half allows for the time Compact spent
	// queued for writeMu before it got it.
	if gap < r.took/2 {
		t.Fatalf("longest writer wait %v during a %v Compact: writers are not excluded", gap, r.took)
	}
}
