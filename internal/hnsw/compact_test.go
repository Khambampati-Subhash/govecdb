package hnsw

import (
	"fmt"
	"maps"
	"math/rand"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
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

// TestConcurrentCompactAndSearch runs a compaction against live readers. The
// guarantee tested here is not that readers keep running (see
// TestCompactDoesNotStopSearches) — it is that they never observe the swap:
// the live population is identical before and after, and no reader may see it
// change.
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

// TestCompactDoesNotStopSearches: the replacement is built outside the graph's
// lock, so searches keep running for the whole rebuild. It used to hold the
// write lock throughout, and no search ran until it finished.
func TestCompactDoesNotStopSearches(t *testing.T) {
	const n, dim = 4000, 32
	g, _ := buildGraph(t, n, dim, 61)
	for i := range n {
		if i%2 == 0 {
			g.Delete(fmt.Sprintf("v%d", i))
		}
	}

	var running atomic.Bool
	running.Store(true)
	done := make(chan int)
	go func() {
		got := g.Compact()
		running.Store(false)
		done <- got
	}()

	rng := rand.New(rand.NewSource(62))
	during := 0
	for running.Load() {
		if _, err := g.Search(randomVector(rng, dim), 10, 64); err != nil {
			t.Fatal(err)
		}
		if running.Load() {
			during++
		}
	}
	if got := <-done; got != n/2 {
		t.Fatalf("Compact reclaimed %d, want %d", got, n/2)
	}
	t.Logf("%d searches completed during Compact", during)
	if during < 3 {
		t.Fatalf("only %d searches completed during Compact: it is excluding readers", during)
	}
}

// TestCompactUnderConcurrentWritesLosesNothing exercises the catch-up: a write
// that lands during the build, or between a catch-up round and the swap, must
// not be thrown away with the old graph. A writer hammers the graph while
// Compact runs in a loop, and every write must survive.
func TestCompactUnderConcurrentWritesLosesNothing(t *testing.T) {
	const dim = 16
	g, _ := buildGraph(t, 200, dim, 63)

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				g.Compact()
			}
		}
	}()

	rng := rand.New(rand.NewSource(64))
	want := make(map[string][]float32)
	for i := range 400 {
		id := fmt.Sprintf("w%d", i%150)
		if i%5 == 4 {
			g.Delete(id)
			delete(want, id)
			continue
		}
		v := randomVector(rng, dim)
		if err := g.Insert(id, v); err != nil {
			t.Fatal(err)
		}
		want[id] = g.prepare(v)
	}
	close(stop)
	wg.Wait()

	for id, v := range want {
		got, ok := g.Vector(id)
		if !ok {
			t.Fatalf("%s was written and is gone after a concurrent Compact", id)
		}
		if !slices.Equal(got, v) {
			t.Fatalf("%s holds a stale vector after a concurrent Compact", id)
		}
	}
	if live := g.Len(); live != 200+len(want) {
		t.Fatalf("Len = %d, want %d", live, 200+len(want))
	}
}

// TestCompactLetsWritesLand: a write issued while the replacement is being
// built returns without waiting for the build, and the compacted graph holds
// it. A hook holds the build open, so nothing here depends on how long a
// rebuild takes. Every kind of write is covered, because each reaches the
// catch-up differently: a new id, a changed vector, an unchanged one (which
// must cost no slot), a delete of a changed id and of an untouched one, an id
// inserted and deleted again, and one deleted and inserted again.
func TestCompactLetsWritesLand(t *testing.T) {
	const n, dim = 600, 16
	g, data := buildGraph(t, n, dim, 65)
	want := make(map[string][]float32, n)
	for id, v := range data {
		want[id] = g.prepare(v)
	}
	for i := 0; i < n; i += 2 {
		id := fmt.Sprintf("v%d", i)
		g.Delete(id)
		delete(want, id)
	}
	captured := maps.Clone(want)

	building, release := make(chan struct{}), make(chan struct{})
	g.onCompactBuild = func() {
		close(building)
		<-release
	}
	done := make(chan int)
	go func() { done <- g.Compact() }()
	<-building

	// changed is every id the replacement was built with and a write then
	// moved: each costs the replacement one tombstone, and nothing else may.
	changed := make(map[string]bool)
	rng := rand.New(rand.NewSource(66))
	put := func(id string) {
		v := randomVector(rng, dim)
		if err := g.Insert(id, v); err != nil {
			t.Error(err)
		}
		want[id] = g.prepare(v)
		if _, ok := captured[id]; ok {
			changed[id] = true
		}
	}
	del := func(id string) {
		g.Delete(id)
		delete(want, id)
		if _, ok := captured[id]; ok {
			changed[id] = true
		}
	}
	wrote := make(chan struct{})
	go func() {
		defer close(wrote)
		for i := range 100 { // past minParallelBatch, so the catch-up takes the batch path
			put(fmt.Sprintf("new%d", i))
		}
		for i := 1; i < 200; i += 4 {
			put(fmt.Sprintf("v%d", i))
		}
		if err := g.Insert("v3", data["v3"]); err != nil {
			t.Error(err)
		}
		for i := 5; i < 200; i += 8 { // every one of these was just changed
			del(fmt.Sprintf("v%d", i))
		}
		for i := 11; i < 200; i += 8 { // and none of these was
			del(fmt.Sprintf("v%d", i))
		}
		put("brief")
		del("brief")
		del("v7")
		put("v7")
	}()
	select {
	case <-wrote:
	case <-time.After(30 * time.Second):
		close(release)
		t.Fatal("writes issued during the build waited for it")
	}
	close(release)
	reclaimed := <-done

	for id, v := range want {
		got, ok := g.Vector(id)
		if !ok {
			t.Fatalf("%s was written during the build and is gone after it", id)
		}
		if !slices.Equal(got, v) {
			t.Fatalf("%s holds a stale vector after the compaction", id)
		}
	}
	g.IDs(func(id string) bool {
		if _, ok := want[id]; !ok {
			t.Errorf("%s was deleted during the build and survived it", id)
		}
		return true
	})
	s := g.Stats()
	t.Logf("reclaimed %d; %d ids changed under the build; after: %+v", reclaimed, len(changed), s)
	if s.Live != len(want) || s.Deleted != len(changed) {
		t.Fatalf("after compaction %+v: want %d live and %d dead (one per changed id)", s, len(want), len(changed))
	}
	if reclaimed < n/2 {
		t.Fatalf("reclaimed %d, want at least the %d dead when it started", reclaimed, n/2)
	}
	assertWellFormed(t, g)
	if u := unreachable(g); u != 0 {
		t.Fatalf("%d live vectors unreachable after catching up", u)
	}
}

// TestCompactOnlineQuiescesOnceAtTheEnd pins the contract a database leans on
// to keep its own writers out of the swap: quiesce runs once, after the build,
// and what it returns runs after the swap — never when there is nothing to do.
func TestCompactOnlineQuiescesOnceAtTheEnd(t *testing.T) {
	g, _ := buildGraph(t, 200, 16, 67)
	if got := g.CompactOnline(func() func() {
		t.Fatal("quiesced a compaction that had nothing to do")
		return nil
	}); got != 0 {
		t.Fatalf("Compact on a clean graph reclaimed %d", got)
	}

	for i := range 100 {
		g.Delete(fmt.Sprintf("v%d", i))
	}
	built := false
	g.onCompactBuild = func() { built = true }
	quiesced, resumed := 0, 0
	got := g.CompactOnline(func() func() {
		quiesced++
		if !built {
			t.Error("quiesced before the build started")
		}
		return func() {
			resumed++
			if s := g.Stats(); s != (Stats{Live: 100, Slots: 100}) {
				t.Errorf("resumed before the swap: %+v", s)
			}
		}
	})
	if got != 100 || quiesced != 1 || resumed != 1 {
		t.Fatalf("reclaimed %d, quiesced %d times, resumed %d times; want 100, 1, 1", got, quiesced, resumed)
	}
}

// TestCompactDoesNotQueueBehindALongReader: a database snapshot streams the
// graph out under its read lock for seconds. A compaction that took the write
// lock to start, or to drain its catch-up, would queue behind that reader —
// and Go's RWMutex then holds every new search until the snapshot ends. So
// everything before the swap runs under the read lock; only the swap takes the
// write lock, and a database runs it inside quiesce, once its snapshot is done.
func TestCompactDoesNotQueueBehindALongReader(t *testing.T) {
	const n, dim = 400, 16
	g, _ := buildGraph(t, n, dim, 68)
	for i := 0; i < n; i += 2 {
		g.Delete(fmt.Sprintf("v%d", i))
	}

	g.mu.RLock() // the snapshot
	quiescing, snapshotDone := make(chan struct{}), make(chan struct{})
	done := make(chan int)
	go func() {
		done <- g.CompactOnline(func() func() {
			close(quiescing)
			<-snapshotDone // as the database's writeMu would wait for it
			return func() {}
		})
	}()
	select {
	case <-quiescing:
	case <-time.After(30 * time.Second):
		g.mu.RUnlock()
		close(snapshotDone)
		t.Fatal("the compaction waited for a long reader before its swap")
	}
	if _, err := g.Search(randomVector(rand.New(rand.NewSource(69)), dim), 10, 64); err != nil {
		t.Fatal(err)
	}
	g.mu.RUnlock()
	close(snapshotDone)
	if got := <-done; got != n/2 {
		t.Fatalf("Compact reclaimed %d, want %d", got, n/2)
	}
}
