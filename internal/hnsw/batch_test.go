package hnsw

import (
	"errors"
	"fmt"
	"math/rand"
	"slices"
	"sync"
	"testing"
	"time"
)

func batchOf(c corpus) ([]string, [][]float32) {
	vecs := make([][]float32, len(c.order))
	for i, id := range c.order {
		vecs[i] = c.vecs[id]
	}
	return c.order, vecs
}

func buildBatch(t *testing.T, cfg Config, c corpus, workers int) (*Graph, time.Duration) {
	t.Helper()
	g, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := batchOf(c)
	start := time.Now()
	if err := g.InsertBatch(ids, vecs, workers); err != nil {
		t.Fatal(err)
	}
	return g, time.Since(start)
}

// assertWellFormed checks the invariants a parallel build could break and a
// serial one cannot: every list within its layer's cap, no self-loops, no
// duplicate edges, no edge to a layer a node does not reach — and that the
// codec, which refuses a structurally invalid graph, accepts it.
func assertWellFormed(t *testing.T, g *Graph) {
	t.Helper()
	for idx, n := range g.nodes {
		for lc, nbrs := range n.neighbors {
			if len(nbrs) > g.maxConn(lc) {
				t.Fatalf("node %d layer %d has %d edges, cap %d", idx, lc, len(nbrs), g.maxConn(lc))
			}
			seen := make(map[int]bool, len(nbrs))
			for _, nb := range nbrs {
				switch {
				case nb == idx:
					t.Fatalf("node %d links to itself on layer %d", idx, lc)
				case seen[nb]:
					t.Fatalf("node %d links to %d twice on layer %d", idx, nb, lc)
				case lc > g.nodes[nb].topLevel():
					t.Fatalf("node %d links to %d on layer %d, which %d does not reach", idx, nb, lc, nb)
				}
				seen[nb] = true
			}
		}
	}
	if g.Len() > 0 && g.nodes[g.entry].topLevel() != g.maxLevel {
		t.Fatalf("entry %d is at level %d, maxLevel %d", g.entry, g.nodes[g.entry].topLevel(), g.maxLevel)
	}
	roundTrip(t, g)
}

// TestInsertBatchRecallMatchesSerial is the guarantee InsertBatch makes in
// place of determinism: a different graph, and an equally good one.
func TestInsertBatchRecallMatchesSerial(t *testing.T) {
	for _, tc := range []struct {
		name string
		c    corpus
	}{
		{"uniform", uniformCorpus(rand.New(rand.NewSource(1)), 4000, 64)},
		{"clustered", clusteredCorpus(rand.New(rand.NewSource(2)), 4000, 64, 40)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{Dimension: 64, Metric: Cosine, Seed: 3}
			queries := makeQueries(rand.New(rand.NewSource(4)), 100, 64)

			serial, _ := buildIndex(t, cfg, tc.c)
			parallel, _ := buildBatch(t, cfg, tc.c, 8)
			assertWellFormed(t, parallel)

			if parallel.Len() != serial.Len() {
				t.Fatalf("Len = %d, serial %d", parallel.Len(), serial.Len())
			}
			for _, ef := range []int{16, 64} {
				rs, _ := evaluate(t, serial, tc.c, Cosine, queries, 10, ef)
				rp, _ := evaluate(t, parallel, tc.c, Cosine, queries, 10, ef)
				t.Logf("ef=%d recall serial %.3f parallel %.3f", ef, rs, rp)
				if rp < rs-0.02 {
					t.Errorf("ef=%d: parallel recall %.3f, serial %.3f", ef, rp, rs)
				}
			}
		})
	}
}

// TestInsertBatchIntoALiveGraph is the shape that matters in practice: a
// rebuild over REST arrives as hundreds of batches, and only the first lands
// in an empty graph. Batches into an existing graph, with tombstones in it,
// must link to what is already there.
func TestInsertBatchIntoALiveGraph(t *testing.T) {
	c := uniformCorpus(rand.New(rand.NewSource(5)), 3000, 32)
	cfg := Config{Dimension: 32, Metric: Euclidean, Seed: 6}
	g, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := batchOf(c)
	for lo := 0; lo < len(ids); lo += 250 {
		hi := min(lo+250, len(ids))
		if err := g.InsertBatch(ids[lo:hi], vecs[lo:hi], 8); err != nil {
			t.Fatal(err)
		}
		// Some deletes between batches, so later batches meet tombstones.
		for i := lo; i < hi; i += 17 {
			g.Delete(ids[i])
			delete(c.vecs, ids[i])
		}
	}
	assertWellFormed(t, g)
	if g.Len() != len(c.vecs) {
		t.Fatalf("Len = %d, want %d", g.Len(), len(c.vecs))
	}
	r, _ := evaluate(t, g, c, Euclidean, makeQueries(rand.New(rand.NewSource(7)), 100, 32), 10, 64)
	if r < 0.95 {
		t.Fatalf("recall %.3f after batched build with deletes, want >= 0.95", r)
	}
}

// TestInsertBatchUpsertSemantics: the same outcome Insert gives, id by id.
func TestInsertBatchUpsertSemantics(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	g, err := New(Config{Dimension: 8, Metric: Euclidean, Seed: 9})
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	var vecs [][]float32
	for i := range 100 {
		ids = append(ids, fmt.Sprintf("v%d", i))
		vecs = append(vecs, randomVector(rng, 8))
	}
	if err := g.InsertBatch(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}

	// Unchanged vectors are no-ops; v0 changes; v1 appears twice and the last
	// wins; one new id.
	again := slices.Clone(ids)
	againV := slices.Clone(vecs)
	changed, first, second, fresh := randomVector(rng, 8), randomVector(rng, 8), randomVector(rng, 8), randomVector(rng, 8)
	againV[0] = changed
	againV[1] = first
	again = append(again, "v1", "new")
	againV = append(againV, second, fresh)
	if err := g.InsertBatch(again, againV, 4); err != nil {
		t.Fatal(err)
	}

	if s := g.Stats(); s.Live != 101 || s.Deleted != 2 {
		t.Fatalf("Stats = %+v, want 101 live and 2 tombstones (v0 and v1 replaced once each)", s)
	}
	for id, want := range map[string][]float32{"v0": changed, "v1": second, "new": fresh, "v2": vecs[2]} {
		if got, ok := g.Vector(id); !ok || !slices.Equal(got, want) {
			t.Errorf("%s = %v, want %v", id, got, want)
		}
	}
	assertWellFormed(t, g)
}

func TestInsertBatchRejectsBeforeWriting(t *testing.T) {
	g, err := New(Config{Dimension: 4})
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, 64)
	vecs := make([][]float32, 64)
	for i := range ids {
		ids[i], vecs[i] = fmt.Sprintf("v%d", i), []float32{1, 2, 3, 4}
	}
	vecs[50] = []float32{1, 2, 3}
	if err := g.InsertBatch(ids, vecs, 4); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("err = %v, want ErrDimensionMismatch", err)
	}
	if err := g.InsertBatch(ids, vecs[:10], 4); !errors.Is(err, ErrBatchMismatch) {
		t.Fatalf("err = %v, want ErrBatchMismatch", err)
	}
	if g.Len() != 0 {
		t.Fatalf("a refused batch inserted %d vectors", g.Len())
	}
}

// TestInsertBatchOneWorkerIsSerial: workers=1 is the reproducible path, and
// must be the same graph a loop over Insert builds.
func TestInsertBatchOneWorkerIsSerial(t *testing.T) {
	c := uniformCorpus(rand.New(rand.NewSource(10)), 500, 16)
	cfg := Config{Dimension: 16, Seed: 11}
	serial, _ := buildIndex(t, cfg, c)
	one, _ := buildBatch(t, cfg, c, 1)
	assertSameGraph(t, serial, one)
}

// TestInsertBatchConcurrentWithSearch: searches take the lock between chunks
// and must never see a half-linked node. Run under -race; the detector is what
// would catch a neighbor list read outside its lock.
func TestInsertBatchConcurrentWithSearch(t *testing.T) {
	c := uniformCorpus(rand.New(rand.NewSource(12)), 2000, 32)
	g, err := New(Config{Dimension: 32, Seed: 13})
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := batchOf(c)

	stop := make(chan struct{})
	var wg sync.WaitGroup
	for r := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := makeQueries(rand.New(rand.NewSource(int64(100+r))), 1, 32)[0]
			for {
				select {
				case <-stop:
					return
				default:
				}
				if _, err := g.Search(q, 10, 32); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	for lo := 0; lo < len(ids); lo += 500 {
		if err := g.InsertBatch(ids[lo:lo+500], vecs[lo:lo+500], 4); err != nil {
			t.Fatal(err)
		}
	}
	close(stop)
	wg.Wait()
	assertWellFormed(t, g)
}

// unreachable counts live nodes no layer-0 walk from the entry can reach.
// Every such node is a vector no search can ever return.
func unreachable(g *Graph) int {
	if g.entry < 0 {
		return 0
	}
	seen := make([]bool, len(g.nodes))
	seen[g.entry] = true
	queue := []int{g.entry}
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		for _, nb := range g.nodes[n].neighbors[0] {
			if !seen[nb] {
				seen[nb] = true
				queue = append(queue, nb)
			}
		}
	}
	count := 0
	for i, n := range g.nodes {
		if !n.deleted && !seen[i] {
			count++
		}
	}
	return count
}

// TestInsertBatchStrandsNothing: no vector a parallel build inserts may be
// unreachable, which is what a serial build gives. Without the in-flight rule
// in InsertBatch, workers seeded from half-linked nodes stranded 17 of 10,000
// vectors at dimension 8 and 16 workers, the worst cell measured. Low dimension
// and many workers is the harshest setting, and each build differs, so it runs
// several.
func TestInsertBatchStrandsNothing(t *testing.T) {
	for run := range 3 {
		for _, dim := range []int{8, 64} {
			c := uniformCorpus(rand.New(rand.NewSource(int64(20+run))), 10000, dim)
			g, _ := buildBatch(t, Config{Dimension: dim, Seed: int64(run)}, c, 16)
			if n := unreachable(g); n != 0 {
				t.Errorf("run %d dim %d: %d of %d vectors unreachable", run, dim, n, g.Len())
			}
		}
	}
}
