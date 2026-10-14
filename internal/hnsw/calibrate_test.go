package hnsw

import (
	"errors"
	"fmt"
	"math/rand"
	"slices"
	"sync"
	"testing"
)

// TestCalibrateNarrowsAnEasyCorpus is the case calibration exists for: data the
// formula over-searches. The suggestion must fall well below the prior *and*
// still hold the target on queries the calibration never saw.
func TestCalibrateNarrowsAnEasyCorpus(t *testing.T) {
	skipUnderRace(t)
	const dim, k, target = 64, 10, 0.95

	rng := rand.New(rand.NewSource(91))
	c, queries := separatedClusters(rng, 8000, dim, 512)
	g, _ := buildIndex(t, DefaultConfig(dim, Cosine), c)

	prior := g.SuggestedEf(k, target)
	cal, err := g.Calibrate(CalibrateOptions{Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	ef := g.SuggestedEf(k, target)
	recall, _ := evaluate(t, g, c, Cosine, queries(100), k, ef)
	t.Logf("calibration %+v: ef %d -> %d, held-out recall %.3f", cal, prior, ef, recall)

	if !cal.Reached || cal.Scale >= 1 {
		t.Fatalf("calibration did not narrow an easy corpus: %+v", cal)
	}
	if ef*2 > prior {
		t.Fatalf("calibrated ef %d is not well below the prior %d", ef, prior)
	}
	if recall < target-0.02 {
		t.Fatalf("calibrated ef %d reached only %.3f on held-out queries", ef, recall)
	}
}

// TestCalibrateHoldsTheTargetOnHardData: uniform random data is the case the
// formula was fitted to and the hard one for a graph. Whatever calibration
// does to the width there, it must not trade away the target.
func TestCalibrateHoldsTheTargetOnHardData(t *testing.T) {
	skipUnderRace(t)
	const dim, k, target = 128, 10, 0.95

	rng := rand.New(rand.NewSource(92))
	c := uniformCorpus(rng, 5000, dim)
	held := uniformCorpus(rand.New(rand.NewSource(93)), 100, dim)
	g, _ := buildIndex(t, DefaultConfig(dim, Cosine), c)

	cal, err := g.Calibrate(CalibrateOptions{Seed: 2})
	if err != nil {
		t.Fatal(err)
	}
	queries := make([][]float32, 0, len(held.order))
	for _, id := range held.order {
		queries = append(queries, held.vecs[id])
	}
	ef := g.SuggestedEf(k, target)
	recall, _ := evaluate(t, g, c, Cosine, queries, k, ef)
	t.Logf("calibration %+v: ef %d, held-out recall %.3f", cal, ef, recall)

	if recall < target-0.02 {
		t.Fatalf("calibrated ef %d reached only %.3f on held-out queries", ef, recall)
	}
}

func TestCalibrateSkipsASmallGraph(t *testing.T) {
	g, _ := buildGraph(t, MinCalibrationSize-1, 8, 3)
	g.efScale.Store(0x4000000000000000) // a stale 2.0 from when it was larger
	cal, err := g.Calibrate(CalibrateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if cal.Live != 0 || g.EfScale() != 1 {
		t.Fatalf("small graph: %+v, scale %v; want no measurement and the scale reset", cal, g.EfScale())
	}
}

func TestCalibrateStops(t *testing.T) {
	g, _ := buildGraph(t, 2000, 8, 4)
	stop := make(chan struct{})
	close(stop)
	if _, err := g.Calibrate(CalibrateOptions{Stop: stop}); !errors.Is(err, ErrCalibrationStopped) {
		t.Fatalf("Calibrate with a closed Stop = %v", err)
	}
	if g.EfScale() != 1 {
		t.Fatalf("a stopped calibration changed the scale to %v", g.EfScale())
	}
}

// TestConcurrentCalibrateAndInsert is for the race detector: the brute force
// runs outside the lock over references copied under it, while writers go on.
func TestConcurrentCalibrateAndInsert(t *testing.T) {
	g, _ := buildGraph(t, 1500, 8, 5)
	rng := rand.New(rand.NewSource(6))
	extra := make([][]float32, 300)
	for i := range extra {
		extra[i] = randomVector(rng, 8)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i, v := range extra {
			if err := g.Insert(fmt.Sprintf("x%d", i), v); err != nil {
				t.Error(err)
			}
			if i%3 == 0 {
				g.Delete(fmt.Sprintf("v%d", i))
			}
		}
	}()
	if _, err := g.Calibrate(CalibrateOptions{Queries: 16}); err != nil {
		t.Fatal(err)
	}
	_ = g.SuggestedEf(10, 0.95)
	wg.Wait()
}

// TestCalibrateSeesThroughCopies: a corpus stored twice — the same documents
// ingested twice — gives every sample a twin whose neighbor list was chosen
// beside its own. Hiding only the sample's slot let the twin answer for it:
// recall 1.000 at ef=k, so calibration chose the narrowest width there is
// while real queries needed several times more. Held differentially against
// the same corpus stored once, which needs no less width per copy.
func TestCalibrateSeesThroughCopies(t *testing.T) {
	skipUnderRace(t)
	const dim, k, target = 64, 10, 0.95

	base := uniformCorpus(rand.New(rand.NewSource(94)), 3000, dim)
	single, _ := buildIndex(t, DefaultConfig(dim, Cosine), base)
	twice, err := New(DefaultConfig(dim, Cosine))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range base.order {
		for _, name := range []string{id, id + "-again"} {
			if err := twice.Insert(name, base.vecs[id]); err != nil {
				t.Fatal(err)
			}
		}
	}

	once, err := single.Calibrate(CalibrateOptions{Target: target, Seed: 3})
	if err != nil {
		t.Fatal(err)
	}
	cal, err := twice.Calibrate(CalibrateOptions{Target: target, Seed: 3})
	if err != nil {
		t.Fatal(err)
	}

	// Truth by radius: a held-out query's nearest come in tied pairs, and
	// which twin a search returns is arbitrary.
	held := uniformCorpus(rand.New(rand.NewSource(95)), 100, dim)
	vecs := twice.refsFrom(0).vecs
	ef := twice.SuggestedEf(k, target)
	hits := 0
	for _, id := range held.order {
		q := twice.prepare(held.vecs[id])
		ds := make([]float32, len(vecs))
		for i, v := range vecs {
			ds[i] = twice.dist(v, q)
		}
		slices.Sort(ds)
		res, err := twice.Search(held.vecs[id], k, ef)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range res {
			if m.Distance <= ds[k-1] {
				hits++
			}
		}
	}
	recall := float64(hits) / float64(k*len(held.order))
	t.Logf("once %+v; twice %+v: ef %d, held-out recall %.3f", once, cal, ef, recall)

	if cal.Queries != defaultCalibrationQueries {
		t.Fatalf("dropped %d samples as boilerplate; two copies is not boilerplate", defaultCalibrationQueries-cal.Queries)
	}
	if !cal.Reached || 2*cal.Ef < once.Ef {
		t.Fatalf("stored twice, calibration measured ef %d against %d stored once: the copies answered for the samples", cal.Ef, once.Ef)
	}
	if recall < target-0.02 {
		t.Fatalf("calibrated ef %d reached only %.3f on held-out queries", ef, recall)
	}
}

// TestCalibrateSeesThroughDeadCopies: every document deleted and stored again,
// as re-indexing one does. Each sample now has a dead twin, and a tombstone
// keeps its edges — the search still walks through it — so a twin left visible
// leads the leave-one-out search back to the sample's neighborhood. Here that
// measured ef 55 on every one of eight seeds against 65–70 stored once; on a
// database churned by kill -9 rounds over real books it made real questions
// miss a 0.99 target (0.983). Held differentially against the corpus stored
// once, to the eighth the bisection resolves to.
func TestCalibrateSeesThroughDeadCopies(t *testing.T) {
	skipUnderRace(t)
	const dim, target = 64, 0.95

	base := uniformCorpus(rand.New(rand.NewSource(94)), 3000, dim)
	single, _ := buildIndex(t, DefaultConfig(dim, Cosine), base)
	churned, _ := buildIndex(t, DefaultConfig(dim, Cosine), base)
	for _, id := range base.order {
		churned.Delete(id)
		if err := churned.Insert(id, base.vecs[id]); err != nil {
			t.Fatal(err)
		}
	}

	once, err := single.Calibrate(CalibrateOptions{Target: target, Seed: 3})
	if err != nil {
		t.Fatal(err)
	}
	cal, err := churned.Calibrate(CalibrateOptions{Target: target, Seed: 3})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("once %+v\nchurned %+v (%d tombstones)", once, cal, churned.Stats().Deleted)
	if !cal.Reached || cal.Ef < once.Ef-once.Ef/8 {
		t.Fatalf("deleted and stored again, calibration measured ef %d against %d stored once: the dead copies answered for the samples", cal.Ef, once.Ef)
	}
}

// TestCalibrationHidesDeadCopies is the same rule at the level of one truth,
// on both paths that build it: the fold outside the lock, and the residue
// hitsWithout folds under it. A dead copy is hidden with the sample, and a
// tombstone is never an answer.
func TestCalibrationHidesDeadCopies(t *testing.T) {
	const dim, k = 32, 10
	c := uniformCorpus(rand.New(rand.NewSource(98)), 2000, dim)
	g, _ := buildIndex(t, DefaultConfig(dim, Cosine), c)
	// Every tenth vector deleted outright, every tenth stored again.
	for i, id := range c.order {
		switch i % 10 {
		case 0:
			g.Delete(id)
		case 1:
			g.Delete(id)
			if err := g.Insert(id, c.vecs[id]); err != nil {
				t.Fatal(err)
			}
		}
	}

	var truths []truth
	for i, id := range c.order {
		if i%10 == 1 && len(truths) < 32 {
			q := g.nodes[g.ids[id]].vector
			truths = append(truths, truth{q: q, self: g.dist(q, q)})
		}
	}
	r := g.refsFrom(0)
	if err := g.fold(truths, r, k, nil); err != nil {
		t.Fatal(err)
	}
	dead := func(tr truth) (hidden int) {
		for _, s := range tr.hide {
			if g.nodes[s].deleted {
				hidden++
			}
		}
		for _, b := range tr.best {
			if g.nodes[b.idx].deleted {
				t.Fatalf("slot %d is a tombstone and an answer", b.idx)
			}
		}
		return hidden
	}
	for _, tr := range truths {
		if len(tr.hide) != 2 || dead(tr) != 1 {
			t.Fatalf("sample hides %v; want itself and its dead copy", tr.hide)
		}
	}

	// A copy stored and deleted after the fold reaches the truth through
	// hitsWithout's residue, under the lock.
	for _, tr := range truths {
		id := g.nodes[tr.hide[0]].id
		if err := g.Insert(id+"-late", c.vecs[id]); err != nil {
			t.Fatal(err)
		}
		g.Delete(id + "-late")
	}
	for i := range truths {
		if _, ok := g.hitsWithout(&truths[i], k, g.Len(), r.layout); !ok {
			t.Fatal("a sample was retired by a dead copy")
		}
		if len(truths[i].hide) != 3 || dead(truths[i]) != 2 {
			t.Fatalf("sample hides %v; want itself and both dead copies", truths[i].hide)
		}
	}
}

// TestCalibrationTruthFollowsWrites: a corpus ingested a second time while it
// is being calibrated. A copy of a sample written after the truth was taken is
// the sample again — left visible, the search walks to it; counted as a miss it
// held every such sample to (k-1)/k and pinned the scale at maxEfScale, which
// is what real books did when the background calibration fired halfway through
// a second ingest. The probe has to fold it into the truth: hidden, like every
// copy the truth already knew about.
//
// And the two writes that cannot be folded in retire the sample instead of
// scoring it: a deleted answer, and a Compact's renumbering.
func TestCalibrationTruthFollowsWrites(t *testing.T) {
	const dim, k = 32, 10
	c := uniformCorpus(rand.New(rand.NewSource(96)), 2000, dim)
	g, _ := buildIndex(t, DefaultConfig(dim, Cosine), c)

	r := g.refsFrom(0)
	truths := make([]truth, 32)
	for i, j := range rand.New(rand.NewSource(4)).Perm(len(r.vecs))[:len(truths)] {
		truths[i] = truth{q: r.vecs[j], self: g.dist(r.vecs[j], r.vecs[j])}
	}
	if err := g.fold(truths, r, k, nil); err != nil {
		t.Fatal(err)
	}
	for _, tr := range truths {
		// The raw vector, so the copy is stored bit-identical to the sample.
		id := g.nodes[tr.hide[0]].id
		if err := g.Insert(id+"-late", c.vecs[id]); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(g.nodes[g.ids[id+"-late"]].vector, tr.q) {
			t.Fatal("late copy is not an exact copy; the test would prove nothing")
		}
	}

	exhaustive, hits := g.Len(), 0
	for i := range truths {
		h, ok := g.hitsWithout(&truths[i], k, exhaustive, r.layout)
		if !ok {
			t.Fatal("a sample was retired by an insert")
		}
		if got := len(truths[i].hide); got != 2 {
			t.Fatalf("sample hides %d slots after its copy arrived, want itself and the copy", got)
		}
		hits += h
	}
	if recall := float64(hits) / float64(k*len(truths)); recall < 0.99 {
		t.Fatalf("exhaustive search scored %.3f once copies arrived after the truth", recall)
	}

	if !g.Delete(g.nodes[truths[0].best[0].idx].id) {
		t.Fatal("answer was not live")
	}
	if _, ok := g.hitsWithout(&truths[0], k, exhaustive, r.layout); ok {
		t.Fatal("scored against a truth whose nearest answer was deleted")
	}
	g.Compact()
	if _, ok := g.hitsWithout(&truths[1], k, exhaustive, r.layout); ok {
		t.Fatal("scored against slot numbers a Compact has since renumbered")
	}
}

// TestCalibrateSurvivesACompaction: a Compact between a calibration's truth and
// its probes renumbers every slot the truth names, so its hidden slots hide
// strangers and the sample is left visible to its own search. The calibration
// has to notice and measure again — and, the graph and sample being the same,
// arrive at exactly what a calibration after the Compact does.
func TestCalibrateSurvivesACompaction(t *testing.T) {
	skipUnderRace(t)
	const dim, target = 32, 0.95
	c := uniformCorpus(rand.New(rand.NewSource(97)), 3000, dim)
	g, _ := buildIndex(t, DefaultConfig(dim, Cosine), c)
	for _, id := range c.order[:900] {
		g.Delete(id)
	}

	compacted := false
	g.onCalibrateProbe = func() {
		if !compacted {
			compacted = true
			g.Compact()
		}
	}
	during, err := g.Calibrate(CalibrateOptions{Target: target, Seed: 5})
	g.onCalibrateProbe = nil
	if err != nil {
		t.Fatal(err)
	}
	after, err := g.Calibrate(CalibrateOptions{Target: target, Seed: 5})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("during %+v\nafter  %+v", during, after)
	if during != after {
		t.Fatalf("a calibration a Compact interrupted measured %+v, one after it %+v", during, after)
	}
}

// TestCalibrationIsNotFlatteredByLateWrites: the same corpus ingested a second
// time between a calibration's truth and its probes — what the database's
// background calibration sees, since it fires when the live count doubles. A
// truth frozen at the start is wrong both ways: by distance, the second copies
// inside its radius made k answers easier to find than the true k, and the
// calibration came out narrower than with no writes at all (0.977 against a
// 0.99 target on real books); passed over, the sample's own late copy was a
// shortcut to it (ef 22 against 70). Folded forward, the probes measure the
// doubled graph, which needs no less width than the single one.
func TestCalibrationIsNotFlatteredByLateWrites(t *testing.T) {
	skipUnderRace(t)
	const dim, k, target = 64, 10, 0.95
	base := uniformCorpus(rand.New(rand.NewSource(94)), 3000, dim)
	g, _ := buildIndex(t, DefaultConfig(dim, Cosine), base)

	quiet, err := g.Calibrate(CalibrateOptions{Target: target, Seed: 3})
	if err != nil {
		t.Fatal(err)
	}
	g.onCalibrateProbe = func() {
		for _, id := range base.order {
			if err := g.Insert(id+"-again", base.vecs[id]); err != nil {
				t.Error(err)
			}
		}
	}
	busy, err := g.Calibrate(CalibrateOptions{Target: target, Seed: 3})
	g.onCalibrateProbe = nil
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("quiet %+v\nbusy  %+v", quiet, busy)

	// Same sample, same truth; the bisection resolves to an eighth.
	if !busy.Reached || busy.Ef < quiet.Ef-quiet.Ef/8 {
		t.Fatalf("with a second ingest landing during it, calibration measured ef %d against %d without: the truth did not follow the writes", busy.Ef, quiet.Ef)
	}
}
