package hnsw

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"
)

// -results turns these sweeps from a guard into a measurement harness: the grid
// widens, the corpora grow, and every cell is written out for docs/benchmarks to
// plot. Without it the same tests run a small grid and assert thresholds, so the
// numbers in the README and the numbers CI defends come from one piece of code
// rather than two that can drift apart.
//
//	go test ./internal/hnsw/ -run TestSweep -results docs/benchmarks/results.csv -timeout 30m
var resultsPath = flag.String("results", "", "write sweep measurements to this CSV path")

// A note on the thresholds in this file, because they look lax next to the 0.999
// that graph_test.go defends.
//
// These sweeps deliberately vary the things recall depends on most — corpus
// size, dimension, search width — and recall at a *fixed* ef falls as either the
// corpus or the dimension grows. A single absolute floor that holds across the
// whole grid can only be the floor of its hardest cell, so absolute floors here
// are set to catch a collapse, nothing finer.
//
// The assertions that actually earn their keep in this file are the ones about
// *shape*: recall must not fall as ef rises, must not fall as M rises, must
// recover under a wide search whatever the metric, and must not drop when a
// graph is compacted. Those hold regardless of where the absolute numbers land,
// and they are what a real regression breaks. Pinning exact recall is the job of
// the fixed-configuration tests — TestRecallVsBruteForce, TestRecallHighDimension,
// TestRecallIsStableAcrossSeeds — which vary nothing.

func measuring() bool { return *resultsPath != "" }

// skipUnderRace keeps `go test ./... -race` inside the default ten-minute
// timeout, which these sweeps blew past.
//
// Skipping is not a dodge here, it is the correct scope. Every sweep in this
// file is single-goroutine — build a graph, query it, compare against brute
// force — so the race detector has nothing to observe in them while costing
// roughly 10x. Race coverage belongs to the five TestConcurrent* tests, which
// exercise readers against writers and finish in about twenty seconds under
// -race. Timings taken under the detector would be meaningless as measurements
// anyway, so this applies in measurement mode too.
func skipUnderRace(t *testing.T) {
	t.Helper()
	if raceDetectorEnabled {
		t.Skip("single-goroutine sweep: nothing for -race to observe, and it costs ~10x (see skipUnderRace)")
	}
}

// sample is one measured cell of one sweep.
type sample struct {
	Sweep  string // which grid this belongs to
	Label  string // the varying value, as it should appear on an axis
	Dim    int
	N      int
	M      int
	Ef     int
	K      int
	Recall float64
	Search time.Duration // mean, per query
	Build  time.Duration // whole corpus
}

// collected accumulates across every sweep in the run; TestMain writes it once.
var collected []sample

func record(t *testing.T, s sample) {
	t.Helper()
	t.Logf("%-12s %-10s dim=%-5d N=%-6d M=%-3d ef=%-4d recall@%d=%.4f search=%v",
		s.Sweep, s.Label, s.Dim, s.N, s.M, s.Ef, s.K, s.Recall, s.Search)
	collected = append(collected, s)
}

func TestMain(m *testing.M) {
	flag.Parse()
	code := m.Run()

	// Written whether or not the run passed. A tripped threshold is exactly when
	// the measurements are most worth having — discarding a multi-minute sweep
	// because one cell fell below a floor would mean re-running it blind to find
	// out by how much.
	if measuring() && len(collected) > 0 {
		if err := writeResults(*resultsPath, collected); err != nil {
			fmt.Fprintf(os.Stderr, "writing results: %v\n", err)
			code = 1
		}
	}
	os.Exit(code)
}

func writeResults(path string, samples []sample) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	if err := w.Write([]string{
		"sweep", "label", "dim", "n", "m", "ef", "k", "recall", "search_ns", "build_ms",
	}); err != nil {
		return err
	}
	for _, s := range samples {
		if err := w.Write([]string{
			s.Sweep, s.Label,
			strconv.Itoa(s.Dim), strconv.Itoa(s.N), strconv.Itoa(s.M),
			strconv.Itoa(s.Ef), strconv.Itoa(s.K),
			strconv.FormatFloat(s.Recall, 'f', 4, 64),
			strconv.FormatInt(s.Search.Nanoseconds(), 10),
			strconv.FormatFloat(float64(s.Build.Nanoseconds())/1e6, 'f', 1, 64),
		}); err != nil {
			return err
		}
	}
	return w.Error()
}

// corpus is a set of vectors plus the order they were inserted in, so a run is
// reproducible rather than dependent on Go's map iteration.
type corpus struct {
	vecs  map[string][]float32
	order []string
	dim   int
}

// uniformCorpus draws every component independently from [-1,1) — centred on
// the origin, which is what a real embedding space looks like.
//
// It was introduced on the theory that the all-positive corpus the rest of the
// package uses (`randomVector`, components in [0,1)) would depress recall: every
// vector shares an orthant, so any two are ~0.75 similar before you look at the
// data, and recall@10 would be resolving a near-tie. **The measurement did not
// support that.** At 5,000 vectors the two score within a point and a half of
// each other, with the positive corpus marginally ahead — see the distribution
// sweep. What actually drives recall down at a fixed ef is corpus size, which
// TestSweepScale isolates.
//
// So centred stays the default for being representative, not for being kinder,
// and positiveCorpus is kept so the comparison stays visible rather than
// becoming folklore.
func uniformCorpus(rng *rand.Rand, n, dim int) corpus {
	c := corpus{vecs: make(map[string][]float32, n), dim: dim}
	for i := range n {
		id := fmt.Sprintf("v%d", i)
		v := make([]float32, dim)
		for j := range v {
			v[j] = rng.Float32()*2 - 1
		}
		c.vecs[id] = v
		c.order = append(c.order, id)
	}
	return c
}

// positiveCorpus is the all-positive corpus the rest of the package's tests use:
// every component in [0,1), so every vector shares an orthant. It is included to
// be measured, not to be built on — see uniformCorpus.
func positiveCorpus(rng *rand.Rand, n, dim int) corpus {
	c := corpus{vecs: make(map[string][]float32, n), dim: dim}
	for i := range n {
		id := fmt.Sprintf("v%d", i)
		c.vecs[id] = randomVector(rng, dim)
		c.order = append(c.order, id)
	}
	return c
}

// clusteredCorpus is what real embeddings look like: points gathered around a
// few dozen centroids rather than sprayed through the whole space. Reporting
// only uniform numbers understates the index on the data anyone actually has.
func clusteredCorpus(rng *rand.Rand, n, dim, clusters int) corpus {
	centroids := make([][]float32, clusters)
	for i := range centroids {
		centroids[i] = randomVector(rng, dim)
	}

	c := corpus{vecs: make(map[string][]float32, n), dim: dim}
	for i := range n {
		id := fmt.Sprintf("v%d", i)
		centre := centroids[i%clusters]
		v := make([]float32, dim)
		for j := range v {
			// Tight spread: the cluster dominates, which is the point.
			v[j] = centre[j] + float32(rng.NormFloat64())*0.05
		}
		c.vecs[id] = v
		c.order = append(c.order, id)
	}
	return c
}

func buildIndex(t *testing.T, cfg Config, c corpus) (*Graph, time.Duration) {
	t.Helper()
	g, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	for _, id := range c.order {
		if err := g.Insert(id, c.vecs[id]); err != nil {
			t.Fatal(err)
		}
	}
	return g, time.Since(start)
}

// evaluate measures recall against brute-force ground truth and mean search
// latency over the same queries.
//
// Ground truth uses the metric's general form against the *raw* corpus, never
// the graph's stored vectors — otherwise normalization bugs would cancel out on
// both sides and the measurement would confirm itself.
func evaluate(t *testing.T, g *Graph, c corpus, metric Metric, queries [][]float32, k, ef int) (recall float64, perQuery time.Duration) {
	t.Helper()
	dist := metric.Func()

	truth := make([][]string, len(queries))
	for i, q := range queries {
		truth[i] = bruteForceNearest(c.vecs, dist, q, k)
	}

	// Warm the pool and any cold paths so the timing below measures steady state.
	for _, q := range queries[:min(8, len(queries))] {
		if _, err := g.Search(q, k, ef); err != nil {
			t.Fatal(err)
		}
	}

	var hits, total int
	start := time.Now()
	for i, q := range queries {
		got, err := g.Search(q, k, ef)
		if err != nil {
			t.Fatal(err)
		}
		want := make(map[string]struct{}, len(truth[i]))
		for _, id := range truth[i] {
			want[id] = struct{}{}
		}
		for _, r := range got {
			if _, ok := want[r.ID]; ok {
				hits++
			}
		}
		total += len(truth[i])
	}
	elapsed := time.Since(start)

	return float64(hits) / float64(total), elapsed / time.Duration(len(queries))
}

func makeQueries(rng *rand.Rand, count, dim int) [][]float32 {
	qs := make([][]float32, count)
	for i := range qs {
		qs[i] = randomVector(rng, dim)
	}
	return qs
}

// sweepSize returns the corpus size and query count for the current mode. The
// assertion grid stays small enough to live in the default `go test` run; the
// measurement grid is what the published charts are drawn from.
func sweepSize() (n, queries int) {
	if measuring() {
		return 5000, 200
	}
	return 800, 50
}

// TestSweepDimension walks the dimensions the index is actually used at, from
// toy sizes to a 1536-dim embedding. Recall is expected to *fall* as dimension
// rises — distances concentrate, so neighborhoods carry less information — and
// the point of measuring it is to know by how much rather than to assume.
func TestSweepDimension(t *testing.T) {
	skipUnderRace(t)

	dims := []int{3, 8, 32, 128, 384}
	if measuring() {
		dims = []int{8, 32, 64, 128, 256, 384, 768, 1536}
	}
	n, nq := sweepSize()
	const (
		k  = 10
		ef = 64
	)

	for _, dim := range dims {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(200 + dim)))
			c := uniformCorpus(rng, n, dim)
			cfg := DefaultConfig(dim, Cosine)

			g, build := buildIndex(t, cfg, c)
			recall, perQuery := evaluate(t, g, c, Cosine, makeQueries(rng, nq, dim), k, ef)

			record(t, sample{
				Sweep: "dimension", Label: strconv.Itoa(dim),
				Dim: dim, N: n, M: cfg.M, Ef: ef, K: k,
				Recall: recall, Search: perQuery, Build: build,
			})

			// A collapse guard, not a target. At a fixed ef=64 across a 200x
			// range of dimensions the slope is the point; see the note at the
			// top of this file on why the floors here are loose.
			if recall < 0.60 {
				t.Fatalf("recall@%d at dim %d collapsed to %.3f", k, dim, recall)
			}
		})
	}
}

// TestSweepScale is the empirical form of the claim the README makes: a search
// visits a tiny fraction of the graph, so latency grows with log N rather than
// with N. Nothing else in the package demonstrates that — the benchmarks all
// run at a single corpus size.
func TestSweepScale(t *testing.T) {
	skipUnderRace(t)

	sizes := []int{500, 2000, 8000}
	if measuring() {
		sizes = []int{500, 1000, 2000, 5000, 10000, 20000}
	}
	const (
		dim = 128
		k   = 10
		ef  = 64
	)
	nq := 50
	if measuring() {
		nq = 200
	}

	var latencies []float64
	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			rng := rand.New(rand.NewSource(901))
			c := uniformCorpus(rng, n, dim)
			cfg := DefaultConfig(dim, Cosine)

			g, build := buildIndex(t, cfg, c)
			recall, perQuery := evaluate(t, g, c, Cosine, makeQueries(rng, nq, dim), k, ef)

			record(t, sample{
				Sweep: "scale", Label: strconv.Itoa(n),
				Dim: dim, N: n, M: cfg.M, Ef: ef, K: k,
				Recall: recall, Search: perQuery, Build: build,
			})
			latencies = append(latencies, float64(perQuery))

			// A floor, not a target. At a FIXED ef, recall necessarily decays as
			// the corpus grows: the search width stays the same while the number
			// of plausible candidates rises, so the beam covers a smaller share
			// of the space. That is the central reason ef is a per-query
			// argument rather than a build-time constant — hold recall steady by
			// raising ef with N. TestSweepEf is the other half of this picture.
			if recall < 0.70 {
				t.Fatalf("recall@%d at N=%d = %.3f, below the floor even for fixed ef", k, n, recall)
			}
		})
	}

	// What this can and cannot assert.
	//
	// Algorithmically a layer-0 search visits about ef nodes and expands each by
	// M, which is independent of N; only the descent grows, and it grows with
	// log N. Wall clock does not follow that cleanly, because once the corpus
	// outgrows L2 the distance kernels start paying for memory rather than
	// arithmetic — measured growth is nearer sqrt(N) than log(N) for that
	// reason, and it is a property of the machine, not of the index.
	//
	// So the bound is set to catch the failure that actually matters: a
	// regression to scanning, where latency would track N outright. Half of
	// linear is far above anything the cache effect produces and far below a
	// scan.
	nGrowth := float64(sizes[len(sizes)-1]) / float64(sizes[0])
	latGrowth := latencies[len(latencies)-1] / latencies[0]
	bound := nGrowth / 2
	t.Logf("corpus grew %.0fx, search latency grew %.2fx (sub-linear bound: %.2fx)", nGrowth, latGrowth, bound)

	if latGrowth > bound {
		t.Fatalf("search latency grew %.2fx for a %.0fx corpus — that is scan-like, not sub-linear", latGrowth, nGrowth)
	}
}

// TestSweepEf is the knob users actually turn, and the only one they can turn
// per query. It buys recall with latency; this measures the exchange rate.
func TestSweepEf(t *testing.T) {
	skipUnderRace(t)

	efs := []int{10, 32, 128}
	if measuring() {
		efs = []int{10, 16, 32, 64, 128, 256, 512}
	}
	n, nq := sweepSize()
	const (
		dim = 128
		k   = 10
	)

	rng := rand.New(rand.NewSource(301))
	c := uniformCorpus(rng, n, dim)
	cfg := DefaultConfig(dim, Cosine)

	// One graph for the whole sweep: ef is a query-time argument, so rebuilding
	// per cell would measure build variance instead of the knob.
	g, build := buildIndex(t, cfg, c)
	queries := makeQueries(rng, nq, dim)

	var recalls []float64
	for _, ef := range efs {
		recall, perQuery := evaluate(t, g, c, Cosine, queries, k, ef)
		record(t, sample{
			Sweep: "ef", Label: strconv.Itoa(ef),
			Dim: dim, N: n, M: cfg.M, Ef: ef, K: k,
			Recall: recall, Search: perQuery, Build: build,
		})
		recalls = append(recalls, recall)
	}

	// Monotonicity is the contract: a wider search may cost more, but it must
	// never find less. A violation means the pruning bound is wrong, not noisy.
	for i := 1; i < len(recalls); i++ {
		if recalls[i] < recalls[i-1]-0.01 {
			t.Fatalf("ef=%d scored %.4f, worse than ef=%d at %.4f — wider search must not find less",
				efs[i], recalls[i], efs[i-1], recalls[i-1])
		}
	}
	if last := recalls[len(recalls)-1]; last < 0.95 {
		t.Fatalf("recall at the widest ef is only %.3f", last)
	}
}

// TestSweepM measures the structural knob. M cannot be changed without a
// rebuild, so this is the chart to consult *before* building an index.
func TestSweepM(t *testing.T) {
	skipUnderRace(t)

	ms := []int{4, 16, 32}
	if measuring() {
		ms = []int{4, 8, 12, 16, 24, 32, 48}
	}
	n, nq := sweepSize()
	const (
		dim = 128
		k   = 10
		ef  = 64
	)

	var recalls []float64
	for _, m := range ms {
		rng := rand.New(rand.NewSource(401))
		c := uniformCorpus(rng, n, dim)
		cfg := DefaultConfig(dim, Cosine)
		cfg.M = m

		g, build := buildIndex(t, cfg, c)
		recall, perQuery := evaluate(t, g, c, Cosine, makeQueries(rng, nq, dim), k, ef)

		record(t, sample{
			Sweep: "M", Label: strconv.Itoa(m),
			Dim: dim, N: n, M: m, Ef: ef, K: k,
			Recall: recall, Search: perQuery, Build: build,
		})
		recalls = append(recalls, recall)
	}

	// More neighbors is more paths out of any node, so recall should climb —
	// and the low-M end is where vectors get stranded, which the delete tests
	// already showed from the other direction.
	if recalls[len(recalls)-1] < recalls[0] {
		t.Fatalf("M=%d scored %.4f, no better than M=%d at %.4f",
			ms[len(ms)-1], recalls[len(recalls)-1], ms[0], recalls[0])
	}
}

// TestSweepMetric closes a real gap: every recall test in this package used
// Cosine, so Euclidean and DotProduct had no accuracy coverage at all — only
// the kernels were checked, never the graph built on top of them.
//
// DotProduct is deliberately held to a lower bar. It is not a metric: it has no
// triangle inequality and rewards magnitude, so a long vector is "close" to
// everything. A graph is a weaker structure over it, and pretending otherwise
// with an equal threshold would just make the test lie.
func TestSweepMetric(t *testing.T) {
	skipUnderRace(t)

	n, nq := sweepSize()
	const (
		dim = 128
		k   = 10
		ef  = 64
	)

	// Each metric is measured twice: at the default ef, and at a wide one. The
	// pair is what makes a low number diagnostic instead of merely alarming.
	//
	// Measured at 5,000 vectors, all three land within two points of each other
	// at ef=64 — Cosine 0.830, Euclidean 0.820, DotProduct 0.848 — and all three
	// recover to 0.98 or better at ef=256. So the dip is not a property of any
	// metric. It is the fixed-ef decay that TestSweepScale isolates: a 64-wide
	// beam covers less of the space at 5,000 vectors than at 800. Reading it as
	// "Euclidean is weak" would have been the obvious wrong conclusion, and the
	// wide-ef column is what rules it out.
	const wideEf = 256

	for _, tc := range []struct {
		metric    Metric
		name      string
		floor     float64 // at ef=64
		wideFloor float64 // at ef=256 — the graph must be navigable, whatever the data
	}{
		{Cosine, "Cosine", 0.70, 0.98},
		{Euclidean, "Euclidean", 0.70, 0.95},
		// DotProduct is not a metric: no triangle inequality, and magnitude
		// counts, so a long vector is "near" everything. A graph is a weaker
		// structure over it, so it gets more headroom — though on this corpus it
		// in fact scores as well as the other two.
		{DotProduct, "DotProduct", 0.60, 0.95},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(501))
			c := uniformCorpus(rng, n, dim)
			cfg := DefaultConfig(dim, tc.metric)

			g, build := buildIndex(t, cfg, c)
			queries := makeQueries(rng, nq, dim)

			recall, perQuery := evaluate(t, g, c, tc.metric, queries, k, ef)
			record(t, sample{
				Sweep: "metric", Label: tc.name,
				Dim: dim, N: n, M: cfg.M, Ef: ef, K: k,
				Recall: recall, Search: perQuery, Build: build,
			})

			wideRecall, widePerQuery := evaluate(t, g, c, tc.metric, queries, k, wideEf)
			record(t, sample{
				Sweep: "metric-wide", Label: tc.name,
				Dim: dim, N: n, M: cfg.M, Ef: wideEf, K: k,
				Recall: wideRecall, Search: widePerQuery, Build: build,
			})
			t.Logf("%s: %.3f at ef=%d, %.3f at ef=%d", tc.name, recall, ef, wideRecall, wideEf)

			if recall < tc.floor {
				t.Fatalf("%s recall@%d at ef=%d = %.3f, below its floor of %.2f", tc.name, k, ef, recall, tc.floor)
			}
			// The load-bearing assertion: whatever the data does to a narrow
			// search, a wide one must still find the neighbours. Failing here
			// means the graph itself is not navigable under this metric.
			if wideRecall < tc.wideFloor {
				t.Fatalf("%s recall@%d at ef=%d = %.3f — a wide search should recover it, so the graph is at fault, not the corpus",
					tc.name, k, wideEf, wideRecall)
			}
		})
	}
}

// TestSweepDistribution contrasts uniform noise with clustered data. Uniform is
// the standard benchmark input and the pessimistic one — real embeddings are
// clustered, and a graph index exploits exactly that structure.
func TestSweepDistribution(t *testing.T) {
	skipUnderRace(t)

	n, nq := sweepSize()
	const (
		dim = 128
		k   = 10
		ef  = 64
	)

	for _, tc := range []struct {
		name  string
		floor float64
		build func(*rand.Rand) corpus
	}{
		{"centered", 0.75, func(r *rand.Rand) corpus { return uniformCorpus(r, n, dim) }},
		// Kept because the expected penalty here turned out not to exist: at
		// 5,000 vectors this scores 0.865 against centred's 0.852. The floor is
		// low because the hypothesis was that it would be bad, and leaving the
		// case in the sweep is how that stays checkable instead of remembered.
		{"positive", 0.40, func(r *rand.Rand) corpus { return positiveCorpus(r, n, dim) }},
		{"clustered", 0.85, func(r *rand.Rand) corpus { return clusteredCorpus(r, n, dim, 32) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(601))
			c := tc.build(rng)
			cfg := DefaultConfig(dim, Cosine)

			g, build := buildIndex(t, cfg, c)

			// Queries come from the same distribution as the corpus. Querying
			// clustered data with uniform noise would measure neither of them.
			queries := make([][]float32, 0, nq)
			qc := tc.build(rand.New(rand.NewSource(602)))
			for _, id := range qc.order[:min(nq, len(qc.order))] {
				queries = append(queries, qc.vecs[id])
			}

			recall, perQuery := evaluate(t, g, c, Cosine, queries, k, ef)
			record(t, sample{
				Sweep: "distribution", Label: tc.name,
				Dim: dim, N: n, M: cfg.M, Ef: ef, K: k,
				Recall: recall, Search: perQuery, Build: build,
			})

			if recall < tc.floor {
				t.Fatalf("%s recall@%d = %.3f, below its floor of %.2f", tc.name, k, recall, tc.floor)
			}
		})
	}
}

// TestSweepTombstones measures what deferred deletion costs *accuracy*, which
// the existing benchmark could not say — it only timed the search. It also
// measures the same graph after Compact, so the rebuild's benefit is a number
// rather than a claim.
func TestSweepTombstones(t *testing.T) {
	skipUnderRace(t)

	n, nq := sweepSize()
	const (
		dim = 128
		k   = 10
		ef  = 64
	)
	pcts := []int{0, 25, 50, 75}

	for _, pct := range pcts {
		t.Run(fmt.Sprintf("dead=%d%%", pct), func(t *testing.T) {
			rng := rand.New(rand.NewSource(701))
			c := uniformCorpus(rng, n, dim)
			cfg := DefaultConfig(dim, Cosine)
			g, build := buildIndex(t, cfg, c)

			// Delete a share of the corpus, and hold ground truth to the
			// survivors — recall is only meaningful against what remains.
			survivors := corpus{vecs: map[string][]float32{}, dim: dim}
			for i, id := range c.order {
				if i%100 < pct {
					if !g.Delete(id) {
						t.Fatalf("Delete(%s) removed nothing", id)
					}
					continue
				}
				survivors.vecs[id] = c.vecs[id]
				survivors.order = append(survivors.order, id)
			}

			queries := makeQueries(rng, nq, dim)
			recall, perQuery := evaluate(t, g, survivors, Cosine, queries, k, ef)
			record(t, sample{
				Sweep: "tombstones", Label: fmt.Sprintf("%d%%", pct),
				Dim: dim, N: len(survivors.order), M: cfg.M, Ef: ef, K: k,
				Recall: recall, Search: perQuery, Build: build,
			})

			// The same graph, compacted: same answers, less memory, and the
			// search width that tombstones were stealing given back.
			g.Compact()
			cRecall, cPerQuery := evaluate(t, g, survivors, Cosine, queries, k, ef)
			record(t, sample{
				Sweep: "compacted", Label: fmt.Sprintf("%d%%", pct),
				Dim: dim, N: len(survivors.order), M: cfg.M, Ef: ef, K: k,
				Recall: cRecall, Search: cPerQuery, Build: build,
			})

			if recall < 0.75 || cRecall < 0.75 {
				t.Fatalf("recall %.3f before compaction, %.3f after", recall, cRecall)
			}

			// Compaction can cost a little recall, and that is not a defect —
			// it is the tombstones' accidental subsidy being withdrawn.
			//
			// Dead slots keep `results` under-filled, which loosens the pruning
			// bound in searchLayer, which makes the search explore *wider* than
			// ef nominally asks for. That buys recall nobody asked for, at a
			// latency nobody wanted: measured at 50% dead, 0.968 recall for
			// 184µs before, 0.949 for 88µs after. The compacted graph wins that
			// trade outright — a slightly larger ef would recover the recall and
			// still be far quicker.
			//
			// The tolerance is what a fair comparison needs, not a threshold the
			// implementation is being held to.
			if cRecall < recall-0.03 {
				t.Fatalf("compaction cost more recall than the widened search it removed: %.3f -> %.3f", recall, cRecall)
			}
		})
	}
}

// TestGraphAcrossSmallDimensions runs the whole index at dimensions where the
// unrolled kernels are all tail and nothing else. Every other graph test uses a
// multiple of four, so this is the only place the integration is exercised.
//
// Recall is not the question here — at dim 1 or 2 the space barely has room for
// neighbors — so it asserts the property that must hold regardless: a stored
// vector retrieves itself.
func TestGraphAcrossSmallDimensions(t *testing.T) {
	for _, dim := range []int{1, 2, 3, 5, 7, 9, 17, 33} {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(800 + dim)))
			const n = 300
			c := uniformCorpus(rng, n, dim)
			g, _ := buildIndex(t, DefaultConfig(dim, Euclidean), c)

			if got := g.Len(); got != n {
				t.Fatalf("Len = %d, want %d", got, n)
			}
			for _, id := range c.order {
				res, err := g.Search(c.vecs[id], 1, 64)
				if err != nil {
					t.Fatal(err)
				}
				if len(res) == 0 {
					t.Fatalf("dim %d: %s returned nothing", dim, id)
				}
				// Duplicate positions are likely in one or two dimensions, so
				// require an exact-distance hit rather than the same id.
				if res[0].Distance > 1e-6 {
					t.Fatalf("dim %d: %s not found at its own position, nearest was %v away",
						dim, id, res[0].Distance)
				}
			}
		})
	}
}

// TestRecallIsStableAcrossSeeds guards against a single lucky seed. Every other
// recall test in this package pins one seed, which measures that configuration
// and not the index.
func TestRecallIsStableAcrossSeeds(t *testing.T) {
	skipUnderRace(t)

	const (
		dim = 64
		n   = 800
		k   = 10
		ef  = 64
	)
	seeds := []int64{1, 2, 3, 4, 5}

	var recalls []float64
	for _, seed := range seeds {
		rng := rand.New(rand.NewSource(seed))
		c := uniformCorpus(rng, n, dim)
		cfg := DefaultConfig(dim, Cosine)
		cfg.Seed = seed // vary the level draw too, not just the data

		g, _ := buildIndex(t, cfg, c)
		recall, _ := evaluate(t, g, c, Cosine, makeQueries(rng, 50, dim), k, ef)
		recalls = append(recalls, recall)
	}

	sort.Float64s(recalls)
	spread := recalls[len(recalls)-1] - recalls[0]
	mean := 0.0
	for _, r := range recalls {
		mean += r
	}
	mean /= float64(len(recalls))
	t.Logf("recall across %d seeds: mean %.4f, spread %.4f, min %.4f", len(seeds), mean, spread, recalls[0])

	if recalls[0] < 0.90 {
		t.Fatalf("worst seed scored %.4f", recalls[0])
	}
	// A configuration whose recall swings with the seed is not a configuration
	// anyone can rely on, whatever its average says.
	if spread > 0.05 {
		t.Fatalf("recall swings %.4f across seeds (%.4f..%.4f)", spread, recalls[0], recalls[len(recalls)-1])
	}
	if math.IsNaN(mean) {
		t.Fatal("mean recall is NaN")
	}
}
