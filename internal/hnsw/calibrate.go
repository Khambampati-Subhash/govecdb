package hnsw

import (
	"errors"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"sync"
)

// Calibration exists because no fixed formula can choose ef well, and the
// measurements say so plainly. At 62,500 vectors and a 0.95 target, uniform
// random data at dimension 512 needs ef ≈ 3,072; tightly clustered data at the
// same size and dimension needs ef ≈ 10. The formula in suggest.go is fitted to
// uniform data at dimension 128 and sits between the two — wrong by 3x in one
// direction and by 100x in the other. Real embeddings are somewhere in that
// range, and where depends on the model and the corpus, not on anything the
// graph's size or configuration can tell it.
//
// So the graph measures itself. Calibrate takes a sample of its own vectors as
// queries, finds their true nearest neighbors by brute force, and finds the
// smallest ef whose search recovers them at the target recall. The ratio of that
// to what the formula would have said becomes a scale on every later
// suggestion, which keeps the formula's shape — how ef grows with n, k and M —
// and replaces only the part of it that was a guess about the data.

// MinCalibrationSize is the smallest graph Calibrate measures. Below it the
// formula's widths are tiny either way, the brute-force truth is cheap enough
// that nothing is saved, and the sample would be a large fraction of the data.
const MinCalibrationSize = 1000

// defaultCalibrationQueries is the sample size when CalibrateOptions.Queries is
// zero — see Queries for the measurement behind it.
const defaultCalibrationQueries = 512

// Scale bounds. A calibration is a measurement on a sample, and a pathological
// sample should not be able to make every later search either exhaustive or
// width-k.
const (
	minEfScale = 1.0 / 256
	maxEfScale = 16
)

// ErrCalibrationStopped is returned when CalibrateOptions.Stop closes first.
// The graph's scale is left as it was.
var ErrCalibrationStopped = errors.New("hnsw: calibration stopped")

// CalibrateOptions configures a calibration. The zero value is usable.
type CalibrateOptions struct {
	// Queries is how many of the graph's own vectors to search for. Zero means
	// 512, or half the graph if that is fewer. Since the margin multiplies
	// whatever is found, a low draw is a search that misses its target, and
	// near 0.99 the recall curve is flat enough that a few samples move the
	// answer a long way: on 6,585 real passages the width found for 0.99
	// ranged 14–35 across twelve seeds at 128 queries, and 20–25 at 512. The
	// lowest draw at 128, times the margin, gave real questions 0.982. The
	// brute force is linear in it.
	Queries int
	// K is the result count recall is measured at. Zero means 10, the k the
	// formula is anchored at.
	K int
	// Target is the recall to find a width for. Zero means 0.95.
	Target float64
	// Seed picks the sample, so a calibration is reproducible.
	Seed int64
	// Stop abandons the calibration when closed. It is checked between queries,
	// so a large graph's brute force does not hold up a shutdown.
	Stop <-chan struct{}
}

// Calibration reports what Calibrate measured.
type Calibration struct {
	// Live is how many vectors the graph held when measured. Zero means there
	// was nothing to measure and the scale was reset to 1: the graph was too
	// small, or most of the sample was boilerplate (see maxHiddenCopies).
	Live int
	// Queries is how many sample queries were run: those asked for, less any
	// dropped as boilerplate.
	Queries int
	// Ef is the smallest width found that reached Target on the sample.
	Ef int
	// Reached is false if even a search as wide as the graph missed Target. Ef
	// is then the graph's size and Scale is capped; the graph is badly
	// connected for this data, and no width will fix that.
	Reached bool
	// Prior is what the uncalibrated formula suggested for the same n, k and
	// target.
	Prior int
	// Scale is now applied to every SuggestedEf: calibrationMargin·Ef/Prior,
	// clamped.
	Scale float64
}

// EfScale is the factor the last Calibrate applied to SuggestedEf; 1 when
// there has been none.
func (g *Graph) EfScale() float64 {
	if s := math.Float64frombits(g.efScale.Load()); s > 0 {
		return s
	}
	return 1
}

// Calibrate measures the search width this graph needs and scales SuggestedEf
// to match. See the note at the top of this file for why.
//
// # Cost, and what it holds
//
// The brute-force truth is Queries × Live distance computations, spread across
// GOMAXPROCS goroutines: at 768 dimensions and the default 512 queries it
// measured 380 ms per 100,000 vectors on a 16-core laptop — ~4 s at a million,
// three times the cost of 128 queries for a width that stops depending on the
// seed. It runs **outside** the graph's lock, over a copy of the (vector, slot,
// dead) references taken under the read lock — 33 bytes per slot, transient —
// so writers are excluded only for that copy. The search probes afterwards are
// ordinary Searches, one goroutine, and run alongside everything else: 130 ms
// in all on 6,585 real passages.
//
// Writes keep landing, and the database calibrates *during* ingest — when the
// live count doubles — so the truth follows them rather than describing a graph
// that no longer exists; see truth. A Compact renumbers every slot the truth
// refers to, and a calibration that sees one starts over, at most
// calibrationAttempts times.
//
// # Leave one out, or the measurement is worthless
//
// The sample queries are stored vectors, and searching for a stored vector is
// trivially easy: the descent lands on the vector itself and its neighbor list
// *is* its approximate nearest neighbors, chosen at insert. Measured that way,
// uniform data "reached" 0.95 at ef=11 while held-out queries got 0.37 at the
// width it then suggested. So each sample is searched with its own slot hidden
// from the traversal — never expanded, never an answer — which is the graph as
// a query that was never inserted would see it. Its neighbors keep edges that
// were chosen with it present, so this is, if anything, slightly pessimistic.
//
// A real query is harder than a left-out passage, and by more than intuition
// says: a question embedded as a query sits off the manifold the passages lie
// on, and on real books it needed 2.5–5.5× the width leave-one-out did at the
// same recall. calibrationMargin covers that — see it for the measurement. This
// is a much better estimate than a formula that knows nothing about the data,
// but it is an estimate of the data's difficulty, not a guarantee about every
// query.
func (g *Graph) Calibrate(o CalibrateOptions) (Calibration, error) {
	if o.K <= 0 {
		o.K = 10
	}
	if o.Target <= 0 || o.Target >= 1 {
		o.Target = 0.95
	}
	for attempt := 1; ; attempt++ {
		c, err := g.calibrate(o)
		if !errors.Is(err, errGraphMoved) || attempt == calibrationAttempts {
			return c, err
		}
	}
}

// calibrationAttempts bounds how often a calibration starts over because the
// graph changed under it. Past that it fails and the previous scale stands: a
// width that is out of date is a latency question, a measurement against a
// truth that is wrong is an accuracy one.
const calibrationAttempts = 3

// errGraphMoved is a calibration whose truths stopped describing the graph
// faster than it could measure it: a Compact renumbered the slots, or deletes
// removed the answers of most of the sample.
var errGraphMoved = errors.New("hnsw: the graph changed too much during calibration")

// calibrate is one attempt at Calibrate.
func (g *Graph) calibrate(o CalibrateOptions) (Calibration, error) {
	r := g.refsFrom(0)
	if o.Queries <= 0 {
		o.Queries = min(defaultCalibrationQueries, r.live/2)
	}
	if r.live < max(MinCalibrationSize, 2*o.Queries) {
		g.efScale.Store(0)
		return Calibration{Live: 0, Scale: 1}, nil
	}

	live := make([]int, 0, r.live)
	for j := range r.vecs {
		if !r.dead[j] {
			live = append(live, j)
		}
	}
	rng := rand.New(rand.NewSource(o.Seed))
	truths := make([]truth, o.Queries)
	for i, j := range rng.Perm(len(live))[:o.Queries] {
		q := r.vecs[live[j]]
		truths[i] = truth{q: q, self: g.dist(q, q), best: make([]candidate, 0, o.K)}
	}
	if err := g.fold(truths, r, o.K, o.Stop); err != nil {
		return Calibration{}, err
	}
	if measurable(truths) < len(truths)/2 {
		// Mostly mass copies: there is no novel query to imitate, so there is
		// nothing to measure. The formula is the honest fallback.
		g.efScale.Store(0)
		return Calibration{Live: 0, Scale: 1}, nil
	}
	if g.onCalibrateProbe != nil {
		g.onCalibrateProbe()
	}

	recallAt := func(ef int) (float64, error) {
		// The bulk of whatever landed since the last round is folded in here,
		// outside the lock; hitsWithout folds the rest under it.
		late := g.refsFrom(slices.MinFunc(truths, func(a, b truth) int { return a.seen - b.seen }).seen)
		if late.layout != r.layout {
			return 0, errGraphMoved
		}
		if err := g.fold(truths, late, o.K, o.Stop); err != nil {
			return 0, err
		}
		hits, used := 0, 0
		for i := range truths {
			t := &truths[i]
			if t.dropped {
				continue
			}
			if stopped(o.Stop) {
				return 0, ErrCalibrationStopped
			}
			h, ok := g.hitsWithout(t, o.K, ef, r.layout)
			if !ok {
				t.dropped = true
				continue
			}
			hits += h
			used++
		}
		if used < len(truths)/2 {
			return 0, errGraphMoved
		}
		return float64(hits) / float64(o.K*used), nil
	}

	// Double until the target is reached, then bisect the last doubling to
	// within an eighth. Recall is monotone in ef up to sampling noise, which is
	// all a bisection needs. lo starts just below k because a width below k is
	// not a search anyone can ask for. The graph may grow while this runs, so
	// the widest search is whatever the graph holds now.
	lo, hi := o.K-1, o.K
	reached := false
	for {
		r, err := recallAt(hi)
		if err != nil {
			return Calibration{}, err
		}
		if r >= o.Target {
			reached = true
			break
		}
		n := g.Len()
		if hi >= n {
			break
		}
		lo, hi = hi, min(2*hi, n)
	}
	if reached {
		for hi-lo > max(1, lo/8) {
			mid := lo + (hi-lo)/2
			r, err := recallAt(mid)
			if err != nil {
				return Calibration{}, err
			}
			if r >= o.Target {
				hi = mid
			} else {
				lo = mid
			}
		}
	}

	// The prior is taken at the size the last probes searched, not the size
	// the calibration started at: hi was measured there, and the scale is a
	// ratio between the two at one size.
	n := g.Len()
	prior := suggestEf(n, o.K, o.Target, g.cfg.M, 1)
	scale := calibrationMargin * float64(hi) / float64(prior)
	if !reached {
		scale = maxEfScale
	}
	scale = min(max(scale, minEfScale), maxEfScale)
	g.efScale.Store(math.Float64bits(scale))

	return Calibration{
		Live: n, Queries: measurable(truths), Ef: hi, Reached: reached, Prior: prior, Scale: scale,
	}, nil
}

// refs is a copy of vector references taken under the read lock: the slots in
// [from, end), and the layout their numbers belong to. Tombstones are included
// and marked: one is never an answer, but one that copies a sample still has
// to be hidden with it — see "Copies are the query too" on truth.
type refs struct {
	vecs   [][]float32
	slots  []int
	dead   []bool
	live   int
	end    int
	layout uint64
}

// refsFrom copies the vector references at slot from and above. Stored
// vectors are never mutated in place — an update builds a new slot, and Compact
// reuses the same slices — so the references stay valid after the lock is
// released and distances to them can be computed outside it. Slots are only
// ever appended, so two copies taken under one layout differ by exactly the
// slots placed in between.
func (g *Graph) refsFrom(from int) refs {
	g.mu.RLock()
	defer g.mu.RUnlock()
	r := refs{end: len(g.nodes), layout: g.layout}
	if from >= len(g.nodes) {
		return r
	}
	room := len(g.nodes) - from
	r.vecs = make([][]float32, room)
	r.slots = make([]int, room)
	r.dead = make([]bool, room)
	for j := range room {
		nd := g.nodes[from+j]
		r.vecs[j], r.slots[j], r.dead[j] = nd.vector, from+j, nd.deleted
		if !nd.deleted {
			r.live++
		}
	}
	return r
}

// maxHiddenCopies bounds how many exact copies of a sample hitsWithout hides.
// A vector stored more often than this is boilerplate — a page footer, a blank
// page — not a stand-in for a query anyone will send, so the sample is dropped
// instead. It also bounds the upper-layer descent's linear scan of the list.
const maxHiddenCopies = 64

// truth is one sample's exact answer: the k nearest vectors in the graph that
// are not copies of it, and the copies, which are hidden with it. fold builds
// it by an exact scan; hitsWithout scores a search against it.
//
// # The truth follows the graph
//
// It is folded forward to every slot placed since it last looked, before each
// probe — the bulk outside the lock, the residue under the read lock the probe
// already holds — so each probe is scored against the exact k nearest of the
// graph it actually searched. Freezing it at the start was measured wrong both
// ways, on real books ingested a second time while they were being calibrated:
//
//   - Counted by distance against the old radius, a late vector inside it was a
//     hit. The radius was drawn around the old population and now held two
//     vectors for every one the truth counted, so filling k from it was easier
//     than finding the true k: the calibration chose a width that gave real
//     questions 0.977 against a 0.99 target.
//   - Passed over, the sample's own late copy was still in the graph, and the
//     search walked straight to it — the self-query the leave-one-out exists to
//     prevent. Ef fell from 70 to 22 on the same graph.
//
// A deleted vector cannot be folded out: the k-th nearest that would replace it
// is not kept. So a truth whose answers lose a member is retired instead, and a
// calibration that loses half its sample that way starts over.
//
// # Why a radius and not a set of ids
//
// Recall measured by id counts an exact tie as a miss. A corpus that holds a
// document twice — the same PDF ingested twice, a boilerplate page in every
// chapter — has a twin at exactly the k-th distance for many queries, and
// which one a search returns is arbitrary. By id, 6,585 real passages ingested
// twice could not reach 0.99 at any width: Calibrate reported Reached false and
// scaled every later search sixteenfold, to ~9,300 on a 13,170-vector graph.
// Any result within the k-th distance is one of the k nearest.
//
// # Copies are the query too
//
// Leaving one out only works if what is left out is everything that makes the
// search trivial, and a copy of the sample does exactly that: its neighbor
// list was chosen beside the sample's, so it is the sample's nearest neighbors
// again. On the same twice-ingested corpus, hiding only the sample's own slot
// measured recall 1.000 at ef=10 for every sample — the twin answered for it —
// so calibration chose ef=10 at a 0.99 target, the narrowest width there is.
// With copies hidden it chooses 35–45, and real questions get 0.995. So every
// vector the metric cannot tell apart from the sample (distance no greater
// than the sample's to itself) is hidden with it and excluded from the truth.
//
// That includes copies that have since been deleted. A tombstone keeps its
// edges — it is still a bridge — so a vector stored, deleted and stored again
// (a document re-indexed, an upsert back to an earlier value) has a dead twin
// that walks the search into its neighborhood as surely as a live one. On a
// database churned by kill -9 rounds of upserts, deletes and re-adds over real
// books — 10,384 live, 3,850 dead — leave-one-out measured ef 22–30 with dead
// copies visible and 30–40 with them hidden; the same vectors compacted
// measured 32–55, and real questions needed ~150 on both. The visible twins
// gave the questions 0.983 against a 0.99 target. A dead copy is never an
// answer: the search cannot return one.
type truth struct {
	q []float32
	// self is the query's distance to itself. Anything this close is a copy.
	self float32
	// best holds the k nearest non-copies, nearest first, by slot.
	best []candidate
	// hide holds the slots of the query and every copy of it.
	hide []int
	// seen is the slot count the truth has been folded up to.
	seen int
	// dropped retires the sample: more than maxHiddenCopies copies, or an
	// answer deleted since it was found.
	dropped bool
}

// radius is the distance to the k-th nearest non-copy: a result is one of the
// k nearest if it lies within it.
func (t *truth) radius(k int) float32 {
	if len(t.best) < k {
		return float32(math.Inf(1))
	}
	return t.best[k-1].dist
}

// add takes one more vector into the truth: a copy into the hidden list, dead
// or alive, and a live non-copy into the answers if it is near enough.
func (t *truth) add(d float32, slot, k int, dead bool) {
	if d <= t.self {
		// The sample itself lands here too.
		if len(t.hide) > maxHiddenCopies {
			t.dropped = true
			return
		}
		t.hide = append(t.hide, slot)
		return
	}
	if dead || len(t.best) == k && d >= t.best[k-1].dist {
		return
	}
	// Insertion into a sorted run of k: k is 10, so this beats a heap and
	// allocates nothing.
	j := len(t.best)
	if j < k {
		t.best = append(t.best, candidate{})
	} else {
		j = k - 1
	}
	for j > 0 && t.best[j-1].dist > d {
		t.best[j] = t.best[j-1]
		j--
	}
	t.best[j] = candidate{idx: slot, dist: d}
}

// measurable counts the samples still in the measurement.
func measurable(truths []truth) int {
	n := 0
	for i := range truths {
		if !truths[i].dropped {
			n++
		}
	}
	return n
}

// fold brings every truth up to date with the references in r, by an exact
// scan spread across GOMAXPROCS goroutines, one truth at a time. It uses the
// graph's own distance function on stored-form vectors, with the arguments in
// the order searchLayer passes them, so a search finding the same vector
// computes the same float. A truth skips the slots it has already seen, so
// the initial brute force and every catch-up after it are the same pass.
func (g *Graph) fold(truths []truth, r refs, k int, stop <-chan struct{}) error {
	work := make(chan int)
	var wg sync.WaitGroup
	var stopErr error

	for range min(runtime.GOMAXPROCS(0), len(truths)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for qi := range work {
				t := &truths[qi]
				j, _ := slices.BinarySearch(r.slots, t.seen)
				for ; j < len(r.slots) && !t.dropped; j++ {
					t.add(g.dist(r.vecs[j], t.q), r.slots[j], k, r.dead[j])
				}
				t.seen = max(t.seen, r.end)
			}
		}()
	}

	for qi := range truths {
		if stopped(stop) {
			stopErr = ErrCalibrationStopped
			break
		}
		if !truths[qi].dropped && truths[qi].seen < r.end {
			work <- qi
		}
	}
	close(work)
	wg.Wait()
	return stopErr
}

// hitsWithout is Search for a sample with it and its copies hidden from the
// traversal, returning how many of its k results are true answers. See "Leave
// one out" on Calibrate. It reports false when the truth no longer describes
// the graph — renumbered by a Compact, or an answer since deleted — and the
// sample is then retired rather than scored.
//
// Under the read lock it first folds in whatever was placed since the truth
// last looked, so the truth, the hidden list and the graph searched are one
// instant's.
//
// The hidden slots are skipped on the way down as well as on layer 0: the
// upper layers are where a stored vector's own position would otherwise pull
// the descent straight to it. If one is the entry point there is nowhere to
// start that does not go through it, so that sample answers nothing — a miss,
// which errs in the safe direction and happens to one slot in the graph.
func (g *Graph) hitsWithout(t *truth, k, ef int, layout uint64) (int, bool) {
	st := g.acquireState()
	defer g.releaseState(st)

	g.mu.RLock()
	defer g.mu.RUnlock()
	if g.layout != layout {
		return 0, false
	}
	for i := t.seen; i < len(g.nodes) && !t.dropped; i++ {
		nd := g.nodes[i]
		t.add(g.dist(nd.vector, t.q), i, k, nd.deleted)
	}
	t.seen = max(t.seen, len(g.nodes))
	if t.dropped {
		return 0, false
	}
	for _, b := range t.best {
		if g.nodes[b.idx].deleted {
			return 0, false
		}
	}
	if g.entry == -1 || slices.Contains(t.hide, g.entry) {
		return 0, true
	}

	cur := g.entry
	for lc := g.maxLevel; lc > 0; lc-- {
		bestDist := g.dist(g.nodes[cur].vector, t.q)
		for improved := true; improved; {
			improved = false
			for _, nb := range g.neighborsAt(cur, lc) {
				if slices.Contains(t.hide, int(nb)) {
					continue
				}
				if d := g.dist(g.nodes[nb].vector, t.q); d < bestDist {
					bestDist, cur, improved = d, int(nb), true
				}
			}
		}
	}

	st.hide = append(st.hide[:0], t.hide...)
	w := g.searchLayer(st, t.q, cur, max(ef, k), 0, nil)
	st.hide = st.hide[:0]

	// Nothing at the sample's own distance can come back — every copy is
	// hidden — but the bound is the definition of a hit, so it is spelled out.
	hits, radius := 0, t.radius(k)
	for _, c := range w[:min(k, len(w))] {
		if t.self < c.dist && c.dist <= radius {
			hits++
		}
	}
	return hits, true
}

func stopped(stop <-chan struct{}) bool {
	if stop == nil {
		return false
	}
	select {
	case <-stop:
		return true
	default:
		return false
	}
}
