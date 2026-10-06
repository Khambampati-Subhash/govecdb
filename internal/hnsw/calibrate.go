package hnsw

import (
	"errors"
	"math"
	"math/rand"
	"runtime"
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
// that nothing is saved, and a 64-query sample is a large fraction of the data.
const MinCalibrationSize = 1000

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
	// 64: enough that one unlucky query moves the measured recall by a point and
	// a half, few enough that the brute-force truth stays well under a second
	// at a million vectors across a few cores.
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
	// Live is how many vectors the graph held when measured. Zero means the
	// graph was too small to calibrate and the scale was reset to 1.
	Live int
	// Queries is how many sample queries were run.
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
// GOMAXPROCS goroutines: at a million 768-dimension vectors and 64 queries that
// is ~6 s of CPU, well under a second of wall time on a laptop. It runs
// **outside** the graph's lock, over a copy of the live (id, vector) references
// taken under the read lock — 40 bytes per vector, transient — so writers are
// excluded only for that copy. The search probes afterwards are ordinary
// Searches and run alongside everything else.
//
// A vector deleted while a calibration runs may still be in its truth, which
// makes the measurement very slightly pessimistic. That is the right direction.
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
// A real query also comes from a slightly different distribution when it is a
// different kind of text (a question against passages), and is usually a
// little harder still. calibrationMargin is there for that: this is a much
// better estimate than a formula that knows nothing about the data, but it is
// an estimate of the data's difficulty, not a guarantee about every query.
func (g *Graph) Calibrate(o CalibrateOptions) (Calibration, error) {
	if o.Queries <= 0 {
		o.Queries = 64
	}
	if o.K <= 0 {
		o.K = 10
	}
	if o.Target <= 0 || o.Target >= 1 {
		o.Target = 0.95
	}

	ids, vecs, slots := g.liveRefs()
	n := len(ids)
	if n < max(MinCalibrationSize, 2*o.Queries) {
		g.efScale.Store(0)
		return Calibration{Live: 0, Scale: 1}, nil
	}

	rng := rand.New(rand.NewSource(o.Seed))
	sample := rng.Perm(n)[:o.Queries]

	truth, err := g.bruteForce(ids, vecs, sample, o.K, o.Stop)
	if err != nil {
		return Calibration{}, err
	}

	recallAt := func(ef int) (float64, error) {
		hits := 0
		for qi, pos := range sample {
			if stopped(o.Stop) {
				return 0, ErrCalibrationStopped
			}
			for _, id := range g.searchWithout(vecs[pos], o.K, ef, slots[pos]) {
				if _, ok := truth[qi][id]; ok {
					hits++
				}
			}
		}
		return float64(hits) / float64(o.K*len(sample)), nil
	}

	// Double until the target is reached, then bisect the last doubling to
	// within an eighth. Recall is monotone in ef up to sampling noise, which is
	// all a bisection needs. lo starts just below k because a width below k is
	// not a search anyone can ask for.
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

	prior := suggestEf(n, o.K, o.Target, g.cfg.M, 1)
	scale := calibrationMargin * float64(hi) / float64(prior)
	if !reached {
		scale = maxEfScale
	}
	scale = min(max(scale, minEfScale), maxEfScale)
	g.efScale.Store(math.Float64bits(scale))

	return Calibration{
		Live: n, Queries: len(sample), Ef: hi, Reached: reached, Prior: prior, Scale: scale,
	}, nil
}

// liveRefs copies the live ids, vector references and slots under the read
// lock. Stored vectors are never mutated in place — an update builds a new
// slot, and Compact reuses the same slices — so the references stay valid after
// the lock is released. A slot can be renumbered by a Compact that runs in the
// meantime; searchWithout then hides the wrong node, which costs one sample's
// accuracy and nothing else.
func (g *Graph) liveRefs() (ids []string, vecs [][]float32, slots []int) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	live := len(g.nodes) - g.numDeleted
	ids = make([]string, 0, live)
	vecs = make([][]float32, 0, live)
	slots = make([]int, 0, live)
	for i, nd := range g.nodes {
		if !nd.deleted {
			ids = append(ids, nd.id)
			vecs = append(vecs, nd.vector)
			slots = append(slots, i)
		}
	}
	return ids, vecs, slots
}

// searchWithout is Search for a stored-form query with one slot hidden from
// the traversal, returning ids only. See "Leave one out" on Calibrate.
//
// The hidden slot is skipped on the way down as well as on layer 0: the upper
// layers are where a stored vector's own position would otherwise pull the
// descent straight to it. If it is the entry point there is nowhere to start
// that does not go through it, so that sample answers nothing — a miss, which
// errs in the safe direction and happens to one slot in the graph.
func (g *Graph) searchWithout(q []float32, k, ef, hide int) []string {
	st := g.acquireState()
	defer g.releaseState(st)

	g.mu.RLock()
	defer g.mu.RUnlock()
	if g.entry == -1 || g.entry == hide || hide >= len(g.nodes) {
		return nil
	}

	cur := g.entry
	for lc := g.maxLevel; lc > 0; lc-- {
		bestDist := g.dist(g.nodes[cur].vector, q)
		for improved := true; improved; {
			improved = false
			for _, nb := range g.neighborsAt(cur, lc) {
				if nb == hide {
					continue
				}
				if d := g.dist(g.nodes[nb].vector, q); d < bestDist {
					bestDist, cur, improved = d, nb, true
				}
			}
		}
	}

	st.hide = hide + 1
	w := g.searchLayer(st, q, cur, max(ef, k), 0, nil)
	st.hide = 0

	out := make([]string, 0, min(k, len(w)))
	for _, c := range w[:min(k, len(w))] {
		out = append(out, g.nodes[c.idx].id)
	}
	return out
}

// bruteForce returns, for each sampled position, the ids of its k nearest
// other vectors. It uses the graph's own distance function on stored-form
// vectors, so the truth is on exactly the scale the search ranks by.
func (g *Graph) bruteForce(ids []string, vecs [][]float32, sample []int, k int, stop <-chan struct{}) ([]map[string]struct{}, error) {
	truth := make([]map[string]struct{}, len(sample))
	work := make(chan int)
	var wg sync.WaitGroup
	var stopErr error

	for range min(runtime.GOMAXPROCS(0), len(sample)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			best := make([]candidate, 0, k+1)
			for qi := range work {
				q := vecs[sample[qi]]
				best = best[:0]
				for i, v := range vecs {
					if i == sample[qi] {
						continue
					}
					d := g.dist(q, v)
					if len(best) == k && d >= best[k-1].dist {
						continue
					}
					// Insertion into a sorted run of k: k is 10, so this
					// beats a heap and allocates nothing.
					j := len(best)
					if j < k {
						best = append(best, candidate{})
					} else {
						j = k - 1
					}
					for j > 0 && best[j-1].dist > d {
						best[j] = best[j-1]
						j--
					}
					best[j] = candidate{idx: i, dist: d}
				}
				set := make(map[string]struct{}, len(best))
				for _, c := range best {
					set[ids[c.idx]] = struct{}{}
				}
				truth[qi] = set
			}
		}()
	}

	for qi := range sample {
		if stopped(stop) {
			stopErr = ErrCalibrationStopped
			break
		}
		work <- qi
	}
	close(work)
	wg.Wait()
	return truth, stopErr
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
