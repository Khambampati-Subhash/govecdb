package hnsw

import (
	"cmp"
	"math"
	"slices"
)

// A filtered search either walks the graph or scans every live slot, and
// SearchFilter picks one per query from how selective the filter looks.
//
// # Why a scan is ever the right answer
//
// The traversal admits a node into its results only if the filter accepts it,
// and it stops once it holds ef admitted nodes and nothing on the frontier is
// closer. So it has to *find* about ef matching nodes, and at selectivity p it
// meets them at rate p: measured, it visits roughly min(N, 3·ef/p) nodes, to
// within ~30% from one in two down to one in five thousand. A visit is a
// random memory access, a neighbor list and a distance — 207 ns at dim 32,
// ~350 ns at 128, 375 ns at 512. A scan reads the slots in order and computes a
// distance only for the nodes the filter accepts: ~15 ns for each one it
// rejects, whatever the dimension.
//
// So the scan wins once the matches it would find, p·N, fall below about
// 3·ef·(visit cost / scan cost) — 41·ef at dim 32, 60·ef at 128, 75·ef at 512.
// At 100K × 128 and one in a thousand, the walk took 25–34 ms and visited the
// whole graph to return ten results; the scan took 1.8 ms and is exact.
// scanPerEf is set below all three fits, so near the line the graph is
// preferred: past it either plan costs within ~1.5× of the other.
//
// # Why sample rather than keep statistics
//
// The predicate is opaque — a closure over a metadata store the index knows
// nothing about, possibly a Filter a caller wrote. Statistics would need the
// store to maintain per-key distributions on every write and a planner to
// combine them through And/Or/Not; asking the predicate itself about a handful
// of nodes answers the question that matters, for any predicate, and costs the
// write path nothing.
//
// The probes stop as soon as the answer is clear. A filter well above the line
// is recognized after a few dozen — on the order of 10 µs, against a search of
// hundreds — and only a filter near or below it is probed to the full count,
// which is where the search it might replace costs milliseconds.
const (
	// scanPerEf is the expected match count, per unit of ef, below which a
	// scan is cheaper than the walk. See the note above for the fits.
	scanPerEf = 48

	// sampleMatches is how many matches the probe expects to see at exactly
	// the threshold rate; it stops early, choosing the walk, at twice this.
	sampleMatches = 16
)

// shouldScan reports whether a filtered search over this graph should scan
// rather than walk. Callers hold the read lock.
func (g *Graph) shouldScan(allow func(id string) bool, ef int) bool {
	live := len(g.nodes) - g.numDeleted
	threshold := scanPerEf * ef
	if live <= threshold {
		// Fewer live nodes than the walk would expect to visit: the scan reads
		// them all once, the walk would read many of them several times over.
		return true
	}

	// Probe enough live nodes to expect sampleMatches matches if the filter sat
	// exactly on the threshold, i.e. at rate threshold/live.
	budget := (sampleMatches*live + threshold - 1) / threshold

	// Slots are visited in a golden-ratio sequence: evenly spread over the
	// whole graph, deterministic, and blind to how slots were filled — so a
	// filter matching one contiguous run of inserts (one tenant's import) is
	// sampled in proportion, which a stride from zero can miss.
	n := len(g.nodes)
	pos := 0.5
	probed, matched := 0, 0
	for tries := 0; probed < budget && tries < 4*budget; tries++ {
		pos += math.Phi - 1
		if pos >= 1 {
			pos--
		}
		nd := g.nodes[int(pos*float64(n))]
		if nd.deleted {
			continue
		}
		probed++
		if allow(nd.id) {
			matched++
			if matched >= 2*sampleMatches {
				return false // well above the threshold: walk
			}
		}
	}
	// probed < budget only when the graph is mostly tombstones; scale up.
	return matched*budget < sampleMatches*max(probed, 1)
}

// scanFilter returns the k live nodes nearest q that allow accepts, exactly,
// by reading every slot once. Callers hold the read lock.
//
// It allocates nothing beyond the result the caller builds: the bounded
// max-heap is the search state's own results buffer.
func (g *Graph) scanFilter(st *searchState, q []float32, k int, allow func(id string) bool) []candidate {
	best := st.results[:0]
	for i, nd := range g.nodes {
		if nd.deleted || !allow(nd.id) {
			continue
		}
		d := g.dist(nd.vector, q)
		if len(best) < k {
			best = maxPush(best, candidate{i, d})
		} else if d < best[0].dist {
			_, best = maxPop(best)
			best = maxPush(best, candidate{i, d})
		}
	}
	st.results = best
	slices.SortFunc(best, func(a, b candidate) int { return cmp.Compare(a.dist, b.dist) })
	return best
}
