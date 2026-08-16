package hnsw

import (
	"errors"
	"math"
	"math/rand"
	"slices"
	"sort"
)

var (
	// ErrDimensionMismatch is returned when a vector's length does not match
	// the dimension the graph was created with.
	ErrDimensionMismatch = errors.New("hnsw: vector dimension mismatch")
	// ErrEmptyVector is returned for a nil/zero-length vector.
	ErrEmptyVector = errors.New("hnsw: empty vector")
)

// Config controls how the graph is built. The zero value is not valid; use
// DefaultConfig and adjust.
type Config struct {
	// Dimension of every vector in the graph. Required.
	Dimension int
	// Metric selects the distance function.
	Metric Metric
	// M is the target number of neighbors per node on layers > 0. Higher M =
	// better recall, more memory, slower inserts. It is structural: it cannot
	// be changed after construction without rebuilding.
	M int
	// EfConstruction is how wide the search is during inserts. Higher = better
	// graph quality, slower inserts.
	EfConstruction int
	// Alpha is the pruning relaxation factor from the DiskANN/Vamana line of
	// work. When selecting neighbors we drop a candidate that sits closer to an
	// already-chosen neighbor than to the node itself — that edge is redundant,
	// you could reach it by hopping. Alpha scales that test:
	//
	//	1.0  classic HNSW heuristic
	//	>1.0 prunes harder, keeping more long-range "shortcut" edges, which
	//	     makes the graph more navigable and measurably lifts recall.
	//
	// 1.0-1.4 is the useful band; DefaultConfig uses 1.2.
	Alpha float32
	// Seed makes level assignment (and therefore the graph) reproducible.
	Seed int64
}

// DefaultConfig returns sensible defaults for the given dimension and metric.
func DefaultConfig(dimension int, metric Metric) Config {
	return Config{
		Dimension:      dimension,
		Metric:         metric,
		M:              16,
		EfConstruction: 200,
		Alpha:          1.2,
		Seed:           1,
	}
}

// Result is a single search hit.
type Result struct {
	ID       string
	Distance float32
}

// Graph is an HNSW index. It is NOT safe for concurrent use in this v1 cut;
// callers serialize access. Concurrency comes in a later step.
type Graph struct {
	cfg  Config
	dist DistanceFunc

	mMax  int     // max neighbors on layers > 0
	mMax0 int     // max neighbors on layer 0 (denser)
	ml    float64 // level-generation normalization factor = 1 / ln(M)
	alpha float32
	rng   *rand.Rand

	// normalized records that stored vectors are unit length, which lets the
	// cosine kernel collapse to a plain dot product.
	normalized bool

	nodes    []*node
	ids      map[string]int // external id -> index into nodes
	entry    int            // index of the entry point, -1 when empty
	maxLevel int

	// Scratch state reused across searches so the hot path allocates nothing.
	visited      visitedList
	scratchCands []candidate
	scratchRes   []candidate
	scratchSel   []candidate
	queryBuf     []float32
}

// New creates an empty graph. No memory is spent on the graph itself until the
// first vector arrives — an empty graph is just an empty container.
func New(cfg Config) (*Graph, error) {
	if cfg.Dimension <= 0 {
		return nil, errors.New("hnsw: dimension must be > 0")
	}
	if cfg.M <= 0 {
		cfg.M = 16
	}
	if cfg.EfConstruction <= 0 {
		cfg.EfConstruction = 200
	}
	if cfg.Alpha <= 0 {
		cfg.Alpha = 1
	}
	normalized := cfg.Metric.normalizes()
	return &Graph{
		cfg:        cfg,
		dist:       cfg.Metric.fastFunc(normalized),
		mMax:       cfg.M,
		mMax0:      cfg.M * 2,
		ml:         1.0 / math.Log(float64(cfg.M)),
		alpha:      cfg.Alpha,
		rng:        rand.New(rand.NewSource(cfg.Seed)),
		normalized: normalized,
		ids:        make(map[string]int),
		entry:      -1,
		maxLevel:   0,
	}, nil
}

// Len reports how many vectors are in the graph.
func (g *Graph) Len() int { return len(g.nodes) }

// randomLevel draws a layer for a new node from an exponentially decaying
// distribution: most nodes land on layer 0, a few reach higher layers.
func (g *Graph) randomLevel() int {
	return int(-math.Log(g.rng.Float64()) * g.ml)
}

// prepare returns a graph-owned copy of v, normalized when the metric wants it.
// Copying matters for correctness as well as normalization: without it the
// graph would alias the caller's slice and silently corrupt if they reused it.
func (g *Graph) prepare(v []float32) []float32 {
	out := make([]float32, len(v))
	copy(out, v)
	if g.normalized {
		Normalize(out)
	}
	return out
}

// Insert adds (or is a no-op for a duplicate id of) a vector into the graph.
func (g *Graph) Insert(id string, vector []float32) error {
	if len(vector) == 0 {
		return ErrEmptyVector
	}
	if len(vector) != g.cfg.Dimension {
		return ErrDimensionMismatch
	}
	if _, exists := g.ids[id]; exists {
		return nil // v1: ignore duplicates; update semantics come later
	}

	vec := g.prepare(vector)
	level := g.randomLevel()
	n := newNode(id, vec, level)
	idx := len(g.nodes)
	g.nodes = append(g.nodes, n)
	g.ids[id] = idx

	// First node ever: it becomes the entry point and we're done.
	if g.entry == -1 {
		g.entry = idx
		g.maxLevel = level
		return nil
	}

	// Phase 1: greedily descend from the top down to level+1 with ef=1, just to
	// find a good entry point near the new node.
	cur := g.entry
	for lc := g.maxLevel; lc > level; lc-- {
		cur = g.greedyClosest(cur, vec, lc)
	}

	// Phase 2: from min(maxLevel, level) down to 0, find neighbors and connect.
	start := min(level, g.maxLevel)
	for lc := start; lc >= 0; lc-- {
		w := g.searchLayer(vec, cur, g.cfg.EfConstruction, lc)
		neighbors := g.selectNeighbors(w, g.maxConn(lc))
		for _, nb := range neighbors {
			g.connect(idx, nb, lc)
			g.connect(nb, idx, lc)
			g.pruneConnections(nb, lc)
		}
		if len(w) > 0 {
			cur = w[0].idx // closest, to seed the next lower layer
		}
	}

	// If the new node reaches higher than any existing node, it becomes entry.
	if level > g.maxLevel {
		g.maxLevel = level
		g.entry = idx
	}
	return nil
}

// Search returns the k nearest neighbors to query. ef controls accuracy and is
// clamped up to at least k.
func (g *Graph) Search(query []float32, k, ef int) ([]Result, error) {
	if len(query) != g.cfg.Dimension {
		return nil, ErrDimensionMismatch
	}
	if g.entry == -1 || k <= 0 {
		return nil, nil
	}
	if ef < k {
		ef = k
	}

	// Normalize into a reusable buffer rather than touching the caller's slice.
	q := query
	if g.normalized {
		g.queryBuf = append(g.queryBuf[:0], query...)
		Normalize(g.queryBuf)
		q = g.queryBuf
	}

	// Descend the upper layers greedily to reach the right region.
	cur := g.entry
	for lc := g.maxLevel; lc > 0; lc-- {
		cur = g.greedyClosest(cur, q, lc)
	}

	// Do the wide search on layer 0.
	w := g.searchLayer(q, cur, ef, 0)

	// w is sorted ascending by distance; take the top k.
	if len(w) > k {
		w = w[:k]
	}
	out := make([]Result, len(w))
	for i, c := range w {
		out[i] = Result{ID: g.nodes[c.idx].id, Distance: c.dist}
	}
	return out, nil
}

// greedyClosest walks layer lc from a starting node, always stepping to the
// neighbor closest to target, until no neighbor is closer. Returns that node.
func (g *Graph) greedyClosest(start int, target []float32, lc int) int {
	best := start
	bestDist := g.dist(g.nodes[start].vector, target)
	for {
		improved := false
		for _, nb := range g.neighborsAt(best, lc) {
			d := g.dist(g.nodes[nb].vector, target)
			if d < bestDist {
				bestDist, best, improved = d, nb, true
			}
		}
		if !improved {
			return best
		}
	}
}

// searchLayer runs the core best-first search on a single layer, returning up
// to ef closest nodes to query, sorted ascending by distance.
func (g *Graph) searchLayer(query []float32, entryPoint, ef, lc int) []candidate {
	g.visited.reset(len(g.nodes))

	// Reuse the heap backing arrays; only the result slice is freshly allocated
	// because the caller keeps it.
	cands := g.scratchCands[:0]
	results := g.scratchRes[:0]

	d := g.dist(g.nodes[entryPoint].vector, query)
	g.visited.visit(entryPoint)
	cands = append(cands, candidate{entryPoint, d})
	results = append(results, candidate{entryPoint, d})

	var c candidate
	for len(cands) > 0 {
		c, cands = minPop(cands)
		// If the closest frontier node is farther than our worst result, every
		// remaining candidate is worse too — stop.
		if len(results) >= ef && c.dist > results[0].dist {
			break
		}
		for _, nb := range g.neighborsAt(c.idx, lc) {
			if g.visited.visit(nb) {
				continue
			}
			nd := g.dist(g.nodes[nb].vector, query)
			if len(results) < ef || nd < results[0].dist {
				cands = minPush(cands, candidate{nb, nd})
				results = maxPush(results, candidate{nb, nd})
				if len(results) > ef {
					_, results = maxPop(results) // drop the farthest
				}
			}
		}
	}

	out := make([]candidate, len(results))
	for i := len(out) - 1; i >= 0; i-- {
		out[i], results = maxPop(results) // farthest first -> fill from end
	}

	g.scratchCands, g.scratchRes = cands, results
	return out // ascending by distance
}

// selectNeighbors picks up to m edges for base out of cands (ascending by
// distance to base), using the alpha-relaxed diversity heuristic.
//
// Taking simply the m closest produces clustered edges that all point the same
// way, which strands the search in local minima. Instead we skip a candidate c
// when it already sits closer to a chosen neighbor s than it does to base —
// c is reachable via s, so that edge buys nothing. Alpha > 1 tightens the test,
// preserving more long-range links and making the graph easier to navigate.
// cands carry their distance to base already, so base itself is not needed.
func (g *Graph) selectNeighbors(cands []candidate, m int) []int {
	if len(cands) <= m {
		out := make([]int, len(cands))
		for i, c := range cands {
			out[i] = c.idx
		}
		return out
	}

	selected := make([]int, 0, m)
	rejected := g.scratchSel[:0]

	for _, c := range cands {
		if len(selected) >= m {
			break
		}
		keep := true
		for _, s := range selected {
			if g.alpha*g.dist(g.nodes[c.idx].vector, g.nodes[s].vector) <= c.dist {
				keep = false
				break
			}
		}
		if keep {
			selected = append(selected, c.idx)
		} else {
			rejected = append(rejected, c)
		}
	}

	// Backfill with the closest rejects rather than returning a thin list: a
	// node with too few edges is a dead end during search.
	for i := 0; len(selected) < m && i < len(rejected); i++ {
		selected = append(selected, rejected[i].idx)
	}

	g.scratchSel = rejected
	return selected
}

// pruneConnections trims a node's neighbor list on layer lc back to the layer
// cap, applying the same diversity heuristic used when inserting.
func (g *Graph) pruneConnections(idx, lc int) {
	maxConn := g.maxConn(lc)
	nbrs := g.neighborsAt(idx, lc)
	if len(nbrs) <= maxConn {
		return
	}

	// Compute each distance exactly once. Doing this inside a sort comparator
	// instead would recompute them O(n log n) times.
	self := g.nodes[idx].vector
	cands := make([]candidate, len(nbrs))
	for i, nb := range nbrs {
		cands[i] = candidate{nb, g.dist(g.nodes[nb].vector, self)}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].dist < cands[j].dist })

	g.nodes[idx].neighbors[lc] = g.selectNeighbors(cands, maxConn)
}

// connect adds `to` to `from`'s neighbor list on layer lc, skipping duplicates.
func (g *Graph) connect(from, to, lc int) {
	n := g.nodes[from]
	if slices.Contains(n.neighbors[lc], to) {
		return
	}
	n.neighbors[lc] = append(n.neighbors[lc], to)
}

// neighborsAt returns node idx's neighbor slice on layer lc (nil if the node
// does not reach that layer).
func (g *Graph) neighborsAt(idx, lc int) []int {
	n := g.nodes[idx]
	if lc > n.topLevel() {
		return nil
	}
	return n.neighbors[lc]
}

// maxConn is the neighbor cap for a layer: denser on layer 0.
func (g *Graph) maxConn(lc int) int {
	if lc == 0 {
		return g.mMax0
	}
	return g.mMax
}
