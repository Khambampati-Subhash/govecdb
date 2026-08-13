package hnsw

import (
	"container/heap"
	"errors"
	"math"
	"math/rand"
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
	rng   *rand.Rand

	nodes    []*node
	ids      map[string]int // external id -> index into nodes
	entry    int            // index of the entry point, -1 when empty
	maxLevel int
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
	return &Graph{
		cfg:      cfg,
		dist:     cfg.Metric.Func(),
		mMax:     cfg.M,
		mMax0:    cfg.M * 2,
		ml:       1.0 / math.Log(float64(cfg.M)),
		rng:      rand.New(rand.NewSource(cfg.Seed)),
		ids:      make(map[string]int),
		entry:    -1,
		maxLevel: 0,
	}, nil
}

// Len reports how many vectors are in the graph.
func (g *Graph) Len() int { return len(g.nodes) }

// randomLevel draws a layer for a new node from an exponentially decaying
// distribution: most nodes land on layer 0, a few reach higher layers.
func (g *Graph) randomLevel() int {
	return int(-math.Log(g.rng.Float64()) * g.ml)
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

	level := g.randomLevel()
	n := newNode(id, vector, level)
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
		cur = g.greedyClosest(cur, vector, lc)
	}

	// Phase 2: from min(maxLevel, level) down to 0, find neighbors and connect.
	start := min(level, g.maxLevel)
	for lc := start; lc >= 0; lc-- {
		w := g.searchLayer(vector, []int{cur}, g.cfg.EfConstruction, lc)
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

	// Descend the upper layers greedily to reach the right region.
	cur := g.entry
	for lc := g.maxLevel; lc > 0; lc-- {
		cur = g.greedyClosest(cur, query, lc)
	}

	// Do the wide search on layer 0.
	w := g.searchLayer(query, []int{cur}, ef, 0)

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
func (g *Graph) searchLayer(query []float32, entryPoints []int, ef, lc int) []candidate {
	visited := make(map[int]struct{}, ef*2)
	cands := &minHeap{} // frontier: explore closest first
	results := &maxHeap{} // best-so-far: drop farthest when over ef

	for _, ep := range entryPoints {
		d := g.dist(g.nodes[ep].vector, query)
		visited[ep] = struct{}{}
		heap.Push(cands, candidate{ep, d})
		heap.Push(results, candidate{ep, d})
	}

	for cands.Len() > 0 {
		c := heap.Pop(cands).(candidate)
		// If the closest frontier node is farther than our worst result, stop.
		if results.Len() >= ef && c.dist > (*results)[0].dist {
			break
		}
		for _, nb := range g.neighborsAt(c.idx, lc) {
			if _, seen := visited[nb]; seen {
				continue
			}
			visited[nb] = struct{}{}
			d := g.dist(g.nodes[nb].vector, query)
			if results.Len() < ef || d < (*results)[0].dist {
				heap.Push(cands, candidate{nb, d})
				heap.Push(results, candidate{nb, d})
				if results.Len() > ef {
					heap.Pop(results) // drop the farthest
				}
			}
		}
	}

	out := make([]candidate, results.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(results).(candidate) // pops farthest-first -> fill from end
	}
	return out // ascending by distance
}

// selectNeighbors keeps the m closest candidates. This is the simple heuristic;
// the fancier "keep diverse neighbors" variant is a later optimization.
func (g *Graph) selectNeighbors(cands []candidate, m int) []int {
	if len(cands) > m {
		cands = cands[:m]
	}
	out := make([]int, len(cands))
	for i, c := range cands {
		out[i] = c.idx
	}
	return out
}

// pruneConnections trims a node's neighbor list on layer lc back down to the
// layer's max, keeping the closest ones.
func (g *Graph) pruneConnections(idx, lc int) {
	max := g.maxConn(lc)
	nbrs := g.neighborsAt(idx, lc)
	if len(nbrs) <= max {
		return
	}
	self := g.nodes[idx].vector
	sort.Slice(nbrs, func(i, j int) bool {
		return g.dist(g.nodes[nbrs[i]].vector, self) < g.dist(g.nodes[nbrs[j]].vector, self)
	})
	g.nodes[idx].neighbors[lc] = nbrs[:max]
}

// connect adds `to` to `from`'s neighbor list on layer lc (no dedup in v1).
func (g *Graph) connect(from, to, lc int) {
	n := g.nodes[from]
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
