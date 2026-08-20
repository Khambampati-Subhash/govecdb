package hnsw

import (
	"errors"
	"math"
	"math/rand"
	"sync"
)

// Graph is an HNSW index. It is safe for concurrent use: any number of Search
// calls run in parallel, and Insert excludes them. Per-search scratch lives in a
// pooled searchState (see state.go) rather than on the graph, which is what lets
// readers share it.
//
// The lock is coarse — one RWMutex over the whole graph — because HNSW inserts
// mutate neighbor lists several hops away from the new node, so there is no
// small region to lock instead. Read-heavy workloads (the expected shape) scale;
// concurrent *writers* serialize, and finer-grained writes are a later step.
type Graph struct {
	// mu guards every mutable field below it. cfg, dist, the M/ml/alpha knobs
	// and normalized are written once in New and only read afterwards.
	mu sync.RWMutex

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

	// numDeleted counts tombstoned slots. Kept as a running total rather than
	// recomputed, because it is what a compaction policy polls.
	numDeleted int

	// pool hands out per-traversal scratch so the hot path allocates nothing.
	// It is internally synchronized, so it sits outside mu's coverage.
	pool sync.Pool
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
	g := &Graph{
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
	}
	g.pool.New = func() any { return new(searchState) }
	return g, nil
}

// Len reports how many vectors Search can return — live vectors only.
// Tombstoned slots still occupy memory and are still traversed, but they are
// not "in the graph" from a caller's point of view. Use Stats to see them.
func (g *Graph) Len() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.nodes) - g.numDeleted
}

// Stats describes how much of the graph is still worth carrying. It exists to
// answer one question — "is it time to compact?" — without exposing the graph's
// internals to whatever ends up deciding that.
type Stats struct {
	// Live is the number of vectors Search can return; the same as Len.
	Live int
	// Deleted is the number of tombstoned slots: memory held, and traversal
	// cost paid, for vectors nobody can retrieve.
	Deleted int
	// Slots is the total allocated slots, Live + Deleted.
	Slots int
}

// Stats reports the graph's live/tombstoned occupancy.
func (g *Graph) Stats() Stats {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return Stats{
		Live:    len(g.nodes) - g.numDeleted,
		Deleted: g.numDeleted,
		Slots:   len(g.nodes),
	}
}

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
