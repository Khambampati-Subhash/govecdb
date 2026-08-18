package hnsw

import (
	"errors"
	"math"
	"math/rand"
)

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
