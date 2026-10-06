package hnsw

import "errors"

var (
	// ErrDimensionMismatch is returned when a vector's length does not match
	// the dimension the graph was created with.
	ErrDimensionMismatch = errors.New("hnsw: vector dimension mismatch")
	// ErrEmptyVector is returned for a nil/zero-length vector.
	ErrEmptyVector = errors.New("hnsw: empty vector")
	// ErrInvalidConfig is returned by New for a Config it cannot build a graph
	// from. The wrapped message names the field and the value.
	ErrInvalidConfig = errors.New("hnsw: invalid config")
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
	//
	// Must be 0, meaning "use the default", or at least 2. M=1 is not a thin
	// graph, it is an undefined one: levels are drawn from -ln(u)/ln(M), and
	// ln(1) is zero, so every node would land on layer +Inf. New refuses it.
	M int
	// EfConstruction is how wide the search is during inserts. Higher = better
	// graph quality, slower inserts.
	EfConstruction int
	// Alpha is the pruning relaxation factor from the DiskANN/Vamana line of
	// work. When selecting neighbors we drop a candidate that sits closer to an
	// already-chosen neighbor than to the node itself — that edge is redundant,
	// you could reach it by hopping. Alpha scales that test, alpha·d(c,s) <=
	// d(c,base) rejecting c:
	//
	//	1.0  the classic HNSW heuristic, and the default
	//	>1.0 rejects *less*: the test gets harder to pass, so more of the
	//	     nearest candidates survive it
	//
	// That direction is the opposite of what the default of 1.2 was chosen
	// for. In DiskANN a larger alpha keeps longer edges because the prune runs
	// against a degree bound with no backfill; here the candidates arrive
	// nearest-first and the M slots fill in that order, so every extra
	// survivor is a near neighbor taking the slot a long-range edge would have
	// had. On clustered data that is what disconnects the clusters. Measured
	// at 62,500 vectors, dim 512, 1,024 tight clusters, M=16: recall@10 at
	// alpha 1.2 never passed 0.954, even at ef=8192; at 1.0 it was 0.95 by
	// ef=10 and 0.999 by ef=256. Alpha 1.4 topped out at 0.62. On uniform
	// data the two needed identical ef at every size and dimension measured
	// (128 to 1024, 20K and 62.5K), so 1.0 costs nothing there.
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
		Alpha:          1.0,
		Seed:           1,
	}
}
