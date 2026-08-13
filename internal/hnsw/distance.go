// Package hnsw is a small, readable implementation of the Hierarchical
// Navigable Small World (HNSW) graph for approximate nearest-neighbor search.
//
// This is the v1 "learn it by building it" core: single-threaded, no memory
// pools or SIMD, optimized for clarity over raw speed. Every distance function
// returns a value where SMALLER MEANS CLOSER, so the whole graph can reason
// about "nearest" without caring which metric is in use.
package hnsw

import "math"

// Metric selects how "distance" between two vectors is measured.
type Metric int

const (
	// Cosine distance: 1 - cosine similarity. Range [0, 2]; 0 == identical direction.
	Cosine Metric = iota
	// Euclidean distance. We return the SQUARED distance (monotonic with the
	// true distance) because ranking only needs the ordering, and skipping the
	// sqrt is cheaper.
	Euclidean
	// DotProduct: negative dot product, so a larger dot (more similar) yields a
	// smaller distance. Best used with normalized vectors.
	DotProduct
)

// DistanceFunc computes distance between two equal-length vectors.
// Smaller result == closer. Callers guarantee len(a) == len(b).
type DistanceFunc func(a, b []float32) float32

// Func returns the distance function for the metric.
func (m Metric) Func() DistanceFunc {
	switch m {
	case Euclidean:
		return squaredEuclidean
	case DotProduct:
		return negativeDot
	case Cosine:
		fallthrough
	default:
		return cosineDistance
	}
}

func squaredEuclidean(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func negativeDot(a, b []float32) float32 {
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return -dot
}

func cosineDistance(a, b []float32) float32 {
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 1 // undefined direction; treat as maximally dissimilar-ish
	}
	sim := dot / float32(math.Sqrt(float64(na))*math.Sqrt(float64(nb)))
	return 1 - sim
}
