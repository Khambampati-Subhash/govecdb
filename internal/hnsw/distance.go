// Package hnsw is a small, readable implementation of the Hierarchical
// Navigable Small World (HNSW) graph for approximate nearest-neighbor search.
//
// Every distance function returns a value where SMALLER MEANS CLOSER, so the
// graph can reason about "nearest" without ever branching on the metric.
//
// Kernels are hand-unrolled with four independent accumulators. This is not
// cosmetic: a single accumulator serialises the FP-add dependency chain, so the
// CPU stalls waiting on the previous add. Four chains let the out-of-order
// engine overlap them, and it also gives the Go compiler a shape it can
// auto-vectorise. Slices are re-sliced to a common length first so the bounds
// checks are hoisted out of the loop.
package hnsw

import "math"

// Metric selects how "distance" between two vectors is measured.
type Metric int

const (
	// Cosine distance: 1 - cosine similarity. Range [0, 2]; 0 == identical
	// direction. Vectors are unit-normalized on insert (see Graph.normalize),
	// which reduces this to 1 - dot(a,b) — no square roots in the hot path.
	Cosine Metric = iota
	// Euclidean distance. We return the SQUARED distance: it is monotonic with
	// the true distance, so ranking is identical, and we skip the sqrt.
	Euclidean
	// DotProduct: negative dot product, so a larger dot (more similar) yields a
	// smaller distance. Vectors are used as given, not normalized.
	DotProduct
)

// DistanceFunc computes distance between two equal-length vectors.
// Smaller result == closer. Callers guarantee len(a) == len(b).
type DistanceFunc func(a, b []float32) float32

// normalizes reports whether this metric wants unit-length vectors. Only Cosine
// does: normalizing would silently change what Euclidean and DotProduct mean.
func (m Metric) normalizes() bool { return m == Cosine }

// Func returns the distance function used when vectors are stored as-is.
func (m Metric) Func() DistanceFunc {
	switch m {
	case Euclidean:
		return SquaredEuclidean
	case DotProduct:
		return NegativeDot
	case Cosine:
		fallthrough
	default:
		return CosineDistance
	}
}

// fastFunc returns the kernel to use given whether stored vectors are already
// unit length. For Cosine that turns a 3-accumulator + 2-sqrt computation into
// a single dot product.
func (m Metric) fastFunc(normalized bool) DistanceFunc {
	if m == Cosine && normalized {
		return oneMinusDot
	}
	return m.Func()
}

// Normalize scales v to unit length in place. A zero vector has no direction,
// so it is left untouched.
func Normalize(v []float32) {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(float64(sum)))
	for i := range v {
		v[i] *= inv
	}
}

// Dot returns the dot product of a and b.
func Dot(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// oneMinusDot is the cosine distance for vectors already unit length.
func oneMinusDot(a, b []float32) float32 { return 1 - Dot(a, b) }

// NegativeDot makes a larger dot product mean a smaller distance.
func NegativeDot(a, b []float32) float32 { return -Dot(a, b) }

// SquaredEuclidean returns |a-b|^2 (monotonic with the true distance).
func SquaredEuclidean(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return (s0 + s1) + (s2 + s3)
}

// CosineDistance is the general form for vectors of arbitrary length. The graph
// avoids this on the hot path by normalizing on insert.
func CosineDistance(a, b []float32) float32 {
	b = b[:len(a)]
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 1 // undefined direction; treat as unrelated
	}
	return 1 - dot/float32(math.Sqrt(float64(na))*math.Sqrt(float64(nb)))
}
