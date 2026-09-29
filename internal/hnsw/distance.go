// Distance metrics and their kernels. Everything here returns SMALLER MEANS
// CLOSER, so the graph never has to branch on which metric is in use.
//
// Kernels are hand-unrolled with eight independent accumulators. This is not
// cosmetic: a single accumulator serialises the FP-add dependency chain, so the
// CPU stalls waiting on the previous add. Go emits scalar FMAs, whose ~4-cycle
// latency against several issue ports means four chains still left the kernel
// latency-bound; eight took Dot from 27.5 to 18.4 ns at dim 128 and from 239 to
// 116 ns at dim 768 (Apple M4). Sixteen is slower again — the accumulators
// spill out of registers. Each step re-slices a fixed 8-wide window with a
// capped capacity (x := a[:8:8]), which is what lets the compiler prove every
// index in range and drop the bounds checks inside the loop.

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
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	for len(a) >= 8 {
		x, y := a[:8:8], b[:8:8]
		s0 += x[0] * y[0]
		s1 += x[1] * y[1]
		s2 += x[2] * y[2]
		s3 += x[3] * y[3]
		s4 += x[4] * y[4]
		s5 += x[5] * y[5]
		s6 += x[6] * y[6]
		s7 += x[7] * y[7]
		a, b = a[8:], b[8:]
	}
	for i := range a {
		s0 += a[i] * b[i]
	}
	return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
}

// oneMinusDot is the cosine distance for vectors already unit length.
func oneMinusDot(a, b []float32) float32 { return 1 - Dot(a, b) }

// NegativeDot makes a larger dot product mean a smaller distance.
func NegativeDot(a, b []float32) float32 { return -Dot(a, b) }

// SquaredEuclidean returns |a-b|^2 (monotonic with the true distance).
func SquaredEuclidean(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	for len(a) >= 8 {
		x, y := a[:8:8], b[:8:8]
		d0, d1, d2, d3 := x[0]-y[0], x[1]-y[1], x[2]-y[2], x[3]-y[3]
		d4, d5, d6, d7 := x[4]-y[4], x[5]-y[5], x[6]-y[6], x[7]-y[7]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
		s4 += d4 * d4
		s5 += d5 * d5
		s6 += d6 * d6
		s7 += d7 * d7
		a, b = a[8:], b[8:]
	}
	for i := range a {
		d := a[i] - b[i]
		s0 += d * d
	}
	return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
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
