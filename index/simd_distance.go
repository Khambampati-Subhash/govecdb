package index

import (
	"math"
	"unsafe"
)

// SIMD-optimized distance functions for better performance
// These implementations use manual vectorization techniques and cache-friendly patterns

// SIMDDistanceFunc represents a SIMD-optimized distance function
type SIMDDistanceFunc func(a, b []float32) float32

// EuclideanSIMD computes Euclidean distance with SIMD-like optimizations
func EuclideanSIMD(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}

	var sum1, sum2, sum3, sum4 float32
	i := 0

	// Process 4 elements at a time for better pipeline utilization
	for i < len(a)-3 {
		diff1 := a[i] - b[i]
		diff2 := a[i+1] - b[i+1]
		diff3 := a[i+2] - b[i+2]
		diff4 := a[i+3] - b[i+3]

		sum1 += diff1 * diff1
		sum2 += diff2 * diff2
		sum3 += diff3 * diff3
		sum4 += diff4 * diff4

		i += 4
	}

	// Handle remaining elements
	sum := sum1 + sum2 + sum3 + sum4
	for i < len(a) {
		diff := a[i] - b[i]
		sum += diff * diff
		i++
	}

	return float32(math.Sqrt(float64(sum)))
}

// CosineSIMD computes cosine distance with SIMD-like optimizations
func CosineSIMD(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}

	var dot1, dot2, dot3, dot4 float32
	var normA1, normA2, normA3, normA4 float32
	var normB1, normB2, normB3, normB4 float32
	i := 0

	// Process 4 elements at a time
	for i < len(a)-3 {
		a1, a2, a3, a4 := a[i], a[i+1], a[i+2], a[i+3]
		b1, b2, b3, b4 := b[i], b[i+1], b[i+2], b[i+3]

		dot1 += a1 * b1
		dot2 += a2 * b2
		dot3 += a3 * b3
		dot4 += a4 * b4

		normA1 += a1 * a1
		normA2 += a2 * a2
		normA3 += a3 * a3
		normA4 += a4 * a4

		normB1 += b1 * b1
		normB2 += b2 * b2
		normB3 += b3 * b3
		normB4 += b4 * b4

		i += 4
	}

	// Combine partial sums
	dot := dot1 + dot2 + dot3 + dot4
	normA := normA1 + normA2 + normA3 + normA4
	normB := normB1 + normB2 + normB3 + normB4

	// Handle remaining elements
	for i < len(a) {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
		i++
	}

	norm := float32(math.Sqrt(float64(normA * normB)))
	if norm == 0 {
		return 1.0
	}

	similarity := dot / norm
	// Clamp to valid range
	if similarity > 1.0 {
		similarity = 1.0
	}
	if similarity < -1.0 {
		similarity = -1.0
	}

	return 1.0 - similarity
}

// DotProductSIMD computes dot product with SIMD-like optimizations
func DotProductSIMD(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(-1))
	}

	var sum1, sum2, sum3, sum4 float32
	i := 0

	// Process 4 elements at a time
	for i < len(a)-3 {
		sum1 += a[i] * b[i]
		sum2 += a[i+1] * b[i+1]
		sum3 += a[i+2] * b[i+2]
		sum4 += a[i+3] * b[i+3]
		i += 4
	}

	// Handle remaining elements
	sum := sum1 + sum2 + sum3 + sum4
	for i < len(a) {
		sum += a[i] * b[i]
		i++
	}

	return -sum // Negative for similarity ranking
}

// ManhattanSIMD computes Manhattan distance with SIMD-like optimizations
func ManhattanSIMD(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}

	var sum1, sum2, sum3, sum4 float32
	i := 0

	// Process 4 elements at a time
	for i < len(a)-3 {
		sum1 += simdAbs32(a[i] - b[i])
		sum2 += simdAbs32(a[i+1] - b[i+1])
		sum3 += simdAbs32(a[i+2] - b[i+2])
		sum4 += simdAbs32(a[i+3] - b[i+3])
		i += 4
	}

	// Handle remaining elements
	sum := sum1 + sum2 + sum3 + sum4
	for i < len(a) {
		sum += simdAbs32(a[i] - b[i])
		i++
	}

	return sum
}

// simdAbs32 computes absolute value for float32 using bit manipulation
func simdAbs32(x float32) float32 {
	// Use bit manipulation to clear sign bit
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
}

// GetSIMDDistanceFunc returns the SIMD-optimized distance function for a metric
func GetSIMDDistanceFunc(metric string) SIMDDistanceFunc {
	switch metric {
	case "euclidean":
		return EuclideanSIMD
	case "cosine":
		return CosineSIMD
	case "dotproduct":
		return DotProductSIMD
	case "manhattan":
		return ManhattanSIMD
	default:
		return EuclideanSIMD
	}
}

// Enhanced batch distance functions for better throughput
type BatchDistanceCalculator struct {
	metric     DistanceMetric
	distFunc   OptimizedDistanceFunc
	vectorPool *VectorPool
}

// NewBatchDistanceCalculator creates a new batch distance calculator
func NewBatchDistanceCalculator(metric DistanceMetric) *BatchDistanceCalculator {
	return &BatchDistanceCalculator{
		metric:     metric,
		distFunc:   getOptimizedDistanceFunc(metric),
		vectorPool: NewVectorPool(),
	}
}

// CalculateBatchDistances computes distances between a query and multiple vectors efficiently
func (bdc *BatchDistanceCalculator) CalculateBatchDistances(query []float32, vectors [][]float32, results []float32) {
	if len(results) != len(vectors) {
		panic("results slice must have same length as vectors slice")
	}

	// Process in batches for better cache locality
	const batchSize = 8

	for i := 0; i < len(vectors); i += batchSize {
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}

		// Process batch
		for j := i; j < end; j++ {
			results[j] = bdc.distFunc(query, vectors[j])
		}
	}
}

// BatchEuclideanSIMD computes multiple Euclidean distances simultaneously
func BatchEuclideanSIMD(query []float32, vectors [][]float32, results []float32) {
	if len(results) != len(vectors) {
		return
	}

	for i, vec := range vectors {
		results[i] = EuclideanSIMD(query, vec)
	}
}

// Enhanced SIMD functions with 8-element vectorization
func EuclideanSIMD8(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}

	var sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8 float32
	i := 0

	// Process 8 elements at a time for better SIMD utilization
	for i < len(a)-7 {
		diff1 := a[i] - b[i]
		diff2 := a[i+1] - b[i+1]
		diff3 := a[i+2] - b[i+2]
		diff4 := a[i+3] - b[i+3]
		diff5 := a[i+4] - b[i+4]
		diff6 := a[i+5] - b[i+5]
		diff7 := a[i+6] - b[i+6]
		diff8 := a[i+7] - b[i+7]

		sum1 += diff1 * diff1
		sum2 += diff2 * diff2
		sum3 += diff3 * diff3
		sum4 += diff4 * diff4
		sum5 += diff5 * diff5
		sum6 += diff6 * diff6
		sum7 += diff7 * diff7
		sum8 += diff8 * diff8

		i += 8
	}

	// Handle remaining elements with 4-element vectorization
	var sum float32 = sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8

	for i < len(a)-3 {
		diff1 := a[i] - b[i]
		diff2 := a[i+1] - b[i+1]
		diff3 := a[i+2] - b[i+2]
		diff4 := a[i+3] - b[i+3]

		sum += diff1*diff1 + diff2*diff2 + diff3*diff3 + diff4*diff4
		i += 4
	}

	// Handle final elements
	for i < len(a) {
		diff := a[i] - b[i]
		sum += diff * diff
		i++
	}

	return float32(math.Sqrt(float64(sum)))
}

// VectorNormalization provides optimized vector normalization
func NormalizeVector(v []float32) {
	var sum1, sum2, sum3, sum4 float32
	i := 0

	// Calculate squared magnitude with loop unrolling
	for i < len(v)-3 {
		sum1 += v[i] * v[i]
		sum2 += v[i+1] * v[i+1]
		sum3 += v[i+2] * v[i+2]
		sum4 += v[i+3] * v[i+3]
		i += 4
	}

	magnitude := sum1 + sum2 + sum3 + sum4
	for i < len(v) {
		magnitude += v[i] * v[i]
		i++
	}

	magnitude = float32(math.Sqrt(float64(magnitude)))
	if magnitude == 0 {
		return
	}

	invMagnitude := 1.0 / magnitude

	// Normalize with loop unrolling
	i = 0
	for i < len(v)-3 {
		v[i] *= invMagnitude
		v[i+1] *= invMagnitude
		v[i+2] *= invMagnitude
		v[i+3] *= invMagnitude
		i += 4
	}

	for i < len(v) {
		v[i] *= invMagnitude
		i++
	}
}

// CacheAwareVectorDistance uses cache-friendly access patterns
func CacheAwareVectorDistance(a, b []float32, metric DistanceMetric) float32 {
	// Pre-check for cache-friendly sizes
	size := len(a)
	if size != len(b) {
		return float32(math.Inf(1))
	}

	// For small vectors, use simple computation
	if size <= 64 {
		return GetSIMDDistanceFunc(metric.String())(a, b)
	}

	// For larger vectors, use block-based computation to improve cache locality
	const blockSize = 64
	var result float32

	switch metric {
	case Euclidean:
		var sum float32
		for i := 0; i < size; i += blockSize {
			end := i + blockSize
			if end > size {
				end = size
			}
			for j := i; j < end; j++ {
				diff := a[j] - b[j]
				sum += diff * diff
			}
		}
		result = float32(math.Sqrt(float64(sum)))

	case Cosine:
		var dot, normA, normB float32
		for i := 0; i < size; i += blockSize {
			end := i + blockSize
			if end > size {
				end = size
			}
			for j := i; j < end; j++ {
				dot += a[j] * b[j]
				normA += a[j] * a[j]
				normB += b[j] * b[j]
			}
		}
		norm := float32(math.Sqrt(float64(normA * normB)))
		if norm == 0 {
			result = 1.0
		} else {
			similarity := dot / norm
			if similarity > 1.0 {
				similarity = 1.0
			}
			if similarity < -1.0 {
				similarity = -1.0
			}
			result = 1.0 - similarity
		}

	default:
		result = GetSIMDDistanceFunc(metric.String())(a, b)
	}

	return result
}

// PrefetchVector hints the CPU to prefetch vector data into cache
func PrefetchVector(v []float32) {
	if len(v) == 0 {
		return
	}
	// Use unsafe pointer to hint prefetching
	ptr := unsafe.Pointer(&v[0])
	_ = ptr // Prevent unused variable warning

	// In a real implementation, you would use compiler intrinsics
	// or assembly code to emit prefetch instructions
	// For now, this serves as a placeholder for the optimization
}
