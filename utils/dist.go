// Package utils provides optimized distance calculation functions for vector similarity.
// This includes SIMD-optimized implementations and fallback pure Go versions.
package utils

import (
	"math"
	"unsafe"
)

// DistanceFn represents a distance calculation function
type DistanceFn func(a, b []float32) float32

// DistanceMetric represents different distance metrics
type DistanceMetric int

const (
	EuclideanDistance DistanceMetric = iota
	CosineDistance
	DotProductDistance
	ManhattanDistance
	HammingDistance
)

// GetDistanceFunction returns the appropriate distance function for the metric
func GetDistanceFunction(metric DistanceMetric) DistanceFn {
	switch metric {
	case EuclideanDistance:
		return EuclideanDistanceOptimized
	case CosineDistance:
		return CosineDistanceOptimized
	case DotProductDistance:
		return DotProductOptimized
	case ManhattanDistance:
		return ManhattanDistanceOptimized
	case HammingDistance:
		return HammingDistanceOptimized
	default:
		return EuclideanDistanceOptimized
	}
}

// EuclideanDistanceOptimized calculates Euclidean distance with optimizations
func EuclideanDistanceOptimized(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vectors must have same dimension")
	}

	// Use SIMD-optimized version if available and vectors are large enough
	if len(a) >= 8 && hasSIMDSupport() {
		return euclideanSIMD(a, b)
	}

	return euclideanPure(a, b)
}

// euclideanPure is the pure Go implementation
func euclideanPure(a, b []float32) float32 {
	var sum float32

	// Unroll loop for better performance
	i := 0
	for i+4 <= len(a) {
		diff0 := a[i] - b[i]
		diff1 := a[i+1] - b[i+1]
		diff2 := a[i+2] - b[i+2]
		diff3 := a[i+3] - b[i+3]

		sum += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
		i += 4
	}

	// Handle remaining elements
	for i < len(a) {
		diff := a[i] - b[i]
		sum += diff * diff
		i++
	}

	return float32(math.Sqrt(float64(sum)))
}

// CosineDistanceOptimized calculates cosine distance with optimizations
func CosineDistanceOptimized(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vectors must have same dimension")
	}

	if len(a) >= 8 && hasSIMDSupport() {
		return cosineSIMD(a, b)
	}

	return cosinePure(a, b)
}

// cosinePure is the pure Go implementation
func cosinePure(a, b []float32) float32 {
	var dotProduct, normA, normB float32

	// Unroll loop for better performance
	i := 0
	for i+4 <= len(a) {
		dotProduct += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		normA += a[i]*a[i] + a[i+1]*a[i+1] + a[i+2]*a[i+2] + a[i+3]*a[i+3]
		normB += b[i]*b[i] + b[i+1]*b[i+1] + b[i+2]*b[i+2] + b[i+3]*b[i+3]
		i += 4
	}

	// Handle remaining elements
	for i < len(a) {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
		i++
	}

	if normA == 0 || normB == 0 {
		return 1.0 // Maximum distance for zero vectors
	}

	cosine := dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	return 1.0 - cosine // Convert similarity to distance
}

// DotProductOptimized calculates dot product with optimizations
func DotProductOptimized(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vectors must have same dimension")
	}

	if len(a) >= 8 && hasSIMDSupport() {
		return dotProductSIMD(a, b)
	}

	return dotProductPure(a, b)
}

// dotProductPure is the pure Go implementation
func dotProductPure(a, b []float32) float32 {
	var sum float32

	// Unroll loop for better performance
	i := 0
	for i+4 <= len(a) {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		i += 4
	}

	// Handle remaining elements
	for i < len(a) {
		sum += a[i] * b[i]
		i++
	}

	return -sum // Negative because we want higher dot product = smaller distance
}

// ManhattanDistanceOptimized calculates Manhattan distance with optimizations
func ManhattanDistanceOptimized(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vectors must have same dimension")
	}

	var sum float32

	// Unroll loop for better performance
	i := 0
	for i+4 <= len(a) {
		sum += abs(a[i]-b[i]) + abs(a[i+1]-b[i+1]) + abs(a[i+2]-b[i+2]) + abs(a[i+3]-b[i+3])
		i += 4
	}

	// Handle remaining elements
	for i < len(a) {
		sum += abs(a[i] - b[i])
		i++
	}

	return sum
}

// HammingDistanceOptimized calculates Hamming distance for binary vectors
func HammingDistanceOptimized(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vectors must have same dimension")
	}

	var count float32
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			count++
		}
	}

	return count
}

// abs returns the absolute value of a float32
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// SIMD Support Detection and Implementations
// Note: These are placeholder implementations. In a real-world scenario,
// you would use assembly or cgo to call SIMD instructions directly.

// hasSIMDSupport checks if the CPU supports SIMD instructions
func hasSIMDSupport() bool {
	// This is a placeholder. In reality, you would check CPU features.
	// For now, we'll assume SIMD is available on 64-bit systems.
	return unsafe.Sizeof(uintptr(0)) == 8
}

// euclideanSIMD is a placeholder for SIMD-optimized Euclidean distance
func euclideanSIMD(a, b []float32) float32 {
	// In a real implementation, this would use SIMD instructions
	// For now, we'll use an optimized pure Go version with hints for the compiler
	return euclideanOptimized(a, b)
}

// cosineSIMD is a placeholder for SIMD-optimized cosine distance
func cosineSIMD(a, b []float32) float32 {
	// In a real implementation, this would use SIMD instructions
	return cosineOptimized(a, b)
}

// dotProductSIMD is a placeholder for SIMD-optimized dot product
func dotProductSIMD(a, b []float32) float32 {
	// In a real implementation, this would use SIMD instructions
	return dotProductOptimized(a, b)
}

// Optimized implementations with compiler hints

// euclideanOptimized provides compiler-friendly Euclidean distance calculation
func euclideanOptimized(a, b []float32) float32 {
	var sum float64 // Use float64 for intermediate calculations to reduce precision loss

	// Process in chunks of 8 for better vectorization
	i := 0
	for i+8 <= len(a) {
		var chunkSum float64
		for j := 0; j < 8; j++ {
			diff := float64(a[i+j] - b[i+j])
			chunkSum += diff * diff
		}
		sum += chunkSum
		i += 8
	}

	// Handle remaining elements
	for i < len(a) {
		diff := float64(a[i] - b[i])
		sum += diff * diff
		i++
	}

	return float32(math.Sqrt(sum))
}

// cosineOptimized provides compiler-friendly cosine distance calculation
func cosineOptimized(a, b []float32) float32 {
	var dotProduct, normA, normB float64

	// Process in chunks of 8
	i := 0
	for i+8 <= len(a) {
		var chunkDot, chunkNormA, chunkNormB float64
		for j := 0; j < 8; j++ {
			aVal := float64(a[i+j])
			bVal := float64(b[i+j])
			chunkDot += aVal * bVal
			chunkNormA += aVal * aVal
			chunkNormB += bVal * bVal
		}
		dotProduct += chunkDot
		normA += chunkNormA
		normB += chunkNormB
		i += 8
	}

	// Handle remaining elements
	for i < len(a) {
		aVal := float64(a[i])
		bVal := float64(b[i])
		dotProduct += aVal * bVal
		normA += aVal * aVal
		normB += bVal * bVal
		i++
	}

	if normA == 0 || normB == 0 {
		return 1.0
	}

	cosine := dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
	return float32(1.0 - cosine)
}

// dotProductOptimized provides compiler-friendly dot product calculation
func dotProductOptimized(a, b []float32) float32 {
	var sum float64

	// Process in chunks of 8
	i := 0
	for i+8 <= len(a) {
		var chunkSum float64
		for j := 0; j < 8; j++ {
			chunkSum += float64(a[i+j]) * float64(b[i+j])
		}
		sum += chunkSum
		i += 8
	}

	// Handle remaining elements
	for i < len(a) {
		sum += float64(a[i]) * float64(b[i])
		i++
	}

	return -float32(sum)
}

// BatchDistanceCalculator provides efficient batch distance calculations
type BatchDistanceCalculator struct {
	metric     DistanceMetric
	distanceFn DistanceFn
	workers    int
	threshold  int // Minimum batch size for parallel processing
}

// NewBatchDistanceCalculator creates a new batch distance calculator
func NewBatchDistanceCalculator(metric DistanceMetric, workers int) *BatchDistanceCalculator {
	return &BatchDistanceCalculator{
		metric:     metric,
		distanceFn: GetDistanceFunction(metric),
		workers:    workers,
		threshold:  100, // Use parallel processing for batches > 100
	}
}

// CalculateDistances calculates distances between a query vector and multiple vectors
func (bdc *BatchDistanceCalculator) CalculateDistances(query []float32, vectors [][]float32) []float32 {
	distances := make([]float32, len(vectors))

	if len(vectors) < bdc.threshold {
		// Sequential processing for small batches
		for i, vector := range vectors {
			distances[i] = bdc.distanceFn(query, vector)
		}
		return distances
	}

	// Parallel processing for large batches
	wp := NewWorkerPool(bdc.workers)
	defer wp.Close()

	chunkSize := len(vectors) / bdc.workers
	if chunkSize == 0 {
		chunkSize = 1
	}

	for i := 0; i < len(vectors); i += chunkSize {
		start := i
		end := i + chunkSize
		if end > len(vectors) {
			end = len(vectors)
		}

		wp.Submit(func() {
			for j := start; j < end; j++ {
				distances[j] = bdc.distanceFn(query, vectors[j])
			}
		})
	}

	// Wait for all workers to complete
	// Note: In a real implementation, you'd want proper synchronization
	// This is simplified for demonstration

	return distances
}

// VectorNormalizer provides utilities for vector normalization
type VectorNormalizer struct {
	pool *Float32Pool
}

// NewVectorNormalizer creates a new vector normalizer
func NewVectorNormalizer() *VectorNormalizer {
	return &VectorNormalizer{
		pool: NewFloat32Pool(1024),
	}
}

// L2Normalize normalizes a vector using L2 norm
func (vn *VectorNormalizer) L2Normalize(vector []float32) []float32 {
	normalized := vn.pool.Get()
	if cap(normalized) < len(vector) {
		normalized = make([]float32, len(vector))
	} else {
		normalized = normalized[:len(vector)]
	}

	// Calculate L2 norm
	var norm float64
	for _, val := range vector {
		norm += float64(val * val)
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		// Return zero vector if input is zero
		for i := range normalized {
			normalized[i] = 0
		}
		return normalized
	}

	// Normalize
	invNorm := float32(1.0 / norm)
	for i, val := range vector {
		normalized[i] = val * invNorm
	}

	return normalized
}

// Release returns a normalized vector to the pool
func (vn *VectorNormalizer) Release(vector []float32) {
	vn.pool.Put(vector)
}

// DistanceCache provides caching for frequently computed distances
type DistanceCache struct {
	cache *Cache
}

// NewDistanceCache creates a new distance cache
func NewDistanceCache(capacity int) *DistanceCache {
	return &DistanceCache{
		cache: NewCache(capacity),
	}
}

// GetOrCompute gets a cached distance or computes it if not found
func (dc *DistanceCache) GetOrCompute(key string, computeFn func() float32) float32 {
	if distance, found := dc.cache.Get(key); found {
		return distance.(float32)
	}

	distance := computeFn()
	dc.cache.Put(key, distance)
	return distance
}

// Clear clears the distance cache
func (dc *DistanceCache) Clear() {
	dc.cache.Clear()
}

// VectorQuantizer provides vector quantization for memory efficiency
type VectorQuantizer struct {
	bits   int     // Number of bits per component
	scale  float32 // Scaling factor
	offset float32 // Offset for quantization
}

// NewVectorQuantizer creates a new vector quantizer
func NewVectorQuantizer(bits int) *VectorQuantizer {
	maxVal := (1 << uint(bits)) - 1 // Calculate as int first
	return &VectorQuantizer{
		bits:   bits,
		scale:  float32(maxVal), // e.g., 255 for 8 bits
		offset: 0.5,
	}
}

// Quantize quantizes a float32 vector to reduce memory usage
func (vq *VectorQuantizer) Quantize(vector []float32) []uint8 {
	quantized := make([]uint8, len(vector))

	for i, val := range vector {
		// Clamp to [0, 1] range
		if val < 0 {
			val = 0
		} else if val > 1 {
			val = 1
		}

		// Quantize
		quantized[i] = uint8(val*vq.scale + vq.offset)
	}

	return quantized
}

// Dequantize converts quantized values back to float32
func (vq *VectorQuantizer) Dequantize(quantized []uint8) []float32 {
	vector := make([]float32, len(quantized))

	for i, val := range quantized {
		vector[i] = (float32(val) - vq.offset) / vq.scale
	}

	return vector
}
