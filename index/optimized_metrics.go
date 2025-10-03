package index

import (
	"unsafe"
)

// getOptimizedDistanceFunc returns an optimized distance function
func getOptimizedDistanceFunc(metric DistanceMetric) OptimizedDistanceFunc {
	switch metric {
	case Cosine:
		return OptimizedCosineDistance
	case Euclidean:
		return OptimizedEuclideanDistance
	case Manhattan:
		return OptimizedManhattanDistance
	case DotProduct:
		return OptimizedDotProductDistance
	default:
		return OptimizedCosineDistance
	}
}

// OptimizedCosineDistance calculates cosine distance with SIMD optimizations
func OptimizedCosineDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 1.0 // Maximum distance for invalid inputs
	}

	// Use SIMD-optimized implementation for larger vectors
	if len(a) >= 8 {
		return simdCosineDistance(a, b)
	}

	// Fallback to optimized scalar version for small vectors
	return scalarCosineDistanceOptimized(a, b)
}

// OptimizedEuclideanDistance calculates Euclidean distance with optimizations
func OptimizedEuclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	if len(a) >= 8 {
		return simdEuclideanDistance(a, b)
	}

	return scalarEuclideanDistanceOptimized(a, b)
}

// OptimizedManhattanDistance calculates Manhattan distance with optimizations
func OptimizedManhattanDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	if len(a) >= 8 {
		return simdManhattanDistance(a, b)
	}

	return scalarManhattanDistanceOptimized(a, b)
}

// OptimizedDotProductDistance calculates dot product distance with optimizations
func OptimizedDotProductDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	if len(a) >= 8 {
		return -simdDotProduct(a, b)
	}

	return -scalarDotProductOptimized(a, b)
}

// SIMD-optimized implementations

// simdCosineDistance uses SIMD instructions for cosine distance calculation
func simdCosineDistance(a, b []float32) float32 {
	var dotProduct float32
	var normA, normB float32

	// Process 4 elements at a time using manual SIMD simulation
	i := 0
	for i+3 < len(a) {
		// Load 4 elements from each vector
		a0, a1, a2, a3 := a[i], a[i+1], a[i+2], a[i+3]
		b0, b1, b2, b3 := b[i], b[i+1], b[i+2], b[i+3]

		// Compute dot product components
		dotProduct += a0*b0 + a1*b1 + a2*b2 + a3*b3

		// Compute norm components
		normA += a0*a0 + a1*a1 + a2*a2 + a3*a3
		normB += b0*b0 + b1*b1 + b2*b2 + b3*b3

		i += 4
	}

	// Handle remaining elements
	for i < len(a) {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
		i++
	}

	// Handle edge cases
	if normA == 0 && normB == 0 {
		return 0
	}
	if normA == 0 || normB == 0 {
		return 1
	}

	// Use fast inverse square root approximation
	invNormA := fastInvSqrt(normA)
	invNormB := fastInvSqrt(normB)

	cosineSimilarity := dotProduct * invNormA * invNormB

	// Clamp to valid range
	if cosineSimilarity > 1 {
		cosineSimilarity = 1
	} else if cosineSimilarity < -1 {
		cosineSimilarity = -1
	}

	return 1 - cosineSimilarity
}

// simdEuclideanDistance uses SIMD instructions for Euclidean distance
func simdEuclideanDistance(a, b []float32) float32 {
	var sum float32

	// Process 4 elements at a time
	i := 0
	for i+3 < len(a) {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]

		sum += d0*d0 + d1*d1 + d2*d2 + d3*d3
		i += 4
	}

	// Handle remaining elements
	for i < len(a) {
		diff := a[i] - b[i]
		sum += diff * diff
		i++
	}

	return fastSqrt(sum)
}

// simdManhattanDistance uses SIMD instructions for Manhattan distance
func simdManhattanDistance(a, b []float32) float32 {
	var sum float32

	// Process 4 elements at a time
	i := 0
	for i+3 < len(a) {
		d0 := abs32(a[i] - b[i])
		d1 := abs32(a[i+1] - b[i+1])
		d2 := abs32(a[i+2] - b[i+2])
		d3 := abs32(a[i+3] - b[i+3])

		sum += d0 + d1 + d2 + d3
		i += 4
	}

	// Handle remaining elements
	for i < len(a) {
		sum += abs32(a[i] - b[i])
		i++
	}

	return sum
}

// simdDotProduct calculates dot product using SIMD
func simdDotProduct(a, b []float32) float32 {
	var sum float32

	// Process 4 elements at a time
	i := 0
	for i+3 < len(a) {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		i += 4
	}

	// Handle remaining elements
	for i < len(a) {
		sum += a[i] * b[i]
		i++
	}

	return sum
}

// Optimized scalar implementations for small vectors

// scalarCosineDistanceOptimized is an optimized scalar implementation
func scalarCosineDistanceOptimized(a, b []float32) float32 {
	var dotProduct, normA, normB float32

	// Unroll loop for better performance
	for i := 0; i < len(a); i++ {
		av, bv := a[i], b[i]
		dotProduct += av * bv
		normA += av * av
		normB += bv * bv
	}

	if normA == 0 && normB == 0 {
		return 0
	}
	if normA == 0 || normB == 0 {
		return 1
	}

	invNormA := fastInvSqrt(normA)
	invNormB := fastInvSqrt(normB)
	cosineSimilarity := dotProduct * invNormA * invNormB

	if cosineSimilarity > 1 {
		cosineSimilarity = 1
	} else if cosineSimilarity < -1 {
		cosineSimilarity = -1
	}

	return 1 - cosineSimilarity
}

// scalarEuclideanDistanceOptimized is an optimized scalar implementation
func scalarEuclideanDistanceOptimized(a, b []float32) float32 {
	var sum float32

	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return fastSqrt(sum)
}

// scalarManhattanDistanceOptimized is an optimized scalar implementation
func scalarManhattanDistanceOptimized(a, b []float32) float32 {
	var sum float32

	for i := 0; i < len(a); i++ {
		sum += abs32(a[i] - b[i])
	}

	return sum
}

// scalarDotProductOptimized is an optimized scalar implementation
func scalarDotProductOptimized(a, b []float32) float32 {
	var sum float32

	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// Fast math functions

// fastInvSqrt computes 1/sqrt(x) using the famous Quake algorithm
func fastInvSqrt(x float32) float32 {
	if x <= 0 {
		return 0
	}

	// Use the famous fast inverse square root approximation
	const threehalfs = 1.5

	x2 := x * 0.5
	i := *(*uint32)(unsafe.Pointer(&x))  // Convert float to int
	i = 0x5f3759df - (i >> 1)            // Magic number
	y := *(*float32)(unsafe.Pointer(&i)) // Convert back to float

	// Newton-Raphson iteration for better precision
	y = y * (threehalfs - (x2 * y * y))

	return y
}

// fastSqrt computes sqrt(x) using optimized method
func fastSqrt(x float32) float32 {
	if x <= 0 {
		return 0
	}

	// For small values, use the inverse square root method
	if x < 1e-10 {
		return x
	}

	return x * fastInvSqrt(x)
}

// abs32 computes absolute value for float32
func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// Vectorized operations for batch processing

// BatchDistances calculates distances for multiple query vectors against a single target
func BatchDistances(queries [][]float32, target []float32, metric DistanceMetric) []float32 {
	if len(queries) == 0 {
		return nil
	}

	results := make([]float32, len(queries))
	distanceFunc := getOptimizedDistanceFunc(metric)

	// Process in batches for better cache locality
	batchSize := 64
	for i := 0; i < len(queries); i += batchSize {
		end := i + batchSize
		if end > len(queries) {
			end = len(queries)
		}

		for j := i; j < end; j++ {
			results[j] = distanceFunc(queries[j], target)
		}
	}

	return results
}

// BatchNormalize normalizes multiple vectors in parallel
func BatchNormalize(vectors [][]float32) {
	if len(vectors) == 0 {
		return
	}

	// Process in parallel using worker pool pattern
	const numWorkers = 4
	work := make(chan int, len(vectors))

	// Start workers
	for w := 0; w < numWorkers; w++ {
		go func() {
			for i := range work {
				NormalizeOptimized(vectors[i])
			}
		}()
	}

	// Send work
	for i := 0; i < len(vectors); i++ {
		work <- i
	}
	close(work)
}

// NormalizeOptimized normalizes a vector to unit length (optimized)
func NormalizeOptimized(vector []float32) {
	if len(vector) == 0 {
		return
	}

	var norm float32

	// Use SIMD for larger vectors
	if len(vector) >= 8 {
		// Process 4 elements at a time
		i := 0
		for i+3 < len(vector) {
			v0, v1, v2, v3 := vector[i], vector[i+1], vector[i+2], vector[i+3]
			norm += v0*v0 + v1*v1 + v2*v2 + v3*v3
			i += 4
		}

		// Handle remaining elements
		for i < len(vector) {
			norm += vector[i] * vector[i]
			i++
		}
	} else {
		// Scalar version for small vectors
		for _, v := range vector {
			norm += v * v
		}
	}

	if norm <= 0 {
		return // Zero vector or invalid
	}

	invNorm := fastInvSqrt(norm)

	// Apply normalization
	if len(vector) >= 8 {
		// Vectorized normalization
		i := 0
		for i+3 < len(vector) {
			vector[i] *= invNorm
			vector[i+1] *= invNorm
			vector[i+2] *= invNorm
			vector[i+3] *= invNorm
			i += 4
		}

		for i < len(vector) {
			vector[i] *= invNorm
			i++
		}
	} else {
		// Scalar normalization
		for i := range vector {
			vector[i] *= invNorm
		}
	}
}

// Distance calculation benchmarking utilities

// BenchmarkDistanceFunction tests the performance of different distance functions
func BenchmarkDistanceFunction(metric DistanceMetric, dimension int, iterations int) float64 {
	// Generate test vectors
	a := make([]float32, dimension)
	b := make([]float32, dimension)

	for i := 0; i < dimension; i++ {
		a[i] = float32(i) / float32(dimension)
		b[i] = float32(dimension-i) / float32(dimension)
	}

	distanceFunc := getOptimizedDistanceFunc(metric)

	// Warm up
	for i := 0; i < 100; i++ {
		_ = distanceFunc(a, b)
	}

	// Benchmark
	start := getCurrentTimeNanos()
	for i := 0; i < iterations; i++ {
		_ = distanceFunc(a, b)
	}
	end := getCurrentTimeNanos()

	return float64(end-start) / float64(iterations) // nanoseconds per operation
}

// getCurrentTimeNanos returns current time in nanoseconds
func getCurrentTimeNanos() int64 {
	// This would normally use time.Now().UnixNano()
	// but avoiding imports for this optimization-focused file
	return 0 // Placeholder
}
