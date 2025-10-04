package index

import (
	"math"
)

// GetDistanceFunc returns the appropriate SIMD-optimized distance function for the given metric
func GetDistanceFunc(metric DistanceMetric) DistanceFunc {
	switch metric {
	case Cosine:
		return CosineDistanceSIMD
	case Euclidean:
		return EuclideanDistanceSIMD
	case Manhattan:
		return ManhattanDistanceSIMD
	case DotProduct:
		return DotProductDistanceSIMD
	default:
		return CosineDistanceSIMD
	}
}

// SIMD-optimized distance function wrappers that match DistanceFunc signature

// CosineDistanceSIMD wraps the SIMD cosine distance function
func CosineDistanceSIMD(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return 0, ErrEmptyVector
	}
	return CosineSIMD(a, b), nil
}

// EuclideanDistanceSIMD wraps the SIMD Euclidean distance function
func EuclideanDistanceSIMD(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return 0, ErrEmptyVector
	}
	return EuclideanSIMD(a, b), nil
}

// ManhattanDistanceSIMD wraps the SIMD Manhattan distance function
func ManhattanDistanceSIMD(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return 0, ErrEmptyVector
	}
	return ManhattanSIMD(a, b), nil
}

// DotProductDistanceSIMD wraps the SIMD dot product distance function
func DotProductDistanceSIMD(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return 0, ErrEmptyVector
	}
	return DotProductSIMD(a, b), nil
}

// CosineDistance calculates the cosine distance between two vectors
// Returns 1 - cosine_similarity, where 0 means identical vectors
func CosineDistance(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return 0, ErrEmptyVector
	}

	var dotProduct, normA, normB float64

	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	// Handle zero vectors
	if normA == 0 && normB == 0 {
		return 0, nil // Both vectors are zero, they are identical
	}
	if normA == 0 || normB == 0 {
		return 1, nil // One vector is zero, maximum distance
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	cosineSimilarity := dotProduct / (normA * normB)

	// Clamp to [-1, 1] to handle floating point errors
	if cosineSimilarity > 1 {
		cosineSimilarity = 1
	} else if cosineSimilarity < -1 {
		cosineSimilarity = -1
	}

	return float32(1 - cosineSimilarity), nil
}

// EuclideanDistance calculates the Euclidean (L2) distance between two vectors
func EuclideanDistance(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return 0, ErrEmptyVector
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}

	return float32(math.Sqrt(sum)), nil
}

// EuclideanDistanceSquared calculates the squared Euclidean distance (faster, no sqrt)
func EuclideanDistanceSquared(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return 0, ErrEmptyVector
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}

	return float32(sum), nil
}

// ManhattanDistance calculates the Manhattan (L1) distance between two vectors
func ManhattanDistance(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return 0, ErrEmptyVector
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		sum += math.Abs(float64(a[i] - b[i]))
	}

	return float32(sum), nil
}

// DotProductDistance calculates the negative dot product (for similarity)
// Returns -dot_product, where smaller values mean more similar vectors
func DotProductDistance(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimensionMismatch
	}
	if len(a) == 0 {
		return 0, ErrEmptyVector
	}

	var dotProduct float64
	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
	}

	return float32(-dotProduct), nil
}

// Normalize normalizes a vector to unit length (in-place)
func Normalize(vector []float32) error {
	if len(vector) == 0 {
		return ErrEmptyVector
	}

	var norm float64
	for _, v := range vector {
		norm += float64(v) * float64(v)
	}

	if norm == 0 {
		return nil // Zero vector remains zero
	}

	norm = math.Sqrt(norm)
	invNorm := 1.0 / norm

	for i := range vector {
		vector[i] = float32(float64(vector[i]) * invNorm)
	}

	return nil
}

// VectorNorm calculates the L2 norm of a vector
func VectorNorm(vector []float32) float32 {
	var sum float64
	for _, v := range vector {
		sum += float64(v) * float64(v)
	}
	return float32(math.Sqrt(sum))
}

// VectorMagnitude calculates the magnitude (L2 norm) of a vector
func VectorMagnitude(vector []float32) float32 {
	return VectorNorm(vector)
}

// VectorAdd adds two vectors element-wise
func VectorAdd(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, ErrDimensionMismatch
	}

	result := make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		result[i] = a[i] + b[i]
	}

	return result, nil
}

// VectorSubtract subtracts vector b from vector a element-wise
func VectorSubtract(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, ErrDimensionMismatch
	}

	result := make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		result[i] = a[i] - b[i]
	}

	return result, nil
}

// VectorScale multiplies a vector by a scalar
func VectorScale(vector []float32, scalar float32) []float32 {
	result := make([]float32, len(vector))
	for i, v := range vector {
		result[i] = v * scalar
	}
	return result
}

// VectorDotProduct calculates the dot product of two vectors
func VectorDotProduct(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimensionMismatch
	}

	var product float64
	for i := 0; i < len(a); i++ {
		product += float64(a[i]) * float64(b[i])
	}

	return float32(product), nil
}
