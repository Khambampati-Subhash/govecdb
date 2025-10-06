package utils

import (
	"math"
)

// NormalizeVectorInPlace normalizes the given float32 slice to unit length.
// It returns true when normalization was applied (norm > 0). Zero vectors are left unchanged.
func NormalizeVectorInPlace(data []float32) bool {
	if len(data) == 0 {
		return false
	}

	var sum float64
	for _, v := range data {
		sum += float64(v) * float64(v)
	}

	if sum == 0 {
		return false
	}

	norm := math.Sqrt(sum)

	// Skip work if vector is already normalized within a tight tolerance.
	if math.Abs(norm-1.0) < 1e-6 {
		return true
	}

	scale := float32(1.0 / norm)
	for i := range data {
		data[i] *= scale
	}

	return true
}

// CloneAndNormalizeVector returns a normalized copy of the provided slice.
// Zero vectors are copied but left unchanged.
func CloneAndNormalizeVector(data []float32) []float32 {
	cloned := make([]float32, len(data))
	copy(cloned, data)
	NormalizeVectorInPlace(cloned)
	return cloned
}
