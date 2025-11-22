package index

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
	"time"
)

func TestDotProductAVX2(t *testing.T) {
	cases := []struct {
		name string
		dim  int
	}{
		{"Small", 8},
		{"Medium", 128},
		{"Large", 1024},
		{"Odd", 123},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := randomVector(tc.dim)
			b := randomVector(tc.dim)

			expected := dotProductGo(a, b)
			got := DotProductAVX2(a, b)

			if math.Abs(float64(expected-got)) > 1e-3 {
				t.Errorf("dim %d: expected %f, got %f", tc.dim, expected, got)
			}
		})
	}
}

func TestEuclideanAVX2(t *testing.T) {
	cases := []struct {
		name string
		dim  int
	}{
		{"Small", 8},
		{"Medium", 128},
		{"Large", 1024},
		{"Odd", 123},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := randomVector(tc.dim)
			b := randomVector(tc.dim)

			expected := euclideanGo(a, b)
			got := EuclideanAVX2(a, b)

			if math.Abs(float64(expected-got)) > 1e-3 {
				t.Errorf("dim %d: expected %f, got %f", tc.dim, expected, got)
			}
		})
	}
}

func BenchmarkDotProduct(b *testing.B) {
	dims := []int{128, 768, 1536}
	for _, dim := range dims {
		a := randomVector(dim)
		bVec := randomVector(dim)

		b.Run("AVX2_"+strconv.Itoa(dim), func(b *testing.B) {
			b.SetBytes(int64(dim * 4))
			for i := 0; i < b.N; i++ {
				DotProductAVX2(a, bVec)
			}
		})

		b.Run("Go_"+strconv.Itoa(dim), func(b *testing.B) {
			b.SetBytes(int64(dim * 4))
			for i := 0; i < b.N; i++ {
				dotProductGo(a, bVec)
			}
		})
	}
}

func BenchmarkEuclidean(b *testing.B) {
	dims := []int{128, 768, 1536}
	for _, dim := range dims {
		a := randomVector(dim)
		bVec := randomVector(dim)

		b.Run("AVX2_"+strconv.Itoa(dim), func(b *testing.B) {
			b.SetBytes(int64(dim * 4))
			for i := 0; i < b.N; i++ {
				EuclideanAVX2(a, bVec)
			}
		})

		b.Run("Go_"+strconv.Itoa(dim), func(b *testing.B) {
			b.SetBytes(int64(dim * 4))
			for i := 0; i < b.N; i++ {
				euclideanGo(a, bVec)
			}
		})
	}
}

func randomVector(dim int) []float32 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	v := make([]float32, dim)
	for i := 0; i < dim; i++ {
		v[i] = r.Float32()
	}
	return v
}

func dotProductGo(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func euclideanGo(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}
