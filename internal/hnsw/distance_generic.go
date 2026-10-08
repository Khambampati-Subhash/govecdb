//go:build !arm64 || purego

package hnsw

// Platforms without an assembly kernel — amd64 included, deliberately: there is
// no AVX2 kernel because nothing here can test one — use the portable code.
// The purego tag forces this path on arm64 too, for A/B measurement.

func dot(a, b []float32) float32              { return dotGeneric(a, b) }
func squaredEuclidean(a, b []float32) float32 { return squaredEuclideanGeneric(a, b) }
