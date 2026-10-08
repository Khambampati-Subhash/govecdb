//go:build arm64 && !purego

package hnsw

// NEON kernels in dot_arm64.s and l2_arm64.s. Both read exactly len(a)
// elements from each argument; the exported wrappers (Dot, SquaredEuclidean)
// re-slice b to len(a) first, which is the bounds check these cannot do.
//
// Floating-point addition is not associative, so these differ from the Go
// kernels by a few ulps: 32 lanes are summed where the Go code sums 8. Every
// graph-vs-graph equality test compares two graphs built with the same kernel,
// so that is invisible to them; a graph serialized on one architecture and
// searched on another sees distances that differ in the last bits, which is
// within what float32 promises anyway.

//go:noescape
func dot(a, b []float32) float32

//go:noescape
func squaredEuclidean(a, b []float32) float32
