//go:build amd64 && !purego

package hnsw

// AVX2+FMA kernels in dot_amd64.s and l2_amd64.s, chosen once at startup.
//
// Unlike arm64, where NEON is architectural, amd64 has to ask: AVX2 is not in
// the baseline GOAMD64=v1 targets, and a CPU that has it may still run under an
// OS (or a hypervisor) that does not save the YMM registers on a context
// switch. So both are checked — CPUID for the instructions, XGETBV for the
// state — and anything short of both runs the Go kernels, which are correct
// everywhere. The branch below is on a value fixed before main and is
// predicted perfectly; it measured inside the noise against calling the
// assembly unconditionally.
//
// Like the NEON kernels these read exactly len(a) elements from each argument,
// and differ from the Go kernels by a few ulps because they sum 32 lanes where
// the Go code sums 8. See distance_arm64.go for why that is invisible to the
// graph-equality tests and harmless across architectures.

var useAVX2 = hasAVX2FMA()

func dot(a, b []float32) float32 {
	if useAVX2 {
		return dotAVX2(a, b)
	}
	return dotGeneric(a, b)
}

func squaredEuclidean(a, b []float32) float32 {
	if useAVX2 {
		return squaredEuclideanAVX2(a, b)
	}
	return squaredEuclideanGeneric(a, b)
}

// hasAVX2FMA is the check the Go runtime's own internal/cpu makes, which this
// package cannot import and will not take a dependency for.
func hasAVX2FMA() bool {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 7 {
		return false
	}
	_, _, ecx1, _ := cpuid(1, 0)
	const (
		fma     = 1 << 12
		osxsave = 1 << 27
		avx     = 1 << 28
	)
	if ecx1&(fma|osxsave|avx) != fma|osxsave|avx {
		return false
	}
	// XMM (bit 1) and YMM (bit 2) state both enabled by the OS.
	if xcr0, _ := xgetbv(); xcr0&0b110 != 0b110 {
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	const avx2 = 1 << 5
	return ebx7&avx2 != 0
}

//go:noescape
func dotAVX2(a, b []float32) float32

//go:noescape
func squaredEuclideanAVX2(a, b []float32) float32

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

func xgetbv() (eax, edx uint32)
