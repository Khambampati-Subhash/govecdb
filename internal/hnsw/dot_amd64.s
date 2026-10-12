//go:build amd64 && !purego

#include "textflag.h"

// func dotAVX2(a, b []float32) float32
//
// 64 floats per iteration into eight 8-lane accumulators (Y0-Y3, Y8-Y11), an
// FMA each with one operand straight from memory. Eight, as on arm64: an FMA
// has ~4 cycles of latency and two ports, so fewer chains leave the ports
// waiting on their own results — four measured 21% slower at dim 512 (23.6 vs
// 18.7 ns, Xeon at 2.3 GHz). That takes all sixteen YMM registers. An 8-wide
// loop into the first accumulator and a scalar loop finish whatever is left.
//
// Callers must have checked AVX2 and FMA; see distance_amd64.go.
TEXT ·dotAVX2(SB), NOSPLIT, $0-52
	MOVQ a_base+0(FP), SI
	MOVQ a_len+8(FP), CX
	MOVQ b_base+24(FP), DI

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11

loop64:
	CMPQ CX, $64
	JL   loop8
	VMOVUPS (SI), Y4
	VMOVUPS 32(SI), Y5
	VMOVUPS 64(SI), Y6
	VMOVUPS 96(SI), Y7
	VMOVUPS 128(SI), Y12
	VMOVUPS 160(SI), Y13
	VMOVUPS 192(SI), Y14
	VMOVUPS 224(SI), Y15
	VFMADD231PS (DI), Y4, Y0
	VFMADD231PS 32(DI), Y5, Y1
	VFMADD231PS 64(DI), Y6, Y2
	VFMADD231PS 96(DI), Y7, Y3
	VFMADD231PS 128(DI), Y12, Y8
	VFMADD231PS 160(DI), Y13, Y9
	VFMADD231PS 192(DI), Y14, Y10
	VFMADD231PS 224(DI), Y15, Y11
	ADDQ $256, SI
	ADDQ $256, DI
	SUBQ $64, CX
	JMP  loop64

loop8:
	CMPQ CX, $8
	JL   reduce
	VMOVUPS (SI), Y4
	VFMADD231PS (DI), Y4, Y0
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP  loop8

reduce:
	// A fixed tree, so the result is deterministic for a given input.
	VADDPS Y8, Y0, Y0
	VADDPS Y9, Y1, Y1
	VADDPS Y10, Y2, Y2
	VADDPS Y11, Y3, Y3
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

tail:
	TESTQ CX, CX
	JE    done
	VMOVSS (SI), X4
	VFMADD231SS (DI), X4, X0
	ADDQ $4, SI
	ADDQ $4, DI
	DECQ CX
	JMP  tail

done:
	// The upper YMM halves are dirty; leaving them so costs the SSE code that
	// follows a transition penalty on older cores.
	VZEROUPPER
	MOVSS X0, ret+48(FP)
	RET
