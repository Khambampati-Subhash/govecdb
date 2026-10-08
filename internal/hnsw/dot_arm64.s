//go:build arm64 && !purego

#include "textflag.h"

// The Go assembler has no mnemonic for the vector FADD/FADDP forms, so they are
// encoded by hand. d, n, m are V-register numbers; the arrangement is .4S.
#define VFADD4S(m, n, d) WORD $(0x4E20D400 | ((m)<<16) | ((n)<<5) | (d))
#define VFADDP4S(m, n, d) WORD $(0x6E20D400 | ((m)<<16) | ((n)<<5) | (d))
// FADDP Sd, Vn.2S: the final pairwise add down to one scalar.
#define FADDP2S(n, d) WORD $(0x7E30D800 | ((n)<<5) | (d))

// func dot(a, b []float32) float32
//
// 32 floats per iteration into eight 4-lane accumulators (V16-V23). Eight, not
// four: an FMLA has ~4 cycles of latency and the M-series cores issue several
// per cycle, so four chains leave the pipes idle waiting on their own results.
// A 4-wide loop and a scalar loop finish whatever is left.
TEXT ·dot(SB), NOSPLIT, $0-52
	MOVD a_base+0(FP), R0
	MOVD a_len+8(FP), R2
	MOVD b_base+24(FP), R1

	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16

loop32:
	CMP  $32, R2
	BLT  loop4
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V12.S4, V13.S4, V14.S4, V15.S4]
	VFMLA V0.S4, V8.S4, V16.S4
	VFMLA V1.S4, V9.S4, V17.S4
	VFMLA V2.S4, V10.S4, V18.S4
	VFMLA V3.S4, V11.S4, V19.S4
	VFMLA V4.S4, V12.S4, V20.S4
	VFMLA V5.S4, V13.S4, V21.S4
	VFMLA V6.S4, V14.S4, V22.S4
	VFMLA V7.S4, V15.S4, V23.S4
	SUB  $32, R2
	B    loop32

loop4:
	CMP  $4, R2
	BLT  reduce
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V8.S4]
	VFMLA V0.S4, V8.S4, V16.S4
	SUB  $4, R2
	B    loop4

reduce:
	// A fixed tree, so the result is deterministic for a given input.
	VFADD4S(17, 16, 16)
	VFADD4S(19, 18, 18)
	VFADD4S(21, 20, 20)
	VFADD4S(23, 22, 22)
	VFADD4S(18, 16, 16)
	VFADD4S(22, 20, 20)
	VFADD4S(20, 16, 16)
	VFADDP4S(16, 16, 16)
	FADDP2S(16, 16)

tail:
	CBZ    R2, done
	FMOVS.P 4(R0), F0
	FMOVS.P 4(R1), F1
	FMULS  F0, F1, F0
	FADDS  F0, F16, F16
	SUB    $1, R2
	B      tail

done:
	FMOVS F16, ret+48(FP)
	RET
