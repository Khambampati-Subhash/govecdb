//go:build arm64 && !purego

#include "textflag.h"

// See dot_arm64.s for why these are hand-encoded.
#define VFADD4S(m, n, d) WORD $(0x4E20D400 | ((m)<<16) | ((n)<<5) | (d))
#define VFSUB4S(m, n, d) WORD $(0x4EA0D400 | ((m)<<16) | ((n)<<5) | (d))
#define VFADDP4S(m, n, d) WORD $(0x6E20D400 | ((m)<<16) | ((n)<<5) | (d))
#define FADDP2S(n, d) WORD $(0x7E30D800 | ((n)<<5) | (d))

// func squaredEuclidean(a, b []float32) float32
//
// Same shape as dot: d = a-b, then FMLA d*d into eight accumulators. The
// subtraction is done explicitly rather than by expanding |a|^2 - 2ab + |b|^2,
// which cancels catastrophically for exactly the near neighbours a search is
// trying to rank.
TEXT ·squaredEuclidean(SB), NOSPLIT, $0-52
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
	VFSUB4S(8, 0, 0)
	VFSUB4S(9, 1, 1)
	VFSUB4S(10, 2, 2)
	VFSUB4S(11, 3, 3)
	VFSUB4S(12, 4, 4)
	VFSUB4S(13, 5, 5)
	VFSUB4S(14, 6, 6)
	VFSUB4S(15, 7, 7)
	VFMLA V0.S4, V0.S4, V16.S4
	VFMLA V1.S4, V1.S4, V17.S4
	VFMLA V2.S4, V2.S4, V18.S4
	VFMLA V3.S4, V3.S4, V19.S4
	VFMLA V4.S4, V4.S4, V20.S4
	VFMLA V5.S4, V5.S4, V21.S4
	VFMLA V6.S4, V6.S4, V22.S4
	VFMLA V7.S4, V7.S4, V23.S4
	SUB  $32, R2
	B    loop32

loop4:
	CMP  $4, R2
	BLT  reduce
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V8.S4]
	VFSUB4S(8, 0, 0)
	VFMLA V0.S4, V0.S4, V16.S4
	SUB  $4, R2
	B    loop4

reduce:
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
	FSUBS  F1, F0, F0
	FMULS  F0, F0, F0
	FADDS  F0, F16, F16
	SUB    $1, R2
	B      tail

done:
	FMOVS F16, ret+48(FP)
	RET
