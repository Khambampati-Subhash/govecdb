package hnsw

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// kernelDims covers the shapes the 4-wide unrolled kernels can get wrong. Every
// kernel processes four lanes per iteration and finishes with a scalar tail, so
// the interesting dimensions are the ones either side of a multiple of four —
// and 1, 2, 3, which are *only* tail. Embedding-sized dims are here too, since
// that is what the index is actually used with.
var kernelDims = []int{
	1, 2, 3, 4, 5, 6, 7, 8, 9,
	15, 16, 17, 31, 32, 33, 63, 64, 65,
	127, 128, 129, 383, 384, 385, 767, 768, 769, 1536,
}

// naiveDot and friends are the reference: the obvious one-accumulator loop, in
// float64 so the comparison is against the real answer rather than against
// another float32 rounding order.
func naiveDot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func naiveSquaredEuclidean(a, b []float32) float64 {
	var s float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		s += d * d
	}
	return s
}

func naiveCosineDistance(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 1
	}
	return 1 - dot/(math.Sqrt(na)*math.Sqrt(nb))
}

// closeEnough compares a float32 kernel result against the float64 truth,
// relative to the magnitude of the answer. float32 carries ~7 significant
// digits and these sums run to 1536 terms, so 1e-4 relative is generous enough
// to never flake and tight enough that a wrong tail element (which shifts the
// result by a whole term) always fails.
func closeEnough(got float32, want float64) bool {
	scale := math.Max(math.Abs(want), 1)
	return math.Abs(float64(got)-want)/scale <= 1e-4
}

// signedVector spans negatives as well as positives. All-positive inputs let
// accumulation errors pile up in one direction and, worse, would hide a sign
// error in a tail element.
func signedVector(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}

// TestKernelsMatchNaiveAcrossDimensions is the test the unrolled kernels never
// had. A wrong tail loop is invisible at dim 128 and wrong at dim 129, and
// every other test in this package uses dimensions divisible by four.
func TestKernelsMatchNaiveAcrossDimensions(t *testing.T) {
	rng := rand.New(rand.NewSource(101))

	for _, dim := range kernelDims {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			for trial := range 20 {
				a := signedVector(rng, dim)
				b := signedVector(rng, dim)

				if got, want := Dot(a, b), naiveDot(a, b); !closeEnough(got, want) {
					t.Fatalf("trial %d: Dot = %v, want %v", trial, got, want)
				}
				if got, want := SquaredEuclidean(a, b), naiveSquaredEuclidean(a, b); !closeEnough(got, want) {
					t.Fatalf("trial %d: SquaredEuclidean = %v, want %v", trial, got, want)
				}
				if got, want := CosineDistance(a, b), naiveCosineDistance(a, b); !closeEnough(got, want) {
					t.Fatalf("trial %d: CosineDistance = %v, want %v", trial, got, want)
				}
				if got, want := NegativeDot(a, b), -naiveDot(a, b); !closeEnough(got, want) {
					t.Fatalf("trial %d: NegativeDot = %v, want %v", trial, got, want)
				}
			}
		})
	}
}

// TestKernelTailIsNotIgnored is a targeted version of the above: a kernel that
// dropped its tail entirely would still pass a loose statistical check, so this
// makes the tail element the *only* difference between two inputs.
func TestKernelTailIsNotIgnored(t *testing.T) {
	for _, dim := range []int{5, 6, 7, 9, 129, 769} {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			a := make([]float32, dim)
			b := make([]float32, dim)
			for i := range a {
				a[i], b[i] = 1, 1
			}
			// Change only the last element, which always lands in the tail.
			b[dim-1] = -1

			if got, want := Dot(a, b), float64(dim-2); !closeEnough(got, want) {
				t.Fatalf("Dot ignored the tail: got %v, want %v", got, want)
			}
			if got, want := SquaredEuclidean(a, b), 4.0; !closeEnough(got, want) {
				t.Fatalf("SquaredEuclidean ignored the tail: got %v, want %v", got, want)
			}
		})
	}
}

// TestKnownDistances pins the actual numbers, so a refactor cannot quietly
// redefine what a metric means while every relative comparison still passes.
func TestKnownDistances(t *testing.T) {
	var (
		e1  = []float32{1, 0, 0, 0}
		e2  = []float32{0, 1, 0, 0}
		neg = []float32{-1, 0, 0, 0}
		two = []float32{2, 0, 0, 0}
	)

	for _, tc := range []struct {
		name string
		got  float32
		want float64
	}{
		{"cosine identical", CosineDistance(e1, e1), 0},
		{"cosine orthogonal", CosineDistance(e1, e2), 1},
		{"cosine opposite", CosineDistance(e1, neg), 2},
		{"cosine ignores magnitude", CosineDistance(e1, two), 0},
		{"cosine zero vector", CosineDistance(e1, []float32{0, 0, 0, 0}), 1},

		{"euclidean identical", SquaredEuclidean(e1, e1), 0},
		{"euclidean orthogonal", SquaredEuclidean(e1, e2), 2},
		{"euclidean opposite", SquaredEuclidean(e1, neg), 4},
		{"euclidean respects magnitude", SquaredEuclidean(e1, two), 1},

		{"dot identical", NegativeDot(e1, e1), -1},
		{"dot orthogonal", NegativeDot(e1, e2), 0},
		{"dot opposite", NegativeDot(e1, neg), 1},
		{"dot rewards magnitude", NegativeDot(e1, two), -2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !closeEnough(tc.got, tc.want) {
				t.Fatalf("got %v, want %v", tc.got, tc.want)
			}
		})
	}
}

// TestSmallerMeansCloser is the one property the whole graph is built on: it
// never branches on the metric, so every metric must agree that a smaller
// number means a nearer vector.
func TestSmallerMeansCloser(t *testing.T) {
	rng := rand.New(rand.NewSource(102))

	for _, tc := range []struct {
		name string
		fn   DistanceFunc
	}{
		{"CosineDistance", CosineDistance},
		{"SquaredEuclidean", SquaredEuclidean},
		{"NegativeDot", NegativeDot},
		{"oneMinusDot", oneMinusDot},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, dim := range []int{3, 4, 33, 128} {
				for range 50 {
					query := signedVector(rng, dim)
					Normalize(query)

					// near is query nudged; far is an independent vector. Both
					// are unit length so oneMinusDot is meaningful too.
					near := make([]float32, dim)
					copy(near, query)
					for i := range near {
						near[i] += (rng.Float32()*2 - 1) * 0.01
					}
					Normalize(near)
					far := signedVector(rng, dim)
					Normalize(far)

					dNear, dFar := tc.fn(query, near), tc.fn(query, far)
					// An independent random vector can land near the query by
					// chance in low dimensions; only assert when they are
					// genuinely different directions.
					if math.Abs(float64(CosineDistance(query, far))) < 0.05 {
						continue
					}
					if dNear > dFar {
						t.Fatalf("dim %d: perturbed copy scored %v, random vector %v — larger should mean farther",
							dim, dNear, dFar)
					}
				}
			}
		})
	}
}

func TestNormalizeAcrossDimensions(t *testing.T) {
	rng := rand.New(rand.NewSource(103))

	for _, dim := range kernelDims {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			v := signedVector(rng, dim)
			// Vary magnitude wildly: normalization must be scale-invariant.
			scale := float32(math.Pow(10, float64(rng.Intn(9)-4)))
			for i := range v {
				v[i] *= scale
			}
			original := append([]float32(nil), v...)

			Normalize(v)

			if norm := math.Sqrt(naiveDot(v, v)); math.Abs(norm-1) > 1e-4 {
				t.Fatalf("norm after Normalize = %v, want 1", norm)
			}
			// Direction preserved: the normalized vector must be a positive
			// multiple of the original, elementwise.
			if got := naiveCosineDistance(original, v); math.Abs(got) > 1e-4 {
				t.Fatalf("Normalize changed direction: cosine distance %v", got)
			}
		})
	}
}

func TestNormalizeZeroVector(t *testing.T) {
	// A zero vector has no direction to preserve, and dividing by its norm
	// would produce NaNs that poison every comparison they touch.
	v := make([]float32, 8)
	Normalize(v)
	for i, x := range v {
		if x != 0 {
			t.Fatalf("Normalize(zero)[%d] = %v, want 0", i, x)
		}
	}
}

// TestFastFuncMatchesGeneralForm justifies the optimization the whole cosine
// path rests on: for unit-length vectors, 1-dot(a,b) is the same number as the
// general cosine formula, not merely the same ranking.
func TestFastFuncMatchesGeneralForm(t *testing.T) {
	rng := rand.New(rand.NewSource(104))

	for _, dim := range kernelDims {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			fast := Cosine.fastFunc(true)
			general := Cosine.Func()

			for range 20 {
				a, b := signedVector(rng, dim), signedVector(rng, dim)
				Normalize(a)
				Normalize(b)

				got, want := fast(a, b), general(a, b)
				if math.Abs(float64(got-want)) > 1e-4 {
					t.Fatalf("fast %v vs general %v", got, want)
				}
			}
		})
	}
}

func TestMetricWiring(t *testing.T) {
	// normalizes() decides whether the graph rewrites stored vectors, so getting
	// it wrong for Euclidean or DotProduct silently redefines those metrics.
	if !Cosine.normalizes() {
		t.Fatal("Cosine must normalize")
	}
	if Euclidean.normalizes() || DotProduct.normalizes() {
		t.Fatal("only Cosine may normalize; the others would change meaning")
	}

	a, b := []float32{1, 2, 3, 4}, []float32{4, 3, 2, 1}
	for _, tc := range []struct {
		metric Metric
		want   float32
	}{
		{Cosine, CosineDistance(a, b)},
		{Euclidean, SquaredEuclidean(a, b)},
		{DotProduct, NegativeDot(a, b)},
	} {
		if got := tc.metric.Func()(a, b); got != tc.want {
			t.Fatalf("metric %d: Func() gave %v, want %v", tc.metric, got, tc.want)
		}
		// Only Cosine has a fast form; the others must be unaffected by the flag.
		if tc.metric != Cosine {
			if got := tc.metric.fastFunc(true)(a, b); got != tc.want {
				t.Fatalf("metric %d: fastFunc(true) diverged from Func()", tc.metric)
			}
		}
	}
}
