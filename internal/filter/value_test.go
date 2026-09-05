package filter

import (
	"errors"
	"math"
	"testing"
)

func TestNormalizeAcceptsEveryIntegerWidth(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want any
	}{
		{"string", "x", "x"},
		{"bool", true, true},
		{"int64", int64(7), int64(7)},
		{"float64", 1.5, 1.5},
		{"int", 7, int64(7)},
		{"int8", int8(7), int64(7)},
		{"int16", int16(7), int64(7)},
		{"int32", int32(7), int64(7)},
		{"uint", uint(7), int64(7)},
		{"uint8", uint8(7), int64(7)},
		{"uint16", uint16(7), int64(7)},
		{"uint32", uint32(7), int64(7)},
		{"uint64", uint64(7), int64(7)},
		{"float32", float32(1.5), 1.5},
		{"negative int", -7, int64(-7)},
		{"max int64", int64(math.MaxInt64), int64(math.MaxInt64)},
		{"uint64 at max int64", uint64(math.MaxInt64), int64(math.MaxInt64)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := normalize(tc.in)
			if err != nil {
				t.Fatalf("normalize(%v): %v", tc.in, err)
			}
			if got != tc.want {
				t.Fatalf("normalize(%v) = %#v, want %#v", tc.in, got, tc.want)
			}
		})
	}
}

// A uint64 past MaxInt64 has to be refused rather than wrapped: converting it
// would produce a negative number, and a filter that silently means the opposite
// of what it says is worse than one that will not run.
func TestNormalizeRefusesWhatItCannotRepresent(t *testing.T) {
	cases := []struct {
		name string
		in   any
	}{
		{"uint64 past int64", uint64(math.MaxInt64) + 1},
		{"slice", []string{"a"}},
		{"map", map[string]any{}},
		{"nil", nil},
		{"struct", struct{}{}},
		{"pointer", new(int)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := normalize(tc.in); !errors.Is(err, ErrOperand) {
				t.Fatalf("normalize(%#v) error = %v, want ErrOperand", tc.in, err)
			}
		})
	}
}

// The reason compareIntFloat exists. float64(i) rounds once |i| passes 2^53, so
// the naive comparison reports two distinct numbers as equal — and the range
// where it starts happening is where ordinary data lives, not a corner.
func TestCompareIntFloatIsExactPastTheFloatMantissa(t *testing.T) {
	const big = int64(1) << 62
	i := big + 1 // odd, so no float64 represents it
	f := float64(big)

	if float64(i) != f {
		t.Fatal("premise broken: float64(i) was expected to round onto f")
	}

	got, ok := compareIntFloat(i, f)
	if !ok {
		t.Fatal("compareIntFloat reported not-comparable for two finite numbers")
	}
	if got != 1 {
		t.Fatalf("compareIntFloat(%d, %v) = %d, want 1 (the int is larger)", i, f, got)
	}

	// And the same through the public surface, which is what actually matters.
	md := map[string]any{"n": i}
	if Eq("n", f).Match(md) {
		t.Error("Eq matched two numbers that differ by one")
	}
	if !Gt("n", f).Match(md) {
		t.Error("Gt failed on an int64 larger than the float it rounds to")
	}
}

// A nanosecond timestamp is ~1.7e18, which is exactly the range above. This is
// the concrete shape the test above defends.
func TestCompareIntFloatOnNanosecondTimestamps(t *testing.T) {
	const ns = int64(1_756_000_000_123_456_789)
	md := map[string]any{"ingested": ns}

	if !Gt("ingested", float64(1_756_000_000_000_000_000)).Match(md) {
		t.Error("timestamp should be greater than the earlier float bound")
	}
	if !Lt("ingested", float64(1_757_000_000_000_000_000)).Match(md) {
		t.Error("timestamp should be less than the later float bound")
	}
}

func TestCompareIntFloatBoundariesAndFractions(t *testing.T) {
	cases := []struct {
		name string
		i    int64
		f    float64
		want int
		ok   bool
	}{
		{"equal whole", 5, 5.0, 0, true},
		{"int below fraction", 5, 5.5, -1, true},
		{"int above fraction", 6, 5.5, 1, true},
		{"negative equal", -5, -5.0, 0, true},
		{"negative int above fraction", -2, -2.5, 1, true},
		{"negative int below fraction", -3, -2.5, -1, true},
		{"zero against tiny positive", 0, 1e-300, -1, true},
		{"zero against tiny negative", 0, -1e-300, 1, true},
		{"positive infinity", math.MaxInt64, math.Inf(1), -1, true},
		{"negative infinity", math.MinInt64, math.Inf(-1), 1, true},
		{"NaN", 0, math.NaN(), 0, false},

		// float64(MaxInt64) rounds *up* to 2^63, so MaxInt64 is strictly below
		// it. Getting this backwards is the classic off-by-one in this function.
		{"max int64 against its rounded float", math.MaxInt64, float64(math.MaxInt64), -1, true},
		{"min int64 against its exact float", math.MinInt64, float64(math.MinInt64), 0, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := compareIntFloat(tc.i, tc.f)
			if ok != tc.ok {
				t.Fatalf("compareIntFloat(%d, %v) ok = %v, want %v", tc.i, tc.f, ok, tc.ok)
			}
			if ok && got != tc.want {
				t.Fatalf("compareIntFloat(%d, %v) = %d, want %d", tc.i, tc.f, got, tc.want)
			}
		})
	}
}

func TestOrderRefusesWhatHasNoOrdering(t *testing.T) {
	cases := []struct {
		name            string
		stored, operand any
	}{
		{"string against int", "5", int64(5)},
		{"int against string", int64(5), "5"},
		{"bool against bool", true, true},
		{"bool against int", true, int64(1)},
		{"int against bool", int64(1), true},
		{"float NaN stored", math.NaN(), 1.0},
		{"float NaN operand", 1.0, math.NaN()},
		{"string against bool", "true", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, ok := order(tc.stored, tc.operand); ok {
				t.Fatalf("order(%#v, %#v) reported an ordering it should not have", tc.stored, tc.operand)
			}
		})
	}
}

func TestEqualAcrossNumericTypes(t *testing.T) {
	cases := []struct {
		name            string
		stored, operand any
		want            bool
	}{
		{"int64 to float64", int64(5), 5.0, true},
		{"float64 to int64", 5.0, int64(5), true},
		{"int64 to fractional float", int64(5), 5.5, false},
		{"strings", "a", "a", true},
		{"different strings", "a", "b", false},
		{"bools", true, true, true},
		{"different bools", true, false, false},
		{"bool against number", true, int64(1), false},
		{"NaN against itself", math.NaN(), math.NaN(), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := equal(tc.stored, tc.operand); got != tc.want {
				t.Fatalf("equal(%#v, %#v) = %v, want %v", tc.stored, tc.operand, got, tc.want)
			}
		})
	}
}
