package filter

import (
	"errors"
	"math"
	"testing"

	"github.com/khambampati-subhash/govecdb/internal/store"
)

func doc() store.Metadata {
	return store.Metadata{
		"source": "handbook.pdf",
		"page":   int64(12),
		"score":  0.75,
		"draft":  false,
	}
}

func TestComparisons(t *testing.T) {
	cases := []struct {
		name string
		f    Filter
		want bool
	}{
		{"eq string", Eq("source", "handbook.pdf"), true},
		{"eq string mismatch", Eq("source", "other.pdf"), false},
		{"eq int", Eq("page", int64(12)), true},
		{"eq int from untyped constant", Eq("page", 12), true},
		{"eq int against float", Eq("page", 12.0), true},
		{"eq float", Eq("score", 0.75), true},
		{"eq bool", Eq("draft", false), true},
		{"eq bool mismatch", Eq("draft", true), false},

		{"ne string", Ne("source", "other.pdf"), true},
		{"ne string same", Ne("source", "handbook.pdf"), false},
		{"ne across kinds counts as different", Ne("page", "12"), true},

		{"lt", Lt("page", 13), true},
		{"lt at boundary", Lt("page", 12), false},
		{"lte at boundary", Lte("page", 12), true},
		{"gt", Gt("page", 11), true},
		{"gt at boundary", Gt("page", 12), false},
		{"gte at boundary", Gte("page", 12), true},
		{"lt float against int value", Lt("page", 12.5), true},
		{"gt on float value", Gt("score", 0.5), true},

		{"lt on strings is lexicographic", Lt("source", "z"), true},
		{"gt on strings is lexicographic", Gt("source", "a"), true},

		// Booleans have no ordering, so every ordering comparison on one is
		// false — including the pair that would otherwise be complementary.
		{"lt on bool", Lt("draft", true), false},
		{"gte on bool", Gte("draft", true), false},

		// Comparing across kinds never orders.
		{"lt string against number", Lt("source", 5), false},
		{"gt number against string", Gt("page", "5"), false},

		{"in", In("source", "a.pdf", "handbook.pdf"), true},
		{"in miss", In("source", "a.pdf", "b.pdf"), false},
		{"in mixed numeric", In("page", 11, 12.0, 13), true},
		{"in empty matches nothing", In("source"), false},

		{"exists", Exists("source"), true},
		{"exists on absent key", Exists("nope"), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.f.Validate(); err != nil {
				t.Fatalf("Validate: %v", err)
			}
			if got := tc.f.Match(doc()); got != tc.want {
				t.Fatalf("Match = %v, want %v", got, tc.want)
			}
		})
	}
}

// The uniform rule, stated once as a test so it cannot drift: every predicate
// about a key is false when the key is not there. Ne is the one that surprises,
// which is exactly why it is in the list.
func TestEveryPredicateIsFalseOnAnAbsentKey(t *testing.T) {
	fs := map[string]Filter{
		"Eq":     Eq("missing", "x"),
		"Ne":     Ne("missing", "x"),
		"Lt":     Lt("missing", 1),
		"Lte":    Lte("missing", 1),
		"Gt":     Gt("missing", 1),
		"Gte":    Gte("missing", 1),
		"In":     In("missing", "x", "y"),
		"Exists": Exists("missing"),
	}
	for name, f := range fs {
		t.Run(name, func(t *testing.T) {
			if f.Match(doc()) {
				t.Fatalf("%s matched a key that is not present", name)
			}
		})
	}
}

// And Not is how the other question gets asked.
func TestNotReachesAbsentKeys(t *testing.T) {
	md := doc()
	if Ne("missing", "x").Match(md) {
		t.Error("Ne matched an absent key")
	}
	if !Not(Eq("missing", "x")).Match(md) {
		t.Error("Not(Eq) should match an absent key")
	}
	if Not(Eq("source", "handbook.pdf")).Match(md) {
		t.Error("Not(Eq) should not match when the value does equal")
	}
}

// Metadata may hold a NaN — Validate only checks the vector values for
// finiteness — so the filter has to answer for one rather than assume it away.
func TestNaNMetadataComparesFalseExceptNe(t *testing.T) {
	md := store.Metadata{"score": math.NaN()}

	for name, f := range map[string]Filter{
		"Eq":  Eq("score", math.NaN()),
		"Lt":  Lt("score", 1.0),
		"Gt":  Gt("score", 1.0),
		"Lte": Lte("score", math.NaN()),
	} {
		if f.Match(md) {
			t.Errorf("%s matched against a NaN", name)
		}
	}

	// Ne is "present and not equal", and a NaN equals nothing — including
	// itself. IEEE's rule, surfaced rather than special-cased.
	if !Ne("score", math.NaN()).Match(md) {
		t.Error("Ne should be true for a NaN, which is unequal to everything")
	}
}

func TestLogicalCombinators(t *testing.T) {
	md := doc()

	cases := []struct {
		name string
		f    Filter
		want bool
	}{
		{"and both", And(Eq("source", "handbook.pdf"), Gt("page", 10)), true},
		{"and one fails", And(Eq("source", "handbook.pdf"), Gt("page", 100)), false},
		{"or first", Or(Eq("source", "handbook.pdf"), Gt("page", 100)), true},
		{"or second", Or(Eq("source", "nope.pdf"), Gt("page", 10)), true},
		{"or neither", Or(Eq("source", "nope.pdf"), Gt("page", 100)), false},
		{"not", Not(Eq("source", "nope.pdf")), true},
		{"nested", And(
			Or(Eq("source", "handbook.pdf"), Eq("source", "manual.pdf")),
			Not(Eq("draft", true)),
			Gte("score", 0.5),
		), true},

		// The identities. An empty conjunction constrains nothing; an empty
		// disjunction offers nothing. Both matter when a filter is assembled
		// from a slice that turns out to be empty.
		{"empty and matches", And(), true},
		{"empty or does not", Or(), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.f.Validate(); err != nil {
				t.Fatalf("Validate: %v", err)
			}
			if got := tc.f.Match(md); got != tc.want {
				t.Fatalf("Match = %v, want %v", got, tc.want)
			}
		})
	}
}

// A vector with no metadata at all reaches Match as a nil map, because "has no
// metadata" is a thing a filter can legitimately be asked about.
func TestNilMetadataIsAnEmptyDocument(t *testing.T) {
	var md store.Metadata

	if Exists("anything").Match(md) {
		t.Error("Exists matched in a nil metadata map")
	}
	if !Not(Exists("anything")).Match(md) {
		t.Error("Not(Exists) should match a vector carrying no metadata")
	}
	if !And().Match(md) {
		t.Error("the empty conjunction should match anything, including nothing")
	}
}

func TestValidateReportsBadOperands(t *testing.T) {
	cases := []struct {
		name string
		f    Filter
	}{
		{"Eq", Eq("k", []string{"a"})},
		{"Ne", Ne("k", struct{}{})},
		{"Lt", Lt("k", nil)},
		{"Lte", Lte("k", map[string]any{})},
		{"Gt", Gt("k", uint64(math.MaxInt64)+1)},
		{"Gte", Gte("k", new(int))},
		{"In first value", In("k", []byte("x"), "ok")},
		{"In later value", In("k", "ok", []byte("x"))},
		{"nested in And", And(Eq("a", 1), Eq("b", []int{2}))},
		{"nested in Or", Or(Eq("a", 1), Eq("b", []int{2}))},
		{"nested in Not", Not(Eq("b", []int{2}))},
		{"deeply nested", And(Or(And(Lt("x", []int{1}))))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.f.Validate(); !errors.Is(err, ErrOperand) {
				t.Fatalf("Validate = %v, want ErrOperand", err)
			}
		})
	}
}

// A nil child is a construction bug. Validate names it rather than letting Match
// panic, and rather than quietly treating it as "matches nothing" — which would
// turn one typo into an empty result set with no explanation.
func TestValidateReportsNilChildren(t *testing.T) {
	cases := []struct {
		name string
		f    Filter
	}{
		{"And", And(Eq("a", 1), nil)},
		{"Or", Or(nil)},
		{"Not", Not(nil)},
		{"nested", And(Or(Eq("a", 1), Not(nil)))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.f.Validate(); !errors.Is(err, ErrNilFilter) {
				t.Fatalf("Validate = %v, want ErrNilFilter", err)
			}
		})
	}
}

func TestValidatePassesOnGoodFilters(t *testing.T) {
	f := And(
		Eq("source", "handbook.pdf"),
		In("page", 1, 2, 3),
		Not(Exists("deleted")),
		Or(Gt("score", 0.5), Lte("score", 0.1)),
	)
	if err := f.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

// An operand error names the operator and the key, because a filter with a dozen
// clauses is where this error actually shows up.
func TestOperandErrorNamesTheClause(t *testing.T) {
	err := Gt("published", []string{"x"}).Validate()
	if err == nil {
		t.Fatal("want an error")
	}
	msg := err.Error()
	for _, want := range []string{"Gt", "published"} {
		if !contains(msg, want) {
			t.Errorf("error %q does not mention %q", msg, want)
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
