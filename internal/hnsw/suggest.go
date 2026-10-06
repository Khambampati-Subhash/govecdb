package hnsw

import "math"

// Choosing ef is the one tuning decision every caller has to make, and the one
// with no safe default: recall at a fixed ef falls as the corpus grows, so a
// constant that works at a thousand vectors is wrong at twenty thousand. This
// file turns the measured surface in docs/benchmarks into a starting point, so
// the guidance lives where it is used rather than only in a README.

// efAnchors is the search width to use for a given recall@10 target on a
// 1,000-vector corpus (dim 128, M 16, uniform random vectors). Everything else
// is scaled off these four points.
//
// They are the widths read off TestSweepEfByScale **plus a margin**, not the
// widths that hit each target exactly. Calibrated on the nose, the function
// undershot on three of four verification corpora — 0.921 against a 0.95 target
// — because the sweep measures one corpus and a caller has a different one, and
// recall on the same configuration spans about eight points across corpora. A
// suggestion that lands under its target more often than not is worse than no
// suggestion, so the anchors aim high enough to absorb that.
var efAnchors = [...]struct{ recall, ef float64 }{
	{0.80, 22},
	{0.90, 36},
	{0.95, 46},
	{0.99, 78},
}

// efGrowthExponent describes how the required width grows with corpus size:
// ef ∝ n^0.78 to hold recall constant.
//
// Measured across a 20x range at three targets, which all agreed:
//
//	recall 0.90   n=1k → ef 26     n=5k → ef ~90    n=20k → ef ~250
//	recall 0.95   n=1k → ef 32     n=5k → ef ~115   n=20k → ef ~330
//	recall 0.99   n=1k → ef 55     n=5k → ef ~190   n=20k → ef ~510
//
// Sub-linear, but a good deal steeper than the log N that the hop count follows
// — holding recall fixed is a stronger demand than "find something nearby".
// This is also the pessimistic case: it is measured on uniform random vectors,
// which have no cluster structure for the graph to exploit.
const efGrowthExponent = 0.78

// efKExponent describes how the required width grows with k: ef ∝ k^0.2.
//
// It used to be assumed linear, which was never measured and turned out to be
// the largest error in the formula. Asked for a 100-candidate pool — the shape a
// retrieval system reranking to 20 asks for — a linear term made every search
// ten times wider than a k=10 one. Measured, the width that holds a recall
// target barely moves with k as long as it is at least k:
//
//	uniform, dim 128, n=62.5k, target 0.95   k=10 → ef 768    k=100 → ef 1024
//	uniform, dim 512, n=62.5k, target 0.95   k=10 → ef 3072   k=100 → ef 3072
//	uniform, dim 128, n=5k,    target 0.95   k=10 → ef 128    k=100 → ef 192
//
// That is an exponent between 0 and 0.18; 0.2 is the top of that range plus a
// little, because undershooting a floor is the expensive direction to be wrong.
const efKExponent = 0.2

// efMExponent describes how the required width falls as M rises: ef ∝
// (16/M)^0.85. Measured at equal recall, M=32 needs half the width M=16 does —
// 384 → 192 at 20k × 128, 3072 → 1536 at 62.5k × 512 — which is an exponent of
// 1. The 0.85 keeps some of that as margin. Without the term, an M=32 index
// searched as wide as an M=16 one with twice the edges per hop, paying for the
// higher M twice.
const efMExponent = 0.85

// calibrationMargin is the headroom Calibrate adds over the width it measured.
// The sample queries are the collection's own vectors, which sit exactly on the
// data; real queries sit near it, and are a little harder. The margin is also
// what absorbs the noise in a 64-query sample.
const calibrationMargin = 1.5

// SuggestedEf returns a starting search width for Search(query, k, ef) on a
// corpus of n vectors.
//
// targetRecall is treated as a **floor to clear, not a point to hit**: the
// anchors carry margin, so a suggestion typically lands a few points above what
// was asked for. That is the useful direction to be wrong in — "at least 0.95"
// is a request someone can act on, where "0.95 give or take four points either
// way" is not. Measured, the 0.95 target returns 0.969 at 1,000 vectors and
// 0.972 at 5,000.
//
// # This is a starting point, not a guarantee
//
// It is fitted to **uniform random 128-dimensional vectors**, and knows nothing
// about the data it is asked about — which is the limit that matters most.
// Measured at 62,500 vectors and a 0.95 target, uniform data at dimension 512
// needs ef ≈ 3,072 and tightly clustered data at the same size needs ≈ 10; this
// formula says 1,158 for both, wrong by 3x one way and 100x the other.
// (*Graph).Calibrate measures the graph's own data and corrects for exactly
// that, and is what a database built on this runs. Further limits:
//
//   - **Recall varies with the corpus, not just its size.** The same
//     configuration measured on three different random corpora returned 0.595,
//     0.646 and 0.677 — an eight-point spread. Variance is widest in the middle
//     of the recall range and compresses near saturation, so a suggestion aimed
//     at 0.95 lands far more reliably than one aimed at 0.65.
//   - **k scales weakly**, as k^0.2 — see efKExponent. The result is never
//     below k.
//
// So: use it to start, then measure on your own data — which is what Calibrate
// does. TestSuggestedEfAchievesTarget is what keeps this function honest: it
// builds real graphs at several sizes, at k=10 and 100 and M=16 and 32, and
// fails if a suggestion misses its target.
//
// This is the M=16 curve; (*Graph).SuggestedEf accounts for the graph's own M
// and for a calibration, when one has run.
func SuggestedEf(n, k int, targetRecall float64) int {
	return suggestEf(n, k, targetRecall, 16, 1)
}

// suggestEf is the whole formula: the anchored base, scaled by corpus size, k
// and M, and then by a calibration factor that is 1 until Calibrate has run.
func suggestEf(n, k int, targetRecall float64, m int, scale float64) int {
	if k < 1 {
		k = 1
	}
	if n < 1 {
		return k
	}

	// Outside the measured band the anchors say nothing, so clamp rather than
	// extrapolate into a number that looks authoritative and is invented.
	lo, hi := efAnchors[0], efAnchors[len(efAnchors)-1]
	switch {
	case targetRecall <= lo.recall:
		targetRecall = lo.recall
	case targetRecall >= hi.recall:
		targetRecall = hi.recall
	}

	base := interpolateAnchors(targetRecall)
	ef := base *
		math.Pow(float64(n)/1000, efGrowthExponent) *
		math.Pow(float64(k)/10, efKExponent) *
		math.Pow(16/float64(max(m, 2)), efMExponent) *
		scale

	out := int(math.Ceil(ef))
	if out < k {
		out = k // Search clamps this anyway; returning it is clearer than relying on that
	}
	return out
}

// SuggestedEf is the same calculation against this graph's live population and
// its own M, scaled by the last Calibrate if there was one, so callers do not
// have to track what the graph already knows.
func (g *Graph) SuggestedEf(k int, targetRecall float64) int {
	return suggestEf(g.Len(), k, targetRecall, g.cfg.M, g.EfScale())
}

// interpolateAnchors reads a base width off the measured points, linearly
// between them. targetRecall is already clamped to the anchored range.
func interpolateAnchors(targetRecall float64) float64 {
	for i := 1; i < len(efAnchors); i++ {
		hi := efAnchors[i]
		if targetRecall > hi.recall {
			continue
		}
		lo := efAnchors[i-1]
		span := hi.recall - lo.recall
		if span == 0 {
			return hi.ef
		}
		frac := (targetRecall - lo.recall) / span
		return lo.ef + frac*(hi.ef-lo.ef)
	}
	return efAnchors[len(efAnchors)-1].ef
}
