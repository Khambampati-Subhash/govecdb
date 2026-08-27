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
// It is calibrated on **uniform random 128-dimensional vectors at M=16**, which
// is the hardest case a graph index faces — real embeddings cluster, and
// clustered data in the same sweep reached higher recall at a third of the
// latency. Two further limits worth knowing before trusting a number from here:
//
//   - **Recall varies with the corpus, not just its size.** The same
//     configuration measured on three different random corpora returned 0.595,
//     0.646 and 0.677 — an eight-point spread. Variance is widest in the middle
//     of the recall range and compresses near saturation, so a suggestion aimed
//     at 0.95 lands far more reliably than one aimed at 0.65.
//   - **Only k=10 was measured.** Scaling with k is assumed linear, which is
//     the reasonable default and is not something the sweeps checked.
//
// So: use it to start, then measure on your own data. TestSuggestedEfAchieves-
// Target is what keeps this function honest — it builds real graphs at several
// sizes and fails if a suggestion misses its target.
//
// Larger M shifts the whole curve: at M=32 the same recall arrives at roughly
// half the ef. Since M is paid once at build and ef on every query, a read-heavy
// workload is usually better off raising M than following this function upward.
func SuggestedEf(n, k int, targetRecall float64) int {
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
	ef := base * math.Pow(float64(n)/1000, efGrowthExponent) * float64(k) / 10

	out := int(math.Ceil(ef))
	if out < k {
		out = k // Search clamps this anyway; returning it is clearer than relying on that
	}
	return out
}

// SuggestedEf is the same calculation against this graph's live population, so
// callers do not have to track a count the graph already knows.
func (g *Graph) SuggestedEf(k int, targetRecall float64) int {
	return SuggestedEf(g.Len(), k, targetRecall)
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
