package hnsw

// Compact rebuilds the graph over its live vectors and reports how many slots
// were reclaimed. Calling it on a graph with no tombstones is free.
//
// Tombstones never release memory on their own — that is the bargain Delete
// makes, and Insert makes it too, once per replaced vector. A workload that
// re-embeds the same corpus therefore grows without bound while its live
// population stays flat: three passes over 300 vectors leave 300 live vectors
// carried by 1,200 slots. Compaction is what pays that debt back, and it is the
// *only* thing that does.
//
// # Why a rebuild and not a repair
//
// Every neighbor list in the graph is a list of indices into g.nodes, so a slot
// cannot be removed in place: dropping one shifts every index above it and
// invalidates every list in the graph at once. Repairing them in place is the
// same O(N·M) scan that made in-place updates unavailable, except now it has to
// run for every dead slot rather than one.
//
// So compaction does not renumber the live graph — it builds a *new* graph from
// the live vectors and swaps it in whole. The slot-index invariant is never
// violated; it is retired along with the graph that held it.
//
// # What the rebuild produces
//
// Exactly the graph you would have built if the dead vectors had never existed:
// the replacement draws its levels from a fresh RNG seeded from the same config,
// and the live vectors are re-inserted in slot order, so the result is
// deterministic and testable by equality rather than by sampling
// (TestCompactMatchesAFreshBuild).
//
// It is also a *better* graph than the one it replaces, not merely a smaller
// one. Edges that pointed at tombstones become edges between live nodes, and
// the pruning bound in searchLayer tightens again because results fill at full
// speed. Recall goes up and search gets faster; only the memory is the headline.
//
// # Compact stops the world
//
// It holds the write lock for the whole rebuild, which is a full index build:
// seconds for a large graph, during which no search runs. That is deliberate
// for v1 — building the replacement outside the lock would mean writes landing
// in the old graph while the new one is being built, and reconciling them needs
// a change log and a double-buffered swap that the durability layer should
// shape first.
//
// So the index does not decide *when*. It exposes Stats().DeadRatio() and lets
// the caller pick a moment that tolerates the pause.
func (g *Graph) Compact() int {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Cheap enough to call on a timer: a clean graph is not worth rebuilding,
	// and rebuilding it would throw away a perfectly good index for nothing.
	if g.numDeleted == 0 {
		return 0
	}

	fresh := newGraph(g.cfg)
	for _, n := range g.nodes {
		if n.deleted {
			continue
		}
		// The vector moves across as-is rather than being re-prepared: it is
		// already this graph's own normalized copy, and the graph it came from
		// is about to be garbage. Dead vectors are the memory being reclaimed —
		// nothing references them once the swap below lands.
		fresh.insertPrepared(n.id, n.vector)
	}

	reclaimed := g.numDeleted
	g.nodes, g.ids = fresh.nodes, fresh.ids
	g.entry, g.maxLevel = fresh.entry, fresh.maxLevel
	g.numDeleted = 0

	// g.pool is deliberately not swapped. Its pooled scratch was sized for the
	// larger graph, which is safe — visitedList.reset re-slices down and its
	// generation counter only ever increases, so a stale stamp can never match
	// the current generation — and the extra capacity dies with the graph.
	return reclaimed
}
