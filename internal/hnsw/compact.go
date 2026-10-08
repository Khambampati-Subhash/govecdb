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
// # Searches keep running; writers wait
//
// The replacement is built under the *read* lock, and the write lock is taken
// only to swap it in. Searches run against the old graph for the whole build —
// seconds for a large one — where they used to stop for all of it. Writers are
// held off by the read lock, so the old graph cannot change underneath the
// build.
//
// It can change in one place: between releasing the read lock and taking the
// write lock, where a writer that was queued gets in first. writes counts every
// slot placed or tombstoned, and if it moved, the replacement is stale and is
// rebuilt under the write lock — the old stop-the-world path, kept for
// correctness rather than speed. A caller that excludes its own writers for
// the duration, as the database does, never takes it.
//
// So the index does not decide *when*. It exposes Stats().DeadRatio() and lets
// the caller pick a moment that tolerates the cost.
func (g *Graph) Compact() int {
	g.mu.RLock()
	// Cheap enough to call on a timer: a clean graph is not worth rebuilding,
	// and rebuilding it would throw away a perfectly good index for nothing.
	if g.numDeleted == 0 {
		g.mu.RUnlock()
		return 0
	}
	fresh, seen := g.rebuild(), g.writes
	g.mu.RUnlock()

	g.mu.Lock()
	defer g.mu.Unlock()
	if g.writes != seen {
		if g.numDeleted == 0 {
			return 0 // another Compact got there first
		}
		fresh = g.rebuild()
	}

	reclaimed := g.numDeleted
	g.nodes, g.ids = fresh.nodes, fresh.ids
	g.entry, g.maxLevel = fresh.entry, fresh.maxLevel
	g.numDeleted = 0
	// The swap is a write too: a Compact that built from the old graph and is
	// waiting for the lock must see its replacement is now stale.
	g.writes++

	// g.pool is deliberately not swapped. Its pooled scratch was sized for the
	// larger graph, which is safe — visitedList.reset re-slices down and its
	// generation counter only ever increases, so a stale stamp can never match
	// the current generation — and the extra capacity dies with the graph.
	return reclaimed
}

// rebuild builds a new graph over the live vectors, in slot order. It only
// reads g, so callers hold either lock.
func (g *Graph) rebuild() *Graph {
	fresh := newGraph(g.cfg)
	for _, n := range g.nodes {
		if n.deleted {
			continue
		}
		// The vector moves across as-is rather than being re-prepared: it is
		// already this graph's own normalized copy, and the graph it came from
		// is about to be garbage. Dead vectors are the memory being reclaimed —
		// nothing references them once the swap lands.
		fresh.insertPrepared(n.id, n.vector)
	}
	return fresh
}
