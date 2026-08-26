package hnsw

import "slices"

// Insert stores vector under id. If the id is already in the graph, the new
// vector replaces the old one — Insert is an upsert, and one operation covers
// both create and replace.
//
// That is a narrowing of the operation set, not a convenience. The WAL record
// format freezes around whatever operations exist, and a separate UPDATE record
// would have to mean "insert if absent" anyway to survive replay against a
// snapshot that may or may not already contain the id. Two record types for one
// state transition, differing only in what they assume about the past. With
// upsert there is a single PUT whose meaning does not depend on history.
//
// # A replacement builds a new slot rather than editing the old one
//
// Overwriting the vector in place would keep the slot's index, and with it every
// neighbor list in the graph that references that index. Those edges were chosen
// for the *old* vector: they encode "these two are close", a claim the new
// vector does not make. Searches would keep being routed into this slot from a
// region it no longer belongs to, and never from the region it does.
//
// Repairing them is not an option either. Pruning makes edges asymmetric, so a
// node's own neighbor list is not the list of nodes pointing at it; finding
// every inbound edge means scanning the whole graph, O(N·M) per update.
//
// So a replacement tombstones the old slot and builds a fresh one, whose edges
// are correct by construction. The price is one dead slot per update — the same
// debt Delete takes on, paid off by the same compaction pass. A workload that
// re-embeds the same ids repeatedly is therefore the one that makes compaction
// load-bearing rather than housekeeping.
//
// Safe to call concurrently, but writers serialize against each other and
// exclude searches for the duration.
func (g *Graph) Insert(id string, vector []float32) error {
	if len(vector) == 0 {
		return ErrEmptyVector
	}
	if len(vector) != g.cfg.Dimension {
		return ErrDimensionMismatch
	}

	// Copy and normalize before taking the lock: it only touches the caller's
	// slice and immutable config, so it need not block searches.
	vec := g.prepare(vector)

	g.mu.Lock()
	defer g.mu.Unlock()

	if prev, exists := g.ids[id]; exists {
		// Re-applying a record the graph already holds is the normal shape of
		// WAL replay across a snapshot boundary, and it must not cost a slot:
		// otherwise every recovery would inflate the graph with tombstones for
		// vectors that never changed. The comparison is against the *stored*
		// form, so for Cosine a rescaled vector is recognized as unchanged —
		// the graph only ever stored its direction.
		if slices.Equal(g.nodes[prev].vector, vec) {
			return nil
		}
		g.tombstone(id)
	}

	g.insertPrepared(id, vec)
	return nil
}

// insertPrepared adds a node for an id that is not currently bound, taking
// ownership of vec — which must already be in stored form: the graph's own
// copy, normalized if the metric wants it.
//
// It is split out for Compact, which rebuilds the graph from vectors that are
// *already* stored form. Sending those back through prepare would copy every
// vector for no reason, and re-normalizing an already-unit vector drifts it by
// an ulp, so a compacted graph would no longer hold quite the same numbers as
// the one it replaced.
//
// Callers hold the write lock.
func (g *Graph) insertPrepared(id string, vec []float32) {
	st := g.acquireState()
	defer g.releaseState(st)

	level := g.randomLevel()
	n := newNode(id, vec, level)
	idx := len(g.nodes)
	g.nodes = append(g.nodes, n)
	g.ids[id] = idx

	// First node ever: it becomes the entry point and we're done.
	if g.entry == -1 {
		g.entry = idx
		g.maxLevel = level
		return
	}

	// Phase 1: greedily descend from the top down to level+1 with ef=1, just to
	// find a good entry point near the new node.
	cur := g.entry
	for lc := g.maxLevel; lc > level; lc-- {
		cur = g.greedyClosest(cur, vec, lc)
	}

	// Phase 2: from min(maxLevel, level) down to 0, find neighbors and connect.
	start := min(level, g.maxLevel)
	for lc := start; lc >= 0; lc-- {
		w := g.searchLayer(st, vec, cur, g.cfg.EfConstruction, lc)
		neighbors := g.selectNeighbors(st, w, g.maxConn(lc))

		// searchLayer yields live nodes only, so inserting into a region whose
		// every member is tombstoned returns nothing to attach to — and a node
		// with no edges is unreachable forever, which is data loss, not just
		// poor recall. Fall back to the node we searched from: edges are
		// bidirectional, and a dead neighbor still routes, so linking to a
		// tombstone beats isolation.
		if len(neighbors) == 0 {
			neighbors = append(neighbors, cur)
		}

		for _, nb := range neighbors {
			g.connect(idx, nb, lc)
			g.connect(nb, idx, lc)
			g.pruneConnections(st, nb, lc)
		}
		if len(w) > 0 {
			cur = w[0].idx // closest, to seed the next lower layer
		}
	}

	// If the new node reaches higher than any existing node, it becomes entry.
	if level > g.maxLevel {
		g.maxLevel = level
		g.entry = idx
	}
}
