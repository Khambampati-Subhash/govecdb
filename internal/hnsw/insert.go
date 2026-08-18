package hnsw

// Insert adds (or is a no-op for a duplicate id of) a vector into the graph.
func (g *Graph) Insert(id string, vector []float32) error {
	if len(vector) == 0 {
		return ErrEmptyVector
	}
	if len(vector) != g.cfg.Dimension {
		return ErrDimensionMismatch
	}
	if _, exists := g.ids[id]; exists {
		return nil // v1: ignore duplicates; update semantics come later
	}

	vec := g.prepare(vector)
	level := g.randomLevel()
	n := newNode(id, vec, level)
	idx := len(g.nodes)
	g.nodes = append(g.nodes, n)
	g.ids[id] = idx

	// First node ever: it becomes the entry point and we're done.
	if g.entry == -1 {
		g.entry = idx
		g.maxLevel = level
		return nil
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
		w := g.searchLayer(vec, cur, g.cfg.EfConstruction, lc)
		neighbors := g.selectNeighbors(w, g.maxConn(lc))
		for _, nb := range neighbors {
			g.connect(idx, nb, lc)
			g.connect(nb, idx, lc)
			g.pruneConnections(nb, lc)
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
	return nil
}
