package hnsw

// Result is a single search hit.
type Result struct {
	ID       string
	Distance float32
}

// Search returns the k nearest neighbors to query. ef controls accuracy and is
// clamped up to at least k.
//
// Safe to call concurrently, with itself and with Insert.
func (g *Graph) Search(query []float32, k, ef int) ([]Result, error) {
	if len(query) != g.cfg.Dimension {
		return nil, ErrDimensionMismatch
	}
	if k <= 0 {
		return nil, nil
	}
	if ef < k {
		ef = k
	}

	st := g.acquireState()
	defer g.releaseState(st)

	// Normalize into scratch rather than touching the caller's slice. This needs
	// no lock — the buffer is ours and g.normalized is fixed at construction —
	// so it stays outside the critical section.
	q := query
	if g.normalized {
		st.queryBuf = append(st.queryBuf[:0], query...)
		Normalize(st.queryBuf)
		q = st.queryBuf
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	if g.entry == -1 {
		return nil, nil
	}

	// Descend the upper layers greedily to reach the right region.
	cur := g.entry
	for lc := g.maxLevel; lc > 0; lc-- {
		cur = g.greedyClosest(cur, q, lc)
	}

	// Do the wide search on layer 0.
	w := g.searchLayer(st, q, cur, ef, 0)

	// w is sorted ascending by distance; take the top k.
	if len(w) > k {
		w = w[:k]
	}
	out := make([]Result, len(w))
	for i, c := range w {
		out[i] = Result{ID: g.nodes[c.idx].id, Distance: c.dist}
	}
	return out, nil
}

// greedyClosest walks layer lc from a starting node, always stepping to the
// neighbor closest to target, until no neighbor is closer. Returns that node.
func (g *Graph) greedyClosest(start int, target []float32, lc int) int {
	best := start
	bestDist := g.dist(g.nodes[start].vector, target)
	for {
		improved := false
		for _, nb := range g.neighborsAt(best, lc) {
			d := g.dist(g.nodes[nb].vector, target)
			if d < bestDist {
				bestDist, best, improved = d, nb, true
			}
		}
		if !improved {
			return best
		}
	}
}

// searchLayer runs the core best-first search on a single layer, returning up
// to ef closest nodes to query, sorted ascending by distance. Callers hold at
// least g.mu.RLock and own st.
func (g *Graph) searchLayer(st *searchState, query []float32, entryPoint, ef, lc int) []candidate {
	st.visited.reset(len(g.nodes))

	// Reuse the heap backing arrays; only the result slice is freshly allocated
	// because the caller keeps it.
	cands := st.cands[:0]
	results := st.results[:0]

	d := g.dist(g.nodes[entryPoint].vector, query)
	st.visited.visit(entryPoint)
	cands = append(cands, candidate{entryPoint, d})
	results = append(results, candidate{entryPoint, d})

	var c candidate
	for len(cands) > 0 {
		c, cands = minPop(cands)
		// If the closest frontier node is farther than our worst result, every
		// remaining candidate is worse too — stop.
		if len(results) >= ef && c.dist > results[0].dist {
			break
		}
		for _, nb := range g.neighborsAt(c.idx, lc) {
			if st.visited.visit(nb) {
				continue
			}
			nd := g.dist(g.nodes[nb].vector, query)
			if len(results) < ef || nd < results[0].dist {
				cands = minPush(cands, candidate{nb, nd})
				results = maxPush(results, candidate{nb, nd})
				if len(results) > ef {
					_, results = maxPop(results) // drop the farthest
				}
			}
		}
	}

	out := make([]candidate, len(results))
	for i := len(out) - 1; i >= 0; i-- {
		out[i], results = maxPop(results) // farthest first -> fill from end
	}

	// Hand the (possibly regrown) backing arrays back to the state so the next
	// traversal inherits the capacity.
	st.cands, st.results = cands, results
	return out // ascending by distance
}
