package hnsw

// Result is a single search hit.
type Result struct {
	ID       string
	Distance float32
}

// Search returns the k nearest neighbors to query. ef controls accuracy and is
// clamped up to at least k.
func (g *Graph) Search(query []float32, k, ef int) ([]Result, error) {
	if len(query) != g.cfg.Dimension {
		return nil, ErrDimensionMismatch
	}
	if g.entry == -1 || k <= 0 {
		return nil, nil
	}
	if ef < k {
		ef = k
	}

	// Normalize into a reusable buffer rather than touching the caller's slice.
	q := query
	if g.normalized {
		g.queryBuf = append(g.queryBuf[:0], query...)
		Normalize(g.queryBuf)
		q = g.queryBuf
	}

	// Descend the upper layers greedily to reach the right region.
	cur := g.entry
	for lc := g.maxLevel; lc > 0; lc-- {
		cur = g.greedyClosest(cur, q, lc)
	}

	// Do the wide search on layer 0.
	w := g.searchLayer(q, cur, ef, 0)

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
// to ef closest nodes to query, sorted ascending by distance.
func (g *Graph) searchLayer(query []float32, entryPoint, ef, lc int) []candidate {
	g.visited.reset(len(g.nodes))

	// Reuse the heap backing arrays; only the result slice is freshly allocated
	// because the caller keeps it.
	cands := g.scratchCands[:0]
	results := g.scratchRes[:0]

	d := g.dist(g.nodes[entryPoint].vector, query)
	g.visited.visit(entryPoint)
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
			if g.visited.visit(nb) {
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

	g.scratchCands, g.scratchRes = cands, results
	return out // ascending by distance
}
