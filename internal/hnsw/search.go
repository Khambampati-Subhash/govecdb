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
	return g.SearchFilter(query, k, ef, nil)
}

// SearchFilter is Search restricted to the ids allow accepts. A nil allow means
// no restriction and is exactly Search.
//
// # The predicate is over ids, not metadata
//
// This package knows nothing about metadata and should keep it that way — the
// index stores vectors, and giving it a second data model to consult would make
// every future index implementation responsible for one too. An opaque
// `func(id string) bool` lets the layer that *does* own metadata close over it,
// and costs this package one parameter.
//
// # It is applied during the traversal, not to the results
//
// Filtering the returned slice instead would be simpler and wrong for the same
// reason it is wrong for tombstones: a selective filter would leave far fewer
// than k hits rather than making the search look wider for k matching ones. So
// allow gates entry to the result set only — rejected nodes still ride the
// frontier, because a node that does not match is very often the bridge to one
// that does.
//
// The honest cost is the same one tombstones carry. A filter that matches little
// keeps the result set under-filled, which loosens the pruning bound and makes
// the search explore wider — self-correcting, but at low selectivity it
// approaches a full scan, and a scan over the metadata is the better tool there.
//
// allow is called at most once per node per search, under the read lock, and
// must not call back into the graph.
func (g *Graph) SearchFilter(query []float32, k, ef int, allow func(id string) bool) ([]Result, error) {
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

	// Do the wide search on layer 0. The descent above stays unfiltered: it is
	// pure routing, and the node it lands on is a seed rather than an answer.
	w := g.searchLayer(st, q, cur, ef, 0, allow)

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

// admits reports whether a node may become an answer: live, and accepted by the
// caller's predicate when there is one.
//
// The two conditions are one function because they are the same rule — "this
// node routes but does not answer" — and keeping them together is what stops a
// future edit from remembering the tombstone half and forgetting the filter half.
func (g *Graph) admits(idx int, allow func(id string) bool) bool {
	if g.nodes[idx].deleted {
		return false
	}
	return allow == nil || allow(g.nodes[idx].id)
}

// greedyClosest walks layer lc from a starting node, always stepping to the
// neighbor closest to target, until no neighbor is closer. Returns that node.
//
// It is deliberately blind to tombstones: this is pure routing on the upper
// layers, and the node it lands on is a seed for the next layer down, never an
// answer. Skipping dead nodes here would only make the descent worse.
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
// to ef closest ADMISSIBLE nodes to query, sorted ascending by distance. Callers
// hold at least g.mu.RLock and own st.
//
// Two things split the structures this runs on, and that split is the whole of
// both the delete design and the filter design:
//
//	cands   (the frontier) admits everything — a rejected node is still a bridge
//	results                admits only nodes that pass admits()
//
// Filtering at the end instead would be simpler and wrong: it would return
// fewer than ef hits as tombstones accumulate or a filter narrows, rather than
// searching wider to find ef admissible ones.
//
// allow may be nil, which is what Insert passes: a graph's shape must never
// depend on a query's filter.
func (g *Graph) searchLayer(st *searchState, query []float32, entryPoint, ef, lc int, allow func(id string) bool) []candidate {
	st.visited.reset(len(g.nodes))

	// Reuse the heap backing arrays; only the result slice is freshly allocated
	// because the caller keeps it.
	cands := st.cands[:0]
	results := st.results[:0]

	d := g.dist(g.nodes[entryPoint].vector, query)
	st.visited.visit(entryPoint)
	cands = append(cands, candidate{entryPoint, d})
	if g.admits(entryPoint, allow) {
		results = append(results, candidate{entryPoint, d})
	}

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

				// A tombstone — or a node the filter rejects — rides the
				// frontier but never becomes an answer. With many of them
				// `results` fills slowly, which loosens the pruning bound above
				// and makes the search explore wider: the honest, self-correcting
				// cost of deferred deletion and of selective filtering alike, and
				// what compaction buys back for the first of the two.
				//
				// admits is evaluated here rather than beside the visited check
				// so it runs only for nodes that clear the distance bound. The
				// predicate reaches outside this package, so it is the expensive
				// half of the loop, and the frontier does not need its answer.
				if g.admits(nb, allow) {
					results = maxPush(results, candidate{nb, nd})
					if len(results) > ef {
						_, results = maxPop(results) // drop the farthest
					}
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
