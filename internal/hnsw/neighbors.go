package hnsw

import (
	"slices"
	"sort"
)

// selectNeighbors picks up to m edges for base out of cands (ascending by
// distance to base), using the alpha-relaxed diversity heuristic.
//
// Taking simply the m closest produces clustered edges that all point the same
// way, which strands the search in local minima. Instead we skip a candidate c
// when it already sits closer to a chosen neighbor s than it does to base —
// c is reachable via s, so that edge buys nothing. Alpha > 1 tightens the test,
// preserving more long-range links and making the graph easier to navigate.
// cands carry their distance to base already, so base itself is not needed.
func (g *Graph) selectNeighbors(st *searchState, cands []candidate, m int) []int {
	if len(cands) <= m {
		out := make([]int, len(cands))
		for i, c := range cands {
			out[i] = c.idx
		}
		return out
	}

	selected := make([]int, 0, m)
	rejected := st.rejected[:0]

	for _, c := range cands {
		if len(selected) >= m {
			break
		}
		keep := true
		for _, s := range selected {
			if g.alpha*g.dist(g.nodes[c.idx].vector, g.nodes[s].vector) <= c.dist {
				keep = false
				break
			}
		}
		if keep {
			selected = append(selected, c.idx)
		} else {
			rejected = append(rejected, c)
		}
	}

	// Backfill with the closest rejects rather than returning a thin list: a
	// node with too few edges is a dead end during search.
	for i := 0; len(selected) < m && i < len(rejected); i++ {
		selected = append(selected, rejected[i].idx)
	}

	st.rejected = rejected
	return selected
}

// pruneConnections trims a node's neighbor list on layer lc back to the layer
// cap, applying the same diversity heuristic used when inserting.
func (g *Graph) pruneConnections(st *searchState, idx, lc int) {
	maxConn := g.maxConn(lc)
	nbrs := g.neighborsAt(idx, lc)
	if len(nbrs) <= maxConn {
		return
	}

	// Compute each distance exactly once. Doing this inside a sort comparator
	// instead would recompute them O(n log n) times.
	self := g.nodes[idx].vector
	cands := make([]candidate, len(nbrs))
	for i, nb := range nbrs {
		cands[i] = candidate{nb, g.dist(g.nodes[nb].vector, self)}
	}

	// Live neighbors outrank tombstones, and only then does distance decide.
	//
	// This is the one place tombstones compete with live nodes for a scarce
	// resource — edge slots — and ranking them purely by distance loses data.
	// Insert connects nb->idx and immediately prunes nb; if a dead node wins
	// that contest, the brand-new live node loses its only inbound edge and
	// becomes unreachable forever. Nothing else can rescue it: a node's own
	// outgoing edges never help anyone find it.
	//
	// Demoting rather than dropping is what keeps this safe. selectNeighbors
	// backfills to the cap regardless, so a node with few live candidates still
	// keeps its tombstone edges and the bridges they provide. Tombstones only
	// lose slots where live alternatives actually exist.
	sort.Slice(cands, func(i, j int) bool {
		di, dj := g.nodes[cands[i].idx].deleted, g.nodes[cands[j].idx].deleted
		if di != dj {
			return !di
		}
		return cands[i].dist < cands[j].dist
	})

	g.nodes[idx].neighbors[lc] = g.selectNeighbors(st, cands, maxConn)
}

// connect adds `to` to `from`'s neighbor list on layer lc, skipping duplicates.
func (g *Graph) connect(from, to, lc int) {
	n := g.nodes[from]
	if slices.Contains(n.neighbors[lc], to) {
		return
	}
	n.neighbors[lc] = append(n.neighbors[lc], to)
}

// neighborsAt returns node idx's neighbor slice on layer lc (nil if the node
// does not reach that layer).
func (g *Graph) neighborsAt(idx, lc int) []int {
	n := g.nodes[idx]
	if lc > n.topLevel() {
		return nil
	}
	return n.neighbors[lc]
}

// maxConn is the neighbor cap for a layer: denser on layer 0.
func (g *Graph) maxConn(lc int) int {
	if lc == 0 {
		return g.mMax0
	}
	return g.mMax
}
