package hnsw

import (
	"cmp"
	"slices"
)

// selectNeighbors picks up to m edges for base out of cands (ascending by
// distance to base), using the alpha-relaxed diversity heuristic.
//
// Taking simply the m closest produces clustered edges that all point the same
// way, which strands the search in local minima. Instead we skip a candidate c
// when it already sits closer to a chosen neighbor s than it does to base —
// c is reachable via s, so that edge buys nothing. Alpha > 1 makes the test
// harder to pass, so it rejects *fewer* near candidates, and since they arrive
// nearest-first they take the slots long-range edges would have had — see
// Config.Alpha for the measurement that moved the default to 1.0.
// cands carry their distance to base already, so base itself is not needed.
//
// The selection is appended to dst[:0] and returned, so the caller decides where
// it lives: pruneConnections hands in the node's own neighbor array and the
// result overwrites it in place. dst must not alias cands.
func (g *Graph) selectNeighbors(st *searchState, cands []candidate, m int, dst []int) []int {
	selected := dst[:0]
	if len(cands) <= m {
		for _, c := range cands {
			selected = append(selected, c.idx)
		}
		return selected
	}

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
	//
	// Suspected once of undoing the heuristic on clustered data, since the
	// closest rejects there are cluster-mates. Measured, it is the opposite:
	// with backfill off, recall on the clustered corpus that motivated the
	// suspicion fell from 0.954 to 0.94. Alpha was the cause; this stays.
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
	//
	// The candidates live in pooled scratch, and the selection is written back
	// over nbrs itself — safe because cands holds a copy of every index. This
	// runs for up to M neighbors on every insert, so allocating either one here
	// was most of an insert's garbage: 206 allocs and 50 KB per op, measured.
	self := g.nodes[idx].vector
	cands := st.pruneBuf[:0]
	for _, nb := range nbrs {
		cands = append(cands, candidate{nb, g.dist(g.nodes[nb].vector, self)})
	}
	st.pruneBuf = cands

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
	//
	// slices.SortFunc rather than sort.Slice: the latter builds a reflection
	// swapper, which allocates on every call.
	slices.SortFunc(cands, func(a, b candidate) int {
		da, db := g.nodes[a.idx].deleted, g.nodes[b.idx].deleted
		if da != db {
			if da {
				return 1
			}
			return -1
		}
		return cmp.Compare(a.dist, b.dist)
	})

	g.nodes[idx].neighbors[lc] = g.selectNeighbors(st, cands, maxConn, nbrs)
}

// connect adds `to` to `from`'s neighbor list on layer lc, skipping duplicates.
//
// A list is given room for maxConn+1 the first time it is full: insert appends
// one edge and immediately prunes back to maxConn in place, so that one spare
// slot is all a list ever needs and it never reallocates again. Plain append
// would double a full 32-edge list to 64 on the first overflow, which is both
// the allocation and twice the memory the list will ever use.
func (g *Graph) connect(from, to, lc int) {
	n := g.nodes[from]
	nbrs := n.neighbors[lc]
	if slices.Contains(nbrs, to) {
		return
	}
	if len(nbrs) == cap(nbrs) {
		nbrs = slices.Grow(nbrs, max(1, g.maxConn(lc)+1-len(nbrs)))
	}
	n.neighbors[lc] = append(nbrs, to)
}

// readNeighbors returns node idx's neighbors on layer lc for a traversal to
// iterate. Outside a parallel batch that is the list itself; inside one, other
// workers rewrite lists in place, so it is a copy taken under the node's lock,
// valid until the next readNeighbors on st — and it leaves out nodes still
// being linked, so a worker only ever builds on finished ones.
func (g *Graph) readNeighbors(st *searchState, idx, lc int) []int {
	if st.locks == nil {
		return g.neighborsAt(idx, lc)
	}
	m := st.locks.of(idx)
	m.Lock()
	buf := st.nbrBuf[:0]
	for _, nb := range g.neighborsAt(idx, lc) {
		if !g.nodes[nb].linking.Load() {
			buf = append(buf, nb)
		}
	}
	m.Unlock()
	st.nbrBuf = buf
	return buf
}

// lockNode and unlockNode bracket a write to idx's neighbor lists. No-ops
// outside a parallel batch. A worker never holds two at once, which is the
// whole of the deadlock argument.
func (g *Graph) lockNode(st *searchState, idx int) {
	if st.locks != nil {
		st.locks.of(idx).Lock()
	}
}

func (g *Graph) unlockNode(st *searchState, idx int) {
	if st.locks != nil {
		st.locks.of(idx).Unlock()
	}
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
