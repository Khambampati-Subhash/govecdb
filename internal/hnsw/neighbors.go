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
func (g *Graph) selectNeighbors(st *searchState, cands []candidate, m int, dst []uint32) []uint32 {
	selected := dst[:0]
	if len(cands) <= m {
		for _, c := range cands {
			selected = append(selected, uint32(c.idx))
		}
		return selected
	}

	rejected := st.rejected[:0]

	for _, c := range cands {
		if len(selected) >= m {
			break
		}
		// A tie keeps the candidate: only a neighbor strictly closer to c than
		// base is makes the edge redundant (hnswlib and Faiss test the same way).
		// Rejecting on equality strands data. Far enough out, every float32
		// distance from an outlier to the data rounds to one value — at 1e9 per
		// component the ulp dwarfs the real differences — so each neighbor
		// tied it with a chosen one and dropped it, and its inbound edges went
		// to zero. Exact duplicates fail the same way: once a copy of base is
		// chosen, every other candidate is exactly as far from it as from
		// base, so all were rejected and backfill filled the list with more
		// copies — 50 copies each of 100 vectors left 4,967 of 5,000
		// unreachable. This rule brings that to 924; the rest is the general
		// cost of storing duplicates as separate nodes, not a tie.
		//
		// A copy of a chosen neighbor (d == 0) is still rejected: it is
		// reachable through that neighbor by construction, and keeping it is
		// what would let a pile of duplicates take every slot.
		keep := true
		for _, s := range selected {
			d := g.dist(g.nodes[c.idx].vector, g.nodes[s].vector)
			if d == 0 || g.alpha*d < c.dist {
				keep = false
				break
			}
		}
		if keep {
			selected = append(selected, uint32(c.idx))
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
		selected = append(selected, uint32(rejected[i].idx))
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
		cands = append(cands, candidate{int(nb), g.dist(g.nodes[nb].vector, self)})
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

	// For layer 0, nbrs is a view of the block itself, so the selection lands
	// in place and only the count is left to write.
	selected := g.selectNeighbors(st, cands, maxConn, nbrs)
	if lc == 0 {
		g.block0(idx)[0] = uint32(len(selected))
		return
	}
	g.nodes[idx].upper[lc-1] = selected
}

// connect adds `to` to `from`'s neighbor list on layer lc, skipping duplicates.
//
// On layer 0 the list is a block with room for maxConn+1 (layer0.go): insert
// appends one edge and immediately prunes back. An upper-layer list is given
// that same room the first time it is full, so it never reallocates again —
// plain append would double a full list on the first overflow, which is both
// the allocation and twice the memory the list will ever use.
func (g *Graph) connect(from, to, lc int) {
	if lc == 0 {
		b := g.block0(from)
		if slices.Contains(b[1:1+b[0]], uint32(to)) {
			return
		}
		b[1+b[0]] = uint32(to) // past the block when full: link prunes first
		b[0]++
		return
	}
	n := g.nodes[from]
	nbrs := n.upper[lc-1]
	if slices.Contains(nbrs, uint32(to)) {
		return
	}
	if len(nbrs) == cap(nbrs) {
		nbrs = slices.Grow(nbrs, max(1, g.maxConn(lc)+1-len(nbrs)))
	}
	n.upper[lc-1] = append(nbrs, uint32(to))
}

// readNeighbors returns node idx's neighbors on layer lc for a traversal to
// iterate. Outside a parallel batch that is the list itself; inside one, other
// workers rewrite lists in place, so it is a copy taken under the node's lock,
// valid until the next readNeighbors on st — and it leaves out nodes still
// being linked, so a worker only ever builds on finished ones.
func (g *Graph) readNeighbors(st *searchState, idx, lc int) []uint32 {
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

// neighborsAt returns node idx's neighbor slots on layer lc (nil if the node
// does not reach that layer). On layer 0 it is a view of the node's block, with
// the block's spare room as its capacity, so pruneConnections can select into
// it in place.
func (g *Graph) neighborsAt(idx, lc int) []uint32 {
	if lc == 0 {
		b := g.block0(idx)
		return b[1 : 1+b[0]]
	}
	n := g.nodes[idx]
	if lc > n.topLevel() {
		return nil
	}
	return n.upper[lc-1]
}

// maxConn is the neighbor cap for a layer: denser on layer 0.
func (g *Graph) maxConn(lc int) int {
	if lc == 0 {
		return g.mMax0
	}
	return g.mMax
}
