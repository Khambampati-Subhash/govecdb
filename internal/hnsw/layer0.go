package hnsw

// Layer 0 is where a search spends nearly all of its time, and every node has
// a list there, so its neighbor lists live in the graph rather than in the
// nodes: fixed-size blocks of uint32, one per slot, packed into chunks.
//
// # Why
//
// A search waits on memory, not arithmetic (see touchLines). Reaching a node's
// layer-0 list used to be three dependent loads — the *node, its neighbors
// slice header, the layer-0 slice header — before the first id arrived, and
// each id was 8 bytes in a list allocated on its own. A block is one load from
// a chunk header that stays in cache, its ids are 4 bytes, and its count sits
// beside them in the same line.
//
// # The block
//
// A slot's block is stride0 = mMax0+2 uint32s: the count, then room for
// mMax0+1 ids — the cap, plus the one edge an insert adds before pruning
// straight back. A list never needs more: link prunes before connecting into a
// full one, and connect would index past the block rather than grow it.
//
// # Why chunks
//
// One flat array would double, and each doubling would copy every list in the
// graph under the write lock — a stall in proportion to the whole index, and
// twice its layer-0 memory at the moment of the copy. Chunks are allocated as
// slots arrive and never move once full; growing the last one moves at most a
// chunk. Slots are only ever added under the write lock (place), so neither a
// search nor a parallel batch's linkers see the chunk list change.
const layer0Chunk = 1024

// block0 is slot idx's layer-0 block: count first, then the ids. Callers hold
// at least the read lock.
func (g *Graph) block0(idx int) []uint32 {
	c := g.layer0[idx/layer0Chunk]
	off := (idx % layer0Chunk) * g.stride0
	return c[off : off+g.stride0 : off+g.stride0]
}

// addSlot0 gives slot idx — the next one, len(g.nodes) — an empty block.
// Callers hold the write lock.
func (g *Graph) addSlot0(idx int) {
	if idx%layer0Chunk == 0 {
		g.layer0 = append(g.layer0, nil)
	}
	last := len(g.layer0) - 1
	c := g.layer0[last]
	n := len(c)
	if cap(c) < n+g.stride0 {
		// Grow toward a full chunk, never past it, so a full chunk is exactly
		// its size and a small graph is not handed a chunk's worth up front.
		grown := make([]uint32, n, min(max(2*cap(c), g.stride0), layer0Chunk*g.stride0))
		copy(grown, c)
		c = grown
	}
	c = c[:n+g.stride0]
	clear(c[n:])
	g.layer0[last] = c
}
