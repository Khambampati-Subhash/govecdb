package hnsw

import "sync/atomic"

// node is a single vector living in the graph.
//
// A node exists on layers 0..topLevel. Layer 0 contains every node; higher
// layers are progressively sparser, forming the "express lanes" that make
// search fast. Its layer-0 neighbors live in the graph's flat store
// (layer0.go), where a search reads them; the sparse upper layers stay here.
type node struct {
	id     string
	vector []float32

	// upper[l-1] = neighbor slots on layer l, for l in 1..topLevel.
	upper [][]uint32

	// deleted marks a tombstone. The slot keeps its index and all of its edges,
	// so searches still route *through* it; it simply stops being an answer.
	// See Delete for why the alternative — actually removing the slot — is not
	// on the table.
	deleted bool

	// linking is set while a parallel batch is still building this node's
	// edges, and parallel traversals step around it (readNeighbors). A node
	// can be found on an upper layer before its worker has reached layer 0,
	// and a search seeded from it then sees an empty list — the newcomer links
	// to one or two nodes and the graph grows an island. Never set outside a
	// batch, and never read by a search.
	linking atomic.Bool
}

// topLevel is the highest layer this node participates in.
func (n *node) topLevel() int {
	return len(n.upper)
}

// newNode allocates a node that lives on layers 0..level.
func newNode(id string, vector []float32, level int) *node {
	return &node{
		id:     id,
		vector: vector,
		upper:  make([][]uint32, level),
	}
}
