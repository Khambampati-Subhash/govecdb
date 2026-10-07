package hnsw

import "sync/atomic"

// node is a single vector living in the graph.
//
// A node exists on layers 0..topLevel. neighbors[l] holds the indexes (into
// Graph.nodes) of this node's connections on layer l. Layer 0 contains every
// node; higher layers are progressively sparser, forming the "express lanes"
// that make search fast.
type node struct {
	id     string
	vector []float32

	// neighbors[l] = neighbor indexes on layer l. len(neighbors) == topLevel+1.
	neighbors [][]int

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
	return len(n.neighbors) - 1
}

// newNode allocates a node that lives on layers 0..level.
func newNode(id string, vector []float32, level int) *node {
	return &node{
		id:        id,
		vector:    vector,
		neighbors: make([][]int, level+1),
	}
}
