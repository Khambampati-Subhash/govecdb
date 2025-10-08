package index

import (
	"fmt"
	"sync"
)

// HNSWNode represents a node in the HNSW graph
type HNSWNode struct {
	// Vector data stored in this node
	Vector *Vector

	// Level of this node (0 is the base layer)
	Level int

	// Connections to other nodes at each layer
	// connections[layer] contains the neighbors at that layer
	connections []map[string]*HNSWNode

	// Mutex for thread-safe operations on this node
	mu sync.RWMutex

	// Mark for deletion (soft delete)
	deleted bool
}

// NewHNSWNode creates a new HNSW node
func NewHNSWNode(vector *Vector, level int) *HNSWNode {
	node := &HNSWNode{
		Vector:      vector,
		Level:       level,
		connections: make([]map[string]*HNSWNode, level+1),
		deleted:     false,
	}

	// Initialize connection maps for each layer
	for i := 0; i <= level; i++ {
		node.connections[i] = make(map[string]*HNSWNode)
	}

	return node
}

// GetConnections returns the connections at the specified layer (thread-safe)
func (n *HNSWNode) GetConnections(layer int) map[string]*HNSWNode {
	n.mu.RLock()
	defer n.mu.RUnlock()

	if layer < 0 || layer >= len(n.connections) {
		return make(map[string]*HNSWNode)
	}

	// Return a copy to prevent race conditions
	connections := make(map[string]*HNSWNode, len(n.connections[layer]))
	for id, node := range n.connections[layer] {
		connections[id] = node
	}

	return connections
}

// GetConnectionsList returns connections as a slice (thread-safe)
func (n *HNSWNode) GetConnectionsList(layer int) []*HNSWNode {
	connections := n.GetConnections(layer)
	nodes := make([]*HNSWNode, 0, len(connections))

	for _, node := range connections {
		if !node.IsDeleted() {
			nodes = append(nodes, node)
		}
	}

	return nodes
}

// AddConnection adds a bidirectional connection between nodes at the specified layer
func (n *HNSWNode) AddConnection(other *HNSWNode, layer int) {
	if n == other || layer < 0 {
		return
	}

	// Ensure both nodes have enough layers
	if layer >= len(n.connections) || layer >= len(other.connections) {
		return
	}

	// Add connection from n to other
	n.mu.Lock()
	if !n.deleted && layer < len(n.connections) {
		n.connections[layer][other.Vector.ID] = other
	}
	n.mu.Unlock()

	// Add connection from other to n
	other.mu.Lock()
	if !other.deleted && layer < len(other.connections) {
		other.connections[layer][n.Vector.ID] = n
	}
	other.mu.Unlock()
}

// AddConnectionUnsafe adds a unidirectional connection without locking (for batch operations)
func (n *HNSWNode) AddConnectionUnsafe(other *HNSWNode, layer int) {
	if n == other || layer < 0 || layer >= len(n.connections) {
		return
	}
	n.connections[layer][other.Vector.ID] = other
}

// RemoveConnection removes a bidirectional connection between nodes at the specified layer
func (n *HNSWNode) RemoveConnection(other *HNSWNode, layer int) {
	if n == other || layer < 0 {
		return
	}

	// Remove connection from n to other
	n.mu.Lock()
	if layer < len(n.connections) {
		delete(n.connections[layer], other.Vector.ID)
	}
	n.mu.Unlock()

	// Remove connection from other to n
	other.mu.Lock()
	if layer < len(other.connections) {
		delete(other.connections[layer], n.Vector.ID)
	}
	other.mu.Unlock()
}

// HasConnection checks if there's a connection to another node at the specified layer
func (n *HNSWNode) HasConnection(other *HNSWNode, layer int) bool {
	n.mu.RLock()
	defer n.mu.RUnlock()

	if layer < 0 || layer >= len(n.connections) {
		return false
	}

	_, exists := n.connections[layer][other.Vector.ID]
	return exists
}

// ConnectionCount returns the number of connections at the specified layer
func (n *HNSWNode) ConnectionCount(layer int) int {
	n.mu.RLock()
	defer n.mu.RUnlock()

	if layer < 0 || layer >= len(n.connections) {
		return 0
	}

	count := 0
	for _, node := range n.connections[layer] {
		if !node.IsDeleted() {
			count++
		}
	}

	return count
}

// TotalConnections returns the total number of connections across all layers
func (n *HNSWNode) TotalConnections() int {
	n.mu.RLock()
	defer n.mu.RUnlock()

	total := 0
	for layer := 0; layer < len(n.connections); layer++ {
		for _, node := range n.connections[layer] {
			if !node.IsDeleted() {
				total++
			}
		}
	}

	return total
}

// MarkDeleted marks the node as deleted (soft delete)
func (n *HNSWNode) MarkDeleted() {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.deleted = true
}

// IsDeleted returns whether the node is marked as deleted
func (n *HNSWNode) IsDeleted() bool {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.deleted
}

// CleanupDeletedConnections removes connections to deleted nodes
func (n *HNSWNode) CleanupDeletedConnections() {
	n.mu.Lock()
	defer n.mu.Unlock()

	for layer := 0; layer < len(n.connections); layer++ {
		for id, node := range n.connections[layer] {
			if node.IsDeleted() {
				delete(n.connections[layer], id)
			}
		}
	}
}

// GetVector returns the vector data (thread-safe)
func (n *HNSWNode) GetVector() *Vector {
	n.mu.RLock()
	defer n.mu.RUnlock()

	if n.deleted {
		return nil
	}

	// Return a copy to prevent modifications
	vectorCopy := &Vector{
		ID:       n.Vector.ID,
		Data:     make([]float32, len(n.Vector.Data)),
		Metadata: make(map[string]interface{}),
	}

	copy(vectorCopy.Data, n.Vector.Data)
	for k, v := range n.Vector.Metadata {
		vectorCopy.Metadata[k] = v
	}

	return vectorCopy
}

// UpdateVector updates the vector data (thread-safe)
func (n *HNSWNode) UpdateVector(vector *Vector) {
	n.mu.Lock()
	defer n.mu.Unlock()

	if !n.deleted && vector != nil {
		n.Vector = vector
	}
}

// String returns a string representation of the node
func (n *HNSWNode) String() string {
	n.mu.RLock()
	defer n.mu.RUnlock()

	if n.deleted {
		return "HNSWNode{deleted}"
	}

	totalConns := 0
	for layer := 0; layer < len(n.connections); layer++ {
		totalConns += len(n.connections[layer])
	}

	return fmt.Sprintf("HNSWNode{ID:%s, Level:%d, Connections:%d, Deleted:%v}",
		n.Vector.ID, n.Level, totalConns, n.deleted)
}

// NodeCandidate represents a node with its distance for priority queue operations
type NodeCandidate struct {
	Node     *HNSWNode
	Distance float32
}

// NodeCandidateHeap implements a min-heap for NodeCandidate
type NodeCandidateHeap []*NodeCandidate

func (h NodeCandidateHeap) Len() int           { return len(h) }
func (h NodeCandidateHeap) Less(i, j int) bool { return h[i].Distance < h[j].Distance }
func (h NodeCandidateHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *NodeCandidateHeap) Push(x interface{}) {
	*h = append(*h, x.(*NodeCandidate))
}

func (h *NodeCandidateHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// MaxNodeCandidateHeap implements a max-heap for NodeCandidate
type MaxNodeCandidateHeap []*NodeCandidate

func (h MaxNodeCandidateHeap) Len() int           { return len(h) }
func (h MaxNodeCandidateHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance }
func (h MaxNodeCandidateHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MaxNodeCandidateHeap) Push(x interface{}) {
	*h = append(*h, x.(*NodeCandidate))
}

func (h *MaxNodeCandidateHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
