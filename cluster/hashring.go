// Package cluster provides hash ring implementation for consistent hashing.
package cluster

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sort"
	"sync"
)

// HashRingImpl implements the HashRing interface for consistent hashing
type HashRingImpl struct {
	mu           sync.RWMutex
	nodes        map[string]bool     // Active nodes
	virtualNodes int                 // Virtual nodes per physical node
	ring         map[uint64]string   // Hash -> Node mapping
	sortedHashes []uint64            // Sorted hash values for binary search
	hashFunc     func([]byte) uint64 // Hash function
}

// NewHashRing creates a new hash ring with the specified virtual nodes per physical node
func NewHashRing(virtualNodes int) *HashRingImpl {
	return &HashRingImpl{
		nodes:        make(map[string]bool),
		virtualNodes: virtualNodes,
		ring:         make(map[uint64]string),
		sortedHashes: make([]uint64, 0),
		hashFunc:     defaultHashFunc,
	}
}

// AddNode adds a node to the hash ring
func (hr *HashRingImpl) AddNode(node string) {
	hr.mu.Lock()
	defer hr.mu.Unlock()

	if hr.nodes[node] {
		return // Node already exists
	}

	hr.nodes[node] = true

	// Add virtual nodes
	for i := 0; i < hr.virtualNodes; i++ {
		virtualNodeKey := fmt.Sprintf("%s-%d", node, i)
		hash := hr.hashFunc([]byte(virtualNodeKey))
		hr.ring[hash] = node
		hr.sortedHashes = append(hr.sortedHashes, hash)
	}

	// Keep hashes sorted
	sort.Slice(hr.sortedHashes, func(i, j int) bool {
		return hr.sortedHashes[i] < hr.sortedHashes[j]
	})
}

// RemoveNode removes a node from the hash ring
func (hr *HashRingImpl) RemoveNode(node string) {
	hr.mu.Lock()
	defer hr.mu.Unlock()

	if !hr.nodes[node] {
		return // Node doesn't exist
	}

	delete(hr.nodes, node)

	// Remove virtual nodes
	newSortedHashes := make([]uint64, 0, len(hr.sortedHashes)-hr.virtualNodes)
	for _, hash := range hr.sortedHashes {
		if hr.ring[hash] != node {
			newSortedHashes = append(newSortedHashes, hash)
		} else {
			delete(hr.ring, hash)
		}
	}

	hr.sortedHashes = newSortedHashes
}

// GetNode returns the node responsible for the given key
func (hr *HashRingImpl) GetNode(key string) (string, error) {
	hr.mu.RLock()
	defer hr.mu.RUnlock()

	if len(hr.nodes) == 0 {
		return "", fmt.Errorf("no nodes available in hash ring")
	}

	hash := hr.hashFunc([]byte(key))

	// Find the first node with hash >= key hash (clockwise)
	idx := sort.Search(len(hr.sortedHashes), func(i int) bool {
		return hr.sortedHashes[i] >= hash
	})

	// Wrap around if we've gone past the end
	if idx == len(hr.sortedHashes) {
		idx = 0
	}

	return hr.ring[hr.sortedHashes[idx]], nil
}

// GetNodes returns the N nodes responsible for the given key (for replication)
func (hr *HashRingImpl) GetNodes(key string, n int) ([]string, error) {
	hr.mu.RLock()
	defer hr.mu.RUnlock()

	if len(hr.nodes) == 0 {
		return nil, fmt.Errorf("no nodes available in hash ring")
	}

	if n > len(hr.nodes) {
		n = len(hr.nodes)
	}

	hash := hr.hashFunc([]byte(key))

	// Find the first node with hash >= key hash
	idx := sort.Search(len(hr.sortedHashes), func(i int) bool {
		return hr.sortedHashes[i] >= hash
	})

	// Wrap around if we've gone past the end
	if idx == len(hr.sortedHashes) {
		idx = 0
	}

	seen := make(map[string]bool)
	result := make([]string, 0, n)

	// Collect unique nodes starting from the found position
	for len(result) < n && len(seen) < len(hr.nodes) {
		node := hr.ring[hr.sortedHashes[idx]]
		if !seen[node] {
			result = append(result, node)
			seen[node] = true
		}
		idx = (idx + 1) % len(hr.sortedHashes)
	}

	return result, nil
}

// GetAllNodes returns all nodes in the hash ring
func (hr *HashRingImpl) GetAllNodes() []string {
	hr.mu.RLock()
	defer hr.mu.RUnlock()

	nodes := make([]string, 0, len(hr.nodes))
	for node := range hr.nodes {
		nodes = append(nodes, node)
	}

	return nodes
}

// NodeCount returns the number of physical nodes in the ring
func (hr *HashRingImpl) NodeCount() int {
	hr.mu.RLock()
	defer hr.mu.RUnlock()
	return len(hr.nodes)
}

// VirtualNodeCount returns the total number of virtual nodes in the ring
func (hr *HashRingImpl) VirtualNodeCount() int {
	hr.mu.RLock()
	defer hr.mu.RUnlock()
	return len(hr.ring)
}

// GetDistribution returns the hash ranges for each node
func (hr *HashRingImpl) GetDistribution() map[string][]HashRange {
	hr.mu.RLock()
	defer hr.mu.RUnlock()

	distribution := make(map[string][]HashRange)

	if len(hr.sortedHashes) == 0 {
		return distribution
	}

	// Calculate ranges for each virtual node
	for i, hash := range hr.sortedHashes {
		node := hr.ring[hash]

		var start uint64
		if i == 0 {
			// First hash range wraps around from the last hash
			start = hr.sortedHashes[len(hr.sortedHashes)-1] + 1
		} else {
			start = hr.sortedHashes[i-1] + 1
		}

		hashRange := HashRange{
			Start: start,
			End:   hash + 1,
		}

		distribution[node] = append(distribution[node], hashRange)
	}

	return distribution
}

// Rebalance redistributes the hash ring with new set of nodes
func (hr *HashRingImpl) Rebalance(nodes []string) {
	hr.mu.Lock()
	defer hr.mu.Unlock()

	// Clear existing ring
	hr.nodes = make(map[string]bool)
	hr.ring = make(map[uint64]string)
	hr.sortedHashes = make([]uint64, 0)

	// Add all nodes
	for _, node := range nodes {
		hr.nodes[node] = true

		// Add virtual nodes
		for i := 0; i < hr.virtualNodes; i++ {
			virtualNodeKey := fmt.Sprintf("%s-%d", node, i)
			hash := hr.hashFunc([]byte(virtualNodeKey))
			hr.ring[hash] = node
			hr.sortedHashes = append(hr.sortedHashes, hash)
		}
	}

	// Keep hashes sorted
	sort.Slice(hr.sortedHashes, func(i, j int) bool {
		return hr.sortedHashes[i] < hr.sortedHashes[j]
	})
}

// Hash returns the hash value for a given key
func (hr *HashRingImpl) Hash(key string) uint64 {
	return hr.hashFunc([]byte(key))
}

// GetLoadDistribution returns load distribution statistics
func (hr *HashRingImpl) GetLoadDistribution() map[string]float64 {
	hr.mu.RLock()
	defer hr.mu.RUnlock()

	distribution := make(map[string]float64)

	if len(hr.nodes) == 0 {
		return distribution
	}

	// Count virtual nodes per physical node
	virtualNodeCounts := make(map[string]int)
	for _, node := range hr.ring {
		virtualNodeCounts[node]++
	}

	// Calculate load percentage
	totalVirtualNodes := len(hr.ring)
	for node, count := range virtualNodeCounts {
		distribution[node] = float64(count) / float64(totalVirtualNodes)
	}

	return distribution
}

// IsBalanced checks if the load distribution is within acceptable limits
func (hr *HashRingImpl) IsBalanced(threshold float64) bool {
	distribution := hr.GetLoadDistribution()

	if len(distribution) <= 1 {
		return true
	}

	expectedLoad := 1.0 / float64(len(distribution))

	for _, load := range distribution {
		deviation := float32(load - expectedLoad)
		if deviation < 0 {
			deviation = -deviation
		}
		if deviation > float32(threshold) {
			return false
		}
	}

	return true
}

// Clone creates a deep copy of the hash ring
func (hr *HashRingImpl) Clone() *HashRingImpl {
	hr.mu.RLock()
	defer hr.mu.RUnlock()

	clone := &HashRingImpl{
		nodes:        make(map[string]bool),
		virtualNodes: hr.virtualNodes,
		ring:         make(map[uint64]string),
		sortedHashes: make([]uint64, len(hr.sortedHashes)),
		hashFunc:     hr.hashFunc,
	}

	// Copy nodes
	for node := range hr.nodes {
		clone.nodes[node] = true
	}

	// Copy ring
	for hash, node := range hr.ring {
		clone.ring[hash] = node
	}

	// Copy sorted hashes
	copy(clone.sortedHashes, hr.sortedHashes)

	return clone
}

// Stats returns statistics about the hash ring
type HashRingStats struct {
	PhysicalNodes      int                `json:"physical_nodes"`
	VirtualNodes       int                `json:"virtual_nodes"`
	VirtualPerPhysical int                `json:"virtual_per_physical"`
	LoadDistribution   map[string]float64 `json:"load_distribution"`
	IsBalanced         bool               `json:"is_balanced"`
	BalanceThreshold   float64            `json:"balance_threshold"`
}

// GetStats returns statistics about the hash ring
func (hr *HashRingImpl) GetStats(balanceThreshold float64) *HashRingStats {
	return &HashRingStats{
		PhysicalNodes:      hr.NodeCount(),
		VirtualNodes:       hr.VirtualNodeCount(),
		VirtualPerPhysical: hr.virtualNodes,
		LoadDistribution:   hr.GetLoadDistribution(),
		IsBalanced:         hr.IsBalanced(balanceThreshold),
		BalanceThreshold:   balanceThreshold,
	}
}

// defaultHashFunc is the default hash function using SHA-256
func defaultHashFunc(data []byte) uint64 {
	hash := sha256.Sum256(data)
	return binary.BigEndian.Uint64(hash[:8])
}

// HashRingManager manages multiple hash rings for different collections
type HashRingManager struct {
	mu    sync.RWMutex
	rings map[string]*HashRingImpl // collectionID -> HashRing
}

// NewHashRingManager creates a new hash ring manager
func NewHashRingManager() *HashRingManager {
	return &HashRingManager{
		rings: make(map[string]*HashRingImpl),
	}
}

// CreateRing creates a new hash ring for a collection
func (hrm *HashRingManager) CreateRing(collectionID string, virtualNodes int, nodes []string) error {
	hrm.mu.Lock()
	defer hrm.mu.Unlock()

	if _, exists := hrm.rings[collectionID]; exists {
		return fmt.Errorf("hash ring for collection %s already exists", collectionID)
	}

	ring := NewHashRing(virtualNodes)
	for _, node := range nodes {
		ring.AddNode(node)
	}

	hrm.rings[collectionID] = ring
	return nil
}

// GetRing returns the hash ring for a collection
func (hrm *HashRingManager) GetRing(collectionID string) (*HashRingImpl, error) {
	hrm.mu.RLock()
	defer hrm.mu.RUnlock()

	ring, exists := hrm.rings[collectionID]
	if !exists {
		return nil, fmt.Errorf("hash ring for collection %s not found", collectionID)
	}

	return ring, nil
}

// DeleteRing removes the hash ring for a collection
func (hrm *HashRingManager) DeleteRing(collectionID string) error {
	hrm.mu.Lock()
	defer hrm.mu.Unlock()

	if _, exists := hrm.rings[collectionID]; !exists {
		return fmt.Errorf("hash ring for collection %s not found", collectionID)
	}

	delete(hrm.rings, collectionID)
	return nil
}

// UpdateRing updates the nodes in a collection's hash ring
func (hrm *HashRingManager) UpdateRing(collectionID string, nodes []string) error {
	hrm.mu.Lock()
	defer hrm.mu.Unlock()

	ring, exists := hrm.rings[collectionID]
	if !exists {
		return fmt.Errorf("hash ring for collection %s not found", collectionID)
	}

	ring.Rebalance(nodes)
	return nil
}

// GetAllRings returns all hash rings
func (hrm *HashRingManager) GetAllRings() map[string]*HashRingImpl {
	hrm.mu.RLock()
	defer hrm.mu.RUnlock()

	result := make(map[string]*HashRingImpl)
	for collectionID, ring := range hrm.rings {
		result[collectionID] = ring.Clone()
	}

	return result
}

// AddNodeToAll adds a node to all hash rings
func (hrm *HashRingManager) AddNodeToAll(node string) {
	hrm.mu.Lock()
	defer hrm.mu.Unlock()

	for _, ring := range hrm.rings {
		ring.AddNode(node)
	}
}

// RemoveNodeFromAll removes a node from all hash rings
func (hrm *HashRingManager) RemoveNodeFromAll(node string) {
	hrm.mu.Lock()
	defer hrm.mu.Unlock()

	for _, ring := range hrm.rings {
		ring.RemoveNode(node)
	}
}

// GetGlobalStats returns statistics for all hash rings
func (hrm *HashRingManager) GetGlobalStats(balanceThreshold float64) map[string]*HashRingStats {
	hrm.mu.RLock()
	defer hrm.mu.RUnlock()

	stats := make(map[string]*HashRingStats)
	for collectionID, ring := range hrm.rings {
		stats[collectionID] = ring.GetStats(balanceThreshold)
	}

	return stats
}
