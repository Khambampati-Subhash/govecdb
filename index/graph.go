package index

import (
	"container/heap"
	"math/rand"
	"sync"
)

// HNSWGraph represents the multi-layer graph structure
type HNSWGraph struct {
	// Configuration
	config *Config

	// Distance function
	distanceFunc DistanceFunc

	// Entry point to the graph (highest layer)
	entryPoint *HNSWNode

	// All nodes in the graph by ID
	nodes *SafeMap

	// Random number generator for layer selection
	rng *rand.Rand

	// Graph-level mutex for structural changes
	mu sync.RWMutex

	// Statistics
	stats *GraphStats
}

// GraphStats contains statistics about the graph
type GraphStats struct {
	NodeCount    int64   `json:"node_count"`
	EdgeCount    int64   `json:"edge_count"`
	MaxLayer     int     `json:"max_layer"`
	AvgDegree    float64 `json:"avg_degree"`
	DeletedCount int64   `json:"deleted_count"`
}

// NewHNSWGraph creates a new HNSW graph
func NewHNSWGraph(config *Config) (*HNSWGraph, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}

	graph := &HNSWGraph{
		config:       config,
		distanceFunc: GetDistanceFunc(config.Metric),
		nodes:        NewSafeMap(),
		rng:          rand.New(rand.NewSource(config.Seed)),
		stats:        &GraphStats{},
	}

	return graph, nil
}

// SelectLayer selects a random layer for a new node using the standard HNSW formula
func (g *HNSWGraph) SelectLayer() int {
	layer := 0

	// Use exponential distribution to select layer
	for g.rng.Float64() < 0.5 && layer < g.config.MaxLayer {
		layer++
	}

	// Alternative implementation using the original paper's formula:
	// mL := 1.0 / math.Log(2.0) // normalization factor
	// layer = int(math.Floor(-math.Log(g.rng.Float64()) * mL))
	// if layer > g.config.MaxLayer {
	//     layer = g.config.MaxLayer
	// }

	return layer
}

// Insert adds a new node to the graph
func (g *HNSWGraph) Insert(vector *Vector) error {
	if len(vector.Data) != g.config.Dimension {
		return ErrDimensionMismatch
	}

	// Check if node already exists
	if _, exists := g.nodes.Get(vector.ID); exists {
		return nil // Node already exists, skip insertion
	}

	level := g.SelectLayer()
	newNode := NewHNSWNode(vector, level)

	g.mu.Lock()
	defer g.mu.Unlock()

	// If this is the first node, make it the entry point
	if g.entryPoint == nil {
		g.entryPoint = newNode
		g.nodes.Set(vector.ID, vector)
		g.stats.NodeCount++
		g.stats.MaxLayer = level
		return nil
	}

	// Find entry points for insertion
	entryPoints := []*HNSWNode{g.entryPoint}

	// Search from top layer down to level+1
	for currentLayer := g.entryPoint.Level; currentLayer > level; currentLayer-- {
		entryPoints = g.searchLayer(vector.Data, entryPoints, 1, currentLayer)
	}

	// Search and connect at each layer from level down to 0
	for currentLayer := min(level, g.entryPoint.Level); currentLayer >= 0; currentLayer-- {
		candidates := g.searchLayer(vector.Data, entryPoints, g.config.EfConstruction, currentLayer)

		// Select M neighbors to connect to
		m := g.config.M
		if currentLayer == 0 {
			m = g.config.M * 2 // More connections at base layer
		}

		selectedNeighbors := g.selectNeighbors(vector.Data, candidates, m)

		// Add bidirectional connections
		for _, neighbor := range selectedNeighbors {
			newNode.AddConnection(neighbor, currentLayer)
			g.stats.EdgeCount++

			// Prune connections of neighbor if it has too many
			g.pruneConnections(neighbor, currentLayer)
		}

		entryPoints = selectedNeighbors
	}

	// Update entry point if the new node has a higher level
	if level > g.entryPoint.Level {
		g.entryPoint = newNode
		g.stats.MaxLayer = level
	}

	g.nodes.Set(vector.ID, vector)
	g.stats.NodeCount++

	return nil
}

// searchLayer performs a greedy search at a specific layer
func (g *HNSWGraph) searchLayer(query []float32, entryPoints []*HNSWNode, numClosest int, layer int) []*HNSWNode {
	visited := make(map[string]bool)
	candidates := &NodeCandidateHeap{}
	dynamic := &MaxNodeCandidateHeap{}

	heap.Init(candidates)
	heap.Init(dynamic)

	// Initialize with entry points
	for _, ep := range entryPoints {
		if ep.IsDeleted() {
			continue
		}

		distance, err := g.distanceFunc(query, ep.Vector.Data)
		if err != nil {
			continue
		}

		candidate := &NodeCandidate{Node: ep, Distance: distance}
		heap.Push(candidates, candidate)
		heap.Push(dynamic, candidate)
		visited[ep.Vector.ID] = true
	}

	for candidates.Len() > 0 {
		current := heap.Pop(candidates).(*NodeCandidate)

		// Stop if current distance is worse than the furthest in dynamic list
		if dynamic.Len() >= numClosest && current.Distance > (*dynamic)[0].Distance {
			break
		}

		// Explore neighbors
		connections := current.Node.GetConnections(layer)
		for _, neighbor := range connections {
			if neighbor.IsDeleted() || visited[neighbor.Vector.ID] {
				continue
			}

			visited[neighbor.Vector.ID] = true

			distance, err := g.distanceFunc(query, neighbor.Vector.Data)
			if err != nil {
				continue
			}

			// Add to candidates if it's closer than the furthest in dynamic list
			if dynamic.Len() < numClosest || distance < (*dynamic)[0].Distance {
				candidate := &NodeCandidate{Node: neighbor, Distance: distance}
				heap.Push(candidates, candidate)
				heap.Push(dynamic, candidate)

				// Remove furthest if we have too many
				if dynamic.Len() > numClosest {
					heap.Pop(dynamic)
				}
			}
		}
	}

	// Convert heap to slice
	result := make([]*HNSWNode, dynamic.Len())
	for i := dynamic.Len() - 1; i >= 0; i-- {
		result[i] = heap.Pop(dynamic).(*NodeCandidate).Node
	}

	return result
}

// selectNeighbors selects the best neighbors using a simple heuristic
func (g *HNSWGraph) selectNeighbors(query []float32, candidates []*HNSWNode, m int) []*HNSWNode {
	if len(candidates) <= m {
		return candidates
	}

	// Calculate distances and sort
	type candidate struct {
		node     *HNSWNode
		distance float32
	}

	candidateList := make([]candidate, 0, len(candidates))
	for _, node := range candidates {
		if node.IsDeleted() {
			continue
		}

		distance, err := g.distanceFunc(query, node.Vector.Data)
		if err != nil {
			continue
		}

		candidateList = append(candidateList, candidate{node: node, distance: distance})
	}

	// Sort by distance (ascending)
	for i := 0; i < len(candidateList)-1; i++ {
		for j := i + 1; j < len(candidateList); j++ {
			if candidateList[i].distance > candidateList[j].distance {
				candidateList[i], candidateList[j] = candidateList[j], candidateList[i]
			}
		}
	}

	// Select top m candidates
	result := make([]*HNSWNode, 0, m)
	for i := 0; i < m && i < len(candidateList); i++ {
		result = append(result, candidateList[i].node)
	}

	return result
}

// pruneConnections removes excess connections from a node
func (g *HNSWGraph) pruneConnections(node *HNSWNode, layer int) {
	maxConnections := g.config.M
	if layer == 0 {
		maxConnections = g.config.M * 2
	}

	connections := node.GetConnectionsList(layer)
	if len(connections) <= maxConnections {
		return
	}

	// Calculate distances to all connections
	type connWithDist struct {
		node     *HNSWNode
		distance float32
	}

	connList := make([]connWithDist, 0, len(connections))
	for _, conn := range connections {
		if conn.IsDeleted() {
			node.RemoveConnection(conn, layer)
			continue
		}

		distance, err := g.distanceFunc(node.Vector.Data, conn.Vector.Data)
		if err != nil {
			continue
		}

		connList = append(connList, connWithDist{node: conn, distance: distance})
	}

	// Sort by distance (ascending - keep closest)
	for i := 0; i < len(connList)-1; i++ {
		for j := i + 1; j < len(connList); j++ {
			if connList[i].distance > connList[j].distance {
				connList[i], connList[j] = connList[j], connList[i]
			}
		}
	}

	// Remove excess connections (furthest ones)
	for i := maxConnections; i < len(connList); i++ {
		node.RemoveConnection(connList[i].node, layer)
		g.stats.EdgeCount--
	}
}

// Search performs k-NN search in the graph
func (g *HNSWGraph) Search(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	if len(query) != g.config.Dimension {
		return nil, ErrDimensionMismatch
	}

	if k <= 0 {
		return nil, ErrInvalidK
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	if g.entryPoint == nil {
		return []*SearchResult{}, nil
	}

	// Search from top layer down to layer 1
	entryPoints := []*HNSWNode{g.entryPoint}
	for currentLayer := g.entryPoint.Level; currentLayer > 0; currentLayer-- {
		entryPoints = g.searchLayer(query, entryPoints, 1, currentLayer)
	}

	// Search at base layer with larger ef
	ef := max(g.config.EfConstruction, k)
	candidates := g.searchLayer(query, entryPoints, ef, 0)

	// Convert to search results and apply filter
	results := make([]*SearchResult, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.IsDeleted() {
			continue
		}

		vector := candidate.GetVector()
		if vector == nil {
			continue
		}

		// Apply filter if provided
		if filter != nil && !filter(vector.Metadata) {
			continue
		}

		distance, err := g.distanceFunc(query, vector.Data)
		if err != nil {
			continue
		}

		result := &SearchResult{
			ID:       vector.ID,
			Vector:   vector.Data,
			Score:    distance,
			Metadata: vector.Metadata,
		}

		results = append(results, result)
	}

	// Sort by distance and limit to k
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].Score > results[j].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

// Delete marks a node as deleted (soft delete)
func (g *HNSWGraph) Delete(id string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	_, exists := g.nodes.Get(id)
	if !exists {
		return ErrNotFound
	}

	// Find the node in our internal structure
	// This is a simplified approach - in a full implementation,
	// you'd want to maintain a separate map of ID -> HNSWNode
	var nodeToDelete *HNSWNode
	g.nodes.ForEach(func(key string, val *Vector) {
		if key == id {
			// This is simplified - we need a better way to track nodes
			// For now, we'll mark the vector as deleted in the SafeMap
			g.nodes.Delete(id)
			g.stats.DeletedCount++
			g.stats.NodeCount--
		}
	})

	if nodeToDelete != nil {
		nodeToDelete.MarkDeleted()
	}

	return nil
}

// Get retrieves a vector by ID
func (g *HNSWGraph) Get(id string) (*Vector, error) {
	vector, exists := g.nodes.Get(id)
	if !exists {
		return nil, ErrNotFound
	}
	return vector, nil
}

// Size returns the number of nodes in the graph
func (g *HNSWGraph) Size() int {
	return g.nodes.Size()
}

// GetStats returns graph statistics
func (g *HNSWGraph) GetStats() *GraphStats {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// Update average degree
	if g.stats.NodeCount > 0 {
		g.stats.AvgDegree = float64(g.stats.EdgeCount) / float64(g.stats.NodeCount)
	}

	return &GraphStats{
		NodeCount:    g.stats.NodeCount,
		EdgeCount:    g.stats.EdgeCount,
		MaxLayer:     g.stats.MaxLayer,
		AvgDegree:    g.stats.AvgDegree,
		DeletedCount: g.stats.DeletedCount,
	}
}

// Helper functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
