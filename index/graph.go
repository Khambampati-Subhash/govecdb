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

// InsertBatch inserts multiple vectors with a single lock acquisition for better performance
func (g *HNSWGraph) InsertBatch(vectors []*Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	// Validate all vectors first
	for _, vector := range vectors {
		if len(vector.Data) != g.config.Dimension {
			return ErrDimensionMismatch
		}
	}

	// Acquire lock ONCE for entire batch
	g.mu.Lock()
	defer g.mu.Unlock()

	// Optimized batch insertion strategy
	batchSize := len(vectors)

	// For large batches, use simplified insertion strategy
	if batchSize > 100 {
		return g.insertBatchOptimized(vectors)
	}

	// For smaller batches, use standard insertion
	return g.insertBatchStandard(vectors)
}

// insertBatchOptimized uses a simplified strategy for large batches
func (g *HNSWGraph) insertBatchOptimized(vectors []*Vector) error {
	// Reduce EfConstruction for batch operations to speed up insertion
	originalEf := g.config.EfConstruction
	g.config.EfConstruction = minInt(g.config.EfConstruction/2, 50) // Reduce search effort
	defer func() { g.config.EfConstruction = originalEf }()

	for _, vector := range vectors {
		// Check if node already exists
		if _, exists := g.nodes.GetUnsafe(vector.ID); exists {
			continue // Skip duplicates
		}

		level := g.SelectLayer()
		newNode := NewHNSWNode(vector, level)

		// If this is the first node, make it the entry point
		if g.entryPoint == nil {
			g.entryPoint = newNode
			g.nodes.SetUnsafe(vector.ID, vector)
			g.stats.NodeCount++
			g.stats.MaxLayer = level
			continue
		}

		// Simplified insertion: only search base layer for batch operations
		// This trades some accuracy for significant speed improvement
		entryPoints := []*HNSWNode{g.entryPoint}

		// Only search to find entry points at target level (skip layer-by-layer descent)
		if level < g.entryPoint.Level {
			entryPoints = g.searchLayer(vector.Data, entryPoints, 1, level)
		}

		// Connect only at base layer (layer 0) for batch operations
		candidates := g.searchLayer(vector.Data, entryPoints, g.config.EfConstruction, 0)

		// Use reduced M for faster connection establishment
		m := minInt(g.config.M/2, 8) // Fewer connections for batch operations
		selectedNeighbors := g.selectNeighborsFast(vector.Data, candidates, m)

		// Add bidirectional connections only at base layer
		for _, neighbor := range selectedNeighbors {
			newNode.AddConnection(neighbor, 0)
			g.stats.EdgeCount++

			// Skip pruning for batch operations to save time
			// This may result in slightly more connections but much faster insertion
		}

		// Update entry point if the new node has a higher level
		if level > g.entryPoint.Level {
			g.entryPoint = newNode
			g.stats.MaxLayer = level
		}

		g.nodes.SetUnsafe(vector.ID, vector)
		g.stats.NodeCount++
	}

	return nil
}

// insertBatchStandard uses full HNSW algorithm for smaller batches
func (g *HNSWGraph) insertBatchStandard(vectors []*Vector) error {
	for _, vector := range vectors {
		// Check if node already exists (use unsafe since we hold g.mu)
		if _, exists := g.nodes.GetUnsafe(vector.ID); exists {
			continue // Skip duplicates
		}

		level := g.SelectLayer()
		newNode := NewHNSWNode(vector, level)

		// If this is the first node, make it the entry point
		if g.entryPoint == nil {
			g.entryPoint = newNode
			g.nodes.SetUnsafe(vector.ID, vector)
			g.stats.NodeCount++
			g.stats.MaxLayer = level
			continue
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

		g.nodes.SetUnsafe(vector.ID, vector)
		g.stats.NodeCount++
	}

	return nil
}

// searchLayer performs a greedy search at a specific layer with optimizations
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

	exploredCount := 0
	maxExplored := g.getOptimalSearchLimit(numClosest) // Dynamic limit based on graph size

	for candidates.Len() > 0 && exploredCount < maxExplored {
		current := heap.Pop(candidates).(*NodeCandidate)

		// Early termination: stop if current distance is much worse than the furthest in dynamic list
		if dynamic.Len() >= numClosest {
			worstDynamicDistance := (*dynamic)[0].Distance
			// More aggressive early termination for large graphs
			toleranceFactor := float32(1.1) // 10% tolerance (reduced from 20%)
			if g.stats.NodeCount > 3000 {
				toleranceFactor = 1.05 // 5% tolerance for large graphs
			}
			if current.Distance > worstDynamicDistance*toleranceFactor {
				break
			}
		}

		exploredCount++

		// Explore neighbors with limited scope for better scaling
		connections := current.Node.GetConnections(layer)
		maxNeighborsToCheck := g.getOptimalNeighborLimit(len(connections)) // Dynamic neighbor limit

		checkedCount := 0
		for _, neighbor := range connections {
			if checkedCount >= maxNeighborsToCheck {
				break // Limit exploration scope
			}
			checkedCount++

			if neighbor.IsDeleted() || visited[neighbor.Vector.ID] {
				continue
			}

			visited[neighbor.Vector.ID] = true

			distance, err := g.distanceFunc(query, neighbor.Vector.Data)
			if err != nil {
				continue
			}

			// More aggressive pruning: only add if significantly better
			shouldAdd := false
			if dynamic.Len() < numClosest {
				shouldAdd = true
			} else {
				worstDistance := (*dynamic)[0].Distance
				// Scale pruning aggressiveness with graph size
				improvementRequired := float32(0.95) // Default: must be 5% better
				if g.stats.NodeCount > 3000 {
					improvementRequired = 0.90 // Large graphs: must be 10% better
				}
				if distance < worstDistance*improvementRequired {
					shouldAdd = true
				}
			}

			if shouldAdd {
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

// getOptimalSearchLimit calculates optimal search limits based on graph size
func (g *HNSWGraph) getOptimalSearchLimit(numClosest int) int {
	nodeCount := g.stats.NodeCount

	// Dynamic scaling based on graph size
	if nodeCount < 1000 {
		return numClosest * 100 // Small graphs: thorough search
	} else if nodeCount < 5000 {
		return numClosest * 50 // Medium graphs: balanced
	} else if nodeCount < 20000 {
		return numClosest * 25 // Large graphs: limited search
	} else {
		return numClosest * 10 // Very large graphs: minimal search
	}
}

// getOptimalNeighborLimit calculates optimal neighbor exploration limits
func (g *HNSWGraph) getOptimalNeighborLimit(connectionCount int) int {
	nodeCount := g.stats.NodeCount

	// Scale neighbor exploration based on graph size
	baseLimit := minInt(connectionCount, 100) // Never check more than 100 neighbors

	if nodeCount < 1000 {
		return baseLimit // Small graphs: check all neighbors
	} else if nodeCount < 5000 {
		return minInt(baseLimit, 50) // Medium graphs: moderate limit
	} else {
		return minInt(baseLimit, 25) // Large graphs: aggressive pruning
	}
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

// selectNeighborsFast is a simplified neighbor selection for batch operations
func (g *HNSWGraph) selectNeighborsFast(query []float32, candidates []*HNSWNode, m int) []*HNSWNode {
	if len(candidates) <= m {
		return candidates
	}

	// Simple distance-based selection without complex heuristics
	// Pre-allocate with exact capacity
	result := make([]*HNSWNode, 0, m)
	minDistances := make([]float32, 0, m)

	for _, node := range candidates {
		if node.IsDeleted() {
			continue
		}

		distance, err := g.distanceFunc(query, node.Vector.Data)
		if err != nil {
			continue
		}

		// Simple insertion sort to maintain top-m closest nodes
		if len(result) < m {
			// Still have space, just add
			result = append(result, node)
			minDistances = append(minDistances, distance)
		} else {
			// Find the worst (highest distance) among current results
			worstIdx := 0
			worstDist := minDistances[0]
			for i := 1; i < len(minDistances); i++ {
				if minDistances[i] > worstDist {
					worstDist = minDistances[i]
					worstIdx = i
				}
			}

			// Replace if current candidate is better
			if distance < worstDist {
				result[worstIdx] = node
				minDistances[worstIdx] = distance
			}
		}

		// Early termination for batch optimization
		if len(result) >= m {
			break
		}
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

// Search performs k-NN search in the graph using proper HNSW algorithm
func (g *HNSWGraph) Search(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	// Use default ef based on k
	ef := max(k, g.config.EfConstruction)
	return g.SearchWithEf(query, k, ef, filter)
}

// SearchWithEf performs k-NN search with specified ef parameter
func (g *HNSWGraph) SearchWithEf(query []float32, k int, ef int, filter FilterFunc) ([]*SearchResult, error) {
	if len(query) != g.config.Dimension {
		return nil, ErrDimensionMismatch
	}

	if k <= 0 {
		return nil, ErrInvalidK
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	// If no entry point, return empty results
	if g.entryPoint == nil {
		return []*SearchResult{}, nil
	}

	// Start search from the top layer
	currentLayer := g.entryPoint.Level
	entryPoints := []*HNSWNode{g.entryPoint}

	// Search from top layer down to layer 1
	for currentLayer > 0 {
		entryPoints = g.searchLayer(query, entryPoints, 1, currentLayer)
		currentLayer--
	}

	// Search at base layer (layer 0) with specified ef
	candidates := g.searchLayer(query, entryPoints, ef, 0)

	// Convert candidates to results, applying filter if provided
	results := make([]*SearchResult, 0, min(k, len(candidates)))
	for _, node := range candidates {
		if len(results) >= k {
			break
		}

		if filter != nil && !filter(node.Vector.Metadata) {
			continue
		}

		distance, err := g.distanceFunc(query, node.Vector.Data)
		if err != nil {
			continue
		}

		result := &SearchResult{
			ID:       node.Vector.ID,
			Vector:   node.Vector.Data,
			Score:    distance,
			Metadata: node.Vector.Metadata,
		}

		results = append(results, result)
	}

	return results, nil
}

// SearchFast performs a fast approximate search optimized for large datasets
func (g *HNSWGraph) SearchFast(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if g.entryPoint == nil {
		return []*SearchResult{}, nil
	}

	// Fast search with reduced precision for better performance
	candidates := g.searchLayerFast(query, []*HNSWNode{g.entryPoint}, k*2, 0)

	results := make([]*SearchResult, 0, k)
	for _, node := range candidates {
		if len(results) >= k {
			break
		}

		if filter != nil && !filter(node.Vector.Metadata) {
			continue
		}

		distance, err := g.distanceFunc(query, node.Vector.Data)
		if err != nil {
			continue
		}

		result := &SearchResult{
			ID:       node.Vector.ID,
			Vector:   node.Vector.Data,
			Score:    distance,
			Metadata: node.Vector.Metadata,
		}

		results = append(results, result)
	}

	// Sort by distance
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

// searchLayerFast performs a fast approximate search with reduced precision
func (g *HNSWGraph) searchLayerFast(query []float32, entryPoints []*HNSWNode, numClosest int, layer int) []*HNSWNode {
	visited := make(map[string]bool)
	candidates := make([]*NodeCandidate, 0, numClosest*2)

	// Initialize with entry points
	for _, ep := range entryPoints {
		if ep.IsDeleted() {
			continue
		}

		distance, err := g.distanceFunc(query, ep.Vector.Data)
		if err != nil {
			continue
		}

		candidates = append(candidates, &NodeCandidate{Node: ep, Distance: distance})
		visited[ep.Vector.ID] = true
	}

	// Limited exploration for speed
	maxExplored := minInt(numClosest*5, 100) // Much more aggressive limiting
	exploredCount := 0

	for len(candidates) > 0 && exploredCount < maxExplored {
		// Find current best candidate (simple linear search for small sets)
		bestIdx := 0
		for i := 1; i < len(candidates); i++ {
			if candidates[i].Distance < candidates[bestIdx].Distance {
				bestIdx = i
			}
		}

		current := candidates[bestIdx]
		// Remove current from candidates
		candidates[bestIdx] = candidates[len(candidates)-1]
		candidates = candidates[:len(candidates)-1]

		exploredCount++

		// Explore neighbors (limited)
		connections := current.Node.GetConnections(layer)
		maxNeighbors := minInt(len(connections), 10) // Very aggressive neighbor limiting

		checkedNeighbors := 0
		for _, neighbor := range connections {
			if checkedNeighbors >= maxNeighbors || visited[neighbor.Vector.ID] || neighbor.IsDeleted() {
				continue
			}
			checkedNeighbors++

			visited[neighbor.Vector.ID] = true

			distance, err := g.distanceFunc(query, neighbor.Vector.Data)
			if err != nil {
				continue
			}

			candidates = append(candidates, &NodeCandidate{Node: neighbor, Distance: distance})
		}

		// Keep only best candidates to prevent memory growth
		if len(candidates) > numClosest*3 {
			// Sort and keep top candidates
			for i := 0; i < len(candidates)-1; i++ {
				for j := i + 1; j < len(candidates); j++ {
					if candidates[i].Distance > candidates[j].Distance {
						candidates[i], candidates[j] = candidates[j], candidates[i]
					}
				}
			}
			candidates = candidates[:numClosest*2]
		}
	}

	// Sort final candidates and return top ones
	for i := 0; i < len(candidates)-1; i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[i].Distance > candidates[j].Distance {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	limit := minInt(len(candidates), numClosest)
	result := make([]*HNSWNode, limit)
	for i := 0; i < limit; i++ {
		result[i] = candidates[i].Node
	}

	return result
}

// Delete removes a node from the graph
func (g *HNSWGraph) Delete(id string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	_, exists := g.nodes.Get(id)
	if !exists {
		return ErrNotFound
	}

	// Hard delete from the nodes map for now
	// TODO: Implement proper soft delete when HNSW structure is fixed
	g.nodes.Delete(id)
	g.stats.DeletedCount++
	g.stats.NodeCount--

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

// minInt returns the minimum of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
