package index

import (
	"container/heap"
	"math"
	"sync"
	"sync/atomic"
	"unsafe"
)

// OptimizedHNSWGraph represents a high-performance HNSW graph implementation
type OptimizedHNSWGraph struct {
	// Configuration
	config *Config

	// Distance function (optimized)
	distanceFunc OptimizedDistanceFunc

	// Entry point to the graph (atomic pointer for lock-free access)
	entryPoint unsafe.Pointer // *OptimizedHNSWNode

	// All nodes in the graph by ID (lock-free map)
	nodes *SyncMap[string, *OptimizedHNSWNode]

	// Memory pools for reducing allocations
	nodePool      sync.Pool
	candidatePool sync.Pool
	resultPool    sync.Pool

	// Statistics (atomic counters)
	nodeCount    int64
	edgeCount    int64
	maxLayer     int32
	searchCount  int64
	insertCount  int64
	deletedCount int64

	// Layer selection probability cache
	layerProbabilities []float64
	layerProbMu        sync.RWMutex
}

// OptimizedHNSWNode represents a memory-efficient node with lock-free operations
type OptimizedHNSWNode struct {
	// Vector data (immutable after creation)
	Vector *Vector

	// Level of this node (immutable)
	Level int32

	// Connections to other nodes at each layer (lock-free)
	connections []atomic.Pointer[ConnectionSet] // Each layer has a ConnectionSet

	// Deletion flag (atomic)
	deleted int32 // 0 = not deleted, 1 = deleted

	// Cache-aligned padding to prevent false sharing
	_ [64 - (unsafe.Sizeof(Vector{})+8+8+8)%64]byte
}

// getNodeByID retrieves a node by its vector ID
func (g *OptimizedHNSWGraph) getNodeByID(id string) *OptimizedHNSWNode {
	node, _ := g.nodes.Load(id)
	return node
}

// ConnectionSet represents a set of connections with lock-free operations
type ConnectionSet struct {
	// connections stored as slice for better cache locality
	nodes []*OptimizedHNSWNode
	ids   []string
	count int32
	mu    sync.RWMutex // Only for structural changes, reads are lock-free when possible
}

// OptimizedDistanceFunc is a function type for optimized distance calculations
type OptimizedDistanceFunc func(a, b []float32) float32

// FastSearchResult represents a search result with minimal allocations
type FastSearchResult struct {
	ID       string
	Distance float64
}

// Helper types for sorting
type candidateWithDist struct {
	node     *OptimizedHNSWNode
	distance float64
}

type connWithDist struct {
	node     *OptimizedHNSWNode
	distance float64
}

// NewOptimizedHNSWGraph creates a new optimized HNSW graph
func NewOptimizedHNSWGraph(config *Config) (*OptimizedHNSWGraph, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}

	graph := &OptimizedHNSWGraph{
		config:       config,
		distanceFunc: getOptimizedDistanceFunc(config.Metric),
		nodes:        NewSyncMap[string, *OptimizedHNSWNode](),
	}

	// Initialize memory pools
	graph.nodePool = sync.Pool{
		New: func() interface{} {
			return &OptimizedHNSWNode{}
		},
	}

	graph.candidatePool = sync.Pool{
		New: func() interface{} {
			return make([]*FastSearchResult, 0, config.EfConstruction*2)
		},
	}

	graph.resultPool = sync.Pool{
		New: func() interface{} {
			return make([]*SearchResult, 0, config.EfConstruction)
		},
	}

	// Pre-calculate layer selection probabilities for better performance
	graph.initLayerProbabilities()

	return graph, nil
}

// initLayerProbabilities pre-calculates layer selection probabilities
func (g *OptimizedHNSWGraph) initLayerProbabilities() {
	g.layerProbMu.Lock()
	defer g.layerProbMu.Unlock()

	g.layerProbabilities = make([]float64, g.config.MaxLayer+1)
	mL := 1.0 / math.Log(2.0)

	for i := 0; i <= g.config.MaxLayer; i++ {
		// Probability of selecting layer i using the mL factor
		g.layerProbabilities[i] = math.Exp(-float64(i)/mL) - math.Exp(-float64(i+1)/mL)
	}
}

// selectLayerOptimized selects a layer using pre-calculated probabilities
func (g *OptimizedHNSWGraph) selectLayerOptimized() int {
	// Fast path for common case (layer 0)
	if fastRand()&0x7FFFFFFF < 0x40000000 { // ~50% chance
		return 0
	}

	g.layerProbMu.RLock()
	defer g.layerProbMu.RUnlock()

	r := fastRandFloat64()
	cumulative := 0.0

	for i, prob := range g.layerProbabilities {
		cumulative += prob
		if r <= cumulative {
			return i
		}
	}

	return g.config.MaxLayer
}

// Insert adds a new node to the graph with optimized performance
func (g *OptimizedHNSWGraph) Insert(vector *Vector) error {
	if len(vector.Data) != g.config.Dimension {
		return ErrDimensionMismatch
	}

	// Check if node already exists (lock-free)
	if existing, exists := g.nodes.Load(vector.ID); exists && !existing.IsDeleted() {
		return nil // Node already exists
	}

	// Create new node
	level := g.selectLayerOptimized()
	newNode := g.createOptimizedNode(vector, level)

	// Atomic increment
	atomic.AddInt64(&g.insertCount, 1)
	atomic.AddInt64(&g.nodeCount, 1)

	// Handle first node case
	currentEntry := (*OptimizedHNSWNode)(atomic.LoadPointer(&g.entryPoint))
	if currentEntry == nil {
		if atomic.CompareAndSwapPointer(&g.entryPoint, nil, unsafe.Pointer(newNode)) {
			g.nodes.Store(vector.ID, newNode)
			atomic.StoreInt32(&g.maxLayer, int32(level))
			return nil
		}
		// Another goroutine set the entry point, continue with normal insertion
		currentEntry = (*OptimizedHNSWNode)(atomic.LoadPointer(&g.entryPoint))
	}

	// Find entry points for insertion starting from the top
	entryPoints := []*OptimizedHNSWNode{currentEntry}

	// Search from top layer down to level+1 (greedy search)
	currentMaxLayer := int(atomic.LoadInt32(&g.maxLayer))
	for currentLayer := currentMaxLayer; currentLayer > level; currentLayer-- {
		entryPoints = g.searchLayerOptimized(vector.Data, entryPoints, 1, currentLayer)
	}

	// Search and connect at each layer from level down to 0
	for currentLayer := min(level, currentMaxLayer); currentLayer >= 0; currentLayer-- {
		candidates := g.searchLayerOptimized(vector.Data, entryPoints, g.config.EfConstruction, currentLayer)

		// Select M neighbors to connect to
		m := g.config.M
		if currentLayer == 0 {
			m = g.config.M * 2 // More connections at base layer
		}

		selectedNeighbors := g.selectNeighborsOptimized(vector.Data, candidates, m)

		// Add bidirectional connections
		for _, neighbor := range selectedNeighbors {
			g.addConnectionOptimized(newNode, neighbor, currentLayer)
			g.pruneConnectionsOptimized(neighbor, currentLayer)
		}

		entryPoints = selectedNeighbors
	}

	// Update entry point if the new node has a higher level
	for {
		currentMaxLayer := int(atomic.LoadInt32(&g.maxLayer))
		if level <= currentMaxLayer {
			break
		}
		if atomic.CompareAndSwapInt32(&g.maxLayer, int32(currentMaxLayer), int32(level)) {
			atomic.StorePointer(&g.entryPoint, unsafe.Pointer(newNode))
			break
		}
	}

	g.nodes.Store(vector.ID, newNode)
	return nil
}

// searchLayerOptimized performs optimized greedy search at a specific layer
func (g *OptimizedHNSWGraph) searchLayerOptimized(query []float32, entryPoints []*OptimizedHNSWNode, numClosest int, layer int) []*OptimizedHNSWNode {
	// Get candidate slice from pool
	candidatesSlice := g.candidatePool.Get().([]*FastSearchResult)
	defer func() {
		candidatesSlice = candidatesSlice[:0] // Reset slice
		g.candidatePool.Put(candidatesSlice)
	}()

	visited := make(map[string]bool, numClosest*4) // Pre-allocate with reasonable size
	candidates := &FastResultHeap{}
	dynamic := &MaxFastResultHeap{}

	heap.Init(candidates)
	heap.Init(dynamic)

	// Initialize with entry points
	for _, ep := range entryPoints {
		if ep.IsDeleted() {
			continue
		}

		distance := g.distanceFunc(query, ep.Vector.Data)
		result := &FastSearchResult{ID: ep.Vector.ID, Distance: float64(distance)}

		heap.Push(candidates, result)
		heap.Push(dynamic, result)
		visited[ep.Vector.ID] = true
	}

	for candidates.Len() > 0 {
		current := heap.Pop(candidates).(*FastSearchResult)

		// Stop if current distance is worse than the furthest in dynamic list
		if dynamic.Len() >= numClosest && current.Distance > (*dynamic)[0].Distance {
			break
		}

		// Explore neighbors efficiently
		// Find the actual node from ID for connections
		nodeID := current.ID
		currentNode := g.getNodeByID(nodeID)
		if currentNode == nil {
			continue
		}
		connections := currentNode.getConnectionsOptimized(layer)
		for _, neighbor := range connections {
			if neighbor.IsDeleted() || visited[neighbor.Vector.ID] {
				continue
			}

			visited[neighbor.Vector.ID] = true
			distance := g.distanceFunc(query, neighbor.Vector.Data)

			// Add to candidates if it's closer than the furthest in dynamic list
			if dynamic.Len() < numClosest || float64(distance) < (*dynamic)[0].Distance {
				result := &FastSearchResult{ID: neighbor.Vector.ID, Distance: float64(distance)}
				heap.Push(candidates, result)
				heap.Push(dynamic, result)

				// Remove furthest if we have too many
				if dynamic.Len() > numClosest {
					heap.Pop(dynamic)
				}
			}
		}
	}

	// Convert heap to slice
	result := make([]*OptimizedHNSWNode, dynamic.Len())
	for i := dynamic.Len() - 1; i >= 0; i-- {
		resultItem := heap.Pop(dynamic).(*FastSearchResult)
		resultNode := g.getNodeByID(resultItem.ID)
		if resultNode != nil {
			result[i] = resultNode
		}
	}

	return result
}

// Search performs optimized k-NN search in the graph
func (g *OptimizedHNSWGraph) Search(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	if len(query) != g.config.Dimension {
		return nil, ErrDimensionMismatch
	}

	if k <= 0 {
		return nil, ErrInvalidK
	}

	atomic.AddInt64(&g.searchCount, 1)

	// Get entry point
	entryPoint := (*OptimizedHNSWNode)(atomic.LoadPointer(&g.entryPoint))
	if entryPoint == nil {
		return []*SearchResult{}, nil // Empty graph
	}

	entryPoints := []*OptimizedHNSWNode{entryPoint}
	maxLayer := int(atomic.LoadInt32(&g.maxLayer))

	// Search from top layer down to layer 1 (greedy search with ef=1)
	for currentLayer := maxLayer; currentLayer >= 1; currentLayer-- {
		entryPoints = g.searchLayerOptimized(query, entryPoints, 1, currentLayer)
	}

	// Search at layer 0 with larger ef
	ef := max(g.config.EfConstruction, k)
	candidates := g.searchLayerOptimized(query, entryPoints, ef, 0)

	// Get results slice from pool
	results := g.resultPool.Get().([]*SearchResult)
	defer func() {
		results = results[:0] // Reset slice
		g.resultPool.Put(results)
	}()

	// Convert to SearchResult format and apply filter
	for _, node := range candidates {
		if node.IsDeleted() {
			continue
		}

		// Apply filter if provided
		if filter != nil && !filter(node.Vector.Metadata) {
			continue
		}

		distance := g.distanceFunc(query, node.Vector.Data)
		result := &SearchResult{
			ID:       node.Vector.ID,
			Vector:   node.Vector.Data,
			Score:    distance,
			Metadata: node.Vector.Metadata,
		}
		results = append(results, result)

		if len(results) >= k {
			break
		}
	}

	// Sort results by distance
	quickSortResults(results)

	// Return copy of results (up to k)
	finalResults := make([]*SearchResult, min(len(results), k))
	copy(finalResults, results[:min(len(results), k)])

	return finalResults, nil
}

// createOptimizedNode creates a new optimized node
func (g *OptimizedHNSWGraph) createOptimizedNode(vector *Vector, level int) *OptimizedHNSWNode {
	node := &OptimizedHNSWNode{
		Vector:      vector,
		Level:       int32(level),
		connections: make([]atomic.Pointer[ConnectionSet], level+1),
	}

	// Initialize connection sets
	for i := 0; i <= level; i++ {
		connSet := &ConnectionSet{
			nodes: make([]*OptimizedHNSWNode, 0, g.config.M*2),
			ids:   make([]string, 0, g.config.M*2),
		}
		node.connections[i].Store(connSet)
	}

	return node
}

// IsDeleted returns whether the node is marked as deleted (atomic)
func (n *OptimizedHNSWNode) IsDeleted() bool {
	return atomic.LoadInt32(&n.deleted) == 1
}

// MarkDeleted marks the node as deleted (atomic)
func (n *OptimizedHNSWNode) MarkDeleted() {
	atomic.StoreInt32(&n.deleted, 1)
}

// getConnectionsOptimized returns connections at a layer (optimized for speed)
func (n *OptimizedHNSWNode) getConnectionsOptimized(layer int) []*OptimizedHNSWNode {
	if layer < 0 || layer >= len(n.connections) {
		return nil
	}

	connSet := n.connections[layer].Load()
	if connSet == nil {
		return nil
	}

	connSet.mu.RLock()
	defer connSet.mu.RUnlock()

	// Return slice directly (caller should not modify)
	return connSet.nodes[:connSet.count]
}

// addConnectionOptimized adds a bidirectional connection (optimized)
func (g *OptimizedHNSWGraph) addConnectionOptimized(node1, node2 *OptimizedHNSWNode, layer int) {
	if node1 == node2 || layer < 0 {
		return
	}

	// Add connection from node1 to node2
	g.addUnidirectionalConnection(node1, node2, layer)

	// Add connection from node2 to node1
	g.addUnidirectionalConnection(node2, node1, layer)

	atomic.AddInt64(&g.edgeCount, 1)
}

// addUnidirectionalConnection adds a connection in one direction
func (g *OptimizedHNSWGraph) addUnidirectionalConnection(from, to *OptimizedHNSWNode, layer int) {
	if layer >= len(from.connections) {
		return
	}

	connSet := from.connections[layer].Load()
	if connSet == nil {
		// Initialize connection set if it doesn't exist
		initialCapacity := g.config.M * 2
		newConnSet := &ConnectionSet{
			nodes: make([]*OptimizedHNSWNode, initialCapacity),
			ids:   make([]string, initialCapacity),
			count: 0,
		}
		from.connections[layer].Store(newConnSet)
		connSet = newConnSet
	}

	connSet.mu.Lock()
	defer connSet.mu.Unlock()

	// Check if connection already exists
	for i := range connSet.nodes[:connSet.count] {
		if connSet.ids[i] == to.Vector.ID {
			return // Connection already exists
		}
	}

	// Add connection
	if int(connSet.count) < len(connSet.nodes) {
		connSet.nodes[connSet.count] = to
		connSet.ids[connSet.count] = to.Vector.ID
		atomic.AddInt32(&connSet.count, 1)
	} else {
		// Need to expand - create new connection set
		newCapacity := len(connSet.nodes) * 2
		if newCapacity == 0 {
			newCapacity = g.config.M * 2
		}
		newNodes := make([]*OptimizedHNSWNode, newCapacity)
		newIds := make([]string, newCapacity)
		copy(newNodes, connSet.nodes)
		copy(newIds, connSet.ids)

		newNodes[connSet.count] = to
		newIds[connSet.count] = to.Vector.ID

		newConnSet := &ConnectionSet{
			nodes: newNodes,
			ids:   newIds,
			count: connSet.count + 1,
		}

		from.connections[layer].Store(newConnSet)
	}
}

// selectNeighborsOptimized selects the best neighbors using optimized heuristics
func (g *OptimizedHNSWGraph) selectNeighborsOptimized(query []float32, candidates []*OptimizedHNSWNode, m int) []*OptimizedHNSWNode {
	if len(candidates) <= m {
		return candidates
	}

	// Use a more efficient selection algorithm

	candidateList := make([]candidateWithDist, 0, len(candidates))
	for _, node := range candidates {
		if node.IsDeleted() {
			continue
		}

		distance := g.distanceFunc(query, node.Vector.Data)
		candidateList = append(candidateList, candidateWithDist{node: node, distance: float64(distance)})
	}

	// Use partial sort (only sort first m elements)
	partialSortCandidates(candidateList, m)

	// Select top m candidates
	result := make([]*OptimizedHNSWNode, min(m, len(candidateList)))
	for i := 0; i < len(result); i++ {
		result[i] = candidateList[i].node
	}

	return result
}

// pruneConnectionsOptimized removes excess connections (optimized)
func (g *OptimizedHNSWGraph) pruneConnectionsOptimized(node *OptimizedHNSWNode, layer int) {
	maxConnections := g.config.M
	if layer == 0 {
		maxConnections = g.config.M * 2
	}

	connections := node.getConnectionsOptimized(layer)
	if len(connections) <= maxConnections {
		return
	}

	// Calculate distances and select best connections

	connList := make([]connWithDist, 0, len(connections))
	for _, conn := range connections {
		if conn.IsDeleted() {
			continue
		}

		distance := g.distanceFunc(node.Vector.Data, conn.Vector.Data)
		connList = append(connList, connWithDist{node: conn, distance: float64(distance)})
	}

	// Sort and keep only maxConnections
	partialSortConnections(connList, maxConnections)

	// Update connection set
	connSet := node.connections[layer].Load()
	if connSet == nil {
		return
	}

	connSet.mu.Lock()
	defer connSet.mu.Unlock()

	// Clear and rebuild with best connections
	for i := 0; i < min(maxConnections, len(connList)); i++ {
		connSet.nodes[i] = connList[i].node
		connSet.ids[i] = connList[i].node.Vector.ID
	}
	atomic.StoreInt32(&connSet.count, int32(min(maxConnections, len(connList))))
}

// Helper functions and data structures

// FastResultHeap implements a min-heap for FastSearchResult
type FastResultHeap []*FastSearchResult

func (h FastResultHeap) Len() int           { return len(h) }
func (h FastResultHeap) Less(i, j int) bool { return h[i].Distance < h[j].Distance }
func (h FastResultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *FastResultHeap) Push(x interface{}) {
	*h = append(*h, x.(*FastSearchResult))
}

func (h *FastResultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// MaxFastResultHeap implements a max-heap for FastSearchResult
type MaxFastResultHeap []*FastSearchResult

func (h MaxFastResultHeap) Len() int           { return len(h) }
func (h MaxFastResultHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance }
func (h MaxFastResultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MaxFastResultHeap) Push(x interface{}) {
	*h = append(*h, x.(*FastSearchResult))
}

func (h *MaxFastResultHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// Utility functions

// min returns the minimum of two integers

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Fast random number generator (thread-local)
var rngState uint64 = 1

func fastRand() uint32 {
	rngState = rngState*1103515245 + 12345
	return uint32(rngState >> 16)
}

func fastRandFloat64() float64 {
	return float64(fastRand()) / float64(1<<32)
}

// partialSort sorts only the first k elements (more efficient than full sort)
func partialSort[T any](slice []T, k int, less func(i, j int) bool) {
	if k >= len(slice) {
		// Full sort needed
		for i := 0; i < len(slice)-1; i++ {
			for j := i + 1; j < len(slice); j++ {
				if less(j, i) {
					slice[i], slice[j] = slice[j], slice[i]
				}
			}
		}
		return
	}

	// Partial selection sort for first k elements
	for i := 0; i < k; i++ {
		minIdx := i
		for j := i + 1; j < len(slice); j++ {
			if less(j, minIdx) {
				minIdx = j
			}
		}
		if minIdx != i {
			slice[i], slice[minIdx] = slice[minIdx], slice[i]
		}
	}
}

// Specific partial sort for candidateWithDist
func partialSortCandidates(candidates []candidateWithDist, k int) {
	if k >= len(candidates) {
		// Full sort needed
		for i := 0; i < len(candidates)-1; i++ {
			for j := i + 1; j < len(candidates); j++ {
				if candidates[j].distance < candidates[i].distance {
					candidates[i], candidates[j] = candidates[j], candidates[i]
				}
			}
		}
		return
	}

	// Partial selection sort for first k elements
	for i := 0; i < k; i++ {
		minIdx := i
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].distance < candidates[minIdx].distance {
				minIdx = j
			}
		}
		if minIdx != i {
			candidates[i], candidates[minIdx] = candidates[minIdx], candidates[i]
		}
	}
}

// Specific partial sort for connWithDist
func partialSortConnections(connections []connWithDist, k int) {
	if k >= len(connections) {
		// Full sort needed
		for i := 0; i < len(connections)-1; i++ {
			for j := i + 1; j < len(connections); j++ {
				if connections[j].distance < connections[i].distance {
					connections[i], connections[j] = connections[j], connections[i]
				}
			}
		}
		return
	}

	// Partial selection sort for first k elements
	for i := 0; i < k; i++ {
		minIdx := i
		for j := i + 1; j < len(connections); j++ {
			if connections[j].distance < connections[minIdx].distance {
				minIdx = j
			}
		}
		if minIdx != i {
			connections[i], connections[minIdx] = connections[minIdx], connections[i]
		}
	}
}

// Size returns the number of nodes in the graph
func (g *OptimizedHNSWGraph) Size() int {
	return g.nodes.Size()
}

// GetStats returns optimized graph statistics
func (g *OptimizedHNSWGraph) GetStats() *GraphStats {
	return &GraphStats{
		NodeCount:    atomic.LoadInt64(&g.nodeCount),
		EdgeCount:    atomic.LoadInt64(&g.edgeCount),
		MaxLayer:     int(atomic.LoadInt32(&g.maxLayer)),
		AvgDegree:    float64(atomic.LoadInt64(&g.edgeCount)) / float64(atomic.LoadInt64(&g.nodeCount)),
		DeletedCount: atomic.LoadInt64(&g.deletedCount),
	}
}

// quickSortResults sorts search results by score
func quickSortResults(results []*SearchResult) {
	if len(results) <= 1 {
		return
	}

	pivot := partition(results)
	quickSortResults(results[:pivot])
	quickSortResults(results[pivot+1:])
}

func partition(results []*SearchResult) int {
	pivot := results[len(results)-1].Score
	i := -1

	for j := 0; j < len(results)-1; j++ {
		if results[j].Score <= pivot {
			i++
			results[i], results[j] = results[j], results[i]
		}
	}

	results[i+1], results[len(results)-1] = results[len(results)-1], results[i+1]
	return i + 1
}
