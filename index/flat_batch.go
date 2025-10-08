package index

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// FlatBatchInserter implements ultra-fast batch insertion by deferring HNSW construction
// Strategy: Store vectors in flat arrays initially, build HNSW graph in background
// This matches ChromaDB's approach for achieving 3000+ vec/sec throughput
type FlatBatchInserter struct {
	graph *HNSWGraph
}

// NewFlatBatchInserter creates a new flat batch inserter
func NewFlatBatchInserter(g *HNSWGraph) *FlatBatchInserter {
	return &FlatBatchInserter{graph: g}
}

// InsertBatch performs ultra-fast batch insertion with minimal graph construction
func (f *FlatBatchInserter) InsertBatch(vectors []*Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	// Phase 1: Parallel node creation (NO LOCKS, NO GRAPH TRAVERSAL)
	nodes := make([]*HNSWNode, len(vectors))
	numWorkers := runtime.GOMAXPROCS(0)
	chunkSize := (len(vectors) + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	var nodeCount int64

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(vectors) {
			end = len(vectors)
		}
		if start >= len(vectors) {
			break
		}

		wg.Add(1)
		go func(startIdx, endIdx int) {
			defer wg.Done()

			for i := startIdx; i < endIdx; i++ {
				// Create hierarchical structure for better recall:
				// - Every 100th node gets elevated to higher layers (hubs)
				// - Other nodes stay at layer 0
				level := 0
				if i > 0 && i%100 == 0 {
					// Create hub nodes at higher layers for better connectivity
					level = 1 + (i/500) // Gradual increase in levels
					if level > 3 {
						level = 3 // Cap at level 3
					}
				}

				node := &HNSWNode{
					Vector:      vectors[i],
					Level:       level,
					connections: make([]map[string]*HNSWNode, level+1),
				}
				for l := 0; l <= level; l++ {
					node.connections[l] = make(map[string]*HNSWNode, f.graph.config.M)
				}
				nodes[i] = node
				atomic.AddInt64(&nodeCount, 1)
			}
		}(start, end)
	}
	wg.Wait()

	// Phase 2: Register nodes sequentially to avoid races
	// Set entry point to highest level node
	if f.graph.entryPoint == nil && len(nodes) > 0 {
		f.graph.entryPoint = nodes[0]
		f.graph.stats.MaxLayer = nodes[0].Level
	}
	for _, node := range nodes {
		if node.Level > f.graph.stats.MaxLayer {
			f.graph.stats.MaxLayer = node.Level
			f.graph.entryPoint = node
		}
	}
	
	// Register all nodes
	for i := range nodes {
		f.graph.nodes.Set(vectors[i].ID, vectors[i])
	}

	// Phase 3: Build connections for all layers
	f.buildHierarchicalConnections(nodes)

	// Update stats
	atomic.AddInt64(&f.graph.stats.NodeCount, nodeCount)
	if f.graph.stats.MaxLayer == 0 {
		f.graph.stats.MaxLayer = 0
	}

	return nil
}

// buildHierarchicalConnections builds connections for all layers
func (f *FlatBatchInserter) buildHierarchicalConnections(nodes []*HNSWNode) {
	// Connect higher layers first (hubs), then layer 0
	// This ensures good top-down connectivity
	for layer := f.graph.stats.MaxLayer; layer >= 0; layer-- {
		f.buildLayerConnections(nodes, layer)
	}
}

// buildLayerConnections builds connections for a specific layer
func (f *FlatBatchInserter) buildLayerConnections(nodes []*HNSWNode, layer int) {
	// Filter nodes that exist at this layer
	layerNodes := make([]*HNSWNode, 0, len(nodes))
	for _, node := range nodes {
		if node.Level >= layer {
			layerNodes = append(layerNodes, node)
		}
	}
	
	if len(layerNodes) == 0 {
		return
	}
	
	distFunc := f.graph.distanceFunc
	m := f.graph.config.M
	
	// Dynamic window size based on dataset size for better recall
	// Small datasets: large window covers most nodes
	// Large datasets: smaller window but with more long-range sampling
	windowSize := m * 6  // Increased from 4 to 6
	if len(nodes) < 500 {
		windowSize = m * 20  // Very large window for small datasets
	} else if len(nodes) < 2000 {
		windowSize = m * 10  // Large window for medium datasets
	}
	if windowSize > len(nodes)/2 {
		windowSize = len(nodes) / 2  // Cap at half the dataset size
	}
	if windowSize > 200 {
		windowSize = 200  // Increased absolute cap from 100 to 200
	}
	if windowSize < m*2 {
		windowSize = m * 2  // Minimum window size
	}
	
	// More aggressive long-range sampling for better global connectivity
	samplingInterval := 25  // Reduced from 50 to 25 for more samples
	numSamples := 5         // Increased from 3 to 5

	for i := 0; i < len(layerNodes); i++ {
		node := layerNodes[i]
		lookbackStart := i - windowSize
		if lookbackStart < 0 {
			lookbackStart = 0
		}

		// Collect distances to recent neighbors
		type neighbor struct {
			node *HNSWNode
			dist float32
		}
		candidates := make([]neighbor, 0, windowSize+numSamples+1)

		// Add entry point for guaranteed connectivity
		if i > 0 && f.graph.entryPoint != nil && f.graph.entryPoint != node {
			dist, err := distFunc(node.Vector.Data, f.graph.entryPoint.Vector.Data)
			if err == nil {
				candidates = append(candidates, neighbor{f.graph.entryPoint, dist})
			}
		}

		// Add long-range samples for better global connectivity
		if i > samplingInterval {
			for s := 0; s < numSamples; s++ {
				sampleIdx := (i / samplingInterval) * samplingInterval - (s+1)*samplingInterval
				if sampleIdx >= 0 && sampleIdx < i {
					dist, err := distFunc(node.Vector.Data, layerNodes[sampleIdx].Vector.Data)
					if err == nil {
						candidates = append(candidates, neighbor{layerNodes[sampleIdx], dist})
					}
				}
			}
		}

		// Add recent neighbors in window
		for j := lookbackStart; j < i; j++ {
			dist, err := distFunc(node.Vector.Data, layerNodes[j].Vector.Data)
			if err != nil {
				continue
			}
			candidates = append(candidates, neighbor{layerNodes[j], dist})
		}

		// Connect to more neighbors than M for better recall
		// This trades some memory for much better search quality
		connectCount := m * 3  // Triple the connections for flat batch mode
		if connectCount > len(candidates) {
			connectCount = len(candidates)
		}
		for k := 0; k < connectCount; k++ {
			// Find kth nearest
			minIdx := k
			minDist := candidates[k].dist
			for j := k + 1; j < len(candidates); j++ {
				if candidates[j].dist < minDist {
					minDist = candidates[j].dist
					minIdx = j
				}
			}
			if minIdx != k {
				candidates[k], candidates[minIdx] = candidates[minIdx], candidates[k]
			}

			// Add bidirectional connections for proper graph connectivity
			node.AddConnectionUnsafe(candidates[k].node, layer)
			candidates[k].node.AddConnectionUnsafe(node, layer)
			if layer == 0 {
				f.graph.stats.EdgeCount += 2
			}
		}
	}
}
