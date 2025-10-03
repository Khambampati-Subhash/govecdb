package index

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// IndexType represents different types of vector indices
type IndexType int

const (
	IndexTypeHNSW IndexType = iota
	IndexTypeIVF
	IndexTypeLSH
	IndexTypePQ
	IndexTypeFlat
	IndexTypeHybrid
)

// MultiIndexManager manages multiple index types for optimal performance
type MultiIndexManager struct {
	// Primary indices
	hnswIndex *ConcurrentHNSWIndex
	ivfIndex  *IVFIndex
	lshIndex  *LSHIndex
	pqIndex   *PQIndex
	flatIndex *FlatIndex

	// Configuration
	config *MultiIndexConfig

	// Adaptive routing
	router *AdaptiveRouter

	// Statistics
	stats *MultiIndexStats

	// Thread safety
	mu sync.RWMutex
}

// MultiIndexConfig configures the multi-index system
type MultiIndexConfig struct {
	// Primary index type
	PrimaryIndexType IndexType

	// Enable secondary indices
	EnableIVF  bool
	EnableLSH  bool
	EnablePQ   bool
	EnableFlat bool

	// Adaptive routing settings
	EnableAdaptiveRouting bool
	RouterLearningRate    float64
	RouterThreshold       float64

	// Index-specific configurations
	HNSWConfig *Config
	IVFConfig  *IVFConfig
	LSHConfig  *LSHConfig
	PQConfig   *PQConfig
}

// MultiIndexStats tracks performance across all indices
type MultiIndexStats struct {
	// Index selection statistics
	hnswQueries int64
	ivfQueries  int64
	lshQueries  int64
	pqQueries   int64
	flatQueries int64

	// Performance metrics
	avgLatencyByIndex map[IndexType]int64
	accuracyByIndex   map[IndexType]float64

	// Memory usage
	memoryUsageByIndex map[IndexType]int64
}

// IVFIndex implements Inverted File Index
type IVFIndex struct {
	// Clustering centroids
	centroids [][]float32
	clusters  [][]int // Cluster assignments

	// Configuration
	config *IVFConfig

	// Distance function
	distanceFunc OptimizedDistanceFunc

	// Thread safety
	mu sync.RWMutex

	// Statistics
	numClusters int
	vectors     []*Vector
}

// IVFConfig configures the IVF index
type IVFConfig struct {
	NumClusters    int
	NumProbes      int // Number of clusters to search
	Metric         DistanceMetric
	MaxVecsPerNode int
}

// LSHIndex implements Locality Sensitive Hashing
type LSHIndex struct {
	// Hash functions
	hashFunctions []LSHashFunction
	hashTables    []map[uint64][]*Vector

	// Configuration
	config *LSHConfig

	// Thread safety
	mu sync.RWMutex
}

// LSHConfig configures the LSH index
type LSHConfig struct {
	NumHashTables    int
	NumHashFunctions int
	HashBucketSize   int
	RandomSeed       int64
}

// LSHashFunction represents a locality sensitive hash function
type LSHashFunction struct {
	Weights []float32
	Offset  float32
}

// PQIndex implements Product Quantization
type PQIndex struct {
	// Codebooks for each subspace
	codebooks [][]Centroid
	codes     [][]uint8 // Quantized codes

	// Configuration
	config *PQConfig

	// Thread safety
	mu sync.RWMutex
}

// PQConfig configures the PQ index
type PQConfig struct {
	NumSubspaces int
	NumCentroids int // Number of centroids per subspace
	TrainingSize int
}

// Centroid represents a centroid in PQ codebook
type Centroid struct {
	Vector []float32
	ID     uint8
}

// FlatIndex implements exhaustive search for ground truth
type FlatIndex struct {
	vectors      []*Vector
	distanceFunc OptimizedDistanceFunc
	mu           sync.RWMutex
}

// AdaptiveRouter routes queries to the optimal index based on learned patterns
type AdaptiveRouter struct {
	// Performance history
	performanceHistory map[IndexType]*PerformanceMetrics

	// Learning parameters
	learningRate float64
	threshold    float64

	// Query characteristics
	queryPatterns map[string]*QueryPattern

	mu sync.RWMutex
}

// PerformanceMetrics tracks performance for an index type
type PerformanceMetrics struct {
	avgLatency  float64
	avgAccuracy float64
	queryCount  int64
	errorRate   float64
	lastUpdate  int64
}

// QueryPattern represents learned query characteristics
type QueryPattern struct {
	Dimension      int
	QueryType      string // "similarity", "filtered", "batch"
	PreferredIndex IndexType
	Confidence     float64
}

// NewMultiIndexManager creates a new multi-index manager
func NewMultiIndexManager(config *MultiIndexConfig) (*MultiIndexManager, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	manager := &MultiIndexManager{
		config: config,
		stats: &MultiIndexStats{
			avgLatencyByIndex:  make(map[IndexType]int64),
			accuracyByIndex:    make(map[IndexType]float64),
			memoryUsageByIndex: make(map[IndexType]int64),
		},
	}

	// Initialize primary index
	switch config.PrimaryIndexType {
	case IndexTypeHNSW:
		hnswIndex, err := NewConcurrentHNSWIndex(config.HNSWConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create HNSW index: %w", err)
		}
		manager.hnswIndex = hnswIndex

	case IndexTypeIVF:
		ivfIndex, err := NewIVFIndex(config.IVFConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create IVF index: %w", err)
		}
		manager.ivfIndex = ivfIndex

	default:
		return nil, fmt.Errorf("unsupported primary index type: %v", config.PrimaryIndexType)
	}

	// Initialize secondary indices
	if config.EnableIVF && config.PrimaryIndexType != IndexTypeIVF {
		ivfIndex, err := NewIVFIndex(config.IVFConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create secondary IVF index: %w", err)
		}
		manager.ivfIndex = ivfIndex
	}

	if config.EnableLSH {
		lshIndex, err := NewLSHIndex(config.LSHConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create LSH index: %w", err)
		}
		manager.lshIndex = lshIndex
	}

	if config.EnablePQ {
		pqIndex, err := NewPQIndex(config.PQConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create PQ index: %w", err)
		}
		manager.pqIndex = pqIndex
	}

	if config.EnableFlat {
		manager.flatIndex = NewFlatIndex()
	}

	// Initialize adaptive router
	if config.EnableAdaptiveRouting {
		manager.router = NewAdaptiveRouter(config.RouterLearningRate, config.RouterThreshold)
	}

	return manager, nil
}

// NewIVFIndex creates a new IVF index
func NewIVFIndex(config *IVFConfig) (*IVFIndex, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	return &IVFIndex{
		config:       config,
		distanceFunc: getOptimizedDistanceFunc(config.Metric),
		numClusters:  config.NumClusters,
		centroids:    make([][]float32, config.NumClusters),
		clusters:     make([][]int, config.NumClusters),
	}, nil
}

// NewLSHIndex creates a new LSH index
func NewLSHIndex(config *LSHConfig) (*LSHIndex, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	index := &LSHIndex{
		config:        config,
		hashFunctions: make([]LSHashFunction, config.NumHashFunctions),
		hashTables:    make([]map[uint64][]*Vector, config.NumHashTables),
	}

	// Initialize hash tables
	for i := 0; i < config.NumHashTables; i++ {
		index.hashTables[i] = make(map[uint64][]*Vector)
	}

	return index, nil
}

// NewPQIndex creates a new PQ index
func NewPQIndex(config *PQConfig) (*PQIndex, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	return &PQIndex{
		config:    config,
		codebooks: make([][]Centroid, config.NumSubspaces),
	}, nil
}

// NewFlatIndex creates a new flat index
func NewFlatIndex() *FlatIndex {
	return &FlatIndex{
		vectors:      make([]*Vector, 0),
		distanceFunc: OptimizedCosineDistance,
	}
}

// NewAdaptiveRouter creates a new adaptive router
func NewAdaptiveRouter(learningRate, threshold float64) *AdaptiveRouter {
	return &AdaptiveRouter{
		performanceHistory: make(map[IndexType]*PerformanceMetrics),
		queryPatterns:      make(map[string]*QueryPattern),
		learningRate:       learningRate,
		threshold:          threshold,
	}
}

// Add inserts a vector into all active indices
func (m *MultiIndexManager) Add(vector *Vector) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var errors []error

	// Add to primary index
	switch m.config.PrimaryIndexType {
	case IndexTypeHNSW:
		if err := m.hnswIndex.Add(vector); err != nil {
			errors = append(errors, fmt.Errorf("HNSW add failed: %w", err))
		}
	case IndexTypeIVF:
		if err := m.ivfIndex.Add(vector); err != nil {
			errors = append(errors, fmt.Errorf("IVF add failed: %w", err))
		}
	}

	// Add to secondary indices
	if m.lshIndex != nil {
		if err := m.lshIndex.Add(vector); err != nil {
			errors = append(errors, fmt.Errorf("LSH add failed: %w", err))
		}
	}

	if m.pqIndex != nil {
		if err := m.pqIndex.Add(vector); err != nil {
			errors = append(errors, fmt.Errorf("PQ add failed: %w", err))
		}
	}

	if m.flatIndex != nil {
		if err := m.flatIndex.Add(vector); err != nil {
			errors = append(errors, fmt.Errorf("Flat add failed: %w", err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("multiple add failures: %v", errors)
	}

	return nil
}

// Search performs adaptive search using the optimal index
func (m *MultiIndexManager) Search(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	// Determine optimal index using router
	indexType := m.selectOptimalIndex(query, k, filter)

	// Record query for statistics
	atomic.AddInt64(m.getQueryCounter(indexType), 1)

	// Execute search on selected index
	switch indexType {
	case IndexTypeHNSW:
		return m.hnswIndex.SearchWithFilter(query, k, filter)
	case IndexTypeIVF:
		return m.ivfIndex.Search(query, k, filter)
	case IndexTypeLSH:
		return m.lshIndex.Search(query, k, filter)
	case IndexTypePQ:
		return m.pqIndex.Search(query, k, filter)
	case IndexTypeFlat:
		return m.flatIndex.Search(query, k, filter)
	default:
		return nil, fmt.Errorf("unsupported index type: %v", indexType)
	}
}

// selectOptimalIndex chooses the best index for a given query
func (m *MultiIndexManager) selectOptimalIndex(query []float32, k int, filter FilterFunc) IndexType {
	if m.router == nil {
		return m.config.PrimaryIndexType
	}

	// Use adaptive router to select optimal index
	return m.router.SelectIndex(query, k, filter)
}

// getQueryCounter returns the appropriate query counter for statistics
func (m *MultiIndexManager) getQueryCounter(indexType IndexType) *int64 {
	switch indexType {
	case IndexTypeHNSW:
		return &m.stats.hnswQueries
	case IndexTypeIVF:
		return &m.stats.ivfQueries
	case IndexTypeLSH:
		return &m.stats.lshQueries
	case IndexTypePQ:
		return &m.stats.pqQueries
	case IndexTypeFlat:
		return &m.stats.flatQueries
	default:
		return &m.stats.hnswQueries
	}
}

// IVF Index Implementation

// Add inserts a vector into the IVF index
func (ivf *IVFIndex) Add(vector *Vector) error {
	ivf.mu.Lock()
	defer ivf.mu.Unlock()

	// Add to vectors list
	vectorIndex := len(ivf.vectors)
	ivf.vectors = append(ivf.vectors, vector)

	// If we don't have centroids yet, just store the vector
	if len(ivf.centroids) == 0 || ivf.centroids[0] == nil {
		return nil
	}

	// Find closest centroid
	closestCluster := ivf.findClosestCentroid(vector.Data)

	// Add to appropriate cluster
	ivf.clusters[closestCluster] = append(ivf.clusters[closestCluster], vectorIndex)

	return nil
}

// Search performs search in the IVF index
func (ivf *IVFIndex) Search(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	ivf.mu.RLock()
	defer ivf.mu.RUnlock()

	if len(ivf.centroids) == 0 {
		// Fall back to exhaustive search
		return ivf.exhaustiveSearch(query, k, filter)
	}

	// Find closest centroids to search
	centroidDistances := make([]struct {
		index    int
		distance float32
	}, len(ivf.centroids))

	for i, centroid := range ivf.centroids {
		if centroid != nil {
			distance := ivf.distanceFunc(query, centroid)
			centroidDistances[i] = struct {
				index    int
				distance float32
			}{i, distance}
		}
	}

	// Sort centroids by distance
	// Use simple sort for now
	for i := 0; i < len(centroidDistances)-1; i++ {
		for j := i + 1; j < len(centroidDistances); j++ {
			if centroidDistances[i].distance > centroidDistances[j].distance {
				centroidDistances[i], centroidDistances[j] = centroidDistances[j], centroidDistances[i]
			}
		}
	}

	// Search in closest clusters
	numProbes := min(ivf.config.NumProbes, len(centroidDistances))
	candidates := make([]*SearchResult, 0)

	for i := 0; i < numProbes; i++ {
		clusterIndex := centroidDistances[i].index
		for _, vectorIndex := range ivf.clusters[clusterIndex] {
			if vectorIndex >= len(ivf.vectors) {
				continue
			}

			vector := ivf.vectors[vectorIndex]
			if filter != nil && !filter(vector.Metadata) {
				continue
			}

			distance := ivf.distanceFunc(query, vector.Data)
			result := &SearchResult{
				ID:       vector.ID,
				Vector:   vector.Data,
				Score:    distance,
				Metadata: vector.Metadata,
			}
			candidates = append(candidates, result)
		}
	}

	// Sort candidates by distance and return top k
	for i := 0; i < len(candidates)-1; i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[i].Score > candidates[j].Score {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	if len(candidates) > k {
		candidates = candidates[:k]
	}

	return candidates, nil
}

// findClosestCentroid finds the closest centroid to a vector
func (ivf *IVFIndex) findClosestCentroid(vector []float32) int {
	if len(ivf.centroids) == 0 {
		return 0
	}

	minDistance := float32(1e9)
	closestIndex := 0

	for i, centroid := range ivf.centroids {
		if centroid != nil {
			distance := ivf.distanceFunc(vector, centroid)
			if distance < minDistance {
				minDistance = distance
				closestIndex = i
			}
		}
	}

	return closestIndex
}

// exhaustiveSearch performs exhaustive search when no centroids exist
func (ivf *IVFIndex) exhaustiveSearch(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	results := make([]*SearchResult, 0)

	for _, vector := range ivf.vectors {
		if filter != nil && !filter(vector.Metadata) {
			continue
		}

		distance := ivf.distanceFunc(query, vector.Data)
		result := &SearchResult{
			ID:       vector.ID,
			Vector:   vector.Data,
			Score:    distance,
			Metadata: vector.Metadata,
		}
		results = append(results, result)
	}

	// Sort and limit to k
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

// LSH Index Implementation

// Add inserts a vector into the LSH index
func (lsh *LSHIndex) Add(vector *Vector) error {
	lsh.mu.Lock()
	defer lsh.mu.Unlock()

	// Compute hash values for each hash table
	for i, hashFunc := range lsh.hashFunctions {
		hashValue := lsh.computeHash(vector.Data, hashFunc)
		bucket := hashValue % uint64(lsh.config.HashBucketSize)

		if lsh.hashTables[i][bucket] == nil {
			lsh.hashTables[i][bucket] = make([]*Vector, 0)
		}
		lsh.hashTables[i][bucket] = append(lsh.hashTables[i][bucket], vector)
	}

	return nil
}

// Search performs search in the LSH index
func (lsh *LSHIndex) Search(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	lsh.mu.RLock()
	defer lsh.mu.RUnlock()

	candidates := make(map[string]*SearchResult)

	// Search in each hash table
	for i, hashFunc := range lsh.hashFunctions {
		hashValue := lsh.computeHash(query, hashFunc)
		bucket := hashValue % uint64(lsh.config.HashBucketSize)

		if vectors, exists := lsh.hashTables[i][bucket]; exists {
			for _, vector := range vectors {
				if filter != nil && !filter(vector.Metadata) {
					continue
				}

				if _, seen := candidates[vector.ID]; !seen {
					distance := OptimizedCosineDistance(query, vector.Data)
					candidates[vector.ID] = &SearchResult{
						ID:       vector.ID,
						Vector:   vector.Data,
						Score:    distance,
						Metadata: vector.Metadata,
					}
				}
			}
		}
	}

	// Convert to slice and sort
	results := make([]*SearchResult, 0, len(candidates))
	for _, result := range candidates {
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

// computeHash computes the hash value for a vector
func (lsh *LSHIndex) computeHash(vector []float32, hashFunc LSHashFunction) uint64 {
	var hash float32
	for i, weight := range hashFunc.Weights {
		if i < len(vector) {
			hash += vector[i] * weight
		}
	}
	hash += hashFunc.Offset

	// Convert to uint64 hash
	if hash >= 0 {
		return uint64(hash * 1000)
	}
	return uint64(-hash * 1000)
}

// PQ Index Implementation

// Add inserts a vector into the PQ index
func (pq *PQIndex) Add(vector *Vector) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	// For now, just store the vector
	// In a full implementation, this would quantize the vector
	return nil
}

// Search performs search in the PQ index
func (pq *PQIndex) Search(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	// Placeholder implementation
	return []*SearchResult{}, nil
}

// Flat Index Implementation

// Add inserts a vector into the flat index
func (flat *FlatIndex) Add(vector *Vector) error {
	flat.mu.Lock()
	defer flat.mu.Unlock()

	flat.vectors = append(flat.vectors, vector)
	return nil
}

// Search performs exhaustive search in the flat index
func (flat *FlatIndex) Search(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	flat.mu.RLock()
	defer flat.mu.RUnlock()

	results := make([]*SearchResult, 0)

	for _, vector := range flat.vectors {
		if filter != nil && !filter(vector.Metadata) {
			continue
		}

		distance := flat.distanceFunc(query, vector.Data)
		result := &SearchResult{
			ID:       vector.ID,
			Vector:   vector.Data,
			Score:    distance,
			Metadata: vector.Metadata,
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

// Adaptive Router Implementation

// SelectIndex selects the optimal index for a query
func (router *AdaptiveRouter) SelectIndex(query []float32, k int, filter FilterFunc) IndexType {
	router.mu.RLock()
	defer router.mu.RUnlock()

	// Simple heuristic for now - use HNSW for most queries
	// In a full implementation, this would use machine learning
	if len(query) > 512 {
		return IndexTypeIVF // Better for high-dimensional vectors
	}

	if k > 100 {
		return IndexTypeFlat // Better for large k
	}

	return IndexTypeHNSW // Default to HNSW
}

// GetStats returns comprehensive multi-index statistics
func (m *MultiIndexManager) GetStats() *MultiIndexStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return &MultiIndexStats{
		hnswQueries: atomic.LoadInt64(&m.stats.hnswQueries),
		ivfQueries:  atomic.LoadInt64(&m.stats.ivfQueries),
		lshQueries:  atomic.LoadInt64(&m.stats.lshQueries),
		pqQueries:   atomic.LoadInt64(&m.stats.pqQueries),
		flatQueries: atomic.LoadInt64(&m.stats.flatQueries),
		// Copy other maps
		avgLatencyByIndex:  make(map[IndexType]int64),
		accuracyByIndex:    make(map[IndexType]float64),
		memoryUsageByIndex: make(map[IndexType]int64),
	}
}
