package index

import (
	"fmt"
	"sync"
	"time"
)

// HNSWIndex implements the HNSW algorithm for approximate nearest neighbor search
type HNSWIndex struct {
	// Core components
	graph  *HNSWGraph
	config *Config

	// Thread safety
	mu sync.RWMutex

	// Statistics
	insertCount  int64
	searchCount  int64
	createdAt    time.Time
	lastUpdateAt time.Time
}

// NewHNSWIndex creates a new HNSW index with the given configuration
func NewHNSWIndex(config *Config) (*HNSWIndex, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	graph, err := NewHNSWGraph(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create graph: %w", err)
	}

	return &HNSWIndex{
		graph:        graph,
		config:       config,
		createdAt:    time.Now(),
		lastUpdateAt: time.Now(),
	}, nil
}

// Add inserts a vector into the index
func (idx *HNSWIndex) Add(vector *Vector) error {
	if vector == nil {
		return fmt.Errorf("vector cannot be nil")
	}

	if len(vector.Data) != idx.config.Dimension {
		return ErrDimensionMismatch
	}

	if vector.ID == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	if idx.config.ThreadSafe {
		idx.mu.Lock()
		defer idx.mu.Unlock()
	}

	err := idx.graph.Insert(vector)
	if err != nil {
		return fmt.Errorf("failed to insert vector: %w", err)
	}

	idx.insertCount++
	idx.lastUpdateAt = time.Now()

	return nil
}

// AddBatch inserts multiple vectors into the index
func (idx *HNSWIndex) AddBatch(vectors []*Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	for i, vector := range vectors {
		if err := idx.Add(vector); err != nil {
			return fmt.Errorf("failed to add vector at index %d: %w", i, err)
		}
	}

	return nil
}

// Search finds the k most similar vectors to the query
func (idx *HNSWIndex) Search(query []float32, k int) ([]*SearchResult, error) {
	return idx.SearchWithFilter(query, k, nil)
}

// SearchWithFilter finds vectors with metadata filtering
func (idx *HNSWIndex) SearchWithFilter(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	if len(query) != idx.config.Dimension {
		return nil, ErrDimensionMismatch
	}

	if k <= 0 {
		return nil, ErrInvalidK
	}

	if idx.config.ThreadSafe {
		idx.mu.RLock()
		defer idx.mu.RUnlock()
	}

	results, err := idx.graph.Search(query, k, filter)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	idx.searchCount++

	return results, nil
}

// Get retrieves a vector by ID
func (idx *HNSWIndex) Get(id string) (*Vector, error) {
	if id == "" {
		return nil, fmt.Errorf("id cannot be empty")
	}

	if idx.config.ThreadSafe {
		idx.mu.RLock()
		defer idx.mu.RUnlock()
	}

	vector, err := idx.graph.Get(id)
	if err != nil {
		return nil, err
	}

	return vector, nil
}

// Delete removes a vector from the index
func (idx *HNSWIndex) Delete(id string) error {
	if id == "" {
		return fmt.Errorf("id cannot be empty")
	}

	if idx.config.ThreadSafe {
		idx.mu.Lock()
		defer idx.mu.Unlock()
	}

	err := idx.graph.Delete(id)
	if err != nil {
		return err
	}

	idx.lastUpdateAt = time.Now()

	return nil
}

// Size returns the number of vectors in the index
func (idx *HNSWIndex) Size() int {
	if idx.config.ThreadSafe {
		idx.mu.RLock()
		defer idx.mu.RUnlock()
	}

	return idx.graph.Size()
}

// Dimension returns the dimension of vectors in the index
func (idx *HNSWIndex) Dimension() int {
	return idx.config.Dimension
}

// GetConfig returns a copy of the index configuration
func (idx *HNSWIndex) GetConfig() *Config {
	configCopy := *idx.config
	return &configCopy
}

// GetStats returns detailed statistics about the index
func (idx *HNSWIndex) GetStats() *IndexStats {
	if idx.config.ThreadSafe {
		idx.mu.RLock()
		defer idx.mu.RUnlock()
	}

	graphStats := idx.graph.GetStats()

	return &IndexStats{
		NodeCount:    graphStats.NodeCount,
		EdgeCount:    graphStats.EdgeCount,
		MaxLayer:     graphStats.MaxLayer,
		AvgDegree:    graphStats.AvgDegree,
		DeletedCount: graphStats.DeletedCount,
		InsertCount:  idx.insertCount,
		SearchCount:  idx.searchCount,
		Dimension:    idx.config.Dimension,
		Metric:       idx.config.Metric.String(),
		CreatedAt:    idx.createdAt,
		LastUpdateAt: idx.lastUpdateAt,
	}
}

// Close cleans up resources used by the index
func (idx *HNSWIndex) Close() error {
	if idx.config.ThreadSafe {
		idx.mu.Lock()
		defer idx.mu.Unlock()
	}

	// Clean up any resources if needed
	// For now, this is a no-op since we don't have external resources

	return nil
}

// Optimize performs maintenance operations on the index
func (idx *HNSWIndex) Optimize() error {
	if idx.config.ThreadSafe {
		idx.mu.Lock()
		defer idx.mu.Unlock()
	}

	// This could include:
	// - Cleaning up deleted nodes
	// - Rebalancing connections
	// - Compacting data structures
	// For now, this is a placeholder

	return nil
}

// SetEf sets the ef parameter for search operations
func (idx *HNSWIndex) SetEf(ef int) error {
	if ef <= 0 {
		return fmt.Errorf("ef must be positive")
	}

	if idx.config.ThreadSafe {
		idx.mu.Lock()
		defer idx.mu.Unlock()
	}

	idx.config.EfConstruction = ef

	return nil
}

// GetEf returns the current ef parameter
func (idx *HNSWIndex) GetEf() int {
	if idx.config.ThreadSafe {
		idx.mu.RLock()
		defer idx.mu.RUnlock()
	}

	return idx.config.EfConstruction
}

// IndexStats contains detailed statistics about the index
type IndexStats struct {
	NodeCount    int64     `json:"node_count"`
	EdgeCount    int64     `json:"edge_count"`
	MaxLayer     int       `json:"max_layer"`
	AvgDegree    float64   `json:"avg_degree"`
	DeletedCount int64     `json:"deleted_count"`
	InsertCount  int64     `json:"insert_count"`
	SearchCount  int64     `json:"search_count"`
	Dimension    int       `json:"dimension"`
	Metric       string    `json:"metric"`
	CreatedAt    time.Time `json:"created_at"`
	LastUpdateAt time.Time `json:"last_update_at"`
}

// String returns a string representation of the index statistics
func (s *IndexStats) String() string {
	return fmt.Sprintf(
		"IndexStats{Nodes:%d, Edges:%d, MaxLayer:%d, AvgDegree:%.2f, Searches:%d, Inserts:%d, Dimension:%d, Metric:%s}",
		s.NodeCount, s.EdgeCount, s.MaxLayer, s.AvgDegree, s.SearchCount, s.InsertCount, s.Dimension, s.Metric,
	)
}

// IsEmpty returns true if the index contains no vectors
func (idx *HNSWIndex) IsEmpty() bool {
	return idx.Size() == 0
}

// Contains checks if a vector with the given ID exists in the index
func (idx *HNSWIndex) Contains(id string) bool {
	_, err := idx.Get(id)
	return err == nil
}

// Clear removes all vectors from the index (creates a new empty graph)
func (idx *HNSWIndex) Clear() error {
	if idx.config.ThreadSafe {
		idx.mu.Lock()
		defer idx.mu.Unlock()
	}

	graph, err := NewHNSWGraph(idx.config)
	if err != nil {
		return fmt.Errorf("failed to create new graph: %w", err)
	}

	idx.graph = graph
	idx.insertCount = 0
	idx.searchCount = 0
	idx.lastUpdateAt = time.Now()

	return nil
}
