// Package collection provides the main collection implementation for GoVecDB.
package collection

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
	"github.com/khambampati-subhash/govecdb/index"
	"github.com/khambampati-subhash/govecdb/store"
)

// VectorCollection implements the Collection interface by combining
// a VectorStore for data persistence and a VectorIndex for similarity search.
// It provides thread-safe operations and proper lifecycle management.
type VectorCollection struct {
	// Core components
	store api.VectorStore
	index api.VectorIndex

	// Configuration and metadata
	config   *api.CollectionConfig
	metadata *CollectionMetadata

	// Synchronization
	mu     sync.RWMutex
	closed bool

	// Background operations
	optimizeInterval time.Duration
	optimizeTicker   *time.Ticker
	stopOptimize     chan bool
}

// CollectionMetadata holds metadata about the collection
type CollectionMetadata struct {
	CreatedAt       time.Time
	UpdatedAt       time.Time
	LastOptimizedAt time.Time
	OperationCount  int64
	LastBackupAt    time.Time
	Version         string
}

// NewVectorCollection creates a new vector collection with the given configuration
func NewVectorCollection(config *api.CollectionConfig, storeConfig *store.StoreConfig) (*VectorCollection, error) {
	if config == nil {
		return nil, api.ErrInvalidConfig
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid collection config: %w", err)
	}

	// Create the memory store
	if storeConfig == nil {
		storeConfig = store.DefaultStoreConfig(config.Name)
	}
	memStore := store.NewMemoryStore(storeConfig)

	// Create the HNSW index configuration
	indexConfig := &index.Config{
		Dimension:      config.Dimension,
		Metric:         convertDistanceMetric(config.Metric),
		M:              config.M,
		EfConstruction: config.EfConstruction,
		MaxLayer:       config.MaxLayer,
		Seed:           config.Seed,
		ThreadSafe:     config.ThreadSafe,
	}

	// Create the HNSW index
	hnswIndex, err := index.NewHNSWIndex(indexConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create HNSW index: %w", err)
	}

	// Create the collection
	collection := &VectorCollection{
		store:  memStore,
		index:  newIndexAdapter(hnswIndex),
		config: config,
		metadata: &CollectionMetadata{
			CreatedAt:       time.Now(),
			UpdatedAt:       time.Now(),
			LastOptimizedAt: time.Now(),
			Version:         "1.0.0",
		},
		closed:           false,
		optimizeInterval: 30 * time.Minute, // Default optimization interval
	}

	// Start background optimization if enabled
	if config.ThreadSafe {
		collection.startBackgroundOptimization()
	}

	return collection, nil
}

// Add adds a vector to the collection
func (c *VectorCollection) Add(ctx context.Context, vector *api.Vector) error {
	if vector == nil {
		return api.ErrEmptyVector
	}

	if err := vector.Validate(); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return api.ErrClosed
	}

	// Validate dimension
	if vector.Dimension() != c.config.Dimension {
		return api.ErrDimensionMismatch
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	// Add to store first
	if err := c.store.Put(ctx, vector); err != nil {
		return fmt.Errorf("failed to store vector: %w", err)
	}

	// Add to index
	if err := c.index.Add(ctx, vector); err != nil {
		// Rollback store operation
		_ = c.store.Delete(ctx, vector.ID)
		return fmt.Errorf("failed to index vector: %w", err)
	}

	c.metadata.UpdatedAt = time.Now()
	c.metadata.OperationCount++

	return nil
}

// Get retrieves a vector by ID
func (c *VectorCollection) Get(ctx context.Context, id string) (*api.Vector, error) {
	if id == "" {
		return nil, api.ErrVectorNotFound
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	return c.store.Get(ctx, id)
}

// Delete removes a vector by ID
func (c *VectorCollection) Delete(ctx context.Context, id string) error {
	if id == "" {
		return api.ErrVectorNotFound
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	// Remove from index first
	if err := c.index.Remove(ctx, id); err != nil && err != api.ErrVectorNotFound {
		return fmt.Errorf("failed to remove from index: %w", err)
	}

	// Remove from store
	if err := c.store.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to remove from store: %w", err)
	}

	c.metadata.UpdatedAt = time.Now()
	c.metadata.OperationCount++

	return nil
}

// Search performs similarity search
func (c *VectorCollection) Search(ctx context.Context, req *api.SearchRequest) ([]*api.SearchResult, error) {
	if req == nil {
		return nil, api.ErrInvalidK
	}

	if err := req.Validate(); err != nil {
		return nil, err
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, api.ErrClosed
	}

	// Validate dimension
	if len(req.Vector) != c.config.Dimension {
		return nil, api.ErrDimensionMismatch
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	// Perform the search
	results, err := c.index.Search(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Apply additional filters if needed
	if req.Filter != nil {
		results = c.applyPostSearchFilters(results, req.Filter)
	}

	// Apply distance and score filters
	results = c.applyDistanceFilters(results, req.MaxDistance, req.MinScore)

	// If we don't need the full data, remove it to save bandwidth
	if !req.IncludeData {
		for _, result := range results {
			if result.Vector != nil {
				result.Vector.Data = nil
			}
		}
	}

	c.metadata.OperationCount++

	return results, nil
}

// AddBatch adds multiple vectors in a single operation
func (c *VectorCollection) AddBatch(ctx context.Context, vectors []*api.Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	// Validate all vectors first
	for _, vector := range vectors {
		if vector == nil {
			return api.ErrEmptyVector
		}
		if err := vector.Validate(); err != nil {
			return err
		}
		if vector.Dimension() != c.config.Dimension {
			return api.ErrDimensionMismatch
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	// Add to store first
	if err := c.store.PutBatch(ctx, vectors); err != nil {
		return fmt.Errorf("failed to store vectors: %w", err)
	}

	// Add to index
	if err := c.index.AddBatch(ctx, vectors); err != nil {
		// Rollback store operations
		ids := make([]string, len(vectors))
		for i, v := range vectors {
			ids[i] = v.ID
		}
		_ = c.store.DeleteBatch(ctx, ids)
		return fmt.Errorf("failed to index vectors: %w", err)
	}

	c.metadata.UpdatedAt = time.Now()
	c.metadata.OperationCount += int64(len(vectors))

	return nil
}

// GetBatch retrieves multiple vectors by their IDs
func (c *VectorCollection) GetBatch(ctx context.Context, ids []string) ([]*api.Vector, error) {
	if len(ids) == 0 {
		return []*api.Vector{}, nil
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	return c.store.GetBatch(ctx, ids)
}

// DeleteBatch removes multiple vectors by their IDs
func (c *VectorCollection) DeleteBatch(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	// Remove from index first
	if err := c.index.RemoveBatch(ctx, ids); err != nil {
		return fmt.Errorf("failed to remove from index: %w", err)
	}

	// Remove from store
	if err := c.store.DeleteBatch(ctx, ids); err != nil {
		return fmt.Errorf("failed to remove from store: %w", err)
	}

	c.metadata.UpdatedAt = time.Now()
	c.metadata.OperationCount += int64(len(ids))

	return nil
}

// Filter returns vectors that match the given filter expression
func (c *VectorCollection) Filter(ctx context.Context, filter api.FilterExpr, limit int) ([]*api.Vector, error) {
	if filter == nil {
		return c.List(ctx, limit, 0)
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	return c.store.Filter(ctx, filter, limit)
}

// Count returns the total number of vectors in the collection
func (c *VectorCollection) Count(ctx context.Context) (int64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return 0, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return 0, api.ErrContextCanceled
	}

	return c.store.Count(ctx)
}

// List returns vectors with pagination
func (c *VectorCollection) List(ctx context.Context, limit int, offset int) ([]*api.Vector, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	return c.store.List(ctx, limit, offset)
}

// Name returns the collection name
func (c *VectorCollection) Name() string {
	return c.config.Name
}

// Config returns the collection configuration
func (c *VectorCollection) Config() *api.CollectionConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Return a copy to prevent external modifications
	configCopy := *c.config
	return &configCopy
}

// Stats returns collection statistics
func (c *VectorCollection) Stats(ctx context.Context) (*api.CollectionStats, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	// Get store stats
	storeStats, err := c.store.Stats(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get store stats: %w", err)
	}

	// Get index stats
	indexStats := c.index.Stats()

	// Combine the stats
	stats := &api.CollectionStats{
		Name:        c.config.Name,
		VectorCount: storeStats.VectorCount,
		Dimension:   c.config.Dimension,
		IndexStats: api.IndexStats{
			NodeCount:   indexStats.NodeCount,
			EdgeCount:   indexStats.EdgeCount,
			MaxLayer:    indexStats.MaxLayer,
			AvgDegree:   indexStats.AvgDegree,
			SearchCount: indexStats.SearchCount,
			InsertCount: indexStats.InsertCount,
		},
		CreatedAt:   c.metadata.CreatedAt,
		UpdatedAt:   c.metadata.UpdatedAt,
		SizeBytes:   storeStats.SizeBytes,
		MemoryUsage: storeStats.MemoryUsage,
	}

	return stats, nil
}

// Optimize performs index optimization
func (c *VectorCollection) Optimize(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	if err := c.index.Optimize(ctx); err != nil {
		return fmt.Errorf("optimization failed: %w", err)
	}

	c.metadata.LastOptimizedAt = time.Now()
	c.metadata.UpdatedAt = time.Now()

	return nil
}

// Clear removes all vectors from the collection
func (c *VectorCollection) Clear(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	// Clear the store
	if err := c.store.Clear(ctx); err != nil {
		return fmt.Errorf("failed to clear store: %w", err)
	}

	// Recreate the index
	indexConfig := &index.Config{
		Dimension:      c.config.Dimension,
		Metric:         convertDistanceMetric(c.config.Metric),
		M:              c.config.M,
		EfConstruction: c.config.EfConstruction,
		MaxLayer:       c.config.MaxLayer,
		Seed:           c.config.Seed,
		ThreadSafe:     c.config.ThreadSafe,
	}

	c.index.Close()
	hnswIndex, err := index.NewHNSWIndex(indexConfig)
	if err != nil {
		return fmt.Errorf("failed to recreate HNSW index: %w", err)
	}
	c.index = newIndexAdapter(hnswIndex)

	c.metadata.UpdatedAt = time.Now()
	c.metadata.OperationCount = 0

	return nil
}

// Close closes the collection and releases resources
func (c *VectorCollection) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return api.ErrClosed
	}

	c.closed = true

	// Stop background optimization
	c.stopBackgroundOptimization()

	// Close store and index
	var storeErr, indexErr error
	if c.store != nil {
		storeErr = c.store.Close()
	}
	if c.index != nil {
		indexErr = c.index.Close()
	}

	// Return the first error encountered
	if storeErr != nil {
		return fmt.Errorf("failed to close store: %w", storeErr)
	}
	if indexErr != nil {
		return fmt.Errorf("failed to close index: %w", indexErr)
	}

	return nil
}

// Private helper methods

// startBackgroundOptimization starts the background optimization routine
func (c *VectorCollection) startBackgroundOptimization() {
	c.mu.Lock()
	if c.optimizeTicker != nil || c.stopOptimize != nil {
		c.mu.Unlock()
		return
	}
	c.optimizeTicker = time.NewTicker(c.optimizeInterval)
	c.stopOptimize = make(chan bool, 1)
	c.mu.Unlock()

	go func() {
		ticker := c.optimizeTicker // Capture ticker to avoid race condition
		defer func() {
			// Ensure ticker is stopped when goroutine exits
			c.mu.Lock()
			if c.optimizeTicker != nil {
				c.optimizeTicker.Stop()
				c.optimizeTicker = nil
			}
			c.mu.Unlock()
		}()

		if ticker == nil {
			return // Exit early if ticker is nil
		}

		for {
			select {
			case <-ticker.C:
				// Perform optimization if the collection has had significant activity
				if c.metadata.OperationCount > 1000 {
					ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
					_ = c.Optimize(ctx)
					cancel()
					c.metadata.OperationCount = 0
				}
			case <-c.stopOptimize:
				return
			}
		}
	}()
}

// stopBackgroundOptimization stops the background optimization routine
// Note: caller must hold the mutex
func (c *VectorCollection) stopBackgroundOptimization() {
	if c.stopOptimize != nil {
		select {
		case c.stopOptimize <- true:
		default:
		}
		close(c.stopOptimize)
		c.stopOptimize = nil
	}
	if c.optimizeTicker != nil {
		c.optimizeTicker.Stop()
		c.optimizeTicker = nil
	}
}

// applyPostSearchFilters applies metadata filters to search results
func (c *VectorCollection) applyPostSearchFilters(results []*api.SearchResult, filter api.FilterExpr) []*api.SearchResult {
	filtered := make([]*api.SearchResult, 0, len(results))

	for _, result := range results {
		if result.Vector != nil && filter.Evaluate(result.Vector.Metadata) {
			filtered = append(filtered, result)
		}
	}

	return filtered
}

// applyDistanceFilters applies distance and score filters to search results
func (c *VectorCollection) applyDistanceFilters(results []*api.SearchResult, maxDistance, minScore *float32) []*api.SearchResult {
	if maxDistance == nil && minScore == nil {
		return results
	}

	filtered := make([]*api.SearchResult, 0, len(results))

	for _, result := range results {
		include := true

		if maxDistance != nil && result.Distance > *maxDistance {
			include = false
		}

		if minScore != nil && result.Score < *minScore {
			include = false
		}

		if include {
			filtered = append(filtered, result)
		}
	}

	return filtered
}

// convertDistanceMetric converts API distance metric to index distance metric
func convertDistanceMetric(metric api.DistanceMetric) index.DistanceMetric {
	switch metric {
	case api.Cosine:
		return index.Cosine
	case api.Euclidean:
		return index.Euclidean
	case api.Manhattan:
		return index.Manhattan
	case api.DotProduct:
		return index.DotProduct
	default:
		return index.Cosine
	}
}

// IndexAdapter adapts the HNSW index to implement the VectorIndex interface
type IndexAdapter struct {
	index *index.HNSWIndex
}

// newIndexAdapter creates a new index adapter
func newIndexAdapter(hnswIndex *index.HNSWIndex) *IndexAdapter {
	return &IndexAdapter{index: hnswIndex}
}

// Add implements VectorIndex.Add
func (a *IndexAdapter) Add(ctx context.Context, vector *api.Vector) error {
	indexVector := &index.Vector{
		ID:   vector.ID,
		Data: vector.Data,
	}
	return a.index.Add(indexVector)
}

// Remove implements VectorIndex.Remove
func (a *IndexAdapter) Remove(ctx context.Context, id string) error {
	return a.index.Delete(id)
}

// Search implements VectorIndex.Search
func (a *IndexAdapter) Search(ctx context.Context, req *api.SearchRequest) ([]*api.SearchResult, error) {
	results, err := a.index.Search(req.Vector, req.K)
	if err != nil {
		return nil, err
	}

	// Convert results
	apiResults := make([]*api.SearchResult, len(results))
	for i, result := range results {
		// For HNSW index, Score contains the actual distance value
		// Score and Distance should be the same for distance-based metrics
		distance := result.Score
		score := result.Score

		// For cosine similarity, score is typically 1 - distance
		// But HNSW returns distances, so we keep it as is

		apiResults[i] = &api.SearchResult{
			Vector: &api.Vector{
				ID:       result.ID,
				Data:     result.Vector,
				Metadata: result.Metadata,
			},
			Score:    score,
			Distance: distance,
		}
	}

	return apiResults, nil
}

// AddBatch implements VectorIndex.AddBatch
func (a *IndexAdapter) AddBatch(ctx context.Context, vectors []*api.Vector) error {
	for _, vector := range vectors {
		indexVector := &index.Vector{
			ID:   vector.ID,
			Data: vector.Data,
		}
		if err := a.index.Add(indexVector); err != nil {
			return err
		}
	}
	return nil
}

// RemoveBatch implements VectorIndex.RemoveBatch
func (a *IndexAdapter) RemoveBatch(ctx context.Context, ids []string) error {
	for _, id := range ids {
		if err := a.index.Delete(id); err != nil {
			// Ignore not found errors
			continue
		}
	}
	return nil
}

// Size implements VectorIndex.Size
func (a *IndexAdapter) Size() int64 {
	return int64(a.index.Size())
}

// Stats implements VectorIndex.Stats
func (a *IndexAdapter) Stats() *api.IndexStats {
	stats := a.index.GetStats()
	return &api.IndexStats{
		NodeCount:   stats.NodeCount,
		EdgeCount:   stats.EdgeCount,
		MaxLayer:    stats.MaxLayer,
		AvgDegree:   stats.AvgDegree,
		SearchCount: stats.SearchCount,
		InsertCount: stats.InsertCount,
	}
}

// Optimize implements VectorIndex.Optimize
func (a *IndexAdapter) Optimize(ctx context.Context) error {
	// HNSW doesn't need explicit optimization, but we can trigger cleanup
	return nil
}

// Close implements VectorIndex.Close
func (a *IndexAdapter) Close() error {
	// HNSW index doesn't need explicit closing
	return nil
}
