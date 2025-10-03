// Package collection provides enhanced collection implementation with filtering and segmented storage.
package collection

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
	"github.com/khambampati-subhash/govecdb/filter"
	"github.com/khambampati-subhash/govecdb/index"
	"github.com/khambampati-subhash/govecdb/segment"
)

// EnhancedVectorCollection implements Collection interface with advanced filtering and segmented storage
type EnhancedVectorCollection struct {
	// Core components
	segments     segment.SegmentManager
	vectorIndex  api.VectorIndex
	filterEngine filter.FilterEngine

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

// NewEnhancedVectorCollection creates a new enhanced vector collection
func NewEnhancedVectorCollection(config *api.CollectionConfig) (*EnhancedVectorCollection, error) {
	if config == nil {
		return nil, api.ErrInvalidConfig
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid collection config: %w", err)
	}

	// Create segment manager
	segmentConfig := segment.DefaultSegmentManagerConfig()
	segmentConfig.DefaultSegmentConfig.MaxVectors = 100000      // 100K vectors per segment
	segmentConfig.DefaultSegmentConfig.MaxSizeBytes = 500 << 20 // 500MB per segment

	segmentManager, err := segment.NewConcurrentSegmentManager(segmentConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create segment manager: %w", err)
	}

	// Create HNSW index for vector similarity search
	indexConfig := &index.Config{
		Dimension:      config.Dimension,
		Metric:         convertDistanceMetric(config.Metric),
		M:              config.M,
		EfConstruction: config.EfConstruction,
		MaxLayer:       config.MaxLayer,
		Seed:           config.Seed,
		ThreadSafe:     config.ThreadSafe,
	}

	hnswIndex, err := index.NewHNSWIndex(indexConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create HNSW index: %w", err)
	}

	// Create filter engine with both inverted and numeric indexes
	invertedConfig := &filter.IndexConfig{
		Field: "category",
		Type:  filter.InvertedIndex,
		Options: map[string]interface{}{
			"cache_size":     50 << 20,
			"max_key_length": 256,
			"bloom_enabled":  true,
		},
		ThreadSafe: true,
	}

	numericConfig := &filter.IndexConfig{
		Field: "value",
		Type:  filter.NumericIndex,
		Options: map[string]interface{}{
			"cache_size":     50 << 20,
			"max_key_length": 256,
		},
		ThreadSafe: true,
	}

	invertedIndex, err := filter.NewInvertedMetadataIndex(invertedConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create inverted index: %w", err)
	}

	numericIndex, err := filter.NewNumericMetadataIndex(numericConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create numeric index: %w", err)
	}

	filterEngine := filter.NewHybridFilterEngine(invertedIndex, numericIndex)

	// Create the enhanced collection
	collection := &EnhancedVectorCollection{
		segments:     segmentManager,
		vectorIndex:  newIndexAdapter(hnswIndex),
		filterEngine: filterEngine,
		config:       config,
		metadata: &CollectionMetadata{
			CreatedAt:       time.Now(),
			UpdatedAt:       time.Now(),
			LastOptimizedAt: time.Now(),
			Version:         "2.0.0",
		},
		closed:           false,
		optimizeInterval: 30 * time.Minute,
	}

	// Start segment manager
	ctx := context.Background()
	if err := segmentManager.Start(ctx); err != nil {
		return nil, fmt.Errorf("failed to start segment manager: %w", err)
	}

	// Start background optimization if enabled
	if config.ThreadSafe {
		collection.startBackgroundOptimization()
	}

	return collection, nil
}

// Add adds a vector to the collection
func (c *EnhancedVectorCollection) Add(ctx context.Context, vector *api.Vector) error {
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

	// Add to segment storage first
	if err := c.segments.Put(ctx, vector); err != nil {
		return fmt.Errorf("failed to store vector in segments: %w", err)
	}

	// Add to vector index for similarity search
	if err := c.vectorIndex.Add(ctx, vector); err != nil {
		// Rollback segment operation
		_ = c.segments.Delete(ctx, vector.ID)
		return fmt.Errorf("failed to index vector: %w", err)
	}

	// Add to filter indexes for metadata search
	if len(vector.Metadata) > 0 {
		if err := c.filterEngine.AddDocument(vector.ID, vector.Metadata); err != nil {
			// Rollback previous operations
			_ = c.vectorIndex.Remove(ctx, vector.ID)
			_ = c.segments.Delete(ctx, vector.ID)
			return fmt.Errorf("failed to index metadata: %w", err)
		}
	}

	c.metadata.UpdatedAt = time.Now()
	c.metadata.OperationCount++

	return nil
}

// Get retrieves a vector by ID
func (c *EnhancedVectorCollection) Get(ctx context.Context, id string) (*api.Vector, error) {
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

	return c.segments.Get(ctx, id)
}

// Delete removes a vector by ID
func (c *EnhancedVectorCollection) Delete(ctx context.Context, id string) error {
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

	// Get the vector first to remove from filter indexes
	vector, err := c.segments.Get(ctx, id)
	if err != nil {
		return err
	}

	// Remove from filter indexes
	if len(vector.Metadata) > 0 {
		if err := c.filterEngine.RemoveDocument(id, vector.Metadata); err != nil {
			// Log warning but continue with deletion
			fmt.Printf("Warning: failed to remove from filter indexes: %v\n", err)
		}
	}

	// Remove from vector index
	if err := c.vectorIndex.Remove(ctx, id); err != nil && err != api.ErrVectorNotFound {
		// Log warning but continue with deletion
		fmt.Printf("Warning: failed to remove from vector index: %v\n", err)
	}

	// Remove from segments
	if err := c.segments.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to remove from segments: %w", err)
	}

	c.metadata.UpdatedAt = time.Now()
	c.metadata.OperationCount++

	return nil
}

// Search performs hybrid similarity and metadata search
func (c *EnhancedVectorCollection) Search(ctx context.Context, req *api.SearchRequest) ([]*api.SearchResult, error) {
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

	var results []*api.SearchResult
	var err error

	// If we have a filter, use hybrid search
	if req.Filter != nil {
		results, err = c.hybridSearch(ctx, req)
	} else {
		// Pure vector similarity search
		results, err = c.vectorIndex.Search(ctx, req)
	}

	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
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

// hybridSearch performs combined vector similarity and metadata filtering
func (c *EnhancedVectorCollection) hybridSearch(ctx context.Context, req *api.SearchRequest) ([]*api.SearchResult, error) {
	// First, filter by metadata to get candidate vector IDs
	filterResult, err := c.filterEngine.Filter(ctx, req.Filter)
	if err != nil {
		return nil, fmt.Errorf("metadata filtering failed: %w", err)
	}

	if len(filterResult) == 0 {
		return []*api.SearchResult{}, nil
	}

	// Get vectors for the filtered IDs
	candidateVectors, err := c.segments.GetBatch(ctx, filterResult)
	if err != nil {
		return nil, fmt.Errorf("failed to get candidate vectors: %w", err)
	}

	if len(candidateVectors) == 0 {
		return []*api.SearchResult{}, nil
	}

	// Perform similarity search on candidate vectors
	results := make([]*api.SearchResult, 0, len(candidateVectors))

	for _, vector := range candidateVectors {
		// Calculate similarity score
		score := c.calculateSimilarity(req.Vector, vector.Data, c.config.Metric)
		distance := c.calculateDistance(score, c.config.Metric)

		results = append(results, &api.SearchResult{
			Vector:   vector,
			Score:    score,
			Distance: distance,
		})
	}

	// Sort by score (descending)
	c.sortResultsByScore(results)

	// Limit to requested K
	if len(results) > req.K {
		results = results[:req.K]
	}

	return results, nil
}

// Filter returns vectors that match the given filter expression
func (c *EnhancedVectorCollection) Filter(ctx context.Context, filter api.FilterExpr, limit int) ([]*api.Vector, error) {
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

	// Use filter engine for efficient metadata filtering
	filterResult, err := c.filterEngine.Filter(ctx, filter)
	if err != nil {
		return nil, fmt.Errorf("filtering failed: %w", err)
	}

	// Limit the number of IDs if requested
	vectorIDs := filterResult
	if limit > 0 && len(vectorIDs) > limit {
		vectorIDs = vectorIDs[:limit]
	}

	// Get the actual vectors
	return c.segments.GetBatch(ctx, vectorIDs)
}

// AddBatch adds multiple vectors in a single operation
func (c *EnhancedVectorCollection) AddBatch(ctx context.Context, vectors []*api.Vector) error {
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

	// Add to segments first
	if err := c.segments.PutBatch(ctx, vectors); err != nil {
		return fmt.Errorf("failed to store vectors in segments: %w", err)
	}

	// Add to vector index
	if err := c.vectorIndex.AddBatch(ctx, vectors); err != nil {
		// Rollback segment operations
		ids := make([]string, len(vectors))
		for i, v := range vectors {
			ids[i] = v.ID
		}
		_ = c.segments.DeleteBatch(ctx, ids)
		return fmt.Errorf("failed to index vectors: %w", err)
	}

	// Add to filter indexes
	for _, vector := range vectors {
		if len(vector.Metadata) > 0 {
			if err := c.filterEngine.AddDocument(vector.ID, vector.Metadata); err != nil {
				// Log warning but continue with batch
				fmt.Printf("Warning: failed to index metadata for vector %s: %v\n", vector.ID, err)
			}
		}
	}

	c.metadata.UpdatedAt = time.Now()
	c.metadata.OperationCount += int64(len(vectors))

	return nil
}

// GetBatch retrieves multiple vectors by their IDs
func (c *EnhancedVectorCollection) GetBatch(ctx context.Context, ids []string) ([]*api.Vector, error) {
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

	return c.segments.GetBatch(ctx, ids)
}

// DeleteBatch removes multiple vectors by their IDs
func (c *EnhancedVectorCollection) DeleteBatch(ctx context.Context, ids []string) error {
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

	// Get vectors first to remove from filter indexes
	vectors, err := c.segments.GetBatch(ctx, ids)
	if err != nil {
		return fmt.Errorf("failed to get vectors for deletion: %w", err)
	}

	// Remove from filter indexes
	for _, vector := range vectors {
		if len(vector.Metadata) > 0 {
			if err := c.filterEngine.RemoveDocument(vector.ID, vector.Metadata); err != nil {
				fmt.Printf("Warning: failed to remove from filter indexes: %v\n", err)
			}
		}
	}

	// Remove from vector index
	if err := c.vectorIndex.RemoveBatch(ctx, ids); err != nil {
		fmt.Printf("Warning: failed to remove from vector index: %v\n", err)
	}

	// Remove from segments
	if err := c.segments.DeleteBatch(ctx, ids); err != nil {
		return fmt.Errorf("failed to remove from segments: %w", err)
	}

	c.metadata.UpdatedAt = time.Now()
	c.metadata.OperationCount += int64(len(ids))

	return nil
}

// Count returns the total number of vectors in the collection
func (c *EnhancedVectorCollection) Count(ctx context.Context) (int64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return 0, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return 0, api.ErrContextCanceled
	}

	// Get count from segment manager stats
	stats, err := c.segments.Stats(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to get segment stats: %w", err)
	}

	return stats.TotalVectors, nil
}

// List returns vectors with pagination
func (c *EnhancedVectorCollection) List(ctx context.Context, limit int, offset int) ([]*api.Vector, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	results := make([]*api.Vector, 0, limit)
	currentOffset := 0
	collected := 0

	// Scan segments to collect vectors with pagination
	err := c.segments.Scan(ctx, func(vector *api.Vector) bool {
		if currentOffset < offset {
			currentOffset++
			return true // Skip this vector
		}

		if collected >= limit {
			return false // Stop scanning
		}

		results = append(results, vector)
		collected++
		return true
	})

	if err != nil {
		return nil, fmt.Errorf("failed to scan segments: %w", err)
	}

	return results, nil
}

// Name returns the collection name
func (c *EnhancedVectorCollection) Name() string {
	return c.config.Name
}

// Config returns the collection configuration
func (c *EnhancedVectorCollection) Config() *api.CollectionConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Return a copy to prevent external modifications
	configCopy := *c.config
	return &configCopy
}

// Stats returns collection statistics
func (c *EnhancedVectorCollection) Stats(ctx context.Context) (*api.CollectionStats, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	// Get segment stats
	segmentStats, err := c.segments.Stats(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get segment stats: %w", err)
	}

	// Get index stats
	indexStats := c.vectorIndex.Stats()

	// Combine the stats
	stats := &api.CollectionStats{
		Name:        c.config.Name,
		VectorCount: segmentStats.TotalVectors,
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
		SizeBytes:   segmentStats.TotalSizeBytes,
		MemoryUsage: segmentStats.TotalMemoryUsage,
	}

	return stats, nil
}

// Optimize performs collection optimization
func (c *EnhancedVectorCollection) Optimize(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	// Optimize vector index
	if err := c.vectorIndex.Optimize(ctx); err != nil {
		return fmt.Errorf("vector index optimization failed: %w", err)
	}

	// Optimize filter engine
	if err := c.filterEngine.Optimize(ctx); err != nil {
		return fmt.Errorf("filter engine optimization failed: %w", err)
	}

	// Trigger segment compaction
	if err := c.segments.TriggerCompaction(ctx); err != nil {
		return fmt.Errorf("segment compaction failed: %w", err)
	}

	c.metadata.LastOptimizedAt = time.Now()
	c.metadata.UpdatedAt = time.Now()

	return nil
}

// Clear removes all vectors from the collection
func (c *EnhancedVectorCollection) Clear(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	// Clear all segments using the segment manager's clear method
	if err := c.segments.Clear(ctx); err != nil {
		return fmt.Errorf("failed to clear segments: %w", err)
	}

	// Clear filter engine by recreating it
	_ = c.filterEngine.Close()

	// Create new filter engine
	invertedConfig := &filter.IndexConfig{
		Field: "category",
		Type:  filter.InvertedIndex,
		Options: map[string]interface{}{
			"cache_size": 50 << 20,
		},
		ThreadSafe: true,
	}

	numericConfig := &filter.IndexConfig{
		Field: "value",
		Type:  filter.NumericIndex,
		Options: map[string]interface{}{
			"cache_size": 50 << 20,
		},
		ThreadSafe: true,
	}

	invertedIndex, err := filter.NewInvertedMetadataIndex(invertedConfig)
	if err != nil {
		return fmt.Errorf("failed to recreate inverted index: %w", err)
	}

	numericIndex, err := filter.NewNumericMetadataIndex(numericConfig)
	if err != nil {
		return fmt.Errorf("failed to recreate numeric index: %w", err)
	}

	c.filterEngine = filter.NewHybridFilterEngine(invertedIndex, numericIndex)

	// Recreate the vector index
	indexConfig := &index.Config{
		Dimension:      c.config.Dimension,
		Metric:         convertDistanceMetric(c.config.Metric),
		M:              c.config.M,
		EfConstruction: c.config.EfConstruction,
		MaxLayer:       c.config.MaxLayer,
		Seed:           c.config.Seed,
		ThreadSafe:     c.config.ThreadSafe,
	}

	c.vectorIndex.Close()
	hnswIndex, err := index.NewHNSWIndex(indexConfig)
	if err != nil {
		return fmt.Errorf("failed to recreate HNSW index: %w", err)
	}
	c.vectorIndex = newIndexAdapter(hnswIndex)

	c.metadata.UpdatedAt = time.Now()
	c.metadata.OperationCount = 0

	return nil
}

// Close closes the collection and releases resources
func (c *EnhancedVectorCollection) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return api.ErrClosed
	}

	c.closed = true

	// Stop background optimization
	c.stopBackgroundOptimization()

	// Close components
	var segmentErr, indexErr, filterErr error

	if c.segments != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		segmentErr = c.segments.Stop(ctx)
		cancel()
	}

	if c.vectorIndex != nil {
		indexErr = c.vectorIndex.Close()
	}

	if c.filterEngine != nil {
		filterErr = c.filterEngine.Close()
	}

	// Return the first error encountered
	if segmentErr != nil {
		return fmt.Errorf("failed to close segments: %w", segmentErr)
	}
	if indexErr != nil {
		return fmt.Errorf("failed to close index: %w", indexErr)
	}
	if filterErr != nil {
		return fmt.Errorf("failed to close filter engine: %w", filterErr)
	}

	return nil
}

// Helper methods

// startBackgroundOptimization starts the background optimization routine
func (c *EnhancedVectorCollection) startBackgroundOptimization() {
	c.mu.Lock()
	if c.optimizeTicker != nil || c.stopOptimize != nil {
		c.mu.Unlock()
		return
	}
	c.optimizeTicker = time.NewTicker(c.optimizeInterval)
	c.stopOptimize = make(chan bool, 1)
	c.mu.Unlock()

	go func() {
		defer func() {
			c.mu.Lock()
			if c.optimizeTicker != nil {
				c.optimizeTicker.Stop()
				c.optimizeTicker = nil
			}
			c.mu.Unlock()
		}()

		for {
			select {
			case <-c.optimizeTicker.C:
				if c.metadata.OperationCount > 10000 {
					ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
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
func (c *EnhancedVectorCollection) stopBackgroundOptimization() {
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

// applyDistanceFilters applies distance and score filters to search results
func (c *EnhancedVectorCollection) applyDistanceFilters(results []*api.SearchResult, maxDistance, minScore *float32) []*api.SearchResult {
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

// calculateSimilarity calculates similarity score between two vectors
func (c *EnhancedVectorCollection) calculateSimilarity(query []float32, target []float32, metric api.DistanceMetric) float32 {
	switch metric {
	case api.Cosine:
		return cosineSimilarity(query, target)
	case api.Euclidean:
		return 1.0 / (1.0 + euclideanDistance(query, target))
	case api.Manhattan:
		return 1.0 / (1.0 + manhattanDistance(query, target))
	case api.DotProduct:
		return dotProduct(query, target)
	default:
		return cosineSimilarity(query, target)
	}
}

// calculateDistance calculates distance from similarity score
func (c *EnhancedVectorCollection) calculateDistance(score float32, metric api.DistanceMetric) float32 {
	switch metric {
	case api.Cosine:
		return 1.0 - score
	case api.Euclidean, api.Manhattan:
		if score == 0 {
			return float32(1e9) // Large distance
		}
		return (1.0 / score) - 1.0
	case api.DotProduct:
		return -score // Negative because higher dot product means closer
	default:
		return 1.0 - score
	}
}

// sortResultsByScore sorts search results by score in descending order
func (c *EnhancedVectorCollection) sortResultsByScore(results []*api.SearchResult) {
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].Score < results[j].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
}

// Vector similarity functions
func cosineSimilarity(a, b []float32) float32 {
	var dotProduct, normA, normB float32
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(sqrt(float64(normA)) * sqrt(float64(normB))))
}

func euclideanDistance(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(sqrt(float64(sum)))
}

func manhattanDistance(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		sum += diff
	}
	return sum
}

func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func sqrt(x float64) float64 {
	if x == 0 {
		return 0
	}

	// Newton's method approximation
	guess := x / 2
	for i := 0; i < 10; i++ {
		guess = (guess + x/guess) / 2
	}
	return guess
}
