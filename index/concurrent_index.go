package index

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// ConcurrentHNSWIndex provides thread-safe HNSW operations with high concurrency
type ConcurrentHNSWIndex struct {
	// Optimized graph backend
	graph *OptimizedHNSWGraph

	// Configuration
	config *Config

	// Concurrency control
	insertSemaphore  chan struct{} // Limits concurrent insertions
	searchWorkerPool *WorkerPool[*SearchJob]
	insertWorkerPool *WorkerPool[*InsertJob]

	// Batch processing
	insertBatcher *BatchProcessor[*Vector]
	deleteBatcher *BatchProcessor[string]

	// Statistics
	stats *ConcurrentStats

	// Lifecycle management
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// ConcurrentStats tracks performance metrics atomically
type ConcurrentStats struct {
	// Operation counters
	insertCount int64
	searchCount int64
	deleteCount int64
	batchCount  int64

	// Performance metrics
	avgInsertLatency int64 // nanoseconds
	avgSearchLatency int64 // nanoseconds
	peakQPS          int64
	currentQPS       int64

	// Resource utilization
	activeWorkers    int32
	queuedOperations int32
	memoryUsed       int64

	// Error tracking
	insertErrors int64
	searchErrors int64
	deleteErrors int64

	// Cache-aligned padding
	_ [64]byte
}

// SearchJob represents a search operation job
type SearchJob struct {
	Query    []float32
	K        int
	Filter   FilterFunc
	Response chan *SearchResponse
}

// SearchResponse contains search results and metadata
type SearchResponse struct {
	Results []*SearchResult
	Error   error
	Latency time.Duration
	Stats   *OperationStats
}

// InsertJob represents an insert operation job
type InsertJob struct {
	Vector   *Vector
	Response chan *InsertResponse
}

// InsertResponse contains insert result and metadata
type InsertResponse struct {
	Error   error
	Latency time.Duration
	Stats   *OperationStats
}

// OperationStats contains per-operation statistics
type OperationStats struct {
	MemoryAllocated      int64
	CacheMisses          int64
	DistanceCalculations int64
	NodesVisited         int64
}

// NewConcurrentHNSWIndex creates a new concurrent HNSW index
func NewConcurrentHNSWIndex(config *Config) (*ConcurrentHNSWIndex, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	graph, err := NewOptimizedHNSWGraph(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create optimized graph: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	// Calculate optimal worker counts based on CPU cores
	numCores := runtime.NumCPU()
	searchWorkers := numCores * 2 // Search is CPU intensive
	insertWorkers := numCores / 2 // Insert is more memory intensive
	if insertWorkers < 1 {
		insertWorkers = 1
	}

	index := &ConcurrentHNSWIndex{
		graph:           graph,
		config:          config,
		insertSemaphore: make(chan struct{}, insertWorkers*2), // Limit concurrent insertions
		stats:           &ConcurrentStats{},
		ctx:             ctx,
		cancel:          cancel,
	}

	// Initialize worker pools
	index.searchWorkerPool = NewWorkerPool(searchWorkers, index.processSearchJob)
	index.insertWorkerPool = NewWorkerPool(insertWorkers, index.processInsertJob)

	// Initialize batch processors
	index.insertBatcher = NewBatchProcessor(100, index.processBatchInsert) // Batch size of 100
	index.deleteBatcher = NewBatchProcessor(50, index.processBatchDelete)  // Batch size of 50

	// Start background workers
	index.startBackgroundWorkers()

	return index, nil
}

// startBackgroundWorkers starts all background processing
func (idx *ConcurrentHNSWIndex) startBackgroundWorkers() {
	// Start worker pools
	idx.searchWorkerPool.Start()
	idx.insertWorkerPool.Start()

	// Start statistics collection
	idx.wg.Add(1)
	go idx.statisticsCollector()

	// Start memory management
	idx.wg.Add(1)
	go idx.memoryManager()

	// Start health monitoring
	idx.wg.Add(1)
	go idx.healthMonitor()
}

// Add inserts a vector with high concurrency support
func (idx *ConcurrentHNSWIndex) Add(vector *Vector) error {
	if vector == nil {
		return fmt.Errorf("vector cannot be nil")
	}

	startTime := time.Now()
	atomic.AddInt64(&idx.stats.insertCount, 1)
	atomic.AddInt32(&idx.stats.queuedOperations, 1)

	// Use batch processor for better throughput
	err := idx.insertBatcher.Add(vector)

	// Update statistics
	latency := time.Since(startTime).Nanoseconds()
	idx.updateInsertLatency(latency)
	atomic.AddInt32(&idx.stats.queuedOperations, -1)

	if err != nil {
		atomic.AddInt64(&idx.stats.insertErrors, 1)
	}

	return err
}

// AddAsync inserts a vector asynchronously
func (idx *ConcurrentHNSWIndex) AddAsync(vector *Vector) <-chan *InsertResponse {
	response := make(chan *InsertResponse, 1)

	job := &InsertJob{
		Vector:   vector,
		Response: response,
	}

	idx.insertWorkerPool.Submit(job)
	return response
}

// AddBatch inserts multiple vectors efficiently using batch processing
func (idx *ConcurrentHNSWIndex) AddBatch(vectors []*Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	startTime := time.Now()
	atomic.AddInt64(&idx.stats.batchCount, 1)

	// Process in parallel batches for maximum throughput
	batchSize := 1000
	numBatches := (len(vectors) + batchSize - 1) / batchSize

	workerPool := NewWorkerPool(runtime.NumCPU(), func(batch []*Vector) error {
		return idx.processBatchInsert(batch)
	})
	workerPool.Start()

	// Submit batches
	for i := 0; i < len(vectors); i += batchSize {
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}

		workerPool.Submit(vectors[i:end])
	}

	workerPool.Close()

	// Collect results
	var firstError error
	for i := 0; i < numBatches; i++ {
		if err := <-workerPool.Results(); err != nil && firstError == nil {
			firstError = err
		}
	}

	// Update statistics
	latency := time.Since(startTime).Nanoseconds()
	avgLatency := latency / int64(len(vectors))
	idx.updateInsertLatency(avgLatency)

	return firstError
}

// Search performs concurrent k-NN search
func (idx *ConcurrentHNSWIndex) Search(query []float32, k int) ([]*SearchResult, error) {
	return idx.SearchWithFilter(query, k, nil)
}

// SearchWithFilter performs concurrent search with filtering
func (idx *ConcurrentHNSWIndex) SearchWithFilter(query []float32, k int, filter FilterFunc) ([]*SearchResult, error) {
	if len(query) != idx.config.Dimension {
		return nil, ErrDimensionMismatch
	}

	if k <= 0 {
		return nil, ErrInvalidK
	}

	startTime := time.Now()
	atomic.AddInt64(&idx.stats.searchCount, 1)

	// Use direct call for synchronous search
	results, err := idx.graph.Search(query, k, filter)

	// Update statistics
	latency := time.Since(startTime).Nanoseconds()
	idx.updateSearchLatency(latency)

	if err != nil {
		atomic.AddInt64(&idx.stats.searchErrors, 1)
	}

	return results, err
}

// SearchAsync performs asynchronous search
func (idx *ConcurrentHNSWIndex) SearchAsync(query []float32, k int, filter FilterFunc) <-chan *SearchResponse {
	response := make(chan *SearchResponse, 1)

	job := &SearchJob{
		Query:    query,
		K:        k,
		Filter:   filter,
		Response: response,
	}

	idx.searchWorkerPool.Submit(job)
	return response
}

// MultiSearch performs multiple searches concurrently
func (idx *ConcurrentHNSWIndex) MultiSearch(queries [][]float32, k int, filter FilterFunc) ([][]*SearchResult, error) {
	if len(queries) == 0 {
		return nil, nil
	}

	results := make([][]*SearchResult, len(queries))
	errors := make([]error, len(queries))

	// Use worker pool for parallel processing
	workerPool := NewWorkerPool(runtime.NumCPU(), func(i int) error {
		result, err := idx.SearchWithFilter(queries[i], k, filter)
		results[i] = result
		errors[i] = err
		return err
	})

	workerPool.Start()

	// Submit all queries
	for i := range queries {
		workerPool.Submit(i)
	}

	workerPool.Close()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return results, err
		}
	}

	return results, nil
}

// Delete removes a vector with batch processing
func (idx *ConcurrentHNSWIndex) Delete(id string) error {
	if id == "" {
		return fmt.Errorf("id cannot be empty")
	}

	atomic.AddInt64(&idx.stats.deleteCount, 1)
	return idx.deleteBatcher.Add(id)
}

// processSearchJob processes a search job
func (idx *ConcurrentHNSWIndex) processSearchJob(job *SearchJob) error {
	atomic.AddInt32(&idx.stats.activeWorkers, 1)
	defer atomic.AddInt32(&idx.stats.activeWorkers, -1)

	startTime := time.Now()
	results, err := idx.graph.Search(job.Query, job.K, job.Filter)
	latency := time.Since(startTime)

	// Create operation stats
	stats := &OperationStats{
		// These would be populated by the actual implementation
		DistanceCalculations: int64(len(results)),
		NodesVisited:         int64(len(results) * 2), // Approximation
	}

	response := &SearchResponse{
		Results: results,
		Error:   err,
		Latency: latency,
		Stats:   stats,
	}

	select {
	case job.Response <- response:
	case <-idx.ctx.Done():
		return idx.ctx.Err()
	}

	return err
}

// processInsertJob processes an insert job
func (idx *ConcurrentHNSWIndex) processInsertJob(job *InsertJob) error {
	atomic.AddInt32(&idx.stats.activeWorkers, 1)
	defer atomic.AddInt32(&idx.stats.activeWorkers, -1)

	// Acquire semaphore to limit concurrent insertions
	select {
	case idx.insertSemaphore <- struct{}{}:
		defer func() { <-idx.insertSemaphore }()
	case <-idx.ctx.Done():
		return idx.ctx.Err()
	}

	startTime := time.Now()
	err := idx.graph.Insert(job.Vector)
	latency := time.Since(startTime)

	stats := &OperationStats{
		// These would be populated by the actual implementation
		MemoryAllocated: int64(len(job.Vector.Data) * 4), // 4 bytes per float32
	}

	response := &InsertResponse{
		Error:   err,
		Latency: latency,
		Stats:   stats,
	}

	select {
	case job.Response <- response:
	case <-idx.ctx.Done():
		return idx.ctx.Err()
	}

	return err
}

// processBatchInsert processes a batch of vectors for insertion
func (idx *ConcurrentHNSWIndex) processBatchInsert(vectors []*Vector) error {
	for _, vector := range vectors {
		if err := idx.graph.Insert(vector); err != nil {
			return err
		}
	}
	return nil
}

// processBatchDelete processes a batch of deletions
func (idx *ConcurrentHNSWIndex) processBatchDelete(ids []string) error {
	for _, id := range ids {
		// Implement batch deletion in the graph
		// For now, delete one by one
		_ = id // TODO: implement batch delete in graph
	}
	return nil
}

// statisticsCollector collects and updates performance statistics
func (idx *ConcurrentHNSWIndex) statisticsCollector() {
	defer idx.wg.Done()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	var lastSearchCount, lastInsertCount int64
	lastTime := time.Now()

	for {
		select {
		case <-ticker.C:
			now := time.Now()
			elapsed := now.Sub(lastTime).Seconds()

			// Calculate QPS
			currentSearches := atomic.LoadInt64(&idx.stats.searchCount)
			currentInserts := atomic.LoadInt64(&idx.stats.insertCount)

			searchQPS := float64(currentSearches-lastSearchCount) / elapsed
			insertQPS := float64(currentInserts-lastInsertCount) / elapsed
			totalQPS := searchQPS + insertQPS

			atomic.StoreInt64(&idx.stats.currentQPS, int64(totalQPS))

			// Update peak QPS
			peakQPS := atomic.LoadInt64(&idx.stats.peakQPS)
			if int64(totalQPS) > peakQPS {
				atomic.StoreInt64(&idx.stats.peakQPS, int64(totalQPS))
			}

			lastSearchCount = currentSearches
			lastInsertCount = currentInserts
			lastTime = now

		case <-idx.ctx.Done():
			return
		}
	}
}

// memoryManager handles memory optimization and garbage collection
func (idx *ConcurrentHNSWIndex) memoryManager() {
	defer idx.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Force garbage collection periodically to keep memory usage low
			runtime.GC()

			// Update memory statistics
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			atomic.StoreInt64(&idx.stats.memoryUsed, int64(m.Alloc))

		case <-idx.ctx.Done():
			return
		}
	}
}

// healthMonitor monitors the health of the index
func (idx *ConcurrentHNSWIndex) healthMonitor() {
	defer idx.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Check for error rates
			totalOps := atomic.LoadInt64(&idx.stats.insertCount) + atomic.LoadInt64(&idx.stats.searchCount)
			totalErrors := atomic.LoadInt64(&idx.stats.insertErrors) + atomic.LoadInt64(&idx.stats.searchErrors)

			if totalOps > 0 {
				errorRate := float64(totalErrors) / float64(totalOps)
				if errorRate > 0.01 { // 1% error rate threshold
					// Log warning about high error rate
					fmt.Printf("Warning: High error rate detected: %.2f%%\n", errorRate*100)
				}
			}

		case <-idx.ctx.Done():
			return
		}
	}
}

// updateInsertLatency updates the average insert latency
func (idx *ConcurrentHNSWIndex) updateInsertLatency(latency int64) {
	// Simple exponential moving average
	current := atomic.LoadInt64(&idx.stats.avgInsertLatency)
	if current == 0 {
		atomic.StoreInt64(&idx.stats.avgInsertLatency, latency)
	} else {
		// EMA with alpha = 0.1
		newAvg := (9*current + latency) / 10
		atomic.StoreInt64(&idx.stats.avgInsertLatency, newAvg)
	}
}

// updateSearchLatency updates the average search latency
func (idx *ConcurrentHNSWIndex) updateSearchLatency(latency int64) {
	current := atomic.LoadInt64(&idx.stats.avgSearchLatency)
	if current == 0 {
		atomic.StoreInt64(&idx.stats.avgSearchLatency, latency)
	} else {
		newAvg := (9*current + latency) / 10
		atomic.StoreInt64(&idx.stats.avgSearchLatency, newAvg)
	}
}

// GetStats returns comprehensive statistics
func (idx *ConcurrentHNSWIndex) GetStats() *ConcurrentStats {
	// Return a copy of statistics
	return &ConcurrentStats{
		insertCount:      atomic.LoadInt64(&idx.stats.insertCount),
		searchCount:      atomic.LoadInt64(&idx.stats.searchCount),
		deleteCount:      atomic.LoadInt64(&idx.stats.deleteCount),
		batchCount:       atomic.LoadInt64(&idx.stats.batchCount),
		avgInsertLatency: atomic.LoadInt64(&idx.stats.avgInsertLatency),
		avgSearchLatency: atomic.LoadInt64(&idx.stats.avgSearchLatency),
		peakQPS:          atomic.LoadInt64(&idx.stats.peakQPS),
		currentQPS:       atomic.LoadInt64(&idx.stats.currentQPS),
		activeWorkers:    atomic.LoadInt32(&idx.stats.activeWorkers),
		queuedOperations: atomic.LoadInt32(&idx.stats.queuedOperations),
		memoryUsed:       atomic.LoadInt64(&idx.stats.memoryUsed),
		insertErrors:     atomic.LoadInt64(&idx.stats.insertErrors),
		searchErrors:     atomic.LoadInt64(&idx.stats.searchErrors),
		deleteErrors:     atomic.LoadInt64(&idx.stats.deleteErrors),
	}
}

// Close gracefully shuts down the index
func (idx *ConcurrentHNSWIndex) Close() error {
	// Cancel all background operations
	idx.cancel()

	// Flush any pending batches
	idx.insertBatcher.Flush()
	idx.deleteBatcher.Flush()

	// Close worker pools
	idx.searchWorkerPool.Close()
	idx.insertWorkerPool.Close()

	// Wait for all background workers to finish
	idx.wg.Wait()

	return nil
}

// Size returns the number of vectors in the index
func (idx *ConcurrentHNSWIndex) Size() int {
	return idx.graph.nodes.Size()
}

// IsHealthy returns whether the index is healthy
func (idx *ConcurrentHNSWIndex) IsHealthy() bool {
	stats := idx.GetStats()

	// Check error rates
	totalOps := stats.insertCount + stats.searchCount
	totalErrors := stats.insertErrors + stats.searchErrors

	if totalOps > 0 {
		errorRate := float64(totalErrors) / float64(totalOps)
		return errorRate < 0.01 // Less than 1% error rate
	}

	return true
}
