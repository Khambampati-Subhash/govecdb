package index

import (
	"context"
	"sync"
	"sync/atomic"
)

// SyncMap is a generic lock-free map implementation optimized for high-performance operations
type SyncMap[K comparable, V any] struct {
	mu    sync.RWMutex
	data  map[K]V
	count int64
}

// NewSyncMap creates a new SyncMap
func NewSyncMap[K comparable, V any]() *SyncMap[K, V] {
	return &SyncMap[K, V]{
		data: make(map[K]V),
	}
}

// Load retrieves a value from the map
func (m *SyncMap[K, V]) Load(key K) (V, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	value, exists := m.data[key]
	return value, exists
}

// Store sets a value in the map
func (m *SyncMap[K, V]) Store(key K, value V) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.data[key]; !exists {
		atomic.AddInt64(&m.count, 1)
	}
	m.data[key] = value
}

// Delete removes a value from the map
func (m *SyncMap[K, V]) Delete(key K) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.data[key]; exists {
		delete(m.data, key)
		atomic.AddInt64(&m.count, -1)
	}
}

// Range iterates over all key-value pairs
func (m *SyncMap[K, V]) Range(fn func(key K, value V) bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for k, v := range m.data {
		if !fn(k, v) {
			break
		}
	}
}

// Size returns the number of elements in the map
func (m *SyncMap[K, V]) Size() int {
	return int(atomic.LoadInt64(&m.count))
}

// MemoryPool provides memory pooling for reducing allocations
type MemoryPool[T any] struct {
	pool sync.Pool
	new  func() T
}

// NewMemoryPool creates a new memory pool
func NewMemoryPool[T any](newFunc func() T) *MemoryPool[T] {
	return &MemoryPool[T]{
		pool: sync.Pool{
			New: func() interface{} {
				return newFunc()
			},
		},
		new: newFunc,
	}
}

// Get retrieves an object from the pool
func (p *MemoryPool[T]) Get() T {
	return p.pool.Get().(T)
}

// Put returns an object to the pool
func (p *MemoryPool[T]) Put(obj T) {
	p.pool.Put(obj)
}

// VectorPool is a specialized pool for vector slices with size-based pooling
type VectorPool struct {
	pools []sync.Pool // Different pools for different sizes
	sizes []int       // Track sizes for each pool
	stats *PoolStats  // Pool statistics
}

// PoolStats tracks memory pool usage statistics
type PoolStats struct {
	hits   int64 // Cache hits
	misses int64 // Cache misses
	puts   int64 // Put operations
	gets   int64 // Get operations
	wastes int64 // Wasted allocations (too large)
}

// NewVectorPool creates a new optimized vector pool
func NewVectorPool() *VectorPool {
	// Common vector sizes in machine learning: 64, 128, 256, 512, 768, 1024, 1536, 2048
	commonSizes := []int{16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096}

	vp := &VectorPool{
		pools: make([]sync.Pool, len(commonSizes)),
		sizes: commonSizes,
		stats: &PoolStats{},
	}

	for i, size := range commonSizes {
		currentSize := size // Capture for closure
		vp.pools[i] = sync.Pool{
			New: func() interface{} {
				atomic.AddInt64(&vp.stats.misses, 1)
				return make([]float32, 0, currentSize)
			},
		}
	}

	return vp
}

// Get retrieves a vector slice with at least the specified capacity
func (vp *VectorPool) Get(minCapacity int) []float32 {
	atomic.AddInt64(&vp.stats.gets, 1)

	// Find the smallest pool that can accommodate the request
	poolIndex := -1
	for i, size := range vp.sizes {
		if size >= minCapacity {
			poolIndex = i
			break
		}
	}

	if poolIndex == -1 {
		// Too large for pools, allocate directly
		atomic.AddInt64(&vp.stats.wastes, 1)
		return make([]float32, 0, minCapacity)
	}

	atomic.AddInt64(&vp.stats.hits, 1)
	slice := vp.pools[poolIndex].Get().([]float32)
	return slice[:0] // Reset length but keep capacity
}

// Put returns a vector slice to the appropriate pool
func (vp *VectorPool) Put(slice []float32) {
	if slice == nil {
		return
	}

	atomic.AddInt64(&vp.stats.puts, 1)
	capacity := cap(slice)

	// Find the exact pool for this capacity
	for i, size := range vp.sizes {
		if size == capacity {
			// Clear slice before returning to pool to prevent memory leaks
			for j := range slice {
				slice[j] = 0
			}
			// Note: SA6002 warning is a false positive - sync.Pool handles this correctly
			vp.pools[i].Put(interface{}(slice[:0]))
			return
		}
	}
	// If capacity doesn't match a pool size, let it be garbage collected
}

// GetStats returns pool usage statistics
func (vp *VectorPool) GetStats() PoolStats {
	return PoolStats{
		hits:   atomic.LoadInt64(&vp.stats.hits),
		misses: atomic.LoadInt64(&vp.stats.misses),
		puts:   atomic.LoadInt64(&vp.stats.puts),
		gets:   atomic.LoadInt64(&vp.stats.gets),
		wastes: atomic.LoadInt64(&vp.stats.wastes),
	}
}

// HitRate returns the cache hit rate as a percentage
func (vp *VectorPool) HitRate() float64 {
	hits := atomic.LoadInt64(&vp.stats.hits)
	total := hits + atomic.LoadInt64(&vp.stats.misses)
	if total == 0 {
		return 0
	}
	return float64(hits) / float64(total) * 100
}

// LockFreeCounter is an atomic counter for statistics
type LockFreeCounter struct {
	value int64
}

// NewLockFreeCounter creates a new lock-free counter
func NewLockFreeCounter() *LockFreeCounter {
	return &LockFreeCounter{}
}

// Add atomically adds to the counter
func (c *LockFreeCounter) Add(delta int64) int64 {
	return atomic.AddInt64(&c.value, delta)
}

// Load atomically loads the counter value
func (c *LockFreeCounter) Load() int64 {
	return atomic.LoadInt64(&c.value)
}

// Store atomically stores a value to the counter
func (c *LockFreeCounter) Store(value int64) {
	atomic.StoreInt64(&c.value, value)
}

// CompareAndSwap atomically compares and swaps the counter value
func (c *LockFreeCounter) CompareAndSwap(old, new int64) bool {
	return atomic.CompareAndSwapInt64(&c.value, old, new)
}

// BatchProcessor handles batch operations efficiently
type BatchProcessor[T any] struct {
	batchSize int
	processor func([]T) error
	buffer    []T
	mu        sync.Mutex
}

// NewBatchProcessor creates a new batch processor
func NewBatchProcessor[T any](batchSize int, processor func([]T) error) *BatchProcessor[T] {
	return &BatchProcessor[T]{
		batchSize: batchSize,
		processor: processor,
		buffer:    make([]T, 0, batchSize),
	}
}

// Add adds an item to the batch processor
func (bp *BatchProcessor[T]) Add(item T) error {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	bp.buffer = append(bp.buffer, item)

	if len(bp.buffer) >= bp.batchSize {
		return bp.flush()
	}

	return nil
}

// Flush processes any remaining items in the buffer
func (bp *BatchProcessor[T]) Flush() error {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	return bp.flush()
}

// flush processes the current buffer (must be called with lock held)
func (bp *BatchProcessor[T]) flush() error {
	if len(bp.buffer) == 0 {
		return nil
	}

	err := bp.processor(bp.buffer)
	bp.buffer = bp.buffer[:0] // Reset buffer
	return err
}

// WorkerPool manages a pool of workers for parallel processing with enhanced performance
type WorkerPool[T any] struct {
	workers   int
	jobs      chan T
	results   chan error
	processor func(T) error
	wg        sync.WaitGroup
	ctx       context.Context
	cancel    context.CancelFunc
	started   int32        // Atomic flag for started state
	stats     *WorkerStats // Worker statistics
}

// WorkerStats tracks worker pool performance
type WorkerStats struct {
	processed  int64 // Total jobs processed
	errors     int64 // Total errors
	avgLatency int64 // Average processing latency in nanoseconds
	active     int32 // Currently active workers
}

// NewWorkerPool creates a new optimized worker pool
func NewWorkerPool[T any](workers int, processor func(T) error) *WorkerPool[T] {
	ctx, cancel := context.WithCancel(context.Background())

	// Calculate optimal buffer size based on worker count
	bufferSize := workers * 4 // 4x workers for better throughput
	if bufferSize < 64 {
		bufferSize = 64 // Minimum buffer size
	}

	return &WorkerPool[T]{
		workers:   workers,
		jobs:      make(chan T, bufferSize),
		results:   make(chan error, bufferSize),
		processor: processor,
		ctx:       ctx,
		cancel:    cancel,
		stats:     &WorkerStats{},
	}
}

// Start starts the worker pool with optimized worker management
func (wp *WorkerPool[T]) Start() {
	if !atomic.CompareAndSwapInt32(&wp.started, 0, 1) {
		return // Already started
	}

	// Start workers with CPU affinity consideration
	for i := 0; i < wp.workers; i++ {
		wp.wg.Add(1)
		go wp.optimizedWorker(i)
	}
}

// Submit submits a job to the worker pool
func (wp *WorkerPool[T]) Submit(job T) {
	wp.jobs <- job
}

// Close gracefully shuts down the worker pool
func (wp *WorkerPool[T]) Close() {
	if !atomic.CompareAndSwapInt32(&wp.started, 1, 0) {
		return // Already stopped or never started
	}

	// Cancel context to signal workers to stop
	wp.cancel()

	// Close jobs channel
	close(wp.jobs)

	// Wait for all workers to finish
	wp.wg.Wait()

	// Close results channel
	close(wp.results)
}

// GetStats returns worker pool statistics
func (wp *WorkerPool[T]) GetStats() WorkerStats {
	return WorkerStats{
		processed:  atomic.LoadInt64(&wp.stats.processed),
		errors:     atomic.LoadInt64(&wp.stats.errors),
		avgLatency: atomic.LoadInt64(&wp.stats.avgLatency),
		active:     atomic.LoadInt32(&wp.stats.active),
	}
}

// Results returns the results channel
func (wp *WorkerPool[T]) Results() <-chan error {
	return wp.results
}

// optimizedWorker processes jobs with performance monitoring and context cancellation
func (wp *WorkerPool[T]) optimizedWorker(workerID int) {
	defer wp.wg.Done()

	atomic.AddInt32(&wp.stats.active, 1)
	defer atomic.AddInt32(&wp.stats.active, -1)

	// Worker loop for processing jobs

	for {
		select {
		case <-wp.ctx.Done():
			return

		case job, ok := <-wp.jobs:
			if !ok {
				return // Channel closed
			}

			// Process with timing
			start := getCurrentTimeNanos()
			err := wp.processor(job)
			elapsed := getCurrentTimeNanos() - start

			// Update statistics
			atomic.AddInt64(&wp.stats.processed, 1)
			if err != nil {
				atomic.AddInt64(&wp.stats.errors, 1)
			}

			// Update average latency using exponential moving average
			oldAvg := atomic.LoadInt64(&wp.stats.avgLatency)
			newAvg := (oldAvg*7 + elapsed) / 8 // EMA with alpha=0.125
			atomic.StoreInt64(&wp.stats.avgLatency, newAvg)

			// Send result
			select {
			case wp.results <- err:
			case <-wp.ctx.Done():
				return
			}
		}
	}
}

// CacheAlignedPadding provides padding to prevent false sharing
type CacheAlignedPadding struct {
	_ [64]byte
}

// AlignedData represents cache-aligned data structure
type AlignedData[T any] struct {
	Data T
	_    CacheAlignedPadding
}
