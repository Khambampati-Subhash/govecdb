package index

import (
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

// VectorPool is a specialized pool for vector slices
type VectorPool struct {
	pools []sync.Pool // Different pools for different sizes
}

// NewVectorPool creates a new vector pool
func NewVectorPool() *VectorPool {
	vp := &VectorPool{
		pools: make([]sync.Pool, 20), // Support vectors up to 2^20 elements
	}

	for i := range vp.pools {
		size := 1 << i // 2^i
		vp.pools[i] = sync.Pool{
			New: func() interface{} {
				return make([]float32, 0, size)
			},
		}
	}

	return vp
}

// Get retrieves a vector slice with at least the specified capacity
func (vp *VectorPool) Get(minCapacity int) []float32 {
	// Find the appropriate pool
	poolIndex := 0
	for poolIndex < len(vp.pools) && (1<<poolIndex) < minCapacity {
		poolIndex++
	}

	if poolIndex >= len(vp.pools) {
		// Too large for pools, allocate directly
		return make([]float32, 0, minCapacity)
	}

	slice := vp.pools[poolIndex].Get().([]float32)
	return slice[:0] // Reset length but keep capacity
}

// Put returns a vector slice to the appropriate pool
func (vp *VectorPool) Put(slice []float32) {
	if slice == nil {
		return
	}

	capacity := cap(slice)

	// Find the appropriate pool
	poolIndex := 0
	for poolIndex < len(vp.pools) && (1<<poolIndex) < capacity {
		poolIndex++
	}

	if poolIndex < len(vp.pools) && (1<<poolIndex) == capacity {
		vp.pools[poolIndex].Put(slice)
	}
	// If capacity doesn't match a pool size, let it be garbage collected
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

// WorkerPool manages a pool of workers for parallel processing
type WorkerPool[T any] struct {
	workers   int
	jobs      chan T
	results   chan error
	processor func(T) error
	wg        sync.WaitGroup
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool[T any](workers int, processor func(T) error) *WorkerPool[T] {
	return &WorkerPool[T]{
		workers:   workers,
		jobs:      make(chan T, workers*2), // Buffer jobs
		results:   make(chan error, workers*2),
		processor: processor,
	}
}

// Start starts the worker pool
func (wp *WorkerPool[T]) Start() {
	for i := 0; i < wp.workers; i++ {
		wp.wg.Add(1)
		go wp.worker()
	}
}

// Submit submits a job to the worker pool
func (wp *WorkerPool[T]) Submit(job T) {
	wp.jobs <- job
}

// Close closes the worker pool and waits for all workers to finish
func (wp *WorkerPool[T]) Close() {
	close(wp.jobs)
	wp.wg.Wait()
	close(wp.results)
}

// Results returns the results channel
func (wp *WorkerPool[T]) Results() <-chan error {
	return wp.results
}

// worker processes jobs from the jobs channel
func (wp *WorkerPool[T]) worker() {
	defer wp.wg.Done()

	for job := range wp.jobs {
		err := wp.processor(job)
		wp.results <- err
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
