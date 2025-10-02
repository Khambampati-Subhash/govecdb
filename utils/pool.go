package utils

// Package utils provides utility functions and optimizations for GoVecDB.
// This includes buffer pooling, SIMD distance functions, and performance utilities.

import (
	"sync"
)

// BufferPool provides thread-safe buffer pooling to reduce GC pressure
type BufferPool struct {
	pool sync.Pool
}

// NewBufferPool creates a new buffer pool with the specified initial capacity
func NewBufferPool(initialCapacity int) *BufferPool {
	return &BufferPool{
		pool: sync.Pool{
			New: func() interface{} {
				return make([]byte, 0, initialCapacity)
			},
		},
	}
}

// Get retrieves a buffer from the pool
func (bp *BufferPool) Get() []byte {
	return bp.pool.Get().([]byte)
}

// Put returns a buffer to the pool after resetting its length
func (bp *BufferPool) Put(buf []byte) {
	// Reset length but keep capacity
	buf = buf[:0]
	bp.pool.Put(buf)
}

// Float32Pool provides pooling for float32 slices
type Float32Pool struct {
	pool sync.Pool
}

// NewFloat32Pool creates a new float32 slice pool
func NewFloat32Pool(initialCapacity int) *Float32Pool {
	return &Float32Pool{
		pool: sync.Pool{
			New: func() interface{} {
				return make([]float32, 0, initialCapacity)
			},
		},
	}
}

// Get retrieves a float32 slice from the pool
func (fp *Float32Pool) Get() []float32 {
	return fp.pool.Get().([]float32)
}

// Put returns a float32 slice to the pool after resetting
func (fp *Float32Pool) Put(slice []float32) {
	slice = slice[:0]
	fp.pool.Put(slice)
}

// StringPool provides pooling for string slices (used for vector IDs)
type StringPool struct {
	pool sync.Pool
}

// NewStringPool creates a new string slice pool
func NewStringPool(initialCapacity int) *StringPool {
	return &StringPool{
		pool: sync.Pool{
			New: func() interface{} {
				return make([]string, 0, initialCapacity)
			},
		},
	}
}

// Get retrieves a string slice from the pool
func (sp *StringPool) Get() []string {
	return sp.pool.Get().([]string)
}

// Put returns a string slice to the pool after resetting
func (sp *StringPool) Put(slice []string) {
	slice = slice[:0]
	sp.pool.Put(slice)
}

// VectorPool provides pooling for vector data to reduce allocations during search
type VectorPool struct {
	dimension int
	pool      sync.Pool
}

// NewVectorPool creates a new vector pool for vectors of specific dimension
func NewVectorPool(dimension int) *VectorPool {
	return &VectorPool{
		dimension: dimension,
		pool: sync.Pool{
			New: func() interface{} {
				return make([]float32, dimension)
			},
		},
	}
}

// Get retrieves a vector from the pool
func (vp *VectorPool) Get() []float32 {
	return vp.pool.Get().([]float32)
}

// Put returns a vector to the pool after clearing
func (vp *VectorPool) Put(vector []float32) {
	if len(vector) == vp.dimension {
		// Clear the vector data
		for i := range vector {
			vector[i] = 0
		}
		vp.pool.Put(vector)
	}
}

// ResultPool provides pooling for search result slices
type ResultPool struct {
	pool sync.Pool
}

// SearchResult represents a lightweight search result for pooling
type SearchResult struct {
	ID       string
	Score    float32
	Distance float32
}

// NewResultPool creates a new search result pool
func NewResultPool(initialCapacity int) *ResultPool {
	return &ResultPool{
		pool: sync.Pool{
			New: func() interface{} {
				return make([]*SearchResult, 0, initialCapacity)
			},
		},
	}
}

// Get retrieves a result slice from the pool
func (rp *ResultPool) Get() []*SearchResult {
	return rp.pool.Get().([]*SearchResult)
}

// Put returns a result slice to the pool after clearing
func (rp *ResultPool) Put(results []*SearchResult) {
	// Clear references to prevent memory leaks
	for i := range results {
		results[i] = nil
	}
	results = results[:0]
	rp.pool.Put(results)
}

// WorkerPool provides a pool of worker goroutines for parallel processing
type WorkerPool struct {
	workers   int
	taskQueue chan func()
	wg        sync.WaitGroup
	quit      chan struct{}
}

// NewWorkerPool creates a new worker pool with specified number of workers
func NewWorkerPool(workers int) *WorkerPool {
	wp := &WorkerPool{
		workers:   workers,
		taskQueue: make(chan func(), workers*2), // Buffer for tasks
		quit:      make(chan struct{}),
	}

	// Start worker goroutines
	for i := 0; i < workers; i++ {
		wp.wg.Add(1)
		go wp.worker()
	}

	return wp
}

// worker runs in a goroutine and processes tasks from the queue
func (wp *WorkerPool) worker() {
	defer wp.wg.Done()

	for {
		select {
		case task := <-wp.taskQueue:
			if task != nil {
				task()
			}
		case <-wp.quit:
			return
		}
	}
}

// Submit adds a task to the worker pool
func (wp *WorkerPool) Submit(task func()) {
	select {
	case wp.taskQueue <- task:
		// Task submitted successfully
	case <-wp.quit:
		// Pool is shutting down
		return
	}
}

// Close shuts down the worker pool and waits for all workers to finish
func (wp *WorkerPool) Close() {
	close(wp.quit)
	wp.wg.Wait()
}

// GlobalPools provides globally accessible pools for common operations
var GlobalPools = struct {
	Buffer  *BufferPool
	Float32 *Float32Pool
	String  *StringPool
	Results *ResultPool
}{
	Buffer:  NewBufferPool(1024),  // 1KB initial buffer
	Float32: NewFloat32Pool(1024), // Space for ~1K floats
	String:  NewStringPool(100),   // Space for ~100 strings
	Results: NewResultPool(50),    // Space for ~50 results
}

// MemoryStats provides memory usage statistics
type MemoryStats struct {
	BuffersInUse   int64
	Float32sInUse  int64
	StringsInUse   int64
	ResultsInUse   int64
	TotalAllocated int64
	TotalRecycled  int64
}

// Stats returns current memory pool statistics
func (bp *BufferPool) Stats() MemoryStats {
	// Note: sync.Pool doesn't provide direct stats, so this is a placeholder
	// In a production system, you might want to wrap sync.Pool to track statistics
	return MemoryStats{}
}

// Batch provides utilities for batch processing
type Batch struct {
	Size     int
	Parallel bool
	Workers  int
}

// NewBatch creates a new batch processor configuration
func NewBatch(size int, parallel bool, workers int) *Batch {
	return &Batch{
		Size:     size,
		Parallel: parallel,
		Workers:  workers,
	}
}

// Process processes items in batches using the provided function
func (b *Batch) Process(items []interface{}, processFn func(interface{}) error) error {
	if !b.Parallel {
		// Sequential processing
		for _, item := range items {
			if err := processFn(item); err != nil {
				return err
			}
		}
		return nil
	}

	// Parallel processing
	wp := NewWorkerPool(b.Workers)
	defer wp.Close()

	errChan := make(chan error, len(items))

	for _, item := range items {
		item := item // Capture loop variable
		wp.Submit(func() {
			if err := processFn(item); err != nil {
				errChan <- err
			} else {
				errChan <- nil
			}
		})
	}

	// Collect results
	for i := 0; i < len(items); i++ {
		if err := <-errChan; err != nil {
			return err
		}
	}

	return nil
}

// Cache provides a simple LRU cache implementation
type Cache struct {
	capacity int
	items    map[string]*cacheItem
	head     *cacheItem
	tail     *cacheItem
	mutex    sync.RWMutex
}

type cacheItem struct {
	key   string
	value interface{}
	prev  *cacheItem
	next  *cacheItem
}

// NewCache creates a new LRU cache with specified capacity
func NewCache(capacity int) *Cache {
	c := &Cache{
		capacity: capacity,
		items:    make(map[string]*cacheItem),
	}

	// Initialize sentinel nodes
	c.head = &cacheItem{}
	c.tail = &cacheItem{}
	c.head.next = c.tail
	c.tail.prev = c.head

	return c
}

// Get retrieves a value from the cache
func (c *Cache) Get(key string) (interface{}, bool) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if item, exists := c.items[key]; exists {
		c.moveToHead(item)
		return item.value, true
	}

	return nil, false
}

// Put stores a value in the cache
func (c *Cache) Put(key string, value interface{}) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if item, exists := c.items[key]; exists {
		item.value = value
		c.moveToHead(item)
		return
	}

	newItem := &cacheItem{
		key:   key,
		value: value,
	}

	c.items[key] = newItem
	c.addToHead(newItem)

	if len(c.items) > c.capacity {
		tail := c.removeTail()
		delete(c.items, tail.key)
	}
}

// moveToHead moves an item to the head of the list
func (c *Cache) moveToHead(item *cacheItem) {
	c.removeItem(item)
	c.addToHead(item)
}

// addToHead adds an item to the head of the list
func (c *Cache) addToHead(item *cacheItem) {
	item.prev = c.head
	item.next = c.head.next
	c.head.next.prev = item
	c.head.next = item
}

// removeItem removes an item from the list
func (c *Cache) removeItem(item *cacheItem) {
	item.prev.next = item.next
	item.next.prev = item.prev
}

// removeTail removes and returns the tail item
func (c *Cache) removeTail() *cacheItem {
	lastItem := c.tail.prev
	c.removeItem(lastItem)
	return lastItem
}

// Size returns the current size of the cache
func (c *Cache) Size() int {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return len(c.items)
}

// Clear removes all items from the cache
func (c *Cache) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.items = make(map[string]*cacheItem)
	c.head.next = c.tail
	c.tail.prev = c.head
}
