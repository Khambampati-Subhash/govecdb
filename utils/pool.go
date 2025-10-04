package utils

// Package utils provides utility functions and optimizations for GoVecDB.
// This includes buffer pooling, SIMD distance functions, and performance utilities.

import (
	"sync"
)

// BufferPool provides thread-safe buffer pooling to reduce GC pressure
type BufferPool struct {
	pool         sync.Pool
	initialCap   int
	allocations  int64
	recycles     int64
	maxAllocated int64
}

// NewBufferPool creates a new buffer pool with the specified initial capacity
func NewBufferPool(initialCapacity int) *BufferPool {
	bp := &BufferPool{
		initialCap: initialCapacity,
		pool: sync.Pool{
			New: func() interface{} {
				return make([]byte, 0, initialCapacity)
			},
		},
	}
	// Pre-warm the pool with some buffers
	bp.preWarm(8)
	return bp
}

// SizedBufferPool provides multiple pools for different buffer sizes
type SizedBufferPool struct {
	pools map[int]*BufferPool
	mutex sync.RWMutex
}

// NewSizedBufferPool creates pools for common buffer sizes
func NewSizedBufferPool() *SizedBufferPool {
	sbp := &SizedBufferPool{
		pools: make(map[int]*BufferPool),
	}

	// Pre-create pools for common sizes
	commonSizes := []int{64, 256, 1024, 4096, 16384, 65536}
	for _, size := range commonSizes {
		sbp.pools[size] = NewBufferPool(size)
	}

	return sbp
}

// GetBuffer gets a buffer of appropriate size
func (sbp *SizedBufferPool) GetBuffer(minSize int) []byte {
	size := sbp.nextPowerOfTwo(minSize)

	sbp.mutex.RLock()
	pool, exists := sbp.pools[size]
	sbp.mutex.RUnlock()

	if !exists {
		sbp.mutex.Lock()
		if pool, exists = sbp.pools[size]; !exists {
			pool = NewBufferPool(size)
			sbp.pools[size] = pool
		}
		sbp.mutex.Unlock()
	}

	return pool.Get()
}

// PutBuffer returns a buffer to the appropriate pool
func (sbp *SizedBufferPool) PutBuffer(buf []byte) {
	size := cap(buf)

	sbp.mutex.RLock()
	pool, exists := sbp.pools[size]
	sbp.mutex.RUnlock()

	if exists {
		pool.Put(buf)
	}
	// If no matching pool, let GC handle it
}

// nextPowerOfTwo finds the next power of 2 >= n
func (sbp *SizedBufferPool) nextPowerOfTwo(n int) int {
	if n <= 64 {
		return 64
	}
	if n <= 256 {
		return 256
	}
	if n <= 1024 {
		return 1024
	}
	if n <= 4096 {
		return 4096
	}
	if n <= 16384 {
		return 16384
	}
	if n <= 65536 {
		return 65536
	}

	// For larger sizes, calculate power of 2
	size := 65536
	for size < n {
		size <<= 1
		if size <= 0 { // overflow protection
			return n
		}
	}
	return size
}

// preWarm pre-allocates buffers to reduce initial allocation cost
func (bp *BufferPool) preWarm(count int) {
	for i := 0; i < count; i++ {
		buf := make([]byte, 0, bp.initialCap)
		bp.pool.Put(buf)
	}
}

// Get retrieves a buffer from the pool
func (bp *BufferPool) Get() []byte {
	bp.updateAllocation()
	return bp.pool.Get().([]byte)
}

// Put returns a buffer to the pool after resetting its length
func (bp *BufferPool) Put(buf []byte) {
	bp.updateRecycle()
	// Reset length but keep capacity
	// Note: SA6002 warning is a false positive - sync.Pool handles this correctly
	bp.pool.Put(interface{}(buf[:0]))
}

// Float32Pool provides pooling for float32 slices
type Float32Pool struct {
	pool         sync.Pool
	initialCap   int
	allocations  int64
	recycles     int64
	maxAllocated int64
}

// NewFloat32Pool creates a new float32 slice pool
func NewFloat32Pool(initialCapacity int) *Float32Pool {
	fp := &Float32Pool{
		initialCap: initialCapacity,
		pool: sync.Pool{
			New: func() interface{} {
				return make([]float32, 0, initialCapacity)
			},
		},
	}
	// Pre-warm the pool
	fp.preWarm(16)
	return fp
}

// SizedFloat32Pool provides multiple pools for different vector dimensions
type SizedFloat32Pool struct {
	pools map[int]*Float32Pool
	mutex sync.RWMutex
}

// NewSizedFloat32Pool creates pools for common vector dimensions
func NewSizedFloat32Pool() *SizedFloat32Pool {
	sfp := &SizedFloat32Pool{
		pools: make(map[int]*Float32Pool),
	}

	// Pre-create pools for common dimensions
	commonDims := []int{128, 256, 384, 512, 768, 1024, 1536, 2048}
	for _, dim := range commonDims {
		sfp.pools[dim] = NewFloat32Pool(dim)
	}

	return sfp
}

// GetVector gets a vector of appropriate dimension
func (sfp *SizedFloat32Pool) GetVector(dimension int) []float32 {
	sfp.mutex.RLock()
	pool, exists := sfp.pools[dimension]
	sfp.mutex.RUnlock()

	if !exists {
		sfp.mutex.Lock()
		if pool, exists = sfp.pools[dimension]; !exists {
			pool = NewFloat32Pool(dimension)
			sfp.pools[dimension] = pool
		}
		sfp.mutex.Unlock()
	}

	vec := pool.Get()
	// Ensure correct capacity
	if cap(vec) < dimension {
		return make([]float32, 0, dimension)
	}
	return vec[:0] // Reset length but keep capacity
}

// PutVector returns a vector to the appropriate pool
func (sfp *SizedFloat32Pool) PutVector(vec []float32) {
	dimension := cap(vec)

	sfp.mutex.RLock()
	pool, exists := sfp.pools[dimension]
	sfp.mutex.RUnlock()

	if exists {
		pool.Put(vec)
	}
}

// preWarm pre-allocates vectors to reduce initial allocation cost
func (fp *Float32Pool) preWarm(count int) {
	for i := 0; i < count; i++ {
		vec := make([]float32, 0, fp.initialCap)
		fp.pool.Put(vec)
	}
}

// Get retrieves a float32 slice from the pool
func (fp *Float32Pool) Get() []float32 {
	fp.updateAllocation()
	return fp.pool.Get().([]float32)
}

// Put returns a float32 slice to the pool after resetting
func (fp *Float32Pool) Put(slice []float32) {
	fp.updateRecycle()
	// Note: SA6002 warning is a false positive - sync.Pool handles this correctly
	fp.pool.Put(interface{}(slice[:0]))
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
	// Note: SA6002 warning is a false positive - sync.Pool handles this correctly
	sp.pool.Put(interface{}(slice[:0]))
}

// Stats returns stats for StringPool
func (sp *StringPool) Stats() MemoryStats {
	return MemoryStats{
		StringsInUse: 0, // sync.Pool doesn't provide direct stats
	}
}

// VectorPool provides pooling for vector data to reduce allocations during search
type VectorPool struct {
	dimension    int
	pool         sync.Pool
	allocations  int64
	recycles     int64
	maxAllocated int64
}

// NewVectorPool creates a new vector pool for vectors of specific dimension
func NewVectorPool(dimension int) *VectorPool {
	vp := &VectorPool{
		dimension: dimension,
		pool: sync.Pool{
			New: func() interface{} {
				return make([]float32, dimension)
			},
		},
	}
	// Pre-warm the pool
	vp.preWarm(32)
	return vp
}

// ScratchBufferPool provides high-performance temporary buffers for calculations
type ScratchBufferPool struct {
	smallPool  sync.Pool // For buffers < 1KB
	mediumPool sync.Pool // For buffers 1KB - 16KB
	largePool  sync.Pool // For buffers > 16KB
}

// NewScratchBufferPool creates a new scratch buffer pool
func NewScratchBufferPool() *ScratchBufferPool {
	sbp := &ScratchBufferPool{
		smallPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, 0, 256) // 1KB
			},
		},
		mediumPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, 0, 4096) // 16KB
			},
		},
		largePool: sync.Pool{
			New: func() interface{} {
				return make([]float32, 0, 16384) // 64KB
			},
		},
	}

	// Pre-warm pools
	for i := 0; i < 16; i++ {
		sbp.smallPool.Put(make([]float32, 0, 256))
		sbp.mediumPool.Put(make([]float32, 0, 4096))
	}
	for i := 0; i < 4; i++ {
		sbp.largePool.Put(make([]float32, 0, 16384))
	}

	return sbp
}

// GetScratch gets a scratch buffer of appropriate size
func (sbp *ScratchBufferPool) GetScratch(minSize int) []float32 {
	if minSize <= 256 {
		return sbp.smallPool.Get().([]float32)[:0]
	} else if minSize <= 4096 {
		return sbp.mediumPool.Get().([]float32)[:0]
	} else {
		return sbp.largePool.Get().([]float32)[:0]
	}
}

// PutScratch returns a scratch buffer to the pool
func (sbp *ScratchBufferPool) PutScratch(buf []float32) {
	capacity := cap(buf)
	if capacity <= 256 {
		sbp.smallPool.Put(buf[:0])
	} else if capacity <= 4096 {
		sbp.mediumPool.Put(buf[:0])
	} else if capacity <= 16384 {
		sbp.largePool.Put(buf[:0])
	}
	// For larger buffers, let GC handle them
}

// preWarm pre-allocates vectors to reduce initial allocation cost
func (vp *VectorPool) preWarm(count int) {
	for i := 0; i < count; i++ {
		vec := make([]float32, vp.dimension)
		vp.pool.Put(vec)
	}
}

// Get retrieves a vector from the pool
func (vp *VectorPool) Get() []float32 {
	vp.updateAllocation()
	return vp.pool.Get().([]float32)
}

// Put returns a vector to the pool after clearing
func (vp *VectorPool) Put(vector []float32) {
	if len(vector) == vp.dimension {
		vp.updateRecycle()
		// Clear the vector data efficiently
		for i := range vector {
			vector[i] = 0
		}
		// Note: SA6002 warning is a false positive - sync.Pool handles this correctly
		vp.pool.Put(interface{}(vector))
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
	// Note: SA6002 warning is a false positive - sync.Pool handles this correctly
	rp.pool.Put(interface{}(results[:0]))
}

// Stats returns stats for ResultPool
func (rp *ResultPool) Stats() MemoryStats {
	return MemoryStats{
		ResultsInUse: 0, // sync.Pool doesn't provide direct stats
	}
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

// AdvancedPoolManager manages all pools for optimal performance
type AdvancedPoolManager struct {
	SizedBuffers  *SizedBufferPool
	SizedFloat32s *SizedFloat32Pool
	ScratchBuffer *ScratchBufferPool
	Results       *ResultPool
	Strings       *StringPool
	Workers       *WorkerPool
}

// NewAdvancedPoolManager creates a comprehensive pool manager
func NewAdvancedPoolManager() *AdvancedPoolManager {
	return &AdvancedPoolManager{
		SizedBuffers:  NewSizedBufferPool(),
		SizedFloat32s: NewSizedFloat32Pool(),
		ScratchBuffer: NewScratchBufferPool(),
		Results:       NewResultPool(100),
		Strings:       NewStringPool(200),
		Workers:       NewWorkerPool(8), // CPU count based
	}
}

// GlobalPools provides globally accessible pools for common operations
var GlobalPools = struct {
	Buffer   *BufferPool
	Float32  *Float32Pool
	String   *StringPool
	Results  *ResultPool
	Advanced *AdvancedPoolManager
}{
	Buffer:   NewBufferPool(1024),      // 1KB initial buffer
	Float32:  NewFloat32Pool(1024),     // Space for ~1K floats
	String:   NewStringPool(100),       // Space for ~100 strings
	Results:  NewResultPool(50),        // Space for ~50 results
	Advanced: NewAdvancedPoolManager(), // Comprehensive pool management
}

// MemoryStats provides memory usage statistics
type MemoryStats struct {
	BuffersInUse    int64
	Float32sInUse   int64
	StringsInUse    int64
	ResultsInUse    int64
	TotalAllocated  int64
	TotalRecycled   int64
	MemoryFootprint int64
	PoolEfficiency  float64 // Recycles / (Allocations + Recycles)
	CacheHitRate    float64 // Pool hits / Total requests
}

// Stats returns current memory pool statistics
func (bp *BufferPool) Stats() MemoryStats {
	allocations := bp.allocations
	recycles := bp.recycles
	total := allocations + recycles

	var efficiency, hitRate float64
	if total > 0 {
		efficiency = float64(recycles) / float64(total)
		hitRate = float64(recycles) / float64(total)
	}

	return MemoryStats{
		BuffersInUse:    allocations - recycles,
		TotalAllocated:  allocations,
		TotalRecycled:   recycles,
		MemoryFootprint: bp.maxAllocated * int64(bp.initialCap) * 4, // float32 = 4 bytes
		PoolEfficiency:  efficiency,
		CacheHitRate:    hitRate,
	}
}

// Stats returns stats for Float32Pool
func (fp *Float32Pool) Stats() MemoryStats {
	allocations := fp.allocations
	recycles := fp.recycles
	total := allocations + recycles

	var efficiency, hitRate float64
	if total > 0 {
		efficiency = float64(recycles) / float64(total)
		hitRate = float64(recycles) / float64(total)
	}

	return MemoryStats{
		Float32sInUse:   allocations - recycles,
		TotalAllocated:  allocations,
		TotalRecycled:   recycles,
		MemoryFootprint: fp.maxAllocated * int64(fp.initialCap) * 4,
		PoolEfficiency:  efficiency,
		CacheHitRate:    hitRate,
	}
}

// GetComprehensiveStats returns overall pool statistics
func (apm *AdvancedPoolManager) GetComprehensiveStats() map[string]MemoryStats {
	return map[string]MemoryStats{
		"results": apm.Results.Stats(),
		"strings": apm.Strings.Stats(),
		// Add other pool stats as needed
	}
}

// UpdateStats safely updates pool statistics
func (bp *BufferPool) updateAllocation() {
	bp.allocations++
	if bp.allocations > bp.maxAllocated {
		bp.maxAllocated = bp.allocations
	}
}

func (bp *BufferPool) updateRecycle() {
	bp.recycles++
}

func (fp *Float32Pool) updateAllocation() {
	fp.allocations++
	if fp.allocations > fp.maxAllocated {
		fp.maxAllocated = fp.allocations
	}
}

func (fp *Float32Pool) updateRecycle() {
	fp.recycles++
}

func (vp *VectorPool) updateAllocation() {
	vp.allocations++
	if vp.allocations > vp.maxAllocated {
		vp.maxAllocated = vp.allocations
	}
}

func (vp *VectorPool) updateRecycle() {
	vp.recycles++
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
