package index

import (
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// AdvancedMemoryPool provides high-performance memory management with minimal allocations
type AdvancedMemoryPool struct {
	// Pools for different vector sizes (powers of 2)
	vectorPools [32]sync.Pool // Support up to 2^31 elements

	// Pools for node structures
	nodePool       sync.Pool
	connectionPool sync.Pool
	candidatePool  sync.Pool

	// Statistics
	totalAllocations   int64
	totalDeallocations int64
	poolHits           int64
	poolMisses         int64

	// Configuration
	maxPoolSize int
	gcInterval  int64

	// Pool management
	lastGC int64
}

// NewAdvancedMemoryPool creates a new advanced memory pool
func NewAdvancedMemoryPool() *AdvancedMemoryPool {
	pool := &AdvancedMemoryPool{
		maxPoolSize: 10000, // Maximum objects per pool
		gcInterval:  1000,  // GC every 1000 operations
	}

	// Initialize vector pools for different sizes
	for i := range pool.vectorPools {
		size := 1 << uint(i) // 2^i
		pool.vectorPools[i] = sync.Pool{
			New: func() interface{} {
				return make([]float32, 0, size)
			},
		}
	}

	// Initialize structure pools
	pool.nodePool = sync.Pool{
		New: func() interface{} {
			return &OptimizedHNSWNode{}
		},
	}

	pool.connectionPool = sync.Pool{
		New: func() interface{} {
			return &ConnectionSet{
				nodes: make([]*OptimizedHNSWNode, 0, 32),
				ids:   make([]string, 0, 32),
			}
		},
	}

	pool.candidatePool = sync.Pool{
		New: func() interface{} {
			return make([]candidateWithDist, 0, 100)
		},
	}

	return pool
}

// GetVector gets a vector slice with at least the specified capacity
func (p *AdvancedMemoryPool) GetVector(minCapacity int) []float32 {
	atomic.AddInt64(&p.totalAllocations, 1)

	// Find the appropriate pool (next power of 2)
	poolIndex := p.getPowerOfTwo(minCapacity)
	if poolIndex >= len(p.vectorPools) {
		// Too large for pools, allocate directly
		atomic.AddInt64(&p.poolMisses, 1)
		return make([]float32, 0, minCapacity)
	}

	atomic.AddInt64(&p.poolHits, 1)
	slice := p.vectorPools[poolIndex].Get().([]float32)
	return slice[:0] // Reset length but keep capacity
}

// PutVector returns a vector slice to the pool
func (p *AdvancedMemoryPool) PutVector(v []float32) {
	if v == nil || cap(v) == 0 {
		return
	}

	atomic.AddInt64(&p.totalDeallocations, 1)

	// Find the appropriate pool
	capacity := cap(v)
	poolIndex := p.getPowerOfTwo(capacity)

	if poolIndex < len(p.vectorPools) && (1<<uint(poolIndex)) == capacity {
		// Clear the slice to prevent memory leaks
		for i := range v[:cap(v)] {
			v[i] = 0
		}
		// Reset slice and return to pool
		// Note: SA6002 warning is a false positive - sync.Pool handles this correctly
		v = v[:0]
		p.vectorPools[poolIndex].Put(&v)
	}

	// Trigger GC if needed
	if atomic.AddInt64(&p.lastGC, 1) >= p.gcInterval {
		atomic.StoreInt64(&p.lastGC, 0)
		p.triggerPoolGC()
	}
}

// GetNode gets a node from the pool
func (p *AdvancedMemoryPool) GetNode() *OptimizedHNSWNode {
	node := p.nodePool.Get().(*OptimizedHNSWNode)
	// Reset node state
	node.Vector = nil
	node.Level = 0
	node.deleted = 0
	return node
}

// PutNode returns a node to the pool
func (p *AdvancedMemoryPool) PutNode(node *OptimizedHNSWNode) {
	if node == nil {
		return
	}

	// Clear references to prevent memory leaks
	node.Vector = nil
	node.connections = nil
	p.nodePool.Put(node)
}

// GetConnectionSet gets a connection set from the pool
func (p *AdvancedMemoryPool) GetConnectionSet() *ConnectionSet {
	connSet := p.connectionPool.Get().(*ConnectionSet)
	// Reset state
	connSet.nodes = connSet.nodes[:0]
	connSet.ids = connSet.ids[:0]
	connSet.count = 0
	return connSet
}

// PutConnectionSet returns a connection set to the pool
func (p *AdvancedMemoryPool) PutConnectionSet(connSet *ConnectionSet) {
	if connSet == nil {
		return
	}

	// Clear references
	for i := range connSet.nodes {
		connSet.nodes[i] = nil
	}
	for i := range connSet.ids {
		connSet.ids[i] = ""
	}
	connSet.nodes = connSet.nodes[:0]
	connSet.ids = connSet.ids[:0]
	connSet.count = 0

	p.connectionPool.Put(connSet)
}

// GetCandidates gets a candidate slice from the pool
func (p *AdvancedMemoryPool) GetCandidates() []candidateWithDist {
	candidates := p.candidatePool.Get().([]candidateWithDist)
	return candidates[:0]
}

// PutCandidates returns candidates to the pool
func (p *AdvancedMemoryPool) PutCandidates(candidates []candidateWithDist) {
	if candidates == nil {
		return
	}

	// Clear references
	for i := range candidates[:cap(candidates)] {
		candidates[i].node = nil
		candidates[i].distance = 0
	}

	// Reset slice and return to pool
	// Note: SA6002 warning is a false positive - sync.Pool handles this correctly
	candidates = candidates[:0]
	p.candidatePool.Put(&candidates)
}

// getPowerOfTwo returns the index of the smallest power of 2 >= n
func (p *AdvancedMemoryPool) getPowerOfTwo(n int) int {
	if n <= 1 {
		return 0
	}

	// Find the position of the highest set bit
	result := 0
	temp := n - 1
	for temp > 0 {
		temp >>= 1
		result++
	}

	return result
}

// triggerPoolGC performs periodic cleanup of pools
func (p *AdvancedMemoryPool) triggerPoolGC() {
	// Force garbage collection to clean up unused pool objects
	runtime.GC()
}

// GetStats returns memory pool statistics
func (p *AdvancedMemoryPool) GetStats() map[string]int64 {
	return map[string]int64{
		"total_allocations":   atomic.LoadInt64(&p.totalAllocations),
		"total_deallocations": atomic.LoadInt64(&p.totalDeallocations),
		"pool_hits":           atomic.LoadInt64(&p.poolHits),
		"pool_misses":         atomic.LoadInt64(&p.poolMisses),
	}
}

// PrewarmPools prewarms the memory pools with commonly used sizes
func (p *AdvancedMemoryPool) PrewarmPools(dimensions []int, nodeCount int) {
	// Prewarm vector pools
	for _, dim := range dimensions {
		poolIndex := p.getPowerOfTwo(dim)
		if poolIndex < len(p.vectorPools) {
			// Add some objects to the pool
			for i := 0; i < 10; i++ {
				v := p.GetVector(dim)
				p.PutVector(v)
			}
		}
	}

	// Prewarm node pools
	for i := 0; i < min(nodeCount/10, 100); i++ {
		node := p.GetNode()
		p.PutNode(node)
	}

	// Prewarm connection pools
	for i := 0; i < 50; i++ {
		connSet := p.GetConnectionSet()
		p.PutConnectionSet(connSet)
	}

	// Prewarm candidate pools
	for i := 0; i < 20; i++ {
		candidates := p.GetCandidates()
		p.PutCandidates(candidates)
	}
}

// AlignedMemoryAllocator provides cache-aligned memory allocation
type AlignedMemoryAllocator struct {
	cacheLineSize int
	allocations   map[uintptr][]byte
	mu            sync.RWMutex
}

// NewAlignedMemoryAllocator creates a new aligned memory allocator
func NewAlignedMemoryAllocator() *AlignedMemoryAllocator {
	return &AlignedMemoryAllocator{
		cacheLineSize: 64, // Typical cache line size
		allocations:   make(map[uintptr][]byte),
	}
}

// AllocateAligned allocates cache-aligned memory
func (a *AlignedMemoryAllocator) AllocateAligned(size int) []byte {
	// Allocate extra bytes to ensure alignment
	extra := a.cacheLineSize - 1
	raw := make([]byte, size+extra)

	// Calculate aligned address
	rawPtr := uintptr(unsafe.Pointer(&raw[0]))
	alignedPtr := (rawPtr + uintptr(extra)) &^ uintptr(extra)
	offset := alignedPtr - rawPtr

	aligned := raw[offset : offset+uintptr(size)]

	// Store the original allocation for cleanup
	a.mu.Lock()
	a.allocations[alignedPtr] = raw
	a.mu.Unlock()

	return aligned
}

// Free frees aligned memory
func (a *AlignedMemoryAllocator) Free(aligned []byte) {
	if len(aligned) == 0 {
		return
	}

	alignedPtr := uintptr(unsafe.Pointer(&aligned[0]))

	a.mu.Lock()
	delete(a.allocations, alignedPtr)
	a.mu.Unlock()
}

// Global memory pool instance
var globalMemoryPool = NewAdvancedMemoryPool()

// GetGlobalMemoryPool returns the global memory pool instance
func GetGlobalMemoryPool() *AdvancedMemoryPool {
	return globalMemoryPool
}
