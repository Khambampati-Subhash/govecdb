# 🔄 Synchronization in GoVecDB

## 🎯 Overview

GoVecDB is designed for high-concurrency environments with sophisticated synchronization mechanisms that ensure data consistency, thread safety, and optimal performance. This document explains how synchronization works at different levels of the system.

## 🏗️ Synchronization Architecture

### Multi-Level Synchronization Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                 Application Level                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Collection Lock │  │ Index Lock      │                  │
│  │ (RWMutex)       │  │ (RWMutex)       │                  │
│  └─────────────────┘  └─────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│                 Node Level                                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Node Mutex      │  │ Connection Lock │                  │
│  │ (RWMutex)       │  │ (Mutex)         │                  │
│  └─────────────────┘  └─────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│                 Storage Level                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ WAL Lock        │  │ Memory Store    │                  │
│  │ (Mutex)         │  │ (Lock-free)     │                  │
│  └─────────────────┘  └─────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│                 Cluster Level                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Raft Consensus  │  │ Hash Ring       │                  │
│  │ (Leader/Follow) │  │ (Atomic Updates)│                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## 🔒 Thread Safety Mechanisms

### 1. HNSW Index Synchronization

The HNSW index uses sophisticated locking strategies for optimal concurrent performance:

```go
// HNSW Index with configurable thread safety
type HNSWIndex struct {
    graph  *HNSWGraph
    config *Config
    
    // Thread safety control
    mu sync.RWMutex  // Primary index lock
    
    // Statistics (atomic for lock-free access)
    insertCount  int64
    searchCount  int64
}

// Thread-safe search operation
func (idx *HNSWIndex) Search(query []float32, k int) ([]*SearchResult, error) {
    if idx.config.ThreadSafe {
        idx.mu.RLock()         // Acquire read lock
        defer idx.mu.RUnlock() // Release on return
    }
    
    // Perform search (multiple readers can execute concurrently)
    results, err := idx.graph.Search(query, k, nil)
    if err != nil {
        return nil, err
    }
    
    // Atomic increment of search counter
    atomic.AddInt64(&idx.searchCount, 1)
    
    return results, nil
}

// Thread-safe insert operation
func (idx *HNSWIndex) Add(vector *Vector) error {
    if idx.config.ThreadSafe {
        idx.mu.Lock()         // Acquire write lock (exclusive)
        defer idx.mu.Unlock() // Release on return
    }
    
    // Perform insertion (only one writer at a time)
    err := idx.graph.Insert(vector)
    if err != nil {
        return err
    }
    
    // Atomic increment of insert counter
    atomic.AddInt64(&idx.insertCount, 1)
    
    return nil
}
```

### 2. Node-Level Synchronization

Each HNSW node has its own synchronization for fine-grained concurrency:

```go
type HNSWNode struct {
    Vector      *Vector
    Level       int
    connections []map[string]*HNSWNode
    
    // Fine-grained synchronization
    mu      sync.RWMutex  // Node-specific lock
    deleted bool          // Soft delete flag
}

// Thread-safe connection management
func (n *HNSWNode) AddConnection(other *HNSWNode, layer int) {
    n.mu.Lock()
    defer n.mu.Unlock()
    
    if !n.deleted && layer <= n.Level {
        n.connections[layer][other.Vector.ID] = other
    }
}

func (n *HNSWNode) GetConnections(layer int) map[string]*HNSWNode {
    n.mu.RLock()
    defer n.mu.RUnlock()
    
    if n.deleted || layer > n.Level {
        return nil
    }
    
    // Return a copy to prevent concurrent modification
    connections := make(map[string]*HNSWNode, len(n.connections[layer]))
    for id, node := range n.connections[layer] {
        connections[id] = node
    }
    
    return connections
}
```

### 3. Lock-Free Data Structures

GoVecDB uses lock-free data structures where possible for maximum performance:

```go
// Lock-free SafeMap implementation
type SafeMap[K comparable, V any] struct {
    data sync.Map  // Go's built-in lock-free map
}

func (sm *SafeMap[K, V]) Get(key K) (V, bool) {
    value, ok := sm.data.Load(key)
    if !ok {
        var zero V
        return zero, false
    }
    return value.(V), true
}

func (sm *SafeMap[K, V]) Set(key K, value V) {
    sm.data.Store(key, value)
}

func (sm *SafeMap[K, V]) Delete(key K) {
    sm.data.Delete(key)
}

// Atomic counters for statistics
type AtomicStats struct {
    nodeCount    int64
    edgeCount    int64
    searchCount  int64
    insertCount  int64
}

func (as *AtomicStats) IncrementNodeCount() {
    atomic.AddInt64(&as.nodeCount, 1)
}

func (as *AtomicStats) GetNodeCount() int64 {
    return atomic.LoadInt64(&as.nodeCount)
}
```

## 🏪 Storage Layer Synchronization

### WAL (Write-Ahead Log) Synchronization

```go
type WAL struct {
    file       *os.File
    mu         sync.Mutex  // Protects write operations
    lsn        int64       // Last sequence number (atomic)
    syncPolicy SyncPolicy
}

func (w *WAL) WriteRecord(record *WALRecord) (LSN, error) {
    // Serialize record
    data, err := record.Marshal()
    if err != nil {
        return 0, err
    }
    
    // Acquire exclusive write lock
    w.mu.Lock()
    defer w.mu.Unlock()
    
    // Assign LSN atomically
    lsn := atomic.AddInt64(&w.lsn, 1)
    record.LSN = LSN(lsn)
    
    // Write to file
    _, err = w.file.Write(data)
    if err != nil {
        return 0, err
    }
    
    // Sync based on policy
    if w.syncPolicy == SyncEveryWrite {
        err = w.file.Sync()
    }
    
    return LSN(lsn), err
}

// Concurrent batch writing
func (w *WAL) WriteBatch(records []*WALRecord) error {
    w.mu.Lock()
    defer w.mu.Unlock()
    
    // Pre-allocate buffer for entire batch
    var buffer bytes.Buffer
    
    for _, record := range records {
        lsn := atomic.AddInt64(&w.lsn, 1)
        record.LSN = LSN(lsn)
        
        data, err := record.Marshal()
        if err != nil {
            return err
        }
        
        buffer.Write(data)
    }
    
    // Single write operation for entire batch
    _, err := w.file.Write(buffer.Bytes())
    if err != nil {
        return err
    }
    
    return w.file.Sync()
}
```

### Memory Store Synchronization

```go
type MemoryStore struct {
    data    *SafeMap[string, *Vector]  // Lock-free map
    filters *SafeMap[string, Filter]   // Lock-free filter storage
    
    // Statistics (atomic)
    size       int64
    operations int64
}

func (ms *MemoryStore) Put(vector *Vector) error {
    // Validate vector
    if vector == nil || vector.ID == "" {
        return ErrInvalidVector
    }
    
    // Check if update or insert
    _, exists := ms.data.Get(vector.ID)
    
    // Store vector (lock-free operation)
    ms.data.Set(vector.ID, vector)
    
    // Update size counter atomically
    if !exists {
        atomic.AddInt64(&ms.size, 1)
    }
    
    // Increment operations counter
    atomic.AddInt64(&ms.operations, 1)
    
    return nil
}

func (ms *MemoryStore) Get(id string) (*Vector, error) {
    // Lock-free read
    vector, exists := ms.data.Get(id)
    if !exists {
        return nil, ErrVectorNotFound
    }
    
    // Increment operations counter
    atomic.AddInt64(&ms.operations, 1)
    
    return vector, nil
}
```

## 🌐 Cluster-Level Synchronization

### Raft Consensus Protocol

GoVecDB uses the Raft consensus algorithm for distributed coordination:

```go
type RaftNode struct {
    id          string
    state       NodeState  // Leader, Follower, Candidate
    currentTerm int64      // Current election term (atomic)
    votedFor    string     // Candidate voted for in current term
    log         []LogEntry // Replicated log entries
    
    // Synchronization
    mu          sync.RWMutex  // Protects node state
    commitIndex int64         // Highest log entry known to be committed (atomic)
    lastApplied int64         // Highest log entry applied to state machine (atomic)
    
    // Leader state (protected by mu)
    nextIndex   map[string]int64  // For each server, index of next log entry to send
    matchIndex  map[string]int64  // For each server, index of highest log entry replicated
}

// Thread-safe term management
func (rn *RaftNode) GetCurrentTerm() int64 {
    return atomic.LoadInt64(&rn.currentTerm)
}

func (rn *RaftNode) SetCurrentTerm(term int64) {
    atomic.StoreInt64(&rn.currentTerm, term)
}

// Leader election with proper synchronization
func (rn *RaftNode) RequestVote(args *RequestVoteArgs) *RequestVoteReply {
    rn.mu.Lock()
    defer rn.mu.Unlock()
    
    reply := &RequestVoteReply{
        Term:        rn.GetCurrentTerm(),
        VoteGranted: false,
    }
    
    // If candidate's term is newer, update our term
    if args.Term > rn.GetCurrentTerm() {
        rn.SetCurrentTerm(args.Term)
        rn.votedFor = ""
        rn.state = Follower
    }
    
    // Grant vote if:
    // 1. Haven't voted in this term, or voted for same candidate
    // 2. Candidate's log is at least as up-to-date as ours
    if (rn.votedFor == "" || rn.votedFor == args.CandidateID) &&
       rn.isLogUpToDate(args.LastLogIndex, args.LastLogTerm) {
        rn.votedFor = args.CandidateID
        reply.VoteGranted = true
    }
    
    return reply
}

// Log replication with consistency guarantees
func (rn *RaftNode) AppendEntries(args *AppendEntriesArgs) *AppendEntriesReply {
    rn.mu.Lock()
    defer rn.mu.Unlock()
    
    reply := &AppendEntriesReply{
        Term:    rn.GetCurrentTerm(),
        Success: false,
    }
    
    // Update term if leader's term is newer
    if args.Term > rn.GetCurrentTerm() {
        rn.SetCurrentTerm(args.Term)
        rn.votedFor = ""
        rn.state = Follower
    }
    
    // Reject if term is outdated
    if args.Term < rn.GetCurrentTerm() {
        return reply
    }
    
    // Check log consistency
    if args.PrevLogIndex > 0 {
        if len(rn.log) < int(args.PrevLogIndex) ||
           rn.log[args.PrevLogIndex-1].Term != args.PrevLogTerm {
            return reply
        }
    }
    
    // Append entries and update commit index
    rn.appendLogEntries(args.Entries, args.PrevLogIndex)
    
    if args.LeaderCommit > atomic.LoadInt64(&rn.commitIndex) {
        newCommitIndex := min(args.LeaderCommit, int64(len(rn.log)))
        atomic.StoreInt64(&rn.commitIndex, newCommitIndex)
    }
    
    reply.Success = true
    return reply
}
```

### Hash Ring Synchronization

Consistent hashing with atomic updates:

```go
type HashRing struct {
    nodes    []string           // Sorted node identifiers
    ring     map[uint32]string  // Hash -> Node mapping
    replicas int               // Virtual nodes per physical node
    
    // Synchronization
    mu sync.RWMutex  // Protects ring modifications
}

func (hr *HashRing) AddNode(node string) {
    hr.mu.Lock()
    defer hr.mu.Unlock()
    
    // Add virtual nodes
    for i := 0; i < hr.replicas; i++ {
        hash := hr.hash(fmt.Sprintf("%s:%d", node, i))
        hr.ring[hash] = node
    }
    
    // Rebuild sorted keys
    hr.rebuildSortedKeys()
}

func (hr *HashRing) GetNode(key string) string {
    hr.mu.RLock()
    defer hr.mu.RUnlock()
    
    if len(hr.nodes) == 0 {
        return ""
    }
    
    hash := hr.hash(key)
    
    // Find first node with hash >= key hash
    idx := sort.Search(len(hr.nodes), func(i int) bool {
        nodeHash := hr.hash(hr.nodes[i])
        return nodeHash >= hash
    })
    
    // Wrap around if necessary
    if idx == len(hr.nodes) {
        idx = 0
    }
    
    return hr.nodes[idx]
}

// Atomic batch updates for rebalancing
func (hr *HashRing) UpdateNodes(newNodes []string) {
    hr.mu.Lock()
    defer hr.mu.Unlock()
    
    // Clear existing ring
    hr.ring = make(map[uint32]string)
    hr.nodes = nil
    
    // Add all new nodes
    for _, node := range newNodes {
        for i := 0; i < hr.replicas; i++ {
            hash := hr.hash(fmt.Sprintf("%s:%d", node, i))
            hr.ring[hash] = node
        }
    }
    
    // Rebuild sorted keys
    hr.rebuildSortedKeys()
}
```

## ⚡ Performance Optimizations

### Lock-Free Fast Paths

```go
// Optimized search with minimal locking
func (idx *HNSWIndex) SearchOptimized(query []float32, k int) ([]*SearchResult, error) {
    // Fast path: check if thread safety is disabled
    if !idx.config.ThreadSafe {
        return idx.graph.Search(query, k, nil)
    }
    
    // Try read lock with timeout to avoid blocking
    if idx.mu.TryRLock() {
        defer idx.mu.RUnlock()
        return idx.graph.Search(query, k, nil)
    }
    
    // Fallback to blocking read lock
    idx.mu.RLock()
    defer idx.mu.RUnlock()
    return idx.graph.Search(query, k, nil)
}

// Lock-free statistics access
func (idx *HNSWIndex) GetStatsLockFree() *IndexStats {
    return &IndexStats{
        NodeCount:   atomic.LoadInt64(&idx.nodeCount),
        SearchCount: atomic.LoadInt64(&idx.searchCount),
        InsertCount: atomic.LoadInt64(&idx.insertCount),
    }
}
```

### Memory Pool Synchronization

```go
type MemoryPool struct {
    vectors sync.Pool
    results sync.Pool
    buffers sync.Pool
}

func NewMemoryPool() *MemoryPool {
    return &MemoryPool{
        vectors: sync.Pool{
            New: func() interface{} {
                return make([]*Vector, 0, 100)
            },
        },
        results: sync.Pool{
            New: func() interface{} {
                return make([]*SearchResult, 0, 100)
            },
        },
        buffers: sync.Pool{
            New: func() interface{} {
                return make([]byte, 0, 4096)
            },
        },
    }
}

// Lock-free pool operations
func (mp *MemoryPool) GetVectorSlice() []*Vector {
    return mp.vectors.Get().([]*Vector)[:0]  // Reset length, keep capacity
}

func (mp *MemoryPool) PutVectorSlice(slice []*Vector) {
    mp.vectors.Put(slice)
}
```

## 🔄 Consistency Guarantees

### Read Consistency Levels

```go
type ConsistencyLevel int

const (
    EventualConsistency ConsistencyLevel = iota  // No synchronization required
    ReadYourWrites                              // Client sees own writes immediately
    MonotonicRead                               // Never read older versions
    StrongConsistency                           // Linearizable reads
)

type ConsistentReader struct {
    cluster   *ClusterManager
    level     ConsistencyLevel
    clientID  string
    lastRead  map[string]int64  // Collection -> LSN mapping
}

func (cr *ConsistentReader) Read(collectionID, vectorID string) (*Vector, error) {
    switch cr.level {
    case EventualConsistency:
        return cr.readFromNearestNode(collectionID, vectorID)
        
    case ReadYourWrites:
        return cr.readWithClientConsistency(collectionID, vectorID)
        
    case MonotonicRead:
        return cr.readMonotonic(collectionID, vectorID)
        
    case StrongConsistency:
        return cr.readFromLeader(collectionID, vectorID)
        
    default:
        return nil, ErrInvalidConsistencyLevel
    }
}
```

### Write Consistency

```go
type WriteConsistency struct {
    RequiredReplicas int  // Minimum replicas that must acknowledge
    Timeout         time.Duration
}

func (wc *WriteConsistency) Write(vector *Vector) error {
    ctx, cancel := context.WithTimeout(context.Background(), wc.Timeout)
    defer cancel()
    
    // Get target nodes from hash ring
    nodes := wc.cluster.GetNodesForKey(vector.ID)
    
    // Send write to all replicas
    responses := make(chan error, len(nodes))
    
    for _, node := range nodes {
        go func(n *ClusterNode) {
            err := n.Write(ctx, vector)
            responses <- err
        }(node)
    }
    
    // Wait for required number of acknowledgments
    acks := 0
    for i := 0; i < len(nodes); i++ {
        select {
        case err := <-responses:
            if err == nil {
                acks++
                if acks >= wc.RequiredReplicas {
                    return nil  // Success - enough replicas acknowledged
                }
            }
        case <-ctx.Done():
            return ErrWriteTimeout
        }
    }
    
    return ErrInsufficientReplicas
}
```

## 🛡️ Deadlock Prevention

### Lock Ordering

```go
// Consistent lock ordering to prevent deadlocks
type LockManager struct {
    collectionLocks map[string]*sync.RWMutex
    indexLocks     map[string]*sync.RWMutex
    nodeLocks      map[string]*sync.RWMutex
    
    // Global ordering based on ID comparison
    mu sync.Mutex  // Protects lock map access
}

func (lm *LockManager) AcquireMultipleLocks(ids []string, exclusive bool) func() {
    // Sort IDs to ensure consistent ordering
    sortedIDs := make([]string, len(ids))
    copy(sortedIDs, ids)
    sort.Strings(sortedIDs)
    
    var unlockFuncs []func()
    
    // Acquire locks in sorted order
    for _, id := range sortedIDs {
        unlock := lm.acquireSingleLock(id, exclusive)
        unlockFuncs = append(unlockFuncs, unlock)
    }
    
    // Return function to release all locks in reverse order
    return func() {
        for i := len(unlockFuncs) - 1; i >= 0; i-- {
            unlockFuncs[i]()
        }
    }
}
```

### Timeout-based Deadlock Detection

```go
type DeadlockDetector struct {
    timeout      time.Duration
    waitGraph    map[string][]string  // goroutine -> waiting for goroutines
    mu          sync.Mutex
}

func (dd *DeadlockDetector) AcquireWithTimeout(lockID string, timeout time.Duration) (bool, error) {
    done := make(chan bool, 1)
    
    go func() {
        // Attempt to acquire lock
        dd.acquireLock(lockID)
        done <- true
    }()
    
    select {
    case <-done:
        return true, nil
    case <-time.After(timeout):
        // Potential deadlock detected
        dd.reportPotentialDeadlock(lockID)
        return false, ErrPotentialDeadlock
    }
}
```

## 📊 Synchronization Monitoring

### Lock Contention Metrics

```go
type SynchronizationMetrics struct {
    LockAcquisitionTime   map[string]time.Duration
    LockContentionCount   map[string]int64
    DeadlockCount         int64
    TimeoutCount          int64
    
    mu sync.RWMutex
}

func (sm *SynchronizationMetrics) RecordLockAcquisition(lockID string, duration time.Duration) {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    
    sm.LockAcquisitionTime[lockID] = duration
    atomic.AddInt64(&sm.LockContentionCount[lockID], 1)
}

func (sm *SynchronizationMetrics) GetContentionReport() *ContentionReport {
    sm.mu.RLock()
    defer sm.mu.RUnlock()
    
    report := &ContentionReport{
        HighContentionLocks: make([]string, 0),
        AverageWaitTime:    make(map[string]time.Duration),
        TotalDeadlocks:     atomic.LoadInt64(&sm.DeadlockCount),
    }
    
    for lockID, count := range sm.LockContentionCount {
        if count > 100 {  // High contention threshold
            report.HighContentionLocks = append(report.HighContentionLocks, lockID)
        }
        
        if avgTime, exists := sm.LockAcquisitionTime[lockID]; exists {
            report.AverageWaitTime[lockID] = avgTime
        }
    }
    
    return report
}
```

## 🎯 Best Practices

### 1. Lock Granularity

- **Collection-level locks**: For schema changes and collection operations
- **Index-level locks**: For index modifications and searches  
- **Node-level locks**: For individual node updates
- **Operation-level locks**: For specific data operations

### 2. Lock-Free Programming

```go
// Prefer atomic operations over locks where possible
type AtomicCounter struct {
    value int64
}

func (ac *AtomicCounter) Increment() int64 {
    return atomic.AddInt64(&ac.value, 1)
}

func (ac *AtomicCounter) Get() int64 {
    return atomic.LoadInt64(&ac.value)
}

// Use lock-free data structures
type LockFreeQueue struct {
    head unsafe.Pointer
    tail unsafe.Pointer
}
```

### 3. Timeout Strategies

```go
// Always use context with timeout for distributed operations
func DistributedOperation(ctx context.Context) error {
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()
    
    // Perform operation with deadline
    return performWithDeadline(ctx)
}
```

### 4. Graceful Degradation

```go
// Implement fallbacks for lock contention
func (idx *HNSWIndex) SearchWithFallback(query []float32, k int) ([]*SearchResult, error) {
    // Try fast path first
    if idx.tryAcquireReadLock(100 * time.Millisecond) {
        defer idx.mu.RUnlock()
        return idx.preciseSearch(query, k)
    }
    
    // Fallback to approximate search without locking
    return idx.approximateSearch(query, k)
}
```

### 5. Monitor and Alert

```go
// Set up monitoring for synchronization issues
type SyncMonitor struct {
    metrics *SynchronizationMetrics
    alerts  chan Alert
}

func (sm *SyncMonitor) CheckForIssues() {
    report := sm.metrics.GetContentionReport()
    
    if report.TotalDeadlocks > 0 {
        sm.alerts <- Alert{
            Type:     "DEADLOCK_DETECTED",
            Severity: "CRITICAL",
            Count:    report.TotalDeadlocks,
        }
    }
    
    for _, lockID := range report.HighContentionLocks {
        if avgTime := report.AverageWaitTime[lockID]; avgTime > 100*time.Millisecond {
            sm.alerts <- Alert{
                Type:     "HIGH_LOCK_CONTENTION",
                Severity: "WARNING",
                LockID:   lockID,
                AvgWait:  avgTime,
            }
        }
    }
}
```

## 📚 Additional Resources

- [Go Memory Model](https://golang.org/ref/mem)
- [Raft Consensus Algorithm](https://raft.github.io/)
- [Lock-Free Programming Guide](./LOCK_FREE_PROGRAMMING.md)
- [Performance Tuning](./PERFORMANCE.md)
- [Troubleshooting Synchronization Issues](./TROUBLESHOOTING.md)

---

*This guide provides comprehensive coverage of synchronization mechanisms in GoVecDB. For specific performance optimizations or advanced configurations, consult the detailed API documentation.*