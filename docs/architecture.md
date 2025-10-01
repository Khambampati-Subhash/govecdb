# HNSW Algorithm Implementation

This document provides technical details about the HNSW (Hierarchical Navigable Small World) algorithm implementation in GoVecDB.

## Overview

The HNSW algorithm is a graph-based approach for approximate nearest neighbor search that provides excellent performance characteristics:

- **Insert**: O(log n) average case
- **Search**: O(log n) average case  
- **Memory**: O(n) for n vectors

## Key Components

### 1. Multi-layer Graph Structure (`graph.go`)

The HNSW graph consists of multiple layers where:
- Layer 0 (base layer) contains all vectors
- Higher layers contain progressively fewer vectors
- Each vector exists on layer 0 and potentially on higher layers

```go
type HNSWGraph struct {
    config       *Config
    distanceFunc DistanceFunc
    entryPoint   *HNSWNode
    nodes        *SafeMap
    rng          *rand.Rand
    mu           sync.RWMutex
    stats        *GraphStats
}
```

### 2. Node Structure (`node.go`)

Each node represents a vector in the graph with connections at multiple layers:

```go
type HNSWNode struct {
    Vector      *Vector
    Level       int
    connections []map[string]*HNSWNode
    mu          sync.RWMutex
    deleted     bool
}
```

### 3. Distance Metrics (`metrics.go`)

Supports multiple distance metrics for multi-dimensional vectors:

- **Cosine Distance**: `1 - cosine_similarity`
- **Euclidean Distance**: L2 norm
- **Manhattan Distance**: L1 norm  
- **Dot Product Distance**: Negative dot product

### 4. Main Index (`hnsw.go`)

The main interface providing thread-safe operations:

```go
type HNSWIndex struct {
    graph        *HNSWGraph
    config       *Config
    mu           sync.RWMutex
    insertCount  int64
    searchCount  int64
    createdAt    time.Time
    lastUpdateAt time.Time
}
```

## Algorithm Details

### Layer Selection

New vectors are assigned to layers using an exponential distribution:

```go
func (g *HNSWGraph) SelectLayer() int {
    layer := 0
    for g.rng.Float64() < 0.5 && layer < g.config.MaxLayer {
        layer++
    }
    return layer
}
```

### Insertion Process

1. **Layer Selection**: Determine the layer for the new vector
2. **Entry Point Search**: Find closest nodes starting from the top layer
3. **Layer-by-layer Search**: Search and connect at each layer from top to layer 0
4. **Neighbor Selection**: Select M best neighbors at each layer
5. **Bidirectional Connections**: Create connections between nodes
6. **Pruning**: Remove excess connections if nodes exceed maximum degree

### Search Process

1. **Top-down Search**: Start from entry point at highest layer
2. **Greedy Search**: Find closest node at each layer
3. **Base Layer Search**: Perform detailed search at layer 0 with larger ef
4. **Result Filtering**: Apply metadata filters if specified
5. **Result Ranking**: Sort by distance and return top-k results

## Configuration Parameters

### Core Parameters

- **Dimension**: Vector dimensionality
- **M**: Maximum number of bidirectional links for each node (except layer 0)
- **EfConstruction**: Size of dynamic candidate list during construction
- **MaxLayer**: Maximum number of layers in the graph
- **Metric**: Distance metric to use

### Default Configuration

```go
func DefaultConfig(dimension int) *Config {
    return &Config{
        Dimension:      dimension,
        Metric:         Cosine,
        M:              16,
        EfConstruction: 200,
        MaxLayer:       16,
        Seed:           42,
        ThreadSafe:     true,
    }
}
```

## Thread Safety

The implementation provides thread-safe operations through:

- **RWMutex** for index-level operations
- **Node-level mutexes** for individual node modifications
- **SafeMap** for concurrent access to vector storage
- **Atomic operations** for statistics updates

## Performance Characteristics

### Time Complexity

- **Insert**: O(M * log(n) * dimension)
- **Search**: O(ef * log(n) * dimension)
- **Memory**: O(n * M * layers)

### Space Complexity

- **Node Storage**: Each vector stored once
- **Connection Storage**: O(M) connections per node per layer
- **Index Overhead**: Minimal metadata and statistics

## Optimization Features

### Memory Optimization

- **Soft Deletion**: Vectors marked as deleted without immediate cleanup
- **Connection Pruning**: Maintains optimal connectivity
- **Lazy Cleanup**: Periodic removal of deleted connections

### Search Optimization

- **Beam Search**: Efficient exploration of graph structure
- **Early Termination**: Stop search when no improvement possible
- **Dynamic Candidate Lists**: Maintain best candidates during search

### Concurrency Optimization

- **Read-Heavy Workloads**: Multiple concurrent searches supported
- **Lock-Free Reads**: Read operations minimize locking
- **Fine-Grained Locking**: Node-level locks for updates

## Usage Examples

### Basic Usage

```go
config := &index.Config{
    Dimension: 384,
    Metric:    index.Cosine,
    M:         16,
    EfConstruction: 200,
}

idx, err := index.NewHNSWIndex(config)
if err != nil {
    log.Fatal(err)
}
defer idx.Close()

// Insert vector
vector := &index.Vector{
    ID:   "doc1",
    Data: []float32{...}, // 384 dimensions
    Metadata: map[string]interface{}{
        "category": "technology",
    },
}

err = idx.Add(vector)
if err != nil {
    log.Fatal(err)
}

// Search
query := []float32{...} // 384 dimensions
results, err := idx.Search(query, 10)
if err != nil {
    log.Fatal(err)
}
```

### Filtered Search

```go
filter := func(metadata map[string]interface{}) bool {
    category, exists := metadata["category"]
    return exists && category == "technology"
}

results, err := idx.SearchWithFilter(query, 10, filter)
```

## Performance Tuning

### Parameter Guidelines

- **M**: Higher values improve search quality but increase memory usage (recommended: 8-64)
- **EfConstruction**: Higher values improve search quality but slow down construction (recommended: 100-800)
- **MaxLayer**: Should accommodate expected dataset size (recommended: 16)

### Memory vs Quality Trade-offs

- **Lower M**: Faster searches, less memory, potentially lower recall
- **Higher M**: Better recall, more memory, slightly slower searches
- **Higher EfConstruction**: Better construction quality, slower insertions

## Benchmarks

The implementation includes comprehensive benchmarks for:

- Distance function performance across different dimensions
- Index construction performance
- Search performance with various parameters
- Concurrent operation performance
- Memory usage analysis

Run benchmarks with:
```bash
go test ./index -bench=. -benchmem
```

## Future Optimizations

### Planned Improvements

1. **SIMD Optimizations**: Vector operations using CPU-specific instructions
2. **Batch Operations**: Optimized batch insertion algorithms
3. **Disk Persistence**: Efficient serialization and loading
4. **Compression**: Vector compression for memory savings
5. **GPU Acceleration**: CUDA/OpenCL support for distance calculations

### Advanced Features

1. **Dynamic Ef**: Adaptive ef parameter based on query characteristics
2. **Graph Rebalancing**: Periodic optimization of graph structure
3. **Incremental Updates**: Efficient updates to existing vectors
4. **Distributed Support**: Multi-node HNSW implementation