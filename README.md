# GoVecDB

A high-performance, distributed vector database written in pure Go for production workloads requiring similarity search and semantic search.

[![Go Report Card](https://goreportcard.com/badge/github.com/khambampati-subhash/govecdb)](https://goreportcard.com/report/github.com/khambampati-subhash/govecdb)
[![Go Version](https://img.shields.io/badge/go-1.23+-blue.svg)](https://golang.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **Pure Go** - Zero dependencies, no CGO, embeddable
- **High Performance** - Sub-millisecond search with HNSW algorithm
- **Production Ready** - WAL persistence, crash recovery, comprehensive testing
- **Thread Safe** - Built with Go's concurrency primitives
- **Distributed** - Clustering with consistent hashing and Raft consensus
- **Smart Filtering** - Complex metadata queries with vector search

## Quick Start

### Installation

```bash
go get github.com/khambampati-subhash/govecdb
```

### Basic Usage

```go
package main

import (
    "log"
    "github.com/khambampati-subhash/govecdb/api"
    "github.com/khambampati-subhash/govecdb/collection"
)

func main() {
    // Create collection
    config := &api.CollectionConfig{
        Name:      "documents",
        Dimension: 384,
        Metric:    api.Cosine,
    }
    
    coll, err := collection.NewPersistentCollection(config, "./data")
    if err != nil {
        log.Fatal(err)
    }
    defer coll.Close()
    
    // Add vectors
    vectors := []*api.Vector{
        {
            ID:   "doc1",
            Data: make([]float32, 384), // Your embeddings here
            Metadata: map[string]interface{}{
                "title": "Introduction to AI",
                "tags":  []string{"ai", "machine-learning"},
            },
        },
    }
    
    if err := coll.AddBatch(vectors); err != nil {
        log.Fatal(err)
    }
    
    // Search
    query := make([]float32, 384) // Your query embedding
    results, err := coll.Search(&api.SearchRequest{
        Vector: query,
        K:      10,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    for _, result := range results {
        log.Printf("ID: %s, Score: %.4f\n", result.ID, result.Score)
    }
}
```

### Advanced Filtering

```go
// Search with metadata filters
filter := &api.LogicalFilter{
    Operator: api.And,
    Filters: []api.Filter{
        &api.FieldFilter{
            Field:    "category",
            Operator: api.Equals,
            Value:    "technology",
        },
        &api.FieldFilter{
            Field:    "tags",
            Operator: api.In,
            Value:    []string{"ai", "machine-learning"},
        },
    },
}

results, err := coll.Search(&api.SearchRequest{
    Vector: query,
    K:      10,
    Filter: filter,
})
```

### Distributed Setup

```go
import "github.com/khambampati-subhash/govecdb/cluster"

// Create cluster
config := &cluster.Config{
    NodeID:            "node1",
    ReplicationFactor: 3,
    ShardCount:        16,
}

manager := cluster.NewClusterManager(config)
coordinator := cluster.NewQueryCoordinator(manager)

// Distributed search
results, err := coordinator.Search(request)
```

## Performance

**Key Metrics** (tested on Apple M1 Pro / Intel i7):

## Comprehensive Benchmarks (N=1000)

| Dimension | Insertion Rate (ops/s) | Recall@10 | Search QPS | Avg Latency |
|-----------|------------------------|-----------|------------|-------------|
| 128       | ~2008                  | 0.66      | ~2813      | ~0.36ms     |
| 256       | ~1646                  | 0.66      | ~2282      | ~0.44ms     |
| 512       | ~1186                  | 0.63      | ~1763      | ~0.57ms     |
| 1024      | ~785                   | 0.66      | ~1199      | ~0.83ms     |
| 2048      | ~446                   | 0.65      | ~757       | ~1.32ms     |
| 4096      | ~243                   | 0.65      | ~442       | ~2.26ms     |
| 6000      | ~133                   | 0.66      | ~206       | ~4.85ms     |
| 8192      | ~123                   | 0.55      | ~241       | ~4.15ms     |
| 16384     | ~63                    | 0.35      | ~127       | ~7.84ms     |

*Note: Search QPS measured with concurrency=1.*

**Highlights**:
- **SIMD Acceleration**: Hand-written Assembly (AVX2 for AMD64, NEON for ARM64) for DotProduct, Euclidean, and Cosine distance.
- **Zero-Allocation Search**: Optimized hot paths to minimize GC pressure.
- **High Throughput**: Up to **60,000 QPS** on a single node for low-dimensional vectors.
- **Data Integrity**: Verified 100% data integrity and recall for exact matches even at 4096 dimensions.

For detailed benchmarks, see [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Architecture

```
┌─────────────────────┐
│     API Layer      │  Type-safe interfaces
├─────────────────────┤
│   Collection Mgmt  │  High-level abstractions
├─────────────────────┤
│   Index Engine     │  HNSW algorithm
├─────────────────────┤
│  Storage Layer     │  Memory management
├─────────────────────┤
│ Persistence Layer  │  WAL, snapshots
├─────────────────────┤
│  Cluster Layer     │  Distribution
└─────────────────────┘
```

**Key Components**:
- **HNSW Index**: Fast approximate nearest neighbor search
- **WAL Persistence**: Durability and crash recovery
- **Consistent Hashing**: Automatic data distribution
- **Raft Consensus**: Distributed coordination

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Configuration

### Collection Configuration

```go
config := &api.CollectionConfig{
    Name:      "my-collection",
    Dimension: 384,
    Metric:    api.Cosine,
    
    // Index tuning
    IndexConfig: &index.Config{
        M:              16,   // Connections per node
        EfConstruction: 200,  // Construction search depth
        MaxLayer:       16,   // Maximum layers
    },
    
    // Persistence
    PersistenceConfig: &collection.PersistenceConfig{
        EnableWAL:        true,
        SyncInterval:     5 * time.Second,
        SnapshotInterval: time.Hour,
    },
}
```

### Cluster Configuration

```go
clusterConfig := &cluster.Config{
    NodeID:            "node-1",
    ReplicationFactor: 3,
    ShardCount:        32,
    
    ConsensusConfig: &cluster.RaftConfig{
        HeartbeatTimeout:  100 * time.Millisecond,
        ElectionTimeout:   500 * time.Millisecond,
    },
}
```

## Use Cases

- **Semantic Search**: Build search engines with natural language understanding
- **RAG Systems**: Retrieval-augmented generation for LLMs
- **Recommendations**: Content-based recommendation engines
- **Anomaly Detection**: High-dimensional data analysis

## Testing

```bash
# Run all tests
go test ./...

# Run benchmarks
go test ./... -bench=. -benchmem

# Run with race detection
go test ./... -race
```

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - Technical deep-dive
- [Performance Guide](docs/PERFORMANCE.md) - Benchmarks and optimization
- [Distributed Systems](docs/DISTRIBUTED_SYSTEMS.md) - Cluster deployment
- [API Reference](https://pkg.go.dev/github.com/khambampati-subhash/govecdb)

## Development

### Prerequisites
- Go 1.23+
- Git

### Setup
```bash
git clone https://github.com/khambampati-subhash/govecdb.git
cd govecdb
go mod download
```

### Code Quality
```bash
go fmt ./...
go vet ./...
golangci-lint run
```

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to Contribute**:
- Bug reports and feature requests
- Code contributions with tests
- Documentation improvements
- Performance benchmarks

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- HNSW algorithm by Yu. A. Malkov and D. A. Yashunin
- Inspired by Chroma, Weaviate, and Qdrant
- Go community for excellent tooling

## Support

- **Issues**: [GitHub Issues](https://github.com/khambampati-subhash/govecdb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/khambampati-subhash/govecdb/discussions)

---

**Built with ❤️ for the Go community**
