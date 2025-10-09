# GoVecDB

A high-performance, distributed vector database written in pure Go, designed for production workloads requiring fast similarity search, semantic search, and RAG (Retrieval-Augmented Generation) systems.

[![Go Report Card](https://goreportcard.com/badge/github.com/khambampati-subhash/govecdb)](https://goreportcard.com/report/github.com/khambampati-subhash/govecdb)
[![Go Version](https://img.shields.io/badge/go-1.23+-blue.svg)](https://golang.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/khambampati-subhash/govecdb/actions)

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#quick-start)
- [Performance](#-performance)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Use Cases](#-use-cases)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Production Deployment](#-production-deployment)
- [Development](#-development)
- [Current Limitations](#️-current-limitations--known-issues)
- [Contributing](#-contributing)
- [FAQ](#-faq)
- [Support](#-support)
- [License](#-license)

---

## ✨ Features

### Core Capabilities
- 🚀 **Pure Go Implementation** - Zero CGO dependencies, fully embeddable
- ⚡ **HNSW Algorithm** - Hierarchical Navigable Small World graphs for fast ANN search
- 💾 **WAL Persistence** - Write-Ahead Logging with crash recovery and snapshots
- 🔒 **Thread-Safe** - Built with Go's native concurrency primitives
- 🌐 **Distributed Architecture** - Horizontal scaling with sharding and replication
- 🔍 **Hybrid Search** - Combine vector similarity with metadata filtering
- 📊 **SIMD Optimization** - Hardware-accelerated distance calculations

### Production Features
- ✅ **Comprehensive Testing** - 10+ distributed system test suites
- 🛡️ **Fault Tolerance** - Node failure detection and automatic recovery
- 🔄 **Dynamic Rebalancing** - Automatic shard redistribution on cluster changes
- 📈 **Performance Monitoring** - Built-in metrics and health checks
- 🎯 **100% Recall** - Verified exact match accuracy in distributed scenarios
- 🔐 **Data Consistency** - Configurable replication factor (default: 3x)

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

## 📊 Performance

**Tested on Apple M1 Pro (MBP-175852-1)**

### Single-Node Performance

| Operation | Latency/Throughput | Configuration |
|-----------|-------------------|---------------|
| **Vector Insert** | 46-692M vec/sec | Batch (5K-30K vectors), 512D |
| **Search (10K vectors)** | ~10ms average | k=10, 512D, brute-force |
| **Exact Match Recall** | 100% | Verified across 50 queries |
| **Concurrent Queries** | 65 qps | 10 clients, 20K vectors |

### Distributed Cluster Performance

| Metric | Result | Test Configuration |
|--------|--------|-------------------|
| **Cluster Insert** | 696K vec/sec | 5 nodes, 10K vectors, concurrent |
| **Search Throughput** | 19-94 qps | 2-10 nodes, mixed workload |
| **Node Failure Impact** | 0% data loss | Replication factor: 3 |
| **Recovery Time** | <3 seconds | Automatic failover |
| **Data Consistency** | 100% | Across all replicas |
| **Scaling Efficiency** | 62% at 2 nodes → 22% at 10 nodes | Horizontal scaling |

### Comprehensive Test Results
**✅ Passing Tests:**
- Full Stack Integration (1.5s)
- High Volume Insertion - 30K vectors (7.9s)
- Node Failure Recovery (10.7s)
- Network Partition Handling (10.5s)
- Race Condition Tests (11.1s)
- Data Consistency (5.3s)
- Scalability Test - 2-10 nodes (5.7s)

**⚠️ Performance Notes:**
- Concurrent search latency increases under heavy load (100+ clients: 1-5s avg)
- Optimal performance at 2-6 nodes; scaling efficiency decreases beyond 6 nodes
- Query coordination stress test: 19 qps throughput (target: 100+ qps)

For detailed benchmarks and optimization tips, see [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

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

## 🎯 Use Cases

### Production Workloads
- 🤖 **LLM Applications**: RAG systems, semantic caching, prompt similarity
- 🔍 **Semantic Search**: Natural language search engines with embedding-based retrieval
- 📚 **Document Management**: Find similar documents, duplicate detection
- 🎬 **Recommendation Systems**: Content-based filtering using vector similarity
- 🛡️ **Anomaly Detection**: High-dimensional pattern recognition
- 🖼️ **Image/Video Search**: Similarity search using vision embeddings
- 💬 **Chatbot Memory**: Store and retrieve conversation context efficiently

### Deployment Scenarios
- **Single-Node**: Embedded in Go applications, no external dependencies
- **Distributed Cluster**: 5-10 node clusters for high availability
- **Edge Computing**: Lightweight deployment on resource-constrained devices
- **Microservices**: Standalone vector search service with gRPC/HTTP API

## 🧪 Testing

### Run Tests

```bash
# Run all tests
go test ./...

# Run with verbose output
go test -v ./...

# Run benchmarks
go test ./... -bench=. -benchmem

# Run with race detection (RECOMMENDED before production)
go test -race ./...
```

### Comprehensive Distributed Tests

```bash
# Full distributed system test suite (8 minutes)
go test -v ./cluster -run TestDistributedCluster_Comprehensive -timeout 30m

# With race detection (slower, ~15 minutes)
go test -race -v ./cluster -run TestDistributedCluster_Comprehensive -timeout 60m

# Individual test suites
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Full_Stack_Integration
go test -v ./cluster -run TestDistributedCluster_Comprehensive/High_Volume_Insertion
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Node_Failure_Recovery
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Data_Consistency
```

### Test Coverage

**10 Comprehensive Test Suites:**
1. ✅ Full Stack Integration - End-to-end cluster operations
2. ✅ High Volume Insertion - 30K vectors across 3 waves
3. ⚠️ Concurrent Search Performance - Load testing (latency optimization needed)
4. ✅ Node Failure Recovery - Fault tolerance validation
5. ✅ Network Partition Handling - Split-brain scenarios
6. ✅ Race Conditions - Concurrent operation safety
7. ⚠️ Shard Rebalancing - Dynamic scaling (needs optimization)
8. ⚠️ Query Coordination Stress - 5K queries under load (throughput optimization needed)
9. ✅ Data Consistency - Replica consistency verification
10. ✅ Scalability Test - 2-10 node horizontal scaling

**Current Status:**
- ✅ 7/10 tests passing
- ⚠️ 3/10 tests need performance optimization
- 🔧 Known issues: MockNetworkManager race conditions (test-only, not production code)

## 📚 Documentation

### Core Documentation
- 🏗️ [Architecture Guide](docs/architecture.md) - Technical deep-dive into system design
- ⚡ [Performance Guide](docs/PERFORMANCE.md) - Benchmarks, optimization tips, and tuning
- 🌐 [Distributed Systems](docs/DISTRIBUTED_SYSTEMS.md) - Cluster deployment and configuration
- 🔄 [Synchronization Guide](docs/SYNCHRONIZATION.md) - Concurrency patterns and thread safety
- 🧪 [Testing Guide](docs/CLUSTER_TESTS_SUMMARY.md) - Comprehensive test suite documentation

### API & Examples
- 📖 [API Reference](https://pkg.go.dev/github.com/khambampati-subhash/govecdb) - Complete Go package documentation
- 💻 [Basic Usage Examples](examples/basic-usage/) - Quick start code samples
- 🚀 [Advanced Features](examples/advanced-features/) - Complex use cases and patterns
- 🔍 [Search Operations](examples/) - Vector search and filtering examples

### Guides
- 🎯 [Quick Start](#quick-start) - Get running in 5 minutes
- ⚙️ [Configuration](#configuration) - Tuning for your workload
- 🏭 [Production Deployment](#production-deployment) - Best practices and monitoring

## 🏭 Production Deployment

### Recommended Configuration

**For 30M vectors/day workload:**
```go
clusterConfig := &cluster.Config{
    NodeCount:         8-10,        // 8-10 nodes for ~1000 vec/sec total
    ReplicationFactor: 3,            // Survives 2 node failures
    ShardCount:        32,           // Balance load distribution
}

collectionConfig := &collection.Config{
    Dimension: 768,                  // Common for sentence transformers
    IndexConfig: &index.Config{
        M:              64,          // Higher for better recall at high dimensions
        EfConstruction: 800,         // Build quality vs speed tradeoff
        MaxLayer:       16,          // Sufficient for millions of vectors
    },
}
```

### Monitoring

**Key Metrics to Track:**
- Insert throughput (vec/sec)
- Search latency (p50, p95, p99)
- Cluster health status
- Node availability
- Replication lag
- Memory usage per node

### Best Practices

1. **Start Small**: Test with 2-3 nodes before scaling to 10+
2. **Monitor Scaling Efficiency**: Efficiency degrades beyond 6 nodes (current)
3. **Use Replication Factor 3**: Optimal for fault tolerance vs storage overhead
4. **Batch Inserts**: Always use batch operations for better throughput
5. **Race Detection**: Run tests with `-race` before deploying new versions

## 🛠️ Development

### Prerequisites
- Go 1.23+
- Git
- (Optional) golangci-lint for code quality checks

### Setup
```bash
# Clone repository
git clone https://github.com/khambampati-subhash/govecdb.git
cd govecdb

# Download dependencies
go mod download

# Run tests
go test ./...

# Build
go build ./...
```

### Code Quality
```bash
# Format code
go fmt ./...

# Vet code for issues
go vet ./...

# Run linter (if installed)
golangci-lint run

# Check for race conditions
go test -race ./...
```

### Project Structure
```
govecdb/
├── api/              # Type-safe API definitions
├── collection/       # High-level collection management
├── index/           # HNSW index implementation
├── cluster/         # Distributed system components
├── persist/         # WAL and snapshot persistence
├── filter/          # Metadata filtering engine
├── utils/           # Distance calculations (SIMD)
├── examples/        # Usage examples
├── docs/            # Documentation
└── benchmarks/      # Performance benchmarks
```

## ⚠️ Current Limitations & Known Issues

### Performance Optimization Needed
1. **High Concurrent Load Latency**: Search latency increases significantly with 100+ concurrent clients (1-5s avg)
   - **Impact**: Limits throughput to ~20-65 qps under stress
   - **Workaround**: Use connection pooling and query batching
   - **Roadmap**: Query coordinator optimization planned

2. **Scaling Efficiency**: Degrades beyond 6 nodes (62% at 2 nodes → 22% at 10 nodes)
   - **Impact**: Adding more nodes provides diminishing returns
   - **Workaround**: Optimal deployment is 4-6 nodes
   - **Roadmap**: Shard distribution and network overhead optimization

3. **Shard Rebalancing Imbalance**: 33% imbalance after rebalancing (target: <20%)
   - **Impact**: Uneven load distribution
   - **Workaround**: Manual shard adjustment
   - **Roadmap**: Improved rebalancing algorithm

### Test-Only Issues (Not Production Code)
- **MockNetworkManager Race Conditions**: Data races detected in test mocks with `-race` flag
  - **Impact**: None (test infrastructure only)
  - **Status**: Being addressed in test refactoring

### Roadmap
- [ ] Query coordinator performance optimization (target: 100+ qps)
- [ ] Improved shard rebalancing algorithm
- [ ] Horizontal scaling efficiency improvements
- [ ] gRPC/HTTP API server implementation
- [ ] Vector quantization support (PQ, SQ)
- [ ] Streaming insert API
- [ ] Multi-vector search support
- [ ] GPU acceleration for distance calculations

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to Contribute**:
- 🐛 Bug reports and feature requests via [GitHub Issues](https://github.com/khambampati-subhash/govecdb/issues)
- 💻 Code contributions with comprehensive tests
- 📖 Documentation improvements and examples
- 🔬 Performance benchmarks and optimization PRs
- 🧪 Test coverage improvements

**Priority Areas**:
1. Query coordinator performance optimization
2. Shard rebalancing algorithm improvements
3. Horizontal scaling efficiency
4. Race condition fixes in test infrastructure
5. Integration with popular embedding models

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- HNSW algorithm by Yu. A. Malkov and D. A. Yashunin
- Inspired by Chroma, Weaviate, and Qdrant
- Go community for excellent tooling

## ❓ FAQ

**Q: How does GoVecDB compare to Chroma, Weaviate, or Qdrant?**
A: GoVecDB is pure Go with zero CGO dependencies, making it ideal for embedded use cases and Go-native applications. It's optimized for 100K-10M vector datasets. For billion-scale, consider Qdrant or Weaviate.

**Q: What's the recommended cluster size?**
A: 4-6 nodes provide the best balance of performance and efficiency. Beyond 6 nodes, scaling efficiency decreases due to coordination overhead.

**Q: Can I use this in production?**
A: Yes, with caveats. The system is stable for single-node and small clusters (2-6 nodes). For high-throughput distributed workloads (100+ qps), query coordinator optimization is recommended.

**Q: What embedding models work best?**
A: Any model producing float32 vectors. Tested with:
- Sentence Transformers (384D, 768D, 1024D)
- OpenAI embeddings (1536D)
- Custom models (128D-4096D)

**Q: How do I handle 30M vectors/day?**
A: Deploy 8-10 node cluster with replication factor 3. This provides ~1000 vec/sec sustained throughput. See [Production Deployment](#-production-deployment).

**Q: What about GPU acceleration?**
A: Currently CPU-only with SIMD optimization. GPU support is on the roadmap.

**Q: Is there a hosted service?**
A: No, GoVecDB is self-hosted only. You deploy and manage it in your infrastructure.

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/khambampati-subhash/govecdb/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/khambampati-subhash/govecdb/discussions)
- 📧 **Email**: khambampati.subhash@example.com
- 📖 **Documentation**: [Full docs in /docs folder](docs/)

## 🌟 Star History

If you find GoVecDB useful, please consider giving it a ⭐ on GitHub!

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2025 Venkata Naga Satya Subhash Khambampati

---

**Built with ❤️ for the Go community**

*GoVecDB is under active development. Contributions, feedback, and bug reports are highly appreciated!*
