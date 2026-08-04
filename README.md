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

## What Is GoVecDB & Why It Exists

AI models turn text, images, and audio into **embeddings** — lists of numbers where
*things that mean similar things sit close together* in number-space. That reframes
"find me content similar to this" into a geometry problem: **find the stored vectors
nearest to my query vector** (nearest-neighbor search).

The naive approach — compare the query against *every* stored vector — is `O(N)` and
collapses at millions of vectors. **GoVecDB's purpose is to make that search fast,
durable, and scalable**, in pure Go with no CGO so it stays embeddable:

> Store millions of embeddings and answer *"what's most similar to this?"* in
> sub-millisecond time, survive crashes, and scale across machines.

This powers **semantic search**, **RAG** for LLMs, **recommendations**, and
**anomaly detection**.

## How It Works

### The core trick: HNSW (skip brute force)

Instead of scanning every vector, GoVecDB builds a **hierarchical graph** (HNSW —
Hierarchical Navigable Small World). Think of finding a house in a country: you don't
knock on every door — you take highways to the right region, then regional roads, then
local streets. HNSW searches the same way, turning `O(N)` into roughly `O(log N)`.

```mermaid
flowchart TB
    Q([Query vector]) --> TOP

    subgraph TOP [Layer 2 · few nodes · long jumps]
        direction LR
        n1(( )) --- n2(( )) --- n3(( ))
    end
    subgraph MID [Layer 1 · more nodes · medium hops]
        direction LR
        m1(( )) --- m2(( )) --- m3(( )) --- m4(( ))
    end
    subgraph BOT [Layer 0 · every vector · short hops]
        direction LR
        b1(( )) --- b2(( )) --- b3(( )) --- b4(( )) --- b5(( ))
    end

    TOP -->|zoom into region| MID
    MID -->|refine| BOT
    BOT --> R([Nearest K results])
```

A search **enters at the sparse top layer, takes big jumps toward the right region,
then drops layer by layer taking smaller steps** until it lands among the true nearest
neighbors — visiting only a tiny fraction of all vectors. Two knobs trade speed for
accuracy: `M` (connections per node) and `EfConstruction`/`ef` (how wide the search
explores).

### The layers that make it production-grade

Each layer of the system solves one part of the problem:

| Layer | Package | What it achieves |
|-------|---------|------------------|
| **Contract** | `api` | Interfaces & types (`Vector`, `SearchRequest`, `VectorIndex`, `VectorStore`) — everything below is swappable behind them |
| **Orchestration** | `collection` | `VectorCollection` ties the index, storage, and filtering together with thread-safe lifecycle management |
| **Speed** | `index` | HNSW graph — the approximate nearest-neighbor engine |
| **Data** | `store` | Holds the actual vectors + metadata in memory |
| **Durability** | `persist` | Write-Ahead Log (WAL) + snapshots so a crash doesn't lose data |
| **Precision** | `filter` | Metadata queries combined with vector search (*similar* **AND** `category = tech`) |
| **Scale** | `cluster` | Spreads data across nodes via consistent hashing + Raft consensus |

**Distance metrics** (`Cosine`, `Euclidean`, `DotProduct`, `Manhattan`) define what
"near" means and are selected per collection. A typical write flow is
`Add → store the vector → insert into the HNSW graph → append to the WAL`; a read is
`Search → HNSW descent → optional metadata filter → top-K results`.

## Quick Start

### Installation

```bash
go get github.com/khambampati-subhash/govecdb
```

### Basic Usage

```go
package main

import (
    "context"
    "log"

    "github.com/khambampati-subhash/govecdb/api"
    "github.com/khambampati-subhash/govecdb/collection"
    "github.com/khambampati-subhash/govecdb/store"
)

func main() {
    ctx := context.Background()

    // Create collection
    config := &api.CollectionConfig{
        Name:           "documents",
        Dimension:      384,
        Metric:         api.Cosine,
        M:              16,
        EfConstruction: 200,
        MaxLayer:       16,
        ThreadSafe:     true,
    }

    coll, err := collection.NewVectorCollection(config, store.DefaultStoreConfig(config.Name))
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

    if err := coll.AddBatch(ctx, vectors); err != nil {
        log.Fatal(err)
    }

    // Search
    query := make([]float32, 384) // Your query embedding
    results, err := coll.Search(ctx, &api.SearchRequest{
        Vector: query,
        K:      10,
    })
    if err != nil {
        log.Fatal(err)
    }

    for _, result := range results {
        log.Printf("ID: %s, Score: %.4f\n", result.Vector.ID, result.Score)
    }
}
```

### Advanced Filtering

```go
// Search with metadata filters
filter := &api.LogicalFilter{
    Op: api.FilterAnd,
    Filters: []api.FilterExpr{
        &api.FieldFilter{
            Field: "category",
            Op:    api.FilterEq,
            Value: "technology",
        },
        &api.FieldFilter{
            Field: "tags",
            Op:    api.FilterIn,
            Value: []interface{}{"ai", "machine-learning"},
        },
    },
}

results, err := coll.Search(ctx, &api.SearchRequest{
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
- **Vectorized Distance Kernels**: Pure-Go distance functions (DotProduct, Euclidean, Cosine) with manual loop unrolling to help the Go compiler auto-vectorize the hot paths — no CGO or hand-written assembly.
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

    // HNSW index tuning (flat fields on CollectionConfig)
    M:              16,  // Connections per node
    EfConstruction: 200, // Construction search depth
    MaxLayer:       16,  // Maximum layers
    Seed:           42,  // Deterministic layer assignment
    ThreadSafe:     true,
}

// For durable, crash-safe storage (WAL + snapshots), use the persistent
// collection instead, which wraps the above config with storage paths:
//
//   pcfg := &collection.PersistentCollectionConfig{
//       CollectionConfig: config,
//       DataDir:          "./data",
//       WALDir:           "./data/wal",
//       SnapshotDir:      "./data/snapshots",
//   }
//   coll, err := collection.NewPersistentVectorCollection(pcfg)
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
