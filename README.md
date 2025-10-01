# govecdb

A high-performance, embeddable vector database written in pure Go.
GoVecDB is designed to be the SQLite of vector databases - lightweight, embeddable, and requiring zero external dependencies. Perfect for building semantic search, RAG applications, and AI-powered features directly into your Go applications.
✨ Features

🔥 Pure Go - No CGO, no Python, no external dependencies
⚡ Fast - HNSW algorithm implementation optimized for Go
💾 Embeddable - Import as a library, runs in-process
🔒 Thread-safe - Built with Go's concurrency primitives
📦 Persistent - Optional disk-based storage
🎯 Multiple distance metrics - Cosine, Euclidean, Dot Product
🔍 Filtering - Combine vector search with metadata filtering
📊 Production-ready - Comprehensive testing and benchmarks

🚀 Quick Start
Installation
bashgo get github.com/khambampati-subhash/govecdb
Basic Usage
gopackage main

import (
    "fmt"
    "github.com/khambampati-subhash/govecdb"
)

func main() {
    // Create a new in-memory vector database
    db, err := govecdb.New(&govecdb.Config{
        Dimension: 384,  // Embedding dimension
        Metric:    govecdb.Cosine,
    })
    if err != nil {
        panic(err)
    }
    defer db.Close()

    // Add vectors
    vectors := [][]float32{
        {0.1, 0.2, 0.3, ...},  // 384 dimensions
        {0.4, 0.5, 0.6, ...},
    }
    
    metadata := []map[string]interface{}{
        {"text": "Hello world", "category": "greeting"},
        {"text": "Goodbye", "category": "farewell"},
    }

    ids, err := db.Add(vectors, metadata)
    if err != nil {
        panic(err)
    }

    // Search for similar vectors
    queryVector := []float32{0.15, 0.25, 0.35, ...}
    
    results, err := db.Search(queryVector, 5, nil)
    if err != nil {
        panic(err)
    }

    for _, result := range results {
        fmt.Printf("ID: %s, Score: %.4f, Metadata: %v\n", 
            result.ID, result.Score, result.Metadata)
    }
}
With Filtering
go// Search with metadata filters
filter := &govecdb.Filter{
    Field:    "category",
    Operator: govecdb.Equals,
    Value:    "greeting",
}

results, err := db.Search(queryVector, 5, filter)
Persistent Storage
go// Create a persistent database
db, err := govecdb.New(&govecdb.Config{
    Dimension:  384,
    Metric:     govecdb.Cosine,
    Persistent: true,
    Path:       "./data/vectors.db",
})
📚 Use Cases

Semantic Search - Find similar documents, code, or content
RAG Applications - Build retrieval-augmented generation systems
Recommendation Systems - Find similar items based on embeddings
Duplicate Detection - Identify similar or duplicate content
Anomaly Detection - Find outliers in high-dimensional data

🏗️ Architecture
GoVecDB uses the HNSW (Hierarchical Navigable Small World) algorithm for approximate nearest neighbor search, providing excellent performance characteristics:

Insert: O(log n) average case
Search: O(log n) average case
Memory: O(n) for n vectors

See docs/architecture.md for detailed design documentation.
📊 Benchmarks
Coming soon! We're working on comprehensive benchmarks against other vector databases.
🛣️ Roadmap

 Core HNSW implementation
 In-memory storage
 Basic metadata filtering
 Persistent storage with compression
 Batch operations optimization
 Advanced filtering (AND, OR, range queries)
 Index snapshots and backups
 gRPC/HTTP server mode
 Distributed mode
 GPU acceleration support

🤝 Contributing
We welcome contributions! Please see CONTRIBUTING.md for guidelines.
Development Setup
bash# Clone the repository
git clone https://github.com/yourusername/govecdb.git
cd govecdb

# Install dependencies
go mod download

# Run tests
make test

# Run benchmarks
make bench

# Run linter
make lint
📖 Documentation

Getting Started Guide
Architecture Overview
API Documentation
Performance Tuning

🌟 Star History
If you find GoVecDB useful, please consider giving it a star! ⭐
📄 License
MIT License - see LICENSE for details.
🙏 Acknowledgments

Inspired by Chroma, Weaviate, and Qdrant
HNSW algorithm by Yu. A. Malkov and D. A. Yashunin

📬 Contact

GitHub Issues: Report a bug or request a feature
Discussions: Join the conversation


Built with ❤️ by the Go community
