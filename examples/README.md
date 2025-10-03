# 📚 GoVecDB Examples

This directory contains comprehensive examples demonstrating how to use GoVecDB in various scenarios.

## 🎯 Example Categories

### 📖 Basic Usage (`basic-usage/`)
- **HNSW Index Demo** (`hnsw_demo.go`): Core index functionality, search performance, and distance metrics

### 🚀 Advanced Features (`advanced-features/`)
- **Collection Demo** (`collection_demo.go`): Collection operations, filtering, real-world scenarios

### 📊 Benchmarks (`benchmarks/`)
- Performance comparisons with other vector databases
- Load testing and stress testing examples

## 🏃 Running Examples

### Basic HNSW Index Demo
```bash
# Run the HNSW index demonstration
go run examples/basic-usage/hnsw_demo.go
```

### Advanced Collection Features
```bash
# Run comprehensive collection demonstrations
go run examples/advanced-features/collection_demo.go
```

### Using Makefile
```bash
# Run all examples
make examples

# Quick demo (limited output)
make quick-demo
```

## 📋 What Each Example Shows

### 🎯 HNSW Index Demo
- ✅ Index creation and configuration
- ✅ Vector insertion and batch operations
- ✅ Similarity search with different metrics
- ✅ Performance benchmarking
- ✅ Distance function comparisons
- ✅ Filtered search capabilities

### 🚀 Collection Demo
- ✅ Collection lifecycle management
- ✅ Vector CRUD operations
- ✅ Advanced filtering (AND, OR, range queries)
- ✅ Batch processing and performance testing
- ✅ Real-world scenarios:
  - 📚 Document search engine
  - 🛍️ Product recommendation system
  - 🛡️ Content moderation

### 📊 Expected Output
- **Search Latency**: ~1-2ms per query
- **Insertion Rate**: 200-400 vectors/second
- **Memory Usage**: Linear scaling with dataset size
- **Accuracy**: 95%+ similarity recall

## 🎨 Customization

Each example is self-contained and can be modified to:
- 🔧 **Change dimensions**: Modify vector dimensions for your use case
- 🎯 **Adjust parameters**: Tune HNSW parameters (M, EfConstruction)
- 📊 **Add metrics**: Include custom distance functions
- 🗄️ **Modify data**: Use your own dataset and metadata

## 💡 Tips for Your Own Applications

1. **Start Small**: Begin with the basic examples and gradually add complexity
2. **Tune Parameters**: Adjust HNSW parameters based on your data characteristics
3. **Monitor Performance**: Use the benchmark patterns for performance testing
4. **Handle Errors**: All examples include proper error handling patterns
5. **Scale Gradually**: Test with small datasets before scaling to production

## 🚀 Next Steps

After running these examples:
1. Check out the `/docs` directory for detailed API documentation
2. Review the `/tests` directory for additional usage patterns
3. Read `CONTRIBUTING.md` to contribute your own examples
4. Explore the production-ready features in the main library

---

**Happy Learning!** 🎉✨