# 🤝 Contributing to GoVecDB

Thank you for your interest in contributing to GoVecDB! We welcome contributions from the community and are excited to work with you to make GoVecDB even better.

## 🚀 Quick Start

### Prerequisites
- **Go 1.23+** installed
- **Git** for version control
- **Make** (optional but recommended)

### 🛠️ Development Setup

1. **Fork & Clone**
   ```bash
   # Fork the repository on GitHub, then:
   git clone https://github.com/khambampati-subhash/govecdb.git
   cd govecdb
   ```

2. **Install Dependencies**
   ```bash
   go mod download
   go mod tidy
   ```

3. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   make test
   # Or without make:
   go test ./...
   ```

4. **Run Demo**
   ```bash
   make demo
   # Or without make:
   go run ./cmd/demo/
   ```

## 🌟 How to Contribute

### 🐛 Bug Reports
- **Search existing issues** first
- Use the **bug report template**
- Include:
  - Go version (`go version`)
  - Operating system
  - Steps to reproduce
  - Expected vs actual behavior
  - Code samples if applicable

### ✨ Feature Requests
- Check **existing feature requests**
- Use the **feature request template**
- Explain:
  - Use case and motivation
  - Proposed API design
  - Implementation considerations

### 🔧 Code Contributions

#### 1. Create a Branch
```bash
git checkout -b feature/awesome-feature
# or
git checkout -b fix/important-bug
```

#### 2. Make Changes
- **Follow Go conventions** and idioms
- **Write tests** for new functionality
- **Update documentation** if needed
- **Add benchmarks** for performance-critical code

#### 3. Test Your Changes
```bash
# Run all tests
make test

# Run benchmarks
make bench

# Check code formatting
make fmt

# Run linting
make lint

# Integration tests
make test-integration
```

#### 4. Commit Guidelines
Follow **Conventional Commits**:
```bash
# Examples:
feat(index): add HNSW parameter tuning
fix(collection): resolve metadata filtering issue
docs(readme): update installation instructions
perf(search): optimize distance calculations
test(cluster): add chaos engineering tests
```

#### 5. Submit Pull Request
- **Clear title** describing the change
- **Detailed description** with context
- **Link related issues**
- **Include test results**
- **Update CHANGELOG.md** if needed

## 📋 Code Standards

### 🎯 Code Quality
- **Test Coverage**: Aim for >90% coverage
- **Documentation**: All public APIs must be documented
- **Error Handling**: Comprehensive error handling with meaningful messages
- **Performance**: Benchmark performance-critical paths
- **Security**: Follow security best practices

### 📝 Go Style Guidelines
```go
// ✅ Good: Clear, documented, tested
// Package collection provides vector collection management.
package collection

// VectorCollection manages a collection of vectors with CRUD operations.
type VectorCollection struct {
    store api.VectorStore
    index api.VectorIndex
}

// Add inserts a vector into the collection.
// Returns ErrVectorExists if the vector ID already exists.
func (c *VectorCollection) Add(ctx context.Context, vector *api.Vector) error {
    if err := vector.Validate(); err != nil {
        return fmt.Errorf("invalid vector: %w", err)
    }
    // ... implementation
}
```

### 🧪 Testing Standards
```go
func TestVectorCollection_Add(t *testing.T) {
    tests := []struct {
        name    string
        vector  *api.Vector
        wantErr bool
    }{
        {
            name: "valid vector",
            vector: &api.Vector{
                ID:   "test1",
                Data: []float32{1.0, 2.0, 3.0},
            },
            wantErr: false,
        },
        // ... more test cases
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // ... test implementation
        })
    }
}
```

## 📦 Project Structure

```
govecdb/
├── 📁 api/           # Core interfaces and types
├── 📁 collection/    # Collection implementation
├── 📁 index/         # HNSW index implementation
├── 📁 store/         # Storage implementations
├── 📁 cluster/       # Distributed features
├── 📁 persist/       # Persistence layer
├── 📁 examples/      # Usage examples
├── 📁 benchmarks/    # Performance comparisons
├── 📁 docs/          # Documentation
├── 📁 internal/      # Internal utilities
└── 📁 tests/         # Integration tests
```

## 🎯 Areas for Contribution

### 🔥 High Priority
- **Performance optimizations** in search algorithms
- **Additional distance metrics** (Hamming, Jaccard, etc.)
- **Query optimization** features
- **Memory usage improvements**
- **Distributed system enhancements**

### 🌱 Good First Issues
- **Documentation improvements**
- **Example applications**
- **Test coverage improvements**
- **Benchmark additions**
- **Error message enhancements**

### 🚀 Advanced Features
- **GPU acceleration** support
- **Approximate nearest neighbor** algorithms
- **Vector quantization** techniques
- **Advanced clustering** features
- **Real-time replication**

## 🔄 Development Workflow

### 🏗️ Building
```bash
# Build the library
make build

# Build examples
make build-examples

# Cross-compilation
make build-all
```

### 🧪 Testing
```bash
# Unit tests
make test

# Integration tests
make test-integration

# Benchmark tests
make bench

# Coverage report
make coverage
```

### 📊 Performance Testing
```bash
# Run performance benchmarks
make perf

# Compare with other vector databases
make compare

# Memory profiling
make profile-mem

# CPU profiling
make profile-cpu
```

## 📖 Documentation

### 📚 Types of Documentation
- **API Documentation**: Go doc comments
- **User Guides**: Markdown in `/docs`
- **Examples**: Working code in `/examples`
- **Benchmarks**: Performance comparisons

### 📝 Documentation Standards
- **Clear examples** for all public APIs
- **Performance characteristics** noted
- **Error conditions** documented
- **Thread safety** guarantees specified

## 🎉 Recognition

Contributors are recognized in:
- **README.md** contributors section
- **CHANGELOG.md** for significant contributions
- **GitHub releases** acknowledgments

## 📞 Getting Help

- **💬 Discussions**: GitHub Discussions for questions
- **🐛 Issues**: GitHub Issues for bugs
- **📧 Contact**: Maintainer email for security issues
- **📖 Documentation**: Check `/docs` for detailed guides

## 🔒 Security

For security vulnerabilities:
- **DO NOT** create public issues
- **Email directly** to maintainer
- **Include** full details and reproduction steps
- **Wait for response** before public disclosure

## 📋 Pull Request Checklist

Before submitting:
- [ ] 🧪 All tests pass
- [ ] 📊 Benchmarks show no regression
- [ ] 📖 Documentation updated
- [ ] 🎯 Code follows style guidelines
- [ ] ✅ Commit messages follow convention
- [ ] 🔗 Related issues linked
- [ ] 📝 CHANGELOG.md updated (if applicable)

## 🙏 Thank You!

Every contribution makes GoVecDB better for everyone. We appreciate your time and effort in making this project successful!

---

**Happy Contributing!** 🚀✨
