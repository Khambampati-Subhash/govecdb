# 🚀 GoVecDB vs ChromaDB Performance Comparison

Comprehensive benchmark comparing GoVecDB and ChromaDB across various dimensions and operations.

## 📋 Test Dimensions

- **Vector Dimensions**: 2, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048
- **Vector Counts**: 1K, 5K, 10K, 25K, 50K, 100K
- **Operations Tested**: Insert, Search, Update, Delete, Filtered Search, Concurrent Operations

## 🏃 Running the Benchmarks

### For GoVecDB (Go)

```bash
# Navigate to benchmarks directory
cd benchmarks

# Run the Go benchmark
go run govecdb_benchmark.go
```

### For ChromaDB (Python) - Google Colab

```python
# Install ChromaDB
!pip install chromadb numpy pandas matplotlib

# Upload the benchmark script to Colab or run directly:
!wget https://raw.githubusercontent.com/YOUR_REPO/benchmarks/chromadb_benchmark.py
!python chromadb_benchmark.py
```

Or paste the entire Python code into a Colab cell and run it.

## 📊 Benchmark Operations

### 1. **Batch Insert** 📥
- Inserts all vectors in a single batch operation
- Measures: Total time, throughput (vectors/sec)

### 2. **Single Insert** ➕
- Inserts vectors one at a time
- Measures: Average, min, max time per insert

### 3. **Exact Search** 🎯
- Searches for exact vector matches (k=1)
- Measures: Average time, recall rate

### 4. **KNN Search** 🔍
- K-nearest neighbor search (k=10)
- Measures: Average time, search quality

### 5. **Large K Search** 📊
- Search with large K (k=100)
- Measures: Scalability with result size

### 6. **Filtered Search** 🔎
- Search with metadata filters
- Measures: Filter overhead

### 7. **Update Operations** 📝
- Updates existing vectors
- Measures: Update performance

### 8. **Delete Operations** 🗑️
- Deletes vectors by ID
- Measures: Delete performance

### 9. **Get by ID** 📌
- Retrieves vectors by ID
- Measures: Lookup speed

### 10. **Concurrent Search** ⚡
- Multiple threads searching simultaneously
- Measures: Throughput, QPS

## 📈 Output Format

Both benchmarks produce:

1. **Console Output** - Real-time progress with emojis
2. **Summary Statistics** - Aggregated performance metrics
3. **JSON Results** (ChromaDB) - Detailed results for analysis

### GoVecDB Output Example:
```
🚀 GoVecDB Comprehensive Performance Benchmark
============================================================

📊 Testing Dimension: 128
────────────────────────────────────────────────────────────

  📦 Vector Count: 10000

    🎲 Generating vectors... ✅
    ⏱️  Batch insert... ✅ (0.05ms, 198234 vec/sec)
    ⏱️  Single insert... ✅ (0.12ms avg)
    🔍 Exact search (k=1)... ✅ (0.08ms avg, recall: 100.00%)
    🔍 KNN search (k=10)... ✅ (0.42ms avg, quality: 95.32%)
    🔍 Large K search (k=100)... ✅ (1.23ms avg)
    🔍 Filtered search... ✅ (0.58ms avg)
    📝 Update operations... ✅ (0.15ms avg)
    🗑️  Delete operations... ✅ (0.09ms avg)
    📌 Get by ID... ✅ (0.03ms avg)
    ⚡ Concurrent search (10 threads)... ✅ (0.45ms avg, 22341 qps)
```

### ChromaDB Output Example:
```
🚀 ChromaDB Comprehensive Performance Benchmark
============================================================

📊 Testing Dimension: 128
────────────────────────────────────────────────────────────

  📦 Vector Count: 10000

    🎲 Generating vectors... ✅
    ⏱️  Batch insert... ✅ (0.08ms, 125000 vec/sec)
    ⏱️  Single insert... ✅ (0.23ms avg)
    🔍 Exact search (k=1)... ✅ (0.15ms avg, recall: 98.50%)
    🔍 KNN search (k=10)... ✅ (0.68ms avg, quality: 92.15%)
    ...
```

## 🔬 Key Metrics Explained

### **Throughput (ops/sec)**
Higher is better. Number of operations per second.

### **Average Time (ms)**
Lower is better. Mean operation latency.

### **Recall (%)**
Higher is better. Accuracy of exact match retrieval.

### **Search Quality (%)**
Higher is better. Quality of approximate nearest neighbor results.

### **QPS (Queries Per Second)**
Higher is better. Concurrent query throughput.

## 🎯 What to Look For

1. **Insertion Speed**: Which DB is faster at bulk inserts?
2. **Search Latency**: Sub-millisecond search times?
3. **Scalability**: Performance with high dimensions?
4. **Concurrent Performance**: How well does it handle multiple threads?
5. **Memory Efficiency**: Resource usage patterns
6. **Search Quality**: Accuracy of approximate results

## 📝 Notes

- **GoVecDB**: Pure Go, HNSW algorithm, optimized for low latency
- **ChromaDB**: Python, DuckDB backend, designed for ease of use
- **Fair Comparison**: Same test vectors, same operations, same metrics
- **Google Colab**: No resource constraints for Python benchmark

## 🚀 Quick Start Commands

### GoVecDB:
```bash
cd /path/to/govecdb/benchmarks
go run govecdb_benchmark.go > govecdb_results.txt
```

### ChromaDB (Colab):
```python
# Cell 1: Install
!pip install -q chromadb numpy pandas

# Cell 2: Run
!python chromadb_benchmark.py

# Cell 3: View Results
import pandas as pd
df = pd.read_csv('chromadb_benchmark_results.csv')
print(df.groupby('operation')['avg_time'].mean() * 1000)  # ms
```

## 📊 Expected Performance Ranges

Based on typical hardware:

| Operation | GoVecDB | ChromaDB |
|-----------|---------|----------|
| Batch Insert | 0.01-0.1ms | 0.05-0.2ms |
| Search (k=10) | 0.2-2ms | 0.5-3ms |
| Exact Match | 0.05-0.5ms | 0.1-1ms |
| Update | 0.1-0.5ms | 0.2-1ms |
| Delete | 0.05-0.2ms | 0.1-0.5ms |
| Concurrent QPS | 10K-50K | 5K-20K |

*Actual results will vary based on hardware and configuration*

## 🎨 Visualization (Optional)

For ChromaDB results in Colab:

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('chromadb_benchmark_results.csv')

# Plot search latency by dimension
search_data = df[df['operation'] == 'search_k10']
plt.figure(figsize=(12, 6))
plt.plot(search_data['dimension'], search_data['avg_time'] * 1000, marker='o')
plt.xlabel('Dimension')
plt.ylabel('Latency (ms)')
plt.title('Search Latency vs Dimension')
plt.grid(True)
plt.show()
```

## ✅ Success Criteria

The benchmark is successful if:
- ✅ All operations complete without errors
- ✅ Results are consistent across runs
- ✅ Performance metrics are reasonable
- ✅ Output is easy to read and interpret

Happy benchmarking! 🎉
