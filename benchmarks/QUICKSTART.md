# 🚀 Quick Start Guide - GoVecDB vs ChromaDB Benchmarking

## Overview
This guide will help you quickly run performance benchmarks comparing GoVecDB and ChromaDB.

---

## ⚡ Quick Start: GoVecDB (5 minutes)

### 1. Navigate to benchmarks directory
```bash
cd /Users/venkatanagasatyasubhash.khambampati/Documents/GitHub/govecdb/benchmarks
```

### 2. Run the benchmark
```bash
go run govecdb_benchmark.go | tee govecdb_results.txt
```

### 3. Wait for completion
The benchmark will test:
- ✅ Dimensions: 2 to 2048
- ✅ Vector counts: 1K to 100K
- ✅ 10+ operations per configuration
- ⏱️ Estimated time: 10-30 minutes (depending on hardware)

### 4. Results will be printed with emojis!
```
🚀 GoVecDB Comprehensive Performance Benchmark
📊 Testing Dimension: 128
  📦 Vector Count: 10000
    🎲 Generating vectors... ✅
    ⏱️  Batch insert... ✅ (0.05ms, 198234 vec/sec)
    🔍 KNN search (k=10)... ✅ (0.42ms avg, quality: 95.32%)
    ...
```

---

## ⚡ Quick Start: ChromaDB (5 minutes in Google Colab)

### Option 1: Using the Jupyter Notebook (Recommended)

1. **Upload to Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `ChromaDB_Benchmark.ipynb`
   - Or: File → Upload notebook

2. **Run All Cells:**
   - Runtime → Run all
   - Or press Ctrl+F9

3. **Download Results:**
   - Results will auto-download at the end
   - Or manually download from Files panel

### Option 2: Using Python Script

1. **Create a new Colab notebook**

2. **Cell 1: Install dependencies**
```python
!pip install -q chromadb numpy pandas matplotlib
```

3. **Cell 2: Upload benchmark script**
```python
from google.colab import files
uploaded = files.upload()  # Upload chromadb_benchmark.py
```

4. **Cell 3: Run benchmark**
```python
!python chromadb_benchmark.py
```

5. **Cell 4: Download results**
```python
files.download('chromadb_benchmark_results.json')
files.download('chromadb_benchmark_results.csv')
```

### Option 3: Direct Code (Copy-Paste)

Copy the entire content of `chromadb_benchmark.py` into a Colab cell and run it!

---

## 📊 Understanding the Output

### GoVecDB Output Example:
```
📊 Testing Dimension: 128
────────────────────────────────────────────────────────────

  📦 Vector Count: 10000

    🎲 Generating vectors... ✅
    ⏱️  Batch insert... ✅ (0.05ms, 198234 vec/sec)
    🔍 Exact search (k=1)... ✅ (0.08ms avg, recall: 100.00%)
    🔍 KNN search (k=10)... ✅ (0.42ms avg, quality: 95.32%)
    ⚡ Concurrent search (10 threads)... ✅ (0.45ms avg, 22341 qps)
```

### What Each Emoji Means:
- 🚀 = Starting benchmark
- 📊 = Testing new dimension
- 📦 = Testing new vector count
- 🎲 = Generating test data
- ⏱️  = Timing operation
- 🔍 = Search operation
- 📝 = Update operation
- 🗑️  = Delete operation
- 📌 = Get by ID
- ⚡ = Concurrent operation
- ✅ = Operation completed successfully
- ❌ = Operation failed
- 🏆 = Final summary

---

## 📈 Key Metrics to Look For

### 1. **Batch Insert** (⏱️)
```
✅ (0.05ms, 198234 vec/sec)
       ↑           ↑
    latency   throughput
```
- **Lower latency = Better**
- **Higher throughput = Better**

### 2. **Search Operations** (🔍)
```
✅ (0.42ms avg, quality: 95.32%)
       ↑              ↑
    latency       quality
```
- **Lower latency = Faster**
- **Higher quality = More accurate**

### 3. **Recall** (🎯)
```
recall: 100.00%
```
- **Higher = More accurate exact matches**
- 100% = Perfect accuracy

### 4. **Concurrent QPS** (⚡)
```
22341 qps
  ↑
queries per second
```
- **Higher = Better concurrent performance**

---

## 🎯 Typical Performance Expectations

### GoVecDB (Expected):
| Operation | Latency | Throughput |
|-----------|---------|------------|
| Batch Insert | 0.01-0.1ms | 100K-500K/sec |
| Search (k=10) | 0.2-2ms | - |
| Exact Match | 0.05-0.5ms | - |
| Concurrent | - | 10K-50K QPS |

### ChromaDB (Expected):
| Operation | Latency | Throughput |
|-----------|---------|------------|
| Batch Insert | 0.05-0.2ms | 50K-200K/sec |
| Search (k=10) | 0.5-3ms | - |
| Exact Match | 0.1-1ms | - |
| Concurrent | - | 5K-20K QPS |

*Actual results depend on your hardware*

---

## 🔧 Customization

### Modify Test Parameters

**For GoVecDB** (edit `govecdb_benchmark.go`):
```go
config := TestConfig{
    Dimensions:   []int{128, 384, 768},  // Test fewer dimensions
    VectorCounts: []int{1000, 10000},     // Test smaller datasets
    SearchK:      10,
    NumSearches:  50,                     // Fewer searches for speed
}
```

**For ChromaDB** (edit `chromadb_benchmark.py`):
```python
DIMENSIONS = [128, 384, 768]      # Test fewer dimensions
VECTOR_COUNTS = [1000, 10000]     # Test smaller datasets
SEARCH_K = 10
NUM_SEARCHES = 50                 # Fewer searches
```

---

## 💾 Saving and Comparing Results

### Step 1: Save GoVecDB Results
```bash
go run govecdb_benchmark.go > govecdb_results.txt
```

### Step 2: Download ChromaDB Results
From Colab, download:
- `chromadb_benchmark_results.json`
- `chromadb_benchmark_results.csv`

### Step 3: Compare (Optional)
If both produce JSON files:
```bash
python compare_results.py govecdb_results.json chromadb_benchmark_results.json
```

---

## 🐛 Troubleshooting

### GoVecDB Issues:

**Error: cannot find package**
```bash
cd /path/to/govecdb
go mod tidy
go mod download
```

**Error: compilation failed**
```bash
go build ./...  # Build first to check for errors
```

### ChromaDB Issues:

**Error: chromadb not found**
```python
!pip install --upgrade chromadb
```

**Error: Out of memory**
- Reduce `VECTOR_COUNTS` to smaller values
- Reduce `DIMENSIONS` list
- Skip large combinations

**Error: Runtime disconnected**
- Colab has a timeout
- Run in smaller batches
- Use Colab Pro for longer runtime

---

## ⚡ Speed Tips

### For Faster Testing:

1. **Test fewer dimensions:**
   ```python
   DIMENSIONS = [128, 384]  # Instead of all
   ```

2. **Test smaller datasets:**
   ```python
   VECTOR_COUNTS = [1000, 5000, 10000]  # Skip 100K
   ```

3. **Fewer search iterations:**
   ```python
   NUM_SEARCHES = 50  # Instead of 100
   ```

4. **Skip operations:**
   Comment out operations you don't need

### For More Thorough Testing:

1. **Add more dimensions:**
   ```python
   DIMENSIONS = list(range(64, 2048, 64))  # Every 64 dims
   ```

2. **Add more vector counts:**
   ```python
   VECTOR_COUNTS = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
   ```

3. **More search iterations:**
   ```python
   NUM_SEARCHES = 500
   ```

---

## ✅ Success Checklist

Before running, ensure:
- ✅ Go 1.23+ installed (for GoVecDB)
- ✅ Python 3.8+ installed (for ChromaDB)
- ✅ Google Colab account (for ChromaDB)
- ✅ Sufficient disk space (results can be large)
- ✅ Stable internet connection (for Colab)

After running, you should have:
- ✅ Console output with performance metrics
- ✅ Results files (JSON/CSV)
- ✅ No errors in the output
- ✅ Visualizations (optional)

---

## 📞 Need Help?

Common questions:

**Q: How long does it take?**
A: 10-30 minutes per database, depending on configuration

**Q: Can I run both at the same time?**
A: Yes! Run GoVecDB locally and ChromaDB in Colab simultaneously

**Q: What if I get errors?**
A: Check the Troubleshooting section above

**Q: Can I test with my own data?**
A: Yes! Modify the `generateVectors` function

**Q: How do I interpret the results?**
A: See "Understanding the Output" section above

---

## 🎉 You're Ready!

1. Run GoVecDB benchmark
2. Run ChromaDB benchmark  
3. Compare the results
4. Celebrate! 🎊

Happy benchmarking! 🚀
