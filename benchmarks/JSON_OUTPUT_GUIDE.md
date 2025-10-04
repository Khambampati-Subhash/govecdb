# 📊 JSON Output Guide - GoVecDB Benchmark

## ✅ Updates Made

I've updated the `govecdb_benchmark.go` file to match the ChromaDB benchmark output format. Now both benchmarks generate consistent JSON and CSV files for easy comparison!

---

## 🎯 What Changed

### 1. **Added JSON Export Functionality**
The Go benchmark now saves results to JSON files with the same structure as ChromaDB:

```json
{
  "metadata": {
    "timestamp": "2025-10-04T15:30:45Z",
    "dimensions": [2, 8, 16, 32, 64, 128, ...],
    "vector_counts": [1000, 5000, 10000, ...],
    "total_tests": 156,
    "database": "govecdb"
  },
  "results": [
    {
      "dimension": 128,
      "num_vectors": 10000,
      "operation": "batch_insert",
      "total_time": 0.523,
      "avg_time": 0.0000523,
      "min_time": 0.0000523,
      "max_time": 0.0000523,
      "throughput": 19120.45,
      "recall": 0.0,
      "search_quality": 0.0
    },
    ...
  ]
}
```

### 2. **Added CSV Export**
Both benchmarks now also save CSV files for spreadsheet analysis:

```csv
dimension,num_vectors,operation,total_time,avg_time,min_time,max_time,throughput,recall,search_quality
128,10000,batch_insert,0.523000,0.000052,0.000052,0.000052,19120.45,0.0000,0.0000
128,10000,search_k10,0.042000,0.000420,0.000320,0.000580,23809.52,0.0000,0.9532
...
```

### 3. **Timestamped Filenames**
Both benchmarks generate timestamped files to avoid overwriting:

**GoVecDB Output:**
- `govecdb_benchmark_results_20251004_153045.json`
- `govecdb_benchmark_results_20251004_153045.csv`

**ChromaDB Output:**
- `chromadb_benchmark_results_20251004_153045.json`
- `chromadb_benchmark_results_20251004_153045.csv`

### 4. **Enhanced Console Output**
Added timestamps to console output:
```
🚀 GoVecDB Comprehensive Performance Benchmark
⏰ Started: 2025-10-04 15:30:45

📊 Testing Dimension: 128
...

✅ Benchmark Complete!
💾 Results saved to: govecdb_benchmark_results_20251004_153045.json
💾 Results saved to: govecdb_benchmark_results_20251004_153045.csv

⏰ Completed: 2025-10-04 15:45:30
⏱️  Total Duration: 14m45s
```

---

## 📁 File Structure Comparison

### GoVecDB Benchmark (Go)
```go
// JSON structs
type JSONBenchmarkResult struct {
    Dimension     int     `json:"dimension"`
    NumVectors    int     `json:"num_vectors"`
    Operation     string  `json:"operation"`
    TotalTime     float64 `json:"total_time"`      // seconds
    AvgTime       float64 `json:"avg_time"`        // seconds
    MinTime       float64 `json:"min_time"`        // seconds
    MaxTime       float64 `json:"max_time"`        // seconds
    Throughput    float64 `json:"throughput"`      // ops/sec
    Recall        float64 `json:"recall"`          // 0-1
    SearchQuality float64 `json:"search_quality"`  // 0-1
}

type BenchmarkMetadata struct {
    Timestamp    string `json:"timestamp"`
    Dimensions   []int  `json:"dimensions"`
    VectorCounts []int  `json:"vector_counts"`
    TotalTests   int    `json:"total_tests"`
    Database     string `json:"database"`
}
```

### ChromaDB Benchmark (Python)
```python
@dataclass
class BenchmarkResult:
    dimension: int
    num_vectors: int
    operation: str
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    throughput: float
    recall: float = 0.0
    search_quality: float = 0.0
```

**✅ Perfect Match!** Both use the same field names and data types!

---

## 🚀 How to Use

### Running GoVecDB Benchmark

```bash
cd benchmarks

# Option 1: Using the script
./run_benchmark.sh

# Option 2: Direct run
go run govecdb_benchmark.go

# Option 3: Build and run
go build govecdb_benchmark.go
./govecdb_benchmark
```

**Output files will be created automatically:**
- `govecdb_benchmark_results_YYYYMMDD_HHMMSS.json`
- `govecdb_benchmark_results_YYYYMMDD_HHMMSS.csv`

### Running ChromaDB Benchmark (Google Colab)

1. Upload `chromadb_benchmark.py` or `ChromaDB_Benchmark.ipynb` to Colab
2. Run all cells
3. Download the generated files:
   - `chromadb_benchmark_results_YYYYMMDD_HHMMSS.json`
   - `chromadb_benchmark_results_YYYYMMDD_HHMMSS.csv`

---

## 📊 Comparing Results

### Method 1: Using compare_results.py

```bash
python compare_results.py \
  govecdb_benchmark_results_20251004_153045.json \
  chromadb_benchmark_results_20251004_153045.json
```

**Output:**
```
🔬 GoVecDB vs ChromaDB Performance Comparison
==================================================

🏆 batch_insert
   GoVecDB:  0.052ms (19,120 ops/sec)
   ChromaDB: 0.087ms (11,494 ops/sec)
   Winner:   🏆 GoVecDB (1.67x faster)

🏆 search_k10
   GoVecDB:  0.420ms (95.32% quality)
   ChromaDB: 0.789ms (92.18% quality)
   Winner:   🏆 GoVecDB (1.88x faster, +3.14% quality)

📊 Overall Summary
   GoVecDB Wins:  8/10 operations 🏆
   Average:       GoVecDB is 1.82x faster ⚡
```

### Method 2: Manual JSON Comparison

Load both JSON files and compare:

```python
import json
import pandas as pd

# Load results
with open('govecdb_benchmark_results_20251004_153045.json') as f:
    govecdb = json.load(f)

with open('chromadb_benchmark_results_20251004_153045.json') as f:
    chromadb = json.load(f)

# Compare metadata
print(f"GoVecDB tests: {govecdb['metadata']['total_tests']}")
print(f"ChromaDB tests: {chromadb['metadata']['total_tests']}")

# Convert to DataFrames for analysis
df_gov = pd.DataFrame(govecdb['results'])
df_chroma = pd.DataFrame(chromadb['results'])

# Compare average latencies
print("\nAverage Latency by Operation:")
print(df_gov.groupby('operation')['avg_time'].mean())
print(df_chroma.groupby('operation')['avg_time'].mean())
```

### Method 3: Excel/Spreadsheet Analysis

1. Open both CSV files in Excel/Google Sheets
2. Create pivot tables by operation
3. Create comparison charts
4. Calculate speedup ratios

---

## 🔍 JSON Field Descriptions

| Field | Type | Description | Unit |
|-------|------|-------------|------|
| `dimension` | int | Vector dimensionality | dimensions |
| `num_vectors` | int | Number of vectors tested | count |
| `operation` | string | Operation name (e.g., "batch_insert") | - |
| `total_time` | float | Total time for all operations | seconds |
| `avg_time` | float | Average time per operation | seconds |
| `min_time` | float | Fastest operation time | seconds |
| `max_time` | float | Slowest operation time | seconds |
| `throughput` | float | Operations per second | ops/sec |
| `recall` | float | Exact match accuracy (0-1) | ratio |
| `search_quality` | float | Search result quality (0-1) | ratio |

---

## 📈 Example Analysis

### Loading and Analyzing Results

```python
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load GoVecDB results
with open('govecdb_benchmark_results_20251004_153045.json') as f:
    data = json.load(f)

# Extract results
results = pd.DataFrame(data['results'])

# Plot throughput by dimension
plt.figure(figsize=(12, 6))
for op in ['batch_insert', 'search_k10']:
    subset = results[results['operation'] == op]
    plt.plot(subset['dimension'], subset['throughput'], 
             marker='o', label=op)

plt.xlabel('Vector Dimension')
plt.ylabel('Throughput (ops/sec)')
plt.title('GoVecDB Performance by Dimension')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.savefig('govecdb_performance.png')
```

### Quick Stats

```python
# Load results
results = pd.DataFrame(data['results'])

# Summary statistics
print("Operation Summary:")
print(results.groupby('operation').agg({
    'avg_time': ['mean', 'std', 'min', 'max'],
    'throughput': ['mean', 'std'],
    'recall': 'mean',
    'search_quality': 'mean'
}))
```

---

## ✅ Verification Checklist

Before comparing results, verify:

- [ ] Both benchmarks completed successfully
- [ ] JSON files are valid (can be parsed)
- [ ] CSV files can be opened in spreadsheet software
- [ ] Metadata contains expected dimensions and vector counts
- [ ] Results array contains multiple operations
- [ ] Timestamps match your test run time
- [ ] File sizes are reasonable (not 0 bytes)

---

## 🐛 Troubleshooting

### JSON file is empty or 0 bytes

**Cause:** Benchmark crashed before saving results

**Solution:**
```bash
# Check for errors in output
go run govecdb_benchmark.go 2>&1 | tee benchmark.log

# Look for error messages before the save step
grep -i error benchmark.log
```

### Cannot parse JSON

**Cause:** File may be corrupted or incomplete

**Solution:**
```bash
# Validate JSON
python -m json.tool govecdb_benchmark_results_*.json

# If invalid, re-run benchmark
rm govecdb_benchmark_results_*.json
go run govecdb_benchmark.go
```

### Different number of tests between databases

**Cause:** Some tests were skipped or failed

**Solution:**
- Check console output for skipped tests
- Verify dimensions and vector counts match in both configs
- Look for error messages in the output

### CSV has wrong format

**Cause:** Locale/encoding issues

**Solution:**
```python
# Read with explicit parameters
df = pd.read_csv('results.csv', encoding='utf-8', sep=',')
```

---

## 🎯 Key Differences from ChromaDB

### Operations Available

**GoVecDB has:**
- ✅ Batch insert
- ✅ Single insert  
- ✅ Exact search
- ✅ KNN search
- ✅ Large K search
- ✅ Filtered search
- ✅ Update (via Delete+Add)
- ✅ Delete
- ✅ Get by ID
- ✅ Concurrent search

**ChromaDB additionally has:**
- Upsert operations
- Count operations

**Both support:**
- Multiple dimensions (2-2048+)
- Metadata filtering
- Cosine distance
- Batch operations

---

## 🔬 Advanced Usage

### Custom Configuration

Edit the benchmark files to test specific scenarios:

**Go (govecdb_benchmark.go):**
```go
config := TestConfig{
    Dimensions:   []int{128, 384, 768},     // Your dimensions
    VectorCounts: []int{1000, 10000},       // Your sizes
    SearchK:      10,
    NumSearches:  100,
}
```

**Python (chromadb_benchmark.py):**
```python
dimensions = [128, 384, 768]
vector_counts = [1000, 10000]
search_k = 10
num_searches = 100
```

### Running Quick Tests

For faster testing, reduce the scope:

```go
// Quick test config
config := TestConfig{
    Dimensions:   []int{128},           // Single dimension
    VectorCounts: []int{1000},          // Small dataset
    SearchK:      10,
    NumSearches:  50,                   // Fewer iterations
}
```

---

## 📚 Related Files

- **govecdb_benchmark.go** - Main Go benchmark (updated with JSON)
- **chromadb_benchmark.py** - Python benchmark (already had JSON)
- **compare_results.py** - Comparison tool
- **run_benchmark.sh** - Easy runner script
- **README.md** - Comprehensive documentation
- **QUICKSTART.md** - Quick start guide
- **START_HERE.md** - Getting started guide

---

## 🎉 Summary

✅ **JSON output added** - Matching ChromaDB format  
✅ **CSV export added** - For spreadsheet analysis  
✅ **Timestamped files** - No more overwrites  
✅ **Enhanced logging** - Start/end times, duration  
✅ **Same structure** - Easy comparison between databases  

**You can now:**
1. Run both benchmarks
2. Get identical JSON/CSV outputs
3. Compare results easily
4. Analyze performance differences
5. Share results with your team

---

**Happy benchmarking!** 🚀📊

*Last updated: October 4, 2025*
