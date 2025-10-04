# 🔄 JSON Format Comparison - GoVecDB vs ChromaDB

## 📋 Identical JSON Structure

Both benchmarks now produce **identical JSON format** for easy comparison!

---

## 🎯 Complete JSON Example

### GoVecDB Output: `govecdb_benchmark_results_20251004_153045.json`

```json
{
  "metadata": {
    "timestamp": "2025-10-04T15:30:45Z",
    "dimensions": [
      2, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048
    ],
    "vector_counts": [
      1000, 5000, 10000, 25000, 50000, 100000
    ],
    "total_tests": 156,
    "database": "govecdb"
  },
  "results": [
    {
      "dimension": 128,
      "num_vectors": 10000,
      "operation": "batch_insert",
      "total_time": 0.523456,
      "avg_time": 0.000052,
      "min_time": 0.000052,
      "max_time": 0.000052,
      "throughput": 19120.45,
      "recall": 0.0,
      "search_quality": 0.0
    },
    {
      "dimension": 128,
      "num_vectors": 10000,
      "operation": "search_k10",
      "total_time": 0.042300,
      "avg_time": 0.000423,
      "min_time": 0.000301,
      "max_time": 0.001234,
      "throughput": 23640.66,
      "recall": 0.0,
      "search_quality": 0.9532
    },
    {
      "dimension": 128,
      "num_vectors": 10000,
      "operation": "exact_search",
      "total_time": 0.008200,
      "avg_time": 0.000082,
      "min_time": 0.000065,
      "max_time": 0.000150,
      "throughput": 12195.12,
      "recall": 1.0,
      "search_quality": 0.0
    }
  ]
}
```

### ChromaDB Output: `chromadb_benchmark_results_20251004_153045.json`

```json
{
  "metadata": {
    "timestamp": "2025-10-04T15:30:45+00:00",
    "dimensions": [
      2, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048
    ],
    "vector_counts": [
      1000, 5000, 10000, 25000, 50000, 100000
    ],
    "total_tests": 168,
    "database": "chromadb"
  },
  "results": [
    {
      "dimension": 128,
      "num_vectors": 10000,
      "operation": "batch_insert",
      "total_time": 0.876543,
      "avg_time": 0.000088,
      "min_time": 0.000088,
      "max_time": 0.000088,
      "throughput": 11408.29,
      "recall": 0.0,
      "search_quality": 0.0
    },
    {
      "dimension": 128,
      "num_vectors": 10000,
      "operation": "search_k10",
      "total_time": 0.078900,
      "avg_time": 0.000789,
      "min_time": 0.000650,
      "max_time": 0.002100,
      "throughput": 12674.27,
      "recall": 0.0,
      "search_quality": 0.9218
    },
    {
      "dimension": 128,
      "num_vectors": 10000,
      "operation": "exact_search",
      "total_time": 0.012400,
      "avg_time": 0.000124,
      "min_time": 0.000098,
      "max_time": 0.000210,
      "throughput": 8064.52,
      "recall": 0.97,
      "search_quality": 0.0
    }
  ]
}
```

---

## 📊 Field-by-Field Comparison

| Field | GoVecDB | ChromaDB | Match? |
|-------|---------|----------|--------|
| **Metadata** |
| `timestamp` | ✅ RFC3339 | ✅ ISO8601 | ✅ Compatible |
| `dimensions` | ✅ Array of ints | ✅ Array of ints | ✅ Identical |
| `vector_counts` | ✅ Array of ints | ✅ Array of ints | ✅ Identical |
| `total_tests` | ✅ Integer | ✅ Integer | ✅ Identical |
| `database` | ✅ "govecdb" | ✅ "chromadb" | ✅ Different (as expected) |
| **Results** |
| `dimension` | ✅ Integer | ✅ Integer | ✅ Identical |
| `num_vectors` | ✅ Integer | ✅ Integer | ✅ Identical |
| `operation` | ✅ String | ✅ String | ✅ Identical |
| `total_time` | ✅ Float (seconds) | ✅ Float (seconds) | ✅ Identical |
| `avg_time` | ✅ Float (seconds) | ✅ Float (seconds) | ✅ Identical |
| `min_time` | ✅ Float (seconds) | ✅ Float (seconds) | ✅ Identical |
| `max_time` | ✅ Float (seconds) | ✅ Float (seconds) | ✅ Identical |
| `throughput` | ✅ Float (ops/sec) | ✅ Float (ops/sec) | ✅ Identical |
| `recall` | ✅ Float (0-1) | ✅ Float (0-1) | ✅ Identical |
| `search_quality` | ✅ Float (0-1) | ✅ Float (0-1) | ✅ Identical |

**✅ 100% Compatible!** Both use the same structure and data types!

---

## 🔍 Quick Comparison Example

### Python Script to Compare

```python
import json
import pandas as pd

# Load both results
with open('govecdb_benchmark_results_20251004_153045.json') as f:
    govecdb = json.load(f)

with open('chromadb_benchmark_results_20251004_153045.json') as f:
    chromadb = json.load(f)

# Convert to DataFrames
df_gov = pd.DataFrame(govecdb['results'])
df_chroma = pd.DataFrame(chromadb['results'])

# Merge on common keys
merged = df_gov.merge(
    df_chroma,
    on=['dimension', 'num_vectors', 'operation'],
    suffixes=('_govecdb', '_chromadb')
)

# Calculate speedup
merged['speedup'] = merged['avg_time_chromadb'] / merged['avg_time_govecdb']
merged['throughput_ratio'] = merged['throughput_govecdb'] / merged['throughput_chromadb']

# Show comparison
print("\n🏆 Performance Comparison (128 dimensions, 10K vectors)")
print("=" * 80)
for _, row in merged[merged['dimension'] == 128].iterrows():
    print(f"\n📊 {row['operation']}")
    print(f"   GoVecDB:  {row['avg_time_govecdb']*1000:.3f}ms")
    print(f"   ChromaDB: {row['avg_time_chromadb']*1000:.3f}ms")
    print(f"   Speedup:  {row['speedup']:.2f}x", end="")
    if row['speedup'] > 1:
        print(" 🏆 GoVecDB faster!")
    else:
        print(" 🏆 ChromaDB faster!")
```

**Output:**
```
🏆 Performance Comparison (128 dimensions, 10K vectors)
================================================================================

📊 batch_insert
   GoVecDB:  0.052ms
   ChromaDB: 0.088ms
   Speedup:  1.69x 🏆 GoVecDB faster!

📊 search_k10
   GoVecDB:  0.423ms
   ChromaDB: 0.789ms
   Speedup:  1.87x 🏆 GoVecDB faster!

📊 exact_search
   GoVecDB:  0.082ms
   ChromaDB: 0.124ms
   Speedup:  1.51x 🏆 GoVecDB faster!
```

---

## 📈 Visualization Example

### Create Comparison Charts

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Plot throughput comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Batch Insert Throughput
ax1 = axes[0, 0]
subset_gov = df_gov[df_gov['operation'] == 'batch_insert']
subset_chroma = df_chroma[df_chroma['operation'] == 'batch_insert']
ax1.plot(subset_gov['dimension'], subset_gov['throughput'], 
         marker='o', label='GoVecDB', linewidth=2)
ax1.plot(subset_chroma['dimension'], subset_chroma['throughput'], 
         marker='s', label='ChromaDB', linewidth=2)
ax1.set_xlabel('Dimension')
ax1.set_ylabel('Throughput (ops/sec)')
ax1.set_title('Batch Insert Performance')
ax1.legend()
ax1.set_xscale('log')
ax1.grid(True)

# Plot 2: Search Latency
ax2 = axes[0, 1]
subset_gov = df_gov[df_gov['operation'] == 'search_k10']
subset_chroma = df_chroma[df_chroma['operation'] == 'search_k10']
ax2.plot(subset_gov['dimension'], subset_gov['avg_time'] * 1000, 
         marker='o', label='GoVecDB', linewidth=2)
ax2.plot(subset_chroma['dimension'], subset_chroma['avg_time'] * 1000, 
         marker='s', label='ChromaDB', linewidth=2)
ax2.set_xlabel('Dimension')
ax2.set_ylabel('Latency (ms)')
ax2.set_title('Search Latency (k=10)')
ax2.legend()
ax2.set_xscale('log')
ax2.grid(True)

# Plot 3: Search Quality
ax3 = axes[1, 0]
ax3.plot(subset_gov['dimension'], subset_gov['search_quality'] * 100, 
         marker='o', label='GoVecDB', linewidth=2)
ax3.plot(subset_chroma['dimension'], subset_chroma['search_quality'] * 100, 
         marker='s', label='ChromaDB', linewidth=2)
ax3.set_xlabel('Dimension')
ax3.set_ylabel('Search Quality (%)')
ax3.set_title('Search Quality Comparison')
ax3.legend()
ax3.set_xscale('log')
ax3.grid(True)

# Plot 4: Speedup Heatmap
ax4 = axes[1, 1]
operations = ['batch_insert', 'search_k10', 'exact_search']
speedups = []
for op in operations:
    gov = df_gov[df_gov['operation'] == op]['avg_time'].values
    chroma = df_chroma[df_chroma['operation'] == op]['avg_time'].values
    speedup = chroma / gov
    speedups.append(speedup)

im = ax4.imshow(speedups, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2)
ax4.set_xticks(range(len(subset_gov['dimension'])))
ax4.set_xticklabels(subset_gov['dimension'])
ax4.set_yticks(range(len(operations)))
ax4.set_yticklabels(operations)
ax4.set_xlabel('Dimension')
ax4.set_title('Speedup (GoVecDB vs ChromaDB)')
plt.colorbar(im, ax=ax4, label='Speedup (x)')

plt.tight_layout()
plt.savefig('govecdb_vs_chromadb_comparison.png', dpi=300)
print("\n📊 Chart saved: govecdb_vs_chromadb_comparison.png")
```

---

## 🎯 Common Operations Comparison

### Operations in Both Databases

| Operation | GoVecDB | ChromaDB | Notes |
|-----------|---------|----------|-------|
| `batch_insert` | ✅ AddBatch | ✅ add | Both support bulk insert |
| `single_insert` | ✅ Add | ✅ add (single) | Insert one vector |
| `exact_search` | ✅ Search (k=1) | ✅ query (n_results=1) | Find exact match |
| `search_k10` | ✅ Search (k=10) | ✅ query (n_results=10) | KNN search |
| `search_k100` | ✅ Search (k=100) | ✅ query (n_results=100) | Large K search |
| `filtered_search` | ✅ Search + Filter | ✅ query + where | Metadata filtering |
| `update` | ✅ Delete + Add | ✅ update | Update vector |
| `delete` | ✅ Delete | ✅ delete | Remove vector |
| `get_by_id` | ✅ Get | ✅ get | Retrieve by ID |

### ChromaDB-Only Operations

| Operation | ChromaDB | GoVecDB Alternative |
|-----------|----------|---------------------|
| `upsert` | ✅ upsert | Use Delete + Add |
| `count` | ✅ count | Use Count() method |

---

## 🔄 Data Type Mapping

### Time Values

**Both use seconds (float64):**
```
GoVecDB:  0.000423 seconds = 0.423 milliseconds
ChromaDB: 0.000789 seconds = 0.789 milliseconds
```

**Convert to milliseconds for display:**
```python
# Python
time_ms = result['avg_time'] * 1000

// Go
timeMs := result.AvgTime.Seconds() * 1000
```

### Recall and Quality

**Both use 0-1 range (float64):**
```
0.0 = 0% quality/recall
0.5 = 50% quality/recall
1.0 = 100% quality/recall
```

**Convert to percentage:**
```python
# Python
recall_pct = result['recall'] * 100

// Go  
recallPct := result.Recall * 100
```

---

## 📦 Loading JSON in Different Languages

### Python
```python
import json

with open('govecdb_benchmark_results_20251004_153045.json') as f:
    data = json.load(f)
    
print(f"Total tests: {data['metadata']['total_tests']}")
print(f"Database: {data['metadata']['database']}")

for result in data['results']:
    print(f"{result['operation']}: {result['avg_time']*1000:.2f}ms")
```

### Go
```go
import (
    "encoding/json"
    "os"
)

type JSONOutput struct {
    Metadata BenchmarkMetadata     `json:"metadata"`
    Results  []JSONBenchmarkResult `json:"results"`
}

func loadResults(filename string) (*JSONOutput, error) {
    data, err := os.ReadFile(filename)
    if err != nil {
        return nil, err
    }
    
    var output JSONOutput
    if err := json.Unmarshal(data, &output); err != nil {
        return nil, err
    }
    
    return &output, nil
}
```

### JavaScript/Node.js
```javascript
const fs = require('fs');

const data = JSON.parse(
    fs.readFileSync('govecdb_benchmark_results_20251004_153045.json', 'utf8')
);

console.log(`Total tests: ${data.metadata.total_tests}`);
console.log(`Database: ${data.metadata.database}`);

data.results.forEach(result => {
    console.log(`${result.operation}: ${(result.avg_time * 1000).toFixed(2)}ms`);
});
```

---

## ✅ Validation Checklist

Use this to verify your JSON files are correct:

- [ ] File exists and is not 0 bytes
- [ ] Can be parsed as valid JSON
- [ ] Has `metadata` and `results` keys at top level
- [ ] `metadata.timestamp` is valid ISO8601/RFC3339
- [ ] `metadata.dimensions` is array of integers
- [ ] `metadata.vector_counts` is array of integers
- [ ] `metadata.total_tests` matches `results` array length
- [ ] All `results` have required fields
- [ ] Time values are in seconds (< 1000 for most operations)
- [ ] Throughput values are positive
- [ ] Recall and search_quality are between 0 and 1

**Quick validation script:**
```python
import json

def validate_json(filename):
    try:
        with open(filename) as f:
            data = json.load(f)
        
        # Check structure
        assert 'metadata' in data, "Missing metadata"
        assert 'results' in data, "Missing results"
        
        # Check metadata
        meta = data['metadata']
        assert 'timestamp' in meta
        assert 'dimensions' in meta
        assert 'vector_counts' in meta
        assert 'total_tests' in meta
        
        # Check results
        for result in data['results']:
            assert all(key in result for key in [
                'dimension', 'num_vectors', 'operation',
                'total_time', 'avg_time', 'throughput'
            ])
            assert 0 <= result['recall'] <= 1
            assert 0 <= result['search_quality'] <= 1
        
        print(f"✅ {filename} is valid!")
        print(f"   Total tests: {meta['total_tests']}")
        print(f"   Database: {meta.get('database', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ {filename} is invalid: {e}")
        return False

# Validate both files
validate_json('govecdb_benchmark_results_20251004_153045.json')
validate_json('chromadb_benchmark_results_20251004_153045.json')
```

---

## 🎉 Summary

✅ **Same JSON structure** across both benchmarks  
✅ **Compatible data types** (float64 for times, int for counts)  
✅ **Easy to compare** using standard JSON tools  
✅ **Standard format** for sharing and analysis  
✅ **Language-agnostic** - load in Python, Go, JS, etc.  

**You can now easily:**
- Load both JSON files side-by-side
- Compare performance metrics directly
- Create visualization dashboards
- Share results with your team
- Automate analysis pipelines

---

**Ready to benchmark!** 🚀📊

*Last updated: October 4, 2025*
