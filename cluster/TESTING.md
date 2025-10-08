# Distributed System Testing Guide

This document describes the comprehensive distributed system tests for GoVecDB cluster implementation.

## Test Suite Overview

The `cluster_comprehensive_test.go` file contains **10 major test categories** covering all aspects of distributed system behavior:

### 1. Full Stack Integration Test
**Purpose**: End-to-end test of the complete distributed system
- ✅ Cluster formation (5 nodes)
- ✅ Distributed insertion (10K vectors)
- ✅ Distributed search coordination
- ✅ Exact match recall verification (≥95%)
- ✅ Cluster health monitoring

**Expected Performance**:
- Insert rate: >100 vec/sec
- Search latency: <100ms average
- Recall: ≥95%

### 2. High Volume Insertion Test
**Purpose**: Test insertion under sustained high load
- ✅ Multiple insertion waves (5K → 10K → 15K vectors)
- ✅ Continuous load simulation
- ✅ Post-insertion verification (≥90% retrievability)
- ✅ Statistics tracking

**Expected Performance**:
- Sustained insertion rate across multiple waves
- Vector retrievability maintained ≥90%

### 3. Concurrent Search Performance Test
**Purpose**: Test search throughput under concurrent load
- ✅ Multiple concurrency levels (10, 50, 100, 200 clients)
- ✅ 20 queries per client
- ✅ Success rate tracking
- ✅ Latency measurement

**Expected Performance**:
- Success rate: ≥95% at all concurrency levels
- Average latency: <500ms even at 200 concurrent clients
- No crashes or deadlocks

### 4. Node Failure Recovery Test
**Purpose**: Validate fault tolerance and recovery
- ✅ Baseline performance measurement
- ✅ Simulated node failure
- ✅ Performance during degraded state (≥80% success)
- ✅ Node recovery simulation
- ✅ Post-recovery verification

**Test Flow**:
1. Insert 10K vectors, measure baseline
2. Simulate node failure
3. Verify degraded performance (≥80%)
4. Recover failed node
5. Verify full recovery

**Expected Behavior**:
- Cluster state transitions: Healthy → Degraded → Healthy
- Search success rate during failure: ≥80%
- Full recovery after node rejoins

### 5. Network Partition Handling Test
**Purpose**: Test split-brain and partition tolerance
- ✅ Cluster split into two groups (3+3 nodes)
- ✅ Majority partition continues operation
- ✅ Partition detection
- ✅ Network healing
- ✅ Data reconciliation

**Test Scenarios**:
- Split-brain detection
- Majority partition operation
- Minority partition behavior
- Partition healing and reconciliation

### 6. Race Condition Test
**Purpose**: Detect concurrency bugs and race conditions
- ✅ Concurrent insertions (20 workers × 500 vectors each)
- ✅ Concurrent searches during insertions
- ✅ Shard redistribution under load
- ✅ Node addition under concurrent operations

**Test Scenarios**:
- **Test 1**: 20 workers inserting concurrently (10K total vectors)
- **Test 2**: 50 clients searching while 5 workers insert (500 searches + 5K inserts)
- **Test 3**: Add new node while 10 clients perform searches

**Run with**:
```bash
go test -race -run TestDistributedCluster_Comprehensive/Race_Conditions
```

### 7. Shard Rebalancing Test
**Purpose**: Validate shard distribution and rebalancing
- ✅ Initial distribution (3 nodes)
- ✅ Add 2 new nodes (scale to 5 nodes)
- ✅ Trigger rebalancing
- ✅ Verify balanced distribution (≤20% imbalance)
- ✅ Data accessibility after rebalancing (≥95%)

**Metrics**:
- Imbalance: (max_shards - min_shards) / max_shards
- Target: ≤20% imbalance
- Data integrity: ≥95% vectors still accessible

### 8. Query Coordination Stress Test
**Purpose**: Stress test the query coordinator
- ✅ 50K vector dataset
- ✅ 100 concurrent clients
- ✅ 50 queries per client (5000 total)
- ✅ Mixed query types (k=5, 50, 500)
- ✅ Latency tracking (min/avg/max)

**Expected Performance**:
- Success rate: ≥95%
- Throughput: ≥100 queries/sec
- No coordinator crashes or hangs

### 9. Data Consistency Test
**Purpose**: Verify consistency across replicas
- ✅ Replication factor = 3
- ✅ Insert 1K vectors
- ✅ Cross-replica consistency check (≥99%)
- ✅ Consistency after node failure (≥90%)
- ✅ Read-after-write consistency

**Test Scenarios**:
- **Test 1**: Search same vector from all nodes → same results
- **Test 2**: Node failure → verify replicas still consistent

### 10. Scalability Test
**Purpose**: Measure horizontal scaling efficiency
- ✅ Test node counts: 2, 4, 6, 8, 10
- ✅ Insert 10K vectors per configuration
- ✅ Measure insert rate, search rate, latency
- ✅ Calculate scaling efficiency

**Metrics**:
- Insert speedup vs baseline (2 nodes)
- Scaling efficiency = speedup / scale_factor × 100%
- Target: ≥50% efficiency up to 10 nodes

---

## Running the Tests

### Run All Tests
```bash
cd /Users/venkatanagasatyasubhash.khambampati/Documents/GitHub/govecdb

# Run all comprehensive tests
go test -v ./cluster -run TestDistributedCluster_Comprehensive -timeout 30m

# Run with race detector (slower but catches concurrency bugs)
go test -race -v ./cluster -run TestDistributedCluster_Comprehensive -timeout 60m
```

### Run Individual Tests

```bash
# Full stack integration
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Full_Stack

# High volume insertion
go test -v ./cluster -run TestDistributedCluster_Comprehensive/High_Volume

# Concurrent search
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Concurrent_Search

# Node failure recovery
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Node_Failure

# Network partition
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Network_Partition

# Race conditions (MUST run with -race)
go test -race -v ./cluster -run TestDistributedCluster_Comprehensive/Race_Conditions

# Shard rebalancing
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Shard_Rebalancing

# Query coordination stress
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Query_Coordination

# Data consistency
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Data_Consistency

# Scalability
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Scalability -timeout 30m
```

### Quick Sanity Check (Fast)
```bash
# Run subset of tests for quick validation
go test -v ./cluster -run TestDistributedCluster_Comprehensive/Full_Stack -timeout 5m
go test -race -v ./cluster -run TestDistributedCluster_Comprehensive/Race_Conditions -timeout 10m
```

---

## Performance Benchmarks

After tests pass, run performance benchmarks:

```bash
# Benchmark distributed insertion
go test -bench=BenchmarkDistributedInsert -benchmem ./cluster

# Benchmark distributed search
go test -bench=BenchmarkDistributedSearch -benchmem ./cluster

# Full benchmark suite
go test -bench=. -benchmem ./cluster -timeout 60m
```

---

## What Each Test Validates

### ✅ Correctness
- Data integrity across operations
- Search result accuracy (recall ≥95%)
- Replica consistency (≥99%)

### ✅ Performance
- Insert throughput (target: 100-1000 vec/sec per node)
- Search latency (<100ms average)
- Concurrent query handling (200+ concurrent clients)

### ✅ Fault Tolerance
- Node failure handling
- Network partition tolerance
- Automatic recovery

### ✅ Scalability
- Linear scaling up to 5-10 nodes
- Minimal performance degradation under load
- Efficient shard rebalancing

### ✅ Concurrency Safety
- No race conditions (verified with `-race`)
- No deadlocks
- Safe concurrent operations

---

## Understanding Test Output

### Success Indicators
```
✓ Cluster initialized with 5 nodes
✓ Inserted 10000 vectors in 15s (666 vec/sec)
✓ Performed 100 searches in 5s (avg: 50ms per search)
✓ Exact match recall: 98.0% (49/50)
✓ Cluster health: healthy
```

### Failure Indicators
```
✗ Insert rate too slow: 45 vec/sec (expected > 100)
✗ Search latency too high: 250ms (expected < 100ms)
✗ Recall too low: 78.0% (expected >= 95%)
✗ Success rate too low during degradation: 65.0%
```

### Performance Metrics
```
--- Wave 1: 5000 vectors ---
✓ Wave 1 completed in 8.5s (588 vec/sec)
  Verification: 96.0% (96/100) vectors retrievable

--- Concurrency 100 results: ---
✓ Total queries: 2000
✓ Success rate: 98.5% (1970/2000)
✓ Avg latency: 45ms
✓ Throughput: 285 queries/sec
```

---

## Expected Test Duration

| Test | Duration | With `-race` |
|------|----------|--------------|
| Full Stack Integration | 30-60s | 60-120s |
| High Volume Insertion | 60-90s | 120-180s |
| Concurrent Search | 60-120s | 120-240s |
| Node Failure Recovery | 30-45s | 60-90s |
| Network Partition | 45-60s | 90-120s |
| Race Conditions | 60-90s | 120-180s |
| Shard Rebalancing | 45-60s | 90-120s |
| Query Coordination Stress | 120-180s | 240-360s |
| Data Consistency | 30-45s | 60-90s |
| Scalability | 300-600s | 600-1200s |
| **TOTAL** | **10-20 min** | **20-40 min** |

---

## Continuous Integration

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml
name: Distributed System Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.23'
      
      # Fast sanity checks
      - name: Quick Tests
        run: |
          go test -v ./cluster -run TestDistributedCluster_Comprehensive/Full_Stack -timeout 5m
      
      # Race detector tests
      - name: Race Condition Tests
        run: |
          go test -race -v ./cluster -run TestDistributedCluster_Comprehensive/Race_Conditions -timeout 15m
      
      # Full test suite (longer timeout)
      - name: Full Test Suite
        run: |
          go test -v ./cluster -run TestDistributedCluster_Comprehensive -timeout 30m
```

---

## Troubleshooting

### Tests Hang
- Check for deadlocks in coordinator or manager
- Verify network mock isn't blocking
- Increase timeout: `-timeout 60m`

### High Failure Rate
- Check node health detection threshold
- Verify replication factor configuration
- Review shard distribution logic

### Race Detector Failures
```bash
# Run specific test with race detector and verbose output
go test -race -v ./cluster -run TestDistributedCluster_Comprehensive/Race_Conditions
```

Common race conditions to check:
- Concurrent map access (use sync.RWMutex)
- Shared state in goroutines (use atomic operations)
- Channel synchronization issues

### Performance Issues
- Profile the slow tests:
```bash
go test -cpuprofile=cpu.prof -memprofile=mem.prof -bench=. ./cluster
go tool pprof cpu.prof
```

---

## Production Readiness Criteria

**Before deploying distributed cluster to production:**

✅ **All tests pass** (without `-race` warnings)
✅ **Performance targets met**:
  - Insert: ≥100 vec/sec per node
  - Search: <100ms p99 latency
  - Concurrent: ≥200 clients supported

✅ **Fault tolerance validated**:
  - Node failure: ≤20% performance degradation
  - Network partition: Majority partition continues
  - Recovery: <10s to restore health

✅ **Scalability confirmed**:
  - Linear scaling up to 10 nodes (≥50% efficiency)
  - Rebalancing: <30s per 10K vectors

✅ **No race conditions** (run all tests with `-race`)

✅ **Data consistency**: ≥99% replica consistency

---

## Next Steps

1. **Run the tests**: `go test -v ./cluster -run TestDistributedCluster_Comprehensive`
2. **Review failures**: Focus on tests with ✗ marks
3. **Fix issues**: Address race conditions, performance bottlenecks
4. **Re-run with `-race`**: Verify no concurrency bugs
5. **Benchmark**: Measure production-level performance
6. **Scale test**: Test with 10+ nodes and 1M+ vectors

For the **30 MILLION vectors/day** requirement:
- Target: 1000+ vec/sec distributed insertion (8 nodes × 125 vec/sec)
- Run scalability test to measure actual throughput
- Adjust shard count and replication factor based on results

---

## Questions?

If tests fail or performance is inadequate:
1. Review test output for specific failure reasons
2. Check cluster health logs
3. Profile slow operations
4. Adjust HNSW parameters for your dimension/scale
5. Consider increasing node count or adjusting replication factor

**The comprehensive test suite validates your cluster is production-ready for massive scale!**
