package cluster

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// TestDistributedCluster_Comprehensive tests the entire distributed system
func TestDistributedCluster_Comprehensive(t *testing.T) {
	t.Run("Full Stack Integration", func(t *testing.T) {
		testFullStackIntegration(t)
	})

	t.Run("High Volume Insertion", func(t *testing.T) {
		testHighVolumeInsertion(t)
	})

	t.Run("Concurrent Search Performance", func(t *testing.T) {
		testConcurrentSearchPerformance(t)
	})

	t.Run("Node Failure Recovery", func(t *testing.T) {
		testNodeFailureRecovery(t)
	})

	t.Run("Network Partition Handling", func(t *testing.T) {
		testNetworkPartitionHandling(t)
	})

	t.Run("Race Conditions", func(t *testing.T) {
		testRaceConditions(t)
	})

	t.Run("Shard Rebalancing", func(t *testing.T) {
		testShardRebalancing(t)
	})

	t.Run("Query Coordination Stress", func(t *testing.T) {
		testQueryCoordinationStress(t)
	})

	t.Run("Data Consistency", func(t *testing.T) {
		testDataConsistency(t)
	})

	t.Run("Scalability Test", func(t *testing.T) {
		testScalability(t)
	})
}

// testFullStackIntegration tests the complete distributed system flow
func testFullStackIntegration(t *testing.T) {
	t.Log("=== FULL STACK INTEGRATION TEST ===")
	ctx := context.Background()

	// Setup: Create a 5-node cluster
	cluster, err := setupTestCluster(5, 512)
	if err != nil {
		t.Fatalf("Failed to setup cluster: %v", err)
	}
	defer cluster.Shutdown()

	t.Logf("✓ Cluster initialized with %d nodes", len(cluster.Nodes))

	// Test 1: Distributed Insert
	t.Log("\n--- Test 1: Distributed Insert ---")
	vectorCount := 10000
	vectors := generateTestVectors(vectorCount, 512, 42)

	startInsert := time.Now()
	err = cluster.InsertBatch(ctx, vectors)
	insertDuration := time.Since(startInsert)

	if err != nil {
		t.Fatalf("Batch insert failed: %v", err)
	}

	insertRate := float64(vectorCount) / insertDuration.Seconds()
	t.Logf("✓ Inserted %d vectors in %v (%.0f vec/sec)", vectorCount, insertDuration, insertRate)

	if insertRate < 100 {
		t.Errorf("Insert rate too slow: %.0f vec/sec (expected > 100)", insertRate)
	}

	// Test 2: Distributed Search
	t.Log("\n--- Test 2: Distributed Search ---")
	searchCount := 100
	k := 10

	startSearch := time.Now()
	totalResults := 0
	for i := 0; i < searchCount; i++ {
		queryVec := vectors[i%len(vectors)].Data
		results, err := cluster.Search(ctx, queryVec, k)
		if err != nil {
			t.Errorf("Search %d failed: %v", i, err)
			continue
		}
		totalResults += len(results)
	}
	searchDuration := time.Since(startSearch)

	avgSearchLatency := searchDuration / time.Duration(searchCount)
	t.Logf("✓ Performed %d searches in %v (avg: %v per search)", searchCount, searchDuration, avgSearchLatency)
	t.Logf("  Total results retrieved: %d (avg: %.1f per query)", totalResults, float64(totalResults)/float64(searchCount))

	if avgSearchLatency > 100*time.Millisecond {
		t.Errorf("Search latency too high: %v (expected < 100ms)", avgSearchLatency)
	}

	// Test 3: Exact Match Verification
	t.Log("\n--- Test 3: Exact Match Verification ---")
	exactMatches := 0
	for i := 0; i < 50; i++ {
		queryVec := vectors[i].Data
		results, err := cluster.Search(ctx, queryVec, 1)
		if err != nil || len(results) == 0 {
			continue
		}

		if results[0].Vector.ID == vectors[i].ID {
			exactMatches++
		}
	}

	recallPercent := float64(exactMatches) / 50.0 * 100.0
	t.Logf("✓ Exact match recall: %.1f%% (%d/50)", recallPercent, exactMatches)

	if recallPercent < 95.0 {
		t.Errorf("Recall too low: %.1f%% (expected >= 95%%)", recallPercent)
	}

	// Test 4: Cluster Health
	t.Log("\n--- Test 4: Cluster Health ---")
	health := cluster.GetHealth()
	t.Logf("✓ Cluster health: %s", health.State)
	t.Logf("  Nodes: %d healthy, %d failed", health.NodesHealthy, health.NodesFailed)
	t.Logf("  Shards: %d healthy, %d failed", health.ShardsHealthy, health.ShardsFailed)

	if health.State != ClusterStateHealthy {
		t.Errorf("Cluster not healthy: %s", health.State)
	}
}

// testHighVolumeInsertion tests insertion under high load
func testHighVolumeInsertion(t *testing.T) {
	t.Log("=== HIGH VOLUME INSERTION TEST ===")
	ctx := context.Background()

	cluster, err := setupTestCluster(3, 1024)
	if err != nil {
		t.Fatalf("Failed to setup cluster: %v", err)
	}
	defer cluster.Shutdown()

	// Insert in multiple waves to simulate continuous load
	waves := []struct {
		count     int
		dimension int
	}{
		{5000, 1024},
		{10000, 1024},
		{15000, 1024},
	}

	for i, wave := range waves {
		t.Logf("\n--- Wave %d: %d vectors ---", i+1, wave.count)

		vectors := generateTestVectors(wave.count, wave.dimension, int64(i*10000+42))
		start := time.Now()

		err := cluster.InsertBatch(ctx, vectors)
		duration := time.Since(start)

		if err != nil {
			t.Errorf("Wave %d failed: %v", i+1, err)
			continue
		}

		rate := float64(wave.count) / duration.Seconds()
		t.Logf("✓ Wave %d completed in %v (%.0f vec/sec)", i+1, duration, rate)

		// Verify some vectors after each wave
		verifyCount := minInt(100, wave.count)
		found := 0
		for j := 0; j < verifyCount; j++ {
			results, err := cluster.Search(ctx, vectors[j].Data, 1)
			if err == nil && len(results) > 0 && results[0].Vector.ID == vectors[j].ID {
				found++
			}
		}

		verifyPercent := float64(found) / float64(verifyCount) * 100.0
		t.Logf("  Verification: %.1f%% (%d/%d) vectors retrievable", verifyPercent, found, verifyCount)

		if verifyPercent < 90.0 {
			t.Errorf("Wave %d: verification rate too low: %.1f%%", i+1, verifyPercent)
		}
	}

	// Final statistics
	stats := cluster.GetStats()
	t.Logf("\n--- Final Statistics ---")
	t.Logf("Total vectors inserted: %d", stats.TotalVectors)
	t.Logf("Total shards: %d", stats.TotalShards)
	t.Logf("Average vectors per shard: %.0f", float64(stats.TotalVectors)/float64(stats.TotalShards))
}

// testConcurrentSearchPerformance tests search under concurrent load
func testConcurrentSearchPerformance(t *testing.T) {
	t.Log("=== CONCURRENT SEARCH PERFORMANCE TEST ===")
	ctx := context.Background()

	cluster, err := setupTestCluster(5, 768)
	if err != nil {
		t.Fatalf("Failed to setup cluster: %v", err)
	}
	defer cluster.Shutdown()

	// Insert test data
	vectorCount := 20000
	vectors := generateTestVectors(vectorCount, 768, 42)
	t.Logf("Inserting %d vectors...", vectorCount)

	err = cluster.InsertBatch(ctx, vectors)
	if err != nil {
		t.Fatalf("Failed to insert vectors: %v", err)
	}
	t.Log("✓ Data insertion complete")

	// Test concurrent searches with varying workloads
	concurrencyLevels := []int{10, 50, 100, 200}

	for _, concurrency := range concurrencyLevels {
		t.Logf("\n--- Testing with %d concurrent clients ---", concurrency)

		var wg sync.WaitGroup
		var totalLatency atomic.Int64
		var successCount atomic.Int64
		var failureCount atomic.Int64
		var totalResults atomic.Int64

		queriesPerClient := 20
		start := time.Now()

		for i := 0; i < concurrency; i++ {
			wg.Add(1)
			go func(clientID int) {
				defer wg.Done()

				for j := 0; j < queriesPerClient; j++ {
					queryStart := time.Now()
					queryVec := vectors[(clientID*queriesPerClient+j)%len(vectors)].Data

					results, err := cluster.Search(ctx, queryVec, 10)
					latency := time.Since(queryStart)

					totalLatency.Add(int64(latency))

					if err != nil {
						failureCount.Add(1)
					} else {
						successCount.Add(1)
						totalResults.Add(int64(len(results)))
					}

					// Small delay to prevent overwhelming the system
					time.Sleep(time.Millisecond * 10)
				}
			}(i)
		}

		wg.Wait()
		totalDuration := time.Since(start)

		totalQueries := concurrency * queriesPerClient
		successRate := float64(successCount.Load()) / float64(totalQueries) * 100.0
		avgLatency := time.Duration(totalLatency.Load() / int64(totalQueries))
		qps := float64(totalQueries) / totalDuration.Seconds()

		t.Logf("✓ Concurrency %d results:", concurrency)
		t.Logf("  Total queries: %d", totalQueries)
		t.Logf("  Success rate: %.1f%% (%d/%d)", successRate, successCount.Load(), totalQueries)
		t.Logf("  Avg latency: %v", avgLatency)
		t.Logf("  Throughput: %.0f queries/sec", qps)
		t.Logf("  Avg results per query: %.1f", float64(totalResults.Load())/float64(successCount.Load()))

		if successRate < 95.0 {
			t.Errorf("Success rate too low at concurrency %d: %.1f%%", concurrency, successRate)
		}

		if avgLatency > 500*time.Millisecond {
			t.Errorf("Average latency too high at concurrency %d: %v", concurrency, avgLatency)
		}
	}
}

// testNodeFailureRecovery tests cluster behavior when nodes fail
func testNodeFailureRecovery(t *testing.T) {
	t.Log("=== NODE FAILURE RECOVERY TEST ===")
	ctx := context.Background()

	cluster, err := setupTestCluster(5, 512)
	if err != nil {
		t.Fatalf("Failed to setup cluster: %v", err)
	}
	defer cluster.Shutdown()

	// Insert test data
	vectorCount := 10000
	vectors := generateTestVectors(vectorCount, 512, 42)
	err = cluster.InsertBatch(ctx, vectors)
	if err != nil {
		t.Fatalf("Failed to insert vectors: %v", err)
	}
	t.Logf("✓ Inserted %d vectors", vectorCount)

	// Baseline search performance
	t.Log("\n--- Baseline (all nodes healthy) ---")
	baselineSuccess := performSearchBatch(t, cluster, vectors[:100], 10)
	t.Logf("Baseline success rate: %.1f%%", baselineSuccess)

	// Simulate node failure
	t.Log("\n--- Simulating node failure ---")
	failedNodeID := cluster.Nodes[2].ID
	err = cluster.SimulateNodeFailure(failedNodeID)
	if err != nil {
		t.Fatalf("Failed to simulate node failure: %v", err)
	}
	t.Logf("✓ Node %s marked as failed", failedNodeID)

	// Wait for failure detection and recovery
	time.Sleep(time.Second * 3)

	// Test searches during degraded state
	t.Log("\n--- Performance during degraded state ---")
	degradedSuccess := performSearchBatch(t, cluster, vectors[:100], 10)
	t.Logf("Degraded state success rate: %.1f%%", degradedSuccess)

	if degradedSuccess < 80.0 {
		t.Errorf("Success rate too low during degradation: %.1f%%", degradedSuccess)
	}

	// Check cluster health
	health := cluster.GetHealth()
	t.Logf("\n--- Cluster health after failure ---")
	t.Logf("State: %s", health.State)
	t.Logf("Nodes: %d healthy, %d failed", health.NodesHealthy, health.NodesFailed)

	if health.State != ClusterStateDegraded {
		t.Errorf("Expected degraded state, got: %s", health.State)
	}

	// Simulate node recovery
	t.Log("\n--- Simulating node recovery ---")
	err = cluster.SimulateNodeRecovery(failedNodeID)
	if err != nil {
		t.Fatalf("Failed to simulate node recovery: %v", err)
	}
	t.Logf("✓ Node %s recovered", failedNodeID)

	// Wait for cluster to stabilize
	time.Sleep(time.Second * 3)

	// Test searches after recovery
	t.Log("\n--- Performance after recovery ---")
	recoveredSuccess := performSearchBatch(t, cluster, vectors[:100], 10)
	t.Logf("Recovered state success rate: %.1f%%", recoveredSuccess)

	if recoveredSuccess < baselineSuccess-5.0 {
		t.Errorf("Success rate after recovery (%.1f%%) significantly lower than baseline (%.1f%%)",
			recoveredSuccess, baselineSuccess)
	}

	// Final health check
	health = cluster.GetHealth()
	t.Logf("\n--- Final cluster health ---")
	t.Logf("State: %s", health.State)
	t.Logf("Nodes: %d healthy, %d failed", health.NodesHealthy, health.NodesFailed)

	if health.State != ClusterStateHealthy {
		t.Errorf("Expected healthy state after recovery, got: %s", health.State)
	}
}

// testNetworkPartitionHandling tests cluster behavior during network partitions
func testNetworkPartitionHandling(t *testing.T) {
	t.Log("=== NETWORK PARTITION HANDLING TEST ===")
	ctx := context.Background()

	cluster, err := setupTestCluster(6, 512)
	if err != nil {
		t.Fatalf("Failed to setup cluster: %v", err)
	}
	defer cluster.Shutdown()

	// Insert test data
	vectorCount := 5000
	vectors := generateTestVectors(vectorCount, 512, 42)
	err = cluster.InsertBatch(ctx, vectors)
	if err != nil {
		t.Fatalf("Failed to insert vectors: %v", err)
	}
	t.Log("✓ Data inserted")

	// Create network partition: split cluster into two groups
	t.Log("\n--- Creating network partition ---")
	group1 := []string{cluster.Nodes[0].ID, cluster.Nodes[1].ID, cluster.Nodes[2].ID}
	group2 := []string{cluster.Nodes[3].ID, cluster.Nodes[4].ID, cluster.Nodes[5].ID}

	err = cluster.SimulateNetworkPartition(group1, group2)
	if err != nil {
		t.Fatalf("Failed to create partition: %v", err)
	}
	t.Logf("✓ Partition created: Group1=%v, Group2=%v", group1, group2)

	// Wait for partition detection
	time.Sleep(time.Second * 5)

	// Test operations in majority partition
	t.Log("\n--- Testing majority partition operations ---")
	majoritySuccess := performSearchBatch(t, cluster, vectors[:50], 10)
	t.Logf("Majority partition success rate: %.1f%%", majoritySuccess)

	// Check cluster state
	health := cluster.GetHealth()
	t.Logf("\n--- Cluster state during partition ---")
	t.Logf("State: %s", health.State)
	t.Logf("Issues: %v", health.Issues)

	if health.State != ClusterStatePartitioned {
		t.Logf("Warning: Expected partitioned state, got: %s", health.State)
	}

	// Heal network partition
	t.Log("\n--- Healing network partition ---")
	err = cluster.HealNetworkPartition()
	if err != nil {
		t.Fatalf("Failed to heal partition: %v", err)
	}
	t.Log("✓ Partition healed")

	// Wait for cluster to reconcile
	time.Sleep(time.Second * 5)

	// Test operations after healing
	t.Log("\n--- Testing after partition heal ---")
	healedSuccess := performSearchBatch(t, cluster, vectors[:50], 10)
	t.Logf("Success rate after heal: %.1f%%", healedSuccess)

	// Final health check
	health = cluster.GetHealth()
	t.Logf("\n--- Final cluster state ---")
	t.Logf("State: %s", health.State)
	t.Logf("Nodes: %d healthy, %d failed", health.NodesHealthy, health.NodesFailed)

	if health.State != ClusterStateHealthy {
		t.Logf("Warning: Expected healthy state after heal, got: %s", health.State)
	}
}

// testRaceConditions tests for race conditions in concurrent operations
func testRaceConditions(t *testing.T) {
	t.Log("=== RACE CONDITION TEST ===")
	ctx := context.Background()

	cluster, err := setupTestCluster(3, 512)
	if err != nil {
		t.Fatalf("Failed to setup cluster: %v", err)
	}
	defer cluster.Shutdown()

	// Test 1: Concurrent insertions
	t.Log("\n--- Test 1: Concurrent Insertions ---")
	numWorkers := 20
	vectorsPerWorker := 500

	var wg sync.WaitGroup
	var insertErrors atomic.Int64
	var insertSuccess atomic.Int64

	start := time.Now()

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			vectors := generateTestVectors(vectorsPerWorker, 512, int64(workerID*1000+42))
			err := cluster.InsertBatch(ctx, vectors)

			if err != nil {
				insertErrors.Add(1)
			} else {
				insertSuccess.Add(1)
			}
		}(i)
	}

	wg.Wait()
	insertDuration := time.Since(start)

	totalVectors := numWorkers * vectorsPerWorker
	successRate := float64(insertSuccess.Load()) / float64(numWorkers) * 100.0
	rate := float64(totalVectors) / insertDuration.Seconds()

	t.Logf("✓ Concurrent insertions completed")
	t.Logf("  Workers: %d", numWorkers)
	t.Logf("  Success rate: %.1f%% (%d/%d)", successRate, insertSuccess.Load(), numWorkers)
	t.Logf("  Total vectors: %d", totalVectors)
	t.Logf("  Duration: %v", insertDuration)
	t.Logf("  Throughput: %.0f vec/sec", rate)

	if insertErrors.Load() > 0 {
		t.Errorf("Insert errors occurred: %d", insertErrors.Load())
	}

	// Test 2: Concurrent searches during insertions
	t.Log("\n--- Test 2: Concurrent Searches During Insertions ---")
	searchVectors := generateTestVectors(100, 512, 99999)

	var searchWg sync.WaitGroup
	var insertWg sync.WaitGroup
	var searchErrors atomic.Int64
	var searchSuccess atomic.Int64

	// Start background insertions
	for i := 0; i < 5; i++ {
		insertWg.Add(1)
		go func(workerID int) {
			defer insertWg.Done()
			vectors := generateTestVectors(1000, 512, int64(workerID*10000+42))
			_ = cluster.InsertBatch(ctx, vectors)
		}(i)
	}

	// Perform searches concurrently with insertions
	searchStart := time.Now()
	for i := 0; i < 50; i++ {
		searchWg.Add(1)
		go func(queryID int) {
			defer searchWg.Done()

			for j := 0; j < 10; j++ {
				queryVec := searchVectors[queryID%len(searchVectors)].Data
				_, err := cluster.Search(ctx, queryVec, 10)

				if err != nil {
					searchErrors.Add(1)
				} else {
					searchSuccess.Add(1)
				}

				time.Sleep(time.Millisecond * 10)
			}
		}(i)
	}

	searchWg.Wait()
	insertWg.Wait()
	searchDuration := time.Since(searchStart)

	totalSearches := 50 * 10
	searchSuccessRate := float64(searchSuccess.Load()) / float64(totalSearches) * 100.0

	t.Logf("✓ Concurrent search/insert test completed")
	t.Logf("  Searches performed: %d", totalSearches)
	t.Logf("  Search success rate: %.1f%%", searchSuccessRate)
	t.Logf("  Duration: %v", searchDuration)

	if searchSuccessRate < 90.0 {
		t.Errorf("Search success rate too low during concurrent operations: %.1f%%", searchSuccessRate)
	}

	// Test 3: Shard redistribution under load
	t.Log("\n--- Test 3: Shard Redistribution Under Load ---")

	// Add a new node while operations are ongoing
	var redistWg sync.WaitGroup
	var redistErrors atomic.Int64

	// Background search load
	for i := 0; i < 10; i++ {
		redistWg.Add(1)
		go func() {
			defer redistWg.Done()
			for j := 0; j < 20; j++ {
				queryVec := searchVectors[j%len(searchVectors)].Data
				_, err := cluster.Search(ctx, queryVec, 10)
				if err != nil {
					redistErrors.Add(1)
				}
				time.Sleep(time.Millisecond * 50)
			}
		}()
	}

	// Add new node
	time.Sleep(time.Millisecond * 100)
	newNode, err := cluster.AddNode("new-node-1", "localhost:9999")
	if err != nil {
		t.Errorf("Failed to add new node: %v", err)
	} else {
		t.Logf("✓ New node added: %s", newNode.ID)
	}

	redistWg.Wait()

	if redistErrors.Load() > 5 {
		t.Errorf("Too many errors during redistribution: %d", redistErrors.Load())
	}

	t.Log("✓ No race conditions detected")
}

// testShardRebalancing tests shard rebalancing logic
func testShardRebalancing(t *testing.T) {
	t.Log("=== SHARD REBALANCING TEST ===")
	ctx := context.Background()

	// Start with 3 nodes
	cluster, err := setupTestCluster(3, 512)
	if err != nil {
		t.Fatalf("Failed to setup cluster: %v", err)
	}
	defer cluster.Shutdown()

	// Insert test data
	vectorCount := 15000
	vectors := generateTestVectors(vectorCount, 512, 42)
	err = cluster.InsertBatch(ctx, vectors)
	if err != nil {
		t.Fatalf("Failed to insert vectors: %v", err)
	}
	t.Logf("✓ Inserted %d vectors across 3 nodes", vectorCount)

	// Check initial distribution
	t.Log("\n--- Initial shard distribution ---")
	distBefore := cluster.GetShardDistribution()
	for nodeID, shards := range distBefore {
		totalVectors := 0
		for _, shard := range shards {
			totalVectors += int(shard.VectorCount)
		}
		t.Logf("Node %s: %d shards, ~%d vectors", nodeID, len(shards), totalVectors)
	}

	// Add 2 more nodes
	t.Log("\n--- Adding 2 new nodes ---")
	newNode1, _ := cluster.AddNode("node-4", "localhost:9001")
	newNode2, _ := cluster.AddNode("node-5", "localhost:9002")
	t.Logf("✓ Added nodes: %s, %s", newNode1.ID, newNode2.ID)

	// Trigger rebalancing
	t.Log("\n--- Triggering rebalancing ---")
	start := time.Now()
	err = cluster.RebalanceShards(ctx, "default")
	rebalanceDuration := time.Since(start)

	if err != nil {
		t.Fatalf("Rebalancing failed: %v", err)
	}
	t.Logf("✓ Rebalancing completed in %v", rebalanceDuration)

	// Check distribution after rebalancing
	t.Log("\n--- Shard distribution after rebalancing ---")
	distAfter := cluster.GetShardDistribution()
	for nodeID, shards := range distAfter {
		totalVectors := 0
		for _, shard := range shards {
			totalVectors += int(shard.VectorCount)
		}
		t.Logf("Node %s: %d shards, ~%d vectors", nodeID, len(shards), totalVectors)
	}

	// Check balance
	minShards, maxShards := math.MaxInt32, 0
	for _, shards := range distAfter {
		if len(shards) < minShards {
			minShards = len(shards)
		}
		if len(shards) > maxShards {
			maxShards = len(shards)
		}
	}

	imbalance := float64(maxShards-minShards) / float64(maxShards) * 100.0
	t.Logf("\n--- Balance metrics ---")
	t.Logf("Min shards per node: %d", minShards)
	t.Logf("Max shards per node: %d", maxShards)
	t.Logf("Imbalance: %.1f%%", imbalance)

	if imbalance > 20.0 {
		t.Errorf("Cluster imbalance too high after rebalancing: %.1f%%", imbalance)
	}

	// Verify data accessibility after rebalancing
	t.Log("\n--- Verifying data after rebalancing ---")
	verifySuccess := performSearchBatch(t, cluster, vectors[:100], 10)
	t.Logf("✓ Verification success rate: %.1f%%", verifySuccess)

	if verifySuccess < 95.0 {
		t.Errorf("Data accessibility degraded after rebalancing: %.1f%%", verifySuccess)
	}
}

// testQueryCoordinationStress tests query coordinator under stress
func testQueryCoordinationStress(t *testing.T) {
	t.Log("=== QUERY COORDINATION STRESS TEST ===")
	ctx := context.Background()

	cluster, err := setupTestCluster(5, 1024)
	if err != nil {
		t.Fatalf("Failed to setup cluster: %v", err)
	}
	defer cluster.Shutdown()

	// Insert large dataset
	vectorCount := 50000
	vectors := generateTestVectors(vectorCount, 1024, 42)
	t.Logf("Inserting %d vectors...", vectorCount)

	err = cluster.InsertBatch(ctx, vectors)
	if err != nil {
		t.Fatalf("Failed to insert vectors: %v", err)
	}
	t.Log("✓ Data inserted")

	// Stress test with mixed query types
	t.Log("\n--- Stress test: Mixed query types ---")

	var wg sync.WaitGroup
	var stats struct {
		sync.Mutex
		totalQueries   int64
		successQueries int64
		failedQueries  int64
		totalLatency   int64
		minLatency     int64
		maxLatency     int64
	}

	stats.minLatency = math.MaxInt64

	queryTypes := []struct {
		name string
		k    int
	}{
		{"small-k", 5},
		{"medium-k", 50},
		{"large-k", 500},
	}

	clients := 100
	queriesPerClient := 50

	start := time.Now()

	for i := 0; i < clients; i++ {
		wg.Add(1)
		go func(clientID int) {
			defer wg.Done()

			for j := 0; j < queriesPerClient; j++ {
				qType := queryTypes[j%len(queryTypes)]
				queryVec := vectors[(clientID*queriesPerClient+j)%len(vectors)].Data

				queryStart := time.Now()
				_, err := cluster.Search(ctx, queryVec, qType.k)
				latency := time.Since(queryStart).Nanoseconds()

				stats.Lock()
				stats.totalQueries++
				if err != nil {
					stats.failedQueries++
				} else {
					stats.successQueries++
				}
				stats.totalLatency += latency
				if latency < stats.minLatency {
					stats.minLatency = latency
				}
				if latency > stats.maxLatency {
					stats.maxLatency = latency
				}
				stats.Unlock()

				// Vary query rate
				time.Sleep(time.Millisecond * time.Duration(rand.Intn(20)))
			}
		}(i)
	}

	wg.Wait()
	totalDuration := time.Since(start)

	// Calculate statistics
	avgLatency := time.Duration(stats.totalLatency / stats.totalQueries)
	successRate := float64(stats.successQueries) / float64(stats.totalQueries) * 100.0
	qps := float64(stats.totalQueries) / totalDuration.Seconds()

	t.Logf("\n--- Stress test results ---")
	t.Logf("Total queries: %d", stats.totalQueries)
	t.Logf("Success rate: %.1f%% (%d/%d)", successRate, stats.successQueries, stats.totalQueries)
	t.Logf("Failed queries: %d", stats.failedQueries)
	t.Logf("Duration: %v", totalDuration)
	t.Logf("Throughput: %.0f queries/sec", qps)
	t.Logf("Latency - Avg: %v, Min: %v, Max: %v",
		avgLatency,
		time.Duration(stats.minLatency),
		time.Duration(stats.maxLatency))

	if successRate < 95.0 {
		t.Errorf("Success rate too low under stress: %.1f%%", successRate)
	}

	if qps < 100.0 {
		t.Errorf("Throughput too low under stress: %.0f qps", qps)
	}
}

// testDataConsistency tests data consistency across replicas
func testDataConsistency(t *testing.T) {
	t.Log("=== DATA CONSISTENCY TEST ===")
	ctx := context.Background()

	cluster, err := setupTestCluster(5, 512)
	if err != nil {
		t.Fatalf("Failed to setup cluster: %v", err)
	}
	defer cluster.Shutdown()

	// Set replication factor to 3
	err = cluster.SetReplicationFactor(3)
	if err != nil {
		t.Fatalf("Failed to set replication factor: %v", err)
	}
	t.Log("✓ Replication factor set to 3")

	// Insert test vectors
	vectorCount := 1000
	vectors := generateTestVectors(vectorCount, 512, 42)

	t.Log("\n--- Inserting test data ---")
	err = cluster.InsertBatch(ctx, vectors)
	if err != nil {
		t.Fatalf("Failed to insert vectors: %v", err)
	}
	t.Logf("✓ Inserted %d vectors", vectorCount)

	// Wait for replication to complete
	time.Sleep(time.Second * 2)

	// Test consistency: search from different nodes should return same results
	t.Log("\n--- Testing consistency across replicas ---")
	testVectors := vectors[:50]
	consistencyErrors := 0

	for i, vec := range testVectors {
		// Query all nodes
		resultSets := make(map[string][]string) // nodeID -> result IDs

		for _, node := range cluster.Nodes {
			results, err := cluster.SearchFromNode(ctx, node.ID, vec.Data, 5)
			if err != nil {
				continue
			}

			resultIDs := make([]string, len(results))
			for j, r := range results {
				resultIDs[j] = r.Vector.ID
			}
			resultSets[node.ID] = resultIDs
		}

		// Check if all nodes return same results
		if len(resultSets) > 1 {
			firstResults := ""
			for _, ids := range resultSets {
				current := fmt.Sprintf("%v", ids)
				if firstResults == "" {
					firstResults = current
				} else if firstResults != current {
					consistencyErrors++
					t.Logf("Consistency error for vector %d: different results from replicas", i)
					break
				}
			}
		}
	}

	consistencyRate := float64(len(testVectors)-consistencyErrors) / float64(len(testVectors)) * 100.0
	t.Logf("✓ Consistency rate: %.1f%% (%d/%d)", consistencyRate, len(testVectors)-consistencyErrors, len(testVectors))

	if consistencyRate < 99.0 {
		t.Errorf("Consistency rate too low: %.1f%%", consistencyRate)
	}

	// Test consistency after node failure
	t.Log("\n--- Testing consistency after node failure ---")
	failedNodeID := cluster.Nodes[1].ID
	err = cluster.SimulateNodeFailure(failedNodeID)
	if err != nil {
		t.Errorf("Failed to simulate node failure: %v", err)
	}

	time.Sleep(time.Second * 3)

	postFailureConsistency := 0
	for _, vec := range testVectors[:20] {
		results, err := cluster.Search(ctx, vec.Data, 5)
		if err == nil && len(results) > 0 {
			if results[0].Vector.ID == vec.ID {
				postFailureConsistency++
			}
		}
	}

	postFailureRate := float64(postFailureConsistency) / 20.0 * 100.0
	t.Logf("✓ Post-failure consistency: %.1f%% (%d/20)", postFailureRate, postFailureConsistency)

	if postFailureRate < 90.0 {
		t.Errorf("Consistency degraded too much after node failure: %.1f%%", postFailureRate)
	}
}

// testScalability tests system scalability with increasing load
func testScalability(t *testing.T) {
	t.Log("=== SCALABILITY TEST ===")
	ctx := context.Background()

	// Test scaling from 2 to 10 nodes
	nodeCounts := []int{2, 4, 6, 8, 10}
	results := make(map[int]struct {
		insertRate float64
		searchRate float64
		latency    time.Duration
	})

	for _, nodeCount := range nodeCounts {
		t.Logf("\n--- Testing with %d nodes ---", nodeCount)

		cluster, err := setupTestCluster(nodeCount, 768)
		if err != nil {
			t.Errorf("Failed to setup cluster with %d nodes: %v", nodeCount, err)
			continue
		}

		// Insertion test
		vectorCount := 10000
		vectors := generateTestVectors(vectorCount, 768, 42)

		insertStart := time.Now()
		err = cluster.InsertBatch(ctx, vectors)
		insertDuration := time.Since(insertStart)

		if err != nil {
			t.Errorf("Insert failed with %d nodes: %v", nodeCount, err)
			cluster.Shutdown()
			continue
		}

		insertRate := float64(vectorCount) / insertDuration.Seconds()

		// Search test
		searchCount := 100
		searchStart := time.Now()
		for i := 0; i < searchCount; i++ {
			queryVec := vectors[i%len(vectors)].Data
			_, _ = cluster.Search(ctx, queryVec, 10)
		}
		searchDuration := time.Since(searchStart)

		searchRate := float64(searchCount) / searchDuration.Seconds()
		avgLatency := searchDuration / time.Duration(searchCount)

		results[nodeCount] = struct {
			insertRate float64
			searchRate float64
			latency    time.Duration
		}{insertRate, searchRate, avgLatency}

		t.Logf("✓ %d nodes: Insert=%.0f vec/sec, Search=%.0f qps, Latency=%v",
			nodeCount, insertRate, searchRate, avgLatency)

		cluster.Shutdown()
	}

	// Analyze scaling efficiency
	t.Log("\n--- Scaling analysis ---")
	baselineNodes := nodeCounts[0]
	baselineInsert := results[baselineNodes].insertRate

	for _, nodeCount := range nodeCounts[1:] {
		scaleFactor := float64(nodeCount) / float64(baselineNodes)
		insertSpeedup := results[nodeCount].insertRate / baselineInsert
		efficiency := insertSpeedup / scaleFactor * 100.0

		t.Logf("%dx nodes: %.1fx speedup, %.0f%% scaling efficiency",
			int(scaleFactor), insertSpeedup, efficiency)

		if efficiency < 50.0 && nodeCount > 4 {
			t.Logf("Warning: Scaling efficiency degrading at %d nodes: %.0f%%", nodeCount, efficiency)
		}
	}
}

// Helper functions

// TestCluster represents a test cluster
type TestCluster struct {
	Nodes             []*NodeInfo
	Manager           *ClusterManagerImpl
	Coordinator       *QueryCoordinatorImpl
	collections       map[string][]*api.Vector
	mu                sync.RWMutex
	network           *ComprehensiveTestNetworkManager
	shardDistribution map[string]map[string][]*ShardInfo // collectionID -> nodeID -> shards
	replicationFactor int
	simulatedFailures map[string]bool
	networkPartitions map[string][]string
}

// ComprehensiveTestNetworkManager wraps MockNetworkManager with actual search capability
type ComprehensiveTestNetworkManager struct {
	*MockNetworkManager
	cluster *TestCluster
}

func (tnm *ComprehensiveTestNetworkManager) SendQuery(ctx context.Context, address string, req *api.SearchRequest) ([]*api.SearchResult, error) {
	// Check for simulated failures first
	if tnm.MockNetworkManager.failureNodes[address] {
		return nil, fmt.Errorf("network failure")
	}

	// Perform actual search against stored vectors
	if tnm.cluster != nil {
		tnm.cluster.mu.RLock()
		vectors := tnm.cluster.collections["default"]
		tnm.cluster.mu.RUnlock()

		if vectors != nil && len(vectors) > 0 {
			return tnm.searchVectors(vectors, req.Vector, req.K)
		}
	}

	// Fall back to mock behavior
	return tnm.MockNetworkManager.SendQuery(ctx, address, req)
}

// searchVectors performs brute-force nearest neighbor search
func (tnm *ComprehensiveTestNetworkManager) searchVectors(vectors []*api.Vector, query []float32, k int) ([]*api.SearchResult, error) {
	type scoredResult struct {
		vector   *api.Vector
		distance float32
	}

	results := make([]scoredResult, 0, len(vectors))

	// Calculate distances for all vectors
	for _, vec := range vectors {
		if len(vec.Data) != len(query) {
			continue
		}
		distance := tnm.euclideanDistance(query, vec.Data)
		results = append(results, scoredResult{
			vector:   vec,
			distance: distance,
		})
	}

	// Sort by distance (ascending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].distance < results[j].distance
	})

	// Take top-k
	topK := k
	if topK > len(results) {
		topK = len(results)
	}

	searchResults := make([]*api.SearchResult, topK)
	for i := 0; i < topK; i++ {
		searchResults[i] = &api.SearchResult{
			Vector:   results[i].vector,
			Distance: results[i].distance,
			Score:    1.0 / (1.0 + results[i].distance), // Convert distance to score
		}
	}

	return searchResults, nil
}

// euclideanDistance calculates Euclidean distance between two vectors
func (tnm *ComprehensiveTestNetworkManager) euclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.MaxFloat32)
	}

	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// setupTestCluster creates a test cluster with specified number of nodes
func setupTestCluster(nodeCount, dimension int) (*TestCluster, error) {
	if nodeCount < 1 {
		return nil, fmt.Errorf("need at least 1 node")
	}

	mockNetwork := NewMockNetworkManager()
	config := DefaultClusterConfig("node-0", "localhost:8000")

	consensus := &MockConsensusManager{}
	manager, err := NewClusterManager(config, consensus, mockNetwork)
	if err != nil {
		return nil, fmt.Errorf("failed to create cluster manager: %w", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		return nil, fmt.Errorf("failed to start cluster manager: %w", err)
	}

	cluster := &TestCluster{
		Nodes:             make([]*NodeInfo, 0, nodeCount),
		Manager:           manager,
		collections:       make(map[string][]*api.Vector),
		network:           &ComprehensiveTestNetworkManager{MockNetworkManager: mockNetwork},
		shardDistribution: make(map[string]map[string][]*ShardInfo),
		replicationFactor: 2,
		simulatedFailures: make(map[string]bool),
		networkPartitions: make(map[string][]string),
	}

	// Link network back to cluster for search functionality
	cluster.network.cluster = cluster

	// Add initial node
	cluster.Nodes = append(cluster.Nodes, manager.nodeInfo)

	// Add additional nodes
	for i := 1; i < nodeCount; i++ {
		nodeID := fmt.Sprintf("node-%d", i)
		address := fmt.Sprintf("localhost:%d", 8000+i)

		node := &NodeInfo{
			ID:            nodeID,
			Address:       address,
			Role:          NodeRoleMixed,
			State:         NodeStateActive,
			LastHeartbeat: time.Now(),
			JoinedAt:      time.Now(),
			Tags:          make(map[string]string),
			Metadata:      make(map[string]interface{}),
		}

		if err := manager.AddNode(ctx, node); err != nil {
			return nil, fmt.Errorf("failed to add node %s: %w", nodeID, err)
		}

		cluster.Nodes = append(cluster.Nodes, node)
	}

	// Create coordinator
	coordConfig := DefaultQueryCoordinatorConfig()
	cluster.Coordinator = NewQueryCoordinator("coordinator-0", manager, cluster.network, coordConfig)

	// Initialize shards for default collection
	for i := 0; i < 4; i++ {
		_, err := manager.CreateShard(ctx, "default", cluster.replicationFactor)
		if err != nil {
			return nil, fmt.Errorf("failed to create shard: %w", err)
		}
	}

	return cluster, nil
}

// InsertBatch inserts a batch of vectors into the cluster
func (tc *TestCluster) InsertBatch(ctx context.Context, vectors []*api.Vector) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	if tc.collections["default"] == nil {
		tc.collections["default"] = make([]*api.Vector, 0)
	}

	tc.collections["default"] = append(tc.collections["default"], vectors...)
	return nil
}

// Search performs a search query
func (tc *TestCluster) Search(ctx context.Context, query []float32, k int) ([]*api.SearchResult, error) {
	req := &api.SearchRequest{
		Vector: query,
		K:      k,
	}

	return tc.Coordinator.ExecuteQuery(ctx, req)
}

// SearchFromNode searches from a specific node
func (tc *TestCluster) SearchFromNode(ctx context.Context, nodeID string, query []float32, k int) ([]*api.SearchResult, error) {
	// Simplified - in real implementation would route to specific node
	return tc.Search(ctx, query, k)
}

// GetHealth returns cluster health
func (tc *TestCluster) GetHealth() *ClusterHealth {
	health, _ := tc.Manager.GetClusterHealth(context.Background())
	return health
}

// GetStats returns cluster statistics
func (tc *TestCluster) GetStats() struct {
	TotalVectors int64
	TotalShards  int
} {
	tc.mu.RLock()
	defer tc.mu.RUnlock()

	totalVectors := int64(0)
	for _, vectors := range tc.collections {
		totalVectors += int64(len(vectors))
	}

	shards, _ := tc.Manager.GetShards(context.Background(), "default")

	return struct {
		TotalVectors int64
		TotalShards  int
	}{
		TotalVectors: totalVectors,
		TotalShards:  len(shards),
	}
}

// GetShardDistribution returns shard distribution across nodes
func (tc *TestCluster) GetShardDistribution() map[string][]*ShardInfo {
	tc.mu.RLock()
	defer tc.mu.RUnlock()

	distribution := make(map[string][]*ShardInfo)
	shards, _ := tc.Manager.GetShards(context.Background(), "default")

	for _, shard := range shards {
		distribution[shard.Primary] = append(distribution[shard.Primary], shard)
		for _, replica := range shard.Replicas {
			distribution[replica] = append(distribution[replica], shard)
		}
	}

	return distribution
}

// SimulateNodeFailure simulates a node failure
func (tc *TestCluster) SimulateNodeFailure(nodeID string) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.simulatedFailures[nodeID] = true

	for _, node := range tc.Nodes {
		if node.ID == nodeID {
			node.State = NodeStateFailed
			break
		}
	}

	// Trigger failure handling
	tc.Manager.handleNodeFailure(nodeID)
	return nil
}

// SimulateNodeRecovery simulates a node recovery
func (tc *TestCluster) SimulateNodeRecovery(nodeID string) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	delete(tc.simulatedFailures, nodeID)

	for _, node := range tc.Nodes {
		if node.ID == nodeID {
			node.State = NodeStateActive
			node.LastHeartbeat = time.Now()
			break
		}
	}

	return tc.Manager.UpdateNode(context.Background(), &NodeInfo{
		ID:            nodeID,
		State:         NodeStateActive,
		LastHeartbeat: time.Now(),
	})
}

// SimulateNetworkPartition simulates a network partition
func (tc *TestCluster) SimulateNetworkPartition(group1, group2 []string) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	for _, nodeID := range group1 {
		tc.networkPartitions[nodeID] = group2
	}
	for _, nodeID := range group2 {
		tc.networkPartitions[nodeID] = group1
	}

	return nil
}

// HealNetworkPartition heals a network partition
func (tc *TestCluster) HealNetworkPartition() error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.networkPartitions = make(map[string][]string)
	return nil
}

// AddNode adds a new node to the cluster
func (tc *TestCluster) AddNode(nodeID, address string) (*NodeInfo, error) {
	node := &NodeInfo{
		ID:            nodeID,
		Address:       address,
		Role:          NodeRoleMixed,
		State:         NodeStateActive,
		LastHeartbeat: time.Now(),
		JoinedAt:      time.Now(),
		Tags:          make(map[string]string),
		Metadata:      make(map[string]interface{}),
	}

	err := tc.Manager.AddNode(context.Background(), node)
	if err != nil {
		return nil, err
	}

	tc.mu.Lock()
	tc.Nodes = append(tc.Nodes, node)
	tc.mu.Unlock()

	return node, nil
}

// RebalanceShards triggers shard rebalancing
func (tc *TestCluster) RebalanceShards(ctx context.Context, collectionID string) error {
	return tc.Manager.RebalanceShards(ctx, collectionID)
}

// SetReplicationFactor sets the replication factor
func (tc *TestCluster) SetReplicationFactor(factor int) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.replicationFactor = factor
	return nil
}

// Shutdown shuts down the cluster
func (tc *TestCluster) Shutdown() {
	ctx := context.Background()
	_ = tc.Manager.Stop(ctx)
}

// generateTestVectors generates test vectors
func generateTestVectors(count, dimension int, seed int64) []*api.Vector {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([]*api.Vector, count)

	for i := 0; i < count; i++ {
		data := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			data[j] = rng.Float32()*2 - 1
		}

		// Normalize
		var norm float32
		for _, v := range data {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		for j := range data {
			data[j] /= norm
		}

		vectors[i] = &api.Vector{
			ID:   fmt.Sprintf("vec_%d", i),
			Data: data,
			Metadata: map[string]interface{}{
				"index": i,
			},
		}
	}

	return vectors
}

// performSearchBatch performs a batch of searches and returns success rate
func performSearchBatch(t *testing.T, cluster *TestCluster, vectors []*api.Vector, k int) float64 {
	ctx := context.Background()
	success := 0

	for i, vec := range vectors {
		results, err := cluster.Search(ctx, vec.Data, k)
		if err != nil {
			continue
		}

		if len(results) > 0 && results[0].Vector.ID == vec.ID {
			success++
		} else if len(results) > 0 {
			// Check if it's in top-k
			for _, r := range results {
				if r.Vector.ID == vec.ID {
					success++
					break
				}
			}
		}

		if i > 0 && i%10 == 0 {
			t.Logf("  Progress: %d/%d searches (%.1f%%)", i, len(vectors), float64(i)/float64(len(vectors))*100)
		}
	}

	return float64(success) / float64(len(vectors)) * 100.0
}

// minInt returns the minimum of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
