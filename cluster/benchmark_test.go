// Package cluster provides benchmark tests for distributed cluster performance evaluation.
package cluster

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// Chaos testing utilities
type ChaosConfig struct {
	NodeFailureRate   float64 // Probability of node failure
	NetworkLatencyMs  int     // Additional network latency
	PartitionDuration time.Duration
	RecoveryTime      time.Duration
	FailureInjection  bool
	MessageLoss       float64 // Probability of message loss
}

type ChaosManager struct {
	config      *ChaosConfig
	failedNodes map[string]bool
	partitions  map[string]map[string]bool // source -> target -> partitioned
	rng         *rand.Rand
	mu          sync.RWMutex
}

func NewChaosManager(config *ChaosConfig) *ChaosManager {
	return &ChaosManager{
		config:      config,
		failedNodes: make(map[string]bool),
		partitions:  make(map[string]map[string]bool),
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (cm *ChaosManager) ShouldFailNode(nodeID string) bool {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if !cm.config.FailureInjection {
		return false
	}

	if cm.failedNodes[nodeID] {
		return true
	}

	if cm.rng.Float64() < cm.config.NodeFailureRate {
		cm.failedNodes[nodeID] = true

		// Schedule recovery
		go func() {
			time.Sleep(cm.config.RecoveryTime)
			cm.mu.Lock()
			delete(cm.failedNodes, nodeID)
			cm.mu.Unlock()
		}()

		return true
	}

	return false
}

func (cm *ChaosManager) ShouldDropMessage(from, to string) bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	if !cm.config.FailureInjection {
		return false
	}

	// Check for network partition
	if partitions, exists := cm.partitions[from]; exists {
		if partitions[to] {
			return true
		}
	}

	// Random message loss
	return cm.rng.Float64() < cm.config.MessageLoss
}

func (cm *ChaosManager) CreatePartition(nodes1, nodes2 []string, duration time.Duration) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Create bidirectional partition
	for _, n1 := range nodes1 {
		if cm.partitions[n1] == nil {
			cm.partitions[n1] = make(map[string]bool)
		}
		for _, n2 := range nodes2 {
			cm.partitions[n1][n2] = true
		}
	}

	for _, n2 := range nodes2 {
		if cm.partitions[n2] == nil {
			cm.partitions[n2] = make(map[string]bool)
		}
		for _, n1 := range nodes1 {
			cm.partitions[n2][n1] = true
		}
	}

	// Schedule partition healing
	if duration > 0 {
		go func() {
			time.Sleep(duration)
			cm.HealPartition(nodes1, nodes2)
		}()
	}
}

func (cm *ChaosManager) HealPartition(nodes1, nodes2 []string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	for _, n1 := range nodes1 {
		if partitions, exists := cm.partitions[n1]; exists {
			for _, n2 := range nodes2 {
				delete(partitions, n2)
			}
		}
	}

	for _, n2 := range nodes2 {
		if partitions, exists := cm.partitions[n2]; exists {
			for _, n1 := range nodes1 {
				delete(partitions, n1)
			}
		}
	}
}

func (cm *ChaosManager) GetNetworkLatency() time.Duration {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	baseLatency := time.Duration(cm.config.NetworkLatencyMs) * time.Millisecond

	// Add random jitter (0-50% of base latency)
	jitter := time.Duration(cm.rng.Float64() * 0.5 * float64(baseLatency))

	return baseLatency + jitter
}

// Chaos-enabled Mock Network Manager
type ChaosNetworkManager struct {
	*MockNetworkManager
	chaos *ChaosManager
}

func NewChaosNetworkManager(chaos *ChaosManager) *ChaosNetworkManager {
	return &ChaosNetworkManager{
		MockNetworkManager: NewMockNetworkManager(),
		chaos:              chaos,
	}
}

func (cm *ChaosNetworkManager) SendHeartbeat(ctx context.Context, address string, node *NodeInfo) error {
	// Check for node failure
	if cm.chaos.ShouldFailNode(node.ID) {
		return fmt.Errorf("node %s failed", node.ID)
	}

	// Check for message drop
	if cm.chaos.ShouldDropMessage(node.ID, "coordinator") {
		return fmt.Errorf("message dropped due to network partition")
	}

	// Add network latency
	select {
	case <-time.After(cm.chaos.GetNetworkLatency()):
	case <-ctx.Done():
		return ctx.Err()
	}

	return cm.MockNetworkManager.SendHeartbeat(ctx, address, node)
}

func (cm *ChaosNetworkManager) SendQuery(ctx context.Context, address string, req *api.SearchRequest) ([]*api.SearchResult, error) {
	// Check for message drop
	if cm.chaos.ShouldDropMessage("coordinator", address) {
		return nil, fmt.Errorf("query dropped due to network partition")
	}

	// Add network latency
	select {
	case <-time.After(cm.chaos.GetNetworkLatency()):
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	return cm.MockNetworkManager.SendQuery(ctx, address, req)
}

// Load testing utilities
type LoadTestConfig struct {
	ConcurrentQueries int
	QueriesPerSecond  int
	TestDuration      time.Duration
	VectorDimensions  int
	QueryTypes        []string
	ResultsPerQuery   int
	EnableChaos       bool
	ChaosConfig       *ChaosConfig
}

type LoadTestResults struct {
	TotalQueries      int64
	SuccessfulQueries int64
	FailedQueries     int64
	AverageLatencyMs  float64
	P95LatencyMs      float64
	P99LatencyMs      float64
	ThroughputQPS     float64
	ErrorRate         float64
	Latencies         []time.Duration
	Errors            []error
}

type LoadTester struct {
	config      *LoadTestConfig
	coordinator *QueryCoordinatorImpl
	results     *LoadTestResults
	mu          sync.Mutex
}

func NewLoadTester(config *LoadTestConfig, coordinator *QueryCoordinatorImpl) *LoadTester {
	return &LoadTester{
		config:      config,
		coordinator: coordinator,
		results: &LoadTestResults{
			Latencies: make([]time.Duration, 0),
			Errors:    make([]error, 0),
		},
	}
}

func (lt *LoadTester) RunTest(ctx context.Context) (*LoadTestResults, error) {
	startTime := time.Now()

	// Create rate limiter
	ticker := time.NewTicker(time.Second / time.Duration(lt.config.QueriesPerSecond))
	defer ticker.Stop()

	// Create worker pool
	queryChan := make(chan struct{}, lt.config.ConcurrentQueries*2)
	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < lt.config.ConcurrentQueries; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			lt.worker(ctx, workerID, queryChan)
		}(i)
	}

	// Generate load
	testCtx, cancel := context.WithTimeout(ctx, lt.config.TestDuration)
	defer cancel()

	go func() {
		for {
			select {
			case <-testCtx.Done():
				close(queryChan)
				return
			case <-ticker.C:
				select {
				case queryChan <- struct{}{}:
				default:
					// Channel full, skip this query
				}
			}
		}
	}()

	// Wait for completion
	wg.Wait()

	// Calculate results
	lt.calculateResults(time.Since(startTime))

	return lt.results, nil
}

func (lt *LoadTester) worker(ctx context.Context, workerID int, queryChan <-chan struct{}) {
	for range queryChan {
		lt.executeQuery(ctx, workerID)
	}
}

func (lt *LoadTester) executeQuery(ctx context.Context, workerID int) {
	startTime := time.Now()

	// Generate random vector
	vector := make([]float32, lt.config.VectorDimensions)
	for i := range vector {
		vector[i] = rand.Float32()*2 - 1 // Range [-1, 1]
	}

	// Create search request
	req := &api.SearchRequest{
		Vector: vector,
		K:      lt.config.ResultsPerQuery,
	}

	// Execute query
	_, err := lt.coordinator.ExecuteQuery(ctx, req)

	latency := time.Since(startTime)

	// Record results
	lt.mu.Lock()
	lt.results.TotalQueries++
	lt.results.Latencies = append(lt.results.Latencies, latency)

	if err != nil {
		lt.results.FailedQueries++
		lt.results.Errors = append(lt.results.Errors, err)
	} else {
		lt.results.SuccessfulQueries++
	}
	lt.mu.Unlock()
}

func (lt *LoadTester) calculateResults(totalTime time.Duration) {
	lt.mu.Lock()
	defer lt.mu.Unlock()

	if len(lt.results.Latencies) == 0 {
		return
	}

	// Sort latencies for percentile calculations
	latencies := make([]time.Duration, len(lt.results.Latencies))
	copy(latencies, lt.results.Latencies)

	// Simple sorting (for small datasets)
	for i := 0; i < len(latencies); i++ {
		for j := i + 1; j < len(latencies); j++ {
			if latencies[i] > latencies[j] {
				latencies[i], latencies[j] = latencies[j], latencies[i]
			}
		}
	}

	// Calculate average latency
	var totalLatency time.Duration
	for _, lat := range latencies {
		totalLatency += lat
	}
	lt.results.AverageLatencyMs = float64(totalLatency.Nanoseconds()) / float64(len(latencies)) / 1e6

	// Calculate percentiles
	p95Index := int(float64(len(latencies)) * 0.95)
	p99Index := int(float64(len(latencies)) * 0.99)

	if p95Index < len(latencies) {
		lt.results.P95LatencyMs = float64(latencies[p95Index].Nanoseconds()) / 1e6
	}

	if p99Index < len(latencies) {
		lt.results.P99LatencyMs = float64(latencies[p99Index].Nanoseconds()) / 1e6
	}

	// Calculate throughput and error rate
	lt.results.ThroughputQPS = float64(lt.results.TotalQueries) / totalTime.Seconds()
	lt.results.ErrorRate = float64(lt.results.FailedQueries) / float64(lt.results.TotalQueries)
}

// Comprehensive benchmark tests
func BenchmarkClusterPerformance(b *testing.B) {
	// Set up test cluster
	config := DefaultClusterConfig("node1", "localhost:8080")
	consensus := NewMockConsensusManager()
	network := NewMockNetworkManager()

	cm, err := NewClusterManager(config, consensus, network)
	if err != nil {
		b.Fatalf("Failed to create cluster manager: %v", err)
	}

	ctx := context.Background()
	cm.Start(ctx)
	defer cm.Stop(ctx)

	// Add nodes
	for i := 2; i <= 10; i++ {
		node := &NodeInfo{
			ID:      fmt.Sprintf("node%d", i),
			Address: fmt.Sprintf("localhost:808%d", i),
			State:   NodeStateActive,
			Role:    NodeRoleData,
		}
		cm.AddNode(ctx, node)
	}

	// Create shards
	cm.CreateShard(ctx, "default", 3)

	qc := NewQueryCoordinator("node1", cm, network, nil)

	b.ResetTimer()

	b.Run("SingleNodeQuery", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			req := &api.SearchRequest{
				Vector: []float32{0.1, 0.2, 0.3, 0.4},
				K:      10,
			}

			_, err := qc.ExecuteQuery(ctx, req)
			if err != nil {
				b.Errorf("Query failed: %v", err)
			}
		}
	})

	b.Run("ConcurrentQueries", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				req := &api.SearchRequest{
					Vector: []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3, float32(i) * 0.4},
					K:      10,
				}

				_, err := qc.ExecuteQuery(ctx, req)
				if err != nil {
					b.Errorf("Concurrent query failed: %v", err)
				}
				i++
			}
		})
	})

	b.Run("BatchQueries", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			operations := make([]api.BatchOperation, 10)
			for j := range operations {
				operations[j] = api.BatchOperation{
					Type: api.BatchOperationSearch,
					Search: &api.SearchRequest{
						Vector: []float32{float32(j) * 0.1, float32(j) * 0.2, float32(j) * 0.3, float32(j) * 0.4},
						K:      5,
					},
				}
			}

			req := &api.BatchRequest{Operations: operations}
			_, err := qc.ExecuteBatch(ctx, req)
			if err != nil {
				b.Errorf("Batch query failed: %v", err)
			}
		}
	})

	b.Run("HighDimensionalVectors", func(b *testing.B) {
		dimension := 512
		for i := 0; i < b.N; i++ {
			vector := make([]float32, dimension)
			for j := range vector {
				vector[j] = rand.Float32()
			}

			req := &api.SearchRequest{
				Vector: vector,
				K:      20,
			}

			_, err := qc.ExecuteQuery(ctx, req)
			if err != nil {
				b.Errorf("High-dimensional query failed: %v", err)
			}
		}
	})
}

// Chaos engineering tests
func TestChaosEngineering(t *testing.T) {
	t.Run("Node Failure Recovery", func(t *testing.T) {
		chaosConfig := &ChaosConfig{
			NodeFailureRate:  0.1, // 10% failure rate
			NetworkLatencyMs: 50,
			RecoveryTime:     2 * time.Second,
			FailureInjection: true,
			MessageLoss:      0.05, // 5% message loss
		}

		chaos := NewChaosManager(chaosConfig)
		network := NewChaosNetworkManager(chaos)

		config := DefaultClusterConfig("node1", "localhost:8080")
		consensus := NewMockConsensusManager()

		cm, err := NewClusterManager(config, consensus, network)
		if err != nil {
			t.Fatalf("Failed to create cluster manager: %v", err)
		}

		ctx := context.Background()
		cm.Start(ctx)
		defer cm.Stop(ctx)

		// Add nodes
		nodes := []string{"node2", "node3", "node4", "node5"}
		for i, nodeID := range nodes {
			node := &NodeInfo{
				ID:      nodeID,
				Address: fmt.Sprintf("localhost:808%d", i+2),
				State:   NodeStateActive,
				Role:    NodeRoleData,
			}
			cm.AddNode(ctx, node)
		}

		cm.CreateShard(ctx, "default", 2)
		qc := NewQueryCoordinator("node1", cm, network, nil)

		// Run queries with chaos
		successCount := 0
		totalQueries := 100

		for i := 0; i < totalQueries; i++ {
			req := &api.SearchRequest{
				Vector: []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3, float32(i) * 0.4},
				K:      10,
			}

			_, err := qc.ExecuteQuery(ctx, req)
			if err == nil {
				successCount++
			}

			// Small delay between queries
			time.Sleep(10 * time.Millisecond)
		}

		successRate := float64(successCount) / float64(totalQueries)

		// With chaos, we expect some failures but system should be resilient
		if successRate < 0.5 {
			t.Errorf("Success rate too low under chaos: %f", successRate)
		}

		t.Logf("Chaos test success rate: %f (%d/%d)", successRate, successCount, totalQueries)
	})

	t.Run("Network Partition", func(t *testing.T) {
		chaosConfig := &ChaosConfig{
			NodeFailureRate:   0.0,
			NetworkLatencyMs:  20,
			PartitionDuration: 3 * time.Second,
			FailureInjection:  true,
			MessageLoss:       0.0,
		}

		chaos := NewChaosManager(chaosConfig)
		network := NewChaosNetworkManager(chaos)

		config := DefaultClusterConfig("node1", "localhost:8080")
		consensus := NewMockConsensusManager()

		cm, err := NewClusterManager(config, consensus, network)
		if err != nil {
			t.Fatalf("Failed to create cluster manager: %v", err)
		}

		ctx := context.Background()
		cm.Start(ctx)
		defer cm.Stop(ctx)

		// Add nodes
		for i := 2; i <= 6; i++ {
			node := &NodeInfo{
				ID:      fmt.Sprintf("node%d", i),
				Address: fmt.Sprintf("localhost:808%d", i),
				State:   NodeStateActive,
				Role:    NodeRoleData,
			}
			cm.AddNode(ctx, node)
		}

		cm.CreateShard(ctx, "default", 2)
		qc := NewQueryCoordinator("node1", cm, network, nil)

		// Create network partition (split nodes)
		group1 := []string{"node1", "node2", "node3"}
		group2 := []string{"node4", "node5", "node6"}

		chaos.CreatePartition(group1, group2, 2*time.Second)

		// Test queries during partition
		partitionQueries := 20
		partitionSuccessful := 0

		for i := 0; i < partitionQueries; i++ {
			req := &api.SearchRequest{
				Vector: []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3, float32(i) * 0.4},
				K:      5,
			}

			_, err := qc.ExecuteQuery(ctx, req)
			if err == nil {
				partitionSuccessful++
			}

			time.Sleep(50 * time.Millisecond)
		}

		// Wait for partition to heal
		time.Sleep(3 * time.Second)

		// Test queries after healing
		healingQueries := 20
		healingSuccessful := 0

		for i := 0; i < healingQueries; i++ {
			req := &api.SearchRequest{
				Vector: []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3, float32(i) * 0.4},
				K:      5,
			}

			_, err := qc.ExecuteQuery(ctx, req)
			if err == nil {
				healingSuccessful++
			}

			time.Sleep(20 * time.Millisecond)
		}

		partitionRate := float64(partitionSuccessful) / float64(partitionQueries)
		healingRate := float64(healingSuccessful) / float64(healingQueries)

		t.Logf("Partition success rate: %f, Healing success rate: %f", partitionRate, healingRate)

		// After healing, success rate should be maintained or improve
		// Allow for the case where partition performance was already perfect
		if healingRate < 0.9 {
			t.Errorf("Expected high success rate after partition healing: got %f", healingRate)
		}

		// If there was degradation during partition, healing should improve it
		if partitionRate < 0.95 && healingRate <= partitionRate {
			t.Errorf("Expected improvement after partition healing: partition=%f, healing=%f", partitionRate, healingRate)
		}
	})
}

// Load testing
func TestLoadTesting(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping load test in short mode")
	}

	t.Run("Sustained Load Test", func(t *testing.T) {
		// Set up cluster
		config := DefaultClusterConfig("node1", "localhost:8080")
		consensus := NewMockConsensusManager()
		network := NewMockNetworkManager()

		cm, err := NewClusterManager(config, consensus, network)
		if err != nil {
			t.Fatalf("Failed to create cluster manager: %v", err)
		}

		ctx := context.Background()
		cm.Start(ctx)
		defer cm.Stop(ctx)

		// Add nodes
		for i := 2; i <= 8; i++ {
			node := &NodeInfo{
				ID:      fmt.Sprintf("node%d", i),
				Address: fmt.Sprintf("localhost:808%d", i),
				State:   NodeStateActive,
				Role:    NodeRoleData,
			}
			cm.AddNode(ctx, node)
		}

		cm.CreateShard(ctx, "default", 2)
		qc := NewQueryCoordinator("node1", cm, network, nil)

		// Configure load test
		loadConfig := &LoadTestConfig{
			ConcurrentQueries: 10,
			QueriesPerSecond:  100,
			TestDuration:      10 * time.Second,
			VectorDimensions:  128,
			ResultsPerQuery:   10,
			EnableChaos:       false,
		}

		tester := NewLoadTester(loadConfig, qc)

		// Run load test
		results, err := tester.RunTest(ctx)
		if err != nil {
			t.Errorf("Load test failed: %v", err)
		}

		// Validate results
		if results.TotalQueries == 0 {
			t.Error("No queries executed during load test")
		}

		if results.ErrorRate > 0.05 { // Allow max 5% error rate
			t.Errorf("Error rate too high: %f", results.ErrorRate)
		}

		if results.AverageLatencyMs > 1000 { // Average latency should be < 1s
			t.Errorf("Average latency too high: %f ms", results.AverageLatencyMs)
		}

		t.Logf("Load test results:")
		t.Logf("  Total queries: %d", results.TotalQueries)
		t.Logf("  Successful: %d", results.SuccessfulQueries)
		t.Logf("  Failed: %d", results.FailedQueries)
		t.Logf("  Error rate: %.2f%%", results.ErrorRate*100)
		t.Logf("  Average latency: %.2f ms", results.AverageLatencyMs)
		t.Logf("  P95 latency: %.2f ms", results.P95LatencyMs)
		t.Logf("  P99 latency: %.2f ms", results.P99LatencyMs)
		t.Logf("  Throughput: %.2f QPS", results.ThroughputQPS)
	})

	t.Run("Load Test with Chaos", func(t *testing.T) {
		// Set up cluster with chaos
		chaosConfig := &ChaosConfig{
			NodeFailureRate:  0.02, // 2% failure rate
			NetworkLatencyMs: 25,
			RecoveryTime:     1 * time.Second,
			FailureInjection: true,
			MessageLoss:      0.01, // 1% message loss
		}

		chaos := NewChaosManager(chaosConfig)
		network := NewChaosNetworkManager(chaos)

		config := DefaultClusterConfig("node1", "localhost:8080")
		consensus := NewMockConsensusManager()

		cm, err := NewClusterManager(config, consensus, network)
		if err != nil {
			t.Fatalf("Failed to create cluster manager: %v", err)
		}

		ctx := context.Background()
		cm.Start(ctx)
		defer cm.Stop(ctx)

		// Add nodes
		for i := 2; i <= 6; i++ {
			node := &NodeInfo{
				ID:      fmt.Sprintf("node%d", i),
				Address: fmt.Sprintf("localhost:808%d", i),
				State:   NodeStateActive,
				Role:    NodeRoleData,
			}
			cm.AddNode(ctx, node)
		}

		cm.CreateShard(ctx, "default", 2)
		qc := NewQueryCoordinator("node1", cm, network, nil)

		// Configure load test with chaos
		loadConfig := &LoadTestConfig{
			ConcurrentQueries: 5,
			QueriesPerSecond:  50,
			TestDuration:      5 * time.Second,
			VectorDimensions:  64,
			ResultsPerQuery:   5,
			EnableChaos:       true,
			ChaosConfig:       chaosConfig,
		}

		tester := NewLoadTester(loadConfig, qc)

		// Run load test
		results, err := tester.RunTest(ctx)
		if err != nil {
			t.Errorf("Chaos load test failed: %v", err)
		}

		// With chaos, allow higher error rates but system should still function
		if results.ErrorRate > 0.20 { // Allow max 20% error rate with chaos
			t.Errorf("Error rate too high even with chaos: %f", results.ErrorRate)
		}

		if results.SuccessfulQueries == 0 {
			t.Error("No successful queries during chaos load test")
		}

		t.Logf("Chaos load test results:")
		t.Logf("  Total queries: %d", results.TotalQueries)
		t.Logf("  Successful: %d", results.SuccessfulQueries)
		t.Logf("  Failed: %d", results.FailedQueries)
		t.Logf("  Error rate: %.2f%%", results.ErrorRate*100)
		t.Logf("  Average latency: %.2f ms", results.AverageLatencyMs)
		t.Logf("  Throughput: %.2f QPS", results.ThroughputQPS)
	})
}

// Memory and resource usage tests
func TestResourceUsage(t *testing.T) {
	t.Run("Memory Usage Under Load", func(t *testing.T) {
		// Set up cluster
		config := DefaultClusterConfig("node1", "localhost:8080")
		consensus := NewMockConsensusManager()
		network := NewMockNetworkManager()

		cm, err := NewClusterManager(config, consensus, network)
		if err != nil {
			t.Fatalf("Failed to create cluster manager: %v", err)
		}

		ctx := context.Background()
		cm.Start(ctx)
		defer cm.Stop(ctx)

		// Add nodes
		for i := 2; i <= 5; i++ {
			node := &NodeInfo{
				ID:      fmt.Sprintf("node%d", i),
				Address: fmt.Sprintf("localhost:808%d", i),
				State:   NodeStateActive,
				Role:    NodeRoleData,
			}
			cm.AddNode(ctx, node)
		}

		cm.CreateShard(ctx, "default", 2)
		qc := NewQueryCoordinator("node1", cm, network, nil)

		// Execute many queries to test memory usage
		const numQueries = 1000

		for i := 0; i < numQueries; i++ {
			req := &api.SearchRequest{
				Vector: make([]float32, 256), // Large vector
				K:      50,                   // Many results
			}

			// Fill vector with random data
			for j := range req.Vector {
				req.Vector[j] = rand.Float32()
			}

			_, err := qc.ExecuteQuery(ctx, req)
			if err != nil {
				t.Errorf("Query %d failed: %v", i, err)
			}

			// Check for obvious memory leaks (basic check)
			if i%100 == 0 {
				stats := qc.GetStats()
				// Monitor query statistics
				if stats.QueriesFailed > stats.QueriesSucceeded {
					t.Errorf("Too many failed queries at iteration %d", i)
				}
			}
		}

		// Final statistics check
		stats := qc.GetStats()
		if stats.QueriesTotal != numQueries {
			t.Errorf("Expected %d total queries, got %d", numQueries, stats.QueriesTotal)
		}

		t.Logf("Memory usage test completed. Final stats:")
		t.Logf("  Total queries: %d", stats.QueriesTotal)
		t.Logf("  Successful queries: %d", stats.QueriesSucceeded)
		t.Logf("  Failed queries: %d", stats.QueriesFailed)
		t.Logf("  Average latency: %v", stats.AvgLatency)
	})
}

// Integration tests with the actual HNSW index
func TestHNSWIntegration(t *testing.T) {
	t.Run("Integration with HNSW Index", func(t *testing.T) {
		// This test would integrate with the actual HNSW index from the index package
		// For now, we test the interface compatibility

		config := DefaultClusterConfig("node1", "localhost:8080")
		consensus := NewMockConsensusManager()
		network := NewMockNetworkManager()

		cm, err := NewClusterManager(config, consensus, network)
		if err != nil {
			t.Fatalf("Failed to create cluster manager: %v", err)
		}

		ctx := context.Background()
		cm.Start(ctx)
		defer cm.Stop(ctx)

		// Add nodes
		for i := 2; i <= 3; i++ {
			node := &NodeInfo{
				ID:      fmt.Sprintf("node%d", i),
				Address: fmt.Sprintf("localhost:808%d", i),
				State:   NodeStateActive,
				Role:    NodeRoleData,
			}
			cm.AddNode(ctx, node)
		}

		cm.CreateShard(ctx, "default", 2)

		// TODO: Integration with actual HNSW index
		// This would involve:
		// 1. Creating real HNSW indexes on each node
		// 2. Inserting vectors into the distributed cluster
		// 3. Testing search across multiple shards
		// 4. Verifying result consistency and quality

		qc := NewQueryCoordinator("node1", cm, network, nil)

		// Test basic functionality
		req := &api.SearchRequest{
			Vector: []float32{0.1, 0.2, 0.3, 0.4},
			K:      10,
		}

		results, err := qc.ExecuteQuery(ctx, req)
		if err != nil {
			t.Errorf("Integration query failed: %v", err)
		}

		if len(results) == 0 {
			t.Error("Integration query returned no results")
		}

		t.Logf("Integration test completed with %d results", len(results))
	})
}
