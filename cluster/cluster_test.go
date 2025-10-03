package cluster

// Package cluster provides unit tests for the distributed cluster components.

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// MockNetworkManager implements NetworkManager for testing
type MockNetworkManager struct {
	nodes        map[string]*NodeInfo
	heartbeats   []HeartbeatRecord
	joinRequests []JoinRecord
	queryResults map[string][]*api.SearchResult
	failureNodes map[string]bool
	latencyMs    int
}

type HeartbeatRecord struct {
	Address string
	Node    *NodeInfo
	Time    time.Time
}

type JoinRecord struct {
	Address string
	Node    *NodeInfo
	Time    time.Time
}

func NewMockNetworkManager() *MockNetworkManager {
	return &MockNetworkManager{
		nodes:        make(map[string]*NodeInfo),
		heartbeats:   make([]HeartbeatRecord, 0),
		joinRequests: make([]JoinRecord, 0),
		queryResults: make(map[string][]*api.SearchResult),
		failureNodes: make(map[string]bool),
		latencyMs:    10,
	}
}

func (m *MockNetworkManager) Start(ctx context.Context) error {
	return nil
}

func (m *MockNetworkManager) Stop(ctx context.Context) error {
	return nil
}

func (m *MockNetworkManager) SendHeartbeat(ctx context.Context, address string, node *NodeInfo) error {
	if m.failureNodes[address] {
		return fmt.Errorf("network failure")
	}

	m.heartbeats = append(m.heartbeats, HeartbeatRecord{
		Address: address,
		Node:    node,
		Time:    time.Now(),
	})

	// Simulate network latency
	time.Sleep(time.Duration(m.latencyMs) * time.Millisecond)
	return nil
}

func (m *MockNetworkManager) JoinRequest(ctx context.Context, address string, node *NodeInfo) error {
	if m.failureNodes[address] {
		return fmt.Errorf("network failure")
	}

	m.joinRequests = append(m.joinRequests, JoinRecord{
		Address: address,
		Node:    node,
		Time:    time.Now(),
	})

	return nil
}

func (m *MockNetworkManager) NodeLeaveNotification(ctx context.Context, address string, nodeID string) error {
	return nil
}

func (m *MockNetworkManager) CreateShardRequest(ctx context.Context, address string, shard *ShardInfo) error {
	return nil
}

func (m *MockNetworkManager) HealthCheck(ctx context.Context) error {
	return nil
}

func (m *MockNetworkManager) SendQuery(ctx context.Context, address string, req *api.SearchRequest) ([]*api.SearchResult, error) {
	if m.failureNodes[address] {
		return nil, fmt.Errorf("network failure")
	}

	// Return mock results
	if results, exists := m.queryResults[address]; exists {
		return results, nil
	}

	// Generate mock results
	results := make([]*api.SearchResult, req.K)
	for i := 0; i < req.K; i++ {
		results[i] = &api.SearchResult{
			Vector: &api.Vector{
				ID:   fmt.Sprintf("mock-vector-%d", i),
				Data: make([]float32, len(req.Vector)),
			},
			Score:    float32(1.0 - float32(i)*0.1),
			Distance: float32(i) * 0.1,
		}
		copy(results[i].Vector.Data, req.Vector)
	}

	return results, nil
}

func (m *MockNetworkManager) SendBatchQuery(ctx context.Context, address string, req *api.BatchRequest) (*api.BatchResponse, error) {
	if m.failureNodes[address] {
		return nil, fmt.Errorf("network failure")
	}

	response := &api.BatchResponse{
		Results: make([]api.BatchResult, len(req.Operations)),
		Success: true,
	}

	for i := range req.Operations {
		response.Results[i] = api.BatchResult{Success: true}
	}

	return response, nil
}

func (m *MockNetworkManager) SetNodeFailure(address string, shouldFail bool) {
	m.failureNodes[address] = shouldFail
}

func (m *MockNetworkManager) SetQueryResults(address string, results []*api.SearchResult) {
	m.queryResults[address] = results
}

func (m *MockNetworkManager) GetHeartbeatCount() int {
	return len(m.heartbeats)
}

func (m *MockNetworkManager) GetJoinRequestCount() int {
	return len(m.joinRequests)
}

// MockConsensusManager implements ConsensusManager for testing
type MockConsensusManager struct {
	isLeader        bool
	currentTerm     uint64
	proposals       []*Proposal
	votes           []Vote
	state           *ConsensusState
	commandsApplied []*Command
}

func NewMockConsensusManager() *MockConsensusManager {
	return &MockConsensusManager{
		isLeader:        true,
		currentTerm:     1,
		proposals:       make([]*Proposal, 0),
		votes:           make([]Vote, 0),
		commandsApplied: make([]*Command, 0),
		state: &ConsensusState{
			Term:        1,
			Leader:      "mock-leader",
			Commitments: make(map[string]uint64),
			LastApplied: 0,
			Log:         make([]*LogEntry, 0),
		},
	}
}

func (m *MockConsensusManager) Start(ctx context.Context) error {
	return nil
}

func (m *MockConsensusManager) Stop(ctx context.Context) error {
	return nil
}

func (m *MockConsensusManager) HealthCheck(ctx context.Context) error {
	return nil
}

func (m *MockConsensusManager) Propose(ctx context.Context, proposal *Proposal) (*ProposalResult, error) {
	m.proposals = append(m.proposals, proposal)

	return &ProposalResult{
		ProposalID: proposal.ID,
		Accepted:   m.isLeader,
		Votes:      []Vote{},
		AppliedAt:  timePtr(time.Now()),
	}, nil
}

func (m *MockConsensusManager) Vote(ctx context.Context, proposalID string, vote Vote) error {
	m.votes = append(m.votes, vote)
	return nil
}

func (m *MockConsensusManager) IsLeader() bool {
	return m.isLeader
}

func (m *MockConsensusManager) GetLeader() (*NodeInfo, error) {
	if m.isLeader {
		return &NodeInfo{ID: "mock-leader"}, nil
	}
	return &NodeInfo{ID: "other-leader"}, nil
}

func (m *MockConsensusManager) GetState(ctx context.Context) (*ConsensusState, error) {
	return m.state, nil
}

func (m *MockConsensusManager) ApplyCommand(ctx context.Context, command *Command) error {
	m.commandsApplied = append(m.commandsApplied, command)
	return nil
}

func (m *MockConsensusManager) SetLeader(isLeader bool) {
	m.isLeader = isLeader
}

func (m *MockConsensusManager) GetProposalCount() int {
	return len(m.proposals)
}

func (m *MockConsensusManager) GetVoteCount() int {
	return len(m.votes)
}

func timePtr(t time.Time) *time.Time {
	return &t
}

// Test HashRing functionality
func TestHashRing(t *testing.T) {
	t.Run("Basic Hash Ring Operations", func(t *testing.T) {
		ring := NewHashRing(160)

		// Test empty ring
		if ring.NodeCount() != 0 {
			t.Errorf("Expected 0 nodes, got %d", ring.NodeCount())
		}

		// Add nodes
		nodes := []string{"node1", "node2", "node3"}
		for _, node := range nodes {
			ring.AddNode(node)
		}

		if ring.NodeCount() != 3 {
			t.Errorf("Expected 3 nodes, got %d", ring.NodeCount())
		}

		if ring.VirtualNodeCount() != 3*160 {
			t.Errorf("Expected %d virtual nodes, got %d", 3*160, ring.VirtualNodeCount())
		}

		// Test node selection
		node, err := ring.GetNode("test-key")
		if err != nil {
			t.Errorf("Failed to get node: %v", err)
		}

		found := false
		for _, n := range nodes {
			if n == node {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("GetNode returned unknown node: %s", node)
		}

		// Test replication
		replicas, err := ring.GetNodes("test-key", 2)
		if err != nil {
			t.Errorf("Failed to get replica nodes: %v", err)
		}

		if len(replicas) != 2 {
			t.Errorf("Expected 2 replicas, got %d", len(replicas))
		}

		// Remove node
		ring.RemoveNode("node1")
		if ring.NodeCount() != 2 {
			t.Errorf("Expected 2 nodes after removal, got %d", ring.NodeCount())
		}
	})

	t.Run("Hash Ring Load Balance", func(t *testing.T) {
		ring := NewHashRing(160)
		nodes := []string{"node1", "node2", "node3", "node4"}

		for _, node := range nodes {
			ring.AddNode(node)
		}

		// Test load distribution
		distribution := ring.GetLoadDistribution()
		if len(distribution) != 4 {
			t.Errorf("Expected distribution for 4 nodes, got %d", len(distribution))
		}

		// Each node should have roughly 25% of the load
		for node, load := range distribution {
			if load < 0.15 || load > 0.35 {
				t.Errorf("Node %s has unbalanced load: %f", node, load)
			}
		}

		// Test balance check
		if !ring.IsBalanced(0.15) {
			t.Error("Ring should be balanced with 15% threshold")
		}
	})
}

// Test Cluster Manager functionality
func TestClusterManager(t *testing.T) {
	t.Run("Cluster Manager Creation and Startup", func(t *testing.T) {
		config := DefaultClusterConfig("node1", "localhost:8080")
		consensus := NewMockConsensusManager()
		network := NewMockNetworkManager()

		cm, err := NewClusterManager(config, consensus, network)
		if err != nil {
			t.Fatalf("Failed to create cluster manager: %v", err)
		}

		ctx := context.Background()
		err = cm.Start(ctx)
		if err != nil {
			t.Fatalf("Failed to start cluster manager: %v", err)
		}

		// Test health check
		err = cm.HealthCheck(ctx)
		if err != nil {
			t.Errorf("Health check failed: %v", err)
		}

		// Test node retrieval
		nodes, err := cm.GetNodes(ctx)
		if err != nil {
			t.Errorf("Failed to get nodes: %v", err)
		}

		if len(nodes) != 1 {
			t.Errorf("Expected 1 node, got %d", len(nodes))
		}

		if nodes[0].ID != "node1" {
			t.Errorf("Expected node1, got %s", nodes[0].ID)
		}

		// Clean up
		cm.Stop(ctx)
	})

	t.Run("Node Join and Leave", func(t *testing.T) {
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

		// Add new node
		newNode := &NodeInfo{
			ID:      "node2",
			Address: "localhost:8081",
			State:   NodeStateActive,
			Role:    NodeRoleData,
		}

		err = cm.AddNode(ctx, newNode)
		if err != nil {
			t.Errorf("Failed to add node: %v", err)
		}

		// Verify node was added
		nodes, _ := cm.GetNodes(ctx)
		if len(nodes) != 2 {
			t.Errorf("Expected 2 nodes, got %d", len(nodes))
		}

		// Remove node
		err = cm.RemoveNode(ctx, "node2")
		if err != nil {
			t.Errorf("Failed to remove node: %v", err)
		}

		// Verify node was removed
		nodes, _ = cm.GetNodes(ctx)
		if len(nodes) != 1 {
			t.Errorf("Expected 1 node after removal, got %d", len(nodes))
		}
	})

	t.Run("Shard Management", func(t *testing.T) {
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

		// Add data nodes
		for i := 2; i <= 4; i++ {
			node := &NodeInfo{
				ID:      fmt.Sprintf("node%d", i),
				Address: fmt.Sprintf("localhost:808%d", i),
				State:   NodeStateActive,
				Role:    NodeRoleData,
			}
			cm.AddNode(ctx, node)
		}

		// Create shard
		shard, err := cm.CreateShard(ctx, "test-collection", 2)
		if err != nil {
			t.Errorf("Failed to create shard: %v", err)
		}

		if shard.ReplicationFactor != 2 {
			t.Errorf("Expected replication factor 2, got %d", shard.ReplicationFactor)
		}

		if shard.Primary == "" {
			t.Error("Shard primary not assigned")
		}

		if len(shard.Replicas) != 1 {
			t.Errorf("Expected 1 replica, got %d", len(shard.Replicas))
		}

		// Get shards
		shards, err := cm.GetShards(ctx, "test-collection")
		if err != nil {
			t.Errorf("Failed to get shards: %v", err)
		}

		if len(shards) != 1 {
			t.Errorf("Expected 1 shard, got %d", len(shards))
		}
	})
}

// Test Query Coordinator functionality
func TestQueryCoordinator(t *testing.T) {
	t.Run("Query Coordinator Creation", func(t *testing.T) {
		config := DefaultClusterConfig("node1", "localhost:8080")
		consensus := NewMockConsensusManager()
		network := NewMockNetworkManager()

		cm, err := NewClusterManager(config, consensus, network)
		if err != nil {
			t.Fatalf("Failed to create cluster manager: %v", err)
		}

		qc := NewQueryCoordinator("node1", cm, network, nil)
		if qc == nil {
			t.Fatal("Failed to create query coordinator")
		}

		stats := qc.GetStats()
		if stats.QueriesTotal != 0 {
			t.Errorf("Expected 0 total queries, got %d", stats.QueriesTotal)
		}
	})

	t.Run("Query Planning", func(t *testing.T) {
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

		// Add nodes and create shards
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

		qc := NewQueryCoordinator("node1", cm, network, nil)

		// Create search request
		req := &api.SearchRequest{
			Vector: []float32{0.1, 0.2, 0.3, 0.4},
			K:      10,
		}

		// Plan query
		plan, err := qc.PlanQuery(ctx, req)
		if err != nil {
			t.Errorf("Failed to plan query: %v", err)
		}

		if plan.QueryType != QueryTypeSearch {
			t.Errorf("Expected search query type, got %s", plan.QueryType)
		}

		if len(plan.Phases) == 0 {
			t.Error("Expected query phases, got none")
		}

		if plan.Collection != "default" {
			t.Errorf("Expected collection 'default', got %s", plan.Collection)
		}
	})

	t.Run("Query Execution", func(t *testing.T) {
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

		// Add nodes and create shards
		node2 := &NodeInfo{
			ID:      "node2",
			Address: "localhost:8082",
			State:   NodeStateActive,
			Role:    NodeRoleData,
		}
		cm.AddNode(ctx, node2)

		cm.CreateShard(ctx, "default", 1)

		qc := NewQueryCoordinator("node1", cm, network, nil)

		// Set up mock results
		mockResults := []*api.SearchResult{
			{
				Vector: &api.Vector{ID: "vec1", Data: []float32{0.1, 0.2}},
				Score:  0.9,
			},
			{
				Vector: &api.Vector{ID: "vec2", Data: []float32{0.3, 0.4}},
				Score:  0.8,
			},
		}
		network.SetQueryResults("localhost:8082", mockResults)

		// Execute query
		req := &api.SearchRequest{
			Vector: []float32{0.1, 0.2, 0.3, 0.4},
			K:      10,
		}

		results, err := qc.ExecuteQuery(ctx, req)
		if err != nil {
			t.Errorf("Failed to execute query: %v", err)
		}

		if len(results) == 0 {
			t.Error("Expected query results, got none")
		}

		// Check statistics
		stats := qc.GetStats()
		if stats.QueriesTotal != 1 {
			t.Errorf("Expected 1 total query, got %d", stats.QueriesTotal)
		}

		if stats.QueriesSucceeded != 1 {
			t.Errorf("Expected 1 successful query, got %d", stats.QueriesSucceeded)
		}
	})

	t.Run("Batch Query Execution", func(t *testing.T) {
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

		qc := NewQueryCoordinator("node1", cm, network, nil)

		// Add node and create shard for batch operations
		node2 := &NodeInfo{
			ID:      "node2",
			Address: "localhost:8082",
			State:   NodeStateActive,
			Role:    NodeRoleData,
		}
		cm.AddNode(ctx, node2)
		cm.CreateShard(ctx, "default", 1)

		// Create batch request
		batchReq := &api.BatchRequest{
			Operations: []api.Operation{
				{
					Type: api.BatchOperationSearch,
					Search: &api.SearchRequest{
						Vector: []float32{0.1, 0.2},
						K:      5,
					},
				},
				{
					Type: api.BatchOperationGet,
					ID:   "test-vector-1",
				},
			},
		}

		// Execute batch
		response, err := qc.ExecuteBatch(ctx, batchReq)
		if err != nil {
			t.Errorf("Failed to execute batch: %v", err)
		}

		if len(response.Results) != 2 {
			t.Errorf("Expected 2 batch results, got %d", len(response.Results))
		}

		// First operation (search) should succeed
		if !response.Results[0].Success {
			t.Error("Expected first batch operation to succeed")
		}

		// Second operation (get) should fail (not implemented)
		if response.Results[1].Success {
			t.Error("Expected second batch operation to fail (not implemented)")
		}
	})
}

// Test Network Manager functionality
func TestNetworkManager(t *testing.T) {
	t.Run("Mock Network Manager", func(t *testing.T) {
		network := NewMockNetworkManager()

		ctx := context.Background()

		// Test startup
		err := network.Start(ctx)
		if err != nil {
			t.Errorf("Failed to start network manager: %v", err)
		}

		// Test health check
		err = network.HealthCheck(ctx)
		if err != nil {
			t.Errorf("Health check failed: %v", err)
		}

		// Test heartbeat
		node := &NodeInfo{ID: "test-node", Address: "localhost:8080"}
		err = network.SendHeartbeat(ctx, "localhost:8081", node)
		if err != nil {
			t.Errorf("Failed to send heartbeat: %v", err)
		}

		if network.GetHeartbeatCount() != 1 {
			t.Errorf("Expected 1 heartbeat, got %d", network.GetHeartbeatCount())
		}

		// Test join request
		err = network.JoinRequest(ctx, "localhost:8081", node)
		if err != nil {
			t.Errorf("Failed to send join request: %v", err)
		}

		if network.GetJoinRequestCount() != 1 {
			t.Errorf("Expected 1 join request, got %d", network.GetJoinRequestCount())
		}

		// Test query
		searchReq := &api.SearchRequest{
			Vector: []float32{0.1, 0.2, 0.3},
			K:      5,
		}

		results, err := network.SendQuery(ctx, "localhost:8081", searchReq)
		if err != nil {
			t.Errorf("Failed to send query: %v", err)
		}

		if len(results) != 5 {
			t.Errorf("Expected 5 results, got %d", len(results))
		}

		// Test network failure
		network.SetNodeFailure("localhost:8081", true)

		err = network.SendHeartbeat(ctx, "localhost:8081", node)
		if err == nil {
			t.Error("Expected heartbeat to fail for failed node")
		}

		// Test shutdown
		err = network.Stop(ctx)
		if err != nil {
			t.Errorf("Failed to stop network manager: %v", err)
		}
	})
}

// Test Consensus Manager functionality
func TestConsensusManager(t *testing.T) {
	t.Run("Mock Consensus Manager", func(t *testing.T) {
		consensus := NewMockConsensusManager()

		ctx := context.Background()

		// Test startup
		err := consensus.Start(ctx)
		if err != nil {
			t.Errorf("Failed to start consensus manager: %v", err)
		}

		// Test leadership
		if !consensus.IsLeader() {
			t.Error("Expected mock consensus to be leader")
		}

		leader, err := consensus.GetLeader()
		if err != nil {
			t.Errorf("Failed to get leader: %v", err)
		}

		if leader.ID != "mock-leader" {
			t.Errorf("Expected leader 'mock-leader', got %s", leader.ID)
		}

		// Test proposal
		proposal := &Proposal{
			ID:   "test-proposal",
			Type: ProposalTypeClusterChange,
			Data: []byte("test data"),
		}

		result, err := consensus.Propose(ctx, proposal)
		if err != nil {
			t.Errorf("Failed to propose: %v", err)
		}

		if !result.Accepted {
			t.Error("Expected proposal to be accepted")
		}

		if consensus.GetProposalCount() != 1 {
			t.Errorf("Expected 1 proposal, got %d", consensus.GetProposalCount())
		}

		// Test voting
		vote := Vote{
			NodeID:     "test-node",
			ProposalID: "test-proposal",
			Accept:     true,
		}

		err = consensus.Vote(ctx, "test-proposal", vote)
		if err != nil {
			t.Errorf("Failed to vote: %v", err)
		}

		if consensus.GetVoteCount() != 1 {
			t.Errorf("Expected 1 vote, got %d", consensus.GetVoteCount())
		}

		// Test state
		state, err := consensus.GetState(ctx)
		if err != nil {
			t.Errorf("Failed to get state: %v", err)
		}

		if state.Term != 1 {
			t.Errorf("Expected term 1, got %d", state.Term)
		}

		// Test command application
		command := &Command{
			Type: CommandTypeNodeJoin,
			Data: []byte("node join data"),
		}

		err = consensus.ApplyCommand(ctx, command)
		if err != nil {
			t.Errorf("Failed to apply command: %v", err)
		}

		// Test health check
		err = consensus.HealthCheck(ctx)
		if err != nil {
			t.Errorf("Health check failed: %v", err)
		}

		// Test shutdown
		err = consensus.Stop(ctx)
		if err != nil {
			t.Errorf("Failed to stop consensus manager: %v", err)
		}
	})

	t.Run("Consensus Leadership Change", func(t *testing.T) {
		consensus := NewMockConsensusManager()

		ctx := context.Background()

		// Initially leader
		if !consensus.IsLeader() {
			t.Error("Expected to be leader initially")
		}

		// Change leadership
		consensus.SetLeader(false)

		if consensus.IsLeader() {
			t.Error("Expected not to be leader after change")
		}

		// Test proposal as non-leader
		proposal := &Proposal{
			ID:   "test-proposal-2",
			Type: ProposalTypeClusterChange,
			Data: []byte("test data 2"),
		}

		result, err := consensus.Propose(ctx, proposal)
		if err != nil {
			t.Errorf("Failed to propose: %v", err)
		}

		if result.Accepted {
			t.Error("Expected proposal to be rejected when not leader")
		}
	})
}

// Integration test for complete distributed scenario
func TestDistributedIntegration(t *testing.T) {
	t.Run("Multi-Node Cluster Setup", func(t *testing.T) {
		// Create a 3-node cluster
		nodes := []struct {
			id      string
			address string
		}{
			{"node1", "localhost:8080"},
			{"node2", "localhost:8081"},
			{"node3", "localhost:8082"},
		}

		clusters := make([]*ClusterManagerImpl, len(nodes))
		networks := make([]*MockNetworkManager, len(nodes))
		consensuses := make([]*MockConsensusManager, len(nodes))

		ctx := context.Background()

		// Set up cluster managers
		for i, node := range nodes {
			config := DefaultClusterConfig(node.id, node.address)
			consensus := NewMockConsensusManager()
			network := NewMockNetworkManager()

			// Only first node is leader
			consensus.SetLeader(i == 0)

			cm, err := NewClusterManager(config, consensus, network)
			if err != nil {
				t.Fatalf("Failed to create cluster manager for %s: %v", node.id, err)
			}

			clusters[i] = cm
			networks[i] = network
			consensuses[i] = consensus

			err = cm.Start(ctx)
			if err != nil {
				t.Fatalf("Failed to start cluster manager for %s: %v", node.id, err)
			}
		}

		// Clean up
		defer func() {
			for _, cm := range clusters {
				cm.Stop(ctx)
			}
		}()

		// Join nodes to cluster
		for i := 1; i < len(clusters); i++ {
			nodeInfo := &NodeInfo{
				ID:      nodes[i].id,
				Address: nodes[i].address,
				State:   NodeStateActive,
				Role:    NodeRoleData,
			}

			err := clusters[0].AddNode(ctx, nodeInfo)
			if err != nil {
				t.Errorf("Failed to add node %s: %v", nodes[i].id, err)
			}
		}

		// Verify cluster has all nodes
		allNodes, err := clusters[0].GetNodes(ctx)
		if err != nil {
			t.Errorf("Failed to get nodes: %v", err)
		}

		if len(allNodes) != 3 {
			t.Errorf("Expected 3 nodes in cluster, got %d", len(allNodes))
		}

		// Create shards for default collection
		shard, err := clusters[0].CreateShard(ctx, "default", 2)
		if err != nil {
			t.Errorf("Failed to create shard: %v", err)
		}

		if shard.ReplicationFactor != 2 {
			t.Errorf("Expected replication factor 2, got %d", shard.ReplicationFactor)
		}

		// Set up query coordinator on leader node
		qc := NewQueryCoordinator(nodes[0].id, clusters[0], networks[0], nil) // Set up mock results
		for _, network := range networks {
			mockResults := []*api.SearchResult{
				{
					Vector: &api.Vector{ID: "result1", Data: []float32{0.1, 0.2}},
					Score:  0.95,
				},
			}
			network.SetQueryResults("localhost:8080", mockResults)
		}

		// Execute distributed query
		req := &api.SearchRequest{
			Vector: []float32{0.5, 0.6, 0.7, 0.8},
			K:      10,
		}

		results, err := qc.ExecuteQuery(ctx, req)
		if err != nil {
			t.Errorf("Failed to execute distributed query: %v", err)
		}

		if len(results) == 0 {
			t.Error("Expected query results from distributed execution")
		}

		// Verify statistics
		stats := qc.GetStats()
		if stats.QueriesTotal != 1 {
			t.Errorf("Expected 1 total query, got %d", stats.QueriesTotal)
		}

		// Test cluster health
		health, err := clusters[0].GetClusterHealth(ctx)
		if err != nil {
			t.Errorf("Failed to get cluster health: %v", err)
		}

		if health.NodesTotal != 3 {
			t.Errorf("Expected 3 total nodes in health, got %d", health.NodesTotal)
		}

		if health.State != ClusterStateHealthy {
			t.Errorf("Expected healthy cluster state, got %s", health.State)
		}
	})
}

// Benchmark tests
func BenchmarkHashRingOperations(b *testing.B) {
	ring := NewHashRing(160)

	// Add nodes
	for i := 0; i < 100; i++ {
		ring.AddNode(fmt.Sprintf("node%d", i))
	}

	b.ResetTimer()

	b.Run("GetNode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("key%d", i)
			_, err := ring.GetNode(key)
			if err != nil {
				b.Errorf("GetNode failed: %v", err)
			}
		}
	})

	b.Run("GetNodes", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("key%d", i)
			_, err := ring.GetNodes(key, 3)
			if err != nil {
				b.Errorf("GetNodes failed: %v", err)
			}
		}
	})

	b.Run("AddNode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			node := fmt.Sprintf("benchmark_node%d", i)
			ring.AddNode(node)
		}
	})

	b.Run("RemoveNode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			node := fmt.Sprintf("benchmark_node%d", i)
			ring.RemoveNode(node)
		}
	})
}

func BenchmarkQueryCoordinator(b *testing.B) {
	config := DefaultClusterConfig("node1", "localhost:8080")
	consensus := NewMockConsensusManager()
	network := NewMockNetworkManager()

	cm, _ := NewClusterManager(config, consensus, network)
	qc := NewQueryCoordinator("node1", cm, network, nil)

	ctx := context.Background()
	cm.Start(ctx)
	defer cm.Stop(ctx)

	// Add some nodes
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

	b.ResetTimer()

	b.Run("QueryPlanning", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			req := &api.SearchRequest{
				Vector: []float32{0.1, 0.2, 0.3, 0.4},
				K:      10,
			}

			_, err := qc.PlanQuery(ctx, req)
			if err != nil {
				b.Errorf("Query planning failed: %v", err)
			}
		}
	})

	b.Run("QueryExecution", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			req := &api.SearchRequest{
				Vector: []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3, float32(i) * 0.4},
				K:      10,
			}

			_, err := qc.ExecuteQuery(ctx, req)
			if err != nil {
				b.Errorf("Query execution failed: %v", err)
			}
		}
	})
}
