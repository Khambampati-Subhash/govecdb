// Package cluster provides cluster management implementation.
package cluster

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// ClusterManagerImpl implements the ClusterManager interface
type ClusterManagerImpl struct {
	config    *ClusterConfig
	nodeInfo  *NodeInfo
	nodes     map[string]*NodeInfo    // nodeID -> NodeInfo
	shards    map[string][]*ShardInfo // collectionID -> shards
	hashRings *HashRingManager
	consensus ConsensusManager
	network   NetworkManager
	mu        sync.RWMutex

	// State
	state    ClusterState
	isLeader bool
	joinedAt time.Time

	// Background tasks
	heartbeatTicker *time.Ticker
	healthTicker    *time.Ticker
	stopChan        chan struct{}

	// Statistics
	stats *ClusterStats
}

// ClusterStats contains cluster statistics
type ClusterStats struct {
	NodeJoins          int64     `json:"node_joins"`
	NodeLeaves         int64     `json:"node_leaves"`
	NodeFailures       int64     `json:"node_failures"`
	ShardMigrations    int64     `json:"shard_migrations"`
	QueriesProcessed   int64     `json:"queries_processed"`
	LastRebalance      time.Time `json:"last_rebalance"`
	HeartbeatsReceived int64     `json:"heartbeats_received"`
	HeartbeatsSent     int64     `json:"heartbeats_sent"`
}

// NewClusterManager creates a new cluster manager
func NewClusterManager(config *ClusterConfig, consensus ConsensusManager, network NetworkManager) (*ClusterManagerImpl, error) {
	if config == nil {
		return nil, fmt.Errorf("cluster config cannot be nil")
	}

	nodeInfo := &NodeInfo{
		ID:       config.NodeID,
		Address:  config.BindAddress,
		Role:     NodeRoleMixed, // Default to mixed role
		State:    NodeStateStarting,
		Version:  "1.0.0", // Should come from build info
		Region:   "default",
		Rack:     "default",
		JoinedAt: time.Now(),
		Tags:     make(map[string]string),
		Metadata: make(map[string]interface{}),
	}

	cm := &ClusterManagerImpl{
		config:    config,
		nodeInfo:  nodeInfo,
		nodes:     make(map[string]*NodeInfo),
		shards:    make(map[string][]*ShardInfo),
		hashRings: NewHashRingManager(),
		consensus: consensus,
		network:   network,

		state:    ClusterStateInitializing,
		joinedAt: time.Now(),
		stopChan: make(chan struct{}),

		stats: &ClusterStats{},
	}

	// Add self to nodes
	cm.nodes[nodeInfo.ID] = nodeInfo

	return cm, nil
}

// Start starts the cluster manager
func (cm *ClusterManagerImpl) Start(ctx context.Context) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if cm.state != ClusterStateInitializing {
		return fmt.Errorf("cluster manager already started")
	}

	// Start consensus if provided
	if cm.consensus != nil {
		if err := cm.consensus.Start(ctx); err != nil {
			return fmt.Errorf("failed to start consensus: %w", err)
		}
	}

	// Start network layer
	if cm.network != nil {
		if err := cm.network.Start(ctx); err != nil {
			return fmt.Errorf("failed to start network layer: %w", err)
		}
	}

	// Start background tasks
	cm.startBackgroundTasks()

	cm.nodeInfo.State = NodeStateActive
	cm.state = ClusterStateHealthy

	return nil
}

// Stop stops the cluster manager
func (cm *ClusterManagerImpl) Stop(ctx context.Context) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if cm.state == ClusterStateFailed {
		return nil // Already stopped
	}

	cm.nodeInfo.State = NodeStateShuttingDown
	cm.state = ClusterStateFailed

	// Stop background tasks
	cm.stopBackgroundTasks()

	// Stop network layer
	if cm.network != nil {
		if err := cm.network.Stop(ctx); err != nil {
			return fmt.Errorf("failed to stop network layer: %w", err)
		}
	}

	// Stop consensus
	if cm.consensus != nil {
		if err := cm.consensus.Stop(ctx); err != nil {
			return fmt.Errorf("failed to stop consensus: %w", err)
		}
	}

	return nil
}

// JoinCluster joins the cluster using seed nodes
func (cm *ClusterManagerImpl) JoinCluster(ctx context.Context, seedNodes []string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if len(seedNodes) == 0 {
		// Bootstrap mode - this is the first node
		cm.isLeader = true
		cm.state = ClusterStateHealthy
		cm.stats.NodeJoins++
		return nil
	}

	// Try to contact seed nodes
	for _, seedAddr := range seedNodes {
		if err := cm.contactSeedNode(ctx, seedAddr); err != nil {
			continue // Try next seed node
		}

		cm.state = ClusterStateHealthy
		cm.stats.NodeJoins++
		return nil
	}

	return fmt.Errorf("failed to join cluster: could not contact any seed nodes")
}

// LeaveCluster gracefully leaves the cluster
func (cm *ClusterManagerImpl) LeaveCluster(ctx context.Context) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.nodeInfo.State = NodeStateDraining

	// Migrate shards owned by this node
	if err := cm.migrateOwnedShards(ctx); err != nil {
		return fmt.Errorf("failed to migrate shards: %w", err)
	}

	// Notify other nodes
	if err := cm.notifyNodeLeave(ctx); err != nil {
		return fmt.Errorf("failed to notify other nodes: %w", err)
	}

	cm.nodeInfo.State = NodeStateShuttingDown
	cm.stats.NodeLeaves++

	return nil
}

// GetNodes returns all nodes in the cluster
func (cm *ClusterManagerImpl) GetNodes(ctx context.Context) ([]*NodeInfo, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	nodes := make([]*NodeInfo, 0, len(cm.nodes))
	for _, node := range cm.nodes {
		// Return a copy to prevent external modification
		nodeCopy := *node
		nodes = append(nodes, &nodeCopy)
	}

	return nodes, nil
}

// GetNode returns information about a specific node
func (cm *ClusterManagerImpl) GetNode(ctx context.Context, nodeID string) (*NodeInfo, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	node, exists := cm.nodes[nodeID]
	if !exists {
		return nil, fmt.Errorf("node %s not found", nodeID)
	}

	// Return a copy
	nodeCopy := *node
	return &nodeCopy, nil
}

// GetShards returns all shards for a collection
func (cm *ClusterManagerImpl) GetShards(ctx context.Context, collectionID string) ([]*ShardInfo, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	shards, exists := cm.shards[collectionID]
	if !exists {
		return []*ShardInfo{}, nil
	}

	// Return copies
	result := make([]*ShardInfo, len(shards))
	for i, shard := range shards {
		shardCopy := *shard
		result[i] = &shardCopy
	}

	return result, nil
}

// CreateShard creates a new shard for a collection
func (cm *ClusterManagerImpl) CreateShard(ctx context.Context, collectionID string, replicationFactor int) (*ShardInfo, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Get or create hash ring for collection
	ring, err := cm.hashRings.GetRing(collectionID)
	if err != nil {
		// Create new ring with current nodes
		nodes := make([]string, 0, len(cm.nodes))
		for nodeID, node := range cm.nodes {
			if node.State == NodeStateActive && (node.Role == NodeRoleData || node.Role == NodeRoleMixed) {
				nodes = append(nodes, nodeID)
			}
		}

		if len(nodes) == 0 {
			return nil, fmt.Errorf("no data nodes available for shard creation")
		}

		if err := cm.hashRings.CreateRing(collectionID, 160, nodes); err != nil {
			return nil, fmt.Errorf("failed to create hash ring: %w", err)
		}

		ring, _ = cm.hashRings.GetRing(collectionID)
	}

	// Generate shard ID
	shardID := fmt.Sprintf("%s-shard-%d", collectionID, len(cm.shards[collectionID]))

	// Determine hash range for this shard
	totalShards := len(cm.shards[collectionID]) + 1
	hashRangeSize := ^uint64(0) / uint64(totalShards)
	shardIndex := len(cm.shards[collectionID])

	hashRange := HashRange{
		Start: uint64(shardIndex) * hashRangeSize,
		End:   uint64(shardIndex+1) * hashRangeSize,
	}

	// Select nodes for replication
	nodes, err := ring.GetNodes(shardID, replicationFactor)
	if err != nil {
		return nil, fmt.Errorf("failed to select nodes for shard: %w", err)
	}

	if len(nodes) == 0 {
		return nil, fmt.Errorf("no nodes available for shard placement")
	}

	primary := nodes[0]
	replicas := nodes[1:]

	shard := &ShardInfo{
		ID:                shardID,
		CollectionID:      collectionID,
		HashRange:         hashRange,
		Primary:           primary,
		Replicas:          replicas,
		ReplicationFactor: replicationFactor,
		State:             ShardStateCreating,
		Size:              0,
		VectorCount:       0,
		CreatedAt:         time.Now(),
		UpdatedAt:         time.Now(),
		HealthScore:       1.0,
		LastAccess:        time.Now(),
	}

	// Add to cluster state
	cm.shards[collectionID] = append(cm.shards[collectionID], shard)

	// Notify nodes to create the shard
	if err := cm.notifyShardCreation(ctx, shard); err != nil {
		return nil, fmt.Errorf("failed to notify shard creation: %w", err)
	}

	shard.State = ShardStateActive
	shard.UpdatedAt = time.Now()

	return shard, nil
}

// RebalanceShards rebalances shards for a collection
func (cm *ClusterManagerImpl) RebalanceShards(ctx context.Context, collectionID string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	shards, exists := cm.shards[collectionID]
	if !exists || len(shards) == 0 {
		return nil // No shards to rebalance
	}

	// Get current hash ring
	ring, err := cm.hashRings.GetRing(collectionID)
	if err != nil {
		return fmt.Errorf("failed to get hash ring: %w", err)
	}

	// Check if rebalancing is needed
	if ring.IsBalanced(cm.config.RebalanceThreshold) {
		return nil // Already balanced
	}

	// Get active data nodes
	nodes := make([]string, 0)
	for nodeID, node := range cm.nodes {
		if node.State == NodeStateActive && (node.Role == NodeRoleData || node.Role == NodeRoleMixed) {
			nodes = append(nodes, nodeID)
		}
	}

	if len(nodes) == 0 {
		return fmt.Errorf("no active data nodes for rebalancing")
	}

	// Update hash ring with current nodes
	ring.Rebalance(nodes)

	// Reassign shards based on new ring
	for _, shard := range shards {
		newNodes, err := ring.GetNodes(shard.ID, shard.ReplicationFactor)
		if err != nil {
			continue // Skip this shard
		}

		if len(newNodes) > 0 {
			newPrimary := newNodes[0]
			newReplicas := newNodes[1:]

			// Check if primary needs to change
			if shard.Primary != newPrimary {
				shard.State = ShardStateMigrating
				if err := cm.migrateShard(ctx, shard, newPrimary); err != nil {
					continue // Skip this shard
				}
				shard.Primary = newPrimary
			}

			shard.Replicas = newReplicas
			shard.UpdatedAt = time.Now()
			shard.State = ShardStateActive
		}
	}

	cm.stats.LastRebalance = time.Now()
	return nil
}

// HealthCheck performs a cluster health check
func (cm *ClusterManagerImpl) HealthCheck(ctx context.Context) error {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Check if this node is healthy
	if cm.nodeInfo.State != NodeStateActive {
		return fmt.Errorf("node is not active")
	}

	// Check consensus layer
	if cm.consensus != nil {
		if err := cm.consensus.HealthCheck(ctx); err != nil {
			return fmt.Errorf("consensus layer unhealthy: %w", err)
		}
	}

	// Check network layer
	if cm.network != nil {
		if err := cm.network.HealthCheck(ctx); err != nil {
			return fmt.Errorf("network layer unhealthy: %w", err)
		}
	}

	return nil
}

// GetClusterHealth returns overall cluster health
func (cm *ClusterManagerImpl) GetClusterHealth(ctx context.Context) (*ClusterHealth, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	health := &ClusterHealth{
		ClusterID:     cm.config.ClusterID,
		State:         cm.state,
		NodesTotal:    len(cm.nodes),
		NodesHealthy:  0,
		NodesFailed:   0,
		ShardsTotal:   0,
		ShardsHealthy: 0,
		ShardsFailed:  0,
		LoadBalanced:  true,
		LastCheck:     time.Now(),
		Issues:        make([]string, 0),
		NodeHealth:    make(map[string]float64),
	}

	// Count node health
	for nodeID, node := range cm.nodes {
		switch node.State {
		case NodeStateActive:
			health.NodesHealthy++
			health.NodeHealth[nodeID] = 1.0
		case NodeStateFailed:
			health.NodesFailed++
			health.NodeHealth[nodeID] = 0.0
		default:
			health.NodeHealth[nodeID] = 0.5 // Transitional state
		}
	}

	// Count shard health
	for _, shards := range cm.shards {
		for _, shard := range shards {
			health.ShardsTotal++
			if shard.State == ShardStateActive {
				health.ShardsHealthy++
			} else {
				health.ShardsFailed++
			}
		}
	}

	// Check load balance
	for collectionID := range cm.shards {
		if ring, err := cm.hashRings.GetRing(collectionID); err == nil {
			if !ring.IsBalanced(cm.config.RebalanceThreshold) {
				health.LoadBalanced = false
				health.Issues = append(health.Issues, fmt.Sprintf("collection %s is not load balanced", collectionID))
			}
		}
	}

	// Determine overall cluster state
	if health.NodesFailed > 0 || health.ShardsFailed > 0 {
		if health.NodesHealthy == 0 {
			health.State = ClusterStateFailed
		} else {
			health.State = ClusterStateDegraded
		}
	}

	return health, nil
}

// Private methods

func (cm *ClusterManagerImpl) startBackgroundTasks() {
	// Heartbeat task
	cm.heartbeatTicker = time.NewTicker(cm.config.HeartbeatInterval)
	go cm.heartbeatLoop()

	// Health check task
	cm.healthTicker = time.NewTicker(time.Minute) // Check health every minute
	go cm.healthCheckLoop()
}

func (cm *ClusterManagerImpl) stopBackgroundTasks() {
	close(cm.stopChan)

	if cm.heartbeatTicker != nil {
		cm.heartbeatTicker.Stop()
	}

	if cm.healthTicker != nil {
		cm.healthTicker.Stop()
	}
}

func (cm *ClusterManagerImpl) heartbeatLoop() {
	for {
		select {
		case <-cm.stopChan:
			return
		case <-cm.heartbeatTicker.C:
			cm.sendHeartbeats()
		}
	}
}

func (cm *ClusterManagerImpl) healthCheckLoop() {
	for {
		select {
		case <-cm.stopChan:
			return
		case <-cm.healthTicker.C:
			cm.performHealthChecks()
		}
	}
}

func (cm *ClusterManagerImpl) sendHeartbeats() {
	cm.mu.RLock()
	nodes := make([]*NodeInfo, 0, len(cm.nodes))
	for _, node := range cm.nodes {
		if node.ID != cm.nodeInfo.ID && node.State == NodeStateActive {
			nodes = append(nodes, node)
		}
	}
	cm.mu.RUnlock()

	// Send heartbeats to all active nodes
	for _, node := range nodes {
		go func(targetNode *NodeInfo) {
			if cm.network != nil {
				if err := cm.network.SendHeartbeat(context.Background(), targetNode.Address, cm.nodeInfo); err != nil {
					cm.handleNodeFailure(targetNode.ID)
				} else {
					cm.stats.HeartbeatsSent++
				}
			}
		}(node)
	}
}

func (cm *ClusterManagerImpl) performHealthChecks() {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	now := time.Now()
	heartbeatTimeout := cm.config.HeartbeatInterval * 3 // 3x heartbeat interval

	for nodeID, node := range cm.nodes {
		if nodeID == cm.nodeInfo.ID {
			continue // Skip self
		}

		// Check if node has missed heartbeats
		if node.State == NodeStateActive && now.Sub(node.LastHeartbeat) > heartbeatTimeout {
			cm.handleNodeFailureInternal(nodeID)
		}
	}
}

func (cm *ClusterManagerImpl) handleNodeFailure(nodeID string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.handleNodeFailureInternal(nodeID)
}

func (cm *ClusterManagerImpl) handleNodeFailureInternal(nodeID string) {
	node, exists := cm.nodes[nodeID]
	if !exists {
		return
	}

	if node.State == NodeStateFailed {
		return // Already marked as failed
	}

	node.State = NodeStateFailed
	cm.stats.NodeFailures++

	// Remove node from hash rings
	cm.hashRings.RemoveNodeFromAll(nodeID)

	// Reassign shards from failed node
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute*5)
		defer cancel()
		cm.reassignShardsFromFailedNode(ctx, nodeID)
	}()
}

func (cm *ClusterManagerImpl) contactSeedNode(ctx context.Context, seedAddr string) error {
	if cm.network == nil {
		return fmt.Errorf("network layer not initialized")
	}

	return cm.network.JoinRequest(ctx, seedAddr, cm.nodeInfo)
}

func (cm *ClusterManagerImpl) migrateOwnedShards(ctx context.Context) error {
	// Migrate all shards where this node is primary
	for _, shards := range cm.shards {
		for _, shard := range shards {
			if shard.Primary == cm.nodeInfo.ID {
				if len(shard.Replicas) > 0 {
					// Promote first replica to primary
					newPrimary := shard.Replicas[0]
					if err := cm.migrateShard(ctx, shard, newPrimary); err != nil {
						return err
					}
					shard.Primary = newPrimary
					shard.Replicas = shard.Replicas[1:]
				}
			}
		}
	}

	return nil
}

func (cm *ClusterManagerImpl) notifyNodeLeave(ctx context.Context) error {
	if cm.network == nil {
		return nil
	}

	// Notify all nodes that this node is leaving
	for _, node := range cm.nodes {
		if node.ID != cm.nodeInfo.ID {
			go cm.network.NodeLeaveNotification(ctx, node.Address, cm.nodeInfo.ID)
		}
	}

	return nil
}

func (cm *ClusterManagerImpl) notifyShardCreation(ctx context.Context, shard *ShardInfo) error {
	if cm.network == nil {
		return nil
	}

	// Notify primary and replica nodes
	nodes := append([]string{shard.Primary}, shard.Replicas...)
	for _, nodeID := range nodes {
		if node, exists := cm.nodes[nodeID]; exists {
			go cm.network.CreateShardRequest(ctx, node.Address, shard)
		}
	}

	return nil
}

func (cm *ClusterManagerImpl) migrateShard(ctx context.Context, shard *ShardInfo, newPrimary string) error {
	if cm.network == nil {
		return fmt.Errorf("network layer not initialized")
	}

	// Implement shard migration logic
	// This would involve copying data from old primary to new primary
	cm.stats.ShardMigrations++
	return nil
}

func (cm *ClusterManagerImpl) reassignShardsFromFailedNode(ctx context.Context, failedNodeID string) {
	// Reassign shards that were on the failed node
	for _, shards := range cm.shards {
		for _, shard := range shards {
			needsReassignment := false

			// Check if failed node was primary
			if shard.Primary == failedNodeID {
				if len(shard.Replicas) > 0 {
					shard.Primary = shard.Replicas[0]
					shard.Replicas = shard.Replicas[1:]
					needsReassignment = true
				}
			}

			// Remove failed node from replicas
			newReplicas := make([]string, 0, len(shard.Replicas))
			for _, replica := range shard.Replicas {
				if replica != failedNodeID {
					newReplicas = append(newReplicas, replica)
				} else {
					needsReassignment = true
				}
			}
			shard.Replicas = newReplicas

			// If replication factor is not met, add new replicas
			if needsReassignment && len(shard.Replicas)+1 < shard.ReplicationFactor {
				if ring, err := cm.hashRings.GetRing(shard.CollectionID); err == nil {
					if nodes, err := ring.GetNodes(shard.ID, shard.ReplicationFactor); err == nil {
						// Add new replicas from available nodes
						existing := make(map[string]bool)
						existing[shard.Primary] = true
						for _, replica := range shard.Replicas {
							existing[replica] = true
						}

						for _, node := range nodes {
							if !existing[node] && len(shard.Replicas)+1 < shard.ReplicationFactor {
								shard.Replicas = append(shard.Replicas, node)
							}
						}
					}
				}
			}

			if needsReassignment {
				shard.UpdatedAt = time.Now()
			}
		}
	}
}

// GetStats returns cluster statistics
func (cm *ClusterManagerImpl) GetStats() *ClusterStats {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Return a copy
	stats := *cm.stats
	return &stats
}

// NodeManager interface defines methods for external node management
type NodeManager interface {
	AddNode(ctx context.Context, node *NodeInfo) error
	RemoveNode(ctx context.Context, nodeID string) error
	UpdateNode(ctx context.Context, node *NodeInfo) error
	HandleHeartbeat(ctx context.Context, node *NodeInfo) error
}

// AddNode adds a new node to the cluster (called when a node joins)
func (cm *ClusterManagerImpl) AddNode(ctx context.Context, node *NodeInfo) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if _, exists := cm.nodes[node.ID]; exists {
		return fmt.Errorf("node %s already exists", node.ID)
	}

	node.JoinedAt = time.Now()
	node.LastHeartbeat = time.Now()
	cm.nodes[node.ID] = node

	// Add node to all hash rings
	cm.hashRings.AddNodeToAll(node.ID)

	cm.stats.NodeJoins++
	return nil
}

// RemoveNode removes a node from the cluster
func (cm *ClusterManagerImpl) RemoveNode(ctx context.Context, nodeID string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if _, exists := cm.nodes[nodeID]; !exists {
		return fmt.Errorf("node %s not found", nodeID)
	}

	delete(cm.nodes, nodeID)

	// Remove node from all hash rings
	cm.hashRings.RemoveNodeFromAll(nodeID)

	// Reassign shards
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute*5)
		defer cancel()
		cm.reassignShardsFromFailedNode(ctx, nodeID)
	}()

	cm.stats.NodeLeaves++
	return nil
}

// UpdateNode updates node information
func (cm *ClusterManagerImpl) UpdateNode(ctx context.Context, node *NodeInfo) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if _, exists := cm.nodes[node.ID]; !exists {
		return fmt.Errorf("node %s not found", node.ID)
	}

	cm.nodes[node.ID] = node
	return nil
}

// HandleHeartbeat handles heartbeat from another node
func (cm *ClusterManagerImpl) HandleHeartbeat(ctx context.Context, node *NodeInfo) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	if existingNode, exists := cm.nodes[node.ID]; exists {
		existingNode.LastHeartbeat = time.Now()
		existingNode.State = node.State
		existingNode.CPUUsage = node.CPUUsage
		existingNode.UsedMemory = node.UsedMemory
		existingNode.UsedStorage = node.UsedStorage
	} else {
		// New node joining
		node.LastHeartbeat = time.Now()
		cm.nodes[node.ID] = node
		cm.hashRings.AddNodeToAll(node.ID)
		cm.stats.NodeJoins++
	}

	cm.stats.HeartbeatsReceived++
	return nil
}

// NetworkManager defines the interface for network communication
type NetworkManager interface {
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	SendHeartbeat(ctx context.Context, address string, node *NodeInfo) error
	JoinRequest(ctx context.Context, address string, node *NodeInfo) error
	NodeLeaveNotification(ctx context.Context, address string, nodeID string) error
	CreateShardRequest(ctx context.Context, address string, shard *ShardInfo) error
	SendQuery(ctx context.Context, address string, req *api.SearchRequest) ([]*api.SearchResult, error)
	SendBatchQuery(ctx context.Context, address string, req *api.BatchRequest) (*api.BatchResponse, error)
	HealthCheck(ctx context.Context) error
}
