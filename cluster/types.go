// Package cluster provides distributed system capabilities for GoVecDB.
// It implements node discovery, consensus, sharding, and distributed query processing.
package cluster

import (
	"context"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// NodeState represents the current state of a cluster node
type NodeState int

const (
	NodeStateUnknown NodeState = iota
	NodeStateStarting
	NodeStateActive
	NodeStateDraining
	NodeStateShuttingDown
	NodeStateFailed
)

// String returns the string representation of the node state
func (s NodeState) String() string {
	switch s {
	case NodeStateStarting:
		return "STARTING"
	case NodeStateActive:
		return "ACTIVE"
	case NodeStateDraining:
		return "DRAINING"
	case NodeStateShuttingDown:
		return "SHUTTING_DOWN"
	case NodeStateFailed:
		return "FAILED"
	default:
		return "UNKNOWN"
	}
}

// NodeRole represents the role of a node in the cluster
type NodeRole int

const (
	NodeRoleUnknown NodeRole = iota
	NodeRoleData             // Stores and processes vector data
	NodeRoleQuery            // Handles query coordination only
	NodeRoleMixed            // Both data and query processing
	NodeRoleWitness          // Participates in consensus but doesn't store data
)

// String returns the string representation of the node role
func (r NodeRole) String() string {
	switch r {
	case NodeRoleData:
		return "DATA"
	case NodeRoleQuery:
		return "QUERY"
	case NodeRoleMixed:
		return "MIXED"
	case NodeRoleWitness:
		return "WITNESS"
	default:
		return "UNKNOWN"
	}
}

// NodeInfo represents information about a cluster node
type NodeInfo struct {
	ID      string    `json:"id"`      // Unique node identifier
	Address string    `json:"address"` // Network address (host:port)
	Role    NodeRole  `json:"role"`    // Node role in cluster
	State   NodeState `json:"state"`   // Current node state
	Version string    `json:"version"` // Software version
	Region  string    `json:"region"`  // Geographic region/zone
	Rack    string    `json:"rack"`    // Rack identifier for topology awareness

	// Capabilities
	MaxMemory        int64 `json:"max_memory"`        // Maximum memory available
	MaxStorage       int64 `json:"max_storage"`       // Maximum storage available
	CPUCores         int   `json:"cpu_cores"`         // Number of CPU cores
	NetworkBandwidth int64 `json:"network_bandwidth"` // Network bandwidth

	// Runtime stats
	UsedMemory    int64     `json:"used_memory"`    // Currently used memory
	UsedStorage   int64     `json:"used_storage"`   // Currently used storage
	CPUUsage      float64   `json:"cpu_usage"`      // CPU usage percentage
	LastHeartbeat time.Time `json:"last_heartbeat"` // Last heartbeat timestamp
	JoinedAt      time.Time `json:"joined_at"`      // When node joined cluster

	// Metadata
	Tags     map[string]string      `json:"tags"`     // Custom node tags
	Metadata map[string]interface{} `json:"metadata"` // Additional metadata
}

// ShardInfo represents information about a data shard
type ShardInfo struct {
	ID           string    `json:"id"`            // Unique shard identifier
	CollectionID string    `json:"collection_id"` // Collection this shard belongs to
	HashRange    HashRange `json:"hash_range"`    // Hash range for this shard

	// Replication
	Primary           string   `json:"primary"`            // Primary node ID
	Replicas          []string `json:"replicas"`           // Replica node IDs
	ReplicationFactor int      `json:"replication_factor"` // Desired replication factor

	// State
	State       ShardState `json:"state"`        // Current shard state
	Size        int64      `json:"size"`         // Approximate size in bytes
	VectorCount int64      `json:"vector_count"` // Number of vectors in shard
	CreatedAt   time.Time  `json:"created_at"`   // When shard was created
	UpdatedAt   time.Time  `json:"updated_at"`   // Last update timestamp

	// Health
	HealthScore float64   `json:"health_score"` // Health score (0.0-1.0)
	LastAccess  time.Time `json:"last_access"`  // Last access timestamp
}

// ShardState represents the state of a shard
type ShardState int

const (
	ShardStateUnknown ShardState = iota
	ShardStateCreating
	ShardStateActive
	ShardStateRebalancing
	ShardStateMigrating
	ShardStateReadOnly
	ShardStateFailed
)

// String returns the string representation of the shard state
func (s ShardState) String() string {
	switch s {
	case ShardStateCreating:
		return "CREATING"
	case ShardStateActive:
		return "ACTIVE"
	case ShardStateRebalancing:
		return "REBALANCING"
	case ShardStateMigrating:
		return "MIGRATING"
	case ShardStateReadOnly:
		return "READ_ONLY"
	case ShardStateFailed:
		return "FAILED"
	default:
		return "UNKNOWN"
	}
}

// HashRange represents a hash range for consistent hashing
type HashRange struct {
	Start uint64 `json:"start"` // Start of hash range (inclusive)
	End   uint64 `json:"end"`   // End of hash range (exclusive)
}

// Contains checks if a hash value falls within this range
func (hr HashRange) Contains(hash uint64) bool {
	if hr.Start <= hr.End {
		return hash >= hr.Start && hash < hr.End
	}
	// Handle wrap-around case
	return hash >= hr.Start || hash < hr.End
}

// Size returns the size of the hash range
func (hr HashRange) Size() uint64 {
	if hr.Start <= hr.End {
		return hr.End - hr.Start
	}
	// Handle wrap-around case
	return (^uint64(0) - hr.Start) + hr.End + 1
}

// QueryPlan represents a plan for executing a distributed query
type QueryPlan struct {
	ID          string    `json:"id"`          // Unique plan identifier
	QueryType   QueryType `json:"query_type"`  // Type of query
	Collection  string    `json:"collection"`  // Target collection
	Coordinator string    `json:"coordinator"` // Coordinating node ID

	// Execution plan
	Phases           []QueryPhase  `json:"phases"`            // Query execution phases
	Shards           []string      `json:"shards"`            // Involved shards
	Nodes            []string      `json:"nodes"`             // Involved nodes
	EstimatedLatency time.Duration `json:"estimated_latency"` // Estimated execution time

	// Configuration
	Parallelism int                    `json:"parallelism"` // Degree of parallelism
	Timeout     time.Duration          `json:"timeout"`     // Query timeout
	Parameters  map[string]interface{} `json:"parameters"`  // Query parameters

	// State
	State       QueryPlanState `json:"state"`        // Current plan state
	CreatedAt   time.Time      `json:"created_at"`   // Plan creation time
	StartedAt   *time.Time     `json:"started_at"`   // Execution start time
	CompletedAt *time.Time     `json:"completed_at"` // Execution completion time
}

// QueryType represents the type of distributed query
type QueryType int

const (
	QueryTypeUnknown   QueryType = iota
	QueryTypeSearch              // Vector similarity search
	QueryTypeInsert              // Insert vectors
	QueryTypeUpdate              // Update vectors
	QueryTypeDelete              // Delete vectors
	QueryTypeBatch               // Batch operations
	QueryTypeAggregate           // Aggregation queries
)

// String returns the string representation of the query type
func (q QueryType) String() string {
	switch q {
	case QueryTypeSearch:
		return "SEARCH"
	case QueryTypeInsert:
		return "INSERT"
	case QueryTypeUpdate:
		return "UPDATE"
	case QueryTypeDelete:
		return "DELETE"
	case QueryTypeBatch:
		return "BATCH"
	case QueryTypeAggregate:
		return "AGGREGATE"
	default:
		return "UNKNOWN"
	}
}

// QueryPhase represents a phase in query execution
type QueryPhase struct {
	ID           string         `json:"id"`           // Phase identifier
	Type         QueryPhaseType `json:"type"`         // Phase type
	Node         string         `json:"node"`         // Executing node
	Dependencies []string       `json:"dependencies"` // Dependent phases

	// Execution details
	Operation  string                 `json:"operation"`  // Operation to perform
	Parameters map[string]interface{} `json:"parameters"` // Phase parameters
	InputData  []byte                 `json:"input_data"` // Input data (if any)

	// State
	State       QueryPhaseState `json:"state"`        // Current phase state
	StartedAt   *time.Time      `json:"started_at"`   // Phase start time
	CompletedAt *time.Time      `json:"completed_at"` // Phase completion time
	Duration    time.Duration   `json:"duration"`     // Actual execution duration
	Error       string          `json:"error"`        // Error message (if failed)
}

// QueryPhaseType represents the type of query phase
type QueryPhaseType int

const (
	QueryPhaseTypeUnknown   QueryPhaseType = iota
	QueryPhaseTypeScatter                  // Scatter query to multiple nodes
	QueryPhaseTypeGather                   // Gather results from multiple nodes
	QueryPhaseTypeMap                      // Map operation on data
	QueryPhaseTypeReduce                   // Reduce operation on results
	QueryPhaseTypeFilter                   // Filter operation
	QueryPhaseTypeSort                     // Sort operation
	QueryPhaseTypeAggregate                // Aggregation operation
)

// String returns the string representation of the query phase type
func (q QueryPhaseType) String() string {
	switch q {
	case QueryPhaseTypeScatter:
		return "SCATTER"
	case QueryPhaseTypeGather:
		return "GATHER"
	case QueryPhaseTypeMap:
		return "MAP"
	case QueryPhaseTypeReduce:
		return "REDUCE"
	case QueryPhaseTypeFilter:
		return "FILTER"
	case QueryPhaseTypeSort:
		return "SORT"
	case QueryPhaseTypeAggregate:
		return "AGGREGATE"
	default:
		return "UNKNOWN"
	}
}

// QueryPlanState represents the state of a query plan
type QueryPlanState int

const (
	QueryPlanStateUnknown QueryPlanState = iota
	QueryPlanStatePending
	QueryPlanStateExecuting
	QueryPlanStateCompleted
	QueryPlanStateFailed
	QueryPlanStateCancelled
)

// String returns the string representation of the query plan state
func (q QueryPlanState) String() string {
	switch q {
	case QueryPlanStatePending:
		return "PENDING"
	case QueryPlanStateExecuting:
		return "EXECUTING"
	case QueryPlanStateCompleted:
		return "COMPLETED"
	case QueryPlanStateFailed:
		return "FAILED"
	case QueryPlanStateCancelled:
		return "CANCELLED"
	default:
		return "UNKNOWN"
	}
}

// QueryPhaseState represents the state of a query phase
type QueryPhaseState int

const (
	QueryPhaseStateUnknown QueryPhaseState = iota
	QueryPhaseStatePending
	QueryPhaseStateExecuting
	QueryPhaseStateCompleted
	QueryPhaseStateFailed
	QueryPhaseStateSkipped
)

// String returns the string representation of the query phase state
func (q QueryPhaseState) String() string {
	switch q {
	case QueryPhaseStatePending:
		return "PENDING"
	case QueryPhaseStateExecuting:
		return "EXECUTING"
	case QueryPhaseStateCompleted:
		return "COMPLETED"
	case QueryPhaseStateFailed:
		return "FAILED"
	case QueryPhaseStateSkipped:
		return "SKIPPED"
	default:
		return "UNKNOWN"
	}
}

// ClusterConfig represents configuration for a cluster
type ClusterConfig struct {
	// Cluster identity
	ClusterID   string `json:"cluster_id"`   // Unique cluster identifier
	ClusterName string `json:"cluster_name"` // Human-readable cluster name

	// Node configuration
	NodeID      string   `json:"node_id"`      // This node's ID
	BindAddress string   `json:"bind_address"` // Address to bind to
	SeedNodes   []string `json:"seed_nodes"`   // Initial seed nodes

	// Consensus settings
	ConsensusAlgorithm string        `json:"consensus_algorithm"` // "raft", "pbft", etc.
	HeartbeatInterval  time.Duration `json:"heartbeat_interval"`  // Heartbeat frequency
	ElectionTimeout    time.Duration `json:"election_timeout"`    // Leader election timeout

	// Replication settings
	DefaultReplicationFactor int    `json:"default_replication_factor"` // Default replication factor
	ConsistencyLevel         string `json:"consistency_level"`          // "strong", "eventual", etc.

	// Sharding settings
	ShardingStrategy   string  `json:"sharding_strategy"`   // "consistent_hash", "range", etc.
	ShardsPerNode      int     `json:"shards_per_node"`     // Target shards per node
	RebalanceThreshold float64 `json:"rebalance_threshold"` // When to trigger rebalancing

	// Performance settings
	MaxConcurrentQueries int           `json:"max_concurrent_queries"` // Max concurrent queries
	QueryTimeout         time.Duration `json:"query_timeout"`          // Default query timeout
	BatchSize            int           `json:"batch_size"`             // Default batch size

	// Network settings
	ConnectionPoolSize int           `json:"connection_pool_size"` // Connection pool size
	MaxMessageSize     int           `json:"max_message_size"`     // Max message size
	NetworkTimeout     time.Duration `json:"network_timeout"`      // Network timeout

	// Storage settings
	DataDir     string `json:"data_dir"`     // Data directory
	WALDir      string `json:"wal_dir"`      // WAL directory
	SnapshotDir string `json:"snapshot_dir"` // Snapshot directory
	MetadataDir string `json:"metadata_dir"` // Metadata directory
}

// DefaultClusterConfig returns a default cluster configuration
func DefaultClusterConfig(nodeID, bindAddress string) *ClusterConfig {
	return &ClusterConfig{
		ClusterID:   "govecdb-cluster",
		ClusterName: "GoVecDB Cluster",
		NodeID:      nodeID,
		BindAddress: bindAddress,
		SeedNodes:   []string{},

		ConsensusAlgorithm: "raft",
		HeartbeatInterval:  time.Second * 5,
		ElectionTimeout:    time.Second * 10,

		DefaultReplicationFactor: 3,
		ConsistencyLevel:         "strong",

		ShardingStrategy:   "consistent_hash",
		ShardsPerNode:      16,
		RebalanceThreshold: 0.1, // 10% imbalance triggers rebalancing

		MaxConcurrentQueries: 100,
		QueryTimeout:         time.Minute * 5,
		BatchSize:            1000,

		ConnectionPoolSize: 10,
		MaxMessageSize:     64 * 1024 * 1024, // 64MB
		NetworkTimeout:     time.Second * 30,

		DataDir:     "./data",
		WALDir:      "./data/wal",
		SnapshotDir: "./data/snapshots",
		MetadataDir: "./data/metadata",
	}
}

// ClusterManager defines the interface for cluster management
type ClusterManager interface {
	// Lifecycle
	Start(ctx context.Context) error
	Stop(ctx context.Context) error

	// Node management
	JoinCluster(ctx context.Context, seedNodes []string) error
	LeaveCluster(ctx context.Context) error
	GetNodes(ctx context.Context) ([]*NodeInfo, error)
	GetNode(ctx context.Context, nodeID string) (*NodeInfo, error)

	// Shard management
	GetShards(ctx context.Context, collectionID string) ([]*ShardInfo, error)
	CreateShard(ctx context.Context, collectionID string, replicationFactor int) (*ShardInfo, error)
	RebalanceShards(ctx context.Context, collectionID string) error

	// Health monitoring
	HealthCheck(ctx context.Context) error
	GetClusterHealth(ctx context.Context) (*ClusterHealth, error)
}

// QueryCoordinator defines the interface for distributed query coordination
type QueryCoordinator interface {
	// Query execution
	ExecuteQuery(ctx context.Context, req *api.SearchRequest) ([]*api.SearchResult, error)
	ExecuteBatch(ctx context.Context, req *api.BatchRequest) (*api.BatchResponse, error)

	// Query planning
	PlanQuery(ctx context.Context, req *api.SearchRequest) (*QueryPlan, error)
	ExecutePlan(ctx context.Context, plan *QueryPlan) ([]*api.SearchResult, error)

	// Query management
	GetActiveQueries(ctx context.Context) ([]*QueryPlan, error)
	CancelQuery(ctx context.Context, queryID string) error
}

// ShardManager defines the interface for shard management
type ShardManager interface {
	// Shard operations
	CreateShard(ctx context.Context, collectionID string, hashRange HashRange) (*ShardInfo, error)
	DeleteShard(ctx context.Context, shardID string) error
	MoveShard(ctx context.Context, shardID, sourceNode, targetNode string) error

	// Shard distribution
	GetShardDistribution(ctx context.Context, collectionID string) (map[string][]*ShardInfo, error)
	RebalanceShards(ctx context.Context, collectionID string) error

	// Hash ring management
	GetHashRing(ctx context.Context, collectionID string) (*HashRing, error)
	UpdateHashRing(ctx context.Context, collectionID string, nodes []string) error
}

// ConsensusManager defines the interface for distributed consensus
type ConsensusManager interface {
	// Lifecycle
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	HealthCheck(ctx context.Context) error

	// Consensus operations
	Propose(ctx context.Context, proposal *Proposal) (*ProposalResult, error)
	Vote(ctx context.Context, proposalID string, vote Vote) error

	// Leadership
	IsLeader() bool
	GetLeader() (*NodeInfo, error)

	// State management
	GetState(ctx context.Context) (*ConsensusState, error)
	ApplyCommand(ctx context.Context, command *Command) error
}

// ClusterHealth represents the health status of the entire cluster
type ClusterHealth struct {
	ClusterID     string             `json:"cluster_id"`
	State         ClusterState       `json:"state"`
	NodesTotal    int                `json:"nodes_total"`
	NodesHealthy  int                `json:"nodes_healthy"`
	NodesFailed   int                `json:"nodes_failed"`
	ShardsTotal   int                `json:"shards_total"`
	ShardsHealthy int                `json:"shards_healthy"`
	ShardsFailed  int                `json:"shards_failed"`
	LoadBalanced  bool               `json:"load_balanced"`
	LastCheck     time.Time          `json:"last_check"`
	Issues        []string           `json:"issues"`
	NodeHealth    map[string]float64 `json:"node_health"` // nodeID -> health score
}

// ClusterState represents the overall state of the cluster
type ClusterState int

const (
	ClusterStateUnknown ClusterState = iota
	ClusterStateInitializing
	ClusterStateHealthy
	ClusterStateDegraded
	ClusterStatePartitioned
	ClusterStateFailed
)

// String returns the string representation of the cluster state
func (c ClusterState) String() string {
	switch c {
	case ClusterStateInitializing:
		return "INITIALIZING"
	case ClusterStateHealthy:
		return "HEALTHY"
	case ClusterStateDegraded:
		return "DEGRADED"
	case ClusterStatePartitioned:
		return "PARTITIONED"
	case ClusterStateFailed:
		return "FAILED"
	default:
		return "UNKNOWN"
	}
}

// HashRing represents a consistent hash ring for data distribution
type HashRing struct {
	Nodes        []string          `json:"nodes"`         // Nodes in the ring
	VirtualNodes int               `json:"virtual_nodes"` // Virtual nodes per physical node
	Ring         map[uint64]string `json:"ring"`          // Hash -> Node mapping
	Sorted       []uint64          `json:"sorted"`        // Sorted hash values
}

// Proposal represents a consensus proposal
type Proposal struct {
	ID        string                 `json:"id"`
	Type      ProposalType           `json:"type"`
	Data      []byte                 `json:"data"`
	Proposer  string                 `json:"proposer"`
	Timestamp time.Time              `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// ProposalType represents the type of consensus proposal
type ProposalType int

const (
	ProposalTypeUnknown ProposalType = iota
	ProposalTypeClusterChange
	ProposalTypeShardChange
	ProposalTypeConfigChange
	ProposalTypeMetadataUpdate
)

// ProposalResult represents the result of a consensus proposal
type ProposalResult struct {
	ProposalID string     `json:"proposal_id"`
	Accepted   bool       `json:"accepted"`
	Votes      []Vote     `json:"votes"`
	AppliedAt  *time.Time `json:"applied_at"`
	Error      string     `json:"error"`
}

// Vote represents a vote on a consensus proposal
type Vote struct {
	NodeID     string    `json:"node_id"`
	ProposalID string    `json:"proposal_id"`
	Accept     bool      `json:"accept"`
	Reason     string    `json:"reason"`
	Timestamp  time.Time `json:"timestamp"`
}

// ConsensusState represents the current state of the consensus protocol
type ConsensusState struct {
	Term        uint64            `json:"term"`
	Leader      string            `json:"leader"`
	Commitments map[string]uint64 `json:"commitments"` // nodeID -> last committed index
	LastApplied uint64            `json:"last_applied"`
	Log         []*LogEntry       `json:"log"`
}

// LogEntry represents an entry in the consensus log
type LogEntry struct {
	Index     uint64    `json:"index"`
	Term      uint64    `json:"term"`
	Command   *Command  `json:"command"`
	Timestamp time.Time `json:"timestamp"`
}

// Command represents a command to be applied to the state machine
type Command struct {
	Type     CommandType            `json:"type"`
	Data     []byte                 `json:"data"`
	Metadata map[string]interface{} `json:"metadata"`
}

// CommandType represents the type of state machine command
type CommandType int

const (
	CommandTypeUnknown CommandType = iota
	CommandTypeNodeJoin
	CommandTypeNodeLeave
	CommandTypeShardCreate
	CommandTypeShardMove
	CommandTypeConfigUpdate
)
