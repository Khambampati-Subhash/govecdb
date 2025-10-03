# 🌐 GoVecDB in Distributed Systems

## 🎯 Overview

GoVecDB is designed from the ground up to operate efficiently in distributed environments. This guide explains how to deploy, configure, and optimize GoVecDB for distributed systems, covering everything from basic clustering to advanced deployment patterns.

## 🏗️ Distributed Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    GoVecDB Cluster                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Node 1  │  │ Node 2  │  │ Node 3  │  │ Node N  │       │
│  │ Leader  │  │Follower │  │Follower │  │Follower │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
├─────────────────────────────────────────────────────────────┤
│           Consistent Hash Ring (Data Distribution)         │
├─────────────────────────────────────────────────────────────┤
│              Raft Consensus (Coordination)                 │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Consistent Hashing**: Automatic data distribution across nodes
2. **Raft Consensus**: Leader election and distributed coordination
3. **Replication**: Configurable replication factor for fault tolerance
4. **Partitioning**: Intelligent data sharding for horizontal scalability

## 🚀 Quick Start: Distributed Setup

### Basic 3-Node Cluster

```go
package main

import (
    "context"
    "log"
    
    "github.com/khambampati-subhash/govecdb/cluster"
    "github.com/khambampati-subhash/govecdb/api"
)

func main() {
    // Node 1 (Leader candidate)
    node1Config := &cluster.Config{
        NodeID:            "node-1",
        ListenAddr:        "localhost:8001",
        ReplicationFactor: 3,
        ShardCount:        32,
        
        // Raft configuration
        ConsensusConfig: &cluster.RaftConfig{
            HeartbeatTimeout:  100 * time.Millisecond,
            ElectionTimeout:   500 * time.Millisecond,
            LeadershipTimeout: 10 * time.Second,
        },
        
        // Peer nodes
        Peers: []string{
            "localhost:8002",
            "localhost:8003",
        },
    }
    
    manager1 := cluster.NewClusterManager(node1Config)
    coordinator1 := cluster.NewQueryCoordinator(manager1)
    
    // Start the cluster node
    ctx := context.Background()
    if err := manager1.Start(ctx); err != nil {
        log.Fatalf("Failed to start node 1: %v", err)
    }
    
    // Create a distributed collection
    collectionConfig := &api.CollectionConfig{
        Name:      "distributed-vectors",
        Dimension: 384,
        Metric:    api.Cosine,
    }
    
    collection, err := coordinator1.CreateCollection(ctx, collectionConfig)
    if err != nil {
        log.Fatalf("Failed to create collection: %v", err)
    }
    
    // Insert vectors (automatically distributed)
    vectors := []*api.Vector{
        {ID: "doc1", Data: generateEmbedding(), Metadata: map[string]interface{}{"type": "document"}},
        {ID: "doc2", Data: generateEmbedding(), Metadata: map[string]interface{}{"type": "image"}},
    }
    
    err = collection.AddBatch(vectors)
    if err != nil {
        log.Fatalf("Failed to insert vectors: %v", err)
    }
    
    // Distributed search across all nodes
    results, err := coordinator1.Search(&api.SearchRequest{
        Vector: generateQueryEmbedding(),
        K:      10,
    })
    
    log.Printf("Found %d results across cluster", len(results))
}
```

## 🔧 Configuration Patterns

### Production Cluster Configuration

```go
type ProductionClusterConfig struct {
    // Node identification
    NodeID     string
    DataCenter string
    Region     string
    
    // Network configuration
    ListenAddr      string
    AdvertiseAddr   string
    BindAddr        string
    
    // Cluster topology
    ReplicationFactor int    // Recommended: 3 for production
    ShardCount        int    // Recommended: 32-128 depending on scale
    
    // Performance tuning
    BatchSize         int    // Default: 100
    MaxConnections    int    // Default: 1000
    QueryTimeout      time.Duration
    
    // Consistency settings
    ConsistencyLevel  cluster.ConsistencyLevel
    ReadQuorum        int
    WriteQuorum       int
    
    // Storage configuration
    DataDir           string
    WALDir            string
    SnapshotInterval  time.Duration
    CompactionInterval time.Duration
}

func NewProductionCluster(config *ProductionClusterConfig) *cluster.Manager {
    clusterConfig := &cluster.Config{
        NodeID:            config.NodeID,
        ListenAddr:       config.ListenAddr,
        ReplicationFactor: config.ReplicationFactor,
        ShardCount:       config.ShardCount,
        
        // Production-tuned Raft settings
        ConsensusConfig: &cluster.RaftConfig{
            HeartbeatTimeout:     50 * time.Millisecond,   // Faster failure detection
            ElectionTimeout:      250 * time.Millisecond,  // Quick leader election
            LeadershipTimeout:    5 * time.Second,         // Reasonable leadership timeout
            SnapshotThreshold:    10000,                   // Snapshot every 10k operations
            MaxAppendEntries:     64,                      // Batch size for log replication
        },
        
        // Network optimization
        NetworkConfig: &cluster.NetworkConfig{
            MaxConnections:    config.MaxConnections,
            ConnectionPoolSize: 10,
            KeepAliveInterval: 30 * time.Second,
            ReadTimeout:       5 * time.Second,
            WriteTimeout:      5 * time.Second,
        },
        
        // Storage optimization
        StorageConfig: &cluster.StorageConfig{
            DataDir:            config.DataDir,
            WALDir:             config.WALDir,
            SnapshotInterval:   config.SnapshotInterval,
            CompactionInterval: config.CompactionInterval,
            MaxWALSize:         500 * 1024 * 1024, // 500MB
        },
    }
    
    return cluster.NewClusterManager(clusterConfig)
}
```

## 🌊 Data Flow and Distribution

### Write Operations

```
Client Write Request
        ↓
    Coordinator
        ↓
   Query Planning ────┐
        ↓            │
   Hash Ring Lookup  │  Determine target nodes
        ↓            │  based on key hash
   Select Nodes ←────┘
        ↓
   Replicate to N nodes (async)
        ↓
   Write to local WAL (sync)
        ↓
   Update HNSW index
        ↓
   Return success to client
```

### Read Operations

```
Client Search Request
        ↓
    Coordinator
        ↓
   Query Planning ────┐
        ↓            │
   Determine Strategy │  • Single node (if key known)
        ↓            │  • Scatter-gather (if range query)
   Execute Query ←────┘  • Parallel search across shards
        ↓
   Collect Results
        ↓
   Merge & Rank
        ↓
   Return to Client
```

## 🎛️ Deployment Patterns

### 1. Single Datacenter Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  govecdb-node1:
    image: govecdb:latest
    environment:
      - NODE_ID=node-1
      - LISTEN_ADDR=0.0.0.0:8001
      - PEERS=node-2:8002,node-3:8003
      - REPLICATION_FACTOR=3
    ports:
      - "8001:8001"
    volumes:
      - ./data/node1:/data

  govecdb-node2:
    image: govecdb:latest
    environment:
      - NODE_ID=node-2
      - LISTEN_ADDR=0.0.0.0:8002
      - PEERS=node-1:8001,node-3:8003
      - REPLICATION_FACTOR=3
    ports:
      - "8002:8002"
    volumes:
      - ./data/node2:/data

  govecdb-node3:
    image: govecdb:latest
    environment:
      - NODE_ID=node-3
      - LISTEN_ADDR=0.0.0.0:8003
      - PEERS=node-1:8001,node-2:8002
      - REPLICATION_FACTOR=3
    ports:
      - "8003:8003"
    volumes:
      - ./data/node3:/data
```

### 2. Multi-Datacenter Deployment

```go
// Multi-region cluster configuration
func SetupMultiRegionCluster() {
    // US East cluster
    usEastNodes := []cluster.NodeConfig{
        {NodeID: "us-east-1", Region: "us-east", DataCenter: "dc1"},
        {NodeID: "us-east-2", Region: "us-east", DataCenter: "dc1"},
        {NodeID: "us-east-3", Region: "us-east", DataCenter: "dc2"},
    }
    
    // EU West cluster
    euWestNodes := []cluster.NodeConfig{
        {NodeID: "eu-west-1", Region: "eu-west", DataCenter: "dc1"},
        {NodeID: "eu-west-2", Region: "eu-west", DataCenter: "dc1"},
        {NodeID: "eu-west-3", Region: "eu-west", DataCenter: "dc2"},
    }
    
    // Global cluster configuration
    globalConfig := &cluster.GlobalConfig{
        Regions: map[string]cluster.RegionConfig{
            "us-east": {
                Nodes:             usEastNodes,
                PreferredReplicas: 2,
                CrossRegionReplicas: 1,
            },
            "eu-west": {
                Nodes:             euWestNodes,
                PreferredReplicas: 2,
                CrossRegionReplicas: 1,
            },
        },
        
        // Global settings
        GlobalReplicationFactor: 3,
        CrossRegionLatencyThreshold: 100 * time.Millisecond,
        ConsistencyMode: cluster.EventualConsistency,
    }
    
    cluster := cluster.NewGlobalCluster(globalConfig)
    cluster.Start()
}
```

### 3. Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: govecdb-cluster
spec:
  serviceName: govecdb
  replicas: 3
  selector:
    matchLabels:
      app: govecdb
  template:
    metadata:
      labels:
        app: govecdb
    spec:
      containers:
      - name: govecdb
        image: govecdb:latest
        ports:
        - containerPort: 8001
        env:
        - name: NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: LISTEN_ADDR
          value: "0.0.0.0:8001"
        - name: PEERS
          value: "govecdb-cluster-0.govecdb:8001,govecdb-cluster-1.govecdb:8001,govecdb-cluster-2.govecdb:8001"
        volumeMounts:
        - name: data
          mountPath: /data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: govecdb
spec:
  clusterIP: None
  selector:
    app: govecdb
  ports:
  - port: 8001
    targetPort: 8001
```

## 🔄 Scaling Strategies

### Horizontal Scaling

```go
// Dynamic node addition
func AddNodeToCluster(manager *cluster.Manager, newNodeConfig *cluster.NodeConfig) error {
    // 1. Prepare new node
    newNode := cluster.NewNode(newNodeConfig)
    
    // 2. Join cluster
    err := manager.AddNode(newNode)
    if err != nil {
        return fmt.Errorf("failed to add node: %w", err)
    }
    
    // 3. Rebalance data (automatic)
    rebalancer := cluster.NewRebalancer(manager)
    err = rebalancer.RebalanceAsync()
    if err != nil {
        return fmt.Errorf("failed to start rebalancing: %w", err)
    }
    
    // 4. Update routing tables
    err = manager.UpdateRoutingTables()
    if err != nil {
        return fmt.Errorf("failed to update routing: %w", err)
    }
    
    return nil
}

// Graceful node removal
func RemoveNodeFromCluster(manager *cluster.Manager, nodeID string) error {
    // 1. Start data migration
    migrator := cluster.NewDataMigrator(manager)
    err := migrator.MigrateFromNode(nodeID)
    if err != nil {
        return fmt.Errorf("failed to migrate data: %w", err)
    }
    
    // 2. Remove from cluster
    err = manager.RemoveNode(nodeID)
    if err != nil {
        return fmt.Errorf("failed to remove node: %w", err)
    }
    
    // 3. Update routing tables
    return manager.UpdateRoutingTables()
}
```

### Auto-scaling Configuration

```go
type AutoScalingConfig struct {
    MinNodes        int
    MaxNodes        int
    TargetCPU       float64 // CPU utilization threshold (0.0-1.0)
    TargetMemory    float64 // Memory utilization threshold (0.0-1.0)
    TargetLatency   time.Duration // P95 latency threshold
    ScaleUpCooldown time.Duration
    ScaleDownCooldown time.Duration
}

func SetupAutoScaling(manager *cluster.Manager, config *AutoScalingConfig) {
    scaler := cluster.NewAutoScaler(manager, config)
    
    // Monitor metrics and scale accordingly
    go scaler.MonitorAndScale()
}
```

## 🛡️ Fault Tolerance and Recovery

### Node Failure Handling

```go
// Implement failure detector
type FailureDetector struct {
    manager     *cluster.Manager
    healthCheck time.Duration
    timeout     time.Duration
}

func (fd *FailureDetector) Start() {
    ticker := time.NewTicker(fd.healthCheck)
    
    go func() {
        for range ticker.C {
            nodes := fd.manager.GetNodes()
            
            for _, node := range nodes {
                ctx, cancel := context.WithTimeout(context.Background(), fd.timeout)
                
                err := node.Ping(ctx)
                if err != nil {
                    log.Printf("Node %s appears unhealthy: %v", node.ID(), err)
                    fd.handleNodeFailure(node)
                }
                
                cancel()
            }
        }
    }()
}

func (fd *FailureDetector) handleNodeFailure(node cluster.Node) {
    // 1. Mark node as failed
    fd.manager.MarkNodeFailed(node.ID())
    
    // 2. Trigger data recovery
    recovery := cluster.NewRecoveryManager(fd.manager)
    recovery.RecoverFromNodeFailure(node.ID())
    
    // 3. Update routing to avoid failed node
    fd.manager.UpdateRoutingTables()
}
```

### Split-Brain Prevention

```go
// Raft consensus prevents split-brain by requiring majority
type ConsensusManager struct {
    nodes       []cluster.Node
    quorumSize  int // (N/2) + 1
    currentTerm int64
    votedFor    string
}

func (cm *ConsensusManager) RequestVote(candidateID string, term int64) bool {
    // Only vote if:
    // 1. Term is newer than current term
    // 2. Haven't voted in this term, or voted for same candidate
    // 3. Candidate's log is at least as up-to-date as ours
    
    if term > cm.currentTerm {
        cm.currentTerm = term
        cm.votedFor = candidateID
        return true
    }
    
    return false
}
```

## 📊 Monitoring and Observability

### Distributed Metrics Collection

```go
type ClusterMetrics struct {
    // Node-level metrics
    NodeHealth      map[string]bool
    NodeLatency     map[string]time.Duration
    NodeThroughput  map[string]float64
    
    // Cluster-level metrics
    TotalNodes      int
    ActiveNodes     int
    LeaderNode      string
    ReplicationLag  time.Duration
    
    // Performance metrics
    QueryLatencyP50 time.Duration
    QueryLatencyP95 time.Duration
    QueryLatencyP99 time.Duration
    IndexSize       int64
    TotalVectors    int64
}

func (cm *ClusterMetrics) CollectMetrics() {
    // Collect from all nodes
    for _, node := range cm.cluster.GetNodes() {
        nodeMetrics := node.GetMetrics()
        cm.NodeHealth[node.ID()] = nodeMetrics.IsHealthy
        cm.NodeLatency[node.ID()] = nodeMetrics.AvgLatency
        cm.NodeThroughput[node.ID()] = nodeMetrics.QPS
    }
    
    // Calculate cluster-wide metrics
    cm.calculateClusterMetrics()
}
```

### Health Checks

```go
type HealthChecker struct {
    manager *cluster.Manager
}

func (hc *HealthChecker) CheckClusterHealth() *HealthStatus {
    status := &HealthStatus{
        Overall: "healthy",
        Nodes:   make(map[string]string),
        Issues:  []string{},
    }
    
    nodes := hc.manager.GetNodes()
    healthyNodes := 0
    
    for _, node := range nodes {
        nodeHealth := hc.checkNodeHealth(node)
        status.Nodes[node.ID()] = nodeHealth
        
        if nodeHealth == "healthy" {
            healthyNodes++
        }
    }
    
    // Check if we have quorum
    if healthyNodes < (len(nodes)/2)+1 {
        status.Overall = "critical"
        status.Issues = append(status.Issues, "Lost quorum")
    }
    
    return status
}
```

## 🔐 Security Considerations

### Authentication and Authorization

```go
type SecurityConfig struct {
    EnableTLS      bool
    CertFile       string
    KeyFile        string
    CAFile         string
    
    EnableAuth     bool
    AuthMethod     string // "jwt", "basic", "mtls"
    
    EnableACL      bool
    ACLFile        string
}

func SetupSecurity(config *SecurityConfig) *cluster.SecurityManager {
    security := cluster.NewSecurityManager()
    
    if config.EnableTLS {
        tlsConfig := &tls.Config{
            CertFile: config.CertFile,
            KeyFile:  config.KeyFile,
            CAFile:   config.CAFile,
        }
        security.EnableTLS(tlsConfig)
    }
    
    if config.EnableAuth {
        auth := cluster.NewAuthenticator(config.AuthMethod)
        security.SetAuthenticator(auth)
    }
    
    if config.EnableACL {
        acl := cluster.NewACLManager(config.ACLFile)
        security.SetACLManager(acl)
    }
    
    return security
}
```

## 🚀 Best Practices

### 1. Cluster Sizing

- **Small clusters** (< 1TB data): 3-5 nodes
- **Medium clusters** (1-10TB data): 5-15 nodes  
- **Large clusters** (> 10TB data): 15+ nodes
- **Replication factor**: 3 for production, 2 for dev/test

### 2. Hardware Recommendations

```yaml
Production Node Specifications:
  CPU: 8+ cores
  Memory: 32GB+ RAM
  Storage: SSD with 1000+ IOPS
  Network: 1Gbps+ bandwidth

Development Node Specifications:
  CPU: 4+ cores
  Memory: 16GB+ RAM
  Storage: SSD recommended
  Network: 100Mbps+ bandwidth
```

### 3. Configuration Tuning

```go
// Production-optimized configuration
func ProductionConfig() *cluster.Config {
    return &cluster.Config{
        // Aggressive failure detection
        HeartbeatTimeout:    50 * time.Millisecond,
        ElectionTimeout:     200 * time.Millisecond,
        
        // Large batch sizes for efficiency
        BatchSize:           500,
        MaxAppendEntries:    128,
        
        // Optimized I/O
        WALSyncInterval:     100 * time.Millisecond,
        SnapshotThreshold:   50000,
        
        // Memory management
        MaxMemoryUsage:      "80%",
        GCTargetPercentage:  100,
    }
}
```

### 4. Monitoring Checklist

- [ ] Node health and availability
- [ ] CPU and memory utilization
- [ ] Disk I/O and storage usage
- [ ] Network latency between nodes
- [ ] Query latency and throughput
- [ ] Replication lag
- [ ] Index size and growth rate
- [ ] Error rates and types

### 5. Backup and Disaster Recovery

```go
// Implement distributed backup strategy
type BackupManager struct {
    cluster   *cluster.Manager
    storage   BackupStorage  // S3, GCS, etc.
    scheduler *BackupScheduler
}

func (bm *BackupManager) CreateClusterBackup() error {
    // 1. Create consistent snapshot across all nodes
    snapshot := bm.cluster.CreateConsistentSnapshot()
    
    // 2. Upload to distributed storage
    return bm.storage.Upload(snapshot)
}

func (bm *BackupManager) RestoreFromBackup(backupID string) error {
    // 1. Download backup
    backup := bm.storage.Download(backupID)
    
    // 2. Coordinate restore across cluster
    return bm.cluster.RestoreFromSnapshot(backup)
}
```

## 🎯 Use Case Examples

### 1. Multi-Tenant SaaS Application

```go
// Implement tenant isolation
type MultiTenantCluster struct {
    cluster     *cluster.Manager
    tenantMap   map[string][]string // tenant -> node mapping
    isolation   TenantIsolation
}

func (mtc *MultiTenantCluster) CreateTenantCollection(tenantID string, config *api.CollectionConfig) error {
    // Assign tenant to specific nodes for isolation
    assignedNodes := mtc.tenantMap[tenantID]
    
    coordinator := cluster.NewQueryCoordinator(mtc.cluster)
    coordinator.SetTargetNodes(assignedNodes)
    
    return coordinator.CreateCollection(config)
}
```

### 2. Global CDN-like Vector Search

```go
// Implement geo-distributed vector search
type GeoDistributedSearch struct {
    regions map[string]*cluster.Manager
    router  *GeographicalRouter
}

func (gds *GeoDistributedSearch) SearchNearUser(userLocation Location, query *api.SearchRequest) ([]*api.SearchResult, error) {
    // Route to nearest region
    nearestRegion := gds.router.FindNearestRegion(userLocation)
    cluster := gds.regions[nearestRegion]
    
    coordinator := cluster.NewQueryCoordinator()
    return coordinator.Search(query)
}
```

### 3. Real-time ML Pipeline

```go
// Integrate with streaming data pipeline
type StreamingVectorPipeline struct {
    cluster     *cluster.Manager
    ingester    *StreamIngester
    processor   *VectorProcessor
}

func (svp *StreamingVectorPipeline) ProcessStream(stream <-chan RawData) {
    for data := range stream {
        // Process raw data to vectors
        vector := svp.processor.Process(data)
        
        // Insert into distributed cluster
        coordinator := cluster.NewQueryCoordinator(svp.cluster)
        coordinator.Insert(vector)
    }
}
```

## 📚 Additional Resources

- [Cluster Administration Guide](./CLUSTER_ADMIN.md)
- [Performance Tuning](./PERFORMANCE.md)
- [Security Guide](./SECURITY.md)
- [Troubleshooting](./TROUBLESHOOTING.md)
- [API Reference](https://pkg.go.dev/github.com/khambampati-subhash/govecdb)

---

*This guide covers the essential aspects of deploying GoVecDB in distributed environments. For specific deployment scenarios or advanced configurations, please refer to the detailed documentation or contact support.*