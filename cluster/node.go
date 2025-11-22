package cluster

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/hashicorp/raft"
	raftboltdb "github.com/hashicorp/raft-boltdb"
	pb "github.com/khambampati-subhash/govecdb/proto"
	"github.com/khambampati-subhash/govecdb/store"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Node represents a node in the distributed vector database
type Node struct {
	ID         string
	Address    string
	Region     string
	Zone       string
	Ring       *Ring
	Store      *store.Store
	Peers      map[string]PeerInfo         // ID -> PeerInfo
	PeerConns  map[string]*grpc.ClientConn // ID -> ClientConn
	mu         sync.RWMutex
	grpcServer *grpc.Server
	Raft       *raft.Raft
	RaftDir    string
}

// NewNode creates a new node
func NewNode(id, address, region, zone string, s *store.Store) *Node {
	if region == "" {
		region = DefaultRegion
	}
	n := &Node{
		ID:        id,
		Address:   address,
		Region:    region,
		Zone:      zone,
		Ring:      NewRing(3), // 3 replicas for consistent hashing
		Store:     s,
		Peers:     make(map[string]PeerInfo),
		PeerConns: make(map[string]*grpc.ClientConn),
		RaftDir:   fmt.Sprintf("raft_%s", id),
	}
	n.Ring.AddNode(id)
	return n
}

// SetupRaft initializes the Raft node
func (n *Node) SetupRaft(raftPort int, bootstrap bool) error {
	config := raft.DefaultConfig()
	config.LocalID = raft.ServerID(n.ID)

	// Setup Raft communication
	addr := fmt.Sprintf("localhost:%d", raftPort)
	transport, err := raft.NewTCPTransport(addr, nil, 3, 10*time.Second, os.Stderr)
	if err != nil {
		return err
	}

	// Create snapshot store
	snapshots, err := raft.NewFileSnapshotStore(n.RaftDir, 2, os.Stderr)
	if err != nil {
		return err
	}

	// Create log store and stable store
	if err := os.MkdirAll(n.RaftDir, 0700); err != nil {
		return err
	}

	boltDB, err := raftboltdb.NewBoltStore(filepath.Join(n.RaftDir, "raft.db"))
	if err != nil {
		return err
	}

	// Create FSM
	fsm := NewFSM(n.Ring)

	// Create Raft instance
	r, err := raft.NewRaft(config, fsm, boltDB, boltDB, snapshots, transport)
	if err != nil {
		return err
	}
	n.Raft = r

	if bootstrap {
		configuration := raft.Configuration{
			Servers: []raft.Server{
				{
					ID:      config.LocalID,
					Address: transport.LocalAddr(),
				},
			},
		}
		n.Raft.BootstrapCluster(configuration)
	}

	return nil
}

// Start starts the gRPC server
func (n *Node) Start() error {
	lis, err := net.Listen("tcp", n.Address)
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	n.grpcServer = grpc.NewServer()
	pb.RegisterVectorServiceServer(n.grpcServer, NewGRPCServer(n))

	go func() {
		if err := n.grpcServer.Serve(lis); err != nil {
			log.Printf("failed to serve: %v", err)
		}
	}()

	return nil
}

// Stop stops the node
func (n *Node) Stop() {
	if n.grpcServer != nil {
		n.grpcServer.Stop()
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	for _, conn := range n.PeerConns {
		conn.Close()
	}
}

// Join adds a peer to the cluster
func (n *Node) Join(peerID, peerAddr, peerRegion, peerZone string) error {
	// If we are the leader, we can add the voter
	if n.Raft != nil && n.Raft.State() == raft.Leader {
		return n.Raft.AddVoter(raft.ServerID(peerID), raft.ServerAddress(peerAddr), 0, 0).Error()
	}

	// Otherwise, we should forward to leader (omitted for brevity, assume we join via leader)
	// For this simplified implementation, we just update local state if not using Raft for everything yet

	n.mu.Lock()
	defer n.mu.Unlock()

	n.Peers[peerID] = PeerInfo{
		ID:      peerID,
		Address: peerAddr,
		Region:  peerRegion,
		Zone:    peerZone,
	}

	if n.Raft == nil {
		n.Ring.AddNode(peerID)
	}

	// Establish connection
	conn, err := grpc.NewClient(peerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("Failed to connect to peer %s: %v", peerID, err)
		return err
	}
	n.PeerConns[peerID] = conn
	return nil
}

// Put inserts a vector into the cluster
func (n *Node) Put(id string, vector []float32) error {
	// 1. Determine replicas
	replicas := n.Ring.GetNodes(id, 3) // Replication factor 3
	if len(replicas) == 0 {
		return fmt.Errorf("no nodes in ring")
	}

	primary := replicas[0]

	// 2. If we are not the primary, forward to primary
	if primary != n.ID {
		return n.forwardPut(primary, id, vector, false)
	}

	// 3. We are the primary. Coordinate replication.
	// We need to write to all replicas (including self).
	// For simplicity, we wait for all successful writes (W=N).

	type result struct {
		nodeID string
		err    error
	}

	results := make(chan result, len(replicas))
	var wg sync.WaitGroup

	for _, nodeID := range replicas {
		wg.Add(1)
		go func(nid string) {
			defer wg.Done()
			if nid == n.ID {
				// Local write
				err := n.Store.Insert(id, vector)
				results <- result{nodeID: nid, err: err}
			} else {
				// Remote replication
				err := n.forwardPut(nid, id, vector, true)
				results <- result{nodeID: nid, err: err}
			}
		}(nodeID)
	}

	wg.Wait()
	close(results)

	// Check for errors
	// For strong consistency, fail if any fail? Or if quorum fails?
	// Let's fail if any fail for now.
	for res := range results {
		if res.err != nil {
			return fmt.Errorf("replication failed on node %s: %w", res.nodeID, res.err)
		}
	}

	return nil
}

// forwardPut forwards a put request to another node
func (n *Node) forwardPut(targetNodeID, id string, vector []float32, isReplication bool) error {
	n.mu.RLock()
	conn, ok := n.PeerConns[targetNodeID]
	n.mu.RUnlock()

	if !ok {
		return fmt.Errorf("no connection to node %s", targetNodeID)
	}

	client := pb.NewVectorServiceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := client.Put(ctx, &pb.PutRequest{
		Vector: &pb.Vector{
			Id:   id,
			Data: vector,
		},
		IsReplication: isReplication,
	})
	if err != nil {
		return err
	}
	if !resp.Success {
		return fmt.Errorf("remote put failed: %s", resp.Error)
	}
	return nil
}

// Get retrieves a vector from the cluster
func (n *Node) Get(id string) ([]float32, error) {
	// 1. Determine responsible node
	targetNodeID := n.Ring.GetNode(id)

	// 2. If local, read from store
	if targetNodeID == n.ID {
		vec, err := n.Store.Index.Get(id) // Accessing internal index for now
		if err != nil {
			return nil, err
		}
		return vec.Data, nil
	}

	// 3. If remote, forward request
	n.mu.RLock()
	conn, ok := n.PeerConns[targetNodeID]
	n.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("no connection to node %s", targetNodeID)
	}

	client := pb.NewVectorServiceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := client.Get(ctx, &pb.GetRequest{Id: id})
	if err != nil {
		return nil, err
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("remote get failed: %s", resp.Error)
	}
	if resp.Vector == nil {
		return nil, fmt.Errorf("vector not found")
	}

	return resp.Vector.Data, nil
}

// DistributeData rebalances data when nodes join/leave (Placeholder)
func (n *Node) DistributeData() {
	// Iterate over local data and move keys that no longer belong to this node
}
