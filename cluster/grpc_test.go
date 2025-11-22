package cluster

import (
	"os"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/index"
	"github.com/khambampati-subhash/govecdb/store"
)

func TestGRPC_Forwarding(t *testing.T) {
	// Setup Node 1
	wal1 := "wal_node1.log"
	defer os.Remove(wal1)
	idx1, _ := index.NewHNSWIndex(index.DefaultConfig(4))
	s1, _ := store.NewStore(wal1, idx1)
	defer s1.Close()

	// Use a free port (0 lets OS choose, but we need to know it for Join)
	// For testing, we'll use fixed ports to avoid complexity, hoping they are free.
	addr1 := "localhost:50051"
	n1 := NewNode("node1", addr1, "us-east", "1a", s1)
	if err := n1.Start(); err != nil {
		t.Fatalf("Failed to start node1: %v", err)
	}
	defer n1.Stop()

	// Setup Node 2
	wal2 := "wal_node2.log"
	defer os.Remove(wal2)
	idx2, _ := index.NewHNSWIndex(index.DefaultConfig(4))
	s2, _ := store.NewStore(wal2, idx2)
	defer s2.Close()

	addr2 := "localhost:50052"
	n2 := NewNode("node2", addr2, "us-west", "1b", s2)
	if err := n2.Start(); err != nil {
		t.Fatalf("Failed to start node2: %v", err)
	}
	defer n2.Stop()

	// Setup Node 3
	wal3 := "wal_node3.log"
	defer os.Remove(wal3)
	idx3, _ := index.NewHNSWIndex(index.DefaultConfig(4))
	s3, _ := store.NewStore(wal3, idx3)
	defer s3.Close()

	addr3 := "localhost:50053"
	n3 := NewNode("node3", addr3, "eu-west", "1c", s3)
	if err := n3.Start(); err != nil {
		t.Fatalf("Failed to start node3: %v", err)
	}
	defer n3.Stop()

	// Join nodes
	// We need to wait a bit for servers to start
	time.Sleep(100 * time.Millisecond)

	n1.Join("node2", addr2, "us-west", "1b")
	n1.Join("node3", addr3, "eu-west", "1c")
	n2.Join("node1", addr1, "us-east", "1a")
	n2.Join("node3", addr3, "eu-west", "1c")
	n3.Join("node1", addr1, "us-east", "1a")
	n3.Join("node2", addr2, "us-west", "1b")

	// Put on Node 1 (should replicate to all 3 nodes)
	vec := []float32{1.0, 2.0, 3.0, 4.0}
	if err := n1.Put("vec1", vec); err != nil {
		t.Fatalf("Failed to put vector via node1: %v", err)
	}

	// Verify on Node 1
	got1, err := n1.Get("vec1")
	if err != nil {
		t.Fatalf("Failed to get vector from node1: %v", err)
	}
	if got1[0] != 1.0 {
		t.Errorf("Data mismatch on node1")
	}

	// Verify on Node 2
	got2, err := n2.Get("vec1")
	if err != nil {
		t.Fatalf("Failed to get vector from node2: %v", err)
	}
	if got2[0] != 1.0 {
		t.Errorf("Data mismatch on node2")
	}

	// Verify on Node 3
	got3, err := n3.Get("vec1")
	if err != nil {
		t.Fatalf("Failed to get vector from node3: %v", err)
	}
	if got3[0] != 1.0 {
		t.Errorf("Data mismatch on node3")
	}
}
