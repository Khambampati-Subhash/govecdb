package cluster

import (
	"os"
	"testing"

	"github.com/khambampati-subhash/govecdb/index"
	"github.com/khambampati-subhash/govecdb/store"
)

func TestNode_LocalOperations(t *testing.T) {
	walPath := "node_test_wal.log"
	defer os.Remove(walPath)

	// Setup Store
	cfg := index.DefaultConfig(4)
	idx, _ := index.NewHNSWIndex(cfg)
	s, err := store.NewStore(walPath, idx)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer s.Close()

	// Setup Node
	node := NewNode("node1", "localhost:8001", "us-east", "1a", s)

	// Test Put (Local)
	vec := []float32{0.1, 0.2, 0.3, 0.4}
	if err := node.Put("vec1", vec); err != nil {
		t.Fatalf("Failed to put vector: %v", err)
	}

	// Test Get (Local)
	got, err := node.Get("vec1")
	if err != nil {
		t.Fatalf("Failed to get vector: %v", err)
	}

	if len(got) != 4 || got[0] != 0.1 {
		t.Errorf("Vector mismatch")
	}
}
