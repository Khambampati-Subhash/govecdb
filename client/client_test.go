package client

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/cluster"
	"github.com/khambampati-subhash/govecdb/index"
	"github.com/khambampati-subhash/govecdb/store"
)

func TestClient_Integration(t *testing.T) {
	// Start a server node
	walPath := "client_test_wal.log"
	defer os.Remove(walPath)

	idx, _ := index.NewHNSWIndex(index.DefaultConfig(4))
	s, _ := store.NewStore(walPath, idx)
	defer s.Close()

	addr := "localhost:50055"
	node := cluster.NewNode("node1", addr, "us-east", "1a", s)
	if err := node.Start(); err != nil {
		t.Fatalf("Failed to start node: %v", err)
	}
	defer node.Stop()

	// Give server time to start
	time.Sleep(100 * time.Millisecond)

	// Connect client
	c, err := NewClient(addr)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	ctx := context.Background()

	// Test Put
	vec := []float32{0.1, 0.2, 0.3, 0.4}
	if err := c.Put(ctx, "vec1", vec); err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Test Get
	got, err := c.Get(ctx, "vec1")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if len(got) != 4 || got[0] != 0.1 {
		t.Errorf("Vector mismatch")
	}

	// Test Search
	results, err := c.Search(ctx, []float32{0.1, 0.2, 0.3, 0.4}, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) == 0 {
		t.Errorf("No results found")
	}
	if results[0].ID != "vec1" {
		t.Errorf("Expected vec1, got %s", results[0].ID)
	}
}
