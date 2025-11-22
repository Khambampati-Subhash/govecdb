package cluster

import (
	"testing"
)

func TestRing(t *testing.T) {
	r := NewRing(3)

	r.AddNode("node1")
	r.AddNode("node2")
	r.AddNode("node3")

	// Verify distribution
	counts := make(map[string]int)
	keys := []string{"key1", "key2", "key3", "key4", "key5", "key6", "key7", "key8", "key9", "key10"}

	for _, key := range keys {
		node := r.GetNode(key)
		counts[node]++
	}

	if len(counts) < 2 {
		t.Errorf("Poor distribution: %v", counts)
	}

	// node1 := r.GetNode("key1") // Unused
	r.AddNode("node4")
	node2 := r.GetNode("key1")

	// In consistent hashing, keys should mostly stay on the same node,
	// but "key1" might move. However, if we remove the node it was on, it MUST move.

	r.RemoveNode(node2)
	node3 := r.GetNode("key1")
	if node3 == node2 {
		t.Errorf("Key should have moved from removed node")
	}
}

func TestRing_GetNodes(t *testing.T) {
	r := NewRing(3)
	r.AddNode("node1")
	r.AddNode("node2")
	r.AddNode("node3")

	nodes := r.GetNodes("key1", 2)
	if len(nodes) != 2 {
		t.Errorf("Expected 2 nodes, got %d", len(nodes))
	}
	if nodes[0] == nodes[1] {
		t.Errorf("Nodes should be distinct")
	}
}
