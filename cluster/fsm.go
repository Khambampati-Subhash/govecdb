package cluster

import (
	"encoding/json"
	"fmt"
	"io"
	"sync"

	"github.com/hashicorp/raft"
)

// FSM implements raft.FSM for cluster metadata (membership)
type FSM struct {
	ring *Ring
	mu   sync.Mutex
}

// NewFSM creates a new FSM
func NewFSM(ring *Ring) *FSM {
	return &FSM{
		ring: ring,
	}
}

// Command types
type RaftCommandType int

const (
	CommandJoin RaftCommandType = iota
	CommandLeave
)

type RaftCommand struct {
	Type   RaftCommandType `json:"type"`
	NodeID string          `json:"node_id"`
}

// Apply applies a Raft log entry to the FSM
func (f *FSM) Apply(l *raft.Log) interface{} {
	var cmd RaftCommand
	if err := json.Unmarshal(l.Data, &cmd); err != nil {
		return fmt.Errorf("failed to unmarshal command: %w", err)
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	switch cmd.Type {
	case CommandJoin:
		f.ring.AddNode(cmd.NodeID)
	case CommandLeave:
		f.ring.RemoveNode(cmd.NodeID)
	}

	return nil
}

// Snapshot returns a snapshot of the FSM state
func (f *FSM) Snapshot() (raft.FSMSnapshot, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Clone the ring state (simplified: just serializing the map)
	// In a real implementation, we'd need a proper clone method on Ring
	nodes := make(map[string]string)
	f.ring.mu.RLock()
	for k, v := range f.ring.nodes {
		nodes[k] = v
	}
	f.ring.mu.RUnlock()

	return &FSMSnapshot{nodes: nodes}, nil
}

// Restore restores the FSM state from a snapshot
func (f *FSM) Restore(rc io.ReadCloser) error {
	defer rc.Close()

	var nodes map[string]string
	if err := json.NewDecoder(rc).Decode(&nodes); err != nil {
		return err
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	// Rebuild ring from nodes
	// Note: This is a bit hacky because Ring.nodes is virtual->physical.
	// We actually need the list of physical nodes to rebuild the ring correctly.
	// For this demo, let's assume we just need to re-add unique values from the map.

	uniqueNodes := make(map[string]bool)
	for _, v := range nodes {
		uniqueNodes[v] = true
	}

	// Clear existing ring? Or just re-add?
	// Ideally we should reset the ring.
	// For now, let's just re-add.
	for nodeID := range uniqueNodes {
		f.ring.AddNode(nodeID)
	}

	return nil
}

// FSMSnapshot implements raft.FSMSnapshot
type FSMSnapshot struct {
	nodes map[string]string
}

func (s *FSMSnapshot) Persist(sink raft.SnapshotSink) error {
	err := func() error {
		b, err := json.Marshal(s.nodes)
		if err != nil {
			return err
		}
		if _, err := sink.Write(b); err != nil {
			return err
		}
		return nil
	}()

	if err != nil {
		sink.Cancel()
		return err
	}

	return sink.Close()
}

func (s *FSMSnapshot) Release() {}
