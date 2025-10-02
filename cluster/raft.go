// Package cluster provides Raft consensus implementation.
package cluster

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// RaftState represents the state of a Raft node
type RaftState int

const (
	RaftStateFollower RaftState = iota
	RaftStateCandidate
	RaftStateLeader
)

// String returns the string representation of the Raft state
func (s RaftState) String() string {
	switch s {
	case RaftStateFollower:
		return "FOLLOWER"
	case RaftStateCandidate:
		return "CANDIDATE"
	case RaftStateLeader:
		return "LEADER"
	default:
		return "UNKNOWN"
	}
}

// RaftConsensusManager implements the ConsensusManager interface using Raft consensus
type RaftConsensusManager struct {
	nodeID    string
	peers     map[string]string // nodeID -> address
	transport Transport         // Network transport
	storage   Storage           // Persistent storage

	// Raft state
	mu          sync.RWMutex
	state       RaftState
	currentTerm uint64
	votedFor    string
	log         []*LogEntry
	commitIndex uint64
	lastApplied uint64

	// Leader state (reset on election)
	nextIndex  map[string]uint64 // nodeID -> next log index to send
	matchIndex map[string]uint64 // nodeID -> highest log index replicated

	// Election state
	electionTimeout  time.Duration
	heartbeatTimeout time.Duration
	lastHeartbeat    time.Time
	votes            map[string]bool // nodeID -> vote received

	// Channels
	proposalChan    chan *ProposalRequest
	commandChan     chan *Command
	appendEntryChan chan *AppendEntryRequest
	voteRequestChan chan *VoteRequest
	shutdownChan    chan struct{}

	// Background tasks
	electionTimer  *time.Timer
	heartbeatTimer *time.Timer

	// Statistics
	stats *RaftStats
}

// RaftStats contains Raft consensus statistics
type RaftStats struct {
	CurrentTerm        uint64    `json:"current_term"`
	LogLength          int       `json:"log_length"`
	CommitIndex        uint64    `json:"commit_index"`
	LastApplied        uint64    `json:"last_applied"`
	ElectionCount      int64     `json:"election_count"`
	HeartbeatsSent     int64     `json:"heartbeats_sent"`
	HeartbeatsRecv     int64     `json:"heartbeats_received"`
	ProposalsTotal     int64     `json:"proposals_total"`
	ProposalsCommitted int64     `json:"proposals_committed"`
	LastElection       time.Time `json:"last_election"`
	LastHeartbeat      time.Time `json:"last_heartbeat"`
}

// ProposalRequest represents a proposal request
type ProposalRequest struct {
	Proposal *Proposal
	Response chan *ProposalResult
}

// AppendEntryRequest represents an append entry RPC request
type AppendEntryRequest struct {
	Term         uint64      `json:"term"`
	LeaderID     string      `json:"leader_id"`
	PrevLogIndex uint64      `json:"prev_log_index"`
	PrevLogTerm  uint64      `json:"prev_log_term"`
	Entries      []*LogEntry `json:"entries"`
	LeaderCommit uint64      `json:"leader_commit"`
}

// AppendEntryResponse represents an append entry RPC response
type AppendEntryResponse struct {
	Term    uint64 `json:"term"`
	Success bool   `json:"success"`
	NodeID  string `json:"node_id"`
}

// VoteRequest represents a vote RPC request
type VoteRequest struct {
	Term         uint64 `json:"term"`
	CandidateID  string `json:"candidate_id"`
	LastLogIndex uint64 `json:"last_log_index"`
	LastLogTerm  uint64 `json:"last_log_term"`
}

// VoteResponse represents a vote RPC response
type VoteResponse struct {
	Term        uint64 `json:"term"`
	VoteGranted bool   `json:"vote_granted"`
	NodeID      string `json:"node_id"`
}

// Transport defines the interface for network communication
type Transport interface {
	SendAppendEntry(ctx context.Context, nodeID string, req *AppendEntryRequest) (*AppendEntryResponse, error)
	SendVoteRequest(ctx context.Context, nodeID string, req *VoteRequest) (*VoteResponse, error)
	Start(nodeID string, handler TransportHandler) error
	Stop() error
}

// TransportHandler defines callbacks for handling incoming RPCs
type TransportHandler interface {
	HandleAppendEntry(req *AppendEntryRequest) *AppendEntryResponse
	HandleVoteRequest(req *VoteRequest) *VoteResponse
}

// Storage defines the interface for persistent storage
type Storage interface {
	SaveState(term uint64, votedFor string) error
	LoadState() (uint64, string, error)
	SaveLog(entries []*LogEntry) error
	LoadLog() ([]*LogEntry, error)
	SaveSnapshot(snapshot []byte, lastIndex uint64, lastTerm uint64) error
	LoadSnapshot() ([]byte, uint64, uint64, error)
}

// NewRaftConsensusManager creates a new Raft consensus manager
func NewRaftConsensusManager(nodeID string, peers map[string]string, transport Transport, storage Storage) (*RaftConsensusManager, error) {
	if nodeID == "" {
		return nil, fmt.Errorf("node ID cannot be empty")
	}

	rcm := &RaftConsensusManager{
		nodeID:    nodeID,
		peers:     make(map[string]string),
		transport: transport,
		storage:   storage,

		state:       RaftStateFollower,
		currentTerm: 0,
		votedFor:    "",
		log:         make([]*LogEntry, 0),
		commitIndex: 0,
		lastApplied: 0,

		nextIndex:  make(map[string]uint64),
		matchIndex: make(map[string]uint64),

		electionTimeout:  randomElectionTimeout(),
		heartbeatTimeout: time.Millisecond * 50,
		votes:            make(map[string]bool),

		proposalChan:    make(chan *ProposalRequest, 100),
		commandChan:     make(chan *Command, 100),
		appendEntryChan: make(chan *AppendEntryRequest, 100),
		voteRequestChan: make(chan *VoteRequest, 100),
		shutdownChan:    make(chan struct{}),

		stats: &RaftStats{},
	}

	// Copy peers
	for id, addr := range peers {
		rcm.peers[id] = addr
	}

	return rcm, nil
}

// Start starts the Raft consensus manager
func (rcm *RaftConsensusManager) Start(ctx context.Context) error {
	rcm.mu.Lock()
	defer rcm.mu.Unlock()

	// Load persistent state
	if rcm.storage != nil {
		if term, votedFor, err := rcm.storage.LoadState(); err == nil {
			rcm.currentTerm = term
			rcm.votedFor = votedFor
		}

		if log, err := rcm.storage.LoadLog(); err == nil {
			rcm.log = log
		}
	}

	// Start transport
	if err := rcm.transport.Start(rcm.nodeID, rcm); err != nil {
		return fmt.Errorf("failed to start transport: %w", err)
	}

	// Start background routines
	go rcm.consensusLoop()
	go rcm.applyLoop()

	// Start election timer
	rcm.resetElectionTimer()

	return nil
}

// Stop stops the Raft consensus manager
func (rcm *RaftConsensusManager) Stop(ctx context.Context) error {
	close(rcm.shutdownChan)

	if rcm.electionTimer != nil {
		rcm.electionTimer.Stop()
	}
	if rcm.heartbeatTimer != nil {
		rcm.heartbeatTimer.Stop()
	}

	return rcm.transport.Stop()
}

// HealthCheck performs a health check
func (rcm *RaftConsensusManager) HealthCheck(ctx context.Context) error {
	rcm.mu.RLock()
	defer rcm.mu.RUnlock()

	// Check if we've received heartbeats recently (if follower)
	if rcm.state == RaftStateFollower {
		if time.Since(rcm.lastHeartbeat) > rcm.electionTimeout*2 {
			return fmt.Errorf("no heartbeats received recently")
		}
	}

	return nil
}

// Propose proposes a new command to the cluster
func (rcm *RaftConsensusManager) Propose(ctx context.Context, proposal *Proposal) (*ProposalResult, error) {
	if rcm.state != RaftStateLeader {
		return &ProposalResult{
			ProposalID: proposal.ID,
			Accepted:   false,
			Error:      "not leader",
		}, nil
	}

	// Create proposal request
	req := &ProposalRequest{
		Proposal: proposal,
		Response: make(chan *ProposalResult, 1),
	}

	// Send to proposal channel
	select {
	case rcm.proposalChan <- req:
		rcm.stats.ProposalsTotal++
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-rcm.shutdownChan:
		return nil, fmt.Errorf("consensus manager shutting down")
	}

	// Wait for response
	select {
	case result := <-req.Response:
		return result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-rcm.shutdownChan:
		return nil, fmt.Errorf("consensus manager shutting down")
	}
}

// Vote votes on a proposal (not implemented in basic Raft)
func (rcm *RaftConsensusManager) Vote(ctx context.Context, proposalID string, vote Vote) error {
	return fmt.Errorf("voting not implemented in basic Raft")
}

// IsLeader returns true if this node is the leader
func (rcm *RaftConsensusManager) IsLeader() bool {
	rcm.mu.RLock()
	defer rcm.mu.RUnlock()
	return rcm.state == RaftStateLeader
}

// GetLeader returns information about the current leader
func (rcm *RaftConsensusManager) GetLeader() (*NodeInfo, error) {
	rcm.mu.RLock()
	defer rcm.mu.RUnlock()

	if rcm.state == RaftStateLeader {
		return &NodeInfo{ID: rcm.nodeID}, nil
	}

	// TODO: Track leader ID from append entries
	return nil, fmt.Errorf("leader unknown")
}

// GetState returns the current consensus state
func (rcm *RaftConsensusManager) GetState(ctx context.Context) (*ConsensusState, error) {
	rcm.mu.RLock()
	defer rcm.mu.RUnlock()

	// Build commitments map
	commitments := make(map[string]uint64)
	for nodeID := range rcm.peers {
		commitments[nodeID] = rcm.matchIndex[nodeID]
	}
	commitments[rcm.nodeID] = rcm.commitIndex

	state := &ConsensusState{
		Term:        rcm.currentTerm,
		Leader:      "",
		Commitments: commitments,
		LastApplied: rcm.lastApplied,
		Log:         make([]*LogEntry, len(rcm.log)),
	}

	if rcm.state == RaftStateLeader {
		state.Leader = rcm.nodeID
	}

	// Copy log entries
	copy(state.Log, rcm.log)

	return state, nil
}

// ApplyCommand applies a command to the state machine
func (rcm *RaftConsensusManager) ApplyCommand(ctx context.Context, command *Command) error {
	select {
	case rcm.commandChan <- command:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-rcm.shutdownChan:
		return fmt.Errorf("consensus manager shutting down")
	}
}

// Transport handler methods

// HandleAppendEntry handles incoming append entry requests
func (rcm *RaftConsensusManager) HandleAppendEntry(req *AppendEntryRequest) *AppendEntryResponse {
	rcm.mu.Lock()
	defer rcm.mu.Unlock()

	rcm.stats.HeartbeatsRecv++
	rcm.lastHeartbeat = time.Now()

	// Reply false if term < currentTerm
	if req.Term < rcm.currentTerm {
		return &AppendEntryResponse{
			Term:    rcm.currentTerm,
			Success: false,
			NodeID:  rcm.nodeID,
		}
	}

	// If term > currentTerm, update term and become follower
	if req.Term > rcm.currentTerm {
		rcm.currentTerm = req.Term
		rcm.votedFor = ""
		rcm.state = RaftStateFollower
		rcm.saveState()
	}

	// Reset election timer
	rcm.resetElectionTimer()

	// Reply false if log doesn't contain entry at prevLogIndex with prevLogTerm
	if req.PrevLogIndex > 0 {
		if len(rcm.log) < int(req.PrevLogIndex) {
			return &AppendEntryResponse{
				Term:    rcm.currentTerm,
				Success: false,
				NodeID:  rcm.nodeID,
			}
		}

		if len(rcm.log) >= int(req.PrevLogIndex) && rcm.log[req.PrevLogIndex-1].Term != req.PrevLogTerm {
			return &AppendEntryResponse{
				Term:    rcm.currentTerm,
				Success: false,
				NodeID:  rcm.nodeID,
			}
		}
	}

	// Append new entries
	if len(req.Entries) > 0 {
		// Delete conflicting entries and append new ones
		if len(rcm.log) > int(req.PrevLogIndex) {
			rcm.log = rcm.log[:req.PrevLogIndex]
		}

		rcm.log = append(rcm.log, req.Entries...)
		rcm.saveLog()
	}

	// Update commit index
	if req.LeaderCommit > rcm.commitIndex {
		logLen := uint64(len(rcm.log))
		if req.LeaderCommit < logLen {
			rcm.commitIndex = req.LeaderCommit
		} else {
			rcm.commitIndex = logLen
		}
	}

	return &AppendEntryResponse{
		Term:    rcm.currentTerm,
		Success: true,
		NodeID:  rcm.nodeID,
	}
}

// HandleVoteRequest handles incoming vote requests
func (rcm *RaftConsensusManager) HandleVoteRequest(req *VoteRequest) *VoteResponse {
	rcm.mu.Lock()
	defer rcm.mu.Unlock()

	// Reply false if term < currentTerm
	if req.Term < rcm.currentTerm {
		return &VoteResponse{
			Term:        rcm.currentTerm,
			VoteGranted: false,
			NodeID:      rcm.nodeID,
		}
	}

	// If term > currentTerm, update term and become follower
	if req.Term > rcm.currentTerm {
		rcm.currentTerm = req.Term
		rcm.votedFor = ""
		rcm.state = RaftStateFollower
		rcm.saveState()
	}

	// Vote for candidate if we haven't voted or voted for them already
	voteGranted := false
	if (rcm.votedFor == "" || rcm.votedFor == req.CandidateID) && rcm.isLogUpToDate(req.LastLogIndex, req.LastLogTerm) {
		rcm.votedFor = req.CandidateID
		voteGranted = true
		rcm.saveState()
		rcm.resetElectionTimer()
	}

	return &VoteResponse{
		Term:        rcm.currentTerm,
		VoteGranted: voteGranted,
		NodeID:      rcm.nodeID,
	}
}

// Private methods

func (rcm *RaftConsensusManager) consensusLoop() {
	for {
		select {
		case <-rcm.shutdownChan:
			return
		default:
			switch rcm.state {
			case RaftStateFollower:
				rcm.runFollower()
			case RaftStateCandidate:
				rcm.runCandidate()
			case RaftStateLeader:
				rcm.runLeader()
			}
		}
	}
}

func (rcm *RaftConsensusManager) runFollower() {
	select {
	case <-rcm.shutdownChan:
		return
	case <-rcm.electionTimer.C:
		rcm.mu.Lock()
		rcm.state = RaftStateCandidate
		rcm.mu.Unlock()
	case req := <-rcm.proposalChan:
		// Reject proposals as follower
		req.Response <- &ProposalResult{
			ProposalID: req.Proposal.ID,
			Accepted:   false,
			Error:      "not leader",
		}
	}
}

func (rcm *RaftConsensusManager) runCandidate() {
	rcm.mu.Lock()

	// Start election
	rcm.currentTerm++
	rcm.votedFor = rcm.nodeID
	rcm.votes = make(map[string]bool)
	rcm.votes[rcm.nodeID] = true
	rcm.stats.ElectionCount++
	rcm.stats.LastElection = time.Now()

	rcm.saveState()
	rcm.resetElectionTimer()

	term := rcm.currentTerm
	lastLogIndex := uint64(len(rcm.log))
	var lastLogTerm uint64
	if lastLogIndex > 0 {
		lastLogTerm = rcm.log[lastLogIndex-1].Term
	}

	rcm.mu.Unlock()

	// Send vote requests to all peers
	for nodeID := range rcm.peers {
		go func(id string) {
			req := &VoteRequest{
				Term:         term,
				CandidateID:  rcm.nodeID,
				LastLogIndex: lastLogIndex,
				LastLogTerm:  lastLogTerm,
			}

			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()

			resp, err := rcm.transport.SendVoteRequest(ctx, id, req)
			if err == nil && resp != nil {
				rcm.handleVoteResponse(resp)
			}
		}(nodeID)
	}

	// Wait for election timeout or becoming leader
	select {
	case <-rcm.shutdownChan:
		return
	case <-rcm.electionTimer.C:
		rcm.mu.Lock()
		if rcm.state == RaftStateCandidate {
			// Election timeout, start new election
			rcm.resetElectionTimer()
		}
		rcm.mu.Unlock()
	case <-time.After(rcm.electionTimeout):
		// Continue to next election
	}
}

func (rcm *RaftConsensusManager) runLeader() {
	rcm.mu.Lock()

	// Initialize leader state
	nextIndex := uint64(len(rcm.log)) + 1
	for nodeID := range rcm.peers {
		rcm.nextIndex[nodeID] = nextIndex
		rcm.matchIndex[nodeID] = 0
	}

	rcm.mu.Unlock()

	// Send initial heartbeats
	rcm.sendHeartbeats()

	// Start heartbeat timer
	rcm.heartbeatTimer = time.NewTimer(rcm.heartbeatTimeout)
	defer rcm.heartbeatTimer.Stop()

	for {
		select {
		case <-rcm.shutdownChan:
			return
		case <-rcm.heartbeatTimer.C:
			rcm.sendHeartbeats()
			rcm.heartbeatTimer.Reset(rcm.heartbeatTimeout)
		case req := <-rcm.proposalChan:
			rcm.handleProposal(req)
		}

		// Check if still leader
		rcm.mu.RLock()
		if rcm.state != RaftStateLeader {
			rcm.mu.RUnlock()
			return
		}
		rcm.mu.RUnlock()
	}
}

func (rcm *RaftConsensusManager) applyLoop() {
	for {
		select {
		case <-rcm.shutdownChan:
			return
		case command := <-rcm.commandChan:
			// Apply command to state machine
			rcm.mu.Lock()
			if rcm.lastApplied < rcm.commitIndex {
				// Apply committed log entries
				for i := rcm.lastApplied; i < rcm.commitIndex; i++ {
					// Apply log entry i+1 (log is 0-indexed, but indices are 1-indexed)
					if int(i) < len(rcm.log) {
						// Apply the command from the log entry
						// This would be application-specific
					}
				}
				rcm.lastApplied = rcm.commitIndex
			}
			rcm.mu.Unlock()

			// Handle external command
			_ = command // Process the command
		}
	}
}

func (rcm *RaftConsensusManager) handleVoteResponse(resp *VoteResponse) {
	rcm.mu.Lock()
	defer rcm.mu.Unlock()

	if rcm.state != RaftStateCandidate || resp.Term != rcm.currentTerm {
		return
	}

	if resp.Term > rcm.currentTerm {
		rcm.currentTerm = resp.Term
		rcm.state = RaftStateFollower
		rcm.votedFor = ""
		rcm.saveState()
		return
	}

	if resp.VoteGranted {
		rcm.votes[resp.NodeID] = true

		// Check if we have majority
		if len(rcm.votes) > (len(rcm.peers)+1)/2 {
			rcm.state = RaftStateLeader
			rcm.resetElectionTimer()
		}
	}
}

func (rcm *RaftConsensusManager) handleProposal(req *ProposalRequest) {
	rcm.mu.Lock()
	defer rcm.mu.Unlock()

	if rcm.state != RaftStateLeader {
		req.Response <- &ProposalResult{
			ProposalID: req.Proposal.ID,
			Accepted:   false,
			Error:      "not leader",
		}
		return
	}

	// Create log entry
	entry := &LogEntry{
		Index:     uint64(len(rcm.log)) + 1,
		Term:      rcm.currentTerm,
		Command:   &Command{Type: CommandTypeConfigUpdate, Data: req.Proposal.Data},
		Timestamp: time.Now(),
	}

	// Append to log
	rcm.log = append(rcm.log, entry)
	rcm.saveLog()

	// Send append entries to replicate
	go rcm.replicateEntry(req, entry)
}

func (rcm *RaftConsensusManager) replicateEntry(req *ProposalRequest, entry *LogEntry) {
	// Send to all followers
	successCount := 1 // Count self
	totalNodes := len(rcm.peers) + 1
	majority := totalNodes/2 + 1

	var wg sync.WaitGroup
	var mu sync.Mutex

	for nodeID := range rcm.peers {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()

			if rcm.sendAppendEntry(id, []*LogEntry{entry}) {
				mu.Lock()
				successCount++
				mu.Unlock()
			}
		}(nodeID)
	}

	wg.Wait()

	// Check if replicated to majority
	result := &ProposalResult{
		ProposalID: req.Proposal.ID,
		Accepted:   successCount >= majority,
	}

	if result.Accepted {
		rcm.mu.Lock()
		rcm.commitIndex = entry.Index
		rcm.stats.ProposalsCommitted++
		rcm.mu.Unlock()
	}

	req.Response <- result
}

func (rcm *RaftConsensusManager) sendHeartbeats() {
	rcm.mu.RLock()
	peers := make([]string, 0, len(rcm.peers))
	for nodeID := range rcm.peers {
		peers = append(peers, nodeID)
	}
	rcm.mu.RUnlock()

	for _, nodeID := range peers {
		go func(id string) {
			rcm.sendAppendEntry(id, []*LogEntry{})
		}(nodeID)
	}

	rcm.stats.HeartbeatsSent += int64(len(peers))
	rcm.stats.LastHeartbeat = time.Now()
}

func (rcm *RaftConsensusManager) sendAppendEntry(nodeID string, entries []*LogEntry) bool {
	rcm.mu.RLock()
	nextIndex := rcm.nextIndex[nodeID]

	prevLogIndex := nextIndex - 1
	var prevLogTerm uint64
	if prevLogIndex > 0 && int(prevLogIndex-1) < len(rcm.log) {
		prevLogTerm = rcm.log[prevLogIndex-1].Term
	}

	req := &AppendEntryRequest{
		Term:         rcm.currentTerm,
		LeaderID:     rcm.nodeID,
		PrevLogIndex: prevLogIndex,
		PrevLogTerm:  prevLogTerm,
		Entries:      entries,
		LeaderCommit: rcm.commitIndex,
	}
	rcm.mu.RUnlock()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	resp, err := rcm.transport.SendAppendEntry(ctx, nodeID, req)
	if err != nil {
		return false
	}

	rcm.mu.Lock()
	defer rcm.mu.Unlock()

	if resp.Term > rcm.currentTerm {
		rcm.currentTerm = resp.Term
		rcm.state = RaftStateFollower
		rcm.votedFor = ""
		rcm.saveState()
		return false
	}

	if resp.Success {
		rcm.nextIndex[nodeID] = nextIndex + uint64(len(entries))
		rcm.matchIndex[nodeID] = nextIndex + uint64(len(entries)) - 1
	} else {
		// Decrement nextIndex and retry
		if rcm.nextIndex[nodeID] > 1 {
			rcm.nextIndex[nodeID]--
		}
	}

	return resp.Success
}

func (rcm *RaftConsensusManager) resetElectionTimer() {
	if rcm.electionTimer != nil {
		rcm.electionTimer.Stop()
	}
	rcm.electionTimeout = randomElectionTimeout()
	rcm.electionTimer = time.NewTimer(rcm.electionTimeout)
}

func (rcm *RaftConsensusManager) isLogUpToDate(lastLogIndex, lastLogTerm uint64) bool {
	if len(rcm.log) == 0 {
		return true
	}

	ourLastLogIndex := uint64(len(rcm.log))
	ourLastLogTerm := rcm.log[ourLastLogIndex-1].Term

	// Candidate's log is up-to-date if:
	// 1. Their last log entry has higher term, or
	// 2. Same term but equal or longer log
	return lastLogTerm > ourLastLogTerm ||
		(lastLogTerm == ourLastLogTerm && lastLogIndex >= ourLastLogIndex)
}

func (rcm *RaftConsensusManager) saveState() {
	if rcm.storage != nil {
		rcm.storage.SaveState(rcm.currentTerm, rcm.votedFor)
	}
}

func (rcm *RaftConsensusManager) saveLog() {
	if rcm.storage != nil {
		rcm.storage.SaveLog(rcm.log)
	}
}

// GetStats returns Raft statistics
func (rcm *RaftConsensusManager) GetStats() *RaftStats {
	rcm.mu.RLock()
	defer rcm.mu.RUnlock()

	stats := *rcm.stats
	stats.CurrentTerm = rcm.currentTerm
	stats.LogLength = len(rcm.log)
	stats.CommitIndex = rcm.commitIndex
	stats.LastApplied = rcm.lastApplied

	return &stats
}

// Helper functions

func randomElectionTimeout() time.Duration {
	// Random timeout between 150-300ms
	min := 150
	max := 300
	timeout := rand.Intn(max-min) + min
	return time.Duration(timeout) * time.Millisecond
}
