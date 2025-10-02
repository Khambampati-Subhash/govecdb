// Package cluster provides network communication layer for distributed operations.
package cluster

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// NetworkManagerImpl implements the NetworkManager interface using HTTP
type NetworkManagerImpl struct {
	nodeID      string
	bindAddress string
	server      *http.Server
	client      *http.Client
	connections map[string]*ConnectionPool
	mu          sync.RWMutex

	// Handlers
	clusterManager ClusterManager

	// Configuration
	timeout        time.Duration
	maxConnections int
	maxRetries     int
	retryDelay     time.Duration

	// Statistics
	stats *NetworkStats
}

// NetworkStats contains network layer statistics
type NetworkStats struct {
	RequestsSent      int64   `json:"requests_sent"`
	RequestsReceived  int64   `json:"requests_received"`
	RequestsFailed    int64   `json:"requests_failed"`
	BytesSent         int64   `json:"bytes_sent"`
	BytesReceived     int64   `json:"bytes_received"`
	ConnectionsActive int     `json:"connections_active"`
	ConnectionsTotal  int64   `json:"connections_total"`
	AvgLatency        float64 `json:"avg_latency_ms"`
}

// ConnectionPool manages HTTP connections to a specific node
type ConnectionPool struct {
	Address    string
	Client     *http.Client
	Mu         sync.RWMutex
	LastUsed   time.Time
	TotalReqs  int64
	FailedReqs int64
}

// NewNetworkManager creates a new network manager
func NewNetworkManager(nodeID, bindAddress string, clusterManager ClusterManager) *NetworkManagerImpl {
	return &NetworkManagerImpl{
		nodeID:      nodeID,
		bindAddress: bindAddress,
		client: &http.Client{
			Timeout: time.Second * 30,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     time.Second * 90,
				DisableKeepAlives:   false,
			},
		},
		connections:    make(map[string]*ConnectionPool),
		clusterManager: clusterManager,

		timeout:        time.Second * 30,
		maxConnections: 100,
		maxRetries:     3,
		retryDelay:     time.Millisecond * 100,

		stats: &NetworkStats{},
	}
}

// Start starts the network manager
func (nm *NetworkManagerImpl) Start(ctx context.Context) error {
	nm.mu.Lock()
	defer nm.mu.Unlock()

	if nm.server != nil {
		return fmt.Errorf("network manager already started")
	}

	// Create HTTP server
	mux := http.NewServeMux()
	nm.setupRoutes(mux)

	nm.server = &http.Server{
		Addr:         nm.bindAddress,
		Handler:      mux,
		ReadTimeout:  time.Second * 30,
		WriteTimeout: time.Second * 30,
		IdleTimeout:  time.Second * 120,
	}

	// Start server in background
	go func() {
		if err := nm.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Printf("HTTP server error: %v\n", err)
		}
	}()

	// Wait for server to be ready
	time.Sleep(time.Millisecond * 100)

	return nil
}

// Stop stops the network manager
func (nm *NetworkManagerImpl) Stop(ctx context.Context) error {
	nm.mu.Lock()
	defer nm.mu.Unlock()

	if nm.server == nil {
		return nil
	}

	// Shutdown HTTP server
	if err := nm.server.Shutdown(ctx); err != nil {
		return fmt.Errorf("failed to shutdown HTTP server: %w", err)
	}

	nm.server = nil
	return nil
}

// SendHeartbeat sends a heartbeat to another node
func (nm *NetworkManagerImpl) SendHeartbeat(ctx context.Context, address string, node *NodeInfo) error {
	url := fmt.Sprintf("http://%s/cluster/heartbeat", address)

	payload, err := json.Marshal(node)
	if err != nil {
		return fmt.Errorf("failed to marshal heartbeat: %w", err)
	}

	resp, err := nm.sendRequest(ctx, "POST", url, payload)
	if err != nil {
		nm.stats.RequestsFailed++
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		nm.stats.RequestsFailed++
		return fmt.Errorf("heartbeat failed with status: %d", resp.StatusCode)
	}

	nm.stats.RequestsSent++
	return nil
}

// JoinRequest sends a join request to a seed node
func (nm *NetworkManagerImpl) JoinRequest(ctx context.Context, address string, node *NodeInfo) error {
	url := fmt.Sprintf("http://%s/cluster/join", address)

	payload, err := json.Marshal(node)
	if err != nil {
		return fmt.Errorf("failed to marshal join request: %w", err)
	}

	resp, err := nm.sendRequest(ctx, "POST", url, payload)
	if err != nil {
		nm.stats.RequestsFailed++
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		nm.stats.RequestsFailed++
		return fmt.Errorf("join request failed with status: %d", resp.StatusCode)
	}

	nm.stats.RequestsSent++
	return nil
}

// NodeLeaveNotification sends a node leave notification
func (nm *NetworkManagerImpl) NodeLeaveNotification(ctx context.Context, address string, nodeID string) error {
	url := fmt.Sprintf("http://%s/cluster/leave", address)

	payload := map[string]string{"node_id": nodeID}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal leave notification: %w", err)
	}

	resp, err := nm.sendRequest(ctx, "POST", url, payloadBytes)
	if err != nil {
		nm.stats.RequestsFailed++
		return err
	}
	defer resp.Body.Close()

	nm.stats.RequestsSent++
	return nil
}

// CreateShardRequest sends a create shard request
func (nm *NetworkManagerImpl) CreateShardRequest(ctx context.Context, address string, shard *ShardInfo) error {
	url := fmt.Sprintf("http://%s/cluster/shard/create", address)

	payload, err := json.Marshal(shard)
	if err != nil {
		return fmt.Errorf("failed to marshal shard create request: %w", err)
	}

	resp, err := nm.sendRequest(ctx, "POST", url, payload)
	if err != nil {
		nm.stats.RequestsFailed++
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		nm.stats.RequestsFailed++
		return fmt.Errorf("shard create request failed with status: %d", resp.StatusCode)
	}

	nm.stats.RequestsSent++
	return nil
}

// SendQuery sends a distributed query to another node
func (nm *NetworkManagerImpl) SendQuery(ctx context.Context, address string, req *api.SearchRequest) ([]*api.SearchResult, error) {
	url := fmt.Sprintf("http://%s/query/search", address)

	payload, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal search request: %w", err)
	}

	resp, err := nm.sendRequest(ctx, "POST", url, payload)
	if err != nil {
		nm.stats.RequestsFailed++
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		nm.stats.RequestsFailed++
		return nil, fmt.Errorf("search request failed with status: %d", resp.StatusCode)
	}

	var results []*api.SearchResult
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		return nil, fmt.Errorf("failed to decode search results: %w", err)
	}

	nm.stats.RequestsSent++
	return results, nil
}

// SendBatchQuery sends a batch query to another node
func (nm *NetworkManagerImpl) SendBatchQuery(ctx context.Context, address string, req *api.BatchRequest) (*api.BatchResponse, error) {
	url := fmt.Sprintf("http://%s/query/batch", address)

	payload, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal batch request: %w", err)
	}

	resp, err := nm.sendRequest(ctx, "POST", url, payload)
	if err != nil {
		nm.stats.RequestsFailed++
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		nm.stats.RequestsFailed++
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("batch request failed with status: %d, body: %s", resp.StatusCode, string(body))
	}

	var response api.BatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode batch response: %w", err)
	}

	nm.stats.RequestsSent++
	return &response, nil
}

// HealthCheck performs a network health check
func (nm *NetworkManagerImpl) HealthCheck(ctx context.Context) error {
	// Check if server is running
	nm.mu.RLock()
	server := nm.server
	nm.mu.RUnlock()

	if server == nil {
		return fmt.Errorf("HTTP server not running")
	}

	// Test connection to self
	url := fmt.Sprintf("http://%s/health", nm.bindAddress)
	ctx, cancel := context.WithTimeout(ctx, time.Second*5)
	defer cancel()

	resp, err := nm.sendRequest(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned status: %d", resp.StatusCode)
	}

	return nil
}

// GetStats returns network statistics
func (nm *NetworkManagerImpl) GetStats() *NetworkStats {
	nm.mu.RLock()
	defer nm.mu.RUnlock()

	// Count active connections
	activeConnections := 0
	for _, pool := range nm.connections {
		if time.Since(pool.LastUsed) < time.Minute*5 {
			activeConnections++
		}
	}

	stats := *nm.stats
	stats.ConnectionsActive = activeConnections
	return &stats
}

// Private methods

func (nm *NetworkManagerImpl) sendRequest(ctx context.Context, method, url string, payload []byte) (*http.Response, error) {
	var req *http.Request
	var err error

	if payload != nil {
		req, err = http.NewRequestWithContext(ctx, method, url, nil)
		req.Header.Set("Content-Type", "application/json")
	} else {
		req, err = http.NewRequestWithContext(ctx, method, url, nil)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	startTime := time.Now()
	resp, err := nm.client.Do(req)
	latency := time.Since(startTime)

	// Update average latency
	nm.updateAvgLatency(latency.Seconds() * 1000) // Convert to milliseconds

	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if payload != nil {
		nm.stats.BytesSent += int64(len(payload))
	}

	return resp, nil
}

func (nm *NetworkManagerImpl) updateAvgLatency(latencyMs float64) {
	nm.mu.Lock()
	defer nm.mu.Unlock()

	// Simple moving average
	alpha := 0.1
	nm.stats.AvgLatency = alpha*latencyMs + (1-alpha)*nm.stats.AvgLatency
}

func (nm *NetworkManagerImpl) setupRoutes(mux *http.ServeMux) {
	// Health endpoint
	mux.HandleFunc("/health", nm.handleHealth)

	// Cluster management endpoints
	mux.HandleFunc("/cluster/heartbeat", nm.handleHeartbeat)
	mux.HandleFunc("/cluster/join", nm.handleJoin)
	mux.HandleFunc("/cluster/leave", nm.handleLeave)
	mux.HandleFunc("/cluster/shard/create", nm.handleShardCreate)

	// Query endpoints
	mux.HandleFunc("/query/search", nm.handleSearch)
	mux.HandleFunc("/query/batch", nm.handleBatch)

	// Raft consensus endpoints
	mux.HandleFunc("/consensus/append", nm.handleAppendEntry)
	mux.HandleFunc("/consensus/vote", nm.handleVoteRequest)
}

func (nm *NetworkManagerImpl) handleHealth(w http.ResponseWriter, r *http.Request) {
	nm.stats.RequestsReceived++

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	health := map[string]interface{}{
		"status":    "healthy",
		"node_id":   nm.nodeID,
		"timestamp": time.Now(),
	}

	json.NewEncoder(w).Encode(health)
}

func (nm *NetworkManagerImpl) handleHeartbeat(w http.ResponseWriter, r *http.Request) {
	nm.stats.RequestsReceived++

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var node NodeInfo
	if err := json.NewDecoder(r.Body).Decode(&node); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Handle heartbeat through cluster manager
	if nm.clusterManager != nil {
		if nodeManager, ok := nm.clusterManager.(NodeManager); ok {
			if err := nodeManager.HandleHeartbeat(r.Context(), &node); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
	}

	w.WriteHeader(http.StatusOK)
}

func (nm *NetworkManagerImpl) handleJoin(w http.ResponseWriter, r *http.Request) {
	nm.stats.RequestsReceived++

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var node NodeInfo
	if err := json.NewDecoder(r.Body).Decode(&node); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Handle join through cluster manager
	if nm.clusterManager != nil {
		if nodeManager, ok := nm.clusterManager.(NodeManager); ok {
			if err := nodeManager.AddNode(r.Context(), &node); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
	}

	w.WriteHeader(http.StatusOK)
}

func (nm *NetworkManagerImpl) handleLeave(w http.ResponseWriter, r *http.Request) {
	nm.stats.RequestsReceived++

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req map[string]string
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	nodeID, exists := req["node_id"]
	if !exists {
		http.Error(w, "Missing node_id", http.StatusBadRequest)
		return
	}

	// Handle leave through cluster manager
	if nm.clusterManager != nil {
		if nodeManager, ok := nm.clusterManager.(NodeManager); ok {
			if err := nodeManager.RemoveNode(r.Context(), nodeID); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
	}

	w.WriteHeader(http.StatusOK)
}

func (nm *NetworkManagerImpl) handleShardCreate(w http.ResponseWriter, r *http.Request) {
	nm.stats.RequestsReceived++

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var shard ShardInfo
	if err := json.NewDecoder(r.Body).Decode(&shard); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Handle shard creation
	// This would create the shard locally on this node
	// Implementation depends on the storage layer

	w.WriteHeader(http.StatusOK)
}

func (nm *NetworkManagerImpl) handleSearch(w http.ResponseWriter, r *http.Request) {
	nm.stats.RequestsReceived++

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req api.SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// This would be handled by the query coordinator
	// For now, return empty results
	results := []*api.SearchResult{}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

func (nm *NetworkManagerImpl) handleBatch(w http.ResponseWriter, r *http.Request) {
	nm.stats.RequestsReceived++

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req api.BatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// This would be handled by the query coordinator
	// For now, return empty response
	response := &api.BatchResponse{
		Results: make([]api.BatchResult, len(req.Operations)),
		Success: true,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (nm *NetworkManagerImpl) handleAppendEntry(w http.ResponseWriter, r *http.Request) {
	nm.stats.RequestsReceived++

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req AppendEntryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// This would be handled by the Raft consensus manager
	// For now, return a default response
	response := &AppendEntryResponse{
		Term:    req.Term,
		Success: true,
		NodeID:  nm.nodeID,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (nm *NetworkManagerImpl) handleVoteRequest(w http.ResponseWriter, r *http.Request) {
	nm.stats.RequestsReceived++

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req VoteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// This would be handled by the Raft consensus manager
	// For now, return a default response
	response := &VoteResponse{
		Term:        req.Term,
		VoteGranted: true,
		NodeID:      nm.nodeID,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// HTTPTransport implements the Transport interface for Raft over HTTP
type HTTPTransport struct {
	nodeID  string
	client  *http.Client
	server  *http.Server
	handler TransportHandler
	mu      sync.RWMutex
	address string
}

// NewHTTPTransport creates a new HTTP transport for Raft
func NewHTTPTransport(address string) *HTTPTransport {
	return &HTTPTransport{
		address: address,
		client: &http.Client{
			Timeout: time.Second * 5,
		},
	}
}

// Start starts the HTTP transport
func (ht *HTTPTransport) Start(nodeID string, handler TransportHandler) error {
	ht.mu.Lock()
	defer ht.mu.Unlock()

	ht.nodeID = nodeID
	ht.handler = handler

	mux := http.NewServeMux()
	mux.HandleFunc("/raft/append", ht.handleAppendEntry)
	mux.HandleFunc("/raft/vote", ht.handleVoteRequest)

	ht.server = &http.Server{
		Addr:    ht.address,
		Handler: mux,
	}

	go func() {
		if err := ht.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Printf("Raft HTTP transport error: %v\n", err)
		}
	}()

	return nil
}

// Stop stops the HTTP transport
func (ht *HTTPTransport) Stop() error {
	ht.mu.Lock()
	defer ht.mu.Unlock()

	if ht.server != nil {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
		defer cancel()
		return ht.server.Shutdown(ctx)
	}

	return nil
}

// SendAppendEntry sends an append entry RPC
func (ht *HTTPTransport) SendAppendEntry(ctx context.Context, nodeID string, req *AppendEntryRequest) (*AppendEntryResponse, error) {
	// This would need to resolve nodeID to address
	address := fmt.Sprintf("http://node-%s:8080/raft/append", nodeID) // Placeholder

	payload, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", address, bytes.NewBuffer(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := ht.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var response AppendEntryResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	return &response, nil
}

// SendVoteRequest sends a vote RPC
func (ht *HTTPTransport) SendVoteRequest(ctx context.Context, nodeID string, req *VoteRequest) (*VoteResponse, error) {
	// This would need to resolve nodeID to address
	address := fmt.Sprintf("http://node-%s:8080/raft/vote", nodeID) // Placeholder

	payload, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", address, bytes.NewBuffer(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := ht.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var response VoteResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	return &response, nil
}

// HTTP handlers for Raft RPCs

func (ht *HTTPTransport) handleAppendEntry(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req AppendEntryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	ht.mu.RLock()
	handler := ht.handler
	ht.mu.RUnlock()

	if handler == nil {
		http.Error(w, "Handler not set", http.StatusInternalServerError)
		return
	}

	response := handler.HandleAppendEntry(&req)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (ht *HTTPTransport) handleVoteRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req VoteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	ht.mu.RLock()
	handler := ht.handler
	ht.mu.RUnlock()

	if handler == nil {
		http.Error(w, "Handler not set", http.StatusInternalServerError)
		return
	}

	response := handler.HandleVoteRequest(&req)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
