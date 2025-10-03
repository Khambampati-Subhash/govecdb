package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// StreamingAPI provides high-performance streaming capabilities
type StreamingAPI struct {
	// Core components
	db     *GoVecDB
	server *http.Server

	// Connection management
	connections   map[string]*StreamConnection
	connectionsMu sync.RWMutex

	// Streaming configuration
	config *StreamConfig

	// Metrics
	totalRequests  int64
	streamingBytes int64
}

// StreamConfig defines streaming configuration
type StreamConfig struct {
	MaxConnections     int
	BufferSize         int
	FlushInterval      time.Duration
	CompressionEnabled bool
	BatchSize          int
	MaxBatchWait       time.Duration
}

// StreamConnection represents a streaming connection
type StreamConnection struct {
	ID           string
	Writer       http.ResponseWriter
	Flusher      http.Flusher
	Context      context.Context
	Cancel       context.CancelFunc
	LastActivity time.Time
	Buffer       []byte
	BufferMu     sync.Mutex
}

// BatchRequest represents a batch operation request
type BatchRequest struct {
	Operations []Operation  `json:"operations"`
	Options    BatchOptions `json:"options,omitempty"`
}

// Operation represents a single operation in a batch
type Operation struct {
	Type     BatchOperationType     `json:"type"` // Changed from string for compatibility
	ID       string                 `json:"id,omitempty"`
	Vector   []float32              `json:"vector,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Query    *SearchQuery           `json:"query,omitempty"`
	Search   *SearchRequest         `json:"search,omitempty"` // For backward compatibility
}

// BatchOptions defines options for batch operations
type BatchOptions struct {
	Parallel        bool          `json:"parallel"`
	MaxWorkers      int           `json:"max_workers"`
	Timeout         time.Duration `json:"timeout"`
	ContinueOnError bool          `json:"continue_on_error"`
}

// SearchQuery represents a search query
type SearchQuery struct {
	Vector   []float32              `json:"vector"`
	K        int                    `json:"k"`
	Filter   map[string]interface{} `json:"filter,omitempty"`
	MinScore float32                `json:"min_score,omitempty"`
}

// BatchResponse represents the response to a batch operation
type BatchResponse struct {
	RequestID  string          `json:"request_id"`
	Results    []BatchResult   `json:"results"` // Changed from OperationResult for compatibility
	Errors     []BatchError    `json:"errors,omitempty"`
	Duration   time.Duration   `json:"duration"`
	Statistics BatchStatistics `json:"statistics"`
	Success    bool            `json:"success"` // For backward compatibility
}

// OperationResult represents the result of a single operation
type OperationResult struct {
	OperationID string        `json:"operation_id"`
	Type        string        `json:"type"`
	Success     bool          `json:"success"`
	Data        interface{}   `json:"data,omitempty"`
	Error       string        `json:"error,omitempty"`
	Duration    time.Duration `json:"duration"`
}

// BatchError represents an error in batch processing
type BatchError struct {
	OperationID string `json:"operation_id"`
	Error       string `json:"error"`
}

// BatchStatistics provides statistics about batch execution
type BatchStatistics struct {
	TotalOperations      int           `json:"total_operations"`
	SuccessfulOperations int           `json:"successful_operations"`
	FailedOperations     int           `json:"failed_operations"`
	AverageLatency       time.Duration `json:"average_latency"`
	TotalDuration        time.Duration `json:"total_duration"`
	Throughput           float64       `json:"throughput"`
}

// NewStreamingAPI creates a new streaming API
func NewStreamingAPI(db *GoVecDB, config *StreamConfig) *StreamingAPI {
	if config == nil {
		config = &StreamConfig{
			MaxConnections:     1000,
			BufferSize:         64 * 1024,
			FlushInterval:      100 * time.Millisecond,
			CompressionEnabled: true,
			BatchSize:          1000,
			MaxBatchWait:       10 * time.Millisecond,
		}
	}

	return &StreamingAPI{
		db:          db,
		connections: make(map[string]*StreamConnection),
		config:      config,
	}
}

// StartServer starts the streaming API server
func (api *StreamingAPI) StartServer(addr string) error {
	mux := http.NewServeMux()

	// Streaming endpoints
	mux.HandleFunc("/v1/stream/search", api.handleStreamSearch)
	mux.HandleFunc("/v1/stream/insert", api.handleStreamInsert)
	mux.HandleFunc("/v1/stream/batch", api.handleStreamBatch)

	// Batch endpoints
	mux.HandleFunc("/v1/batch", api.handleBatch)
	mux.HandleFunc("/v1/batch/async", api.handleAsyncBatch)

	// Enhanced search endpoints
	mux.HandleFunc("/v1/search/advanced", api.handleAdvancedSearch)
	mux.HandleFunc("/v1/search/multi", api.handleMultiSearch)

	// Utility endpoints
	mux.HandleFunc("/v1/stats", api.handleStats)
	mux.HandleFunc("/v1/health", api.handleHealth)

	api.server = &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	return api.server.ListenAndServe()
}

// handleStreamSearch handles streaming search requests
func (api *StreamingAPI) handleStreamSearch(w http.ResponseWriter, r *http.Request) {
	// Set streaming headers
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Create stream connection
	ctx, cancel := context.WithCancel(r.Context())
	conn := &StreamConnection{
		ID:           generateConnectionID(),
		Writer:       w,
		Flusher:      flusher,
		Context:      ctx,
		Cancel:       cancel,
		LastActivity: time.Now(),
		Buffer:       make([]byte, 0, api.config.BufferSize),
	}

	// Register connection
	api.registerConnection(conn)
	defer api.unregisterConnection(conn.ID)

	// Process streaming search
	for {
		select {
		case <-ctx.Done():
			return
		default:
			// Read search query from request stream
			query, err := api.readSearchQuery(r)
			if err != nil {
				if err == io.EOF {
					return
				}
				api.writeError(conn, fmt.Sprintf("Failed to read query: %v", err))
				continue
			}

			// Perform search
			results, err := api.performSearch(ctx, query)
			if err != nil {
				api.writeError(conn, fmt.Sprintf("Search failed: %v", err))
				continue
			}

			// Stream results
			if err := api.streamResults(conn, results); err != nil {
				return
			}
		}
	}
}

// handleBatch handles batch operations
func (api *StreamingAPI) handleBatch(w http.ResponseWriter, r *http.Request) {
	var batchReq BatchRequest
	if err := json.NewDecoder(r.Body).Decode(&batchReq); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Execute batch
	response, err := api.executeBatch(r.Context(), &batchReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("Batch execution failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// executeBatch executes a batch of operations
func (api *StreamingAPI) executeBatch(ctx context.Context, req *BatchRequest) (*BatchResponse, error) {
	startTime := time.Now()
	requestID := generateRequestID()

	var results []OperationResult
	var errors []BatchError
	var successCount, failCount int
	var totalDuration time.Duration

	// Execute operations
	if req.Options.Parallel && len(req.Operations) > 1 {
		// Parallel execution
		results, errors = api.executeParallel(ctx, req.Operations, req.Options)
	} else {
		// Sequential execution
		results, errors = api.executeSequential(ctx, req.Operations, req.Options)
	}

	// Calculate statistics
	for _, result := range results {
		totalDuration += result.Duration
		if result.Success {
			successCount++
		} else {
			failCount++
		}
	}

	batchDuration := time.Since(startTime)
	avgLatency := time.Duration(0)
	if len(results) > 0 {
		avgLatency = totalDuration / time.Duration(len(results))
	}

	throughput := float64(len(req.Operations)) / batchDuration.Seconds()

	// Convert OperationResult to BatchResult for compatibility
	batchResults := make([]BatchResult, len(results))
	for i, result := range results {
		batchResults[i] = BatchResult{
			Success: result.Success,
			Error:   result.Error,
			// Handle data conversion based on result type
		}
	}

	response := &BatchResponse{
		RequestID: requestID,
		Results:   batchResults,
		Errors:    errors,
		Duration:  batchDuration,
		Statistics: BatchStatistics{
			TotalOperations:      len(req.Operations),
			SuccessfulOperations: successCount,
			FailedOperations:     failCount,
			AverageLatency:       avgLatency,
			TotalDuration:        batchDuration,
			Throughput:           throughput,
		},
		Success: len(errors) == 0,
	}

	return response, nil
}

// executeParallel executes operations in parallel
func (api *StreamingAPI) executeParallel(ctx context.Context, operations []Operation, options BatchOptions) ([]OperationResult, []BatchError) {
	maxWorkers := options.MaxWorkers
	if maxWorkers <= 0 {
		maxWorkers = 10
	}

	results := make([]OperationResult, len(operations))
	var errors []BatchError
	var wg sync.WaitGroup
	var errorsMu sync.Mutex

	// Create worker pool
	opChan := make(chan operationJob, len(operations))

	// Start workers
	for i := 0; i < maxWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range opChan {
				result := api.executeOperation(ctx, job.operation)
				results[job.index] = result

				if !result.Success {
					errorsMu.Lock()
					errors = append(errors, BatchError{
						OperationID: job.operation.ID,
						Error:       result.Error,
					})
					errorsMu.Unlock()

					if !options.ContinueOnError {
						return
					}
				}
			}
		}()
	}

	// Send operations to workers
	for i, op := range operations {
		opChan <- operationJob{index: i, operation: op}
	}
	close(opChan)

	wg.Wait()
	return results, errors
}

// executeSequential executes operations sequentially
func (api *StreamingAPI) executeSequential(ctx context.Context, operations []Operation, options BatchOptions) ([]OperationResult, []BatchError) {
	results := make([]OperationResult, len(operations))
	var errors []BatchError

	for i, op := range operations {
		result := api.executeOperation(ctx, op)
		results[i] = result

		if !result.Success {
			errors = append(errors, BatchError{
				OperationID: op.ID,
				Error:       result.Error,
			})

			if !options.ContinueOnError {
				break
			}
		}
	}

	return results, errors
}

// executeOperation executes a single operation
func (api *StreamingAPI) executeOperation(ctx context.Context, op Operation) OperationResult {
	startTime := time.Now()

	result := OperationResult{
		OperationID: op.ID,
		Type:        string(op.Type),
		Success:     false,
	}

	switch op.Type {
	case BatchOperationInsert:
		err := api.performInsert(ctx, op.ID, op.Vector, op.Metadata)
		if err != nil {
			result.Error = err.Error()
		} else {
			result.Success = true
			result.Data = map[string]string{"status": "inserted"}
		}

	case BatchOperationSearch:
		if op.Query == nil {
			result.Error = "Query is required for search operation"
		} else {
			searchResults, err := api.performSearch(ctx, op.Query)
			if err != nil {
				result.Error = err.Error()
			} else {
				result.Success = true
				result.Data = searchResults
			}
		}

	case BatchOperationUpdate:
		err := api.performUpdate(ctx, op.ID, op.Vector, op.Metadata)
		if err != nil {
			result.Error = err.Error()
		} else {
			result.Success = true
			result.Data = map[string]string{"status": "updated"}
		}

	case BatchOperationDelete:
		err := api.performDelete(ctx, op.ID)
		if err != nil {
			result.Error = err.Error()
		} else {
			result.Success = true
			result.Data = map[string]string{"status": "deleted"}
		}

	default:
		result.Error = fmt.Sprintf("Unknown operation type: %s", op.Type)
	}

	result.Duration = time.Since(startTime)
	return result
}

// Helper types and functions

type operationJob struct {
	index     int
	operation Operation
}

type GoVecDB struct {
	// Placeholder for actual GoVecDB implementation
}

// Placeholder implementations (replace with actual GoVecDB operations)

func (api *StreamingAPI) performInsert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error {
	// Implement actual insert operation
	return nil
}

func (api *StreamingAPI) performSearch(ctx context.Context, query *SearchQuery) (interface{}, error) {
	// Implement actual search operation
	return map[string]interface{}{
		"results": []map[string]interface{}{
			{"id": "1", "score": 0.95},
			{"id": "2", "score": 0.87},
		},
	}, nil
}

func (api *StreamingAPI) performUpdate(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error {
	// Implement actual update operation
	return nil
}

func (api *StreamingAPI) performDelete(ctx context.Context, id string) error {
	// Implement actual delete operation
	return nil
}

func (api *StreamingAPI) readSearchQuery(r *http.Request) (*SearchQuery, error) {
	// Implement query reading from request stream
	return &SearchQuery{
		Vector: []float32{0.1, 0.2, 0.3},
		K:      10,
	}, nil
}

func (api *StreamingAPI) streamResults(conn *StreamConnection, results interface{}) error {
	// Implement result streaming
	data, _ := json.Marshal(results)
	conn.Writer.Write(data)
	conn.Writer.Write([]byte("\n"))
	conn.Flusher.Flush()
	return nil
}

func (api *StreamingAPI) writeError(conn *StreamConnection, error string) {
	// Implement error writing to stream
	errorData := map[string]string{"error": error}
	data, _ := json.Marshal(errorData)
	conn.Writer.Write(data)
	conn.Writer.Write([]byte("\n"))
	conn.Flusher.Flush()
}

func (api *StreamingAPI) registerConnection(conn *StreamConnection) {
	api.connectionsMu.Lock()
	defer api.connectionsMu.Unlock()
	api.connections[conn.ID] = conn
}

func (api *StreamingAPI) unregisterConnection(id string) {
	api.connectionsMu.Lock()
	defer api.connectionsMu.Unlock()
	delete(api.connections, id)
}

func generateConnectionID() string {
	return fmt.Sprintf("conn_%d", time.Now().UnixNano())
}

func generateRequestID() string {
	return fmt.Sprintf("req_%d", time.Now().UnixNano())
}

// Additional endpoint handlers (simplified implementations)

func (api *StreamingAPI) handleStreamInsert(w http.ResponseWriter, r *http.Request) {
	// Implement streaming insert
	w.WriteHeader(http.StatusNotImplemented)
}

func (api *StreamingAPI) handleStreamBatch(w http.ResponseWriter, r *http.Request) {
	// Implement streaming batch
	w.WriteHeader(http.StatusNotImplemented)
}

func (api *StreamingAPI) handleAsyncBatch(w http.ResponseWriter, r *http.Request) {
	// Implement async batch
	w.WriteHeader(http.StatusNotImplemented)
}

func (api *StreamingAPI) handleAdvancedSearch(w http.ResponseWriter, r *http.Request) {
	// Implement advanced search with filters, facets, etc.
	w.WriteHeader(http.StatusNotImplemented)
}

func (api *StreamingAPI) handleMultiSearch(w http.ResponseWriter, r *http.Request) {
	// Implement multi-vector search
	w.WriteHeader(http.StatusNotImplemented)
}

func (api *StreamingAPI) handleStats(w http.ResponseWriter, r *http.Request) {
	stats := map[string]interface{}{
		"active_connections": len(api.connections),
		"total_requests":     api.totalRequests,
		"streaming_bytes":    api.streamingBytes,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func (api *StreamingAPI) handleHealth(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"uptime":    time.Since(time.Now()), // Placeholder
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// Additional types for backward compatibility with cluster package
type BatchResult struct {
	Success       bool            `json:"success"`
	Error         string          `json:"error,omitempty"`
	Vector        *Vector         `json:"vector,omitempty"`         // For get operations
	SearchResults []*SearchResult `json:"search_results,omitempty"` // For search operations
}

type BatchOperation struct {
	Type     BatchOperationType `json:"type"`
	VectorID string             `json:"vector_id,omitempty"`
	Vector   *Vector            `json:"vector,omitempty"`
	Search   *SearchRequest     `json:"search,omitempty"`
}

type BatchOperationType string

const (
	BatchOperationInsert BatchOperationType = "insert"
	BatchOperationUpdate BatchOperationType = "update"
	BatchOperationDelete BatchOperationType = "delete"
	BatchOperationSearch BatchOperationType = "search"
	BatchOperationGet    BatchOperationType = "get"
)
