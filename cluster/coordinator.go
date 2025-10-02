// Package cluster provides distributed query coordination for GoVecDB.
package cluster

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// QueryCoordinatorImpl implements the QueryCoordinator interface
type QueryCoordinatorImpl struct {
	nodeID         string
	clusterManager ClusterManager
	network        NetworkManager
	planner        QueryPlanner

	// Query execution
	activeQueries map[string]*ActiveQuery
	queryHistory  map[string]*QueryExecutionStats
	mu            sync.RWMutex

	// Configuration
	config *QueryCoordinatorConfig

	// Statistics
	stats *QueryCoordinatorStats
}

// QueryCoordinatorConfig represents configuration for the query coordinator
type QueryCoordinatorConfig struct {
	MaxConcurrentQueries int           `json:"max_concurrent_queries"`
	QueryTimeout         time.Duration `json:"query_timeout"`
	RetryAttempts        int           `json:"retry_attempts"`
	RetryDelay           time.Duration `json:"retry_delay"`

	// Result merging
	MaxResultsPerShard int     `json:"max_results_per_shard"`
	ScoreThreshold     float32 `json:"score_threshold"`
	DiversityFactor    float32 `json:"diversity_factor"`

	// Performance
	ParallelismFactor  int `json:"parallelism_factor"`
	BatchSize          int `json:"batch_size"`
	ConnectionPoolSize int `json:"connection_pool_size"`
}

// DefaultQueryCoordinatorConfig returns default configuration
func DefaultQueryCoordinatorConfig() *QueryCoordinatorConfig {
	return &QueryCoordinatorConfig{
		MaxConcurrentQueries: 1000,
		QueryTimeout:         time.Minute * 5,
		RetryAttempts:        3,
		RetryDelay:           time.Millisecond * 100,

		MaxResultsPerShard: 1000,
		ScoreThreshold:     0.0,
		DiversityFactor:    0.1,

		ParallelismFactor:  4,
		BatchSize:          100,
		ConnectionPoolSize: 10,
	}
}

// QueryCoordinatorStats contains query coordinator statistics
type QueryCoordinatorStats struct {
	QueriesTotal     int64         `json:"queries_total"`
	QueriesSucceeded int64         `json:"queries_succeeded"`
	QueriesFailed    int64         `json:"queries_failed"`
	QueriesTimeout   int64         `json:"queries_timeout"`
	AvgLatency       time.Duration `json:"avg_latency"`
	AvgResultCount   float64       `json:"avg_result_count"`
	ShardsQueried    int64         `json:"shards_queried"`
	NodesQueried     int64         `json:"nodes_queried"`
	RetryCount       int64         `json:"retry_count"`
}

// ActiveQuery represents an active query execution
type ActiveQuery struct {
	ID           string                         `json:"id"`
	Plan         *QueryPlan                     `json:"plan"`
	Request      *api.SearchRequest             `json:"request"`
	StartTime    time.Time                      `json:"start_time"`
	Status       QueryExecutionStatus           `json:"status"`
	Results      []*api.SearchResult            `json:"results"`
	Errors       []error                        `json:"errors"`
	ShardResults map[string][]*api.SearchResult `json:"shard_results"`

	// Synchronization
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	mu     sync.RWMutex
}

// QueryExecutionStatus represents the status of query execution
type QueryExecutionStatus int

const (
	QueryStatusPending QueryExecutionStatus = iota
	QueryStatusPlanning
	QueryStatusExecuting
	QueryStatusMerging
	QueryStatusCompleted
	QueryStatusFailed
	QueryStatusTimeout
	QueryStatusCancelled
)

// String returns the string representation of query execution status
func (s QueryExecutionStatus) String() string {
	switch s {
	case QueryStatusPending:
		return "PENDING"
	case QueryStatusPlanning:
		return "PLANNING"
	case QueryStatusExecuting:
		return "EXECUTING"
	case QueryStatusMerging:
		return "MERGING"
	case QueryStatusCompleted:
		return "COMPLETED"
	case QueryStatusFailed:
		return "FAILED"
	case QueryStatusTimeout:
		return "TIMEOUT"
	case QueryStatusCancelled:
		return "CANCELLED"
	default:
		return "UNKNOWN"
	}
}

// QueryExecutionStats contains statistics for a query execution
type QueryExecutionStats struct {
	QueryID       string               `json:"query_id"`
	StartTime     time.Time            `json:"start_time"`
	EndTime       time.Time            `json:"end_time"`
	Duration      time.Duration        `json:"duration"`
	Status        QueryExecutionStatus `json:"status"`
	ResultCount   int                  `json:"result_count"`
	ShardsQueried int                  `json:"shards_queried"`
	NodesQueried  int                  `json:"nodes_queried"`
	RetryCount    int                  `json:"retry_count"`
	Error         string               `json:"error,omitempty"`
}

// QueryPlanner defines the interface for query planning
type QueryPlanner interface {
	PlanQuery(ctx context.Context, req *api.SearchRequest, collection string) (*QueryPlan, error)
	OptimizePlan(ctx context.Context, plan *QueryPlan) (*QueryPlan, error)
	EstimateQueryCost(ctx context.Context, plan *QueryPlan) (*QueryCostEstimate, error)
}

// QueryCostEstimate represents the estimated cost of executing a query
type QueryCostEstimate struct {
	EstimatedLatency     time.Duration `json:"estimated_latency"`
	EstimatedNetworkIO   int64         `json:"estimated_network_io"`
	EstimatedComputeCost float64       `json:"estimated_compute_cost"`
	NodesInvolved        int           `json:"nodes_involved"`
	ShardsInvolved       int           `json:"shards_involved"`
	ParallelismDegree    int           `json:"parallelism_degree"`
}

// NewQueryCoordinator creates a new query coordinator
func NewQueryCoordinator(nodeID string, clusterManager ClusterManager, network NetworkManager, config *QueryCoordinatorConfig) *QueryCoordinatorImpl {
	if config == nil {
		config = DefaultQueryCoordinatorConfig()
	}

	return &QueryCoordinatorImpl{
		nodeID:         nodeID,
		clusterManager: clusterManager,
		network:        network,
		planner:        NewQueryPlannerImpl(clusterManager),

		activeQueries: make(map[string]*ActiveQuery),
		queryHistory:  make(map[string]*QueryExecutionStats),

		config: config,
		stats:  &QueryCoordinatorStats{},
	}
}

// ExecuteQuery executes a distributed search query
func (qc *QueryCoordinatorImpl) ExecuteQuery(ctx context.Context, req *api.SearchRequest) ([]*api.SearchResult, error) {
	// Generate query ID
	queryID := fmt.Sprintf("query-%d-%s", time.Now().UnixNano(), qc.nodeID)

	// Check concurrent query limit
	qc.mu.RLock()
	if len(qc.activeQueries) >= qc.config.MaxConcurrentQueries {
		qc.mu.RUnlock()
		qc.stats.QueriesFailed++
		return nil, fmt.Errorf("too many concurrent queries")
	}
	qc.mu.RUnlock()

	// Create active query
	queryCtx, cancel := context.WithTimeout(ctx, qc.config.QueryTimeout)
	activeQuery := &ActiveQuery{
		ID:           queryID,
		Request:      req,
		StartTime:    time.Now(),
		Status:       QueryStatusPending,
		ShardResults: make(map[string][]*api.SearchResult),
		ctx:          queryCtx,
		cancel:       cancel,
	}

	// Register active query
	qc.mu.Lock()
	qc.activeQueries[queryID] = activeQuery
	qc.mu.Unlock()

	defer func() {
		cancel()
		qc.mu.Lock()
		delete(qc.activeQueries, queryID)
		qc.mu.Unlock()
	}()

	// Execute query with error handling
	results, err := qc.executeQueryInternal(activeQuery)

	// Update statistics
	qc.updateQueryStats(activeQuery, err)

	return results, err
}

// ExecuteBatch executes a batch of operations
func (qc *QueryCoordinatorImpl) ExecuteBatch(ctx context.Context, req *api.BatchRequest) (*api.BatchResponse, error) {
	_ = fmt.Sprintf("batch-%d-%s", time.Now().UnixNano(), qc.nodeID) // batchID for future use

	response := &api.BatchResponse{
		Results: make([]api.BatchResult, len(req.Operations)),
		Success: true,
	}

	// Execute operations based on type
	for i, op := range req.Operations {
		switch op.Type {
		case api.BatchOperationSearch:
			if op.Search == nil {
				response.Results[i] = api.BatchResult{
					Success: false,
					Error:   "missing search request",
				}
				response.Success = false
				continue
			}

			results, err := qc.ExecuteQuery(ctx, op.Search)
			if err != nil {
				response.Results[i] = api.BatchResult{
					Success: false,
					Error:   err.Error(),
				}
				response.Success = false
			} else {
				response.Results[i] = api.BatchResult{
					Success:       true,
					SearchResults: results,
				}
			}

		case api.BatchOperationGet:
			// Handle get operation
			response.Results[i] = api.BatchResult{
				Success: false,
				Error:   "get operation not implemented",
			}
			response.Success = false

		default:
			response.Results[i] = api.BatchResult{
				Success: false,
				Error:   fmt.Sprintf("unsupported operation type: %s", op.Type),
			}
			response.Success = false
		}
	}

	return response, nil
}

// PlanQuery creates a query execution plan
func (qc *QueryCoordinatorImpl) PlanQuery(ctx context.Context, req *api.SearchRequest) (*QueryPlan, error) {
	// For now, assume single collection named "default"
	collection := "default"

	return qc.planner.PlanQuery(ctx, req, collection)
}

// ExecutePlan executes a pre-planned query
func (qc *QueryCoordinatorImpl) ExecutePlan(ctx context.Context, plan *QueryPlan) ([]*api.SearchResult, error) {
	// Create active query from plan
	activeQuery := &ActiveQuery{
		ID:           plan.ID,
		Plan:         plan,
		StartTime:    time.Now(),
		Status:       QueryStatusExecuting,
		ShardResults: make(map[string][]*api.SearchResult),
		ctx:          ctx,
	}

	return qc.executePlanInternal(activeQuery)
}

// GetActiveQueries returns all active queries
func (qc *QueryCoordinatorImpl) GetActiveQueries(ctx context.Context) ([]*QueryPlan, error) {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	plans := make([]*QueryPlan, 0, len(qc.activeQueries))
	for _, query := range qc.activeQueries {
		if query.Plan != nil {
			plans = append(plans, query.Plan)
		}
	}

	return plans, nil
}

// CancelQuery cancels an active query
func (qc *QueryCoordinatorImpl) CancelQuery(ctx context.Context, queryID string) error {
	qc.mu.RLock()
	query, exists := qc.activeQueries[queryID]
	qc.mu.RUnlock()

	if !exists {
		return fmt.Errorf("query %s not found", queryID)
	}

	query.mu.Lock()
	query.Status = QueryStatusCancelled
	query.mu.Unlock()

	query.cancel()
	return nil
}

// Private methods

func (qc *QueryCoordinatorImpl) executeQueryInternal(activeQuery *ActiveQuery) ([]*api.SearchResult, error) {
	// Phase 1: Planning
	activeQuery.mu.Lock()
	activeQuery.Status = QueryStatusPlanning
	activeQuery.mu.Unlock()

	plan, err := qc.planner.PlanQuery(activeQuery.ctx, activeQuery.Request, "default")
	if err != nil {
		activeQuery.mu.Lock()
		activeQuery.Status = QueryStatusFailed
		activeQuery.Errors = append(activeQuery.Errors, err)
		activeQuery.mu.Unlock()
		return nil, fmt.Errorf("query planning failed: %w", err)
	}

	activeQuery.mu.Lock()
	activeQuery.Plan = plan
	activeQuery.mu.Unlock()

	// Phase 2: Execution
	return qc.executePlanInternal(activeQuery)
}

func (qc *QueryCoordinatorImpl) executePlanInternal(activeQuery *ActiveQuery) ([]*api.SearchResult, error) {
	activeQuery.mu.Lock()
	activeQuery.Status = QueryStatusExecuting
	plan := activeQuery.Plan
	activeQuery.mu.Unlock()

	if plan == nil {
		return nil, fmt.Errorf("no execution plan available")
	}

	// Execute phases in order
	for _, phase := range plan.Phases {
		if err := qc.executePhase(activeQuery, &phase); err != nil {
			activeQuery.mu.Lock()
			activeQuery.Status = QueryStatusFailed
			activeQuery.Errors = append(activeQuery.Errors, err)
			activeQuery.mu.Unlock()
			return nil, fmt.Errorf("phase execution failed: %w", err)
		}

		// Check for cancellation
		select {
		case <-activeQuery.ctx.Done():
			activeQuery.mu.Lock()
			activeQuery.Status = QueryStatusCancelled
			activeQuery.mu.Unlock()
			return nil, activeQuery.ctx.Err()
		default:
		}
	}

	// Phase 3: Merge results
	activeQuery.mu.Lock()
	activeQuery.Status = QueryStatusMerging
	activeQuery.mu.Unlock()

	results, err := qc.mergeResults(activeQuery)
	if err != nil {
		activeQuery.mu.Lock()
		activeQuery.Status = QueryStatusFailed
		activeQuery.Errors = append(activeQuery.Errors, err)
		activeQuery.mu.Unlock()
		return nil, fmt.Errorf("result merging failed: %w", err)
	}

	activeQuery.mu.Lock()
	activeQuery.Status = QueryStatusCompleted
	activeQuery.Results = results
	activeQuery.mu.Unlock()

	return results, nil
}

func (qc *QueryCoordinatorImpl) executePhase(activeQuery *ActiveQuery, phase *QueryPhase) error {
	switch phase.Type {
	case QueryPhaseTypeScatter:
		return qc.executeScatterPhase(activeQuery, phase)
	case QueryPhaseTypeGather:
		return qc.executeGatherPhase(activeQuery, phase)
	case QueryPhaseTypeMap:
		return qc.executeMapPhase(activeQuery, phase)
	case QueryPhaseTypeReduce:
		return qc.executeReducePhase(activeQuery, phase)
	default:
		return fmt.Errorf("unsupported phase type: %s", phase.Type)
	}
}

func (qc *QueryCoordinatorImpl) executeScatterPhase(activeQuery *ActiveQuery, phase *QueryPhase) error {
	// Scatter search requests to multiple shards/nodes
	shards, err := qc.clusterManager.GetShards(activeQuery.ctx, activeQuery.Plan.Collection)
	if err != nil {
		return fmt.Errorf("failed to get shards: %w", err)
	}

	// Execute queries in parallel across shards
	var wg sync.WaitGroup
	var mu sync.Mutex
	errors := make([]error, 0)

	for _, shard := range shards {
		if shard.State != ShardStateActive {
			continue // Skip inactive shards
		}

		wg.Add(1)
		go func(s *ShardInfo) {
			defer wg.Done()

			// Send query to shard's primary node
			node, err := qc.clusterManager.GetNode(activeQuery.ctx, s.Primary)
			if err != nil {
				mu.Lock()
				errors = append(errors, fmt.Errorf("failed to get node %s: %w", s.Primary, err))
				mu.Unlock()
				return
			}

			results, err := qc.network.SendQuery(activeQuery.ctx, node.Address, activeQuery.Request)
			if err != nil {
				// Try replicas if primary fails
				for _, replicaID := range s.Replicas {
					replicaNode, err := qc.clusterManager.GetNode(activeQuery.ctx, replicaID)
					if err != nil {
						continue
					}

					results, err = qc.network.SendQuery(activeQuery.ctx, replicaNode.Address, activeQuery.Request)
					if err == nil {
						break
					}
				}

				if err != nil {
					mu.Lock()
					errors = append(errors, fmt.Errorf("failed to query shard %s: %w", s.ID, err))
					mu.Unlock()
					return
				}
			}

			// Store shard results
			activeQuery.mu.Lock()
			activeQuery.ShardResults[s.ID] = results
			activeQuery.mu.Unlock()

		}(shard)
	}

	wg.Wait()

	if len(errors) > 0 && len(errors) == len(shards) {
		return fmt.Errorf("all shard queries failed: %v", errors)
	}

	qc.stats.ShardsQueried += int64(len(shards))
	return nil
}

func (qc *QueryCoordinatorImpl) executeGatherPhase(activeQuery *ActiveQuery, phase *QueryPhase) error {
	// Gather results from scatter phase - results are already collected
	return nil
}

func (qc *QueryCoordinatorImpl) executeMapPhase(activeQuery *ActiveQuery, phase *QueryPhase) error {
	// Apply transformations to results
	// Implementation depends on specific map operation
	return nil
}

func (qc *QueryCoordinatorImpl) executeReducePhase(activeQuery *ActiveQuery, phase *QueryPhase) error {
	// Reduce/aggregate results
	// Implementation depends on specific reduce operation
	return nil
}

func (qc *QueryCoordinatorImpl) mergeResults(activeQuery *ActiveQuery) ([]*api.SearchResult, error) {
	activeQuery.mu.RLock()
	shardResults := activeQuery.ShardResults
	k := activeQuery.Request.K
	activeQuery.mu.RUnlock()

	// Collect all results
	allResults := make([]*api.SearchResult, 0)
	for _, results := range shardResults {
		allResults = append(allResults, results...)
	}

	if len(allResults) == 0 {
		return []*api.SearchResult{}, nil
	}

	// Sort by score (assuming higher is better)
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score > allResults[j].Score
	})

	// Apply score threshold
	if activeQuery.Request.MinScore != nil {
		filtered := make([]*api.SearchResult, 0)
		for _, result := range allResults {
			if result.Score >= *activeQuery.Request.MinScore {
				filtered = append(filtered, result)
			}
		}
		allResults = filtered
	}

	// Take top K results
	if k > 0 && len(allResults) > k {
		allResults = allResults[:k]
	}

	// Apply diversity if configured
	if qc.config.DiversityFactor > 0 {
		allResults = qc.applyDiversity(allResults, qc.config.DiversityFactor)
	}

	return allResults, nil
}

func (qc *QueryCoordinatorImpl) applyDiversity(results []*api.SearchResult, diversityFactor float32) []*api.SearchResult {
	// Simple diversity implementation - remove very similar results
	if len(results) <= 1 {
		return results
	}

	diverse := make([]*api.SearchResult, 0, len(results))
	diverse = append(diverse, results[0]) // Always include top result

	for i := 1; i < len(results); i++ {
		candidate := results[i]
		tooSimilar := false

		// Check similarity with already selected results
		for _, selected := range diverse {
			// Simple similarity check - could be enhanced with vector similarity
			if abs(candidate.Score-selected.Score) < diversityFactor {
				tooSimilar = true
				break
			}
		}

		if !tooSimilar {
			diverse = append(diverse, candidate)
		}
	}

	return diverse
}

func (qc *QueryCoordinatorImpl) updateQueryStats(activeQuery *ActiveQuery, err error) {
	duration := time.Since(activeQuery.StartTime)

	activeQuery.mu.RLock()
	status := activeQuery.Status
	resultCount := len(activeQuery.Results)
	shardCount := len(activeQuery.ShardResults)
	activeQuery.mu.RUnlock()

	// Update coordinator stats
	qc.mu.Lock()
	qc.stats.QueriesTotal++

	if err != nil {
		qc.stats.QueriesFailed++
	} else {
		qc.stats.QueriesSucceeded++
	}

	if status == QueryStatusTimeout {
		qc.stats.QueriesTimeout++
	}

	// Update average latency
	alpha := 0.1
	qc.stats.AvgLatency = time.Duration(alpha*float64(duration) + (1-alpha)*float64(qc.stats.AvgLatency))

	// Update average result count
	qc.stats.AvgResultCount = alpha*float64(resultCount) + (1-alpha)*qc.stats.AvgResultCount

	qc.stats.ShardsQueried += int64(shardCount)
	qc.mu.Unlock()

	// Store query execution stats
	execStats := &QueryExecutionStats{
		QueryID:       activeQuery.ID,
		StartTime:     activeQuery.StartTime,
		EndTime:       time.Now(),
		Duration:      duration,
		Status:        status,
		ResultCount:   resultCount,
		ShardsQueried: shardCount,
		NodesQueried:  shardCount, // Simplified - one node per shard
		RetryCount:    0,          // TODO: track retries
	}

	if err != nil {
		execStats.Error = err.Error()
	}

	qc.mu.Lock()
	qc.queryHistory[activeQuery.ID] = execStats
	// Keep only recent history (last 1000 queries)
	if len(qc.queryHistory) > 1000 {
		// Simple cleanup - remove oldest entries
		// In production, this should be more sophisticated
		for id := range qc.queryHistory {
			delete(qc.queryHistory, id)
			if len(qc.queryHistory) <= 900 {
				break
			}
		}
	}
	qc.mu.Unlock()
}

// GetStats returns query coordinator statistics
func (qc *QueryCoordinatorImpl) GetStats() *QueryCoordinatorStats {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	stats := *qc.stats
	return &stats
}

// GetQueryHistory returns query execution history
func (qc *QueryCoordinatorImpl) GetQueryHistory(limit int) []*QueryExecutionStats {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	history := make([]*QueryExecutionStats, 0, len(qc.queryHistory))
	for _, stats := range qc.queryHistory {
		history = append(history, stats)
	}

	// Sort by start time (newest first)
	sort.Slice(history, func(i, j int) bool {
		return history[i].StartTime.After(history[j].StartTime)
	})

	if limit > 0 && len(history) > limit {
		history = history[:limit]
	}

	return history
}

// Helper function for absolute value
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// QueryPlannerImpl implements the QueryPlanner interface
type QueryPlannerImpl struct {
	clusterManager ClusterManager
	mu             sync.RWMutex
}

// NewQueryPlannerImpl creates a new query planner
func NewQueryPlannerImpl(clusterManager ClusterManager) *QueryPlannerImpl {
	return &QueryPlannerImpl{
		clusterManager: clusterManager,
	}
}

// PlanQuery creates an execution plan for a search query
func (qp *QueryPlannerImpl) PlanQuery(ctx context.Context, req *api.SearchRequest, collection string) (*QueryPlan, error) {
	planID := fmt.Sprintf("plan-%d", time.Now().UnixNano())

	// Get shards for collection
	shards, err := qp.clusterManager.GetShards(ctx, collection)
	if err != nil {
		return nil, fmt.Errorf("failed to get shards: %w", err)
	}

	if len(shards) == 0 {
		return nil, fmt.Errorf("no shards found for collection %s", collection)
	}

	// Create phases
	phases := []QueryPhase{
		{
			ID:        "scatter",
			Type:      QueryPhaseTypeScatter,
			Operation: "scatter_search",
			Parameters: map[string]interface{}{
				"collection": collection,
				"k":          req.K,
			},
			State: QueryPhaseStatePending,
		},
		{
			ID:           "gather",
			Type:         QueryPhaseTypeGather,
			Dependencies: []string{"scatter"},
			Operation:    "gather_results",
			State:        QueryPhaseStatePending,
		},
	}

	// Extract shard IDs and involved nodes
	shardIDs := make([]string, len(shards))
	nodeSet := make(map[string]bool)

	for i, shard := range shards {
		shardIDs[i] = shard.ID
		nodeSet[shard.Primary] = true
		for _, replica := range shard.Replicas {
			nodeSet[replica] = true
		}
	}

	nodes := make([]string, 0, len(nodeSet))
	for nodeID := range nodeSet {
		nodes = append(nodes, nodeID)
	}

	plan := &QueryPlan{
		ID:               planID,
		QueryType:        QueryTypeSearch,
		Collection:       collection,
		Coordinator:      "", // Will be set by coordinator
		Phases:           phases,
		Shards:           shardIDs,
		Nodes:            nodes,
		EstimatedLatency: time.Millisecond * 100 * time.Duration(len(shards)), // Simple estimate
		Parallelism:      len(shards),
		Timeout:          time.Minute * 5,
		Parameters: map[string]interface{}{
			"k":      req.K,
			"vector": req.Vector,
		},
		State:     QueryPlanStatePending,
		CreatedAt: time.Now(),
	}

	return plan, nil
}

// OptimizePlan optimizes a query execution plan
func (qp *QueryPlannerImpl) OptimizePlan(ctx context.Context, plan *QueryPlan) (*QueryPlan, error) {
	// Simple optimization - could be enhanced with cost-based optimization
	optimized := *plan

	// Adjust parallelism based on cluster size
	nodes, err := qp.clusterManager.GetNodes(ctx)
	if err == nil {
		activeNodes := 0
		for _, node := range nodes {
			if node.State == NodeStateActive {
				activeNodes++
			}
		}

		if activeNodes > 0 {
			optimized.Parallelism = min(optimized.Parallelism, activeNodes*2)
		}
	}

	return &optimized, nil
}

// EstimateQueryCost estimates the cost of executing a query plan
func (qp *QueryPlannerImpl) EstimateQueryCost(ctx context.Context, plan *QueryPlan) (*QueryCostEstimate, error) {
	estimate := &QueryCostEstimate{
		EstimatedLatency:     plan.EstimatedLatency,
		EstimatedNetworkIO:   int64(len(plan.Shards) * 1024), // Simple estimate
		EstimatedComputeCost: float64(len(plan.Shards)) * 1.0,
		NodesInvolved:        len(plan.Nodes),
		ShardsInvolved:       len(plan.Shards),
		ParallelismDegree:    plan.Parallelism,
	}

	return estimate, nil
}

// Helper function for minimum
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
