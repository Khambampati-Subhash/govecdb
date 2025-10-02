// Package filter implements efficient numeric index for range queries and numeric filtering.
package filter

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// NumericMetadataIndex implements MetadataIndex interface for numeric fields.
// It uses a B+Tree-like structure for efficient range queries and comparisons.
// The implementation focuses on read performance with sorted arrays and binary search.
type NumericMetadataIndex struct {
	// Core index data structures
	tree *NumericBTree      // B+tree for range queries
	docs map[string]float64 // vector ID -> numeric value

	// Configuration
	config *IndexConfig
	field  string

	// Statistics and monitoring
	stats *NumericIndexStats

	// Thread safety
	mu     sync.RWMutex
	closed bool

	// Performance optimizations
	cache     *NumericQueryCache
	minValue  float64
	maxValue  float64
	allowNaN  bool
	precision int // Decimal precision for comparisons
}

// NumericBTree implements a B+Tree structure optimized for numeric range queries
type NumericBTree struct {
	root      *NumericNode
	order     int // Maximum number of children per node
	height    int
	nodeCount int64

	// Thread safety for tree operations
	mu sync.RWMutex
}

// NumericNode represents a node in the B+Tree
type NumericNode struct {
	// Node properties
	isLeaf   bool
	keys     []float64      // Sorted keys
	values   [][]string     // For leaf nodes: vector IDs for each key
	children []*NumericNode // For internal nodes: child pointers

	// For leaf nodes: linked list for range queries
	next *NumericNode
	prev *NumericNode

	// Statistics
	accessCount  int64
	lastAccessed time.Time

	// Thread safety
	mu sync.RWMutex
}

// NumericIndexStats provides detailed statistics for numeric index
type NumericIndexStats struct {
	*IndexStats

	// Numeric index specific metrics
	MinValue   float64 `json:"min_value"`
	MaxValue   float64 `json:"max_value"`
	TreeHeight int     `json:"tree_height"`
	NodeCount  int64   `json:"node_count"`
	LeafCount  int64   `json:"leaf_count"`

	// Query performance by type
	EqualityQueries   int64   `json:"equality_queries"`
	RangeQueries      int64   `json:"range_queries"`
	ComparisonQueries int64   `json:"comparison_queries"`
	AvgEqualityTime   float64 `json:"avg_equality_time"`
	AvgRangeTime      float64 `json:"avg_range_time"`
	AvgComparisonTime float64 `json:"avg_comparison_time"`

	// Tree statistics
	AvgNodeFill float64 `json:"avg_node_fill"`
	TreeBalance float64 `json:"tree_balance"`
	SplitCount  int64   `json:"split_count"`
	MergeCount  int64   `json:"merge_count"`

	// Value distribution
	UniqueValues   int64 `json:"unique_values"`
	NullValues     int64 `json:"null_values"`
	NaNValues      int64 `json:"nan_values"`
	InfiniteValues int64 `json:"infinite_values"`
}

// NumericQueryCache implements caching for numeric queries
type NumericQueryCache struct {
	rangeCache      map[string]*NumericCacheEntry
	comparisonCache map[string]*NumericCacheEntry
	maxSize         int
	usage           map[string]time.Time
	mu              sync.RWMutex
}

// NumericCacheEntry represents a cached numeric query result
type NumericCacheEntry struct {
	VectorIDs    []string
	ResultCount  int64
	LastAccessed time.Time
	AccessCount  int64
	QueryType    string // "range", "comparison", "equality"
}

// NumericRange represents a numeric range for queries
type NumericRange struct {
	Min        float64
	Max        float64
	IncludeMin bool
	IncludeMax bool
}

// NewNumericMetadataIndex creates a new numeric index for the specified field
func NewNumericMetadataIndex(config *IndexConfig) (*NumericMetadataIndex, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	if config.Type != NumericIndex {
		return nil, fmt.Errorf("%w: expected %s, got %s", ErrInvalidIndexType, NumericIndex, config.Type)
	}

	// Extract options
	order := 64 // B+Tree order (max children per node)
	allowNaN := false
	precision := 10 // Decimal precision
	cacheSize := 1000

	if options := config.Options; options != nil {
		if val, ok := options["btree_order"].(int); ok && val > 3 {
			order = val
		}
		if val, ok := options["allow_nan"].(bool); ok {
			allowNaN = val
		}
		if val, ok := options["precision"].(int); ok && val >= 0 {
			precision = val
		}
		if val, ok := options["cache_size"].(int); ok && val > 0 {
			cacheSize = val
		}
	}

	index := &NumericMetadataIndex{
		tree:      NewNumericBTree(order),
		docs:      make(map[string]float64),
		config:    config,
		field:     config.Field,
		closed:    false,
		cache:     NewNumericQueryCache(cacheSize),
		minValue:  math.Inf(1),  // +Inf
		maxValue:  math.Inf(-1), // -Inf
		allowNaN:  allowNaN,
		precision: precision,
		stats: &NumericIndexStats{
			IndexStats: &IndexStats{
				Field:         config.Field,
				Type:          NumericIndex,
				EntryCount:    0,
				DocumentCount: 0,
				MemoryUsage:   0,
				UpdateCount:   0,
				QueryCount:    0,
				AverageTime:   0.0,
				LastUpdate:    time.Now().Unix(),
			},
			MinValue:   math.Inf(1),
			MaxValue:   math.Inf(-1),
			TreeHeight: 0,
			NodeCount:  0,
			LeafCount:  0,
		},
	}

	return index, nil
}

// Field returns the field name this index covers
func (idx *NumericMetadataIndex) Field() string {
	return idx.field
}

// Type returns the index type
func (idx *NumericMetadataIndex) Type() IndexType {
	return NumericIndex
}

// Config returns the index configuration
func (idx *NumericMetadataIndex) Config() *IndexConfig {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Return a copy to prevent modification
	configCopy := *idx.config
	return &configCopy
}

// Add adds a vector ID to the index for the given numeric value
func (idx *NumericMetadataIndex) Add(vectorID string, value interface{}) error {
	if vectorID == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return ErrIndexClosed
	}

	// Convert value to float64
	numValue, err := idx.normalizeValue(value)
	if err != nil {
		return fmt.Errorf("failed to normalize value: %w", err)
	}

	// Check if document already exists
	if existingValue, exists := idx.docs[vectorID]; exists {
		// Remove from old value's position first
		if err := idx.tree.Remove(existingValue, vectorID); err != nil {
			return fmt.Errorf("failed to remove from old position: %w", err)
		}
	} else {
		idx.stats.DocumentCount++
	}

	// Add to new position in tree
	if err := idx.tree.Insert(numValue, vectorID); err != nil {
		return fmt.Errorf("failed to insert into tree: %w", err)
	}

	// Update document mapping
	idx.docs[vectorID] = numValue

	// Update min/max values
	if numValue < idx.minValue {
		idx.minValue = numValue
		idx.stats.MinValue = numValue
	}
	if numValue > idx.maxValue {
		idx.maxValue = numValue
		idx.stats.MaxValue = numValue
	}

	// Clear cache as it may be invalidated
	idx.cache.Clear()

	// Update statistics
	idx.stats.UpdateCount++
	idx.stats.LastUpdate = time.Now().Unix()

	return nil
}

// Remove removes a vector ID from the index
func (idx *NumericMetadataIndex) Remove(vectorID string, value interface{}) error {
	if vectorID == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return ErrIndexClosed
	}

	// Get current value for this document
	currentValue, exists := idx.docs[vectorID]
	if !exists {
		return nil // Already removed or never existed
	}

	// Remove from tree
	if err := idx.tree.Remove(currentValue, vectorID); err != nil {
		return fmt.Errorf("failed to remove from tree: %w", err)
	}

	// Remove document mapping
	delete(idx.docs, vectorID)
	idx.stats.DocumentCount--

	// Update min/max if necessary (recalculate from remaining docs)
	if len(idx.docs) == 0 {
		idx.minValue = math.Inf(1)
		idx.maxValue = math.Inf(-1)
		idx.stats.MinValue = math.Inf(1)
		idx.stats.MaxValue = math.Inf(-1)
	} else if currentValue == idx.minValue || currentValue == idx.maxValue {
		idx.recalculateMinMax()
	}

	// Clear cache
	idx.cache.Clear()

	// Update statistics
	idx.stats.UpdateCount++
	idx.stats.LastUpdate = time.Now().Unix()

	return nil
}

// Update updates a vector ID's value in the index
func (idx *NumericMetadataIndex) Update(vectorID string, oldValue, newValue interface{}) error {
	if vectorID == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return ErrIndexClosed
	}

	// Normalize values
	oldNum, err := idx.normalizeValue(oldValue)
	if err != nil {
		return fmt.Errorf("failed to normalize old value: %w", err)
	}

	newNum, err := idx.normalizeValue(newValue)
	if err != nil {
		return fmt.Errorf("failed to normalize new value: %w", err)
	}

	// If values are effectively the same, no update needed
	if idx.compareValues(oldNum, newNum) == 0 {
		return nil
	}

	// Remove from old position
	if err := idx.tree.Remove(oldNum, vectorID); err != nil {
		return fmt.Errorf("failed to remove from old position: %w", err)
	}

	// Add to new position
	if err := idx.tree.Insert(newNum, vectorID); err != nil {
		return fmt.Errorf("failed to insert to new position: %w", err)
	}

	// Update document mapping
	idx.docs[vectorID] = newNum

	// Update min/max values
	if newNum < idx.minValue {
		idx.minValue = newNum
		idx.stats.MinValue = newNum
	}
	if newNum > idx.maxValue {
		idx.maxValue = newNum
		idx.stats.MaxValue = newNum
	}

	// Clear cache
	idx.cache.Clear()

	// Update statistics
	idx.stats.UpdateCount++
	idx.stats.LastUpdate = time.Now().Unix()

	return nil
}

// Clear removes all entries from the index
func (idx *NumericMetadataIndex) Clear() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return ErrIndexClosed
	}

	idx.tree = NewNumericBTree(idx.tree.order)
	idx.docs = make(map[string]float64)
	idx.cache.Clear()

	// Reset min/max
	idx.minValue = math.Inf(1)
	idx.maxValue = math.Inf(-1)

	// Reset statistics
	idx.stats.EntryCount = 0
	idx.stats.DocumentCount = 0
	idx.stats.MinValue = math.Inf(1)
	idx.stats.MaxValue = math.Inf(-1)
	idx.stats.TreeHeight = 0
	idx.stats.NodeCount = 0
	idx.stats.LeafCount = 0
	idx.stats.UpdateCount++
	idx.stats.LastUpdate = time.Now().Unix()

	return nil
}

// Query processes a filter expression and returns matching vector IDs
func (idx *NumericMetadataIndex) Query(ctx context.Context, expr api.FilterExpr) ([]string, error) {
	if expr == nil {
		return nil, fmt.Errorf("filter expression cannot be nil")
	}

	start := time.Now()
	defer func() {
		duration := time.Since(start)
		idx.updateQueryStats(duration)
	}()

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.closed {
		return nil, ErrIndexClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	return idx.processFilterExpression(ctx, expr)
}

// Contains checks if the index contains the given value
func (idx *NumericMetadataIndex) Contains(value interface{}) bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.closed {
		return false
	}

	numValue, err := idx.normalizeValue(value)
	if err != nil {
		return false
	}

	return idx.tree.Contains(numValue)
}

// Size returns the number of unique values in the index
func (idx *NumericMetadataIndex) Size() int64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return idx.tree.UniqueValueCount()
}

// Stats returns index statistics
func (idx *NumericMetadataIndex) Stats() *IndexStats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Update computed statistics
	idx.updateComputedStats()

	// Return a copy
	statsCopy := *idx.stats.IndexStats
	return &statsCopy
}

// Validate validates the index consistency
func (idx *NumericMetadataIndex) Validate() error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.closed {
		return ErrIndexClosed
	}

	// Check document count consistency
	if int64(len(idx.docs)) != idx.stats.DocumentCount {
		return fmt.Errorf("document count mismatch: expected %d, got %d",
			idx.stats.DocumentCount, len(idx.docs))
	}

	// Validate tree structure
	if err := idx.tree.Validate(); err != nil {
		return fmt.Errorf("tree validation failed: %w", err)
	}

	// Validate all document values exist in tree
	for vectorID, value := range idx.docs {
		if !idx.tree.ContainsValue(value, vectorID) {
			return fmt.Errorf("document %s with value %f not found in tree", vectorID, value)
		}
	}

	return nil
}

// Close closes the index and releases resources
func (idx *NumericMetadataIndex) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return ErrIndexClosed
	}

	idx.closed = true
	idx.tree = nil
	idx.docs = nil
	idx.cache.Clear()

	return nil
}

// IsClosed returns whether the index is closed
func (idx *NumericMetadataIndex) IsClosed() bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return idx.closed
}

// Private helper methods

// normalizeValue converts a value to float64 for indexing
func (idx *NumericMetadataIndex) normalizeValue(value interface{}) (float64, error) {
	if value == nil {
		return math.NaN(), nil
	}

	var numValue float64
	switch v := value.(type) {
	case float64:
		numValue = v
	case float32:
		numValue = float64(v)
	case int:
		numValue = float64(v)
	case int32:
		numValue = float64(v)
	case int64:
		numValue = float64(v)
	case uint:
		numValue = float64(v)
	case uint32:
		numValue = float64(v)
	case uint64:
		numValue = float64(v)
	default:
		return 0, fmt.Errorf("cannot convert %T to numeric value", value)
	}

	// Check for special values
	if math.IsNaN(numValue) && !idx.allowNaN {
		return 0, fmt.Errorf("NaN values not allowed")
	}

	return numValue, nil
}

// compareValues compares two float64 values with precision consideration
func (idx *NumericMetadataIndex) compareValues(a, b float64) int {
	if idx.precision <= 0 {
		// Direct comparison
		if a < b {
			return -1
		} else if a > b {
			return 1
		}
		return 0
	}

	// Precision-based comparison
	multiplier := math.Pow(10, float64(idx.precision))
	aRounded := math.Round(a*multiplier) / multiplier
	bRounded := math.Round(b*multiplier) / multiplier

	if aRounded < bRounded {
		return -1
	} else if aRounded > bRounded {
		return 1
	}
	return 0
}

// recalculateMinMax recalculates min and max values from all documents
func (idx *NumericMetadataIndex) recalculateMinMax() {
	if len(idx.docs) == 0 {
		idx.minValue = math.Inf(1)
		idx.maxValue = math.Inf(-1)
		idx.stats.MinValue = math.Inf(1)
		idx.stats.MaxValue = math.Inf(-1)
		return
	}

	min := math.Inf(1)
	max := math.Inf(-1)

	for _, value := range idx.docs {
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
	}

	idx.minValue = min
	idx.maxValue = max
	idx.stats.MinValue = min
	idx.stats.MaxValue = max
}

// processFilterExpression processes a filter expression and returns matching vector IDs
func (idx *NumericMetadataIndex) processFilterExpression(ctx context.Context, expr api.FilterExpr) ([]string, error) {
	switch filter := expr.(type) {
	case *api.FieldFilter:
		return idx.processFieldFilter(ctx, filter)
	case *api.LogicalFilter:
		return idx.processLogicalFilter(ctx, filter)
	default:
		return nil, fmt.Errorf("unsupported filter type: %T", expr)
	}
}

// processFieldFilter processes a field-based filter
func (idx *NumericMetadataIndex) processFieldFilter(ctx context.Context, filter *api.FieldFilter) ([]string, error) {
	// Only process filters for our field
	if filter.Field != idx.field {
		return nil, fmt.Errorf("filter field %s does not match index field %s", filter.Field, idx.field)
	}

	switch filter.Op {
	case api.FilterEq:
		return idx.queryEqual(ctx, filter.Value)
	case api.FilterNe:
		return idx.queryNotEqual(ctx, filter.Value)
	case api.FilterGt:
		return idx.queryGreaterThan(ctx, filter.Value, false)
	case api.FilterGte:
		return idx.queryGreaterThan(ctx, filter.Value, true)
	case api.FilterLt:
		return idx.queryLessThan(ctx, filter.Value, false)
	case api.FilterLte:
		return idx.queryLessThan(ctx, filter.Value, true)
	case api.FilterIn:
		return idx.queryIn(ctx, filter.Value)
	case api.FilterNin:
		return idx.queryNotIn(ctx, filter.Value)
	default:
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedOperator, filter.Op)
	}
}

// processLogicalFilter processes logical filters (AND, OR, NOT)
func (idx *NumericMetadataIndex) processLogicalFilter(ctx context.Context, filter *api.LogicalFilter) ([]string, error) {
	switch filter.Op {
	case api.FilterAnd:
		return idx.queryAnd(ctx, filter.Filters)
	case api.FilterOr:
		return idx.queryOr(ctx, filter.Filters)
	case api.FilterNot:
		if len(filter.Filters) != 1 {
			return nil, fmt.Errorf("NOT filter must have exactly one sub-filter")
		}
		return idx.queryNot(ctx, filter.Filters[0])
	default:
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedOperator, filter.Op)
	}
}

// Query implementations for different operators

// queryEqual handles equality queries
func (idx *NumericMetadataIndex) queryEqual(ctx context.Context, value interface{}) ([]string, error) {
	start := time.Now()
	defer func() {
		idx.stats.EqualityQueries++
		idx.stats.AvgEqualityTime = updateAverage(idx.stats.AvgEqualityTime,
			float64(time.Since(start).Nanoseconds())/1e6, idx.stats.EqualityQueries)
	}()

	numValue, err := idx.normalizeValue(value)
	if err != nil {
		return nil, err
	}

	// Check cache first
	cacheKey := fmt.Sprintf("eq:%g", numValue)
	if cached := idx.cache.GetComparison(cacheKey); cached != nil {
		return cached.VectorIDs, nil
	}

	// Query tree
	vectorIDs := idx.tree.FindEqual(numValue)

	// Cache result
	idx.cache.PutComparison(cacheKey, &NumericCacheEntry{
		VectorIDs:    vectorIDs,
		ResultCount:  int64(len(vectorIDs)),
		LastAccessed: time.Now(),
		AccessCount:  1,
		QueryType:    "equality",
	})

	return vectorIDs, nil
}

// queryNotEqual handles not-equal queries
func (idx *NumericMetadataIndex) queryNotEqual(ctx context.Context, value interface{}) ([]string, error) {
	// Get all equal values
	equalResults, err := idx.queryEqual(ctx, value)
	if err != nil {
		return nil, err
	}

	// Get all vector IDs
	allResults := make([]string, 0, len(idx.docs))
	for id := range idx.docs {
		allResults = append(allResults, id)
	}

	// Return complement
	return removeStrings(allResults, equalResults), nil
}

// queryGreaterThan handles greater than queries
func (idx *NumericMetadataIndex) queryGreaterThan(ctx context.Context, value interface{}, inclusive bool) ([]string, error) {
	start := time.Now()
	defer func() {
		idx.stats.ComparisonQueries++
		idx.stats.AvgComparisonTime = updateAverage(idx.stats.AvgComparisonTime,
			float64(time.Since(start).Nanoseconds())/1e6, idx.stats.ComparisonQueries)
	}()

	numValue, err := idx.normalizeValue(value)
	if err != nil {
		return nil, err
	}

	// Check cache
	var cacheKey string
	if inclusive {
		cacheKey = fmt.Sprintf("gte:%g", numValue)
	} else {
		cacheKey = fmt.Sprintf("gt:%g", numValue)
	}

	if cached := idx.cache.GetComparison(cacheKey); cached != nil {
		return cached.VectorIDs, nil
	}

	// Query tree for range
	var vectorIDs []string
	if inclusive {
		vectorIDs = idx.tree.FindRange(numValue, math.Inf(1), true, false)
	} else {
		vectorIDs = idx.tree.FindRange(numValue, math.Inf(1), false, false)
	}

	// Cache result
	idx.cache.PutComparison(cacheKey, &NumericCacheEntry{
		VectorIDs:    vectorIDs,
		ResultCount:  int64(len(vectorIDs)),
		LastAccessed: time.Now(),
		AccessCount:  1,
		QueryType:    "comparison",
	})

	return vectorIDs, nil
}

// queryLessThan handles less than queries
func (idx *NumericMetadataIndex) queryLessThan(ctx context.Context, value interface{}, inclusive bool) ([]string, error) {
	start := time.Now()
	defer func() {
		idx.stats.ComparisonQueries++
		idx.stats.AvgComparisonTime = updateAverage(idx.stats.AvgComparisonTime,
			float64(time.Since(start).Nanoseconds())/1e6, idx.stats.ComparisonQueries)
	}()

	numValue, err := idx.normalizeValue(value)
	if err != nil {
		return nil, err
	}

	// Check cache
	var cacheKey string
	if inclusive {
		cacheKey = fmt.Sprintf("lte:%g", numValue)
	} else {
		cacheKey = fmt.Sprintf("lt:%g", numValue)
	}

	if cached := idx.cache.GetComparison(cacheKey); cached != nil {
		return cached.VectorIDs, nil
	}

	// Query tree for range
	var vectorIDs []string
	if inclusive {
		vectorIDs = idx.tree.FindRange(math.Inf(-1), numValue, false, true)
	} else {
		vectorIDs = idx.tree.FindRange(math.Inf(-1), numValue, false, false)
	}

	// Cache result
	idx.cache.PutComparison(cacheKey, &NumericCacheEntry{
		VectorIDs:    vectorIDs,
		ResultCount:  int64(len(vectorIDs)),
		LastAccessed: time.Now(),
		AccessCount:  1,
		QueryType:    "comparison",
	})

	return vectorIDs, nil
}

// queryIn handles membership queries (IN operator)
func (idx *NumericMetadataIndex) queryIn(ctx context.Context, values interface{}) ([]string, error) {
	// Convert to slice of interfaces
	valueSlice, ok := values.([]interface{})
	if !ok {
		return nil, fmt.Errorf("IN operator requires array of values")
	}

	if len(valueSlice) == 0 {
		return []string{}, nil
	}

	// Union all results for the specified values
	result := make(map[string]struct{})

	for _, value := range valueSlice {
		vectorIDs, err := idx.queryEqual(ctx, value)
		if err != nil {
			continue // Skip invalid values
		}

		for _, id := range vectorIDs {
			result[id] = struct{}{}
		}
	}

	// Convert to slice
	vectorIDs := make([]string, 0, len(result))
	for id := range result {
		vectorIDs = append(vectorIDs, id)
	}

	return vectorIDs, nil
}

// queryNotIn handles not-in-membership queries (NIN operator)
func (idx *NumericMetadataIndex) queryNotIn(ctx context.Context, values interface{}) ([]string, error) {
	// Get all vector IDs that are IN the specified values
	inResults, err := idx.queryIn(ctx, values)
	if err != nil {
		return nil, err
	}

	// Get all vector IDs in the index
	allResults := make([]string, 0, idx.stats.DocumentCount)
	for vectorID := range idx.docs {
		allResults = append(allResults, vectorID)
	}

	// Return complement
	return removeStrings(allResults, inResults), nil
}

// Logical operators (reuse from inverted index)

// queryAnd performs intersection of multiple filter results
func (idx *NumericMetadataIndex) queryAnd(ctx context.Context, filters []api.FilterExpr) ([]string, error) {
	if len(filters) == 0 {
		return []string{}, nil
	}

	// Process first filter
	result, err := idx.processFilterExpression(ctx, filters[0])
	if err != nil {
		return nil, err
	}

	// Intersect with remaining filters
	for i := 1; i < len(filters); i++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		nextResult, err := idx.processFilterExpression(ctx, filters[i])
		if err != nil {
			return nil, err
		}

		result = intersectStrings(result, nextResult)

		// Early termination if result is empty
		if len(result) == 0 {
			break
		}
	}

	return result, nil
}

// queryOr performs union of multiple filter results
func (idx *NumericMetadataIndex) queryOr(ctx context.Context, filters []api.FilterExpr) ([]string, error) {
	if len(filters) == 0 {
		return []string{}, nil
	}

	resultSet := make(map[string]struct{})

	for _, filter := range filters {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		result, err := idx.processFilterExpression(ctx, filter)
		if err != nil {
			return nil, err
		}

		for _, id := range result {
			resultSet[id] = struct{}{}
		}
	}

	// Convert to slice
	result := make([]string, 0, len(resultSet))
	for id := range resultSet {
		result = append(result, id)
	}

	return result, nil
}

// queryNot performs complement of a filter result
func (idx *NumericMetadataIndex) queryNot(ctx context.Context, filter api.FilterExpr) ([]string, error) {
	// Get filter result
	filterResult, err := idx.processFilterExpression(ctx, filter)
	if err != nil {
		return nil, err
	}

	// Get all vector IDs
	allIDs := make([]string, 0, len(idx.docs))
	for id := range idx.docs {
		allIDs = append(allIDs, id)
	}

	// Return complement
	return removeStrings(allIDs, filterResult), nil
}

// updateQueryStats updates query performance statistics
func (idx *NumericMetadataIndex) updateQueryStats(duration time.Duration) {
	idx.stats.QueryCount++
	durationMs := float64(duration.Nanoseconds()) / 1e6
	idx.stats.AverageTime = updateAverage(idx.stats.AverageTime, durationMs, idx.stats.QueryCount)
}

// updateComputedStats updates computed statistics
func (idx *NumericMetadataIndex) updateComputedStats() {
	// Update tree statistics
	idx.stats.TreeHeight = idx.tree.height
	idx.stats.NodeCount = idx.tree.nodeCount
	idx.stats.UniqueValues = idx.tree.UniqueValueCount()

	// Count null/special values
	nullCount := int64(0)
	nanCount := int64(0)
	infCount := int64(0)

	for _, value := range idx.docs {
		if math.IsNaN(value) {
			nanCount++
		} else if math.IsInf(value, 0) {
			infCount++
		}
	}

	idx.stats.NullValues = nullCount
	idx.stats.NaNValues = nanCount
	idx.stats.InfiniteValues = infCount

	// Estimate memory usage
	memoryUsage := int64(0)
	memoryUsage += int64(len(idx.docs) * 100) // Document mapping
	memoryUsage += idx.tree.EstimateMemoryUsage()

	idx.stats.MemoryUsage = memoryUsage
}

// NumericBTree implementation

// NewNumericBTree creates a new B+Tree for numeric values
func NewNumericBTree(order int) *NumericBTree {
	if order < 4 {
		order = 4 // Minimum order for B+Tree
	}

	return &NumericBTree{
		root:      nil,
		order:     order,
		height:    0,
		nodeCount: 0,
	}
}

// Insert inserts a value with associated vector ID into the tree
func (tree *NumericBTree) Insert(value float64, vectorID string) error {
	tree.mu.Lock()
	defer tree.mu.Unlock()

	if tree.root == nil {
		// Create root node
		tree.root = &NumericNode{
			isLeaf:   true,
			keys:     []float64{value},
			values:   [][]string{{vectorID}},
			children: nil,
		}
		tree.height = 1
		tree.nodeCount = 1
		return nil
	}

	// Insert into existing tree
	newRoot, err := tree.insertIntoNode(tree.root, value, vectorID)
	if err != nil {
		return err
	}

	if newRoot != nil {
		// Root was split, create new root
		tree.root = newRoot
		tree.height++
	}

	return nil
}

// Remove removes a value-vectorID pair from the tree
func (tree *NumericBTree) Remove(value float64, vectorID string) error {
	tree.mu.Lock()
	defer tree.mu.Unlock()

	if tree.root == nil {
		return nil // Empty tree
	}

	removed := tree.removeFromNode(tree.root, value, vectorID)
	if !removed {
		return nil // Not found
	}

	// Check if root became empty
	if len(tree.root.keys) == 0 && !tree.root.isLeaf {
		if len(tree.root.children) > 0 {
			tree.root = tree.root.children[0]
			tree.height--
			tree.nodeCount--
		} else {
			tree.root = nil
			tree.height = 0
			tree.nodeCount = 0
		}
	}

	return nil
}

// FindEqual finds all vector IDs with the exact value
func (tree *NumericBTree) FindEqual(value float64) []string {
	tree.mu.RLock()
	defer tree.mu.RUnlock()

	if tree.root == nil {
		return []string{}
	}

	return tree.findEqualInNode(tree.root, value)
}

// FindRange finds all vector IDs within the specified range
func (tree *NumericBTree) FindRange(minVal, maxVal float64, includeMin, includeMax bool) []string {
	tree.mu.RLock()
	defer tree.mu.RUnlock()

	if tree.root == nil {
		return []string{}
	}

	var result []string
	tree.findRangeInNode(tree.root, minVal, maxVal, includeMin, includeMax, &result)
	return result
}

// Contains checks if the tree contains the specified value
func (tree *NumericBTree) Contains(value float64) bool {
	return len(tree.FindEqual(value)) > 0
}

// ContainsValue checks if the tree contains the value-vectorID pair
func (tree *NumericBTree) ContainsValue(value float64, vectorID string) bool {
	vectorIDs := tree.FindEqual(value)
	for _, id := range vectorIDs {
		if id == vectorID {
			return true
		}
	}
	return false
}

// UniqueValueCount returns the number of unique values in the tree
func (tree *NumericBTree) UniqueValueCount() int64 {
	tree.mu.RLock()
	defer tree.mu.RUnlock()

	if tree.root == nil {
		return 0
	}

	return tree.countUniqueValues(tree.root)
}

// EstimateMemoryUsage estimates the memory usage of the tree
func (tree *NumericBTree) EstimateMemoryUsage() int64 {
	// Rough estimate based on node count and average node size
	avgNodeSize := int64(tree.order * 100) // Rough estimate per node
	return tree.nodeCount * avgNodeSize
}

// Validate validates the tree structure
func (tree *NumericBTree) Validate() error {
	tree.mu.RLock()
	defer tree.mu.RUnlock()

	if tree.root == nil {
		return nil
	}

	return tree.validateNode(tree.root, math.Inf(-1), math.Inf(1))
}

// Private B+Tree methods (simplified implementation for brevity)

// insertIntoNode inserts a value into a node (recursive)
func (tree *NumericBTree) insertIntoNode(node *NumericNode, value float64, vectorID string) (*NumericNode, error) {
	node.mu.Lock()
	defer node.mu.Unlock()

	if node.isLeaf {
		// Find insertion point
		i := sort.SearchFloat64s(node.keys, value)

		if i < len(node.keys) && node.keys[i] == value {
			// Value exists, add vector ID to existing list
			node.values[i] = append(node.values[i], vectorID)
		} else {
			// Insert new key-value pair
			node.keys = insertFloat64At(node.keys, i, value)
			node.values = insertStringSliceAt(node.values, i, []string{vectorID})
		}

		// Check if node needs to be split
		if len(node.keys) > tree.order-1 {
			return tree.splitLeafNode(node), nil
		}

		return nil, nil
	} else {
		// Internal node - find child to insert into
		i := sort.SearchFloat64s(node.keys, value)
		if i >= len(node.children) {
			i = len(node.children) - 1
		}

		newChild, err := tree.insertIntoNode(node.children[i], value, vectorID)
		if err != nil {
			return nil, err
		}

		if newChild != nil {
			// Child was split, insert new child
			if len(newChild.keys) > 0 {
				promotedKey := newChild.keys[0]
				node.keys = insertFloat64At(node.keys, i, promotedKey)
				node.children = insertNodeAt(node.children, i+1, newChild)
			}

			// Check if this node needs to be split
			if len(node.keys) > tree.order-1 {
				return tree.splitInternalNode(node), nil
			}
		}

		return nil, nil
	}
}

// Helper methods for B+Tree operations (simplified)

func (tree *NumericBTree) splitLeafNode(node *NumericNode) *NumericNode {
	mid := len(node.keys) / 2

	newNode := &NumericNode{
		isLeaf: true,
		keys:   make([]float64, len(node.keys)-mid),
		values: make([][]string, len(node.values)-mid),
		next:   node.next,
		prev:   node,
	}

	copy(newNode.keys, node.keys[mid:])
	copy(newNode.values, node.values[mid:])

	node.keys = node.keys[:mid]
	node.values = node.values[:mid]
	node.next = newNode

	if newNode.next != nil {
		newNode.next.prev = newNode
	}

	tree.nodeCount++
	return newNode
}

func (tree *NumericBTree) splitInternalNode(node *NumericNode) *NumericNode {
	mid := len(node.keys) / 2

	newNode := &NumericNode{
		isLeaf:   false,
		keys:     make([]float64, len(node.keys)-mid-1),
		children: make([]*NumericNode, len(node.children)-mid-1),
	}

	copy(newNode.keys, node.keys[mid+1:])
	copy(newNode.children, node.children[mid+1:])

	promotedKey := node.keys[mid]
	node.keys = node.keys[:mid]
	node.children = node.children[:mid+1]

	// Create new root
	newRoot := &NumericNode{
		isLeaf:   false,
		keys:     []float64{promotedKey},
		children: []*NumericNode{node, newNode},
	}

	tree.nodeCount += 2
	return newRoot
}

func (tree *NumericBTree) removeFromNode(node *NumericNode, value float64, vectorID string) bool {
	node.mu.Lock()
	defer node.mu.Unlock()

	if node.isLeaf {
		i := sort.SearchFloat64s(node.keys, value)
		if i < len(node.keys) && node.keys[i] == value {
			// Remove vector ID from the list
			node.values[i] = removeStringFromSlice(node.values[i], vectorID)

			// If no more vector IDs for this value, remove the key
			if len(node.values[i]) == 0 {
				node.keys = removeFloat64At(node.keys, i)
				node.values = removeStringSliceAt(node.values, i)
			}
			return true
		}
		return false
	} else {
		// Internal node - find child
		i := sort.SearchFloat64s(node.keys, value)
		if i >= len(node.children) {
			i = len(node.children) - 1
		}
		return tree.removeFromNode(node.children[i], value, vectorID)
	}
}

func (tree *NumericBTree) findEqualInNode(node *NumericNode, value float64) []string {
	node.mu.RLock()
	defer node.mu.RUnlock()

	if node.isLeaf {
		i := sort.SearchFloat64s(node.keys, value)
		if i < len(node.keys) && node.keys[i] == value {
			result := make([]string, len(node.values[i]))
			copy(result, node.values[i])
			return result
		}
		return []string{}
	} else {
		i := sort.SearchFloat64s(node.keys, value)
		if i >= len(node.children) {
			i = len(node.children) - 1
		}
		return tree.findEqualInNode(node.children[i], value)
	}
}

func (tree *NumericBTree) findRangeInNode(node *NumericNode, minVal, maxVal float64, includeMin, includeMax bool, result *[]string) {
	node.mu.RLock()
	defer node.mu.RUnlock()

	if node.isLeaf {
		for i, key := range node.keys {
			include := false

			if includeMin && includeMax {
				include = key >= minVal && key <= maxVal
			} else if includeMin {
				include = key >= minVal && key < maxVal
			} else if includeMax {
				include = key > minVal && key <= maxVal
			} else {
				include = key > minVal && key < maxVal
			}

			if include {
				*result = append(*result, node.values[i]...)
			}
		}
	} else {
		for i, child := range node.children {
			// Determine if this child might contain values in range
			childMinKey := math.Inf(-1)
			childMaxKey := math.Inf(1)

			if i > 0 {
				childMinKey = node.keys[i-1]
			}
			if i < len(node.keys) {
				childMaxKey = node.keys[i]
			}

			// Check if child range overlaps with query range
			if childMaxKey >= minVal && childMinKey <= maxVal {
				tree.findRangeInNode(child, minVal, maxVal, includeMin, includeMax, result)
			}
		}
	}
}

func (tree *NumericBTree) countUniqueValues(node *NumericNode) int64 {
	node.mu.RLock()
	defer node.mu.RUnlock()

	if node.isLeaf {
		return int64(len(node.keys))
	}

	count := int64(0)
	for _, child := range node.children {
		count += tree.countUniqueValues(child)
	}
	return count
}

func (tree *NumericBTree) validateNode(node *NumericNode, minVal, maxVal float64) error {
	node.mu.RLock()
	defer node.mu.RUnlock()

	// Check key ordering
	for i := 1; i < len(node.keys); i++ {
		if node.keys[i-1] >= node.keys[i] {
			return fmt.Errorf("keys not in ascending order")
		}
	}

	// Check bounds
	for _, key := range node.keys {
		if key <= minVal || key >= maxVal {
			return fmt.Errorf("key %f out of bounds [%f, %f]", key, minVal, maxVal)
		}
	}

	if !node.isLeaf {
		// Validate children
		for i, child := range node.children {
			childMin := minVal
			childMax := maxVal

			if i > 0 {
				childMin = node.keys[i-1]
			}
			if i < len(node.keys) {
				childMax = node.keys[i]
			}

			if err := tree.validateNode(child, childMin, childMax); err != nil {
				return err
			}
		}
	}

	return nil
}

// NumericQueryCache implementation

// NewNumericQueryCache creates a new query cache for numeric queries
func NewNumericQueryCache(maxSize int) *NumericQueryCache {
	return &NumericQueryCache{
		rangeCache:      make(map[string]*NumericCacheEntry),
		comparisonCache: make(map[string]*NumericCacheEntry),
		maxSize:         maxSize,
		usage:           make(map[string]time.Time),
	}
}

// GetRange retrieves a cached range query result
func (cache *NumericQueryCache) GetRange(key string) *NumericCacheEntry {
	cache.mu.RLock()
	defer cache.mu.RUnlock()

	entry, exists := cache.rangeCache[key]
	if !exists {
		return nil
	}

	cache.usage[key] = time.Now()
	entry.LastAccessed = time.Now()
	entry.AccessCount++

	return entry
}

// GetComparison retrieves a cached comparison query result
func (cache *NumericQueryCache) GetComparison(key string) *NumericCacheEntry {
	cache.mu.RLock()
	defer cache.mu.RUnlock()

	entry, exists := cache.comparisonCache[key]
	if !exists {
		return nil
	}

	cache.usage[key] = time.Now()
	entry.LastAccessed = time.Now()
	entry.AccessCount++

	return entry
}

// PutRange stores a range query result in the cache
func (cache *NumericQueryCache) PutRange(key string, entry *NumericCacheEntry) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	cache.evictIfNeeded()
	cache.rangeCache[key] = entry
	cache.usage[key] = time.Now()
}

// PutComparison stores a comparison query result in the cache
func (cache *NumericQueryCache) PutComparison(key string, entry *NumericCacheEntry) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	cache.evictIfNeeded()
	cache.comparisonCache[key] = entry
	cache.usage[key] = time.Now()
}

// Clear clears all cached entries
func (cache *NumericQueryCache) Clear() {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	cache.rangeCache = make(map[string]*NumericCacheEntry)
	cache.comparisonCache = make(map[string]*NumericCacheEntry)
	cache.usage = make(map[string]time.Time)
}

// evictIfNeeded evicts entries if cache is at capacity
func (cache *NumericQueryCache) evictIfNeeded() {
	totalSize := len(cache.rangeCache) + len(cache.comparisonCache)
	if totalSize < cache.maxSize {
		return
	}

	// Find and evict LRU entry
	var oldestKey string
	var oldestTime time.Time

	for key, accessTime := range cache.usage {
		if oldestKey == "" || accessTime.Before(oldestTime) {
			oldestKey = key
			oldestTime = accessTime
		}
	}

	if oldestKey != "" {
		delete(cache.rangeCache, oldestKey)
		delete(cache.comparisonCache, oldestKey)
		delete(cache.usage, oldestKey)
	}
}

// Utility functions for slice operations

func insertFloat64At(slice []float64, index int, value float64) []float64 {
	if index == len(slice) {
		return append(slice, value)
	}
	slice = append(slice, 0)
	copy(slice[index+1:], slice[index:])
	slice[index] = value
	return slice
}

func insertStringSliceAt(slice [][]string, index int, value []string) [][]string {
	if index == len(slice) {
		return append(slice, value)
	}
	slice = append(slice, nil)
	copy(slice[index+1:], slice[index:])
	slice[index] = value
	return slice
}

func insertNodeAt(slice []*NumericNode, index int, node *NumericNode) []*NumericNode {
	if index == len(slice) {
		return append(slice, node)
	}
	slice = append(slice, nil)
	copy(slice[index+1:], slice[index:])
	slice[index] = node
	return slice
}

func removeFloat64At(slice []float64, index int) []float64 {
	return append(slice[:index], slice[index+1:]...)
}

func removeStringSliceAt(slice [][]string, index int) [][]string {
	return append(slice[:index], slice[index+1:]...)
}

func removeStringFromSlice(slice []string, target string) []string {
	for i, s := range slice {
		if s == target {
			return append(slice[:i], slice[i+1:]...)
		}
	}
	return slice
}
