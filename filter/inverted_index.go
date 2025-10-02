// Package filter implements efficient inverted index for categorical/string metadata filtering.
package filter

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// InvertedMetadataIndex implements MetadataIndex interface for categorical/string fields.
// It maintains an inverted index mapping from field values to lists of vector IDs,
// enabling extremely fast equality, membership, and prefix matching operations.
type InvertedMetadataIndex struct {
	// Core index data structures
	index map[string]*PostingList // value -> posting list of vector IDs
	docs  map[string]interface{}  // vector ID -> current value for this field

	// Configuration
	config *IndexConfig
	field  string

	// Statistics and monitoring
	stats *InvertedIndexStats

	// Thread safety
	mu     sync.RWMutex
	closed bool

	// Performance optimizations
	cache          *QueryCache
	caseSensitive  bool
	maxValueLength int
}

// PostingList represents a list of vector IDs for a particular field value.
// It uses efficient data structures for set operations (union, intersection).
type PostingList struct {
	// Vector IDs that have this value
	vectorIDs map[string]struct{} // Using map for O(1) operations

	// Cached sorted slice for range operations
	sortedIDs []string
	sorted    bool

	// Statistics
	lastAccessed time.Time
	accessCount  int64

	// Thread safety for individual posting lists
	mu sync.RWMutex
}

// InvertedIndexStats provides detailed statistics for inverted index
type InvertedIndexStats struct {
	*IndexStats

	// Inverted index specific metrics
	UniqueValues   int64   `json:"unique_values"`
	AvgPostingSize float64 `json:"avg_posting_size"`
	MaxPostingSize int64   `json:"max_posting_size"`
	EmptyPostings  int64   `json:"empty_postings"`

	// Query performance
	EqualityQueries   int64   `json:"equality_queries"`
	MembershipQueries int64   `json:"membership_queries"`
	PrefixQueries     int64   `json:"prefix_queries"`
	AvgEqualityTime   float64 `json:"avg_equality_time"`
	AvgMembershipTime float64 `json:"avg_membership_time"`
	AvgPrefixTime     float64 `json:"avg_prefix_time"`

	// Cache statistics
	CacheHits   int64 `json:"cache_hits"`
	CacheMisses int64 `json:"cache_misses"`
	CacheSize   int64 `json:"cache_size"`
}

// QueryCache implements LRU cache for frequently accessed queries
type QueryCache struct {
	cache   map[string]*CacheEntry
	usage   map[string]time.Time
	maxSize int
	mu      sync.RWMutex
}

// CacheEntry represents a cached query result
type CacheEntry struct {
	VectorIDs    []string
	LastAccessed time.Time
	AccessCount  int64
}

// NewInvertedMetadataIndex creates a new inverted index for the specified field
func NewInvertedMetadataIndex(config *IndexConfig) (*InvertedMetadataIndex, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	if config.Type != InvertedIndex {
		return nil, fmt.Errorf("%w: expected %s, got %s", ErrInvalidIndexType, InvertedIndex, config.Type)
	}

	// Extract options
	caseSensitive := true
	maxValueLength := 1000
	cacheSize := 1000

	if options := config.Options; options != nil {
		if val, ok := options["case_sensitive"].(bool); ok {
			caseSensitive = val
		}
		if val, ok := options["max_value_length"].(int); ok {
			maxValueLength = val
		}
		if val, ok := options["cache_size"].(int); ok {
			cacheSize = val
		}
	}

	index := &InvertedMetadataIndex{
		index:          make(map[string]*PostingList),
		docs:           make(map[string]interface{}),
		config:         config,
		field:          config.Field,
		closed:         false,
		caseSensitive:  caseSensitive,
		maxValueLength: maxValueLength,
		cache:          NewQueryCache(cacheSize),
		stats: &InvertedIndexStats{
			IndexStats: &IndexStats{
				Field:         config.Field,
				Type:          InvertedIndex,
				EntryCount:    0,
				DocumentCount: 0,
				MemoryUsage:   0,
				UpdateCount:   0,
				QueryCount:    0,
				AverageTime:   0.0,
				LastUpdate:    time.Now().Unix(),
			},
		},
	}

	return index, nil
}

// Field returns the field name this index covers
func (idx *InvertedMetadataIndex) Field() string {
	return idx.field
}

// Type returns the index type
func (idx *InvertedMetadataIndex) Type() IndexType {
	return InvertedIndex
}

// Config returns the index configuration
func (idx *InvertedMetadataIndex) Config() *IndexConfig {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Return a copy to prevent modification
	configCopy := *idx.config
	return &configCopy
}

// Add adds a vector ID to the index for the given value
func (idx *InvertedMetadataIndex) Add(vectorID string, value interface{}) error {
	if vectorID == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return ErrIndexClosed
	}

	// Convert value to string
	strValue, err := idx.normalizeValue(value)
	if err != nil {
		return fmt.Errorf("failed to normalize value: %w", err)
	}

	// Check if document already exists
	if existingValue, exists := idx.docs[vectorID]; exists {
		// Remove from old value's posting list first
		if err := idx.removeFromPostingList(vectorID, existingValue); err != nil {
			return fmt.Errorf("failed to remove from old posting list: %w", err)
		}
	} else {
		idx.stats.DocumentCount++
	}

	// Add to new value's posting list
	if err := idx.addToPostingList(vectorID, strValue); err != nil {
		return fmt.Errorf("failed to add to posting list: %w", err)
	}

	// Update document mapping
	idx.docs[vectorID] = strValue

	// Clear cache as it may be invalidated
	idx.cache.Clear()

	// Update statistics
	idx.stats.UpdateCount++
	idx.stats.LastUpdate = time.Now().Unix()

	return nil
}

// Remove removes a vector ID from the index
func (idx *InvertedMetadataIndex) Remove(vectorID string, value interface{}) error {
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

	// Remove from posting list
	if err := idx.removeFromPostingList(vectorID, currentValue); err != nil {
		return fmt.Errorf("failed to remove from posting list: %w", err)
	}

	// Remove document mapping
	delete(idx.docs, vectorID)
	idx.stats.DocumentCount--

	// Clear cache
	idx.cache.Clear()

	// Update statistics
	idx.stats.UpdateCount++
	idx.stats.LastUpdate = time.Now().Unix()

	return nil
}

// Update updates a vector ID's value in the index
func (idx *InvertedMetadataIndex) Update(vectorID string, oldValue, newValue interface{}) error {
	if vectorID == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return ErrIndexClosed
	}

	// Normalize values
	oldStr, err := idx.normalizeValue(oldValue)
	if err != nil {
		return fmt.Errorf("failed to normalize old value: %w", err)
	}

	newStr, err := idx.normalizeValue(newValue)
	if err != nil {
		return fmt.Errorf("failed to normalize new value: %w", err)
	}

	// If values are the same, no update needed
	if oldStr == newStr {
		return nil
	}

	// Remove from old posting list
	if err := idx.removeFromPostingList(vectorID, oldStr); err != nil {
		return fmt.Errorf("failed to remove from old posting list: %w", err)
	}

	// Add to new posting list
	if err := idx.addToPostingList(vectorID, newStr); err != nil {
		return fmt.Errorf("failed to add to new posting list: %w", err)
	}

	// Update document mapping
	idx.docs[vectorID] = newStr

	// Clear cache
	idx.cache.Clear()

	// Update statistics
	idx.stats.UpdateCount++
	idx.stats.LastUpdate = time.Now().Unix()

	return nil
}

// Clear removes all entries from the index
func (idx *InvertedMetadataIndex) Clear() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return ErrIndexClosed
	}

	idx.index = make(map[string]*PostingList)
	idx.docs = make(map[string]interface{})
	idx.cache.Clear()

	// Reset statistics
	idx.stats.EntryCount = 0
	idx.stats.DocumentCount = 0
	idx.stats.UniqueValues = 0
	idx.stats.UpdateCount++
	idx.stats.LastUpdate = time.Now().Unix()

	return nil
}

// Query processes a filter expression and returns matching vector IDs
func (idx *InvertedMetadataIndex) Query(ctx context.Context, expr api.FilterExpr) ([]string, error) {
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
func (idx *InvertedMetadataIndex) Contains(value interface{}) bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.closed {
		return false
	}

	strValue, err := idx.normalizeValue(value)
	if err != nil {
		return false
	}

	_, exists := idx.index[strValue]
	return exists
}

// Size returns the number of unique values in the index
func (idx *InvertedMetadataIndex) Size() int64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return int64(len(idx.index))
}

// Stats returns index statistics
func (idx *InvertedMetadataIndex) Stats() *IndexStats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Update computed statistics
	idx.updateComputedStats()

	// Return a copy
	statsCopy := *idx.stats.IndexStats
	return &statsCopy
}

// Validate validates the index consistency
func (idx *InvertedMetadataIndex) Validate() error {
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

	// Check posting list consistency
	docCount := int64(0)
	for value, postingList := range idx.index {
		if postingList == nil {
			return fmt.Errorf("nil posting list for value: %s", value)
		}

		postingList.mu.RLock()
		docCount += int64(len(postingList.vectorIDs))
		postingList.mu.RUnlock()
	}

	// Validate each posting list
	for _, postingList := range idx.index {
		if err := postingList.validate(); err != nil {
			return fmt.Errorf("posting list validation failed: %w", err)
		}
	}

	return nil
}

// Close closes the index and releases resources
func (idx *InvertedMetadataIndex) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return ErrIndexClosed
	}

	idx.closed = true
	idx.index = nil
	idx.docs = nil
	idx.cache.Clear()

	return nil
}

// IsClosed returns whether the index is closed
func (idx *InvertedMetadataIndex) IsClosed() bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return idx.closed
}

// Private helper methods

// normalizeValue converts a value to its string representation for indexing
func (idx *InvertedMetadataIndex) normalizeValue(value interface{}) (string, error) {
	if value == nil {
		return "", nil
	}

	var strValue string
	switch v := value.(type) {
	case string:
		strValue = v
	case fmt.Stringer:
		strValue = v.String()
	default:
		strValue = fmt.Sprintf("%v", v)
	}

	// Apply case sensitivity
	if !idx.caseSensitive {
		strValue = strings.ToLower(strValue)
	}

	// Check length limit
	if len(strValue) > idx.maxValueLength {
		return "", fmt.Errorf("value too long: %d > %d", len(strValue), idx.maxValueLength)
	}

	return strValue, nil
}

// addToPostingList adds a vector ID to the posting list for a value
func (idx *InvertedMetadataIndex) addToPostingList(vectorID, value string) error {
	postingList, exists := idx.index[value]
	if !exists {
		postingList = NewPostingList()
		idx.index[value] = postingList
		idx.stats.EntryCount++
		idx.stats.UniqueValues++
	}

	return postingList.Add(vectorID)
}

// removeFromPostingList removes a vector ID from the posting list for a value
func (idx *InvertedMetadataIndex) removeFromPostingList(vectorID string, value interface{}) error {
	strValue, err := idx.normalizeValue(value)
	if err != nil {
		return err
	}

	postingList, exists := idx.index[strValue]
	if !exists {
		return nil // Nothing to remove
	}

	if err := postingList.Remove(vectorID); err != nil {
		return err
	}

	// Remove empty posting lists to save memory
	if postingList.Size() == 0 {
		delete(idx.index, strValue)
		idx.stats.EntryCount--
		idx.stats.UniqueValues--
		idx.stats.EmptyPostings++
	}

	return nil
}

// processFilterExpression processes a filter expression and returns matching vector IDs
func (idx *InvertedMetadataIndex) processFilterExpression(ctx context.Context, expr api.FilterExpr) ([]string, error) {
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
func (idx *InvertedMetadataIndex) processFieldFilter(ctx context.Context, filter *api.FieldFilter) ([]string, error) {
	// Only process filters for our field
	if filter.Field != idx.field {
		return nil, fmt.Errorf("filter field %s does not match index field %s", filter.Field, idx.field)
	}

	switch filter.Op {
	case api.FilterEq:
		return idx.queryEqual(ctx, filter.Value)
	case api.FilterNe:
		return idx.queryNotEqual(ctx, filter.Value)
	case api.FilterIn:
		return idx.queryIn(ctx, filter.Value)
	case api.FilterNin:
		return idx.queryNotIn(ctx, filter.Value)
	case api.FilterMatch:
		return idx.queryMatch(ctx, filter.Value)
	default:
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedOperator, filter.Op)
	}
}

// processLogicalFilter processes logical filters (AND, OR, NOT)
func (idx *InvertedMetadataIndex) processLogicalFilter(ctx context.Context, filter *api.LogicalFilter) ([]string, error) {
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
func (idx *InvertedMetadataIndex) queryEqual(ctx context.Context, value interface{}) ([]string, error) {
	start := time.Now()
	defer func() {
		idx.stats.EqualityQueries++
		idx.stats.AvgEqualityTime = updateAverage(idx.stats.AvgEqualityTime,
			float64(time.Since(start).Nanoseconds())/1e6, idx.stats.EqualityQueries)
	}()

	strValue, err := idx.normalizeValue(value)
	if err != nil {
		return nil, err
	}

	// Check cache first
	cacheKey := fmt.Sprintf("eq:%s", strValue)
	if cached := idx.cache.Get(cacheKey); cached != nil {
		idx.stats.CacheHits++
		return cached.VectorIDs, nil
	}
	idx.stats.CacheMisses++

	// Get posting list
	postingList, exists := idx.index[strValue]
	if !exists {
		return []string{}, nil
	}

	vectorIDs := postingList.GetVectorIDs()

	// Cache result
	idx.cache.Put(cacheKey, &CacheEntry{
		VectorIDs:    vectorIDs,
		LastAccessed: time.Now(),
		AccessCount:  1,
	})

	return vectorIDs, nil
}

// queryNotEqual handles not-equal queries
func (idx *InvertedMetadataIndex) queryNotEqual(ctx context.Context, value interface{}) ([]string, error) {
	strValue, err := idx.normalizeValue(value)
	if err != nil {
		return nil, err
	}

	result := make([]string, 0)

	// Collect all vector IDs except those with the specified value
	for _, postingList := range idx.index {
		vectorIDs := postingList.GetVectorIDs()
		result = append(result, vectorIDs...)
	}

	// Remove vector IDs that have the specified value
	if postingList, exists := idx.index[strValue]; exists {
		excludeIDs := postingList.GetVectorIDs()
		result = removeStrings(result, excludeIDs)
	}

	return result, nil
}

// queryIn handles membership queries (IN operator)
func (idx *InvertedMetadataIndex) queryIn(ctx context.Context, values interface{}) ([]string, error) {
	start := time.Now()
	defer func() {
		idx.stats.MembershipQueries++
		idx.stats.AvgMembershipTime = updateAverage(idx.stats.AvgMembershipTime,
			float64(time.Since(start).Nanoseconds())/1e6, idx.stats.MembershipQueries)
	}()

	// Convert to slice of interfaces
	valueSlice, ok := values.([]interface{})
	if !ok {
		return nil, fmt.Errorf("IN operator requires array of values")
	}

	if len(valueSlice) == 0 {
		return []string{}, nil
	}

	// Union all posting lists for the specified values
	result := make(map[string]struct{})

	for _, value := range valueSlice {
		strValue, err := idx.normalizeValue(value)
		if err != nil {
			continue // Skip invalid values
		}

		if postingList, exists := idx.index[strValue]; exists {
			vectorIDs := postingList.GetVectorIDs()
			for _, id := range vectorIDs {
				result[id] = struct{}{}
			}
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
func (idx *InvertedMetadataIndex) queryNotIn(ctx context.Context, values interface{}) ([]string, error) {
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

// queryMatch handles prefix/pattern matching
func (idx *InvertedMetadataIndex) queryMatch(ctx context.Context, pattern interface{}) ([]string, error) {
	start := time.Now()
	defer func() {
		idx.stats.PrefixQueries++
		idx.stats.AvgPrefixTime = updateAverage(idx.stats.AvgPrefixTime,
			float64(time.Since(start).Nanoseconds())/1e6, idx.stats.PrefixQueries)
	}()

	strPattern, err := idx.normalizeValue(pattern)
	if err != nil {
		return nil, err
	}

	if strPattern == "" {
		return []string{}, nil
	}

	result := make(map[string]struct{})

	// Simple prefix matching for now
	for value, postingList := range idx.index {
		if strings.HasPrefix(value, strPattern) {
			vectorIDs := postingList.GetVectorIDs()
			for _, id := range vectorIDs {
				result[id] = struct{}{}
			}
		}
	}

	// Convert to slice
	vectorIDs := make([]string, 0, len(result))
	for id := range result {
		vectorIDs = append(vectorIDs, id)
	}

	return vectorIDs, nil
}

// Logical operators

// queryAnd performs intersection of multiple filter results
func (idx *InvertedMetadataIndex) queryAnd(ctx context.Context, filters []api.FilterExpr) ([]string, error) {
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
func (idx *InvertedMetadataIndex) queryOr(ctx context.Context, filters []api.FilterExpr) ([]string, error) {
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
func (idx *InvertedMetadataIndex) queryNot(ctx context.Context, filter api.FilterExpr) ([]string, error) {
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
func (idx *InvertedMetadataIndex) updateQueryStats(duration time.Duration) {
	idx.stats.QueryCount++
	durationMs := float64(duration.Nanoseconds()) / 1e6
	idx.stats.AverageTime = updateAverage(idx.stats.AverageTime, durationMs, idx.stats.QueryCount)
}

// updateComputedStats updates computed statistics
func (idx *InvertedMetadataIndex) updateComputedStats() {
	// Update unique values count
	idx.stats.UniqueValues = int64(len(idx.index))

	// Calculate average and max posting list sizes
	if len(idx.index) > 0 {
		totalSize := int64(0)
		maxSize := int64(0)

		for _, postingList := range idx.index {
			size := postingList.Size()
			totalSize += size
			if size > maxSize {
				maxSize = size
			}
		}

		idx.stats.AvgPostingSize = float64(totalSize) / float64(len(idx.index))
		idx.stats.MaxPostingSize = maxSize
	} else {
		idx.stats.AvgPostingSize = 0
		idx.stats.MaxPostingSize = 0
	}

	// Update memory usage (rough estimate)
	memoryUsage := int64(0)
	memoryUsage += int64(len(idx.index) * 200) // Index overhead
	memoryUsage += int64(len(idx.docs) * 100)  // Document mapping

	for value, postingList := range idx.index {
		memoryUsage += int64(len(value))              // Key storage
		memoryUsage += int64(postingList.Size() * 50) // Vector ID storage
	}

	idx.stats.MemoryUsage = memoryUsage

	// Update cache statistics
	idx.stats.CacheSize = int64(idx.cache.Size())
}

// PostingList implementation

// NewPostingList creates a new posting list
func NewPostingList() *PostingList {
	return &PostingList{
		vectorIDs:    make(map[string]struct{}),
		sortedIDs:    nil,
		sorted:       false,
		lastAccessed: time.Now(),
		accessCount:  0,
	}
}

// Add adds a vector ID to the posting list
func (pl *PostingList) Add(vectorID string) error {
	pl.mu.Lock()
	defer pl.mu.Unlock()

	pl.vectorIDs[vectorID] = struct{}{}
	pl.sorted = false // Invalidate sorted cache
	pl.lastAccessed = time.Now()
	pl.accessCount++

	return nil
}

// Remove removes a vector ID from the posting list
func (pl *PostingList) Remove(vectorID string) error {
	pl.mu.Lock()
	defer pl.mu.Unlock()

	delete(pl.vectorIDs, vectorID)
	pl.sorted = false // Invalidate sorted cache
	pl.lastAccessed = time.Now()

	return nil
}

// Contains checks if the posting list contains a vector ID
func (pl *PostingList) Contains(vectorID string) bool {
	pl.mu.RLock()
	defer pl.mu.RUnlock()

	_, exists := pl.vectorIDs[vectorID]
	return exists
}

// Size returns the number of vector IDs in the posting list
func (pl *PostingList) Size() int64 {
	pl.mu.RLock()
	defer pl.mu.RUnlock()

	return int64(len(pl.vectorIDs))
}

// GetVectorIDs returns all vector IDs in the posting list
func (pl *PostingList) GetVectorIDs() []string {
	pl.mu.Lock()
	defer pl.mu.Unlock()

	// Use cached sorted slice if available
	if pl.sorted && pl.sortedIDs != nil && len(pl.sortedIDs) == len(pl.vectorIDs) {
		result := make([]string, len(pl.sortedIDs))
		copy(result, pl.sortedIDs)
		return result
	}

	// Build and cache sorted slice
	pl.sortedIDs = make([]string, 0, len(pl.vectorIDs))
	for id := range pl.vectorIDs {
		pl.sortedIDs = append(pl.sortedIDs, id)
	}
	sort.Strings(pl.sortedIDs)
	pl.sorted = true

	// Return copy
	result := make([]string, len(pl.sortedIDs))
	copy(result, pl.sortedIDs)
	return result
}

// validate validates the posting list consistency
func (pl *PostingList) validate() error {
	pl.mu.RLock()
	defer pl.mu.RUnlock()

	// Check for empty vector IDs
	for id := range pl.vectorIDs {
		if id == "" {
			return fmt.Errorf("empty vector ID in posting list")
		}
	}

	return nil
}

// QueryCache implementation

// NewQueryCache creates a new query cache
func NewQueryCache(maxSize int) *QueryCache {
	return &QueryCache{
		cache:   make(map[string]*CacheEntry),
		usage:   make(map[string]time.Time),
		maxSize: maxSize,
	}
}

// Get retrieves a cached entry
func (qc *QueryCache) Get(key string) *CacheEntry {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	entry, exists := qc.cache[key]
	if !exists {
		return nil
	}

	// Update access time
	qc.usage[key] = time.Now()
	entry.LastAccessed = time.Now()
	entry.AccessCount++

	return entry
}

// Put stores an entry in the cache
func (qc *QueryCache) Put(key string, entry *CacheEntry) {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	// Evict if necessary
	if len(qc.cache) >= qc.maxSize {
		qc.evictLRU()
	}

	qc.cache[key] = entry
	qc.usage[key] = time.Now()
}

// Clear clears the cache
func (qc *QueryCache) Clear() {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	qc.cache = make(map[string]*CacheEntry)
	qc.usage = make(map[string]time.Time)
}

// Size returns the cache size
func (qc *QueryCache) Size() int {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	return len(qc.cache)
}

// evictLRU evicts the least recently used entry
func (qc *QueryCache) evictLRU() {
	var oldestKey string
	var oldestTime time.Time

	for key, accessTime := range qc.usage {
		if oldestKey == "" || accessTime.Before(oldestTime) {
			oldestKey = key
			oldestTime = accessTime
		}
	}

	if oldestKey != "" {
		delete(qc.cache, oldestKey)
		delete(qc.usage, oldestKey)
	}
}

// Utility functions

// updateAverage updates a running average
func updateAverage(current float64, newValue float64, count int64) float64 {
	if count <= 1 {
		return newValue
	}
	return (current*float64(count-1) + newValue) / float64(count)
}

// intersectStrings returns the intersection of two string slices
func intersectStrings(a, b []string) []string {
	setA := make(map[string]struct{})
	for _, s := range a {
		setA[s] = struct{}{}
	}

	result := make([]string, 0)
	for _, s := range b {
		if _, exists := setA[s]; exists {
			result = append(result, s)
		}
	}

	return result
}

// removeStrings removes all strings in 'remove' from 'from'
func removeStrings(from, remove []string) []string {
	removeSet := make(map[string]struct{})
	for _, s := range remove {
		removeSet[s] = struct{}{}
	}

	result := make([]string, 0)
	for _, s := range from {
		if _, exists := removeSet[s]; !exists {
			result = append(result, s)
		}
	}

	return result
}
