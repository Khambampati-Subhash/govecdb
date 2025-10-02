// Package filter provides a hybrid filter engine implementation.
package filter

import (
	"context"
	"fmt"
	"sync"

	"github.com/khambampati-subhash/govecdb/api"
)

// HybridFilterEngine implements FilterEngine using multiple metadata indexes
type HybridFilterEngine struct {
	indexes map[string]MetadataIndex
	mu      sync.RWMutex
	closed  bool
}

// NewHybridFilterEngine creates a new hybrid filter engine
func NewHybridFilterEngine(invertedIndex, numericIndex MetadataIndex) *HybridFilterEngine {
	engine := &HybridFilterEngine{
		indexes: make(map[string]MetadataIndex),
		closed:  false,
	}

	// Add the provided indexes
	if invertedIndex != nil {
		engine.indexes[invertedIndex.Field()] = invertedIndex
	}
	if numericIndex != nil {
		engine.indexes[numericIndex.Field()] = numericIndex
	}

	return engine
}

// CreateIndex creates a new index for the specified field
func (h *HybridFilterEngine) CreateIndex(config *IndexConfig) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.closed {
		return ErrIndexClosed
	}

	if _, exists := h.indexes[config.Field]; exists {
		return ErrIndexExists
	}

	var index MetadataIndex
	var err error

	switch config.Type {
	case InvertedIndex:
		index, err = NewInvertedMetadataIndex(config)
	case NumericIndex:
		index, err = NewNumericMetadataIndex(config)
	default:
		return ErrInvalidIndexType
	}

	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	h.indexes[config.Field] = index
	return nil
}

// GetIndex returns the index for the specified field
func (h *HybridFilterEngine) GetIndex(field string) (MetadataIndex, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.closed {
		return nil, ErrIndexClosed
	}

	index, exists := h.indexes[field]
	if !exists {
		return nil, ErrIndexNotFound
	}

	return index, nil
}

// DropIndex removes the index for the specified field
func (h *HybridFilterEngine) DropIndex(field string) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.closed {
		return ErrIndexClosed
	}

	index, exists := h.indexes[field]
	if !exists {
		return ErrIndexNotFound
	}

	if err := index.Close(); err != nil {
		return fmt.Errorf("failed to close index: %w", err)
	}

	delete(h.indexes, field)
	return nil
}

// ListIndexes returns a list of all index field names
func (h *HybridFilterEngine) ListIndexes() []string {
	h.mu.RLock()
	defer h.mu.RUnlock()

	fields := make([]string, 0, len(h.indexes))
	for field := range h.indexes {
		fields = append(fields, field)
	}

	return fields
}

// HasIndex checks if an index exists for the specified field
func (h *HybridFilterEngine) HasIndex(field string) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()

	_, exists := h.indexes[field]
	return exists
}

// AddDocument adds a document with its metadata to all relevant indexes
func (h *HybridFilterEngine) AddDocument(vectorID string, metadata map[string]interface{}) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.closed {
		return ErrIndexClosed
	}

	for field, value := range metadata {
		if index, exists := h.indexes[field]; exists {
			if err := index.Add(vectorID, value); err != nil {
				return fmt.Errorf("failed to add to index %s: %w", field, err)
			}
		}
	}

	return nil
}

// RemoveDocument removes a document from all relevant indexes
func (h *HybridFilterEngine) RemoveDocument(vectorID string, metadata map[string]interface{}) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.closed {
		return ErrIndexClosed
	}

	for field, value := range metadata {
		if index, exists := h.indexes[field]; exists {
			if err := index.Remove(vectorID, value); err != nil {
				// Log warning but continue
				fmt.Printf("Warning: failed to remove from index %s: %v\n", field, err)
			}
		}
	}

	return nil
}

// UpdateDocument updates a document in all relevant indexes
func (h *HybridFilterEngine) UpdateDocument(vectorID string, oldMetadata, newMetadata map[string]interface{}) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.closed {
		return ErrIndexClosed
	}

	// Get all fields from both old and new metadata
	allFields := make(map[string]bool)
	for field := range oldMetadata {
		allFields[field] = true
	}
	for field := range newMetadata {
		allFields[field] = true
	}

	// Update each field
	for field := range allFields {
		if index, exists := h.indexes[field]; exists {
			oldValue := oldMetadata[field]
			newValue := newMetadata[field]

			if err := index.Update(vectorID, oldValue, newValue); err != nil {
				return fmt.Errorf("failed to update index %s: %w", field, err)
			}
		}
	}

	return nil
}

// Filter executes a filter expression and returns matching vector IDs
func (h *HybridFilterEngine) Filter(ctx context.Context, expr api.FilterExpr) ([]string, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.closed {
		return nil, ErrIndexClosed
	}

	return h.executeFilter(ctx, expr)
}

// executeFilter recursively executes filter expressions
func (h *HybridFilterEngine) executeFilter(ctx context.Context, expr api.FilterExpr) ([]string, error) {
	switch filter := expr.(type) {
	case *api.FieldFilter:
		return h.executeFieldFilter(ctx, filter)
	case *api.LogicalFilter:
		return h.executeLogicalFilter(ctx, filter)
	default:
		return nil, fmt.Errorf("unsupported filter type: %T", expr)
	}
}

// executeFieldFilter executes a field-based filter
func (h *HybridFilterEngine) executeFieldFilter(ctx context.Context, filter *api.FieldFilter) ([]string, error) {
	index, exists := h.indexes[filter.Field]
	if !exists {
		return []string{}, nil // No index for this field
	}

	return index.Query(ctx, filter)
}

// executeLogicalFilter executes a logical filter (AND, OR, NOT)
func (h *HybridFilterEngine) executeLogicalFilter(ctx context.Context, filter *api.LogicalFilter) ([]string, error) {
	switch filter.Op {
	case api.FilterAnd:
		return h.executeAndLogical(ctx, filter.Filters)
	case api.FilterOr:
		return h.executeOrLogical(ctx, filter.Filters)
	case api.FilterNot:
		return h.executeNotLogical(ctx, filter.Filters)
	default:
		return nil, fmt.Errorf("unsupported logical operator: %s", filter.Op)
	}
}

// executeAndLogical executes an AND logical filter
func (h *HybridFilterEngine) executeAndLogical(ctx context.Context, filters []api.FilterExpr) ([]string, error) {
	if len(filters) == 0 {
		return []string{}, nil
	}

	// Execute first filter
	result, err := h.executeFilter(ctx, filters[0])
	if err != nil {
		return nil, err
	}

	// Intersect with remaining filters
	for i := 1; i < len(filters); i++ {
		otherResult, err := h.executeFilter(ctx, filters[i])
		if err != nil {
			return nil, err
		}

		result = intersect(result, otherResult)
		if len(result) == 0 {
			break // Early termination
		}
	}

	return result, nil
}

// executeOrLogical executes an OR logical filter
func (h *HybridFilterEngine) executeOrLogical(ctx context.Context, filters []api.FilterExpr) ([]string, error) {
	if len(filters) == 0 {
		return []string{}, nil
	}

	allResults := make([]string, 0)

	// Execute all filters and union results
	for _, subFilter := range filters {
		result, err := h.executeFilter(ctx, subFilter)
		if err != nil {
			return nil, err
		}

		allResults = union(allResults, result)
	}

	return allResults, nil
}

// executeNotLogical executes a NOT logical filter
func (h *HybridFilterEngine) executeNotLogical(ctx context.Context, filters []api.FilterExpr) ([]string, error) {
	// NOT filters are complex as they require knowing all possible vector IDs
	// For now, return an error as this would require a full scan
	return nil, fmt.Errorf("NOT filters not yet supported in hybrid engine")
}

// EstimateSelectivity estimates the selectivity of a filter expression
func (h *HybridFilterEngine) EstimateSelectivity(expr api.FilterExpr) float64 {
	// Simple estimation - can be improved with statistics
	switch filter := expr.(type) {
	case *api.FieldFilter:
		if index, exists := h.indexes[filter.Field]; exists {
			stats := index.Stats()
			if stats.EntryCount > 0 {
				switch filter.Op {
				case api.FilterEq:
					return 1.0 / float64(stats.EntryCount)
				case api.FilterIn:
					return 0.2 // Membership queries are moderately selective
				case api.FilterGt, api.FilterGte, api.FilterLt, api.FilterLte:
					return 0.3 // Range queries typically return more results
				default:
					return 0.1
				}
			}
		}
		return 0.1 // Default estimate
	case *api.LogicalFilter:
		switch filter.Op {
		case api.FilterAnd:
			// AND reduces selectivity
			selectivity := 1.0
			for _, subFilter := range filter.Filters {
				selectivity *= h.EstimateSelectivity(subFilter)
			}
			return selectivity
		case api.FilterOr:
			// OR increases selectivity
			selectivity := 0.0
			for _, subFilter := range filter.Filters {
				selectivity += h.EstimateSelectivity(subFilter)
			}
			if selectivity > 1.0 {
				selectivity = 1.0
			}
			return selectivity
		case api.FilterNot:
			// NOT inverts selectivity
			if len(filter.Filters) == 1 {
				return 1.0 - h.EstimateSelectivity(filter.Filters[0])
			}
		}
		return 0.5
	default:
		return 0.5 // Default estimate
	}
}

// AddDocumentBatch adds multiple documents in a batch
func (h *HybridFilterEngine) AddDocumentBatch(documents map[string]map[string]interface{}) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.closed {
		return ErrIndexClosed
	}

	for vectorID, metadata := range documents {
		for field, value := range metadata {
			if index, exists := h.indexes[field]; exists {
				if err := index.Add(vectorID, value); err != nil {
					return fmt.Errorf("failed to add document %s to index %s: %w", vectorID, field, err)
				}
			}
		}
	}

	return nil
}

// RemoveDocumentBatch removes multiple documents in a batch
func (h *HybridFilterEngine) RemoveDocumentBatch(vectorIDs []string, metadataMap map[string]map[string]interface{}) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.closed {
		return ErrIndexClosed
	}

	for _, vectorID := range vectorIDs {
		if metadata, exists := metadataMap[vectorID]; exists {
			for field, value := range metadata {
				if index, exists := h.indexes[field]; exists {
					if err := index.Remove(vectorID, value); err != nil {
						fmt.Printf("Warning: failed to remove document %s from index %s: %v\n", vectorID, field, err)
					}
				}
			}
		}
	}

	return nil
}

// Stats returns statistics for all indexes
func (h *HybridFilterEngine) Stats() map[string]*IndexStats {
	h.mu.RLock()
	defer h.mu.RUnlock()

	stats := make(map[string]*IndexStats)
	for field, index := range h.indexes {
		stats[field] = index.Stats()
	}

	return stats
}

// Optimize optimizes all indexes
func (h *HybridFilterEngine) Optimize(ctx context.Context) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.closed {
		return ErrIndexClosed
	}

	for field, index := range h.indexes {
		if err := index.Validate(); err != nil {
			return fmt.Errorf("optimization failed for index %s: %w", field, err)
		}
	}

	return nil
}

// Validate validates all indexes
func (h *HybridFilterEngine) Validate() error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.closed {
		return ErrIndexClosed
	}

	for field, index := range h.indexes {
		if err := index.Validate(); err != nil {
			return fmt.Errorf("validation failed for index %s: %w", field, err)
		}
	}

	return nil
}

// Close closes all indexes
func (h *HybridFilterEngine) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.closed {
		return ErrIndexClosed
	}

	h.closed = true

	for field, index := range h.indexes {
		if err := index.Close(); err != nil {
			fmt.Printf("Warning: failed to close index %s: %v\n", field, err)
		}
	}

	h.indexes = make(map[string]MetadataIndex)
	return nil
}

// Helper functions for set operations

// intersect returns the intersection of two string slices
func intersect(a, b []string) []string {
	if len(a) == 0 || len(b) == 0 {
		return []string{}
	}

	// Create a map for faster lookup
	bMap := make(map[string]bool)
	for _, item := range b {
		bMap[item] = true
	}

	result := make([]string, 0)
	for _, item := range a {
		if bMap[item] {
			result = append(result, item)
		}
	}

	return result
}

// union returns the union of two string slices
func union(a, b []string) []string {
	if len(a) == 0 {
		return b
	}
	if len(b) == 0 {
		return a
	}

	// Use a map to avoid duplicates
	itemMap := make(map[string]bool)

	// Add items from a
	for _, item := range a {
		itemMap[item] = true
	}

	// Add items from b
	for _, item := range b {
		itemMap[item] = true
	}

	// Convert back to slice
	result := make([]string, 0, len(itemMap))
	for item := range itemMap {
		result = append(result, item)
	}

	return result
}
