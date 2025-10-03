// Package api defines the core interfaces, types, and contracts for GoVecDB.
package api

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Common errors used throughout the API
var (
	ErrVectorNotFound     = errors.New("vector not found")
	ErrVectorExists       = errors.New("vector already exists")
	ErrCollectionNotFound = errors.New("collection not found")
	ErrDimensionMismatch  = errors.New("vector dimension mismatch")
	ErrEmptyVector        = errors.New("vector cannot be empty")
	ErrInvalidK           = errors.New("k must be positive")
	ErrInvalidMetric      = errors.New("invalid distance metric")
	ErrCollectionExists   = errors.New("collection already exists")
	ErrInvalidConfig      = errors.New("invalid configuration")
	ErrClosed             = errors.New("database is closed")
	ErrContextCanceled    = errors.New("operation canceled")
	ErrDuplicateID        = errors.New("duplicate vector ID")
	ErrCapacityExceeded   = errors.New("storage capacity exceeded")
	ErrOperationTimeout   = errors.New("operation timeout")
)

// DistanceMetric represents the type of distance calculation to use
type DistanceMetric int

const (
	Cosine DistanceMetric = iota
	Euclidean
	Manhattan
	DotProduct
)

// String returns the string representation of the distance metric
func (m DistanceMetric) String() string {
	switch m {
	case Cosine:
		return "cosine"
	case Euclidean:
		return "euclidean"
	case Manhattan:
		return "manhattan"
	case DotProduct:
		return "dot_product"
	default:
		return "unknown"
	}
}

// Vector represents a multi-dimensional vector with metadata
type Vector struct {
	ID       string                 `json:"id"`
	Data     []float32              `json:"data"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Validate checks if the vector is valid
func (v *Vector) Validate() error {
	if v.ID == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}
	if len(v.Data) == 0 {
		return ErrEmptyVector
	}
	return nil
}

// Clone creates a deep copy of the vector
func (v *Vector) Clone() *Vector {
	if v == nil {
		return nil
	}

	clone := &Vector{
		ID:       v.ID,
		Data:     make([]float32, len(v.Data)),
		Metadata: make(map[string]interface{}),
	}

	copy(clone.Data, v.Data)
	for k, val := range v.Metadata {
		clone.Metadata[k] = val
	}

	return clone
}

// Dimension returns the dimension of the vector
func (v *Vector) Dimension() int {
	return len(v.Data)
}

// SearchResult represents a search result with distance/similarity score
type SearchResult struct {
	Vector   *Vector `json:"vector"`
	Score    float32 `json:"score"`
	Distance float32 `json:"distance"` // Raw distance value
}

// SearchRequest represents a search query
type SearchRequest struct {
	Vector      []float32  `json:"vector"`
	K           int        `json:"k"`
	Filter      FilterExpr `json:"filter,omitempty"`
	IncludeData bool       `json:"include_data"`
	MaxDistance *float32   `json:"max_distance,omitempty"`
	MinScore    *float32   `json:"min_score,omitempty"`
}

// Validate checks if the search request is valid
func (r *SearchRequest) Validate() error {
	if len(r.Vector) == 0 {
		return ErrEmptyVector
	}
	if r.K <= 0 {
		return ErrInvalidK
	}
	if r.MaxDistance != nil && *r.MaxDistance < 0 {
		return fmt.Errorf("max distance cannot be negative")
	}
	if r.MinScore != nil && (*r.MinScore < 0 || *r.MinScore > 1) {
		return fmt.Errorf("min score must be between 0 and 1")
	}
	return nil
}

// CollectionConfig represents configuration for a collection
type CollectionConfig struct {
	Name           string         `json:"name"`
	Dimension      int            `json:"dimension"`
	Metric         DistanceMetric `json:"metric"`
	M              int            `json:"m"`               // Max connections per node
	EfConstruction int            `json:"ef_construction"` // Size of dynamic candidate list
	MaxLayer       int            `json:"max_layer"`
	Seed           int64          `json:"seed"`
	ThreadSafe     bool           `json:"thread_safe"`
	Description    string         `json:"description,omitempty"`
	Tags           []string       `json:"tags,omitempty"`
}

// Validate checks if the collection configuration is valid
func (c *CollectionConfig) Validate() error {
	if c.Name == "" {
		return fmt.Errorf("collection name cannot be empty")
	}
	if c.Dimension <= 0 {
		return fmt.Errorf("dimension must be positive")
	}
	if c.M <= 0 {
		return fmt.Errorf("m must be positive")
	}
	if c.EfConstruction <= 0 {
		return fmt.Errorf("ef_construction must be positive")
	}
	if c.MaxLayer <= 0 {
		return fmt.Errorf("max_layer must be positive")
	}
	if c.Metric < Cosine || c.Metric > DotProduct {
		return ErrInvalidMetric
	}
	return nil
}

// DefaultCollectionConfig returns a default configuration
func DefaultCollectionConfig(name string, dimension int) *CollectionConfig {
	return &CollectionConfig{
		Name:           name,
		Dimension:      dimension,
		Metric:         Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		Seed:           time.Now().UnixNano(),
		ThreadSafe:     true,
	}
}

// FilterOp represents filter operations
type FilterOp string

const (
	FilterEq    FilterOp = "eq"    // Equals
	FilterNe    FilterOp = "ne"    // Not equals
	FilterIn    FilterOp = "in"    // In array
	FilterNin   FilterOp = "nin"   // Not in array
	FilterGt    FilterOp = "gt"    // Greater than
	FilterGte   FilterOp = "gte"   // Greater than or equal
	FilterLt    FilterOp = "lt"    // Less than
	FilterLte   FilterOp = "lte"   // Less than or equal
	FilterAnd   FilterOp = "and"   // Logical AND
	FilterOr    FilterOp = "or"    // Logical OR
	FilterNot   FilterOp = "not"   // Logical NOT
	FilterMatch FilterOp = "match" // Text match
)

// FilterExpr represents a filter expression
type FilterExpr interface {
	// Evaluate checks if the given metadata matches this filter
	Evaluate(metadata map[string]interface{}) bool
	// Validate checks if the filter expression is valid
	Validate() error
}

// FieldFilter represents a field-based filter
type FieldFilter struct {
	Field string      `json:"field"`
	Op    FilterOp    `json:"op"`
	Value interface{} `json:"value"`
}

// Evaluate implements FilterExpr
func (f *FieldFilter) Evaluate(metadata map[string]interface{}) bool {
	val, exists := metadata[f.Field]
	if !exists {
		return false
	}

	switch f.Op {
	case FilterEq:
		return val == f.Value
	case FilterNe:
		return val != f.Value
	case FilterIn:
		if arr, ok := f.Value.([]interface{}); ok {
			for _, item := range arr {
				if val == item {
					return true
				}
			}
		}
		return false
	case FilterNin:
		if arr, ok := f.Value.([]interface{}); ok {
			for _, item := range arr {
				if val == item {
					return false
				}
			}
			return true
		}
		return false
	case FilterGt, FilterGte, FilterLt, FilterLte:
		return compareNumbers(val, f.Value, f.Op)
	case FilterMatch:
		if strVal, ok := val.(string); ok {
			if strMatch, ok := f.Value.(string); ok {
				return matchString(strVal, strMatch)
			}
		}
		return false
	default:
		return false
	}
}

// Validate implements FilterExpr
func (f *FieldFilter) Validate() error {
	if f.Field == "" {
		return fmt.Errorf("filter field cannot be empty")
	}
	if f.Value == nil {
		return fmt.Errorf("filter value cannot be nil")
	}
	return nil
}

// LogicalFilter represents a logical filter (AND, OR, NOT)
type LogicalFilter struct {
	Op      FilterOp     `json:"op"`
	Filters []FilterExpr `json:"filters"`
}

// Evaluate implements FilterExpr
func (f *LogicalFilter) Evaluate(metadata map[string]interface{}) bool {
	switch f.Op {
	case FilterAnd:
		for _, filter := range f.Filters {
			if !filter.Evaluate(metadata) {
				return false
			}
		}
		return true
	case FilterOr:
		for _, filter := range f.Filters {
			if filter.Evaluate(metadata) {
				return true
			}
		}
		return false
	case FilterNot:
		if len(f.Filters) == 1 {
			return !f.Filters[0].Evaluate(metadata)
		}
		return false
	default:
		return false
	}
}

// Validate implements FilterExpr
func (f *LogicalFilter) Validate() error {
	if len(f.Filters) == 0 {
		return fmt.Errorf("logical filter must have at least one sub-filter")
	}
	if f.Op == FilterNot && len(f.Filters) != 1 {
		return fmt.Errorf("NOT filter must have exactly one sub-filter")
	}
	for _, filter := range f.Filters {
		if err := filter.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// CollectionStats represents statistics about a collection
type CollectionStats struct {
	Name        string     `json:"name"`
	VectorCount int64      `json:"vector_count"`
	Dimension   int        `json:"dimension"`
	IndexStats  IndexStats `json:"index_stats"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
	SizeBytes   int64      `json:"size_bytes"`
	MemoryUsage int64      `json:"memory_usage"`
}

// IndexStats represents statistics about the index
type IndexStats struct {
	NodeCount   int64   `json:"node_count"`
	EdgeCount   int64   `json:"edge_count"`
	MaxLayer    int     `json:"max_layer"`
	AvgDegree   float64 `json:"avg_degree"`
	SearchCount int64   `json:"search_count"`
	InsertCount int64   `json:"insert_count"`
}

// VectorStore defines the interface for vector storage
type VectorStore interface {
	// Store operations
	Put(ctx context.Context, vector *Vector) error
	Update(ctx context.Context, vector *Vector) error
	Get(ctx context.Context, id string) (*Vector, error)
	Delete(ctx context.Context, id string) error
	List(ctx context.Context, limit int, offset int) ([]*Vector, error)
	Count(ctx context.Context) (int64, error)

	// Batch operations
	PutBatch(ctx context.Context, vectors []*Vector) error
	GetBatch(ctx context.Context, ids []string) ([]*Vector, error)
	DeleteBatch(ctx context.Context, ids []string) error

	// Filter operations
	Filter(ctx context.Context, filter FilterExpr, limit int) ([]*Vector, error)

	// Lifecycle
	Close() error
	Clear(ctx context.Context) error
	Stats(ctx context.Context) (*CollectionStats, error)
}

// VectorIndex defines the interface for vector indexing and search
type VectorIndex interface {
	// Index operations
	Add(ctx context.Context, vector *Vector) error
	Remove(ctx context.Context, id string) error
	Search(ctx context.Context, req *SearchRequest) ([]*SearchResult, error)

	// Batch operations
	AddBatch(ctx context.Context, vectors []*Vector) error
	RemoveBatch(ctx context.Context, ids []string) error

	// Management
	Size() int64
	Stats() *IndexStats
	Optimize(ctx context.Context) error
	Close() error
}

// Collection defines the main interface for a vector collection
type Collection interface {
	// Basic operations
	Add(ctx context.Context, vector *Vector) error
	Get(ctx context.Context, id string) (*Vector, error)
	Delete(ctx context.Context, id string) error
	Search(ctx context.Context, req *SearchRequest) ([]*SearchResult, error)

	// Batch operations
	AddBatch(ctx context.Context, vectors []*Vector) error
	GetBatch(ctx context.Context, ids []string) ([]*Vector, error)
	DeleteBatch(ctx context.Context, ids []string) error

	// Query operations
	Filter(ctx context.Context, filter FilterExpr, limit int) ([]*Vector, error)
	Count(ctx context.Context) (int64, error)
	List(ctx context.Context, limit int, offset int) ([]*Vector, error)

	// Management
	Name() string
	Config() *CollectionConfig
	Stats(ctx context.Context) (*CollectionStats, error)
	Optimize(ctx context.Context) error
	Clear(ctx context.Context) error
	Close() error
}

// Database defines the main interface for the vector database
type Database interface {
	// Collection management
	CreateCollection(ctx context.Context, config *CollectionConfig) (Collection, error)
	GetCollection(ctx context.Context, name string) (Collection, error)
	DropCollection(ctx context.Context, name string) error
	ListCollections(ctx context.Context) ([]string, error)

	// Lifecycle
	Close() error
	Stats(ctx context.Context) (map[string]*CollectionStats, error)
}

// Helper functions

// compareNumbers compares two numeric values based on the operation
func compareNumbers(a, b interface{}, op FilterOp) bool {
	aFloat, aOk := toFloat64(a)
	bFloat, bOk := toFloat64(b)

	if !aOk || !bOk {
		return false
	}

	switch op {
	case FilterGt:
		return aFloat > bFloat
	case FilterGte:
		return aFloat >= bFloat
	case FilterLt:
		return aFloat < bFloat
	case FilterLte:
		return aFloat <= bFloat
	default:
		return false
	}
}

// toFloat64 converts various numeric types to float64
func toFloat64(val interface{}) (float64, bool) {
	switch v := val.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case uint:
		return float64(v), true
	case uint32:
		return float64(v), true
	case uint64:
		return float64(v), true
	default:
		return 0, false
	}
}

// matchString performs simple string matching (can be enhanced with regex later)
func matchString(text, pattern string) bool {
	// Simple contains match for now
	// Can be enhanced with regex or fuzzy matching
	return len(pattern) > 0 && len(text) >= len(pattern) &&
		text[:len(pattern)] == pattern
}

// CreateFieldFilter creates a field-based filter
func CreateFieldFilter(field string, op FilterOp, value interface{}) FilterExpr {
	return &FieldFilter{
		Field: field,
		Op:    op,
		Value: value,
	}
}

// CreateLogicalFilter creates a logical filter
func CreateLogicalFilter(op FilterOp, filters ...FilterExpr) FilterExpr {
	return &LogicalFilter{
		Op:      op,
		Filters: filters,
	}
}

// And creates an AND filter
func And(filters ...FilterExpr) FilterExpr {
	return CreateLogicalFilter(FilterAnd, filters...)
}

// Or creates an OR filter
func Or(filters ...FilterExpr) FilterExpr {
	return CreateLogicalFilter(FilterOr, filters...)
}

// Not creates a NOT filter
func Not(filter FilterExpr) FilterExpr {
	return CreateLogicalFilter(FilterNot, filter)
}

// Eq creates an equality filter
func Eq(field string, value interface{}) FilterExpr {
	return CreateFieldFilter(field, FilterEq, value)
}

// Ne creates a not-equal filter
func Ne(field string, value interface{}) FilterExpr {
	return CreateFieldFilter(field, FilterNe, value)
}

// In creates an "in array" filter
func In(field string, values []interface{}) FilterExpr {
	return CreateFieldFilter(field, FilterIn, values)
}

// Gt creates a greater-than filter
func Gt(field string, value interface{}) FilterExpr {
	return CreateFieldFilter(field, FilterGt, value)
}

// Gte creates a greater-than-or-equal filter
func Gte(field string, value interface{}) FilterExpr {
	return CreateFieldFilter(field, FilterGte, value)
}

// Lt creates a less-than filter
func Lt(field string, value interface{}) FilterExpr {
	return CreateFieldFilter(field, FilterLt, value)
}

// Lte creates a less-than-or-equal filter
func Lte(field string, value interface{}) FilterExpr {
	return CreateFieldFilter(field, FilterLte, value)
}

// FuncFilter wraps a function to implement FilterExpr interface
type FuncFilter struct {
	Fn func(metadata map[string]interface{}) bool
}

// Evaluate implements FilterExpr
func (f *FuncFilter) Evaluate(metadata map[string]interface{}) bool {
	if f.Fn == nil {
		return true
	}
	return f.Fn(metadata)
}

// Validate implements FilterExpr
func (f *FuncFilter) Validate() error {
	if f.Fn == nil {
		return fmt.Errorf("filter function cannot be nil")
	}
	return nil
}

// NewFuncFilter creates a function-based filter
func NewFuncFilter(fn func(metadata map[string]interface{}) bool) FilterExpr {
	return &FuncFilter{Fn: fn}
}
