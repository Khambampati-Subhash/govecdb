// Package filter provides efficient metadata indexing and filtering capabilities for GoVecDB.
// It implements inverted indexes for categorical data and numeric indexes for range queries,
// enabling fast hybrid search combining vector similarity with metadata constraints.
package filter

import (
	"context"
	"errors"

	"github.com/khambampati-subhash/govecdb/api"
)

// Common errors for filtering operations
var (
	ErrIndexNotFound       = errors.New("index not found")
	ErrInvalidIndexType    = errors.New("invalid index type")
	ErrIndexExists         = errors.New("index already exists")
	ErrUnsupportedOperator = errors.New("unsupported filter operator")
	ErrInvalidRange        = errors.New("invalid range parameters")
	ErrIndexCorrupted      = errors.New("index corrupted")
	ErrIndexClosed         = errors.New("index is closed")
)

// IndexType represents the type of metadata index
type IndexType string

const (
	InvertedIndex IndexType = "inverted" // For categorical/string fields
	NumericIndex  IndexType = "numeric"  // For numeric range queries
	TextIndex     IndexType = "text"     // For full-text search (future)
	GeoIndex      IndexType = "geo"      // For geospatial queries (future)
)

// IndexConfig holds configuration for creating an index
type IndexConfig struct {
	Field       string                 `json:"field"`        // Metadata field to index
	Type        IndexType              `json:"type"`         // Type of index
	Options     map[string]interface{} `json:"options"`      // Index-specific options
	Description string                 `json:"description"`  // Human-readable description
	ThreadSafe  bool                   `json:"thread_safe"`  // Enable concurrent access
	AutoRebuild bool                   `json:"auto_rebuild"` // Auto-rebuild on corruption
}

// Validate validates the index configuration
func (c *IndexConfig) Validate() error {
	if c.Field == "" {
		return errors.New("index field cannot be empty")
	}
	if c.Type == "" {
		return errors.New("index type cannot be empty")
	}
	return nil
}

// IndexStats provides statistics about an index
type IndexStats struct {
	Field         string    `json:"field"`
	Type          IndexType `json:"type"`
	EntryCount    int64     `json:"entry_count"`    // Number of unique values
	DocumentCount int64     `json:"document_count"` // Number of documents indexed
	MemoryUsage   int64     `json:"memory_usage"`   // Memory usage in bytes
	UpdateCount   int64     `json:"update_count"`   // Number of updates
	QueryCount    int64     `json:"query_count"`    // Number of queries processed
	AverageTime   float64   `json:"average_time"`   // Average query time in ms
	LastUpdate    int64     `json:"last_update"`    // Unix timestamp of last update
}

// MetadataIndex defines the interface for metadata indexes
type MetadataIndex interface {
	// Configuration and metadata
	Field() string
	Type() IndexType
	Config() *IndexConfig

	// Index management
	Add(vectorID string, value interface{}) error
	Remove(vectorID string, value interface{}) error
	Update(vectorID string, oldValue, newValue interface{}) error
	Clear() error

	// Querying
	Query(ctx context.Context, expr api.FilterExpr) ([]string, error)
	Contains(value interface{}) bool
	Size() int64

	// Statistics and introspection
	Stats() *IndexStats
	Validate() error

	// Lifecycle
	Close() error
	IsClosed() bool
}

// FilterEngine coordinates multiple metadata indexes to provide fast filtering
type FilterEngine interface {
	// Index management
	CreateIndex(config *IndexConfig) error
	GetIndex(field string) (MetadataIndex, error)
	DropIndex(field string) error
	ListIndexes() []string
	HasIndex(field string) bool

	// Document management
	AddDocument(vectorID string, metadata map[string]interface{}) error
	RemoveDocument(vectorID string, metadata map[string]interface{}) error
	UpdateDocument(vectorID string, oldMetadata, newMetadata map[string]interface{}) error

	// Querying
	Filter(ctx context.Context, expr api.FilterExpr) ([]string, error)
	EstimateSelectivity(expr api.FilterExpr) float64

	// Batch operations
	AddDocumentBatch(documents map[string]map[string]interface{}) error
	RemoveDocumentBatch(vectorIDs []string, metadataMap map[string]map[string]interface{}) error

	// Statistics and management
	Stats() map[string]*IndexStats
	Optimize(ctx context.Context) error
	Validate() error

	// Lifecycle
	Close() error
}

// QueryPlan represents an execution plan for a filter query
type QueryPlan struct {
	Steps         []QueryStep `json:"steps"`
	EstimatedCost float64     `json:"estimated_cost"`
	UseIndexes    []string    `json:"use_indexes"`
}

// QueryStep represents a single step in query execution
type QueryStep struct {
	Type        string      `json:"type"`        // "index_scan", "merge", "intersect", etc.
	Index       string      `json:"index"`       // Index name if applicable
	Operation   string      `json:"operation"`   // Specific operation
	Cardinality int64       `json:"cardinality"` // Expected result size
	Cost        float64     `json:"cost"`        // Estimated cost
	Details     interface{} `json:"details"`     // Step-specific details
}

// QueryOptimizer optimizes filter expressions for efficient execution
type QueryOptimizer interface {
	// Query planning
	CreatePlan(expr api.FilterExpr, availableIndexes []string) (*QueryPlan, error)
	OptimizeExpression(expr api.FilterExpr) (api.FilterExpr, error)
	EstimateCost(expr api.FilterExpr, plan *QueryPlan) float64

	// Statistics for optimization
	UpdateStatistics(field string, stats *IndexStats)
	GetStatistics(field string) *IndexStats
}

// ConcurrentFilterEngine provides thread-safe filtering operations
type ConcurrentFilterEngine struct {
}

// FilterEngineConfig holds configuration for the filter engine
type FilterEngineConfig struct {
	// Performance tuning
	MaxConcurrentQueries int `json:"max_concurrent_queries"`
	QueryTimeout         int `json:"query_timeout_ms"`
	CacheSize            int `json:"cache_size"`

	// Index management
	AutoCreateIndexes bool                   `json:"auto_create_indexes"`
	DefaultIndexType  IndexType              `json:"default_index_type"`
	IndexOptions      map[string]interface{} `json:"index_options"`

	// Optimization
	EnableOptimization bool `json:"enable_optimization"`
	StatisticsInterval int  `json:"statistics_interval"`

	// Memory management
	MaxMemoryUsage int64   `json:"max_memory_usage"`
	GCThreshold    float64 `json:"gc_threshold"`
}

// DefaultFilterEngineConfig returns default configuration for filter engine
func DefaultFilterEngineConfig() *FilterEngineConfig {
	return &FilterEngineConfig{
		MaxConcurrentQueries: 100,
		QueryTimeout:         5000, // 5 seconds
		CacheSize:            10000,
		AutoCreateIndexes:    true,
		DefaultIndexType:     InvertedIndex,
		IndexOptions:         make(map[string]interface{}),
		EnableOptimization:   true,
		StatisticsInterval:   60,  // 60 seconds
		MaxMemoryUsage:       0,   // Unlimited
		GCThreshold:          0.8, // 80%
	}
}

// FilterEngineStats provides overall statistics about the filter engine
type FilterEngineStats struct {
	// Index counts
	TotalIndexes    int64 `json:"total_indexes"`
	InvertedIndexes int64 `json:"inverted_indexes"`
	NumericIndexes  int64 `json:"numeric_indexes"`

	// Document counts
	TotalDocuments   int64 `json:"total_documents"`
	IndexedDocuments int64 `json:"indexed_documents"`

	// Query statistics
	TotalQueries     int64   `json:"total_queries"`
	CachedQueries    int64   `json:"cached_queries"`
	AverageQueryTime float64 `json:"average_query_time"`
	SlowQueries      int64   `json:"slow_queries"`

	// Memory usage
	TotalMemory int64 `json:"total_memory"`
	IndexMemory int64 `json:"index_memory"`
	CacheMemory int64 `json:"cache_memory"`

	// Performance metrics
	IndexHitRate      float64 `json:"index_hit_rate"`
	CacheHitRate      float64 `json:"cache_hit_rate"`
	OptimizationCount int64   `json:"optimization_count"`

	// Error tracking
	ErrorCount    int64  `json:"error_count"`
	LastError     string `json:"last_error"`
	LastErrorTime int64  `json:"last_error_time"`
}

// IndexOption represents a configuration option for indexes
type IndexOption func(*IndexConfig)

// WithThreadSafe enables thread-safe operations for the index
func WithThreadSafe(enabled bool) IndexOption {
	return func(config *IndexConfig) {
		config.ThreadSafe = enabled
	}
}

// WithAutoRebuild enables automatic index rebuilding on corruption
func WithAutoRebuild(enabled bool) IndexOption {
	return func(config *IndexConfig) {
		config.AutoRebuild = enabled
	}
}

// WithDescription sets a description for the index
func WithDescription(description string) IndexOption {
	return func(config *IndexConfig) {
		config.Description = description
	}
}

// WithOptions sets custom options for the index
func WithOptions(options map[string]interface{}) IndexOption {
	return func(config *IndexConfig) {
		if config.Options == nil {
			config.Options = make(map[string]interface{})
		}
		for k, v := range options {
			config.Options[k] = v
		}
	}
}

// NewIndexConfig creates a new index configuration with options
func NewIndexConfig(field string, indexType IndexType, options ...IndexOption) *IndexConfig {
	config := &IndexConfig{
		Field:       field,
		Type:        indexType,
		Options:     make(map[string]interface{}),
		ThreadSafe:  true,
		AutoRebuild: false,
	}

	for _, option := range options {
		option(config)
	}

	return config
}

// FilterResult represents the result of a filter operation
type FilterResult struct {
	VectorIDs   []string   `json:"vector_ids"`
	Count       int64      `json:"count"`
	Selectivity float64    `json:"selectivity"`  // Fraction of total documents matched
	QueryTime   float64    `json:"query_time"`   // Query execution time in ms
	IndexesUsed []string   `json:"indexes_used"` // Indexes used in query
	PlanUsed    *QueryPlan `json:"plan_used"`    // Query plan used
}

// QueryContext provides context for query execution
type QueryContext struct {
	// Request context
	Context context.Context

	// Query parameters
	MaxResults int
	Timeout    int64
	UseCache   bool

	// Performance hints
	PreferredIndexes []string
	AvoidIndexes     []string
	ForceFullScan    bool

	// Debugging
	IncludeStats   bool
	IncludePlan    bool
	VerboseLogging bool
}

// NewQueryContext creates a new query context with default values
func NewQueryContext(ctx context.Context) *QueryContext {
	return &QueryContext{
		Context:          ctx,
		MaxResults:       0,    // No limit
		Timeout:          5000, // 5 seconds
		UseCache:         true,
		PreferredIndexes: nil,
		AvoidIndexes:     nil,
		ForceFullScan:    false,
		IncludeStats:     false,
		IncludePlan:      false,
		VerboseLogging:   false,
	}
}

// FilterCallback defines a callback function for streaming filter results
type FilterCallback func(vectorID string, metadata map[string]interface{}) bool

// StreamingFilter provides streaming interface for large result sets
type StreamingFilter interface {
	// Streaming query execution
	FilterStream(ctx context.Context, expr api.FilterExpr, callback FilterCallback) error
	FilterStreamWithPlan(ctx context.Context, plan *QueryPlan, callback FilterCallback) error

	// Batch streaming for better performance
	FilterBatchStream(ctx context.Context, expr api.FilterExpr, batchSize int, callback func([]string) bool) error
}

// IndexBuilder helps build indexes efficiently
type IndexBuilder interface {
	// Build index from existing data
	BuildFromStore(ctx context.Context, store api.VectorStore, config *IndexConfig) (MetadataIndex, error)
	BuildFromVectors(vectors []*api.Vector, config *IndexConfig) (MetadataIndex, error)

	// Incremental building
	CreateEmpty(config *IndexConfig) (MetadataIndex, error)
	Rebuild(ctx context.Context, index MetadataIndex, store api.VectorStore) error

	// Index validation and repair
	ValidateIndex(index MetadataIndex, store api.VectorStore) error
	RepairIndex(ctx context.Context, index MetadataIndex, store api.VectorStore) error
}
