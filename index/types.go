package index

import (
	"errors"
	"sync"
)

// Common errors
var (
	ErrDimensionMismatch = errors.New("vector dimension mismatch")
	ErrEmptyVector       = errors.New("vector cannot be empty")
	ErrInvalidK          = errors.New("k must be positive")
	ErrNotFound          = errors.New("vector not found")
	ErrInvalidMetric     = errors.New("invalid distance metric")
)

// DistanceMetric represents the type of distance calculation to use
type DistanceMetric int

const (
	// Cosine distance (1 - cosine similarity)
	Cosine DistanceMetric = iota
	// Euclidean distance (L2 norm)
	Euclidean
	// Manhattan distance (L1 norm)
	Manhattan
	// Dot product (negative for similarity)
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

// Dimension returns the dimension of the vector
func (v *Vector) Dimension() int {
	return len(v.Data)
}

// SearchResult represents a search result with distance/similarity score
type SearchResult struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Score    float32                `json:"score"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Config represents the configuration for HNSW index
type Config struct {
	// Dimension of vectors
	Dimension int `json:"dimension"`

	// Distance metric to use
	Metric DistanceMetric `json:"metric"`

	// Maximum number of connections for every new element during construction
	M int `json:"m"`

	// Size of the dynamic candidate list
	EfConstruction int `json:"ef_construction"`

	// Maximum layer for HNSW
	MaxLayer int `json:"max_layer"`

	// Random seed for reproducibility
	Seed int64 `json:"seed"`

	// Enable thread safety
	ThreadSafe bool `json:"thread_safe"`
}

// DefaultConfig returns a default configuration for HNSW
func DefaultConfig(dimension int) *Config {
	return &Config{
		Dimension:      dimension,
		Metric:         Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		Seed:           42,
		ThreadSafe:     true,
	}
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.Dimension <= 0 {
		return errors.New("dimension must be positive")
	}
	if c.M <= 0 {
		return errors.New("m must be positive")
	}
	if c.EfConstruction <= 0 {
		return errors.New("ef_construction must be positive")
	}
	if c.MaxLayer <= 0 {
		return errors.New("max_layer must be positive")
	}
	if c.Metric < Cosine || c.Metric > DotProduct {
		return ErrInvalidMetric
	}
	return nil
}

// DistanceFunc represents a function that calculates distance between two vectors
type DistanceFunc func(a, b []float32) (float32, error)

// Index represents the interface for vector index implementations
type Index interface {
	// Add inserts a vector into the index
	Add(vector *Vector) error

	// AddBatch inserts multiple vectors into the index
	AddBatch(vectors []*Vector) error

	// Search finds the k most similar vectors to the query
	Search(query []float32, k int) ([]*SearchResult, error)

	// SearchWithFilter finds vectors with metadata filtering
	SearchWithFilter(query []float32, k int, filter FilterFunc) ([]*SearchResult, error)

	// Get retrieves a vector by ID
	Get(id string) (*Vector, error)

	// Delete removes a vector from the index
	Delete(id string) error

	// Size returns the number of vectors in the index
	Size() int

	// Dimension returns the dimension of vectors in the index
	Dimension() int

	// Close cleans up resources
	Close() error
}

// FilterFunc represents a function to filter vectors based on metadata
type FilterFunc func(metadata map[string]interface{}) bool

// SafeMap provides thread-safe access to a map
type SafeMap struct {
	mu   sync.RWMutex
	data map[string]*Vector
}

// NewSafeMap creates a new thread-safe map
func NewSafeMap() *SafeMap {
	return &SafeMap{
		data: make(map[string]*Vector),
	}
}

// Get retrieves a value from the map
func (sm *SafeMap) Get(key string) (*Vector, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	val, ok := sm.data[key]
	return val, ok
}

// Set stores a value in the map
func (sm *SafeMap) Set(key string, value *Vector) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.data[key] = value
}

// Delete removes a value from the map
func (sm *SafeMap) Delete(key string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	delete(sm.data, key)
}

// Size returns the number of elements in the map
func (sm *SafeMap) Size() int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return len(sm.data)
}

// Keys returns all keys in the map
func (sm *SafeMap) Keys() []string {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	keys := make([]string, 0, len(sm.data))
	for k := range sm.data {
		keys = append(keys, k)
	}
	return keys
}

// ForEach iterates over all key-value pairs
func (sm *SafeMap) ForEach(fn func(key string, value *Vector)) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	for k, v := range sm.data {
		fn(k, v)
	}
}
