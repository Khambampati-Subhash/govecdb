// Package store provides storage implementations for GoVecDB.
package store

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// MemoryStore implements VectorStore interface with in-memory storage.
// It provides thread-safe operations with proper locking and supports
// concurrent read/write operations using RWMutex.
type MemoryStore struct {
	// Primary storage map: id -> vector
	vectors map[string]*api.Vector

	// Metadata for statistics and management
	metadata *StoreMetadata

	// Read-write mutex for thread-safe operations
	mu sync.RWMutex

	// Configuration
	config *StoreConfig

	// Closed flag to prevent operations on closed store
	closed bool
}

// StoreMetadata holds metadata about the store
type StoreMetadata struct {
	CreatedAt   time.Time
	UpdatedAt   time.Time
	VectorCount int64
	SizeBytes   int64
	Stats       *StoreStats
}

// StoreStats holds operational statistics
type StoreStats struct {
	TotalPuts    int64
	TotalGets    int64
	TotalDeletes int64
	TotalScans   int64
	CacheHits    int64
	CacheMisses  int64
}

// StoreConfig holds configuration for the memory store
type StoreConfig struct {
	Name          string
	MaxVectors    int64 // Maximum number of vectors (0 = unlimited)
	MaxSizeBytes  int64 // Maximum size in bytes (0 = unlimited)
	EnableStats   bool  // Whether to collect detailed statistics
	PreallocSize  int   // Pre-allocate map size
	EnableMetrics bool  // Enable performance metrics
}

// DefaultStoreConfig returns a default store configuration
func DefaultStoreConfig(name string) *StoreConfig {
	return &StoreConfig{
		Name:          name,
		MaxVectors:    0,
		MaxSizeBytes:  0,
		EnableStats:   true,
		PreallocSize:  1000,
		EnableMetrics: true,
	}
}

// NewMemoryStore creates a new memory store instance
func NewMemoryStore(config *StoreConfig) *MemoryStore {
	if config == nil {
		config = DefaultStoreConfig("default")
	}

	store := &MemoryStore{
		vectors: make(map[string]*api.Vector, config.PreallocSize),
		config:  config,
		metadata: &StoreMetadata{
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			VectorCount: 0,
			SizeBytes:   0,
			Stats:       &StoreStats{},
		},
		closed: false,
	}

	return store
}

// Put stores a vector in the memory store
func (m *MemoryStore) Put(ctx context.Context, vector *api.Vector) error {
	if vector == nil {
		return api.ErrEmptyVector
	}

	if err := vector.Validate(); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	// Check capacity limits
	if err := m.checkCapacityLocked(vector); err != nil {
		return err
	}

	// Check if vector already exists
	existing, exists := m.vectors[vector.ID]
	var sizeDelta int64

	if exists {
		// Calculate size difference
		oldSize := m.calculateVectorSize(existing)
		newSize := m.calculateVectorSize(vector)
		sizeDelta = newSize - oldSize
	} else {
		// New vector
		sizeDelta = m.calculateVectorSize(vector)
		m.metadata.VectorCount++
	}

	// Clone the vector to prevent external modifications
	m.vectors[vector.ID] = vector.Clone()
	m.metadata.SizeBytes += sizeDelta
	m.metadata.UpdatedAt = time.Now()

	// Update statistics
	if m.config.EnableStats {
		m.metadata.Stats.TotalPuts++
	}

	return nil
}

// Get retrieves a vector by ID
func (m *MemoryStore) Get(ctx context.Context, id string) (*api.Vector, error) {
	if id == "" {
		return nil, api.ErrVectorNotFound
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	vector, exists := m.vectors[id]
	if !exists {
		if m.config.EnableStats {
			m.metadata.Stats.CacheMisses++
		}
		return nil, api.ErrVectorNotFound
	}

	// Update statistics
	if m.config.EnableStats {
		m.metadata.Stats.TotalGets++
		m.metadata.Stats.CacheHits++
	}

	// Return a clone to prevent external modifications
	return vector.Clone(), nil
}

// Delete removes a vector by ID
func (m *MemoryStore) Delete(ctx context.Context, id string) error {
	if id == "" {
		return api.ErrVectorNotFound
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	vector, exists := m.vectors[id]
	if !exists {
		return api.ErrVectorNotFound
	}

	// Calculate size to subtract
	sizeToRemove := m.calculateVectorSize(vector)

	// Remove the vector
	delete(m.vectors, id)
	m.metadata.VectorCount--
	m.metadata.SizeBytes -= sizeToRemove
	m.metadata.UpdatedAt = time.Now()

	// Update statistics
	if m.config.EnableStats {
		m.metadata.Stats.TotalDeletes++
	}

	return nil
}

// List returns vectors with pagination
func (m *MemoryStore) List(ctx context.Context, limit int, offset int) ([]*api.Vector, error) {
	if limit < 0 || offset < 0 {
		return nil, api.ErrInvalidConfig
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	// Get all vector IDs and sort them for consistent ordering
	ids := make([]string, 0, len(m.vectors))
	for id := range m.vectors {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	// Apply pagination
	start := offset
	if start >= len(ids) {
		return []*api.Vector{}, nil
	}

	end := start + limit
	if limit == 0 || end > len(ids) {
		end = len(ids)
	}

	// Build result
	result := make([]*api.Vector, 0, end-start)
	for i := start; i < end; i++ {
		if vector, exists := m.vectors[ids[i]]; exists {
			result = append(result, vector.Clone())
		}
	}

	// Update statistics
	if m.config.EnableStats {
		m.metadata.Stats.TotalScans++
	}

	return result, nil
}

// Count returns the total number of vectors
func (m *MemoryStore) Count(ctx context.Context) (int64, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return 0, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return 0, api.ErrContextCanceled
	}

	return m.metadata.VectorCount, nil
}

// PutBatch stores multiple vectors in a single operation
func (m *MemoryStore) PutBatch(ctx context.Context, vectors []*api.Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	// Validate all vectors first
	for i, vector := range vectors {
		if vector == nil {
			return api.ErrEmptyVector
		}
		if err := vector.Validate(); err != nil {
			return err
		}

		// Check for duplicates within the batch
		for j := i + 1; j < len(vectors); j++ {
			if vectors[j] != nil && vector.ID == vectors[j].ID {
				return api.ErrCollectionExists // Reusing error for duplicate IDs
			}
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	// Calculate total size impact and check capacity
	var totalSizeDelta int64
	var newVectorCount int64

	for _, vector := range vectors {
		if existing, exists := m.vectors[vector.ID]; exists {
			// Existing vector - calculate size difference
			oldSize := m.calculateVectorSize(existing)
			newSize := m.calculateVectorSize(vector)
			totalSizeDelta += newSize - oldSize
		} else {
			// New vector
			totalSizeDelta += m.calculateVectorSize(vector)
			newVectorCount++
		}
	}

	// Check capacity constraints
	if m.config.MaxVectors > 0 && m.metadata.VectorCount+newVectorCount > m.config.MaxVectors {
		return api.ErrInvalidConfig
	}
	if m.config.MaxSizeBytes > 0 && m.metadata.SizeBytes+totalSizeDelta > m.config.MaxSizeBytes {
		return api.ErrInvalidConfig
	}

	// Perform batch insertion
	for _, vector := range vectors {
		if _, exists := m.vectors[vector.ID]; !exists {
			m.metadata.VectorCount++
		}
		m.vectors[vector.ID] = vector.Clone()
	}

	m.metadata.SizeBytes += totalSizeDelta
	m.metadata.UpdatedAt = time.Now()

	// Update statistics
	if m.config.EnableStats {
		m.metadata.Stats.TotalPuts += int64(len(vectors))
	}

	return nil
}

// GetBatch retrieves multiple vectors by their IDs
func (m *MemoryStore) GetBatch(ctx context.Context, ids []string) ([]*api.Vector, error) {
	if len(ids) == 0 {
		return []*api.Vector{}, nil
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	result := make([]*api.Vector, 0, len(ids))

	for _, id := range ids {
		if vector, exists := m.vectors[id]; exists {
			result = append(result, vector.Clone())
			if m.config.EnableStats {
				m.metadata.Stats.CacheHits++
			}
		} else {
			if m.config.EnableStats {
				m.metadata.Stats.CacheMisses++
			}
		}
	}

	// Update statistics
	if m.config.EnableStats {
		m.metadata.Stats.TotalGets += int64(len(ids))
	}

	return result, nil
}

// DeleteBatch removes multiple vectors by their IDs
func (m *MemoryStore) DeleteBatch(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	var deletedCount int64
	var sizeReduction int64

	for _, id := range ids {
		if vector, exists := m.vectors[id]; exists {
			sizeReduction += m.calculateVectorSize(vector)
			delete(m.vectors, id)
			deletedCount++
		}
	}

	m.metadata.VectorCount -= deletedCount
	m.metadata.SizeBytes -= sizeReduction
	m.metadata.UpdatedAt = time.Now()

	// Update statistics
	if m.config.EnableStats {
		m.metadata.Stats.TotalDeletes += int64(len(ids))
	}

	return nil
}

// Filter returns vectors that match the given filter expression
func (m *MemoryStore) Filter(ctx context.Context, filter api.FilterExpr, limit int) ([]*api.Vector, error) {
	if filter == nil {
		return m.List(ctx, limit, 0)
	}

	if err := filter.Validate(); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	result := make([]*api.Vector, 0)
	count := 0

	for _, vector := range m.vectors {
		// Check context periodically for long-running filters
		if count%1000 == 0 && ctx.Err() != nil {
			return nil, api.ErrContextCanceled
		}

		if filter.Evaluate(vector.Metadata) {
			result = append(result, vector.Clone())
			count++

			if limit > 0 && count >= limit {
				break
			}
		}
	}

	// Update statistics
	if m.config.EnableStats {
		m.metadata.Stats.TotalScans++
	}

	return result, nil
}

// Clear removes all vectors from the store
func (m *MemoryStore) Clear(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return api.ErrContextCanceled
	}

	m.vectors = make(map[string]*api.Vector, m.config.PreallocSize)
	m.metadata.VectorCount = 0
	m.metadata.SizeBytes = 0
	m.metadata.UpdatedAt = time.Now()

	return nil
}

// Close closes the memory store and prevents further operations
func (m *MemoryStore) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return api.ErrClosed
	}

	m.closed = true
	m.vectors = nil
	return nil
}

// Stats returns statistics about the store
func (m *MemoryStore) Stats(ctx context.Context) (*api.CollectionStats, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return nil, api.ErrClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, api.ErrContextCanceled
	}

	stats := &api.CollectionStats{
		Name:        m.config.Name,
		VectorCount: m.metadata.VectorCount,
		CreatedAt:   m.metadata.CreatedAt,
		UpdatedAt:   m.metadata.UpdatedAt,
		SizeBytes:   m.metadata.SizeBytes,
		MemoryUsage: m.metadata.SizeBytes, // For memory store, these are the same
	}

	return stats, nil
}

// Config returns the store configuration
func (m *MemoryStore) Config() *StoreConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Return a copy to prevent external modifications
	configCopy := *m.config
	return &configCopy
}

// IsClosed returns whether the store is closed
func (m *MemoryStore) IsClosed() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.closed
}

// Private helper methods

// checkCapacityLocked checks capacity constraints (must be called with write lock held)
func (m *MemoryStore) checkCapacityLocked(vector *api.Vector) error {
	// Check vector count limit
	if m.config.MaxVectors > 0 {
		if _, exists := m.vectors[vector.ID]; !exists {
			// New vector
			if m.metadata.VectorCount >= m.config.MaxVectors {
				return api.ErrInvalidConfig
			}
		}
	}

	// Check size limit
	if m.config.MaxSizeBytes > 0 {
		vectorSize := m.calculateVectorSize(vector)
		if existing, exists := m.vectors[vector.ID]; exists {
			// Replacing existing - check size difference
			existingSize := m.calculateVectorSize(existing)
			sizeDelta := vectorSize - existingSize
			if m.metadata.SizeBytes+sizeDelta > m.config.MaxSizeBytes {
				return api.ErrInvalidConfig
			}
		} else {
			// New vector
			if m.metadata.SizeBytes+vectorSize > m.config.MaxSizeBytes {
				return api.ErrInvalidConfig
			}
		}
	}

	return nil
}

// calculateVectorSize estimates the memory size of a vector
func (m *MemoryStore) calculateVectorSize(vector *api.Vector) int64 {
	if vector == nil {
		return 0
	}

	size := int64(len(vector.ID))       // ID string
	size += int64(len(vector.Data) * 4) // float32 data

	// Estimate metadata size
	for key, val := range vector.Metadata {
		size += int64(len(key))
		switch v := val.(type) {
		case string:
			size += int64(len(v))
		case int, int32, int64, uint, uint32, uint64:
			size += 8
		case float32, float64:
			size += 8
		case bool:
			size += 1
		default:
			size += 16 // Rough estimate for other types
		}
	}

	return size
}

// Validate checks if the store is in a valid state
func (m *MemoryStore) Validate() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return api.ErrClosed
	}

	// Validate vector count consistency
	if int64(len(m.vectors)) != m.metadata.VectorCount {
		return api.ErrInvalidConfig
	}

	// Validate all stored vectors
	for id, vector := range m.vectors {
		if vector == nil {
			return api.ErrEmptyVector
		}
		if vector.ID != id {
			return api.ErrInvalidConfig
		}
		if err := vector.Validate(); err != nil {
			return err
		}
	}

	return nil
}
