// Package segment implements memory-based and file-based segment storage.
package segment

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// MemorySegment implements an in-memory segment for fast access and testing
type MemorySegment struct {
	// Configuration and metadata
	id       string
	config   *SegmentConfig
	metadata *SegmentMetadata

	// Data storage
	vectors map[string]*api.Vector

	// State management
	state  SegmentState
	frozen bool
	closed bool

	// Statistics (atomic counters for thread safety)
	readCount   int64
	writeCount  int64
	errorCount  int64
	sizeBytes   int64
	vectorCount int64

	// Performance tracking
	lastAccessed time.Time
	stats        *SegmentStats

	// Thread safety
	mu sync.RWMutex

	// Bloom filter for membership testing
	bloom BloomFilter
}

// NewMemorySegment creates a new in-memory segment
func NewMemorySegment(config *SegmentConfig) (*MemorySegment, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	segment := &MemorySegment{
		id:           config.ID,
		config:       config,
		vectors:      make(map[string]*api.Vector),
		state:        SegmentStateActive,
		frozen:       false,
		closed:       false,
		lastAccessed: time.Now(),
		bloom:        NewSimpleBloomFilter(config.BloomFilterBits),
		metadata: &SegmentMetadata{
			ID:                 config.ID,
			Type:               config.Type,
			State:              SegmentStateActive,
			Generation:         1,
			VectorCount:        0,
			SizeBytes:          0,
			CreatedAt:          time.Now(),
			ModifiedAt:         time.Now(),
			AccessedAt:         time.Now(),
			CompactionLevel:    0,
			CompressionRatio:   1.0,
			FragmentationRatio: 0.0,
		},
		stats: &SegmentStats{
			VectorCount:    0,
			UniqueVectors:  0,
			DeletedVectors: 0,
			SizeBytes:      0,
			ReadLatency:    0.0,
			WriteLatency:   0.0,
			CacheHitRate:   1.0, // Memory segments always hit
		},
	}

	return segment, nil
}

// ID returns the segment ID
func (s *MemorySegment) ID() string {
	return s.id
}

// Type returns the segment type
func (s *MemorySegment) Type() SegmentType {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.config.Type
}

// State returns the current segment state
func (s *MemorySegment) State() SegmentState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state
}

// Metadata returns segment metadata
func (s *MemorySegment) Metadata() *SegmentMetadata {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Update metadata with current values
	s.metadata.VectorCount = s.vectorCount
	s.metadata.SizeBytes = s.sizeBytes
	s.metadata.ModifiedAt = time.Now()
	s.metadata.AccessedAt = s.lastAccessed
	s.metadata.ReadCount = s.readCount
	s.metadata.WriteCount = s.writeCount
	s.metadata.ErrorCount = s.errorCount

	// Return a copy
	metadataCopy := *s.metadata
	return &metadataCopy
}

// Stats returns segment statistics
func (s *MemorySegment) Stats() *SegmentStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Update stats with current values
	s.stats.VectorCount = s.vectorCount
	s.stats.SizeBytes = s.sizeBytes
	s.stats.UniqueVectors = int64(len(s.vectors))

	// Return a copy
	statsCopy := *s.stats
	return &statsCopy
}

// Put stores a vector in the segment
func (s *MemorySegment) Put(ctx context.Context, vector *api.Vector) error {
	if vector == nil {
		return fmt.Errorf("vector cannot be nil")
	}

	if err := vector.Validate(); err != nil {
		return fmt.Errorf("invalid vector: %w", err)
	}

	start := time.Now()
	defer func() {
		duration := time.Since(start)
		s.updateWriteLatency(float64(duration.Nanoseconds()) / 1e6)
		atomic.AddInt64(&s.writeCount, 1)
	}()

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		atomic.AddInt64(&s.errorCount, 1)
		return ErrSegmentClosed
	}

	if s.frozen {
		atomic.AddInt64(&s.errorCount, 1)
		return fmt.Errorf("segment is frozen")
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return ctx.Err()
	}

	// Check if segment is full
	if s.isFull() {
		atomic.AddInt64(&s.errorCount, 1)
		return ErrSegmentFull
	}

	// Calculate size delta
	sizeDelta := s.calculateVectorSize(vector)

	// Check if adding this vector would exceed limits
	if s.config.MaxVectors > 0 && s.vectorCount >= s.config.MaxVectors {
		atomic.AddInt64(&s.errorCount, 1)
		return ErrSegmentFull
	}

	if s.config.MaxSizeBytes > 0 && s.sizeBytes+sizeDelta > s.config.MaxSizeBytes {
		atomic.AddInt64(&s.errorCount, 1)
		return ErrSegmentFull
	}

	// Check if vector already exists
	existing, exists := s.vectors[vector.ID]
	if exists {
		// Update existing vector
		oldSize := s.calculateVectorSize(existing)
		sizeDelta = sizeDelta - oldSize
	} else {
		// New vector
		atomic.AddInt64(&s.vectorCount, 1)
		s.bloom.Add([]byte(vector.ID))
	}

	// Store the vector (clone to prevent external modifications)
	s.vectors[vector.ID] = vector.Clone()
	atomic.AddInt64(&s.sizeBytes, sizeDelta)
	s.lastAccessed = time.Now()

	// Update state
	if s.state == SegmentStateActive {
		s.metadata.ModifiedAt = time.Now()
	}

	return nil
}

// Get retrieves a vector by ID
func (s *MemorySegment) Get(ctx context.Context, id string) (*api.Vector, error) {
	if id == "" {
		return nil, fmt.Errorf("vector ID cannot be empty")
	}

	start := time.Now()
	defer func() {
		duration := time.Since(start)
		s.updateReadLatency(float64(duration.Nanoseconds()) / 1e6)
		atomic.AddInt64(&s.readCount, 1)
	}()

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		atomic.AddInt64(&s.errorCount, 1)
		return nil, ErrSegmentClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	// Check bloom filter first (if available)
	if s.bloom != nil && !s.bloom.Contains([]byte(id)) {
		return nil, api.ErrVectorNotFound
	}

	// Get the vector
	vector, exists := s.vectors[id]
	if !exists {
		return nil, api.ErrVectorNotFound
	}

	s.lastAccessed = time.Now()

	// Return a clone to prevent external modifications
	return vector.Clone(), nil
}

// Delete removes a vector by ID
func (s *MemorySegment) Delete(ctx context.Context, id string) error {
	if id == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	start := time.Now()
	defer func() {
		duration := time.Since(start)
		s.updateWriteLatency(float64(duration.Nanoseconds()) / 1e6)
		atomic.AddInt64(&s.writeCount, 1)
	}()

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		atomic.AddInt64(&s.errorCount, 1)
		return ErrSegmentClosed
	}

	if s.frozen {
		atomic.AddInt64(&s.errorCount, 1)
		return fmt.Errorf("segment is frozen")
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return ctx.Err()
	}

	// Check if vector exists
	vector, exists := s.vectors[id]
	if !exists {
		return api.ErrVectorNotFound
	}

	// Calculate size to remove
	sizeToRemove := s.calculateVectorSize(vector)

	// Remove the vector
	delete(s.vectors, id)
	atomic.AddInt64(&s.vectorCount, -1)
	atomic.AddInt64(&s.sizeBytes, -sizeToRemove)
	s.lastAccessed = time.Now()

	// Update deleted vectors count
	atomic.AddInt64(&s.stats.DeletedVectors, 1)

	// Update state
	if s.state == SegmentStateActive {
		s.metadata.ModifiedAt = time.Now()
	}

	return nil
}

// Contains checks if the segment contains a vector with the given ID
func (s *MemorySegment) Contains(ctx context.Context, id string) (bool, error) {
	if id == "" {
		return false, fmt.Errorf("vector ID cannot be empty")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return false, ErrSegmentClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return false, ctx.Err()
	}

	// Check bloom filter first
	if s.bloom != nil && !s.bloom.Contains([]byte(id)) {
		return false, nil
	}

	_, exists := s.vectors[id]
	return exists, nil
}

// PutBatch stores multiple vectors in a single operation
func (s *MemorySegment) PutBatch(ctx context.Context, vectors []*api.Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	// Validate all vectors first
	for _, vector := range vectors {
		if vector == nil {
			return fmt.Errorf("batch contains nil vector")
		}
		if err := vector.Validate(); err != nil {
			return fmt.Errorf("invalid vector in batch: %w", err)
		}
	}

	start := time.Now()
	defer func() {
		duration := time.Since(start)
		s.updateWriteLatency(float64(duration.Nanoseconds()) / 1e6)
		atomic.AddInt64(&s.writeCount, int64(len(vectors)))
	}()

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		atomic.AddInt64(&s.errorCount, 1)
		return ErrSegmentClosed
	}

	if s.frozen {
		atomic.AddInt64(&s.errorCount, 1)
		return fmt.Errorf("segment is frozen")
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return ctx.Err()
	}

	// Calculate total size impact
	var totalSizeDelta int64
	var newVectorCount int64

	for _, vector := range vectors {
		sizeDelta := s.calculateVectorSize(vector)

		if existing, exists := s.vectors[vector.ID]; exists {
			// Update existing vector
			oldSize := s.calculateVectorSize(existing)
			totalSizeDelta += sizeDelta - oldSize
		} else {
			// New vector
			totalSizeDelta += sizeDelta
			newVectorCount++
		}
	}

	// Check capacity limits
	if s.config.MaxVectors > 0 && s.vectorCount+newVectorCount > s.config.MaxVectors {
		atomic.AddInt64(&s.errorCount, 1)
		return ErrSegmentFull
	}

	if s.config.MaxSizeBytes > 0 && s.sizeBytes+totalSizeDelta > s.config.MaxSizeBytes {
		atomic.AddInt64(&s.errorCount, 1)
		return ErrSegmentFull
	}

	// Perform batch insertion
	for _, vector := range vectors {
		if _, exists := s.vectors[vector.ID]; !exists {
			atomic.AddInt64(&s.vectorCount, 1)
			s.bloom.Add([]byte(vector.ID))
		}
		s.vectors[vector.ID] = vector.Clone()
	}

	atomic.AddInt64(&s.sizeBytes, totalSizeDelta)
	s.lastAccessed = time.Now()

	// Update state
	if s.state == SegmentStateActive {
		s.metadata.ModifiedAt = time.Now()
	}

	return nil
}

// GetBatch retrieves multiple vectors by their IDs
func (s *MemorySegment) GetBatch(ctx context.Context, ids []string) ([]*api.Vector, error) {
	if len(ids) == 0 {
		return []*api.Vector{}, nil
	}

	start := time.Now()
	defer func() {
		duration := time.Since(start)
		s.updateReadLatency(float64(duration.Nanoseconds()) / 1e6)
		atomic.AddInt64(&s.readCount, int64(len(ids)))
	}()

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		atomic.AddInt64(&s.errorCount, 1)
		return nil, ErrSegmentClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	result := make([]*api.Vector, 0, len(ids))

	for _, id := range ids {
		// Check bloom filter first
		if s.bloom != nil && !s.bloom.Contains([]byte(id)) {
			continue
		}

		if vector, exists := s.vectors[id]; exists {
			result = append(result, vector.Clone())
		}
	}

	s.lastAccessed = time.Now()
	return result, nil
}

// DeleteBatch removes multiple vectors by their IDs
func (s *MemorySegment) DeleteBatch(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	start := time.Now()
	defer func() {
		duration := time.Since(start)
		s.updateWriteLatency(float64(duration.Nanoseconds()) / 1e6)
		atomic.AddInt64(&s.writeCount, int64(len(ids)))
	}()

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		atomic.AddInt64(&s.errorCount, 1)
		return ErrSegmentClosed
	}

	if s.frozen {
		atomic.AddInt64(&s.errorCount, 1)
		return fmt.Errorf("segment is frozen")
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return ctx.Err()
	}

	var deletedCount int64
	var sizeReduction int64

	for _, id := range ids {
		if vector, exists := s.vectors[id]; exists {
			sizeReduction += s.calculateVectorSize(vector)
			delete(s.vectors, id)
			deletedCount++
		}
	}

	atomic.AddInt64(&s.vectorCount, -deletedCount)
	atomic.AddInt64(&s.sizeBytes, -sizeReduction)
	atomic.AddInt64(&s.stats.DeletedVectors, deletedCount)
	s.lastAccessed = time.Now()

	// Update state
	if s.state == SegmentStateActive {
		s.metadata.ModifiedAt = time.Now()
	}

	return nil
}

// Scan iterates over all vectors in the segment
func (s *MemorySegment) Scan(ctx context.Context, callback func(*api.Vector) bool) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return ErrSegmentClosed
	}

	// Create sorted list of keys for consistent iteration
	keys := make([]string, 0, len(s.vectors))
	for key := range s.vectors {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	// Iterate over vectors
	for _, key := range keys {
		// Check context cancellation periodically
		if ctx.Err() != nil {
			return ctx.Err()
		}

		if vector, exists := s.vectors[key]; exists {
			if !callback(vector.Clone()) {
				break // Stop iteration if callback returns false
			}
		}
	}

	return nil
}

// ScanRange iterates over vectors in a key range
func (s *MemorySegment) ScanRange(ctx context.Context, start, end string, callback func(*api.Vector) bool) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return ErrSegmentClosed
	}

	// Create sorted list of keys for consistent iteration
	keys := make([]string, 0, len(s.vectors))
	for key := range s.vectors {
		if key >= start && (end == "" || key <= end) {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)

	// Iterate over vectors in range
	for _, key := range keys {
		// Check context cancellation periodically
		if ctx.Err() != nil {
			return ctx.Err()
		}

		if vector, exists := s.vectors[key]; exists {
			if !callback(vector.Clone()) {
				break // Stop iteration if callback returns false
			}
		}
	}

	return nil
}

// Keys returns all vector IDs in the segment
func (s *MemorySegment) Keys(ctx context.Context) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return nil, ErrSegmentClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	keys := make([]string, 0, len(s.vectors))
	for key := range s.vectors {
		keys = append(keys, key)
	}

	sort.Strings(keys)
	return keys, nil
}

// Filter returns vectors that match the given filter expression
func (s *MemorySegment) Filter(ctx context.Context, filter api.FilterExpr) ([]*api.Vector, error) {
	if filter == nil {
		return nil, fmt.Errorf("filter expression cannot be nil")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return nil, ErrSegmentClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	result := make([]*api.Vector, 0)

	for _, vector := range s.vectors {
		// Check context periodically for long-running filters
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		if filter.Evaluate(vector.Metadata) {
			result = append(result, vector.Clone())
		}
	}

	return result, nil
}

// FilterKeys returns vector IDs that match the given filter expression
func (s *MemorySegment) FilterKeys(ctx context.Context, filter api.FilterExpr) ([]string, error) {
	if filter == nil {
		return nil, fmt.Errorf("filter expression cannot be nil")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return nil, ErrSegmentClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	result := make([]string, 0)

	for id, vector := range s.vectors {
		// Check context periodically
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		if filter.Evaluate(vector.Metadata) {
			result = append(result, id)
		}
	}

	sort.Strings(result)
	return result, nil
}

// Freeze makes the segment immutable
func (s *MemorySegment) Freeze() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrSegmentClosed
	}

	s.frozen = true
	s.state = SegmentStateImmutable
	s.metadata.State = SegmentStateImmutable
	s.metadata.ModifiedAt = time.Now()

	return nil
}

// Compact compacts the segment (removes deleted entries, optimizes storage)
func (s *MemorySegment) Compact(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrSegmentClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return ctx.Err()
	}

	// For memory segments, compaction mainly involves updating statistics
	// and potentially rebuilding the bloom filter

	s.state = SegmentStateCompacting
	s.metadata.State = SegmentStateCompacting

	// Rebuild bloom filter with current keys
	s.bloom.Clear()
	for key := range s.vectors {
		s.bloom.Add([]byte(key))
	}

	// Reset deleted vectors count
	s.stats.DeletedVectors = 0

	// Update fragmentation ratio
	if s.vectorCount > 0 {
		s.metadata.FragmentationRatio = float64(s.stats.DeletedVectors) / float64(s.vectorCount)
	}

	// Mark as compacted
	s.state = SegmentStateImmutable
	s.metadata.State = SegmentStateImmutable
	s.metadata.CompactedAt = time.Now()
	s.metadata.CompactionLevel++

	return nil
}

// Merge merges this segment with another segment
func (s *MemorySegment) Merge(ctx context.Context, other Segment) (Segment, error) {
	if other == nil {
		return nil, fmt.Errorf("other segment cannot be nil")
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	// Create new segment for merge result
	mergeConfig := *s.config
	mergeConfig.ID = fmt.Sprintf("%s-%s-merged", s.id, other.ID())
	mergeConfig.Type = CompactedSegment

	mergedSegment, err := NewMemorySegment(&mergeConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create merged segment: %w", err)
	}

	// Copy vectors from this segment
	err = s.Scan(ctx, func(vector *api.Vector) bool {
		if err := mergedSegment.Put(ctx, vector); err != nil {
			return false // Stop on error
		}
		return true
	})
	if err != nil {
		return nil, fmt.Errorf("failed to copy vectors from source segment: %w", err)
	}

	// Copy vectors from other segment
	err = other.Scan(ctx, func(vector *api.Vector) bool {
		if err := mergedSegment.Put(ctx, vector); err != nil {
			return false // Stop on error
		}
		return true
	})
	if err != nil {
		return nil, fmt.Errorf("failed to copy vectors from other segment: %w", err)
	}

	// Freeze the merged segment
	if err := mergedSegment.Freeze(); err != nil {
		return nil, fmt.Errorf("failed to freeze merged segment: %w", err)
	}

	// Update metadata
	mergedSegment.metadata.SourceSegments = []string{s.id, other.ID()}
	mergedSegment.metadata.CompactionLevel = maxInt(s.metadata.CompactionLevel, other.Metadata().CompactionLevel) + 1

	return mergedSegment, nil
}

// Size returns the size of the segment in bytes
func (s *MemorySegment) Size() int64 {
	return atomic.LoadInt64(&s.sizeBytes)
}

// VectorCount returns the number of vectors in the segment
func (s *MemorySegment) VectorCount() int64 {
	return atomic.LoadInt64(&s.vectorCount)
}

// IsEmpty returns true if the segment contains no vectors
func (s *MemorySegment) IsEmpty() bool {
	return atomic.LoadInt64(&s.vectorCount) == 0
}

// IsFull returns true if the segment has reached its capacity
func (s *MemorySegment) IsFull() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.isFull()
}

// CanAcceptWrites returns true if the segment can accept write operations
func (s *MemorySegment) CanAcceptWrites() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return !s.closed && !s.frozen && s.state == SegmentStateActive && !s.isFull()
}

// Validate validates the segment integrity
func (s *MemorySegment) Validate(ctx context.Context) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return ErrSegmentClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return ctx.Err()
	}

	// Validate vector count consistency
	if int64(len(s.vectors)) != s.vectorCount {
		return fmt.Errorf("vector count mismatch: expected %d, got %d", s.vectorCount, len(s.vectors))
	}

	// Validate all vectors
	for id, vector := range s.vectors {
		if vector == nil {
			return fmt.Errorf("nil vector found for ID: %s", id)
		}

		if vector.ID != id {
			return fmt.Errorf("vector ID mismatch: expected %s, got %s", id, vector.ID)
		}

		if err := vector.Validate(); err != nil {
			return fmt.Errorf("invalid vector %s: %w", id, err)
		}
	}

	return nil
}

// Repair attempts to repair segment corruption
func (s *MemorySegment) Repair(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrSegmentClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return ctx.Err()
	}

	// For memory segments, repair mainly involves:
	// 1. Remove invalid vectors
	// 2. Rebuild bloom filter
	// 3. Recalculate statistics

	var repaired int64
	var sizeReduction int64

	for id, vector := range s.vectors {
		if vector == nil || vector.ID != id || vector.Validate() != nil {
			// Remove invalid vector
			sizeReduction += s.calculateVectorSize(vector)
			delete(s.vectors, id)
			repaired++
		}
	}

	// Update counters
	atomic.AddInt64(&s.vectorCount, -repaired)
	atomic.AddInt64(&s.sizeBytes, -sizeReduction)

	// Rebuild bloom filter
	s.bloom.Clear()
	for key := range s.vectors {
		s.bloom.Add([]byte(key))
	}

	// Update statistics
	s.stats.UniqueVectors = int64(len(s.vectors))

	return nil
}

// Checkpoint creates a checkpoint of the segment
func (s *MemorySegment) Checkpoint(ctx context.Context) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return ErrSegmentClosed
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return ctx.Err()
	}

	// For memory segments, checkpoint mainly involves updating metadata
	s.metadata.AccessedAt = time.Now()

	// Could also serialize to disk if needed
	if s.config.BasePath != "" {
		return s.serializeMetadata()
	}

	return nil
}

// Open opens the segment (no-op for memory segments)
func (s *MemorySegment) Open(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		s.closed = false
		s.state = SegmentStateActive
		s.metadata.State = SegmentStateActive
	}

	return nil
}

// Close closes the segment and releases resources
func (s *MemorySegment) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrSegmentClosed
	}

	s.closed = true
	s.state = SegmentStateDeleted
	s.metadata.State = SegmentStateDeleted

	// Clear data to help GC
	s.vectors = nil
	s.bloom = nil

	return nil
}

// IsClosed returns true if the segment is closed
func (s *MemorySegment) IsClosed() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.closed
}

// Private helper methods

// isFull checks if the segment has reached capacity (must be called with lock held)
func (s *MemorySegment) isFull() bool {
	if s.config.MaxVectors > 0 && s.vectorCount >= s.config.MaxVectors {
		return true
	}

	if s.config.MaxSizeBytes > 0 && s.sizeBytes >= s.config.MaxSizeBytes {
		return true
	}

	return false
}

// calculateVectorSize estimates the memory size of a vector
func (s *MemorySegment) calculateVectorSize(vector *api.Vector) int64 {
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

// updateReadLatency updates the running average read latency
func (s *MemorySegment) updateReadLatency(latency float64) {
	count := atomic.LoadInt64(&s.readCount)
	if count == 0 {
		s.stats.ReadLatency = latency
	} else {
		// Running average
		s.stats.ReadLatency = (s.stats.ReadLatency*float64(count-1) + latency) / float64(count)
	}
}

// updateWriteLatency updates the running average write latency
func (s *MemorySegment) updateWriteLatency(latency float64) {
	count := atomic.LoadInt64(&s.writeCount)
	if count == 0 {
		s.stats.WriteLatency = latency
	} else {
		// Running average
		s.stats.WriteLatency = (s.stats.WriteLatency*float64(count-1) + latency) / float64(count)
	}
}

// serializeMetadata serializes segment metadata to disk
func (s *MemorySegment) serializeMetadata() error {
	if s.config.BasePath == "" {
		return nil
	}

	// Ensure directory exists
	if err := os.MkdirAll(s.config.BasePath, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Serialize metadata to JSON
	metadataPath := filepath.Join(s.config.BasePath, s.id+".meta")
	data, err := json.MarshalIndent(s.metadata, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	// Write to file
	if err := os.WriteFile(metadataPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write metadata file: %w", err)
	}

	return nil
}

// SimpleBloomFilter implements a basic bloom filter
type SimpleBloomFilter struct {
	bits      []bool
	size      int
	hashFuncs int
	elements  int64
}

// NewSimpleBloomFilter creates a new simple bloom filter
func NewSimpleBloomFilter(bitsPerElement int) *SimpleBloomFilter {
	if bitsPerElement < 1 {
		bitsPerElement = 10
	}

	size := 1000 * bitsPerElement                     // Start with space for 1000 elements
	hashFuncs := int(float64(bitsPerElement) * 0.693) // ln(2) * bits_per_element
	if hashFuncs < 1 {
		hashFuncs = 1
	}

	return &SimpleBloomFilter{
		bits:      make([]bool, size),
		size:      size,
		hashFuncs: hashFuncs,
		elements:  0,
	}
}

// Add adds a key to the bloom filter
func (bf *SimpleBloomFilter) Add(key []byte) {
	for i := 0; i < bf.hashFuncs; i++ {
		hash := bf.hash(key, i)
		bf.bits[hash%bf.size] = true
	}
	bf.elements++
}

// Contains checks if a key might be in the set
func (bf *SimpleBloomFilter) Contains(key []byte) bool {
	for i := 0; i < bf.hashFuncs; i++ {
		hash := bf.hash(key, i)
		if !bf.bits[hash%bf.size] {
			return false
		}
	}
	return true
}

// Size returns the number of elements added to the filter
func (bf *SimpleBloomFilter) Size() int64 {
	return bf.elements
}

// Clear clears the bloom filter
func (bf *SimpleBloomFilter) Clear() {
	for i := range bf.bits {
		bf.bits[i] = false
	}
	bf.elements = 0
}

// FalsePositiveRate estimates the false positive rate
func (bf *SimpleBloomFilter) FalsePositiveRate() float64 {
	if bf.elements == 0 {
		return 0.0
	}

	// Approximate false positive rate calculation
	ratio := float64(bf.elements) / float64(bf.size)
	return 1.0 - (1.0-ratio)*float64(bf.hashFuncs)
}

// hash implements a simple hash function with seed
func (bf *SimpleBloomFilter) hash(key []byte, seed int) int {
	hash := 2166136261 // FNV offset basis
	hash ^= seed

	for _, b := range key {
		hash ^= int(b)
		hash *= 16777619 // FNV prime
	}

	if hash < 0 {
		hash = -hash
	}

	return hash
}

// Utility functions

// maxInt returns the maximum of two integers
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
