// Package segment implements the segment manager for coordinating multiple segments.
package segment

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// NewConcurrentSegmentManager creates a new concurrent segment manager
func NewConcurrentSegmentManager(config *SegmentManagerConfig) (*ConcurrentSegmentManager, error) {
	if config == nil {
		config = DefaultSegmentManagerConfig()
	}

	// Create compaction policy
	policy := NewDefaultCompactionPolicy()

	// Create compactor
	compactionConfig := &CompactionConfig{
		WorkerCount:          config.CompactionThreads,
		MaxConcurrentTasks:   config.CompactionThreads * 2,
		TaskTimeout:          300,     // 5 minutes
		BufferSize:           1 << 20, // 1MB
		CompressionLevel:     6,
		ChecksumVerification: true,
		MaxMemoryUsage:       config.MaxMemoryUsage / 2, // Half for compaction
		MaxDiskUsage:         0,                         // Unlimited
		IOLimits: IOLimits{
			ReadBytesPerSec:  100 << 20, // 100MB/s
			WriteBytesPerSec: 100 << 20, // 100MB/s
			ReadOpsPerSec:    10000,
			WriteOpsPerSec:   10000,
		},
	}

	compactor, err := NewSegmentCompactor(compactionConfig, policy)
	if err != nil {
		return nil, fmt.Errorf("failed to create compactor: %w", err)
	}

	manager := &ConcurrentSegmentManager{
		segments:        make(map[string]Segment),
		activeSegment:   nil,
		config:          config,
		compactor:       compactor,
		policy:          policy,
		events:          make(chan *SegmentEvent, 1000),
		listeners:       make([]SegmentEventListener, 0),
		running:         false,
		segmentSequence: 0,
		stats: &SegmentManagerStats{
			TotalSegments:         0,
			ActiveSegments:        0,
			ImmutableSegments:     0,
			CompactedSegments:     0,
			TotalVectors:          0,
			TotalSizeBytes:        0,
			CompactionsInProgress: 0,
			TotalCompactions:      0,
			CompactionErrors:      0,
		},
		health: &SegmentHealthStatus{
			HealthScore:         1.0,
			Status:              "healthy",
			LastCheckTime:       time.Now(),
			HealthySegments:     0,
			DegradedSegments:    0,
			CorruptedSegments:   0,
			UnreachableSegments: 0,
			PerformanceScore:    1.0,
			LatencyIssues:       false,
			ThroughputIssues:    false,
			MemoryUsageRatio:    0.0,
			DiskUsageRatio:      0.0,
			IOQueueDepth:        0,
			CompactionNeeded:    false,
			RepairNeeded:        false,
			CleanupNeeded:       false,
			Issues:              make([]string, 0),
			Recommendations:     make([]string, 0),
		},
	}

	return manager, nil
}

// Start starts the segment manager and background processes
func (sm *ConcurrentSegmentManager) Start(ctx context.Context) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.running {
		return fmt.Errorf("segment manager is already running")
	}

	// Start compactor
	if err := sm.compactor.Start(ctx); err != nil {
		return fmt.Errorf("failed to start compactor: %w", err)
	}

	// Create initial active segment if none exists
	if sm.activeSegment == nil {
		if err := sm.createActiveSegment(ctx); err != nil {
			return fmt.Errorf("failed to create initial active segment: %w", err)
		}
	}

	// Initialize background tasks
	sm.stopChan = make(chan struct{})
	sm.running = true

	// Start background tasks if enabled
	if sm.config.BackgroundTasksEnabled {
		sm.startBackgroundTasks()
	}

	// Emit start event
	sm.emitEvent(&SegmentEvent{
		Type:      "manager_started",
		SegmentID: "",
		Timestamp: time.Now(),
		Details:   nil,
	})

	return nil
}

// Stop stops the segment manager and background processes
func (sm *ConcurrentSegmentManager) Stop(ctx context.Context) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if !sm.running {
		return fmt.Errorf("segment manager is not running")
	}

	sm.running = false

	// Stop background tasks
	close(sm.stopChan)
	sm.wg.Wait()

	// Stop compactor
	if err := sm.compactor.Stop(ctx); err != nil {
		return fmt.Errorf("failed to stop compactor: %w", err)
	}

	// Close all segments
	for id, segment := range sm.segments {
		if err := segment.Close(); err != nil {
			fmt.Printf("Warning: failed to close segment %s: %v\n", id, err)
		}
	}

	// Clear segments
	sm.segments = make(map[string]Segment)
	sm.activeSegment = nil

	// Emit stop event
	sm.emitEvent(&SegmentEvent{
		Type:      "manager_stopped",
		SegmentID: "",
		Timestamp: time.Now(),
		Details:   nil,
	})

	return nil
}

// IsRunning returns whether the segment manager is running
func (sm *ConcurrentSegmentManager) IsRunning() bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.running
}

// CreateSegment creates a new segment with the given configuration
func (sm *ConcurrentSegmentManager) CreateSegment(ctx context.Context, config *SegmentConfig) (Segment, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid segment config: %w", err)
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	if !sm.running {
		return nil, fmt.Errorf("segment manager is not running")
	}

	// Check if segment already exists
	if _, exists := sm.segments[config.ID]; exists {
		return nil, ErrSegmentExists
	}

	// Check segment count limit
	if sm.config.MaxTotalSegments > 0 && len(sm.segments) >= sm.config.MaxTotalSegments {
		return nil, fmt.Errorf("maximum number of segments reached: %d", sm.config.MaxTotalSegments)
	}

	// Create the segment
	segment, err := NewMemorySegment(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create segment: %w", err)
	}

	// Open the segment
	if err := segment.Open(ctx); err != nil {
		return nil, fmt.Errorf("failed to open segment: %w", err)
	}

	// Add to segments map
	sm.segments[config.ID] = segment

	// Update statistics
	atomic.AddInt64(&sm.stats.TotalSegments, 1)
	if config.Type == ActiveSegment {
		atomic.AddInt64(&sm.stats.ActiveSegments, 1)
	}

	// Emit creation event
	sm.emitEvent(&SegmentEvent{
		Type:      SegmentCreated,
		SegmentID: config.ID,
		Timestamp: time.Now(),
		Details:   config,
	})

	return segment, nil
}

// OpenSegment opens an existing segment by ID
func (sm *ConcurrentSegmentManager) OpenSegment(ctx context.Context, id string) (Segment, error) {
	sm.mu.RLock()
	segment, exists := sm.segments[id]
	sm.mu.RUnlock()

	if !exists {
		return nil, ErrSegmentNotFound
	}

	if err := segment.Open(ctx); err != nil {
		return nil, fmt.Errorf("failed to open segment: %w", err)
	}

	// Emit open event
	sm.emitEvent(&SegmentEvent{
		Type:      SegmentOpened,
		SegmentID: id,
		Timestamp: time.Now(),
		Details:   nil,
	})

	return segment, nil
}

// CloseSegment closes a segment by ID
func (sm *ConcurrentSegmentManager) CloseSegment(ctx context.Context, id string) error {
	sm.mu.RLock()
	segment, exists := sm.segments[id]
	sm.mu.RUnlock()

	if !exists {
		return ErrSegmentNotFound
	}

	if err := segment.Close(); err != nil {
		return fmt.Errorf("failed to close segment: %w", err)
	}

	// Emit close event
	sm.emitEvent(&SegmentEvent{
		Type:      SegmentClosed,
		SegmentID: id,
		Timestamp: time.Now(),
		Details:   nil,
	})

	return nil
}

// DeleteSegment deletes a segment by ID
func (sm *ConcurrentSegmentManager) DeleteSegment(ctx context.Context, id string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	segment, exists := sm.segments[id]
	if !exists {
		return ErrSegmentNotFound
	}

	// Cannot delete active segment
	if segment == sm.activeSegment {
		return fmt.Errorf("cannot delete active segment")
	}

	// Close segment first
	if err := segment.Close(); err != nil {
		return fmt.Errorf("failed to close segment before deletion: %w", err)
	}

	// Remove from segments map
	delete(sm.segments, id)

	// Update statistics
	atomic.AddInt64(&sm.stats.TotalSegments, -1)
	segmentType := segment.Type()
	if segmentType == ActiveSegment {
		atomic.AddInt64(&sm.stats.ActiveSegments, -1)
	} else if segmentType == ImmutableSegment {
		atomic.AddInt64(&sm.stats.ImmutableSegments, -1)
	} else if segmentType == CompactedSegment {
		atomic.AddInt64(&sm.stats.CompactedSegments, -1)
	}

	// Emit deletion event
	sm.emitEvent(&SegmentEvent{
		Type:      SegmentDeleted,
		SegmentID: id,
		Timestamp: time.Now(),
		Details:   nil,
	})

	return nil
}

// Clear removes all segments and creates a new empty active segment
func (sm *ConcurrentSegmentManager) Clear(ctx context.Context) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Close all existing segments
	for _, segment := range sm.segments {
		if err := segment.Close(); err != nil {
			// Log error but continue clearing
			continue
		}
	}

	// Clear segments map
	sm.segments = make(map[string]Segment)
	sm.activeSegment = nil

	// Reset statistics
	atomic.StoreInt64(&sm.stats.TotalSegments, 0)
	atomic.StoreInt64(&sm.stats.ActiveSegments, 0)
	atomic.StoreInt64(&sm.stats.ImmutableSegments, 0)
	atomic.StoreInt64(&sm.stats.CompactedSegments, 0)

	// Create new active segment
	if err := sm.createActiveSegment(ctx); err != nil {
		return fmt.Errorf("failed to create new active segment after clear: %w", err)
	}

	return nil
}

// ListSegments returns a list of all segment IDs
func (sm *ConcurrentSegmentManager) ListSegments(ctx context.Context) ([]string, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	ids := make([]string, 0, len(sm.segments))
	for id := range sm.segments {
		ids = append(ids, id)
	}

	sort.Strings(ids)
	return ids, nil
}

// GetSegment retrieves a segment by ID
func (sm *ConcurrentSegmentManager) GetSegment(ctx context.Context, id string) (Segment, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	segment, exists := sm.segments[id]
	if !exists {
		return nil, ErrSegmentNotFound
	}

	return segment, nil
}

// GetActiveSegment returns the currently active segment
func (sm *ConcurrentSegmentManager) GetActiveSegment(ctx context.Context) (Segment, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	if sm.activeSegment == nil {
		return nil, fmt.Errorf("no active segment available")
	}

	return sm.activeSegment, nil
}

// Put stores a vector in the appropriate segment
func (sm *ConcurrentSegmentManager) Put(ctx context.Context, vector *api.Vector) error {
	if vector == nil {
		return fmt.Errorf("vector cannot be nil")
	}

	// Get active segment
	activeSegment, err := sm.GetActiveSegment(ctx)
	if err != nil {
		return fmt.Errorf("no active segment available: %w", err)
	}

	// Try to put in active segment
	err = activeSegment.Put(ctx, vector)
	if err == ErrSegmentFull {
		// Rotate to new active segment and retry
		if rotateErr := sm.RotateActiveSegment(ctx); rotateErr != nil {
			return fmt.Errorf("failed to rotate active segment: %w", rotateErr)
		}

		// Get new active segment and retry
		newActiveSegment, getErr := sm.GetActiveSegment(ctx)
		if getErr != nil {
			return fmt.Errorf("failed to get new active segment: %w", getErr)
		}

		err = newActiveSegment.Put(ctx, vector)
	}

	if err != nil {
		return fmt.Errorf("failed to put vector: %w", err)
	}

	// Update statistics
	atomic.AddInt64(&sm.stats.TotalVectors, 1)

	return nil
}

// Get retrieves a vector by ID from any segment
func (sm *ConcurrentSegmentManager) Get(ctx context.Context, id string) (*api.Vector, error) {
	if id == "" {
		return nil, fmt.Errorf("vector ID cannot be empty")
	}

	sm.mu.RLock()
	segments := make([]Segment, 0, len(sm.segments))
	for _, segment := range sm.segments {
		segments = append(segments, segment)
	}
	sm.mu.RUnlock()

	// Search segments in reverse order (newest first)
	for i := len(segments) - 1; i >= 0; i-- {
		segment := segments[i]

		// Check if segment might contain the vector
		contains, err := segment.Contains(ctx, id)
		if err != nil {
			continue // Skip segments with errors
		}

		if contains {
			vector, err := segment.Get(ctx, id)
			if err == nil {
				return vector, nil
			}
		}
	}

	return nil, api.ErrVectorNotFound
}

// Delete removes a vector by ID from all segments
func (sm *ConcurrentSegmentManager) Delete(ctx context.Context, id string) error {
	if id == "" {
		return fmt.Errorf("vector ID cannot be empty")
	}

	sm.mu.RLock()
	segments := make([]Segment, 0, len(sm.segments))
	for _, segment := range sm.segments {
		segments = append(segments, segment)
	}
	sm.mu.RUnlock()

	var deleted bool
	var lastError error

	// Try to delete from all segments
	for _, segment := range segments {
		if segment.CanAcceptWrites() {
			err := segment.Delete(ctx, id)
			if err == nil {
				deleted = true
			} else if err != api.ErrVectorNotFound {
				lastError = err
			}
		}
	}

	if !deleted && lastError != nil {
		return lastError
	}

	if deleted {
		atomic.AddInt64(&sm.stats.TotalVectors, -1)
	}

	return nil
}

// PutBatch stores multiple vectors
func (sm *ConcurrentSegmentManager) PutBatch(ctx context.Context, vectors []*api.Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	// Process vectors in batches to handle segment rotation
	for _, vector := range vectors {
		if err := sm.Put(ctx, vector); err != nil {
			return fmt.Errorf("failed to put vector %s in batch: %w", vector.ID, err)
		}
	}

	return nil
}

// GetBatch retrieves multiple vectors by their IDs
func (sm *ConcurrentSegmentManager) GetBatch(ctx context.Context, ids []string) ([]*api.Vector, error) {
	if len(ids) == 0 {
		return []*api.Vector{}, nil
	}

	result := make([]*api.Vector, 0, len(ids))

	for _, id := range ids {
		vector, err := sm.Get(ctx, id)
		if err == nil {
			result = append(result, vector)
		}
		// Ignore not found errors in batch operations
	}

	return result, nil
}

// DeleteBatch removes multiple vectors by their IDs
func (sm *ConcurrentSegmentManager) DeleteBatch(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	for _, id := range ids {
		if err := sm.Delete(ctx, id); err != nil && err != api.ErrVectorNotFound {
			return fmt.Errorf("failed to delete vector %s in batch: %w", id, err)
		}
	}

	return nil
}

// Filter returns vectors that match the given filter expression
func (sm *ConcurrentSegmentManager) Filter(ctx context.Context, filter api.FilterExpr) ([]*api.Vector, error) {
	sm.mu.RLock()
	segments := make([]Segment, 0, len(sm.segments))
	for _, segment := range sm.segments {
		segments = append(segments, segment)
	}
	sm.mu.RUnlock()

	result := make([]*api.Vector, 0)

	// Filter from all segments
	for _, segment := range segments {
		segmentResults, err := segment.Filter(ctx, filter)
		if err != nil {
			continue // Skip segments with errors
		}
		result = append(result, segmentResults...)
	}

	return result, nil
}

// Scan iterates over all vectors in all segments
func (sm *ConcurrentSegmentManager) Scan(ctx context.Context, callback func(*api.Vector) bool) error {
	sm.mu.RLock()
	segments := make([]Segment, 0, len(sm.segments))
	for _, segment := range sm.segments {
		segments = append(segments, segment)
	}
	sm.mu.RUnlock()

	// Scan all segments
	for _, segment := range segments {
		err := segment.Scan(ctx, func(vector *api.Vector) bool {
			return callback(vector)
		})
		if err != nil {
			return fmt.Errorf("error scanning segment %s: %w", segment.ID(), err)
		}
	}

	return nil
}

// RotateActiveSegment creates a new active segment and freezes the current one
func (sm *ConcurrentSegmentManager) RotateActiveSegment(ctx context.Context) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Freeze current active segment
	if sm.activeSegment != nil {
		if err := sm.activeSegment.Freeze(); err != nil {
			return fmt.Errorf("failed to freeze current active segment: %w", err)
		}

		// Update statistics
		atomic.AddInt64(&sm.stats.ActiveSegments, -1)
		atomic.AddInt64(&sm.stats.ImmutableSegments, 1)

		// Emit freeze event
		sm.emitEvent(&SegmentEvent{
			Type:      SegmentFrozen,
			SegmentID: sm.activeSegment.ID(),
			Timestamp: time.Now(),
			Details:   nil,
		})
	}

	// Create new active segment
	return sm.createActiveSegment(ctx)
}

// FreezeSegment freezes a segment by ID
func (sm *ConcurrentSegmentManager) FreezeSegment(ctx context.Context, id string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	segment, exists := sm.segments[id]
	if !exists {
		return ErrSegmentNotFound
	}

	if err := segment.Freeze(); err != nil {
		return fmt.Errorf("failed to freeze segment: %w", err)
	}

	// Update active segment if this was the active one
	if segment == sm.activeSegment {
		sm.activeSegment = nil
		atomic.AddInt64(&sm.stats.ActiveSegments, -1)
		atomic.AddInt64(&sm.stats.ImmutableSegments, 1)

		// Create new active segment
		if err := sm.createActiveSegment(ctx); err != nil {
			return fmt.Errorf("failed to create new active segment: %w", err)
		}
	}

	// Emit freeze event
	sm.emitEvent(&SegmentEvent{
		Type:      SegmentFrozen,
		SegmentID: id,
		Timestamp: time.Now(),
		Details:   nil,
	})

	return nil
}

// TriggerCompaction triggers compaction of eligible segments
func (sm *ConcurrentSegmentManager) TriggerCompaction(ctx context.Context) error {
	sm.mu.RLock()
	segments := make([]Segment, 0, len(sm.segments))
	for _, segment := range sm.segments {
		if segment.State() == SegmentStateImmutable {
			segments = append(segments, segment)
		}
	}
	sm.mu.RUnlock()

	if len(segments) == 0 {
		return nil // Nothing to compact
	}

	// Use policy to select segments for compaction
	selectedSegments := sm.policy.SelectSegmentsForCompaction(segments)
	if len(selectedSegments) == 0 {
		return nil // No segments selected
	}

	// Create compaction task
	task := &CompactionTask{
		ID:            fmt.Sprintf("compact-%d", time.Now().Unix()),
		Type:          MinorCompaction,
		InputSegments: make([]string, len(selectedSegments)),
		OutputSegment: fmt.Sprintf("compacted-%d", atomic.AddInt64(&sm.segmentSequence, 1)),
		Priority:      1,
		CreatedAt:     time.Now(),
		Status:        TaskPending,
		Progress:      0.0,
	}

	for i, segment := range selectedSegments {
		task.InputSegments[i] = segment.ID()
	}

	// Submit task to compactor
	return sm.compactor.SubmitTask(ctx, task)
}

// TriggerMajorCompaction triggers major compaction of all eligible segments
func (sm *ConcurrentSegmentManager) TriggerMajorCompaction(ctx context.Context) error {
	// Major compaction involves all immutable and compacted segments
	sm.mu.RLock()
	segments := make([]Segment, 0, len(sm.segments))
	for _, segment := range sm.segments {
		state := segment.State()
		if state == SegmentStateImmutable || state == SegmentStateActive {
			segments = append(segments, segment)
		}
	}
	sm.mu.RUnlock()

	if len(segments) <= 1 {
		return nil // Not enough segments for major compaction
	}

	// Create major compaction task
	task := &CompactionTask{
		ID:            fmt.Sprintf("major-compact-%d", time.Now().Unix()),
		Type:          MajorCompaction,
		InputSegments: make([]string, len(segments)),
		OutputSegment: fmt.Sprintf("major-compacted-%d", atomic.AddInt64(&sm.segmentSequence, 1)),
		Priority:      2,
		CreatedAt:     time.Now(),
		Status:        TaskPending,
		Progress:      0.0,
	}

	for i, segment := range segments {
		task.InputSegments[i] = segment.ID()
	}

	// Submit task to compactor
	return sm.compactor.SubmitTask(ctx, task)
}

// GetCompactionStatus returns the current compaction status
func (sm *ConcurrentSegmentManager) GetCompactionStatus(ctx context.Context) (*CompactionStatus, error) {
	return sm.compactor.GetStatus(ctx), nil
}

// Stats returns segment manager statistics
func (sm *ConcurrentSegmentManager) Stats(ctx context.Context) (*SegmentManagerStats, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Update computed statistics
	sm.updateStats()

	// Return a copy
	statsCopy := *sm.stats
	return &statsCopy, nil
}

// Health returns segment health status
func (sm *ConcurrentSegmentManager) Health(ctx context.Context) (*SegmentHealthStatus, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Update health status
	sm.updateHealth()

	// Return a copy
	healthCopy := *sm.health
	return &healthCopy, nil
}

// UpdateConfig updates the segment manager configuration
func (sm *ConcurrentSegmentManager) UpdateConfig(ctx context.Context, config *SegmentManagerConfig) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Validate new configuration
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	// Update configuration
	sm.config = config

	return nil
}

// GetConfig returns the current configuration
func (sm *ConcurrentSegmentManager) GetConfig(ctx context.Context) *SegmentManagerConfig {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Return a copy
	configCopy := *sm.config
	return &configCopy
}

// Private helper methods

// createActiveSegment creates a new active segment (must be called with write lock held)
func (sm *ConcurrentSegmentManager) createActiveSegment(ctx context.Context) error {
	// Generate unique segment ID
	segmentID := fmt.Sprintf("active-%d", atomic.AddInt64(&sm.segmentSequence, 1))

	// Create segment configuration
	segmentConfig := *sm.config.DefaultSegmentConfig
	segmentConfig.ID = segmentID
	segmentConfig.Type = ActiveSegment

	// Create the segment
	segment, err := NewMemorySegment(&segmentConfig)
	if err != nil {
		return fmt.Errorf("failed to create active segment: %w", err)
	}

	// Open the segment
	if err := segment.Open(ctx); err != nil {
		return fmt.Errorf("failed to open active segment: %w", err)
	}

	// Set as active segment
	sm.activeSegment = segment
	sm.segments[segmentID] = segment

	// Update statistics
	atomic.AddInt64(&sm.stats.TotalSegments, 1)
	atomic.AddInt64(&sm.stats.ActiveSegments, 1)

	// Emit creation event
	sm.emitEvent(&SegmentEvent{
		Type:      SegmentCreated,
		SegmentID: segmentID,
		Timestamp: time.Now(),
		Details:   &segmentConfig,
	})

	return nil
}

// startBackgroundTasks starts background maintenance tasks
func (sm *ConcurrentSegmentManager) startBackgroundTasks() {
	// Start compaction task
	if sm.config.AutoCompactionEnabled {
		sm.compactionTicker = time.NewTicker(time.Duration(sm.config.StatisticsInterval) * time.Second)
		sm.wg.Add(1)
		go sm.compactionTask()
	}

	// Start health check task
	if sm.config.HealthCheckInterval > 0 {
		sm.healthTicker = time.NewTicker(time.Duration(sm.config.HealthCheckInterval) * time.Second)
		sm.wg.Add(1)
		go sm.healthCheckTask()
	}

	// Start statistics update task
	if sm.config.StatsUpdateInterval > 0 {
		sm.statsTicker = time.NewTicker(time.Duration(sm.config.StatsUpdateInterval) * time.Second)
		sm.wg.Add(1)
		go sm.statsUpdateTask()
	}

	// Start event processing task
	sm.wg.Add(1)
	go sm.eventProcessingTask()
}

// compactionTask runs periodic compaction checks
func (sm *ConcurrentSegmentManager) compactionTask() {
	defer sm.wg.Done()
	defer sm.compactionTicker.Stop()

	for {
		select {
		case <-sm.compactionTicker.C:
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

			// Check if compaction is needed
			sm.mu.RLock()
			segments := make([]Segment, 0, len(sm.segments))
			for _, segment := range sm.segments {
				segments = append(segments, segment)
			}
			sm.mu.RUnlock()

			if sm.policy.ShouldCompact(segments) {
				if err := sm.TriggerCompaction(ctx); err != nil {
					fmt.Printf("Warning: compaction failed: %v\n", err)
				}
			}

			cancel()

		case <-sm.stopChan:
			return
		}
	}
}

// healthCheckTask runs periodic health checks
func (sm *ConcurrentSegmentManager) healthCheckTask() {
	defer sm.wg.Done()
	defer sm.healthTicker.Stop()

	for {
		select {
		case <-sm.healthTicker.C:
			sm.mu.Lock()
			sm.updateHealth()
			sm.mu.Unlock()

		case <-sm.stopChan:
			return
		}
	}
}

// statsUpdateTask runs periodic statistics updates
func (sm *ConcurrentSegmentManager) statsUpdateTask() {
	defer sm.wg.Done()
	defer sm.statsTicker.Stop()

	for {
		select {
		case <-sm.statsTicker.C:
			sm.mu.Lock()
			sm.updateStats()
			sm.mu.Unlock()

		case <-sm.stopChan:
			return
		}
	}
}

// eventProcessingTask processes segment events
func (sm *ConcurrentSegmentManager) eventProcessingTask() {
	defer sm.wg.Done()

	for {
		select {
		case event := <-sm.events:
			// Notify all listeners
			for _, listener := range sm.listeners {
				if err := listener.OnSegmentEvent(event); err != nil {
					fmt.Printf("Warning: event listener error: %v\n", err)
				}
			}

		case <-sm.stopChan:
			return
		}
	}
}

// updateStats updates computed statistics (must be called with lock held)
func (sm *ConcurrentSegmentManager) updateStats() {
	var totalVectors int64
	var totalSize int64
	var activeCount int64
	var immutableCount int64
	var compactedCount int64

	for _, segment := range sm.segments {
		totalVectors += segment.VectorCount()
		totalSize += segment.Size()

		switch segment.State() {
		case SegmentStateActive:
			activeCount++
		case SegmentStateImmutable:
			immutableCount++
		case SegmentStateCompacting:
			// Count as immutable for now
			immutableCount++
		}

		if segment.Type() == CompactedSegment {
			compactedCount++
		}
	}

	sm.stats.TotalVectors = totalVectors
	sm.stats.TotalSizeBytes = totalSize
	sm.stats.ActiveSegments = activeCount
	sm.stats.ImmutableSegments = immutableCount
	sm.stats.CompactedSegments = compactedCount

	if len(sm.segments) > 0 {
		sm.stats.AverageSegmentSize = totalSize / int64(len(sm.segments))
	}
}

// updateHealth updates health status (must be called with lock held)
func (sm *ConcurrentSegmentManager) updateHealth() {
	var healthyCount int64
	var degradedCount int64
	var corruptedCount int64
	var unreachableCount int64

	issues := make([]string, 0)
	recommendations := make([]string, 0)

	for _, segment := range sm.segments {
		// Simple health check - could be enhanced
		if segment.IsClosed() {
			unreachableCount++
		} else if segment.State() == SegmentStateCorrupted {
			corruptedCount++
		} else {
			// Validate segment
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			if err := segment.Validate(ctx); err != nil {
				degradedCount++
				issues = append(issues, fmt.Sprintf("Segment %s validation failed: %v", segment.ID(), err))
			} else {
				healthyCount++
			}
			cancel()
		}
	}

	totalSegments := int64(len(sm.segments))

	// Calculate health score
	var healthScore float64 = 1.0
	if totalSegments > 0 {
		healthScore = float64(healthyCount) / float64(totalSegments)
	}

	// Determine status
	status := "healthy"
	if healthScore < 0.8 {
		status = "degraded"
	}
	if healthScore < 0.5 {
		status = "critical"
	}

	// Check for compaction needs
	compactionNeeded := false
	if len(sm.segments) > sm.config.MergeThreshold {
		compactionNeeded = true
		recommendations = append(recommendations, "Consider running compaction to merge segments")
	}

	// Update health status
	sm.health.HealthScore = healthScore
	sm.health.Status = status
	sm.health.LastCheckTime = time.Now()
	sm.health.HealthySegments = healthyCount
	sm.health.DegradedSegments = degradedCount
	sm.health.CorruptedSegments = corruptedCount
	sm.health.UnreachableSegments = unreachableCount
	sm.health.CompactionNeeded = compactionNeeded
	sm.health.Issues = issues
	sm.health.Recommendations = recommendations
}

// emitEvent emits a segment event
func (sm *ConcurrentSegmentManager) emitEvent(event *SegmentEvent) {
	select {
	case sm.events <- event:
		// Event queued successfully
	default:
		// Event queue is full, drop the event
		fmt.Printf("Warning: segment event queue is full, dropping event: %+v\n", event)
	}
}

// DefaultCompactionPolicy implements a default compaction policy
type DefaultCompactionPolicy struct {
	thresholds CompactionThresholds
	mu         sync.RWMutex
}

// NewDefaultCompactionPolicy creates a new default compaction policy
func NewDefaultCompactionPolicy() CompactionPolicy {
	return &DefaultCompactionPolicy{
		thresholds: CompactionThresholds{
			MinSegmentSize:        1 << 20, // 1MB
			MaxSegmentSize:        1 << 30, // 1GB
			FragmentationRatio:    0.3,     // 30% fragmentation
			MinSegmentsToCompact:  2,
			MaxSegmentsToCompact:  10,
			ReadAmplificationMax:  5.0,
			WriteAmplificationMax: 5.0,
			MaxSegmentAge:         7 * 24 * 3600, // 7 days
			CompactionCooldown:    3600,          // 1 hour
		},
	}
}

// ShouldCompact determines if compaction should be triggered
func (dcp *DefaultCompactionPolicy) ShouldCompact(segments []Segment) bool {
	dcp.mu.RLock()
	defer dcp.mu.RUnlock()

	immutableSegments := make([]Segment, 0)
	for _, segment := range segments {
		if segment.State() == SegmentStateImmutable {
			immutableSegments = append(immutableSegments, segment)
		}
	}

	// Check if we have enough segments to compact
	if len(immutableSegments) < dcp.thresholds.MinSegmentsToCompact {
		return false
	}

	// Check fragmentation
	for _, segment := range immutableSegments {
		metadata := segment.Metadata()
		if metadata.FragmentationRatio > dcp.thresholds.FragmentationRatio {
			return true
		}
	}

	// Check segment count threshold
	return len(immutableSegments) >= dcp.thresholds.MaxSegmentsToCompact
}

// SelectSegmentsForCompaction selects segments that should be compacted
func (dcp *DefaultCompactionPolicy) SelectSegmentsForCompaction(segments []Segment) []Segment {
	dcp.mu.RLock()
	defer dcp.mu.RUnlock()

	// Filter immutable segments
	candidates := make([]Segment, 0)
	for _, segment := range segments {
		if segment.State() == SegmentStateImmutable {
			candidates = append(candidates, segment)
		}
	}

	if len(candidates) < dcp.thresholds.MinSegmentsToCompact {
		return []Segment{}
	}

	// Sort by size (smaller segments first for better compaction)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Size() < candidates[j].Size()
	})

	// Select up to MaxSegmentsToCompact
	maxToCompact := dcp.thresholds.MaxSegmentsToCompact
	if len(candidates) < maxToCompact {
		maxToCompact = len(candidates)
	}

	return candidates[:maxToCompact]
}

// EstimateCompactionBenefit estimates the benefit of compacting the given segments
func (dcp *DefaultCompactionPolicy) EstimateCompactionBenefit(segments []Segment) CompactionBenefit {
	if len(segments) == 0 {
		return CompactionBenefit{}
	}

	var totalSize int64
	var totalVectors int64
	var totalDeleted int64

	for _, segment := range segments {
		totalSize += segment.Size()
		stats := segment.Stats()
		totalVectors += stats.VectorCount
		totalDeleted += stats.DeletedVectors
	}

	// Estimate space savings from removing deleted vectors
	deletionRatio := float64(totalDeleted) / float64(totalVectors)
	spaceSavings := int64(float64(totalSize) * deletionRatio)

	// Estimate I/O reduction from having fewer segments
	ioReduction := float64(len(segments)-1) / float64(len(segments))

	return CompactionBenefit{
		SpaceSavings:    spaceSavings,
		IOReduction:     ioReduction,
		PerformanceGain: ioReduction * 0.5,                // Rough estimate
		MaintenanceCost: float64(totalSize) / (100 << 20), // Cost per 100MB
		NetBenefit:      (ioReduction * 0.5) - (float64(totalSize) / (100 << 20)),
	}
}

// GetThresholds returns current compaction thresholds
func (dcp *DefaultCompactionPolicy) GetThresholds() CompactionThresholds {
	dcp.mu.RLock()
	defer dcp.mu.RUnlock()
	return dcp.thresholds
}

// SetThresholds updates compaction thresholds
func (dcp *DefaultCompactionPolicy) SetThresholds(thresholds CompactionThresholds) {
	dcp.mu.Lock()
	defer dcp.mu.Unlock()
	dcp.thresholds = thresholds
}
