// Package segment implements segment compaction functionality.
package segment

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"
)

// NewSegmentCompactor creates a new segment compactor
func NewSegmentCompactor(config *CompactionConfig, policy CompactionPolicy) (*SegmentCompactor, error) {
	if config == nil {
		return nil, fmt.Errorf("compaction config cannot be nil")
	}
	if policy == nil {
		return nil, fmt.Errorf("compaction policy cannot be nil")
	}

	compactor := &SegmentCompactor{
		config:     config,
		policy:     policy,
		inProgress: make(map[string]*CompactionTask),
		queue:      make(chan *CompactionTask, config.MaxConcurrentTasks*2),
		workers:    make([]*CompactionWorker, config.WorkerCount),
		stats: &CompactionStats{
			TotalTasks:         0,
			CompletedTasks:     0,
			FailedTasks:        0,
			PendingTasks:       0,
			RunningTasks:       0,
			AverageTaskTime:    0.0,
			TotalDataProcessed: 0,
			TotalSpaceSaved:    0,
			MemoryUsage:        0,
			DiskUsage:          0,
			CPUUsage:           0.0,
		},
		running: false,
	}

	// Create workers
	for i := 0; i < config.WorkerCount; i++ {
		compactor.workers[i] = &CompactionWorker{
			ID:        i,
			compactor: compactor,
			stopChan:  make(chan struct{}),
		}
	}

	return compactor, nil
}

// Start starts the compactor and its workers
func (sc *SegmentCompactor) Start(ctx context.Context) error {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	if sc.running {
		return fmt.Errorf("compactor is already running")
	}

	sc.running = true

	// Start workers
	for _, worker := range sc.workers {
		sc.wg.Add(1)
		go worker.run()
	}

	return nil
}

// Stop stops the compactor and its workers
func (sc *SegmentCompactor) Stop(ctx context.Context) error {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	if !sc.running {
		return fmt.Errorf("compactor is not running")
	}

	sc.running = false

	// Stop workers
	for _, worker := range sc.workers {
		close(worker.stopChan)
	}

	// Wait for workers to finish
	sc.wg.Wait()

	// Close queue
	close(sc.queue)

	return nil
}

// SubmitTask submits a compaction task
func (sc *SegmentCompactor) SubmitTask(ctx context.Context, task *CompactionTask) error {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	if !sc.running {
		return fmt.Errorf("compactor is not running")
	}

	// Check if task already exists
	if _, exists := sc.inProgress[task.ID]; exists {
		return fmt.Errorf("task %s already exists", task.ID)
	}

	// Add to in-progress map
	sc.inProgress[task.ID] = task
	atomic.AddInt64(&sc.stats.PendingTasks, 1)

	// Submit to queue
	select {
	case sc.queue <- task:
		return nil
	case <-ctx.Done():
		// Remove from in-progress if submission failed
		delete(sc.inProgress, task.ID)
		atomic.AddInt64(&sc.stats.PendingTasks, -1)
		return ctx.Err()
	}
}

// GetStatus returns the current compaction status
func (sc *SegmentCompactor) GetStatus(ctx context.Context) *CompactionStatus {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	status := &CompactionStatus{
		InProgress:          len(sc.inProgress) > 0,
		StartTime:           time.Time{},
		EstimatedCompletion: time.Time{},
		Progress:            0.0,
		CurrentSegment:      "",
		CurrentOperation:    "",
		SegmentsProcessed:   0,
		SegmentsTotal:       0,
		RecordsProcessed:    0,
		RecordsTotal:        0,
		BytesProcessed:      0,
		BytesTotal:          0,
		ProcessingRate:      0.0,
		InputSegments:       make([]string, 0),
		OutputSegments:      make([]string, 0),
		SpaceSaved:          0,
		CompressionRatio:    1.0,
		Errors:              make([]string, 0),
		Warnings:            make([]string, 0),
	}

	// Find the earliest started task
	var earliestTask *CompactionTask
	for _, task := range sc.inProgress {
		if task.Status == TaskRunning {
			if earliestTask == nil || task.StartedAt.Before(earliestTask.StartedAt) {
				earliestTask = task
			}
		}
	}

	if earliestTask != nil {
		status.StartTime = earliestTask.StartedAt
		status.Progress = earliestTask.Progress
		status.CurrentOperation = fmt.Sprintf("Compacting task %s", earliestTask.ID)
		status.InputSegments = earliestTask.InputSegments
		status.OutputSegments = []string{earliestTask.OutputSegment}

		// Estimate completion time based on progress
		if earliestTask.Progress > 0 {
			elapsed := time.Since(earliestTask.StartedAt)
			totalEstimated := time.Duration(float64(elapsed) / earliestTask.Progress)
			status.EstimatedCompletion = earliestTask.StartedAt.Add(totalEstimated)
		}
	}

	return status
}

// GetStats returns compaction statistics
func (sc *SegmentCompactor) GetStats() *CompactionStats {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	// Return a copy
	stats := *sc.stats
	return &stats
}

// CompactionWorker runs compaction tasks
func (cw *CompactionWorker) run() {
	defer cw.compactor.wg.Done()

	for {
		select {
		case task := <-cw.compactor.queue:
			if task == nil {
				return // Channel closed
			}
			cw.processTask(task)

		case <-cw.stopChan:
			return
		}
	}
}

// processTask processes a single compaction task
func (cw *CompactionWorker) processTask(task *CompactionTask) {
	cw.mu.Lock()
	cw.currentTask = task
	cw.mu.Unlock()

	defer func() {
		cw.mu.Lock()
		cw.currentTask = nil
		cw.mu.Unlock()
	}()

	// Update task status
	task.Status = TaskRunning
	task.StartedAt = time.Now()

	// Update statistics
	atomic.AddInt64(&cw.compactor.stats.PendingTasks, -1)
	atomic.AddInt64(&cw.compactor.stats.RunningTasks, 1)

	// Execute the compaction
	err := cw.executeCompaction(task)

	// Update task completion
	task.CompletedAt = time.Now()
	duration := task.CompletedAt.Sub(task.StartedAt)

	// Update statistics
	atomic.AddInt64(&cw.compactor.stats.RunningTasks, -1)
	atomic.AddInt64(&cw.compactor.stats.TotalTasks, 1)

	if err != nil {
		task.Status = TaskFailed
		task.Error = err.Error()
		atomic.AddInt64(&cw.compactor.stats.FailedTasks, 1)
	} else {
		task.Status = TaskCompleted
		task.Progress = 1.0
		atomic.AddInt64(&cw.compactor.stats.CompletedTasks, 1)
	}

	// Try to acquire lock with timeout to avoid deadlock during shutdown
	lockAcquired := make(chan struct{})
	go func() {
		cw.compactor.mu.Lock()
		close(lockAcquired)
	}()

	select {
	case <-lockAcquired:
		// Successfully acquired lock
		delete(cw.compactor.inProgress, task.ID)

		// Update average task time while holding the lock
		if cw.compactor.running {
			currentAvg := cw.compactor.stats.AverageTaskTime
			totalTasks := cw.compactor.stats.TotalTasks

			if totalTasks == 1 {
				cw.compactor.stats.AverageTaskTime = duration.Seconds()
			} else {
				// Running average calculation
				newAvg := (currentAvg*float64(totalTasks-1) + duration.Seconds()) / float64(totalTasks)
				cw.compactor.stats.AverageTaskTime = newAvg
			}
		}

		cw.compactor.mu.Unlock()

	case <-time.After(100 * time.Millisecond):
		// Timeout - skip cleanup during shutdown to avoid deadlock
		return
	}
}

// executeCompaction performs the actual compaction work
func (cw *CompactionWorker) executeCompaction(task *CompactionTask) error {
	// This is a simplified implementation
	// In a real implementation, this would:
	// 1. Open input segments
	// 2. Create output segment
	// 3. Merge data while removing duplicates and tombstones
	// 4. Update indexes and bloom filters
	// 5. Verify integrity
	// 6. Atomically replace input segments with output

	steps := []string{
		"Opening input segments",
		"Creating output segment",
		"Merging data",
		"Building indexes",
		"Verifying integrity",
		"Finalizing compaction",
	}

	for i, step := range steps {
		// Simulate work
		time.Sleep(100 * time.Millisecond)

		// Update progress
		task.Progress = float64(i+1) / float64(len(steps))

		// Check for cancellation
		select {
		case <-cw.stopChan:
			return fmt.Errorf("compaction cancelled")
		default:
		}

		// Simulate potential failure
		if i == 2 && len(task.InputSegments) > 5 {
			// Simulate failure for very large compactions
			if time.Now().UnixNano()%10 == 0 {
				return fmt.Errorf("compaction failed at step: %s", step)
			}
		}
	}

	return nil
}

// GetCurrentTask returns the current task being processed by this worker
func (cw *CompactionWorker) GetCurrentTask() *CompactionTask {
	cw.mu.RLock()
	defer cw.mu.RUnlock()
	return cw.currentTask
}

// IsIdle returns whether the worker is currently idle
func (cw *CompactionWorker) IsIdle() bool {
	cw.mu.RLock()
	defer cw.mu.RUnlock()
	return cw.currentTask == nil
}
