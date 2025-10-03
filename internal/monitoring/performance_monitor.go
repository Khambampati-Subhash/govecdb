package monitoring

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// PerformanceMonitor tracks system and application metrics
type PerformanceMonitor struct {
	mu sync.RWMutex

	// System metrics
	memoryUsage    int64
	goroutineCount int64
	gcCount        uint32
	gcPauseTime    int64

	// Database metrics
	queryLatency     *LatencyTracker
	insertLatency    *LatencyTracker
	searchThroughput *ThroughputTracker
	insertThroughput *ThroughputTracker

	// Index metrics
	indexSize     int64
	nodeCount     int64
	edgeCount     int64
	averageDegree float64

	// Error metrics
	errorCount    int64
	errorRate     float64
	lastErrorTime time.Time

	// Configuration
	sampleInterval time.Duration
	retentionTime  time.Duration

	// Background monitoring
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// LatencyTracker tracks latency percentiles and statistics
type LatencyTracker struct {
	mu       sync.RWMutex
	samples  []time.Duration
	count    int64
	sum      int64
	min      int64
	max      int64
	capacity int
}

// ThroughputTracker tracks operations per second
type ThroughputTracker struct {
	mu         sync.RWMutex
	timestamps []time.Time
	operations int64
	capacity   int
}

// NewPerformanceMonitor creates a new performance monitor
func NewPerformanceMonitor(sampleInterval, retentionTime time.Duration) *PerformanceMonitor {
	ctx, cancel := context.WithCancel(context.Background())

	pm := &PerformanceMonitor{
		queryLatency:     NewLatencyTracker(10000),
		insertLatency:    NewLatencyTracker(10000),
		searchThroughput: NewThroughputTracker(1000),
		insertThroughput: NewThroughputTracker(1000),
		sampleInterval:   sampleInterval,
		retentionTime:    retentionTime,
		ctx:              ctx,
		cancel:           cancel,
	}

	pm.start()
	return pm
}

// NewLatencyTracker creates a new latency tracker
func NewLatencyTracker(capacity int) *LatencyTracker {
	return &LatencyTracker{
		samples:  make([]time.Duration, 0, capacity),
		capacity: capacity,
		min:      int64(time.Hour), // Initialize to high value
	}
}

// NewThroughputTracker creates a new throughput tracker
func NewThroughputTracker(capacity int) *ThroughputTracker {
	return &ThroughputTracker{
		timestamps: make([]time.Time, 0, capacity),
		capacity:   capacity,
	}
}

// RecordLatency records a latency measurement
func (lt *LatencyTracker) RecordLatency(duration time.Duration) {
	lt.mu.Lock()
	defer lt.mu.Unlock()

	nanos := duration.Nanoseconds()
	atomic.AddInt64(&lt.count, 1)
	atomic.AddInt64(&lt.sum, nanos)

	// Update min/max atomically
	for {
		current := atomic.LoadInt64(&lt.min)
		if nanos >= current || atomic.CompareAndSwapInt64(&lt.min, current, nanos) {
			break
		}
	}

	for {
		current := atomic.LoadInt64(&lt.max)
		if nanos <= current || atomic.CompareAndSwapInt64(&lt.max, current, nanos) {
			break
		}
	}

	// Store sample for percentile calculation
	if len(lt.samples) >= lt.capacity {
		// Remove oldest sample (circular buffer)
		copy(lt.samples, lt.samples[1:])
		lt.samples = lt.samples[:len(lt.samples)-1]
	}
	lt.samples = append(lt.samples, duration)
}

// GetStatistics returns latency statistics
func (lt *LatencyTracker) GetStatistics() LatencyStats {
	lt.mu.RLock()
	defer lt.mu.RUnlock()

	count := atomic.LoadInt64(&lt.count)
	if count == 0 {
		return LatencyStats{}
	}

	sum := atomic.LoadInt64(&lt.sum)
	min := atomic.LoadInt64(&lt.min)
	max := atomic.LoadInt64(&lt.max)

	stats := LatencyStats{
		Count:   count,
		Min:     time.Duration(min),
		Max:     time.Duration(max),
		Average: time.Duration(sum / count),
	}

	// Calculate percentiles from samples
	if len(lt.samples) > 0 {
		samples := make([]time.Duration, len(lt.samples))
		copy(samples, lt.samples)
		stats.Percentiles = calculatePercentiles(samples)
	}

	return stats
}

// RecordOperation records a throughput measurement
func (tt *ThroughputTracker) RecordOperation() {
	tt.mu.Lock()
	defer tt.mu.Unlock()

	now := time.Now()
	atomic.AddInt64(&tt.operations, 1)

	// Remove old timestamps
	cutoff := now.Add(-time.Minute)
	for len(tt.timestamps) > 0 && tt.timestamps[0].Before(cutoff) {
		tt.timestamps = tt.timestamps[1:]
	}

	// Add new timestamp
	if len(tt.timestamps) >= tt.capacity {
		copy(tt.timestamps, tt.timestamps[1:])
		tt.timestamps = tt.timestamps[:len(tt.timestamps)-1]
	}
	tt.timestamps = append(tt.timestamps, now)
}

// GetThroughput returns operations per second
func (tt *ThroughputTracker) GetThroughput() float64 {
	tt.mu.RLock()
	defer tt.mu.RUnlock()

	now := time.Now()
	cutoff := now.Add(-time.Minute)

	count := 0
	for _, ts := range tt.timestamps {
		if ts.After(cutoff) {
			count++
		}
	}

	return float64(count) / 60.0 // Operations per second
}

// start begins background monitoring
func (pm *PerformanceMonitor) start() {
	pm.wg.Add(1)
	go pm.monitorLoop()
}

// monitorLoop runs the monitoring loop
func (pm *PerformanceMonitor) monitorLoop() {
	defer pm.wg.Done()

	ticker := time.NewTicker(pm.sampleInterval)
	defer ticker.Stop()

	for {
		select {
		case <-pm.ctx.Done():
			return
		case <-ticker.C:
			pm.collectSystemMetrics()
		}
	}
}

// collectSystemMetrics collects system-level metrics
func (pm *PerformanceMonitor) collectSystemMetrics() {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	atomic.StoreInt64(&pm.memoryUsage, int64(ms.Alloc))
	atomic.StoreInt64(&pm.goroutineCount, int64(runtime.NumGoroutine()))
	atomic.StoreUint32(&pm.gcCount, ms.NumGC)
	atomic.StoreInt64(&pm.gcPauseTime, int64(ms.PauseTotalNs))
}

// RecordQuery records a query operation
func (pm *PerformanceMonitor) RecordQuery(duration time.Duration) {
	pm.queryLatency.RecordLatency(duration)
	pm.searchThroughput.RecordOperation()
}

// RecordInsert records an insert operation
func (pm *PerformanceMonitor) RecordInsert(duration time.Duration) {
	pm.insertLatency.RecordLatency(duration)
	pm.insertThroughput.RecordOperation()
}

// RecordError records an error occurrence
func (pm *PerformanceMonitor) RecordError() {
	atomic.AddInt64(&pm.errorCount, 1)
	pm.mu.Lock()
	pm.lastErrorTime = time.Now()
	pm.mu.Unlock()
}

// UpdateIndexMetrics updates index-related metrics
func (pm *PerformanceMonitor) UpdateIndexMetrics(size, nodeCount, edgeCount int64, avgDegree float64) {
	atomic.StoreInt64(&pm.indexSize, size)
	atomic.StoreInt64(&pm.nodeCount, nodeCount)
	atomic.StoreInt64(&pm.edgeCount, edgeCount)
	pm.mu.Lock()
	pm.averageDegree = avgDegree
	pm.mu.Unlock()
}

// GetMetrics returns current performance metrics
func (pm *PerformanceMonitor) GetMetrics() PerformanceMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	return PerformanceMetrics{
		Timestamp: time.Now(),

		// System metrics
		MemoryUsage:    atomic.LoadInt64(&pm.memoryUsage),
		GoroutineCount: atomic.LoadInt64(&pm.goroutineCount),
		GCCount:        atomic.LoadUint32(&pm.gcCount),
		GCPauseTime:    time.Duration(atomic.LoadInt64(&pm.gcPauseTime)),

		// Query metrics
		QueryLatency:     pm.queryLatency.GetStatistics(),
		InsertLatency:    pm.insertLatency.GetStatistics(),
		SearchThroughput: pm.searchThroughput.GetThroughput(),
		InsertThroughput: pm.insertThroughput.GetThroughput(),

		// Index metrics
		IndexSize:     atomic.LoadInt64(&pm.indexSize),
		NodeCount:     atomic.LoadInt64(&pm.nodeCount),
		EdgeCount:     atomic.LoadInt64(&pm.edgeCount),
		AverageDegree: pm.averageDegree,

		// Error metrics
		ErrorCount:    atomic.LoadInt64(&pm.errorCount),
		ErrorRate:     pm.calculateErrorRate(),
		LastErrorTime: pm.lastErrorTime,
	}
}

// calculateErrorRate calculates the error rate over the last minute
func (pm *PerformanceMonitor) calculateErrorRate() float64 {
	// Simplified error rate calculation
	totalOps := pm.searchThroughput.GetThroughput() + pm.insertThroughput.GetThroughput()
	if totalOps == 0 {
		return 0
	}

	errorCount := atomic.LoadInt64(&pm.errorCount)
	return float64(errorCount) / (totalOps * 60) // Errors per operation
}

// Stop stops the performance monitor
func (pm *PerformanceMonitor) Stop() {
	pm.cancel()
	pm.wg.Wait()
}

// LatencyStats represents latency statistics
type LatencyStats struct {
	Count       int64
	Min         time.Duration
	Max         time.Duration
	Average     time.Duration
	Percentiles map[string]time.Duration
}

// PerformanceMetrics represents all performance metrics
type PerformanceMetrics struct {
	Timestamp time.Time

	// System metrics
	MemoryUsage    int64
	GoroutineCount int64
	GCCount        uint32
	GCPauseTime    time.Duration

	// Query metrics
	QueryLatency     LatencyStats
	InsertLatency    LatencyStats
	SearchThroughput float64
	InsertThroughput float64

	// Index metrics
	IndexSize     int64
	NodeCount     int64
	EdgeCount     int64
	AverageDegree float64

	// Error metrics
	ErrorCount    int64
	ErrorRate     float64
	LastErrorTime time.Time
}

// calculatePercentiles calculates percentile values from samples
func calculatePercentiles(samples []time.Duration) map[string]time.Duration {
	if len(samples) == 0 {
		return nil
	}

	// Simple sorting for percentile calculation
	// In production, use a more efficient algorithm
	sorted := make([]time.Duration, len(samples))
	copy(sorted, samples)

	// Bubble sort (replace with quicksort for better performance)
	for i := 0; i < len(sorted); i++ {
		for j := 0; j < len(sorted)-1-i; j++ {
			if sorted[j] > sorted[j+1] {
				sorted[j], sorted[j+1] = sorted[j+1], sorted[j]
			}
		}
	}

	percentiles := map[string]time.Duration{
		"p50":  sorted[len(sorted)*50/100],
		"p90":  sorted[len(sorted)*90/100],
		"p95":  sorted[len(sorted)*95/100],
		"p99":  sorted[len(sorted)*99/100],
		"p999": sorted[len(sorted)*999/1000],
	}

	return percentiles
}
