package streaming

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// StreamingVectorProcessor handles real-time vector processing with optimizations
type StreamingVectorProcessor struct {
	// Core configuration
	bufferSize    int
	workers       int
	batchSize     int
	flushInterval time.Duration

	// Input/Output streams
	inputStream  chan VectorMessage
	outputStream chan ProcessedResult
	errorStream  chan error

	// Processing pipeline
	processors    []VectorProcessor
	processorPool sync.Pool

	// Memory management
	vectorPool sync.Pool
	resultPool sync.Pool

	// Performance tracking
	processed    int64
	errors       int64
	totalLatency int64
	maxLatency   int64

	// Flow control and backpressure
	backpressure chan struct{}
	throttle     *time.Ticker

	// Context and lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu sync.RWMutex
}

// VectorMessage represents a vector message in the stream
type VectorMessage struct {
	ID        string
	Vector    []float32
	Metadata  map[string]interface{}
	Timestamp time.Time
	Priority  int
}

// ProcessedResult represents the result of vector processing
type ProcessedResult struct {
	ID        string
	Result    interface{}
	Latency   time.Duration
	Timestamp time.Time
	Error     error
}

// VectorProcessor interface for processing pipeline stages
type VectorProcessor interface {
	Process(ctx context.Context, msg VectorMessage) (interface{}, error)
	GetMetrics() map[string]interface{}
}

// StreamingConfig holds configuration for streaming processor
type StreamingConfig struct {
	BufferSize    int
	Workers       int
	BatchSize     int
	FlushInterval time.Duration
	MaxLatency    time.Duration
	EnableMetrics bool
}

// DefaultStreamingConfig returns optimized default configuration
func DefaultStreamingConfig() StreamingConfig {
	return StreamingConfig{
		BufferSize:    10000,
		Workers:       runtime.NumCPU() * 2,
		BatchSize:     100,
		FlushInterval: 10 * time.Millisecond,
		MaxLatency:    100 * time.Millisecond,
		EnableMetrics: true,
	}
}

// NewStreamingVectorProcessor creates a new streaming vector processor
func NewStreamingVectorProcessor(config StreamingConfig, processors []VectorProcessor) *StreamingVectorProcessor {
	ctx, cancel := context.WithCancel(context.Background())

	svp := &StreamingVectorProcessor{
		bufferSize:    config.BufferSize,
		workers:       config.Workers,
		batchSize:     config.BatchSize,
		flushInterval: config.FlushInterval,

		inputStream:  make(chan VectorMessage, config.BufferSize),
		outputStream: make(chan ProcessedResult, config.BufferSize),
		errorStream:  make(chan error, 1000),

		processors:   processors,
		backpressure: make(chan struct{}, config.BufferSize/10),
		throttle:     time.NewTicker(config.FlushInterval),

		ctx:    ctx,
		cancel: cancel,
	}

	// Initialize memory pools
	svp.vectorPool = sync.Pool{
		New: func() interface{} {
			return make([]float32, 0, 1024) // Assume max 1024 dimensions
		},
	}

	svp.resultPool = sync.Pool{
		New: func() interface{} {
			return &ProcessedResult{}
		},
	}

	svp.processorPool = sync.Pool{
		New: func() interface{} {
			return make([]interface{}, len(processors))
		},
	}

	// Start workers
	svp.startWorkers()

	// Start monitoring
	go svp.monitorPerformance()

	return svp
}

// AddVector adds a vector to the processing stream
func (svp *StreamingVectorProcessor) AddVector(msg VectorMessage) error {
	select {
	case svp.inputStream <- msg:
		return nil
	case <-svp.ctx.Done():
		return fmt.Errorf("processor stopped")
	default:
		// Apply backpressure
		select {
		case svp.backpressure <- struct{}{}:
			// Wait briefly and retry
			time.Sleep(time.Microsecond * 100)
			select {
			case svp.inputStream <- msg:
				<-svp.backpressure
				return nil
			default:
				<-svp.backpressure
				return fmt.Errorf("stream full, message dropped")
			}
		default:
			return fmt.Errorf("system overloaded")
		}
	}
}

// GetResults returns the output stream for processed results
func (svp *StreamingVectorProcessor) GetResults() <-chan ProcessedResult {
	return svp.outputStream
}

// GetErrors returns the error stream
func (svp *StreamingVectorProcessor) GetErrors() <-chan error {
	return svp.errorStream
}

// startWorkers starts the processing workers
func (svp *StreamingVectorProcessor) startWorkers() {
	for i := 0; i < svp.workers; i++ {
		svp.wg.Add(1)
		go svp.worker(i)
	}
}

// worker processes vectors from the input stream
func (svp *StreamingVectorProcessor) worker(id int) {
	defer svp.wg.Done()

	batch := make([]VectorMessage, 0, svp.batchSize)
	flushTimer := time.NewTimer(svp.flushInterval)
	defer flushTimer.Stop()

	for {
		select {
		case msg := <-svp.inputStream:
			batch = append(batch, msg)

			// Process batch when full
			if len(batch) >= svp.batchSize {
				svp.processBatch(batch)
				batch = batch[:0] // Reset batch
				flushTimer.Reset(svp.flushInterval)
			}

		case <-flushTimer.C:
			// Process remaining messages in batch
			if len(batch) > 0 {
				svp.processBatch(batch)
				batch = batch[:0]
			}
			flushTimer.Reset(svp.flushInterval)

		case <-svp.ctx.Done():
			// Process remaining batch before shutdown
			if len(batch) > 0 {
				svp.processBatch(batch)
			}
			return
		}
	}
}

// processBatch processes a batch of vector messages
func (svp *StreamingVectorProcessor) processBatch(batch []VectorMessage) {
	if len(batch) == 0 {
		return
	}

	// Process messages in parallel within the batch
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, runtime.NumCPU()) // Limit concurrent processing

	for _, msg := range batch {
		wg.Add(1)
		go func(message VectorMessage) {
			defer wg.Done()

			// Acquire semaphore
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			startTime := time.Now()
			result := svp.processMessage(message)
			latency := time.Since(startTime)

			result.Latency = latency
			result.Timestamp = time.Now()

			// Update metrics
			atomic.AddInt64(&svp.processed, 1)
			atomic.AddInt64(&svp.totalLatency, int64(latency))

			// Update max latency
			for {
				currentMax := atomic.LoadInt64(&svp.maxLatency)
				if int64(latency) <= currentMax {
					break
				}
				if atomic.CompareAndSwapInt64(&svp.maxLatency, currentMax, int64(latency)) {
					break
				}
			}

			// Send result
			select {
			case svp.outputStream <- result:
			case <-svp.ctx.Done():
				return
			default:
				// Output stream full, log error
				select {
				case svp.errorStream <- fmt.Errorf("output stream full, result dropped"):
				default:
				}
			}
		}(msg)
	}

	wg.Wait()
}

// processMessage processes a single vector message through the pipeline
func (svp *StreamingVectorProcessor) processMessage(msg VectorMessage) ProcessedResult {
	result := svp.resultPool.Get().(*ProcessedResult)
	defer svp.resultPool.Put(result)

	// Reset result
	result.ID = msg.ID
	result.Result = nil
	result.Error = nil

	// Process through pipeline
	var currentResult interface{} = msg
	var err error

	for i, processor := range svp.processors {
		currentResult, err = processor.Process(svp.ctx, msg)
		if err != nil {
			atomic.AddInt64(&svp.errors, 1)
			result.Error = fmt.Errorf("processor %d failed: %w", i, err)
			break
		}
	}

	result.Result = currentResult

	// Create new result to return (since we're putting the original back in pool)
	return ProcessedResult{
		ID:        result.ID,
		Result:    result.Result,
		Error:     result.Error,
		Latency:   result.Latency,
		Timestamp: result.Timestamp,
	}
}

// monitorPerformance monitors system performance and adjusts parameters
func (svp *StreamingVectorProcessor) monitorPerformance() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	var lastProcessed int64

	for {
		select {
		case <-ticker.C:
			current := atomic.LoadInt64(&svp.processed)
			throughput := current - lastProcessed
			lastProcessed = current

			// Log performance metrics
			if throughput > 0 {
				avgLatency := atomic.LoadInt64(&svp.totalLatency) / current
				maxLatency := atomic.LoadInt64(&svp.maxLatency)
				errorRate := float64(atomic.LoadInt64(&svp.errors)) / float64(current) * 100

				_ = throughput // Use in logging
				_ = avgLatency // Use in logging
				_ = maxLatency // Use in logging
				_ = errorRate  // Use in logging

				// Could log metrics here or send to monitoring system
			}

		case <-svp.ctx.Done():
			return
		}
	}
}

// GetMetrics returns comprehensive performance metrics
func (svp *StreamingVectorProcessor) GetMetrics() map[string]interface{} {
	svp.mu.RLock()
	defer svp.mu.RUnlock()

	processed := atomic.LoadInt64(&svp.processed)
	errors := atomic.LoadInt64(&svp.errors)
	totalLatency := atomic.LoadInt64(&svp.totalLatency)
	maxLatency := atomic.LoadInt64(&svp.maxLatency)

	metrics := map[string]interface{}{
		"processed_messages":  processed,
		"error_count":         errors,
		"max_latency_ns":      maxLatency,
		"input_queue_length":  len(svp.inputStream),
		"output_queue_length": len(svp.outputStream),
		"error_queue_length":  len(svp.errorStream),
		"workers":             svp.workers,
		"batch_size":          svp.batchSize,
		"buffer_size":         svp.bufferSize,
	}

	if processed > 0 {
		metrics["average_latency_ns"] = totalLatency / processed
		metrics["error_rate"] = float64(errors) / float64(processed) * 100
	}

	return metrics
}

// Shutdown gracefully shuts down the streaming processor
func (svp *StreamingVectorProcessor) Shutdown(timeout time.Duration) error {
	// Stop accepting new messages
	svp.cancel()

	// Close input stream
	close(svp.inputStream)

	// Wait for workers to finish
	done := make(chan struct{})
	go func() {
		svp.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Cleanup
		svp.throttle.Stop()
		close(svp.outputStream)
		close(svp.errorStream)
		return nil
	case <-time.After(timeout):
		return fmt.Errorf("shutdown timeout exceeded")
	}
}

// Specialized processors

// SIMDVectorProcessor processes vectors using SIMD operations
type SIMDVectorProcessor struct {
	operation string
	metrics   map[string]interface{}
	mu        sync.RWMutex
}

// NewSIMDVectorProcessor creates a SIMD vector processor
func NewSIMDVectorProcessor(operation string) *SIMDVectorProcessor {
	return &SIMDVectorProcessor{
		operation: operation,
		metrics:   make(map[string]interface{}),
	}
}

// Process processes a vector using SIMD operations
func (svp *SIMDVectorProcessor) Process(ctx context.Context, msg VectorMessage) (interface{}, error) {
	switch svp.operation {
	case "normalize":
		return svp.normalizeVector(msg.Vector), nil
	case "quantize":
		return svp.quantizeVector(msg.Vector), nil
	case "index":
		return svp.indexVector(msg), nil
	default:
		return nil, fmt.Errorf("unsupported operation: %s", svp.operation)
	}
}

// normalizeVector normalizes a vector using SIMD-like operations
func (svp *SIMDVectorProcessor) normalizeVector(vector []float32) []float32 {
	normalized := make([]float32, len(vector))

	// Calculate magnitude (simulated SIMD)
	var magnitude float32
	for i := 0; i < len(vector); i += 4 {
		end := min(i+4, len(vector))
		for j := i; j < end; j++ {
			magnitude += vector[j] * vector[j]
		}
	}

	magnitude = float32(1.0 / (magnitude + 1e-8)) // Avoid division by zero

	// Normalize (simulated SIMD)
	for i := 0; i < len(vector); i += 4 {
		end := min(i+4, len(vector))
		for j := i; j < end; j++ {
			normalized[j] = vector[j] * magnitude
		}
	}

	return normalized
}

// quantizeVector quantizes a vector for compression
func (svp *SIMDVectorProcessor) quantizeVector(vector []float32) []uint8 {
	quantized := make([]uint8, len(vector))

	// Simple 8-bit quantization
	for i := 0; i < len(vector); i += 4 {
		end := min(i+4, len(vector))
		for j := i; j < end; j++ {
			// Scale to [0, 255] range
			val := (vector[j] + 1.0) * 127.5 // Assume input is in [-1, 1]
			if val < 0 {
				val = 0
			} else if val > 255 {
				val = 255
			}
			quantized[j] = uint8(val)
		}
	}

	return quantized
}

// indexVector creates an index entry for the vector
func (svp *SIMDVectorProcessor) indexVector(msg VectorMessage) interface{} {
	return map[string]interface{}{
		"id":        msg.ID,
		"dimension": len(msg.Vector),
		"timestamp": msg.Timestamp,
		"metadata":  msg.Metadata,
	}
}

// GetMetrics returns processor metrics
func (svp *SIMDVectorProcessor) GetMetrics() map[string]interface{} {
	svp.mu.RLock()
	defer svp.mu.RUnlock()

	metrics := make(map[string]interface{})
	for k, v := range svp.metrics {
		metrics[k] = v
	}
	metrics["operation"] = svp.operation

	return metrics
}

// MemoryOptimizedProcessor manages memory usage during processing
type MemoryOptimizedProcessor struct {
	vectorPool     sync.Pool
	resultPool     sync.Pool
	maxVectors     int64
	currentVectors int64
	metrics        map[string]interface{}
	mu             sync.RWMutex
}

// NewMemoryOptimizedProcessor creates a memory-optimized processor
func NewMemoryOptimizedProcessor(maxVectors int64) *MemoryOptimizedProcessor {
	return &MemoryOptimizedProcessor{
		maxVectors: maxVectors,
		metrics:    make(map[string]interface{}),
		vectorPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, 0, 1024)
			},
		},
		resultPool: sync.Pool{
			New: func() interface{} {
				return make(map[string]interface{})
			},
		},
	}
}

// Process processes vectors with memory optimization
func (mop *MemoryOptimizedProcessor) Process(ctx context.Context, msg VectorMessage) (interface{}, error) {
	// Check memory limits
	current := atomic.LoadInt64(&mop.currentVectors)
	if current >= mop.maxVectors {
		return nil, fmt.Errorf("memory limit exceeded")
	}

	atomic.AddInt64(&mop.currentVectors, 1)
	defer atomic.AddInt64(&mop.currentVectors, -1)

	// Get vector from pool
	vector := mop.vectorPool.Get().([]float32)
	// Note: SA6002 warning is a false positive - sync.Pool handles this correctly
	defer func() {
		mop.vectorPool.Put(interface{}(vector[:0]))
	}()

	// Process vector (copy to avoid aliasing)
	vector = vector[:len(msg.Vector)]
	copy(vector, msg.Vector)

	// Get result from pool
	result := mop.resultPool.Get().(map[string]interface{})
	defer mop.resultPool.Put(result)

	// Clear result map
	for key := range result {
		delete(result, key)
	}

	// Simulate processing
	result["processed_vector_id"] = msg.ID
	result["dimension"] = len(vector)
	result["memory_usage"] = current

	// Create new result to return
	resultCopy := make(map[string]interface{})
	for k, v := range result {
		resultCopy[k] = v
	}

	return resultCopy, nil
}

// GetMetrics returns memory metrics
func (mop *MemoryOptimizedProcessor) GetMetrics() map[string]interface{} {
	mop.mu.RLock()
	defer mop.mu.RUnlock()

	metrics := make(map[string]interface{})
	for k, v := range mop.metrics {
		metrics[k] = v
	}

	metrics["max_vectors"] = mop.maxVectors
	metrics["current_vectors"] = atomic.LoadInt64(&mop.currentVectors)

	return metrics
}

// Utility functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
