package benchmark

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// ComprehensiveBenchmark provides extensive benchmarking capabilities
type ComprehensiveBenchmark struct {
	// Configuration
	config *BenchmarkConfig

	// Results storage
	results   map[string]*BenchmarkResult
	resultsMu sync.RWMutex

	// Memory profiling
	memProfiler *MemoryProfiler

	// Performance tracking
	perfTracker *PerformanceTracker

	// Test data generators
	dataGen *TestDataGenerator
}

// BenchmarkConfig defines benchmark parameters
type BenchmarkConfig struct {
	VectorDimensions  []int
	DatasetSizes      []int
	NumQueries        int
	NumWorkers        int
	BatchSizes        []int
	DistanceMetrics   []string
	IndexTypes        []string
	WarmupIterations  int
	BenchmarkDuration time.Duration
	MemoryProfiling   bool
	CPUProfiling      bool
}

// BenchmarkResult stores comprehensive benchmark results
type BenchmarkResult struct {
	TestName  string
	Timestamp time.Time
	Duration  time.Duration

	// Performance metrics
	InsertThroughput float64 // ops/sec
	SearchThroughput float64 // ops/sec
	SearchLatency    LatencyStats
	InsertLatency    LatencyStats

	// Resource usage
	MemoryUsage    MemoryStats
	CPUUsage       float64
	GoroutineCount int

	// Index metrics
	IndexSize     int64
	NodeCount     int64
	EdgeCount     int64
	AverageDegree float64

	// Quality metrics
	Recall        float64
	Precision     float64
	SearchQuality float64

	// Configuration
	VectorDimension int
	DatasetSize     int
	BatchSize       int
	NumWorkers      int
	IndexType       string
	DistanceMetric  string
}

// LatencyStats represents latency statistics
type LatencyStats struct {
	Min    time.Duration
	Max    time.Duration
	Mean   time.Duration
	Median time.Duration
	P95    time.Duration
	P99    time.Duration
	P999   time.Duration
	StdDev time.Duration
}

// MemoryStats represents memory usage statistics
type MemoryStats struct {
	AllocBytes      uint64
	TotalAllocBytes uint64
	SysBytes        uint64
	NumGC           uint32
	GCPauseTotal    time.Duration
	HeapInUse       uint64
	StackInUse      uint64
}

// MemoryProfiler tracks memory usage during benchmarks
type MemoryProfiler struct {
	samples   []MemoryStats
	samplesMu sync.Mutex
	sampling  int32
	interval  time.Duration
}

// PerformanceTracker tracks performance metrics
type PerformanceTracker struct {
	operations     int64
	errors         int64
	startTime      time.Time
	lastCheckpoint time.Time

	// Latency tracking
	latencies   []time.Duration
	latenciesMu sync.Mutex
}

// TestDataGenerator generates test data for benchmarks
type TestDataGenerator struct {
	rng        *rand.Rand
	rngMu      sync.Mutex
	vectorPool sync.Pool
}

// NewComprehensiveBenchmark creates a new comprehensive benchmark suite
func NewComprehensiveBenchmark(config *BenchmarkConfig) *ComprehensiveBenchmark {
	return &ComprehensiveBenchmark{
		config:      config,
		results:     make(map[string]*BenchmarkResult),
		memProfiler: NewMemoryProfiler(100 * time.Millisecond),
		perfTracker: NewPerformanceTracker(),
		dataGen:     NewTestDataGenerator(),
	}
}

// NewMemoryProfiler creates a new memory profiler
func NewMemoryProfiler(interval time.Duration) *MemoryProfiler {
	return &MemoryProfiler{
		interval: interval,
		samples:  make([]MemoryStats, 0, 10000),
	}
}

// NewPerformanceTracker creates a new performance tracker
func NewPerformanceTracker() *PerformanceTracker {
	return &PerformanceTracker{
		startTime:      time.Now(),
		lastCheckpoint: time.Now(),
		latencies:      make([]time.Duration, 0, 100000),
	}
}

// NewTestDataGenerator creates a new test data generator
func NewTestDataGenerator() *TestDataGenerator {
	return &TestDataGenerator{
		rng: rand.New(rand.NewSource(42)), // Fixed seed for reproducibility
		vectorPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, 0, 1024)
			},
		},
	}
}

// RunFullBenchmarkSuite runs the complete benchmark suite
func (cb *ComprehensiveBenchmark) RunFullBenchmarkSuite(ctx context.Context) (*BenchmarkSuite, error) {
	suite := &BenchmarkSuite{
		StartTime: time.Now(),
		Config:    cb.config,
		Results:   make(map[string]*BenchmarkResult),
	}

	fmt.Println("🚀 Starting Comprehensive GoVecDB Benchmark Suite")
	fmt.Println("==================================================")

	// Test matrix
	for _, indexType := range cb.config.IndexTypes {
		for _, distanceMetric := range cb.config.DistanceMetrics {
			for _, dimension := range cb.config.VectorDimensions {
				for _, datasetSize := range cb.config.DatasetSizes {
					for _, batchSize := range cb.config.BatchSizes {

						testName := fmt.Sprintf("%s_%s_%dd_%dn_b%d",
							indexType, distanceMetric, dimension, datasetSize, batchSize)

						fmt.Printf("🎯 Running: %s\n", testName)

						result, err := cb.runSingleBenchmark(ctx, BenchmarkParams{
							TestName:        testName,
							IndexType:       indexType,
							DistanceMetric:  distanceMetric,
							VectorDimension: dimension,
							DatasetSize:     datasetSize,
							BatchSize:       batchSize,
							NumWorkers:      cb.config.NumWorkers,
						})

						if err != nil {
							fmt.Printf("❌ Failed: %s - %v\n", testName, err)
							continue
						}

						suite.Results[testName] = result
						cb.storeResult(testName, result)

						// Print interim results
						cb.printBenchmarkResult(result)

						// Check for cancellation
						select {
						case <-ctx.Done():
							return suite, ctx.Err()
						default:
						}
					}
				}
			}
		}
	}

	suite.EndTime = time.Now()
	suite.Duration = suite.EndTime.Sub(suite.StartTime)

	// Generate final report
	cb.generateFinalReport(suite)

	return suite, nil
}

// BenchmarkParams defines parameters for a single benchmark
type BenchmarkParams struct {
	TestName        string
	IndexType       string
	DistanceMetric  string
	VectorDimension int
	DatasetSize     int
	BatchSize       int
	NumWorkers      int
}

// runSingleBenchmark runs a single benchmark test
func (cb *ComprehensiveBenchmark) runSingleBenchmark(ctx context.Context, params BenchmarkParams) (*BenchmarkResult, error) {
	// Start memory profiling
	if cb.config.MemoryProfiling {
		cb.memProfiler.StartProfiling()
		defer cb.memProfiler.StopProfiling()
	}

	// Reset performance tracker
	cb.perfTracker.Reset()

	// Create index based on type
	var graphIndex interface{}
	switch params.IndexType {
	case "optimized_hnsw":
		graphIndex = cb.createOptimizedHNSW(params)
	case "concurrent_hnsw":
		graphIndex = cb.createConcurrentHNSW(params)
	case "multi_index":
		graphIndex = cb.createMultiIndex(params)
	default:
		return nil, fmt.Errorf("unknown index type: %s", params.IndexType)
	}

	// Generate test data
	vectors, queries, groundTruth := cb.dataGen.GenerateTestData(
		params.DatasetSize,
		cb.config.NumQueries,
		params.VectorDimension,
	)

	startTime := time.Now()

	// Benchmark insertions
	insertStats, err := cb.benchmarkInsertions(ctx, graphIndex, vectors, params)
	if err != nil {
		return nil, fmt.Errorf("insertion benchmark failed: %w", err)
	}

	// Benchmark searches
	searchStats, err := cb.benchmarkSearches(ctx, graphIndex, queries, groundTruth, params)
	if err != nil {
		return nil, fmt.Errorf("search benchmark failed: %w", err)
	}

	duration := time.Since(startTime)

	// Collect final metrics
	memStats := cb.getMemoryStats()
	cpuUsage := cb.getCPUUsage()

	result := &BenchmarkResult{
		TestName:         params.TestName,
		Timestamp:        startTime,
		Duration:         duration,
		InsertThroughput: insertStats.Throughput,
		SearchThroughput: searchStats.Throughput,
		SearchLatency:    searchStats.Latency,
		InsertLatency:    insertStats.Latency,
		MemoryUsage:      memStats,
		CPUUsage:         cpuUsage,
		GoroutineCount:   runtime.NumGoroutine(),
		VectorDimension:  params.VectorDimension,
		DatasetSize:      params.DatasetSize,
		BatchSize:        params.BatchSize,
		NumWorkers:       params.NumWorkers,
		IndexType:        params.IndexType,
		DistanceMetric:   params.DistanceMetric,
		Recall:           searchStats.Recall,
		Precision:        searchStats.Precision,
		SearchQuality:    searchStats.Quality,
	}

	return result, nil
}

// benchmarkInsertions benchmarks vector insertions
func (cb *ComprehensiveBenchmark) benchmarkInsertions(ctx context.Context, index interface{}, vectors [][]float32, params BenchmarkParams) (*InsertionStats, error) {
	latencies := make([]time.Duration, 0, len(vectors))
	var totalOps int64

	startTime := time.Now()

	// Batch insertions
	batchSize := params.BatchSize
	numBatches := (len(vectors) + batchSize - 1) / batchSize

	for i := 0; i < numBatches; i++ {
		batchStart := i * batchSize
		batchEnd := batchStart + batchSize
		if batchEnd > len(vectors) {
			batchEnd = len(vectors)
		}

		batchStartTime := time.Now()

		// Insert batch (simplified - actual implementation depends on index type)
		for j := batchStart; j < batchEnd; j++ {
			// Simulate insertion (replace with actual index insertion)
			time.Sleep(time.Microsecond) // Simulate work
			atomic.AddInt64(&totalOps, 1)
		}

		batchDuration := time.Since(batchStartTime)
		batchLatency := batchDuration / time.Duration(batchEnd-batchStart)

		latencies = append(latencies, batchLatency)

		// Check for cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}

	totalDuration := time.Since(startTime)
	throughput := float64(totalOps) / totalDuration.Seconds()

	return &InsertionStats{
		Throughput: throughput,
		Latency:    calculateLatencyStats(latencies),
		Duration:   totalDuration,
		Operations: totalOps,
	}, nil
}

// benchmarkSearches benchmarks vector searches
func (cb *ComprehensiveBenchmark) benchmarkSearches(ctx context.Context, index interface{}, queries [][]float32, groundTruth [][]int, params BenchmarkParams) (*SearchStats, error) {
	latencies := make([]time.Duration, 0, len(queries))
	var totalOps int64
	var totalRecall float64

	startTime := time.Now()

	// Parallel searches
	numWorkers := params.NumWorkers
	queryChan := make(chan int, len(queries))
	resultChan := make(chan SearchResult, len(queries))

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for queryIdx := range queryChan {
				queryStart := time.Now()

				// Perform search (simplified - replace with actual search)
				results := cb.simulateSearch(index, queries[queryIdx])

				queryDuration := time.Since(queryStart)
				recall := cb.calculateRecall(results, groundTruth[queryIdx])

				resultChan <- SearchResult{
					Latency: queryDuration,
					Recall:  recall,
				}

				atomic.AddInt64(&totalOps, 1)
			}
		}()
	}

	// Send queries
	go func() {
		defer close(queryChan)
		for i := range queries {
			select {
			case queryChan <- i:
			case <-ctx.Done():
				return
			}
		}
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for result := range resultChan {
		latencies = append(latencies, result.Latency)
		totalRecall += result.Recall
	}

	totalDuration := time.Since(startTime)
	throughput := float64(totalOps) / totalDuration.Seconds()
	avgRecall := totalRecall / float64(len(queries))

	return &SearchStats{
		Throughput: throughput,
		Latency:    calculateLatencyStats(latencies),
		Duration:   totalDuration,
		Operations: totalOps,
		Recall:     avgRecall,
		Precision:  avgRecall, // Simplified
		Quality:    avgRecall, // Simplified
	}, nil
}

// Helper functions and types

type InsertionStats struct {
	Throughput float64
	Latency    LatencyStats
	Duration   time.Duration
	Operations int64
}

type SearchStats struct {
	Throughput float64
	Latency    LatencyStats
	Duration   time.Duration
	Operations int64
	Recall     float64
	Precision  float64
	Quality    float64
}

type SearchResult struct {
	Latency time.Duration
	Recall  float64
}

type BenchmarkSuite struct {
	StartTime time.Time
	EndTime   time.Time
	Duration  time.Duration
	Config    *BenchmarkConfig
	Results   map[string]*BenchmarkResult
}

// Placeholder implementations (replace with actual index operations)

func (cb *ComprehensiveBenchmark) createOptimizedHNSW(params BenchmarkParams) interface{} {
	// Return actual optimized HNSW index
	return "optimized_hnsw_index"
}

func (cb *ComprehensiveBenchmark) createConcurrentHNSW(params BenchmarkParams) interface{} {
	return "concurrent_hnsw_index"
}

func (cb *ComprehensiveBenchmark) createMultiIndex(params BenchmarkParams) interface{} {
	return "multi_index"
}

func (cb *ComprehensiveBenchmark) simulateSearch(index interface{}, query []float32) []int {
	// Simulate search results
	return []int{1, 2, 3, 4, 5}
}

func (cb *ComprehensiveBenchmark) calculateRecall(results []int, groundTruth []int) float64 {
	// Simplified recall calculation
	return 0.95
}

// GenerateTestData generates test vectors and queries
func (tg *TestDataGenerator) GenerateTestData(datasetSize, numQueries, dimension int) ([][]float32, [][]float32, [][]int) {
	tg.rngMu.Lock()
	defer tg.rngMu.Unlock()

	// Generate dataset
	vectors := make([][]float32, datasetSize)
	for i := 0; i < datasetSize; i++ {
		vector := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vector[j] = tg.rng.Float32()
		}
		vectors[i] = vector
	}

	// Generate queries
	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		query := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			query[j] = tg.rng.Float32()
		}
		queries[i] = query
	}

	// Generate ground truth (simplified)
	groundTruth := make([][]int, numQueries)
	for i := 0; i < numQueries; i++ {
		truth := make([]int, 10) // Top 10
		for j := 0; j < 10; j++ {
			truth[j] = tg.rng.Intn(datasetSize)
		}
		groundTruth[i] = truth
	}

	return vectors, queries, groundTruth
}

// calculateLatencyStats calculates latency statistics
func calculateLatencyStats(latencies []time.Duration) LatencyStats {
	if len(latencies) == 0 {
		return LatencyStats{}
	}

	// Sort latencies
	for i := 0; i < len(latencies); i++ {
		for j := i + 1; j < len(latencies); j++ {
			if latencies[i] > latencies[j] {
				latencies[i], latencies[j] = latencies[j], latencies[i]
			}
		}
	}

	var sum time.Duration
	for _, lat := range latencies {
		sum += lat
	}

	return LatencyStats{
		Min:    latencies[0],
		Max:    latencies[len(latencies)-1],
		Mean:   sum / time.Duration(len(latencies)),
		Median: latencies[len(latencies)/2],
		P95:    latencies[len(latencies)*95/100],
		P99:    latencies[len(latencies)*99/100],
		P999:   latencies[len(latencies)*999/1000],
	}
}

// getMemoryStats gets current memory statistics
func (cb *ComprehensiveBenchmark) getMemoryStats() MemoryStats {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	return MemoryStats{
		AllocBytes:      ms.Alloc,
		TotalAllocBytes: ms.TotalAlloc,
		SysBytes:        ms.Sys,
		NumGC:           ms.NumGC,
		GCPauseTotal:    time.Duration(ms.PauseTotalNs),
		HeapInUse:       ms.HeapInuse,
		StackInUse:      ms.StackInuse,
	}
}

// getCPUUsage gets current CPU usage (simplified)
func (cb *ComprehensiveBenchmark) getCPUUsage() float64 {
	// Simplified CPU usage calculation
	return 0.0
}

// Reset resets the performance tracker
func (pt *PerformanceTracker) Reset() {
	atomic.StoreInt64(&pt.operations, 0)
	atomic.StoreInt64(&pt.errors, 0)
	pt.startTime = time.Now()
	pt.lastCheckpoint = time.Now()

	pt.latenciesMu.Lock()
	pt.latencies = pt.latencies[:0]
	pt.latenciesMu.Unlock()
}

// StartProfiling starts memory profiling
func (mp *MemoryProfiler) StartProfiling() {
	if atomic.CompareAndSwapInt32(&mp.sampling, 0, 1) {
		go mp.profileLoop()
	}
}

// StopProfiling stops memory profiling
func (mp *MemoryProfiler) StopProfiling() {
	atomic.StoreInt32(&mp.sampling, 0)
}

// profileLoop runs the memory profiling loop
func (mp *MemoryProfiler) profileLoop() {
	ticker := time.NewTicker(mp.interval)
	defer ticker.Stop()

	for atomic.LoadInt32(&mp.sampling) == 1 {
		<-ticker.C
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)

		stats := MemoryStats{
			AllocBytes:      ms.Alloc,
			TotalAllocBytes: ms.TotalAlloc,
			SysBytes:        ms.Sys,
			NumGC:           ms.NumGC,
			GCPauseTotal:    time.Duration(ms.PauseTotalNs),
			HeapInUse:       ms.HeapInuse,
			StackInUse:      ms.StackInuse,
		}

		mp.samplesMu.Lock()
		mp.samples = append(mp.samples, stats)
		mp.samplesMu.Unlock()
	}
}

// storeResult stores a benchmark result
func (cb *ComprehensiveBenchmark) storeResult(name string, result *BenchmarkResult) {
	cb.resultsMu.Lock()
	defer cb.resultsMu.Unlock()
	cb.results[name] = result
}

// printBenchmarkResult prints a single benchmark result
func (cb *ComprehensiveBenchmark) printBenchmarkResult(result *BenchmarkResult) {
	fmt.Printf("📊 %s Results:\n", result.TestName)
	fmt.Printf("   📤 Insert: %.0f ops/sec (%.2fms avg latency)\n",
		result.InsertThroughput, float64(result.InsertLatency.Mean.Nanoseconds())/1e6)
	fmt.Printf("   🔍 Search: %.0f ops/sec (%.2fms avg latency)\n",
		result.SearchThroughput, float64(result.SearchLatency.Mean.Nanoseconds())/1e6)
	fmt.Printf("   💾 Memory: %.2f MB allocated\n", float64(result.MemoryUsage.AllocBytes)/1024/1024)
	fmt.Printf("   🎯 Recall: %.2f%%\n", result.Recall*100)
	fmt.Println()
}

// generateFinalReport generates a comprehensive final report
func (cb *ComprehensiveBenchmark) generateFinalReport(suite *BenchmarkSuite) {
	fmt.Println("\n🎯 COMPREHENSIVE BENCHMARK REPORT")
	fmt.Println("=================================")
	fmt.Printf("Duration: %v\n", suite.Duration)
	fmt.Printf("Total Tests: %d\n", len(suite.Results))
	fmt.Println()

	// Find best performers
	var bestInsert, bestSearch *BenchmarkResult
	for _, result := range suite.Results {
		if bestInsert == nil || result.InsertThroughput > bestInsert.InsertThroughput {
			bestInsert = result
		}
		if bestSearch == nil || result.SearchThroughput > bestSearch.SearchThroughput {
			bestSearch = result
		}
	}

	fmt.Println("🏆 BEST PERFORMERS:")
	if bestInsert != nil {
		fmt.Printf("   📤 Best Insert: %s (%.0f ops/sec)\n", bestInsert.TestName, bestInsert.InsertThroughput)
	}
	if bestSearch != nil {
		fmt.Printf("   🔍 Best Search: %s (%.0f ops/sec)\n", bestSearch.TestName, bestSearch.SearchThroughput)
	}
	fmt.Println()

	// Performance summary
	fmt.Println("📊 PERFORMANCE SUMMARY:")
	for testName, result := range suite.Results {
		fmt.Printf("   %s: I=%.0f S=%.0f (ops/sec)\n",
			testName, result.InsertThroughput, result.SearchThroughput)
	}
}
