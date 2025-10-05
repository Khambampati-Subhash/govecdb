package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/khambampati-subhash/govecdb/index"
)

// BenchmarkResult represents the results of a single benchmark
type BenchmarkResult struct {
	TestName         string  `json:"test_name"`
	Dimension        int     `json:"dimension"`
	NumVectors       int     `json:"num_vectors"`
	Operation        string  `json:"operation"`
	TotalTimeMs      float64 `json:"total_time_ms"`
	AvgTimeMs        float64 `json:"avg_time_ms"`
	MinTimeMs        float64 `json:"min_time_ms"`
	MaxTimeMs        float64 `json:"max_time_ms"`
	ThroughputVecSec float64 `json:"throughput_vec_per_sec"`
	SuccessfulOps    int     `json:"successful_operations"`
	FailedOps        int     `json:"failed_operations"`
}

func main() {
	fmt.Println("GoVecDB Simple Benchmark - SIMD Distance Function Test")
	fmt.Println("=====================================================")

	results := []BenchmarkResult{}

	// Test configurations
	configs := []struct {
		dimension  int
		numVectors int
	}{
		{128, 1000},
		{128, 5000},
	}

	for _, config := range configs {
		fmt.Printf("\n🧪 Testing %d vectors (dimension %d)...\n", config.numVectors, config.dimension)

		result := runBenchmark(config.dimension, config.numVectors)
		results = append(results, result...)

		fmt.Println("✅ Test completed")
	}

	// Output JSON results
	jsonOutput, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		fmt.Printf("Error marshaling results: %v\n", err)
		return
	}

	fmt.Printf("\n📊 Final Results:\n%s\n", string(jsonOutput))

	// Save to file
	if err := os.WriteFile("govecdb_benchmark_results.json", jsonOutput, 0644); err != nil {
		fmt.Printf("Error saving results: %v\n", err)
	} else {
		fmt.Println("💾 Results saved to govecdb_benchmark_results.json")
	}
}

func runBenchmark(dimension, numVectors int) []BenchmarkResult {
	results := []BenchmarkResult{}

	// Create index
	config := &index.Config{
		Dimension:      dimension,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		Metric:         index.Cosine,
		ThreadSafe:     true,
	}

	idx, err := index.NewHNSWIndex(config)
	if err != nil {
		fmt.Printf("Error creating index: %v\n", err)
		return results
	}
	defer idx.Close()

	// Generate test vectors
	vectors := make([]*index.Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		data := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			data[j] = rand.Float32()*2 - 1 // Range [-1, 1]
		}

		// Normalize vector
		var norm float32
		for _, val := range data {
			norm += val * val
		}
		norm = float32(1.0 / (float64(norm) + 1e-8))
		for j := range data {
			data[j] *= norm
		}

		vectors[i] = &index.Vector{
			ID:   fmt.Sprintf("vec_%d", i),
			Data: data,
		}
	}

	// Benchmark single insert
	fmt.Print("  📝 Single insert... ")
	result := benchmarkSingleInsert(idx, vectors[:100]) // Test with first 100 vectors
	results = append(results, result)
	fmt.Printf("✅ (%.2fms avg, %d success)\n", result.AvgTimeMs, result.SuccessfulOps)

	// Clear index for batch test
	idx.Close()
	idx, err = index.NewHNSWIndex(config)
	if err != nil {
		fmt.Printf("Error recreating index: %v\n", err)
		return results
	}
	defer idx.Close()

	// Benchmark batch insert
	fmt.Print("  📦 Batch insert... ")
	result = benchmarkBatchInsert(idx, vectors)
	results = append(results, result)
	fmt.Printf("✅ (%.2f vec/sec)\n", result.ThroughputVecSec)

	// Benchmark search
	fmt.Print("  🔍 Search operations... ")
	queryVector := vectors[0].Data
	result = benchmarkSearch(idx, queryVector, 10, 100)
	results = append(results, result)
	fmt.Printf("✅ (%.2fms avg)\n", result.AvgTimeMs)

	return results
}

func benchmarkSingleInsert(idx *index.HNSWIndex, vectors []*index.Vector) BenchmarkResult {
	times := make([]float64, 0, len(vectors))
	successCount := 0
	failCount := 0

	for _, vector := range vectors {
		start := time.Now()
		err := idx.Add(vector)
		elapsed := time.Since(start)

		if err != nil {
			failCount++
		} else {
			successCount++
			times = append(times, float64(elapsed.Nanoseconds())/1e6) // Convert to milliseconds
		}
	}

	var totalTime, minTime, maxTime, avgTime float64
	if len(times) > 0 {
		totalTime = times[0]
		minTime = times[0]
		maxTime = times[0]

		for _, t := range times {
			totalTime += t
			if t < minTime {
				minTime = t
			}
			if t > maxTime {
				maxTime = t
			}
		}
		avgTime = totalTime / float64(len(times))
	}

	return BenchmarkResult{
		TestName:         "Single Insert",
		Dimension:        len(vectors[0].Data),
		NumVectors:       len(vectors),
		Operation:        "single_insert",
		TotalTimeMs:      totalTime,
		AvgTimeMs:        avgTime,
		MinTimeMs:        minTime,
		MaxTimeMs:        maxTime,
		ThroughputVecSec: float64(successCount) / (totalTime / 1000.0),
		SuccessfulOps:    successCount,
		FailedOps:        failCount,
	}
}

func benchmarkBatchInsert(idx *index.HNSWIndex, vectors []*index.Vector) BenchmarkResult {
	start := time.Now()
	err := idx.AddBatch(vectors)
	elapsed := time.Since(start)

	successCount := len(vectors)
	failCount := 0
	if err != nil {
		failCount = len(vectors)
		successCount = 0
	}

	totalTimeMs := float64(elapsed.Nanoseconds()) / 1e6
	avgTimeMs := totalTimeMs / float64(len(vectors))
	throughput := float64(successCount) / (totalTimeMs / 1000.0)

	return BenchmarkResult{
		TestName:         "Batch Insert",
		Dimension:        len(vectors[0].Data),
		NumVectors:       len(vectors),
		Operation:        "batch_insert",
		TotalTimeMs:      totalTimeMs,
		AvgTimeMs:        avgTimeMs,
		MinTimeMs:        avgTimeMs, // Same for batch
		MaxTimeMs:        avgTimeMs, // Same for batch
		ThroughputVecSec: throughput,
		SuccessfulOps:    successCount,
		FailedOps:        failCount,
	}
}

func benchmarkSearch(idx *index.HNSWIndex, query []float32, k int, numSearches int) BenchmarkResult {
	times := make([]float64, 0, numSearches)
	successCount := 0
	failCount := 0

	for i := 0; i < numSearches; i++ {
		start := time.Now()
		_, err := idx.Search(query, k)
		elapsed := time.Since(start)

		if err != nil {
			failCount++
		} else {
			successCount++
			times = append(times, float64(elapsed.Nanoseconds())/1e6) // Convert to milliseconds
		}
	}

	var totalTime, minTime, maxTime, avgTime float64
	if len(times) > 0 {
		totalTime = times[0]
		minTime = times[0]
		maxTime = times[0]

		for _, t := range times {
			totalTime += t
			if t < minTime {
				minTime = t
			}
			if t > maxTime {
				maxTime = t
			}
		}
		avgTime = totalTime / float64(len(times))
	}

	return BenchmarkResult{
		TestName:         "Search",
		Dimension:        len(query),
		NumVectors:       numSearches,
		Operation:        "search",
		TotalTimeMs:      totalTime,
		AvgTimeMs:        avgTime,
		MinTimeMs:        minTime,
		MaxTimeMs:        maxTime,
		ThroughputVecSec: float64(successCount) / (totalTime / 1000.0),
		SuccessfulOps:    successCount,
		FailedOps:        failCount,
	}
}
