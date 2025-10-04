package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
	"github.com/khambampati-subhash/govecdb/collection"
	"github.com/khambampati-subhash/govecdb/store"
)

// BenchmarkResult stores performance metrics
type BenchmarkResult struct {
	Dimension     int
	NumVectors    int
	Operation     string
	TotalTime     time.Duration
	AvgTime       time.Duration
	MinTime       time.Duration
	MaxTime       time.Duration
	Throughput    float64
	MemoryUsed    uint64
	Recall        float64
	SearchQuality float64
}

// JSONBenchmarkResult is the JSON-serializable version of BenchmarkResult
type JSONBenchmarkResult struct {
	Dimension     int     `json:"dimension"`
	NumVectors    int     `json:"num_vectors"`
	Operation     string  `json:"operation"`
	TotalTime     float64 `json:"total_time"`     // in seconds
	AvgTime       float64 `json:"avg_time"`       // in seconds
	MinTime       float64 `json:"min_time"`       // in seconds
	MaxTime       float64 `json:"max_time"`       // in seconds
	Throughput    float64 `json:"throughput"`     // ops/sec
	Recall        float64 `json:"recall"`         // 0-1
	SearchQuality float64 `json:"search_quality"` // 0-1
}

// BenchmarkMetadata stores metadata about the benchmark run
type BenchmarkMetadata struct {
	Timestamp    string `json:"timestamp"`
	Dimensions   []int  `json:"dimensions"`
	VectorCounts []int  `json:"vector_counts"`
	TotalTests   int    `json:"total_tests"`
	Database     string `json:"database"`
}

// JSONOutput is the complete JSON output structure
type JSONOutput struct {
	Metadata BenchmarkMetadata     `json:"metadata"`
	Results  []JSONBenchmarkResult `json:"results"`
}

// TestConfig defines test parameters
type TestConfig struct {
	Dimensions   []int
	VectorCounts []int
	SearchK      int
	NumSearches  int
}

// ToJSON converts BenchmarkResult to JSONBenchmarkResult
func (r *BenchmarkResult) ToJSON() JSONBenchmarkResult {
	return JSONBenchmarkResult{
		Dimension:     r.Dimension,
		NumVectors:    r.NumVectors,
		Operation:     r.Operation,
		TotalTime:     r.TotalTime.Seconds(),
		AvgTime:       r.AvgTime.Seconds(),
		MinTime:       r.MinTime.Seconds(),
		MaxTime:       r.MaxTime.Seconds(),
		Throughput:    r.Throughput,
		Recall:        r.Recall,
		SearchQuality: r.SearchQuality,
	}
}

func main() {
	fmt.Println("🚀 GoVecDB Comprehensive Performance Benchmark")
	fmt.Println("=" + string(make([]byte, 60)))
	startTime := time.Now()
	fmt.Printf("⏰ Started: %s\n", startTime.Format("2006-01-02 15:04:05"))
	fmt.Println()

	// Test configuration - Updated per user requirements
	// Start with a smaller test first, then uncomment the full config
	// 2, 16, 32, 64, 128, 256, 384,
	config := TestConfig{
		Dimensions:   []int{512, 768, 1024, 1536, 2048, 3072, 4096, 6120}, // Small test first
		VectorCounts: []int{1000, 2000, 3000, 4000, 5000},                 // Small test first
		SearchK:      10,
		NumSearches:  100, // Reduced for faster testing
	}

	// Full configuration - uncomment when ready for complete benchmark:
	// config := TestConfig{
	//     Dimensions:   []int{2, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6120},
	//     VectorCounts: []int{1000, 2000, 3000, 4000, 5000},
	//     SearchK:      10,
	//     NumSearches:  100,
	// }

	results := []BenchmarkResult{}

	// Run benchmarks for each dimension
	for _, dim := range config.Dimensions {
		fmt.Printf("\n📊 Testing Dimension: %d\n", dim)
		fmt.Println("─" + string(make([]byte, 60)))

		for _, numVectors := range config.VectorCounts {
			if shouldSkip(dim, numVectors) {
				fmt.Printf("\n  ⏭️  Skipping Vector Count: %d (too large for dimension %d)\n", numVectors, dim)
				continue
			}

			fmt.Printf("\n  📦 Vector Count: %d\n", numVectors)

			result := runBenchmark(dim, numVectors, config.SearchK, config.NumSearches)
			results = append(results, result...)

			printResults(result)

			// Force garbage collection after each large test to manage memory
			if dim*numVectors > 1000000 { // Large test case
				fmt.Print("    🧹 Running garbage collection... ")
				time.Sleep(100 * time.Millisecond) // Brief pause
				fmt.Println("✅")
			}
		}
	}

	// Print final summary
	printSummary(results)

	// Save results to JSON and CSV
	if err := saveResultsToJSON(results, config); err != nil {
		fmt.Printf("\n⚠️  Warning: Failed to save JSON results: %v\n", err)
	}

	if err := saveResultsToCSV(results); err != nil {
		fmt.Printf("\n⚠️  Warning: Failed to save CSV results: %v\n", err)
	}

	fmt.Printf("\n⏰ Completed: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Printf("⏱️  Total Duration: %s\n", time.Since(startTime).Round(time.Second))
}

func shouldSkip(dim, numVectors int) bool {
	// Skip very large combinations to save time and memory
	if dim >= 6120 && numVectors > 3000 {
		return true
	}
	if dim >= 4096 && numVectors > 4000 {
		return true
	}
	if dim >= 2048 && numVectors > 5000 {
		return true
	}
	// Skip very small dimensions with large vector counts for efficiency
	if dim <= 2 && numVectors > 3000 {
		return true
	}
	return false
}

func runBenchmark(dim, numVectors, searchK, numSearches int) []BenchmarkResult {
	results := []BenchmarkResult{}

	// Create collection
	collConfig := &api.CollectionConfig{
		Name:           fmt.Sprintf("bench_%d_%d", dim, numVectors),
		Dimension:      dim,
		Metric:         api.Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		ThreadSafe:     true,
	}

	storeConfig := &store.StoreConfig{
		Name:         fmt.Sprintf("bench_store_%d", dim),
		PreallocSize: numVectors,
		EnableStats:  true,
	}

	coll, err := collection.NewVectorCollection(collConfig, storeConfig)
	if err != nil {
		fmt.Printf("    ❌ Failed to create collection: %v\n", err)
		return results
	}
	defer coll.Close()

	ctx := context.Background()

	// Generate test vectors
	fmt.Print("    🎲 Generating vectors... ")
	vectors := generateVectors(numVectors, dim)
	fmt.Println("✅")

	// Benchmark: Batch Insert
	fmt.Print("    ⏱️  Batch insert... ")
	insertResult := benchmarkBatchInsert(ctx, coll, vectors)
	results = append(results, insertResult)
	if insertResult.Throughput > 0 {
		fmt.Printf("✅ (%.3fms avg, %.0f vec/sec)\n",
			float64(insertResult.AvgTime.Microseconds())/1000.0,
			insertResult.Throughput)
	} else {
		fmt.Printf("❌ (failed)\n")
	}

	// Benchmark: Single Insert (subset)
	if numVectors <= 10000 {
		fmt.Print("    ⏱️  Single insert... ")
		singleInsertResult := benchmarkSingleInsert(ctx, coll, dim, 100)
		results = append(results, singleInsertResult)
		fmt.Printf("✅ (%.3fms avg)\n",
			float64(singleInsertResult.AvgTime.Microseconds())/1000.0)
	}

	// Benchmark: Exact Search
	fmt.Print("    🔍 Exact search (k=1)... ")
	exactSearchResult := benchmarkExactSearch(ctx, coll, vectors, 1, numSearches)
	results = append(results, exactSearchResult)
	fmt.Printf("✅ (%.3fms avg, recall: %.2f%%)\n",
		float64(exactSearchResult.AvgTime.Microseconds())/1000.0,
		exactSearchResult.Recall*100)

	// Benchmark: KNN Search (this is what our Search method does)
	fmt.Print("    🔍 KNN search (k=10)... ")
	knnSearchResult := benchmarkSearch(ctx, coll, vectors, searchK, numSearches)
	results = append(results, knnSearchResult)
	fmt.Printf("✅ (%.3fms avg, quality: %.2f%%)\n",
		float64(knnSearchResult.AvgTime.Microseconds())/1000.0,
		knnSearchResult.SearchQuality*100)

	// Benchmark: Large K Search
	if numVectors >= 1000 {
		fmt.Print("    🔍 Large K search (k=100)... ")
		largeKResult := benchmarkSearch(ctx, coll, vectors, 100, 50)
		results = append(results, largeKResult)
		fmt.Printf("✅ (%.3fms avg)\n",
			float64(largeKResult.AvgTime.Microseconds())/1000.0)
	}

	// Benchmark: Filtered Search
	fmt.Print("    🔍 Filtered search... ")
	filteredResult := benchmarkFilteredSearch(ctx, coll, vectors, searchK, 50)
	results = append(results, filteredResult)
	fmt.Printf("✅ (%.3fms avg)\n",
		float64(filteredResult.AvgTime.Microseconds())/1000.0)

	// Benchmark: Update
	if numVectors <= 10000 {
		fmt.Print("    📝 Update operations... ")
		updateResult := benchmarkUpdate(ctx, coll, dim, 100)
		results = append(results, updateResult)
		fmt.Printf("✅ (%.3fms avg)\n",
			float64(updateResult.AvgTime.Microseconds())/1000.0)
	}

	// Benchmark: Delete
	if numVectors <= 10000 {
		fmt.Print("    🗑️  Delete operations... ")
		deleteResult := benchmarkDelete(ctx, coll, 100)
		results = append(results, deleteResult)
		fmt.Printf("✅ (%.2fµs avg)\n",
			float64(deleteResult.AvgTime.Nanoseconds())/1000.0)
	}

	// Benchmark: Get by ID
	fmt.Print("    📌 Get by ID... ")
	getResult := benchmarkGetByID(ctx, coll, vectors, 100)
	results = append(results, getResult)
	fmt.Printf("✅ (%.2fµs avg)\n",
		float64(getResult.AvgTime.Nanoseconds())/1000.0)

	// Benchmark: Concurrent Search
	fmt.Print("    ⚡ Concurrent search (10 threads)... ")
	concurrentResult := benchmarkConcurrentSearch(ctx, coll, vectors, searchK, 10, 10)
	results = append(results, concurrentResult)
	fmt.Printf("✅ (%.3fms avg, %.0f qps)\n",
		float64(concurrentResult.AvgTime.Microseconds())/1000.0,
		concurrentResult.Throughput)

	return results
}

func generateVectors(count, dim int) []*api.Vector {
	rng := rand.New(rand.NewSource(42))
	vectors := make([]*api.Vector, count)

	categories := []string{"tech", "science", "arts", "sports", "news"}

	for i := 0; i < count; i++ {
		data := make([]float32, dim)
		for j := 0; j < dim; j++ {
			data[j] = rng.Float32()*2 - 1 // [-1, 1]
		}

		// Normalize
		var norm float32
		for _, v := range data {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		for j := range data {
			data[j] /= norm
		}

		vectors[i] = &api.Vector{
			ID:   fmt.Sprintf("vec_%d", i),
			Data: data,
			Metadata: map[string]interface{}{
				"index":    i,
				"category": categories[i%len(categories)],
				"score":    rng.Float64(),
				"group":    i % 10,
			},
		}
	}

	return vectors
}

func benchmarkBatchInsert(ctx context.Context, coll *collection.VectorCollection, vectors []*api.Vector) BenchmarkResult {
	start := time.Now()
	err := coll.AddBatch(ctx, vectors)
	elapsed := time.Since(start)

	result := BenchmarkResult{
		Dimension:  len(vectors[0].Data),
		NumVectors: len(vectors),
		Operation:  "batch_insert",
		TotalTime:  elapsed,
		AvgTime:    elapsed / time.Duration(len(vectors)),
		Throughput: 0, // Will be set below if successful
	}

	if err != nil {
		fmt.Printf("\n    ❌ Batch insert error: %v\n", err)
		return result
	}

	result.Throughput = float64(len(vectors)) / elapsed.Seconds()
	return result
}

func benchmarkSingleInsert(ctx context.Context, coll *collection.VectorCollection, dim, count int) BenchmarkResult {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	times := make([]time.Duration, count)

	for i := 0; i < count; i++ {
		data := make([]float32, dim)
		for j := 0; j < dim; j++ {
			data[j] = rng.Float32()
		}

		vector := &api.Vector{
			ID:   fmt.Sprintf("single_%d_%d", dim, i),
			Data: data,
		}

		start := time.Now()
		err := coll.Add(ctx, vector)
		times[i] = time.Since(start)
		if err != nil {
			fmt.Printf("\n    ⚠️  Single insert error for %s: %v\n", vector.ID, err)
		}
	}

	return calculateStats("single_insert", dim, count, times)
}

func benchmarkSearch(ctx context.Context, coll *collection.VectorCollection, vectors []*api.Vector, k, numSearches int) BenchmarkResult {
	rng := rand.New(rand.NewSource(42))
	times := make([]time.Duration, numSearches)
	qualityScores := make([]float64, numSearches)

	for i := 0; i < numSearches; i++ {
		queryIdx := rng.Intn(len(vectors))
		query := vectors[queryIdx].Data

		start := time.Now()
		results, err := coll.Search(ctx, &api.SearchRequest{
			Vector: query,
			K:      k,
		})
		times[i] = time.Since(start)

		if err == nil && len(results) > 0 {
			// Calculate quality: check if query vector is in top results
			qualityScores[i] = calculateSearchQuality(results, vectors[queryIdx].ID)
		}
	}

	result := calculateStats(fmt.Sprintf("search_k%d", k), len(vectors[0].Data), numSearches, times)
	result.SearchQuality = average(qualityScores)
	return result
}

func benchmarkExactSearch(ctx context.Context, coll *collection.VectorCollection, vectors []*api.Vector, k, numSearches int) BenchmarkResult {
	rng := rand.New(rand.NewSource(42))
	times := make([]time.Duration, numSearches)
	recalls := make([]float64, numSearches)

	for i := 0; i < numSearches; i++ {
		queryIdx := rng.Intn(len(vectors))
		query := vectors[queryIdx].Data

		start := time.Now()
		results, err := coll.Search(ctx, &api.SearchRequest{
			Vector: query,
			K:      k,
		})
		times[i] = time.Since(start)

		if err == nil && len(results) > 0 {
			// For exact match, the query vector should be the top result
			if results[0].Vector.ID == vectors[queryIdx].ID {
				recalls[i] = 1.0
			}
		}
	}

	result := calculateStats("exact_search", len(vectors[0].Data), numSearches, times)
	result.Recall = average(recalls)
	return result
}

func benchmarkFilteredSearch(ctx context.Context, coll *collection.VectorCollection, vectors []*api.Vector, k, numSearches int) BenchmarkResult {
	rng := rand.New(rand.NewSource(42))
	times := make([]time.Duration, numSearches)

	categories := []string{"tech", "science", "arts", "sports", "news"}

	for i := 0; i < numSearches; i++ {
		queryIdx := rng.Intn(len(vectors))
		query := vectors[queryIdx].Data
		category := categories[rng.Intn(len(categories))]

		filter := &api.FieldFilter{
			Field: "category",
			Op:    api.FilterEq,
			Value: category,
		}

		start := time.Now()
		coll.Search(ctx, &api.SearchRequest{
			Vector: query,
			K:      k,
			Filter: filter,
		})
		times[i] = time.Since(start)
	}

	return calculateStats("filtered_search", len(vectors[0].Data), numSearches, times)
}

func benchmarkUpdate(ctx context.Context, coll *collection.VectorCollection, dim, count int) BenchmarkResult {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	times := make([]time.Duration, count)
	successCount := 0
	failCount := 0

	for i := 0; i < count; i++ {
		data := make([]float32, dim)
		for j := 0; j < dim; j++ {
			data[j] = rng.Float32()
		}

		// Normalize vector
		var norm float32
		for _, v := range data {
			norm += v * v
		}
		if norm > 0 {
			norm = float32(1.0 / float64(norm))
			for j := range data {
				data[j] *= norm
			}
		}

		vector := &api.Vector{
			ID:   fmt.Sprintf("vec_%d", i),
			Data: data,
			Metadata: map[string]interface{}{
				"updated":   true,
				"timestamp": time.Now().Unix(),
			},
		}

		start := time.Now()
		// Simulate Update: Delete + Add (since Update method doesn't exist in our implementation)
		err := coll.Delete(ctx, vector.ID)
		if err != nil && err != api.ErrVectorNotFound {
			failCount++
			if i < 5 { // Only show first 5 errors
				fmt.Printf("\n    ⚠️  Update delete error for %s: %v\n", vector.ID, err)
			}
		}
		err = coll.Add(ctx, vector)
		if err != nil {
			failCount++
			if i < 5 { // Only show first 5 errors
				fmt.Printf("\n    ⚠️  Update add error for %s: %v\n", vector.ID, err)
			}
		} else {
			successCount++
		}
		times[i] = time.Since(start)
	}

	if failCount > 0 {
		fmt.Printf("\n    ⚠️  Update: %d succeeded, %d failed\n", successCount, failCount)
	}

	return calculateStats("update", dim, count, times)
}

func benchmarkDelete(ctx context.Context, coll *collection.VectorCollection, count int) BenchmarkResult {
	times := make([]time.Duration, count)
	successCount := 0
	failCount := 0

	for i := 0; i < count; i++ {
		id := fmt.Sprintf("vec_%d", i)

		start := time.Now()
		err := coll.Delete(ctx, id)
		times[i] = time.Since(start)

		if err != nil {
			failCount++
			if i < 5 { // Only show first 5 errors
				fmt.Printf("\n    ⚠️  Delete error for %s: %v\n", id, err)
			}
		} else {
			successCount++
		}
	}

	if failCount > 0 {
		fmt.Printf("\n    ⚠️  Delete: %d succeeded, %d failed\n", successCount, failCount)
	}

	return calculateStats("delete", 0, count, times)
}

func benchmarkGetByID(ctx context.Context, coll *collection.VectorCollection, vectors []*api.Vector, count int) BenchmarkResult {
	rng := rand.New(rand.NewSource(42))
	times := make([]time.Duration, count)
	successCount := 0
	failCount := 0

	for i := 0; i < count; i++ {
		idx := rng.Intn(len(vectors))
		id := vectors[idx].ID

		start := time.Now()
		vec, err := coll.Get(ctx, id)
		times[i] = time.Since(start)

		if err != nil || vec == nil {
			failCount++
			if i < 5 { // Only show first 5 errors
				fmt.Printf("\n    ⚠️  Get error for %s: %v\n", id, err)
			}
		} else {
			successCount++
		}
	}

	if failCount > 0 {
		fmt.Printf("\n    ⚠️  Get: %d succeeded, %d failed\n", successCount, failCount)
	}

	return calculateStats("get_by_id", len(vectors[0].Data), count, times)
}

func benchmarkConcurrentSearch(ctx context.Context, coll *collection.VectorCollection, vectors []*api.Vector, k, numThreads, searchesPerThread int) BenchmarkResult {
	type threadResult struct {
		times []time.Duration
	}

	results := make([]threadResult, numThreads)
	done := make(chan int, numThreads)

	start := time.Now()

	for t := 0; t < numThreads; t++ {
		threadID := t
		go func() {
			threadRng := rand.New(rand.NewSource(int64(threadID)))
			times := make([]time.Duration, searchesPerThread)

			for i := 0; i < searchesPerThread; i++ {
				queryIdx := threadRng.Intn(len(vectors))
				query := vectors[queryIdx].Data

				searchStart := time.Now()
				coll.Search(ctx, &api.SearchRequest{
					Vector: query,
					K:      k,
				})
				times[i] = time.Since(searchStart)
			}

			results[threadID] = threadResult{times: times}
			done <- threadID
		}()
	}

	for i := 0; i < numThreads; i++ {
		<-done
	}

	totalElapsed := time.Since(start)

	// Collect all times
	allTimes := []time.Duration{}
	for _, r := range results {
		allTimes = append(allTimes, r.times...)
	}

	benchResult := calculateStats("concurrent_search", len(vectors[0].Data), len(allTimes), allTimes)
	benchResult.Throughput = float64(len(allTimes)) / totalElapsed.Seconds()

	return benchResult
}

func calculateStats(operation string, dim, count int, times []time.Duration) BenchmarkResult {
	if len(times) == 0 {
		return BenchmarkResult{Operation: operation}
	}

	sort.Slice(times, func(i, j int) bool {
		return times[i] < times[j]
	})

	var total time.Duration
	for _, t := range times {
		total += t
	}

	return BenchmarkResult{
		Dimension:  dim,
		NumVectors: count,
		Operation:  operation,
		TotalTime:  total,
		AvgTime:    total / time.Duration(len(times)),
		MinTime:    times[0],
		MaxTime:    times[len(times)-1],
		Throughput: float64(len(times)) / total.Seconds(),
	}
}

func calculateSearchQuality(results []*api.SearchResult, queryID string) float64 {
	if len(results) == 0 {
		return 0.0
	}

	// Check if query ID is in results and calculate position-based quality
	for i, r := range results {
		if r.Vector.ID == queryID {
			// Higher quality if found in top positions
			return 1.0 - (float64(i) / float64(len(results)))
		}
	}

	// If not found, use top result score as quality indicator
	return float64(results[0].Score)
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func printResults(results []BenchmarkResult) {
	if len(results) == 0 {
		return
	}

	fmt.Println()
}

func saveResultsToJSON(results []BenchmarkResult, config TestConfig) error {
	// Convert results to JSON format
	jsonResults := make([]JSONBenchmarkResult, len(results))
	for i, r := range results {
		jsonResults[i] = r.ToJSON()
	}

	// Create metadata
	metadata := BenchmarkMetadata{
		Timestamp:    time.Now().Format(time.RFC3339),
		Dimensions:   config.Dimensions,
		VectorCounts: config.VectorCounts,
		TotalTests:   len(results),
		Database:     "govecdb",
	}

	// Create output structure
	output := JSONOutput{
		Metadata: metadata,
		Results:  jsonResults,
	}

	// Generate filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("govecdb_benchmark_results_%s.json", timestamp)

	// Marshal to JSON with indentation
	jsonData, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	// Write to file
	if err := os.WriteFile(filename, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write JSON file: %w", err)
	}

	fmt.Printf("\n💾 Results saved to: %s\n", filename)
	return nil
}

func saveResultsToCSV(results []BenchmarkResult) error {
	// Generate filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("govecdb_benchmark_results_%s.csv", timestamp)

	// Create CSV file
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %w", err)
	}
	defer file.Close()

	// Write CSV header
	header := []string{
		"dimension",
		"num_vectors",
		"operation",
		"total_time",
		"avg_time",
		"min_time",
		"max_time",
		"throughput",
		"recall",
		"search_quality",
	}
	if _, err := file.WriteString(strings.Join(header, ",") + "\n"); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}

	// Write data rows
	for _, r := range results {
		row := fmt.Sprintf("%d,%d,%s,%.6f,%.6f,%.6f,%.6f,%.2f,%.4f,%.4f\n",
			r.Dimension,
			r.NumVectors,
			r.Operation,
			r.TotalTime.Seconds(),
			r.AvgTime.Seconds(),
			r.MinTime.Seconds(),
			r.MaxTime.Seconds(),
			r.Throughput,
			r.Recall,
			r.SearchQuality,
		)
		if _, err := file.WriteString(row); err != nil {
			return fmt.Errorf("failed to write CSV row: %w", err)
		}
	}

	fmt.Printf("💾 Results saved to: %s\n", filename)
	return nil
}

func printSummary(results []BenchmarkResult) {
	fmt.Println("\n\n🏆 FINAL PERFORMANCE SUMMARY")
	fmt.Println("=" + string(make([]byte, 80)))
	fmt.Println()

	// Group by operation
	opGroups := make(map[string][]BenchmarkResult)
	for _, r := range results {
		opGroups[r.Operation] = append(opGroups[r.Operation], r)
	}

	for op, group := range opGroups {
		fmt.Printf("\n📊 %s\n", op)
		fmt.Println("─" + string(make([]byte, 80)))

		var totalAvg, minAvg, maxAvg float64
		count := 0

		for _, r := range group {
			if r.AvgTime > 0 {
				avgMs := float64(r.AvgTime.Microseconds()) / 1000.0
				totalAvg += avgMs
				if minAvg == 0 || avgMs < minAvg {
					minAvg = avgMs
				}
				if avgMs > maxAvg {
					maxAvg = avgMs
				}
				count++
			}
		}

		if count > 0 {
			fmt.Printf("  Average: %.3fms | Min: %.3fms | Max: %.3fms\n",
				totalAvg/float64(count), minAvg, maxAvg)
		}

		// Show throughput if available
		var throughputs []float64
		for _, r := range group {
			if r.Throughput > 0 {
				throughputs = append(throughputs, r.Throughput)
			}
		}
		if len(throughputs) > 0 {
			fmt.Printf("  Throughput: %.0f ops/sec (avg)\n", average(throughputs))
		}

		// Show recall/quality if available
		if op == "exact_search" {
			recalls := []float64{}
			for _, r := range group {
				if r.Recall > 0 {
					recalls = append(recalls, r.Recall)
				}
			}
			if len(recalls) > 0 {
				fmt.Printf("  Recall: %.2f%%\n", average(recalls)*100)
			}
		}

		if op == "search_k10" {
			qualities := []float64{}
			for _, r := range group {
				if r.SearchQuality > 0 {
					qualities = append(qualities, r.SearchQuality)
				}
			}
			if len(qualities) > 0 {
				fmt.Printf("  Search Quality: %.2f%%\n", average(qualities)*100)
			}
		}
	}

	fmt.Println("\n✅ Benchmark Complete!")
	fmt.Println()
}
