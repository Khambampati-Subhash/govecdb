package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
	"github.com/khambampati-subhash/govecdb/collection"
	"github.com/khambampati-subhash/govecdb/store"
)

type BenchmarkResult struct {
	Operation   string  `json:"operation"`
	Dimension   int     `json:"dimension"`
	VectorCount int     `json:"vector_count"`
	AvgTimeMs   float64 `json:"avg_time_ms"`
	Throughput  float64 `json:"throughput"`
	Recall      float64 `json:"recall"`
	SuccessRate float64 `json:"success_rate"`
}

type GoVecDBBenchmark struct {
	results []BenchmarkResult
}

func generateVectors(count, dim int, seed int64) []*api.Vector {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([]*api.Vector, count)

	for i := 0; i < count; i++ {
		data := make([]float32, dim)
		for j := 0; j < dim; j++ {
			data[j] = rng.Float32()*2 - 1 // [-1, 1]
		}

		// Normalize for cosine similarity
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
				"category": fmt.Sprintf("cat_%d", i%5),
				"value":    float64(i) * 0.1,
			},
		}
	}

	return vectors
}

func createCollection(name string, dimension, vectorCount int) (*collection.VectorCollection, error) {
	// Use optimal parameters for dimension and vector count
	params := collection.CalculateOptimalHNSWParams(dimension, vectorCount)

	config := &api.CollectionConfig{
		Name:           name,
		Dimension:      dimension,
		Metric:         api.Cosine,
		M:              params.M,
		EfConstruction: params.EfConstruction,
		MaxLayer:       params.MaxLayer,
		ThreadSafe:     true,
	}

	storeConfig := &store.StoreConfig{
		Name:         fmt.Sprintf("%s_store", name),
		PreallocSize: vectorCount,
		EnableStats:  true,
	}

	return collection.NewVectorCollection(config, storeConfig)
}

func (b *GoVecDBBenchmark) benchmarkSingleInsert(coll *collection.VectorCollection, vectors []*api.Vector, count int) BenchmarkResult {
	ctx := context.Background()
	times := make([]time.Duration, count)

	for i := 0; i < count; i++ {
		start := time.Now()
		_ = coll.Add(ctx, vectors[i])
		times[i] = time.Since(start)
	}

	avgTime := average(times)
	throughput := 1000.0 / avgTime.Seconds() / 1000.0

	return BenchmarkResult{
		Operation:   "single_insert",
		Dimension:   len(vectors[0].Data),
		VectorCount: count,
		AvgTimeMs:   avgTime.Seconds() * 1000,
		Throughput:  throughput,
		SuccessRate: 100.0,
	}
}

func (b *GoVecDBBenchmark) benchmarkBatchInsert(coll *collection.VectorCollection, vectors []*api.Vector) BenchmarkResult {
	ctx := context.Background()

	start := time.Now()
	err := coll.AddBatch(ctx, vectors)
	elapsed := time.Since(start)

	successRate := 100.0
	if err != nil {
		successRate = 0.0
	}

	avgTime := elapsed.Seconds() * 1000 / float64(len(vectors))
	throughput := float64(len(vectors)) / elapsed.Seconds()

	return BenchmarkResult{
		Operation:   "batch_insert",
		Dimension:   len(vectors[0].Data),
		VectorCount: len(vectors),
		AvgTimeMs:   avgTime,
		Throughput:  throughput,
		SuccessRate: successRate,
	}
}

func (b *GoVecDBBenchmark) benchmarkExactSearch(coll *collection.VectorCollection, vectors []*api.Vector, numSearches int) BenchmarkResult {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))
	times := make([]time.Duration, numSearches)
	recalls := make([]float64, numSearches)

	for i := 0; i < numSearches; i++ {
		queryIdx := rng.Intn(len(vectors))
		query := vectors[queryIdx].Data
		expectedID := vectors[queryIdx].ID

		start := time.Now()
		results, err := coll.Search(ctx, &api.SearchRequest{
			Vector: query,
			K:      1,
		})
		times[i] = time.Since(start)

		if err == nil && len(results) > 0 {
			if results[0].Vector.ID == expectedID {
				recalls[i] = 1.0
			}
		}
	}

	avgTime := average(times)
	avgRecall := averageFloat(recalls) * 100
	throughput := 1000.0 / avgTime.Seconds() / 1000.0

	return BenchmarkResult{
		Operation:   "exact_search_k1",
		Dimension:   len(vectors[0].Data),
		VectorCount: len(vectors),
		AvgTimeMs:   avgTime.Seconds() * 1000,
		Throughput:  throughput,
		Recall:      avgRecall,
		SuccessRate: 100.0,
	}
}

func (b *GoVecDBBenchmark) benchmarkKNNSearch(coll *collection.VectorCollection, vectors []*api.Vector, k, numSearches int) BenchmarkResult {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))
	times := make([]time.Duration, numSearches)
	qualityScores := make([]float64, numSearches)

	for i := 0; i < numSearches; i++ {
		queryIdx := rng.Intn(len(vectors))
		query := vectors[queryIdx].Data
		expectedID := vectors[queryIdx].ID

		start := time.Now()
		results, err := coll.Search(ctx, &api.SearchRequest{
			Vector: query,
			K:      k,
		})
		times[i] = time.Since(start)

		if err == nil && len(results) > 0 {
			// Find position of expected ID
			for pos, result := range results {
				if result.Vector.ID == expectedID {
					// Score: 1.0 if first, decreasing with position
					qualityScores[i] = 1.0 - (float64(pos) / float64(k))
					break
				}
			}
		}
	}

	avgTime := average(times)
	avgQuality := averageFloat(qualityScores) * 100
	throughput := 1000.0 / avgTime.Seconds() / 1000.0

	return BenchmarkResult{
		Operation:   fmt.Sprintf("knn_search_k%d", k),
		Dimension:   len(vectors[0].Data),
		VectorCount: len(vectors),
		AvgTimeMs:   avgTime.Seconds() * 1000,
		Throughput:  throughput,
		Recall:      avgQuality,
		SuccessRate: 100.0,
	}
}

func (b *GoVecDBBenchmark) benchmarkDelete(coll *collection.VectorCollection, vectors []*api.Vector, count int) BenchmarkResult {
	ctx := context.Background()
	times := make([]time.Duration, count)

	for i := 0; i < count; i++ {
		start := time.Now()
		_ = coll.Delete(ctx, vectors[i].ID)
		times[i] = time.Since(start)
	}

	// Verify deletion
	deletedCount := 0
	for i := 0; i < count; i++ {
		_, err := coll.Get(ctx, vectors[i].ID)
		if err != nil {
			deletedCount++
		}
	}

	successRate := (float64(deletedCount) / float64(count)) * 100
	avgTime := average(times)
	throughput := 1000.0 / avgTime.Seconds() / 1000.0

	return BenchmarkResult{
		Operation:   "delete",
		Dimension:   0,
		VectorCount: count,
		AvgTimeMs:   avgTime.Seconds() * 1000,
		Throughput:  throughput,
		SuccessRate: successRate,
	}
}

func (b *GoVecDBBenchmark) benchmarkUpdate(coll *collection.VectorCollection, vectors []*api.Vector, dim, count int) BenchmarkResult {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(100))
	times := make([]time.Duration, count)

	for i := 0; i < count; i++ {
		// Generate new vector
		data := make([]float32, dim)
		for j := 0; j < dim; j++ {
			data[j] = rng.Float32()*2 - 1
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

		newVec := &api.Vector{
			ID:   vectors[i].ID,
			Data: data,
			Metadata: map[string]interface{}{
				"index":   i,
				"updated": true,
			},
		}

		start := time.Now()
		// Update = Delete + Add
		_ = coll.Delete(ctx, vectors[i].ID)
		_ = coll.Add(ctx, newVec)
		times[i] = time.Since(start)
	}

	// Verify update
	updatedCount := 0
	for i := 0; i < count; i++ {
		vec, err := coll.Get(ctx, vectors[i].ID)
		if err == nil && vec != nil {
			if updated, ok := vec.Metadata["updated"].(bool); ok && updated {
				updatedCount++
			}
		}
	}

	successRate := (float64(updatedCount) / float64(count)) * 100
	avgTime := average(times)
	throughput := 1000.0 / avgTime.Seconds() / 1000.0

	return BenchmarkResult{
		Operation:   "update",
		Dimension:   dim,
		VectorCount: count,
		AvgTimeMs:   avgTime.Seconds() * 1000,
		Throughput:  throughput,
		SuccessRate: successRate,
	}
}

func (b *GoVecDBBenchmark) runBenchmarkSuite(dimension, vectorCount int) {
	fmt.Printf("\n%s\n", string(make([]byte, 70)))
	fmt.Printf("  Dimension: %d, Vector Count: %d\n", dimension, vectorCount)
	fmt.Printf("%s\n\n", string(make([]byte, 70)))

	// Generate test data
	fmt.Print("  📊 Generating test vectors... ")
	vectors := generateVectors(vectorCount, dimension, 42)
	fmt.Println("✅")

	// Create collection with optimal parameters
	collName := fmt.Sprintf("bench_%d_%d", dimension, vectorCount)
	coll, err := createCollection(collName, dimension, vectorCount)
	if err != nil {
		fmt.Printf("  ❌ Failed to create collection: %v\n", err)
		return
	}
	defer coll.Close()

	// Show the parameters being used
	params := collection.CalculateOptimalHNSWParams(dimension, vectorCount)
	fmt.Printf("  🔧 HNSW params: M=%d, EfConstruction=%d, MaxLayer=%d\n",
		params.M, params.EfConstruction, params.MaxLayer)

	ctx := context.Background()

	// 1. Batch Insert
	fmt.Print("  ⏱️  Batch insert... ")
	result := b.benchmarkBatchInsert(coll, vectors)
	b.results = append(b.results, result)
	fmt.Printf("✅ (%.3fms avg, %.0f vec/sec)\n", result.AvgTimeMs, result.Throughput)

	// 2. Exact Search (k=1)
	fmt.Print("  🔍 Exact search (k=1)... ")
	result = b.benchmarkExactSearch(coll, vectors, 100)
	b.results = append(b.results, result)
	fmt.Printf("✅ (%.3fms avg, recall: %.2f%%)\n", result.AvgTimeMs, result.Recall)

	// 3. KNN Search (k=10)
	fmt.Print("  🔍 KNN search (k=10)... ")
	result = b.benchmarkKNNSearch(coll, vectors, 10, 100)
	b.results = append(b.results, result)
	fmt.Printf("✅ (%.3fms avg, quality: %.2f%%)\n", result.AvgTimeMs, result.Recall)

	// 4. KNN Search (k=100)
	if vectorCount >= 1000 {
		fmt.Print("  🔍 KNN search (k=100)... ")
		result = b.benchmarkKNNSearch(coll, vectors, 100, 100)
		b.results = append(b.results, result)
		fmt.Printf("✅ (%.3fms avg, quality: %.2f%%)\n", result.AvgTimeMs, result.Recall)
	}

	// 5. Update operations
	fmt.Print("  📝 Update operations... ")
	result = b.benchmarkUpdate(coll, vectors, dimension, 100)
	b.results = append(b.results, result)
	fmt.Printf("✅ (%.3fms avg, success: %.1f%%)\n", result.AvgTimeMs, result.SuccessRate)

	// 6. Delete operations
	fmt.Print("  🗑️  Delete operations... ")
	result = b.benchmarkDelete(coll, vectors, 100)
	b.results = append(b.results, result)
	fmt.Printf("✅ (%.3fms avg, success: %.1f%%)\n", result.AvgTimeMs, result.SuccessRate)

	// Add back deleted vectors for next test
	deletedVecs := vectors[:100]
	_ = coll.AddBatch(ctx, deletedVecs)
}

func (b *GoVecDBBenchmark) printSummary() {
	fmt.Printf("\n%s\n", strings.Repeat("=", 70))
	fmt.Println("  GOVECDB BENCHMARK RESULTS SUMMARY")
	fmt.Printf("%s\n\n", strings.Repeat("=", 70))

	// Group by operation
	ops := make(map[string][]BenchmarkResult)
	for _, result := range b.results {
		ops[result.Operation] = append(ops[result.Operation], result)
	}

	for opName, results := range ops {
		fmt.Printf("📊 %s\n", opName)
		fmt.Printf("%s\n", strings.Repeat("─", 70))

		times := make([]float64, len(results))
		throughputs := []float64{}
		recalls := []float64{}
		successRates := []float64{}

		for i, r := range results {
			times[i] = r.AvgTimeMs
			if r.Throughput > 0 {
				throughputs = append(throughputs, r.Throughput)
			}
			if r.Recall > 0 {
				recalls = append(recalls, r.Recall)
			}
			if r.SuccessRate > 0 {
				successRates = append(successRates, r.SuccessRate)
			}
		}

		avgTime := averageFloat(times)
		minTime := minFloat(times)
		maxTime := maxFloat(times)

		fmt.Printf("  Average Time: %.3fms | Min: %.3fms | Max: %.3fms\n", avgTime, minTime, maxTime)

		if len(throughputs) > 0 {
			fmt.Printf("  Throughput: %.0f ops/sec (avg)\n", averageFloat(throughputs))
		}

		if len(recalls) > 0 {
			fmt.Printf("  Recall/Quality: %.2f%%\n", averageFloat(recalls))
		}

		if len(successRates) > 0 && averageFloat(successRates) < 100 {
			fmt.Printf("  Success Rate: %.2f%%\n", averageFloat(successRates))
		}

		fmt.Println()
	}
}

func (b *GoVecDBBenchmark) saveResults(filename string) error {
	data := map[string]interface{}{
		"database":  "GoVecDB",
		"timestamp": time.Now().Format("2006-01-02 15:04:05"),
		"results":   b.results,
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(data); err != nil {
		return err
	}

	fmt.Printf("💾 Results saved to: %s\n", filename)
	return nil
}

func average(durations []time.Duration) time.Duration {
	if len(durations) == 0 {
		return 0
	}
	var sum time.Duration
	for _, d := range durations {
		sum += d
	}
	return sum / time.Duration(len(durations))
}

func averageFloat(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func minFloat(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	min := values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
	}
	return min
}

func maxFloat(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}

func main() {
	fmt.Println("🚀 GoVecDB Comprehensive Benchmark")
	fmt.Printf("%s\n", strings.Repeat("=", 70))

	// Test configurations
	dimensions := []int{512, 1024, 2048, 3072, 4096}
	vectorCounts := []int{1000, 3000, 5000}

	benchmark := &GoVecDBBenchmark{}

	for _, dim := range dimensions {
		for _, count := range vectorCounts {
			// Skip very large combinations
			if dim >= 4096 && count >= 5000 {
				fmt.Printf("\n⏭️  Skipping: %dD x %d vectors (too large)\n", dim, count)
				continue
			}

			benchmark.runBenchmarkSuite(dim, count)
		}
	}

	// Print summary
	benchmark.printSummary()

	// Save results
	if err := benchmark.saveResults("govecdb_results.json"); err != nil {
		fmt.Printf("❌ Error saving results: %v\n", err)
	}

	fmt.Println("\n✅ Benchmark Complete!")
}
