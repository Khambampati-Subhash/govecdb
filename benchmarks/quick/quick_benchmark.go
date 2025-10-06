package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
	"github.com/khambampati-subhash/govecdb/collection"
	"github.com/khambampati-subhash/govecdb/store"
)

// BenchmarkResult1 stores performance metrics
type BenchmarkResult1 struct {
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

// JSONBenchmarkResult11 is the JSON-serializable version of BenchmarkResult1
type JSONBenchmarkResult11 struct {
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

func main() {
	fmt.Println("🚀 GoVecDB Quick Test Benchmark")
	fmt.Println("===============================")
	startTime := time.Now()
	fmt.Printf("⏰ Started: %s\n", startTime.Format("2006-01-02 15:04:05"))
	fmt.Println()

	// Quick test with smaller dimensions and vector counts
	testConfigs := []struct {
		dimension int
		vectors   int
	}{
		{128, 1000},
		{256, 2000},
		{512, 1000},
	}

	results := []BenchmarkResult1{}

	for _, cfg := range testConfigs {
		fmt.Printf("📊 Testing %dD vectors with %d vectors\n", cfg.dimension, cfg.vectors)
		result := runQuickBenchmark(cfg.dimension, cfg.vectors)
		results = append(results, result...)
	}

	// Save results
	saveResultsToJSON(results)
	fmt.Printf("\n⏰ Completed: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Printf("⏱️  Total Duration: %s\n", time.Since(startTime).Round(time.Second))
}

func runQuickBenchmark(dim, numVectors int) []BenchmarkResult1 {
	results := []BenchmarkResult1{}

	// Create collection
	collConfig := &api.CollectionConfig{
		Name:           fmt.Sprintf("test_%d_%d", dim, numVectors),
		Dimension:      dim,
		Metric:         api.Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		ThreadSafe:     true,
	}

	storeConfig := &store.StoreConfig{
		Name:         fmt.Sprintf("test_store_%d", dim),
		PreallocSize: numVectors,
		EnableStats:  true,
	}

	coll, err := collection.NewVectorCollection(collConfig, storeConfig)
	if err != nil {
		fmt.Printf("❌ Failed to create collection: %v\n", err)
		return results
	}
	defer coll.Close()

	ctx := context.Background()

	// Generate test vectors
	fmt.Print("  🎲 Generating vectors... ")
	vectors := generateVectors(numVectors, dim)
	fmt.Println("✅")

	// Test Batch Insert
	fmt.Print("  📦 Batch insert... ")
	start := time.Now()
	err = coll.AddBatch(ctx, vectors)
	elapsed := time.Since(start)

	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		return results
	}

	throughput := float64(numVectors) / elapsed.Seconds()
	fmt.Printf("✅ (%.0f vec/sec)\n", throughput)

	results = append(results, BenchmarkResult1{
		Dimension:  dim,
		NumVectors: numVectors,
		Operation:  "batch_insert",
		TotalTime:  elapsed,
		AvgTime:    elapsed / time.Duration(numVectors),
		Throughput: throughput,
	})

	// Test Search
	fmt.Print("  🔍 Search test... ")
	searchTimes := []time.Duration{}
	for i := 0; i < 10; i++ {
		query := vectors[i].Data
		start := time.Now()
		_, err := coll.Search(ctx, &api.SearchRequest{
			Vector: query,
			K:      10,
		})
		if err != nil {
			fmt.Printf("❌ Search error: %v\n", err)
			continue
		}
		searchTimes = append(searchTimes, time.Since(start))
	}

	if len(searchTimes) > 0 {
		var totalSearch time.Duration
		for _, t := range searchTimes {
			totalSearch += t
		}
		avgSearch := totalSearch / time.Duration(len(searchTimes))
		fmt.Printf("✅ (%.2fms avg)\n", float64(avgSearch.Microseconds())/1000.0)

		results = append(results, BenchmarkResult1{
			Dimension:  dim,
			NumVectors: len(searchTimes),
			Operation:  "search",
			AvgTime:    avgSearch,
			Throughput: float64(len(searchTimes)) / totalSearch.Seconds(),
		})
	}

	return results
}

func generateVectors(count, dim int) []*api.Vector {
	rng := rand.New(rand.NewSource(42))
	vectors := make([]*api.Vector, count)

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
		if norm > 0 {
			for j := range data {
				data[j] /= norm
			}
		}

		vectors[i] = &api.Vector{
			ID:   fmt.Sprintf("vec_%d", i),
			Data: data,
			Metadata: map[string]interface{}{
				"index": i,
			},
		}
	}

	return vectors
}

func saveResultsToJSON(results []BenchmarkResult1) error {
	// Convert results to JSON format
	jsonResults := make([]JSONBenchmarkResult11, len(results))
	for i, r := range results {
		jsonResults[i] = JSONBenchmarkResult11{
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

	// Generate filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("govecdb_quick_benchmark_%s.json", timestamp)

	// Marshal to JSON with indentation
	jsonData, err := json.MarshalIndent(jsonResults, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	// Write to file
	if err := os.WriteFile(filename, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write JSON file: %w", err)
	}

	fmt.Printf("💾 Results saved to: %s\n", filename)
	return nil
}
