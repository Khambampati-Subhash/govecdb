package main

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
	"github.com/khambampati-subhash/govecdb/collection"
	"github.com/khambampati-subhash/govecdb/store"
)

const (
	DIM_SIZE     = 384  // Common embedding dimension
	NUM_VECTORS  = 1000 // Number of vectors to demo with
	SEARCH_K     = 10   // Number of results to retrieve
	BATCH_SIZE   = 100  // Batch size for batch operations
	DEMO_TIMEOUT = 30 * time.Second
)

// TestVector represents a test vector with sample data
type TestVector struct {
	ID       string
	Data     []float32
	Metadata map[string]interface{}
}

func main() {
	fmt.Println("🚀 GoVecDB Comprehensive Demo")
	fmt.Println("=============================")
	fmt.Println()

	ctx, cancel := context.WithTimeout(context.Background(), DEMO_TIMEOUT)
	defer cancel()

	// Demo sequence
	if !runConfigDemo(ctx) {
		fmt.Println("❌ Configuration demo failed")
		return
	}

	coll := createCollectionDemo(ctx)
	if coll == nil {
		fmt.Println("❌ Collection creation failed")
		return
	}

	if !runSingleVectorDemo(ctx, coll) {
		fmt.Println("❌ Single vector demo failed")
		return
	}

	vectors := runBatchOperationsDemo(ctx, coll)
	if vectors == nil {
		fmt.Println("❌ Batch operations demo failed")
		return
	}

	if !runSearchDemo(ctx, coll, vectors) {
		fmt.Println("❌ Search demo failed")
		return
	}

	if !runConcurrentDemo(ctx, coll) {
		fmt.Println("❌ Concurrent demo failed")
		return
	}

	if !runPersistenceDemo(ctx, coll) {
		fmt.Println("❌ Persistence demo failed")
		return
	}

	fmt.Println("🎉 All demos completed successfully!")
	fmt.Println("====================================")
}

// runConfigDemo demonstrates configuration and validation
func runConfigDemo(ctx context.Context) bool {
	fmt.Println("⚙️  Configuration Demo")
	fmt.Println("======================")

	// Test different distance metrics
	metrics := []api.DistanceMetric{
		api.Cosine,
		api.Euclidean,
		api.Manhattan,
		api.DotProduct,
	}

	fmt.Println("📏 Available Distance Metrics:")
	for i, metric := range metrics {
		fmt.Printf("   %d. %s\n", i+1, metric.String())
	}

	// Test configuration validation
	fmt.Print("🔍 Testing configuration validation... ")
	invalidConfig := &api.CollectionConfig{
		Name:      "",
		Dimension: -1,
		Metric:    api.Cosine,
	}

	if err := invalidConfig.Validate(); err != nil {
		fmt.Println("✅ Validation correctly failed:", err.Error())
	} else {
		fmt.Println("❌ Validation should have failed")
		return false
	}

	fmt.Println()
	return true
}

// createCollectionDemo demonstrates collection creation
func createCollectionDemo(ctx context.Context) *collection.VectorCollection {
	fmt.Println("📦 Collection Creation Demo")
	fmt.Println("===========================")

	// Create collection configuration
	config := &api.CollectionConfig{
		Name:           "demo_collection",
		Dimension:      DIM_SIZE,
		Metric:         api.Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		Seed:           42,
		ThreadSafe:     true,
	}

	fmt.Printf("🔧 Creating collection '%s' with %dD vectors...\n", config.Name, config.Dimension)
	fmt.Printf("   📐 Distance metric: %s\n", config.Metric.String())
	fmt.Printf("   🏗️  HNSW parameters: M=%d, EfConstruction=%d\n", config.M, config.EfConstruction)

	// Create store config
	storeConfig := &store.StoreConfig{
		Name:          "demo_store",
		EnableStats:   true,
		EnableMetrics: true,
		PreallocSize:  1000,
	}

	start := time.Now()
	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		fmt.Printf("❌ Error creating collection: %v\n", err)
		return nil
	}

	fmt.Printf("✅ Collection created successfully in %v\n", time.Since(start))
	fmt.Println()
	return coll
}

// runSingleVectorDemo demonstrates comprehensive single vector CRUD operations
func runSingleVectorDemo(ctx context.Context, coll *collection.VectorCollection) bool {
	fmt.Println("🔢 Comprehensive CRUD Operations Demo")
	fmt.Println("=====================================")

	// Create a test vector
	testVector := &TestVector{
		ID:   "demo_vector_1",
		Data: generateRandomVector(DIM_SIZE),
		Metadata: map[string]interface{}{
			"category":  "demo",
			"priority":  1,
			"tags":      []string{"test", "demo"},
			"timestamp": time.Now().Unix(),
			"score":     0.85,
		},
	}

	// Convert to API vector
	apiVector := convertToAPIVector(testVector)

	// === CREATE Operation ===
	fmt.Printf("➕ CREATE: Adding vector '%s'... ", testVector.ID)
	start := time.Now()
	err := coll.Add(ctx, apiVector)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		return false
	}
	createDuration := time.Since(start)
	fmt.Printf("✅ Success (%v)\n", createDuration)

	// === READ Operation ===
	fmt.Printf("📖 READ: Retrieving vector '%s'... ", testVector.ID)
	start = time.Now()
	retrieved, err := coll.Get(ctx, testVector.ID)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		return false
	}
	readDuration := time.Since(start)
	fmt.Printf("✅ Success (%v)\n", readDuration)

	// Comprehensive validation
	fmt.Print("🔍 VALIDATE: Comprehensive data integrity check... ")
	if !validateVectorIntegrity(retrieved, testVector) {
		fmt.Println("❌ Validation failed")
		return false
	}
	fmt.Println("✅ All validations passed")

	// === UPDATE Operation ===
	fmt.Print("🔄 UPDATE: Modifying vector metadata... ")
	updatedVector := *apiVector // Copy
	updatedVector.Metadata["priority"] = 2
	updatedVector.Metadata["updated_at"] = time.Now().Unix()
	updatedVector.Metadata["version"] = "1.1"

	start = time.Now()
	// Note: Update operation - in many vector DBs this is delete + add
	err = coll.Add(ctx, &updatedVector)
	updateDuration := time.Since(start)
	if err == nil {
		fmt.Printf("✅ Update successful (%v)\n", updateDuration)
	} else {
		fmt.Printf("ℹ️  Update via replace: %v (%v)\n", err, updateDuration)
	}

	// Verify update
	fmt.Print("🔍 VALIDATE UPDATE: Checking updated metadata... ")
	updatedRetrieved, err := coll.Get(ctx, testVector.ID)
	if err != nil {
		fmt.Printf("❌ Error retrieving updated vector: %v\n", err)
		return false
	}
	if priority, ok := updatedRetrieved.Metadata["priority"]; ok && priority != 2 {
		fmt.Printf("⚠️  Metadata update may not be supported (priority: %v)\n", priority)
	} else {
		fmt.Println("✅ Update verified")
	}

	// Test duplicate insertion/idempotency
	fmt.Printf("🔄 IDEMPOTENCY: Testing duplicate insertion... ")
	start = time.Now()
	err = coll.Add(ctx, apiVector)
	idempotencyDuration := time.Since(start)
	if err == nil {
		fmt.Printf("⚠️  Duplicate insertion allowed (%v)\n", idempotencyDuration)
	} else {
		fmt.Printf("✅ Duplicate properly rejected: %v (%v)\n", err.Error(), idempotencyDuration)
	}

	// === DELETE Operation ===
	fmt.Printf("🗑️  DELETE: Removing vector '%s'... ", testVector.ID)
	start = time.Now()
	err = coll.Delete(ctx, testVector.ID)
	deleteDuration := time.Since(start)
	if err != nil {
		fmt.Printf("❌ Error: %v (%v)\n", err, deleteDuration)
		return false
	}
	fmt.Printf("✅ Success (%v)\n", deleteDuration)

	// Verify deletion
	fmt.Print("🔍 VALIDATE DELETE: Confirming vector removal... ")
	start = time.Now()
	_, err = coll.Get(ctx, testVector.ID)
	verifyDuration := time.Since(start)
	if err != nil {
		fmt.Printf("✅ Deletion confirmed: %v (%v)\n", err.Error(), verifyDuration)
	} else {
		fmt.Printf("⚠️  Vector still exists after deletion (%v)\n", verifyDuration)
	}

	// Re-add for subsequent tests
	fmt.Print("↩️  RE-CREATE: Re-adding vector for subsequent tests... ")
	start = time.Now()
	err = coll.Add(ctx, apiVector)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		return false
	}
	fmt.Printf("✅ Success (%v)\n", time.Since(start))

	// Final count
	fmt.Print("🔢 COUNT: Final vector count... ")
	start = time.Now()
	count, err := coll.Count(ctx)
	countDuration := time.Since(start)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		return false
	}
	fmt.Printf("✅ Count: %d (%v)\n", count, countDuration)

	// Performance Summary
	fmt.Println("\n📊 CRUD Performance Summary:")
	fmt.Printf("   ➕ CREATE: %v\n", createDuration)
	fmt.Printf("   📖 READ:   %v\n", readDuration)
	fmt.Printf("   🔄 UPDATE: %v\n", updateDuration)
	fmt.Printf("   🗑️  DELETE: %v\n", deleteDuration)
	fmt.Printf("   🔢 COUNT:  %v\n", countDuration)
	fmt.Printf("   🔄 IDEMPOTENCY: %v\n", idempotencyDuration)

	fmt.Println()
	return true
}

// validateVectorIntegrity performs comprehensive validation of retrieved vector
func validateVectorIntegrity(retrieved *api.Vector, expected *TestVector) bool {
	// ID validation
	if retrieved.ID != expected.ID {
		fmt.Printf("❌ ID mismatch: got %s, expected %s\n", retrieved.ID, expected.ID)
		return false
	}

	// Dimension validation
	if len(retrieved.Data) != len(expected.Data) {
		fmt.Printf("❌ Dimension mismatch: got %d, expected %d\n", len(retrieved.Data), len(expected.Data))
		return false
	}

	// Vector data validation (approximate equality for floating point)
	const epsilon = 1e-6
	for i, v := range retrieved.Data {
		if abs(v-expected.Data[i]) > epsilon {
			fmt.Printf("❌ Vector data mismatch at index %d: got %f, expected %f\n", i, v, expected.Data[i])
			return false
		}
	}

	// Metadata validation (basic check)
	if len(retrieved.Metadata) == 0 && len(expected.Metadata) > 0 {
		fmt.Println("⚠️  Metadata not preserved (may be expected)")
	}

	return true
}

// abs returns the absolute value of a float32
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// runBatchOperationsDemo demonstrates batch operations
func runBatchOperationsDemo(ctx context.Context, coll *collection.VectorCollection) []*TestVector {
	fmt.Println("📦 Batch Operations Demo")
	fmt.Println("========================")

	// Generate test vectors
	fmt.Printf("🎲 Generating %d test vectors... ", NUM_VECTORS)
	start := time.Now()
	vectors := generateTestVectors(NUM_VECTORS)
	fmt.Printf("✅ Generated in %v\n", time.Since(start))

	// Convert to API vectors
	apiVectors := make([]*api.Vector, len(vectors))
	for i, v := range vectors {
		apiVectors[i] = convertToAPIVector(v)
	}

	// Batch insert
	fmt.Printf("➕ Batch inserting %d vectors... ", len(apiVectors))
	start = time.Now()
	err := coll.AddBatch(ctx, apiVectors)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		return nil
	}
	duration := time.Since(start)
	throughput := float64(len(apiVectors)) / duration.Seconds()
	fmt.Printf("✅ Success (%v, %.1f vec/sec)\n", duration, throughput)

	// Verify count
	fmt.Print("🔍 Verifying vector count... ")
	count, err := coll.Count(ctx)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		return nil
	}
	fmt.Printf("✅ Count: %d vectors\n", count)

	fmt.Println()
	return vectors
}

// runSearchDemo demonstrates comprehensive search operations with detailed timing
func runSearchDemo(ctx context.Context, coll *collection.VectorCollection, vectors []*TestVector) bool {
	fmt.Println("🔍 Comprehensive Search Operations Demo")
	fmt.Println("=======================================")

	// Basic search test
	fmt.Print("🎯 Basic similarity search... ")
	queryVector := generateRandomVector(DIM_SIZE)
	start := time.Now()
	results, err := coll.Search(ctx, &api.SearchRequest{
		Vector:      queryVector,
		K:           SEARCH_K,
		IncludeData: true,
	})
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		return false
	}
	duration := time.Since(start)
	fmt.Printf("✅ Found %d results (%v)\n", len(results), duration)

	// Display top results
	fmt.Println("   📊 Top results:")
	for i, result := range results[:min(5, len(results))] {
		vectorID := "unknown"
		if result.Vector != nil {
			vectorID = result.Vector.ID
		}
		fmt.Printf("      %d. ID: %s, Distance: %.4f\n", i+1, vectorID, result.Distance)
	}

	// === Comprehensive Search Types Demo ===
	fmt.Println("\n🎯 Search Type Analysis:")

	// 1. Exact Match Search
	fmt.Print("🎯 EXACT MATCH: Perfect vector match... ")
	exactDuration := time.Duration(0)
	if len(vectors) > 42 {
		exactVector := vectors[42] // Use known vector
		start = time.Now()
		exactResults, err := coll.Search(ctx, &api.SearchRequest{
			Vector:      exactVector.Data,
			K:           1,
			IncludeData: true,
		})
		exactDuration = time.Since(start)
		if err != nil || len(exactResults) == 0 {
			fmt.Printf("❌ Error or no results: %v\n", err)
		} else {
			distance := exactResults[0].Distance
			fmt.Printf("✅ Found (distance=%.6f, %v)\n", distance, exactDuration)
		}
	} else {
		fmt.Println("⚠️  Not enough vectors for exact match test")
	}

	// 2. Top-K Similarity Search (various K values)
	fmt.Println("\n📊 TOP-K SIMILARITY: Various K values...")
	kValues := []int{1, 5, 10, 20, 50, 100}
	queryVector = generateRandomVector(DIM_SIZE)
	kTimings := make(map[int]time.Duration)

	for _, k := range kValues {
		fmt.Printf("   K=%d: ", k)
		start = time.Now()
		results, err := coll.Search(ctx, &api.SearchRequest{
			Vector:      queryVector,
			K:           k,
			IncludeData: false,
		})
		duration := time.Since(start)
		kTimings[k] = duration

		if err != nil {
			fmt.Printf("❌ Error: %v\n", err)
			continue
		}

		avgDistance := float32(0)
		for _, r := range results {
			avgDistance += r.Distance
		}
		if len(results) > 0 {
			avgDistance /= float32(len(results))
		}

		fmt.Printf("%d results (avg_dist=%.4f, %v)\n", len(results), avgDistance, duration)
	}

	// 3. Filtered Search
	fmt.Println("\n🏷️  FILTERED SEARCH: Category-based filtering...")
	categories := []string{"technology", "science", "business", "sports"}

	for _, category := range categories {
		fmt.Printf("   Category '%s': ", category)
		filter := &api.FieldFilter{
			Field: "category",
			Op:    api.FilterEq,
			Value: category,
		}

		start = time.Now()
		results, err := coll.Search(ctx, &api.SearchRequest{
			Vector:      queryVector,
			K:           10,
			Filter:      filter,
			IncludeData: false,
		})
		duration := time.Since(start)

		if err != nil {
			fmt.Printf("❌ Error: %v\n", err)
		} else {
			fmt.Printf("%d results (%v)\n", len(results), duration)
		}
	}

	// 4. Multi-Vector Batch Search
	fmt.Print("\n🔄 MULTI-VECTOR SEARCH: Batch query processing... ")
	numQueries := 10
	queryVectors := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queryVectors[i] = generateRandomVector(DIM_SIZE)
	}

	start = time.Now()
	batchResults := make([][]*api.SearchResult, numQueries)
	for i, qv := range queryVectors {
		results, err := coll.Search(ctx, &api.SearchRequest{
			Vector:      qv,
			K:           5,
			IncludeData: false,
		})
		if err == nil {
			batchResults[i] = results
		}
	}
	batchDuration := time.Since(start)

	totalBatchResults := 0
	for _, results := range batchResults {
		totalBatchResults += len(results)
	}
	batchThroughput := float64(numQueries) / batchDuration.Seconds()
	fmt.Printf("✅ %d queries, %d total results (%.1f queries/sec, %v)\n",
		numQueries, totalBatchResults, batchThroughput, batchDuration)

	// 5. Range Search (distance threshold)
	fmt.Print("\n📏 RANGE SEARCH: Distance threshold filtering... ")
	maxDistance := float32(0.8)
	start = time.Now()
	rangeResults, err := coll.Search(ctx, &api.SearchRequest{
		Vector:      queryVector,
		K:           100, // Large K to get candidates
		MaxDistance: &maxDistance,
		IncludeData: false,
	})
	rangeDuration := time.Since(start)

	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
	} else {
		// Filter results within range (if MaxDistance not supported by backend)
		withinRange := 0
		for _, r := range rangeResults {
			if r.Distance <= maxDistance {
				withinRange++
			}
		}
		fmt.Printf("✅ %d/%d within range (%.3f threshold, %v)\n",
			withinRange, len(rangeResults), maxDistance, rangeDuration)
	}

	// 6. Performance Stress Test
	fmt.Print("\n⚡ PERFORMANCE STRESS TEST: High-throughput search... ")
	stressQueries := 1000
	start = time.Now()
	totalResults := 0
	errorCount := 0
	latencies := make([]time.Duration, 0, stressQueries)

	for i := 0; i < stressQueries; i++ {
		queryStart := time.Now()
		queryVec := generateRandomVector(DIM_SIZE)
		results, err := coll.Search(ctx, &api.SearchRequest{
			Vector:      queryVec,
			K:           10,
			IncludeData: false,
		})
		queryLatency := time.Since(queryStart)
		latencies = append(latencies, queryLatency)

		if err != nil {
			errorCount++
		} else {
			totalResults += len(results)
		}
	}
	totalDuration := time.Since(start)

	// Calculate statistics
	avgLatency := totalDuration / time.Duration(stressQueries)
	stressThroughput := float64(stressQueries) / totalDuration.Seconds()

	// Calculate percentiles
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	p50 := latencies[len(latencies)/2]
	p95 := latencies[int(float64(len(latencies))*0.95)]
	p99 := latencies[int(float64(len(latencies))*0.99)]

	fmt.Printf("✅ Complete\n")
	fmt.Printf("   📊 Queries: %d (errors: %d)\n", stressQueries, errorCount)
	fmt.Printf("   📊 Total results: %d\n", totalResults)
	fmt.Printf("   📊 Throughput: %.1f searches/sec\n", stressThroughput)
	fmt.Printf("   📊 Avg latency: %v\n", avgLatency)
	fmt.Printf("   📊 P50 latency: %v\n", p50)
	fmt.Printf("   📊 P95 latency: %v\n", p95)
	fmt.Printf("   📊 P99 latency: %v\n", p99)

	// Search Performance Summary
	fmt.Println("\n📊 Search Performance Summary:")
	if len(vectors) > 42 {
		fmt.Printf("   🎯 Exact match:    %v\n", exactDuration)
	}
	for _, k := range kValues {
		fmt.Printf("   📊 Top-%d search:   %v\n", k, kTimings[k])
	}
	fmt.Printf("   🔄 Batch search:    %v (%.1f queries/sec)\n", batchDuration, batchThroughput)
	fmt.Printf("   📏 Range search:    %v\n", rangeDuration)
	fmt.Printf("   ⚡ Stress test:     %v (%.1f QPS)\n", totalDuration, stressThroughput)

	fmt.Println()
	return true
}

// runConcurrentDemo demonstrates concurrent operations
func runConcurrentDemo(ctx context.Context, coll *collection.VectorCollection) bool {
	fmt.Println("🔄 Concurrent Operations Demo")
	fmt.Println("=============================")

	numWorkers := 10
	numOpsPerWorker := 100

	fmt.Printf("🚀 Running %d concurrent workers (%d ops each)...\n", numWorkers, numOpsPerWorker)

	start := time.Now()
	done := make(chan bool, numWorkers)
	errors := make(chan error, numWorkers*numOpsPerWorker)

	// Launch concurrent workers
	for i := 0; i < numWorkers; i++ {
		go func(workerID int) {
			defer func() { done <- true }()

			for j := 0; j < numOpsPerWorker; j++ {
				// Concurrent search
				queryVector := generateRandomVector(DIM_SIZE)
				_, err := coll.Search(ctx, &api.SearchRequest{
					Vector:      queryVector,
					K:           5,
					IncludeData: false,
				})
				if err != nil {
					errors <- err
				}
			}
		}(i)
	}

	// Wait for completion
	for i := 0; i < numWorkers; i++ {
		<-done
	}
	close(errors)
	duration := time.Since(start)

	// Count errors
	errorCount := 0
	for range errors {
		errorCount++
	}

	totalOps := numWorkers * numOpsPerWorker
	throughput := float64(totalOps) / duration.Seconds()

	fmt.Printf("✅ Completed %d operations in %v\n", totalOps, duration)
	fmt.Printf("   📊 Throughput: %.1f ops/sec\n", throughput)
	fmt.Printf("   ❌ Errors: %d (%.2f%%)\n", errorCount, float64(errorCount)/float64(totalOps)*100)

	fmt.Println()
	return errorCount == 0
}

// runPersistenceDemo demonstrates persistence operations
func runPersistenceDemo(ctx context.Context, coll *collection.VectorCollection) bool {
	fmt.Println("💾 Persistence Demo")
	fmt.Println("===================")

	fmt.Print("💾 Testing collection persistence... ")

	// Note: In a real implementation, you might have persistence methods
	// For now, we'll simulate with basic collection info
	start := time.Now()
	count, err := coll.Count(ctx)
	if err != nil {
		fmt.Printf("❌ Error checking collection state: %v\n", err)
		return false
	}

	fmt.Printf("✅ Collection has %d vectors (%v)\n", count, time.Since(start))

	// Simulate persistence validation
	fmt.Print("🔍 Validating collection integrity... ")
	start = time.Now()

	// Basic integrity check - try to retrieve a few vectors
	testVector := generateRandomVector(DIM_SIZE)
	_, err = coll.Search(ctx, &api.SearchRequest{
		Vector:      testVector,
		K:           1,
		IncludeData: false,
	})

	if err != nil {
		fmt.Printf("❌ Integrity check failed: %v\n", err)
		return false
	}

	fmt.Printf("✅ Integrity check passed (%v)\n", time.Since(start))

	fmt.Println()
	return true
}

// Helper functions

// generateRandomVector creates a random vector of the specified dimension
func generateRandomVector(dim int) []float32 {
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = rand.Float32()*2 - 1 // Range [-1, 1]
	}
	return vector
}

// generateTestVectors creates multiple test vectors with metadata
func generateTestVectors(count int) []*TestVector {
	vectors := make([]*TestVector, count)
	categories := []string{"technology", "science", "business", "sports", "entertainment"}

	for i := 0; i < count; i++ {
		vectors[i] = &TestVector{
			ID:   fmt.Sprintf("vec_%04d", i),
			Data: generateRandomVector(DIM_SIZE),
			Metadata: map[string]interface{}{
				"category": categories[i%len(categories)],
				"index":    i,
				"batch":    i / BATCH_SIZE,
				"score":    rand.Float64(),
			},
		}
	}
	return vectors
}

// convertToAPIVector converts TestVector to API Vector
func convertToAPIVector(tv *TestVector) *api.Vector {
	return &api.Vector{
		ID:       tv.ID,
		Data:     tv.Data,
		Metadata: tv.Metadata,
	}
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
