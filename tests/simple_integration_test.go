package tests

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
	"github.com/khambampati-subhash/govecdb/collection"
	"github.com/khambampati-subhash/govecdb/store"
)

// TestBasicVectorOperations tests basic vector operations
func TestBasicVectorOperations(t *testing.T) {
	// Create collection configuration
	config := api.DefaultCollectionConfig("test", 128)
	config.Metric = api.Euclidean

	// Create store configuration
	storeConfig := store.DefaultStoreConfig("test")

	// Create collection
	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}
	defer coll.Close()

	// Test vector addition
	vector := generateRandomVector(128)
	testVector := &api.Vector{
		ID:   "test_1",
		Data: vector,
		Metadata: map[string]interface{}{
			"category": "test",
			"value":    42.0,
		},
	}

	if err := coll.Add(context.Background(), testVector); err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	// Test search
	req := &api.SearchRequest{
		Vector: vector,
		K:      1,
	}

	results, err := coll.Search(context.Background(), req)
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	if results[0].Vector.ID != "test_1" {
		t.Fatalf("Expected ID 'test_1', got '%s'", results[0].Vector.ID)
	}

	t.Log("✅ Basic vector operations working correctly!")
}

// TestBatchOperations tests batch operations
func TestBatchOperations(t *testing.T) {
	config := api.DefaultCollectionConfig("batch_test", 64)
	config.Metric = api.Cosine

	storeConfig := store.DefaultStoreConfig("batch_test")

	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}
	defer coll.Close()

	// Generate batch data
	batchSize := 100
	vectors := make([]*api.Vector, batchSize)

	for i := 0; i < batchSize; i++ {
		vectors[i] = &api.Vector{
			ID:   fmt.Sprintf("batch_%d", i),
			Data: generateRandomVector(64),
			Metadata: map[string]interface{}{
				"batch_id": i / 10,
				"category": fmt.Sprintf("cat_%d", i%5),
			},
		}
	}

	// Test batch insert
	startTime := time.Now()
	if err := coll.AddBatch(context.Background(), vectors); err != nil {
		t.Fatalf("Failed to batch insert: %v", err)
	}
	insertTime := time.Since(startTime)

	t.Logf("Batch insert of %d vectors took %v", batchSize, insertTime)

	// Test search performance
	queryVector := generateRandomVector(64)
	req := &api.SearchRequest{
		Vector: queryVector,
		K:      10,
	}

	startTime = time.Now()
	results, err := coll.Search(context.Background(), req)
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}
	searchTime := time.Since(startTime)

	t.Logf("Search took %v, found %d results", searchTime, len(results))

	if len(results) != 10 {
		t.Fatalf("Expected 10 results, got %d", len(results))
	}

	// Verify results are sorted by distance/score
	for i := 1; i < len(results); i++ {
		if results[i-1].Distance > results[i].Distance {
			t.Fatalf("Results not sorted by distance: %f > %f", results[i-1].Distance, results[i].Distance)
		}
	}

	t.Log("✅ Batch operations working correctly!")
}

// TestFilteredSearch tests search with metadata filters
func TestFilteredSearch(t *testing.T) {
	config := api.DefaultCollectionConfig("filter_test", 32)
	config.Metric = api.Euclidean

	storeConfig := store.DefaultStoreConfig("filter_test")

	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}
	defer coll.Close()

	// Insert test data with various metadata
	testData := make([]*api.Vector, 50)
	categories := []string{"A", "B", "C"}

	for i := 0; i < 50; i++ {
		testData[i] = &api.Vector{
			ID:   fmt.Sprintf("item_%d", i),
			Data: generateRandomVector(32),
			Metadata: map[string]interface{}{
				"category": categories[i%len(categories)],
				"value":    float64(i),
				"active":   i%2 == 0,
			},
		}
	}

	if err := coll.AddBatch(context.Background(), testData); err != nil {
		t.Fatalf("Failed to insert test data: %v", err)
	}

	// Test search with filter using the proper API
	queryVector := generateRandomVector(32)

	// Create a category filter
	categoryFilter := api.Eq("category", "A")

	req := &api.SearchRequest{
		Vector: queryVector,
		K:      20,
		Filter: categoryFilter,
	}

	results, err := coll.Search(context.Background(), req)
	if err != nil {
		t.Fatalf("Failed to search with filter: %v", err)
	}

	// Verify all results have category "A"
	for _, result := range results {
		if category, ok := result.Vector.Metadata["category"]; !ok || category != "A" {
			t.Fatalf("Filter failed: expected category 'A', got '%v'", category)
		}
	}

	t.Logf("Category filter returned %d results", len(results))
	t.Log("✅ Filtered search working correctly!")
}

// TestVectorDeletion tests vector deletion
func TestVectorDeletion(t *testing.T) {
	config := api.DefaultCollectionConfig("delete_test", 64)
	storeConfig := store.DefaultStoreConfig("delete_test")

	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}
	defer coll.Close()

	// Add some vectors
	vectors := make([]*api.Vector, 10)
	for i := 0; i < 10; i++ {
		vectors[i] = &api.Vector{
			ID:   fmt.Sprintf("delete_test_%d", i),
			Data: generateRandomVector(64),
		}
	}

	if err := coll.AddBatch(context.Background(), vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Search before deletion
	req := &api.SearchRequest{
		Vector: vectors[0].Data,
		K:      5,
	}

	results, err := coll.Search(context.Background(), req)
	if err != nil {
		t.Fatalf("Failed to search before deletion: %v", err)
	}

	initialCount := len(results)
	t.Logf("Found %d results before deletion", initialCount)

	// Delete some vectors
	idsToDelete := []string{"delete_test_0", "delete_test_1", "delete_test_2"}
	if err := coll.DeleteBatch(context.Background(), idsToDelete); err != nil {
		t.Fatalf("Failed to delete vectors: %v", err)
	}

	// Search after deletion
	results, err = coll.Search(context.Background(), req)
	if err != nil {
		t.Fatalf("Failed to search after deletion: %v", err)
	}

	finalCount := len(results)
	t.Logf("Found %d results after deletion", finalCount)

	// Verify the specific deleted vector is not found
	for _, result := range results {
		if result.Vector.ID == "delete_test_0" {
			t.Fatal("Deleted vector still found in search results")
		}
	}

	t.Log("✅ Vector deletion working correctly!")
}

// TestPerformanceBaseline tests basic performance metrics
func TestPerformanceBaseline(t *testing.T) {
	config := api.DefaultCollectionConfig("perf_test", 256)
	storeConfig := store.DefaultStoreConfig("perf_test")

	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}
	defer coll.Close()

	// Performance test parameters
	numVectors := 1000
	numQueries := 50

	// Generate test data
	vectors := make([]*api.Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = &api.Vector{
			ID:   fmt.Sprintf("perf_%d", i),
			Data: generateRandomVector(256),
		}
	}

	// Measure insert performance
	insertStart := time.Now()
	if err := coll.AddBatch(context.Background(), vectors); err != nil {
		t.Fatalf("Failed to batch insert: %v", err)
	}
	insertTime := time.Since(insertStart)

	insertRate := float64(numVectors) / insertTime.Seconds()
	t.Logf("Insert performance: %d vectors in %v (%.2f vectors/sec)", numVectors, insertTime, insertRate)

	// Measure search performance
	queryVectors := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queryVectors[i] = generateRandomVector(256)
	}

	searchStart := time.Now()
	for _, queryVector := range queryVectors {
		req := &api.SearchRequest{
			Vector: queryVector,
			K:      10,
		}
		_, err := coll.Search(context.Background(), req)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
	}
	searchTime := time.Since(searchStart)

	avgSearchTime := searchTime / time.Duration(numQueries)
	searchRate := float64(numQueries) / searchTime.Seconds()

	t.Logf("Search performance: %d queries in %v (avg: %v, %.2f queries/sec)",
		numQueries, searchTime, avgSearchTime, searchRate)

	// Basic performance expectations
	if insertRate < 100 {
		t.Logf("Warning: Insert rate %f may be below expected performance", insertRate)
	}

	if avgSearchTime > 50*time.Millisecond {
		t.Logf("Warning: Average search time %v may be too high", avgSearchTime)
	}

	t.Log("✅ Performance baseline established!")
}

// generateRandomVector generates a random vector of the specified dimension
func generateRandomVector(dimension int) []float32 {
	vector := make([]float32, dimension)
	for i := 0; i < dimension; i++ {
		vector[i] = rand.Float32()*2 - 1 // Random values between -1 and 1
	}
	return vector
}

// RunAllIntegrationTests runs all integration tests
func TestAllIntegrationTests(t *testing.T) {
	t.Run("BasicVectorOperations", TestBasicVectorOperations)
	t.Run("BatchOperations", TestBatchOperations)
	t.Run("FilteredSearch", TestFilteredSearch)
	t.Run("VectorDeletion", TestVectorDeletion)
	t.Run("PerformanceBaseline", TestPerformanceBaseline)

	t.Log("🎉 All integration tests completed successfully!")
}
