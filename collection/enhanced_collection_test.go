// Package collection provides integration tests for the enhanced collection.
package collection

import (
	"context"
	"fmt"
	"testing"

	"github.com/khambampati-subhash/govecdb/api"
)

// TestEnhancedCollectionIntegration tests the enhanced collection with filtering and segments
func TestEnhancedCollectionIntegration(t *testing.T) {
	// Create a test configuration
	config := &api.CollectionConfig{
		Name:           "test-enhanced-collection",
		Dimension:      3,
		Metric:         api.Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		ThreadSafe:     true,
	}

	// Create enhanced collection
	collection, err := NewEnhancedVectorCollection(config)
	if err != nil {
		t.Fatalf("Failed to create enhanced collection: %v", err)
	}
	defer collection.Close()

	ctx := context.Background()

	// Test adding vectors with metadata
	vectors := []*api.Vector{
		{
			ID:   "vec-1",
			Data: []float32{1.0, 2.0, 3.0},
			Metadata: map[string]interface{}{
				"category": "type_a",
				"value":    10,
			},
		},
		{
			ID:   "vec-2",
			Data: []float32{2.0, 3.0, 4.0},
			Metadata: map[string]interface{}{
				"category": "type_b",
				"value":    20,
			},
		},
		{
			ID:   "vec-3",
			Data: []float32{3.0, 4.0, 5.0},
			Metadata: map[string]interface{}{
				"category": "type_a",
				"value":    30,
			},
		},
	}

	// Add vectors
	for _, vector := range vectors {
		if err := collection.Add(ctx, vector); err != nil {
			t.Fatalf("Failed to add vector %s: %v", vector.ID, err)
		}
	}

	// Test retrieval
	retrieved, err := collection.Get(ctx, "vec-1")
	if err != nil {
		t.Fatalf("Failed to get vector: %v", err)
	}
	if retrieved.ID != "vec-1" {
		t.Fatalf("Retrieved wrong vector: expected vec-1, got %s", retrieved.ID)
	}

	// Test count
	count, err := collection.Count(ctx)
	if err != nil {
		t.Fatalf("Failed to get count: %v", err)
	}
	if count != 3 {
		t.Fatalf("Expected count 3, got %d", count)
	}

	// Test filtering by category
	categoryFilter := &api.FieldFilter{
		Field: "category",
		Op:    api.FilterEq,
		Value: "type_a",
	}

	filtered, err := collection.Filter(ctx, categoryFilter, 10)
	if err != nil {
		t.Fatalf("Failed to filter by category: %v", err)
	}

	if len(filtered) != 2 {
		t.Fatalf("Expected 2 vectors with category 'type_a', got %d", len(filtered))
	}

	for _, vector := range filtered {
		if vector.Metadata["category"] != "type_a" {
			t.Fatalf("Filter returned vector with wrong category: %v", vector.Metadata["category"])
		}
	}

	// Test search with filter
	searchReq := &api.SearchRequest{
		Vector: []float32{1.5, 2.5, 3.5},
		K:      5,
		Filter: categoryFilter,
	}

	results, err := collection.Search(ctx, searchReq)
	if err != nil {
		t.Fatalf("Failed to search with filter: %v", err)
	}

	// Should return only vectors matching the category filter
	if len(results) > 2 {
		t.Fatalf("Search with filter returned too many results: %d", len(results))
	}

	for _, result := range results {
		if result.Vector.Metadata["category"] != "type_a" {
			t.Fatalf("Search result has wrong category: %v", result.Vector.Metadata["category"])
		}
	}

	// Test batch operations
	batchVectors := []*api.Vector{
		{
			ID:   "batch-1",
			Data: []float32{4.0, 5.0, 6.0},
			Metadata: map[string]interface{}{
				"category": "type_c",
				"value":    40,
			},
		},
		{
			ID:   "batch-2",
			Data: []float32{5.0, 6.0, 7.0},
			Metadata: map[string]interface{}{
				"category": "type_c",
				"value":    50,
			},
		},
	}

	if err := collection.AddBatch(ctx, batchVectors); err != nil {
		t.Fatalf("Failed to add batch: %v", err)
	}

	// Verify batch addition
	newCount, err := collection.Count(ctx)
	if err != nil {
		t.Fatalf("Failed to get count after batch: %v", err)
	}
	if newCount != 5 {
		t.Fatalf("Expected count 5 after batch, got %d", newCount)
	}

	// Test stats
	stats, err := collection.Stats(ctx)
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.VectorCount != 5 {
		t.Fatalf("Stats show wrong vector count: expected 5, got %d", stats.VectorCount)
	}

	if stats.Dimension != 3 {
		t.Fatalf("Stats show wrong dimension: expected 3, got %d", stats.Dimension)
	}

	// Test clear
	if err := collection.Clear(ctx); err != nil {
		t.Fatalf("Failed to clear collection: %v", err)
	}

	finalCount, err := collection.Count(ctx)
	if err != nil {
		t.Fatalf("Failed to get count after clear: %v", err)
	}
	if finalCount != 0 {
		t.Fatalf("Expected count 0 after clear, got %d", finalCount)
	}
}

// TestEnhancedCollectionConcurrency tests concurrent operations
func TestEnhancedCollectionConcurrency(t *testing.T) {
	config := &api.CollectionConfig{
		Name:           "test-concurrent-collection",
		Dimension:      2,
		Metric:         api.Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		ThreadSafe:     true,
	}

	collection, err := NewEnhancedVectorCollection(config)
	if err != nil {
		t.Fatalf("Failed to create enhanced collection: %v", err)
	}
	defer collection.Close()

	ctx := context.Background()

	// Test concurrent adds
	numWorkers := 10
	vectorsPerWorker := 10
	done := make(chan bool, numWorkers)

	for i := 0; i < numWorkers; i++ {
		go func(workerID int) {
			defer func() { done <- true }()

			for j := 0; j < vectorsPerWorker; j++ {
				vector := &api.Vector{
					ID:   fmt.Sprintf("worker-%d-vec-%d", workerID, j),
					Data: []float32{float32(workerID), float32(j)},
					Metadata: map[string]interface{}{
						"worker": workerID,
						"seq":    j,
					},
				}

				if err := collection.Add(ctx, vector); err != nil {
					t.Errorf("Worker %d failed to add vector %d: %v", workerID, j, err)
					return
				}
			}
		}(i)
	}

	// Wait for all workers to complete
	for i := 0; i < numWorkers; i++ {
		<-done
	}

	// Verify total count
	count, err := collection.Count(ctx)
	if err != nil {
		t.Fatalf("Failed to get count: %v", err)
	}

	expected := int64(numWorkers * vectorsPerWorker)
	if count != expected {
		t.Fatalf("Expected count %d, got %d", expected, count)
	}

	// Test concurrent searches
	searchDone := make(chan bool, numWorkers)

	for i := 0; i < numWorkers; i++ {
		go func(workerID int) {
			defer func() { searchDone <- true }()

			searchReq := &api.SearchRequest{
				Vector: []float32{float32(workerID), 5.0},
				K:      5,
			}

			_, err := collection.Search(ctx, searchReq)
			if err != nil {
				t.Errorf("Worker %d search failed: %v", workerID, err)
			}
		}(i)
	}

	// Wait for all search workers to complete
	for i := 0; i < numWorkers; i++ {
		<-searchDone
	}
}
