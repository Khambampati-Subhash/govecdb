package store

import (
	"context"
	"fmt"
	"testing"

	"github.com/khambampati-subhash/govecdb/api"
)

func TestMemoryStore(t *testing.T) {
	config := DefaultStoreConfig("test-store")
	store := NewMemoryStore(config)
	ctx := context.Background()

	t.Run("Put", func(t *testing.T) {
		vector := &api.Vector{
			ID:   "test1",
			Data: []float32{1.0, 2.0, 3.0},
			Metadata: map[string]interface{}{
				"category": "test",
			},
		}

		err := store.Put(ctx, vector)
		if err != nil {
			t.Errorf("Failed to put vector: %v", err)
		}

		// Try putting same ID again should fail
		err = store.Put(ctx, vector)
		if err != api.ErrVectorExists {
			t.Errorf("Expected ErrVectorExists, got %v", err)
		}
	})

	t.Run("Get", func(t *testing.T) {
		// Get existing vector
		vector, err := store.Get(ctx, "test1")
		if err != nil {
			t.Errorf("Failed to get vector: %v", err)
		}
		if vector.ID != "test1" {
			t.Errorf("Expected ID test1, got %s", vector.ID)
		}

		// Get non-existent vector
		_, err = store.Get(ctx, "nonexistent")
		if err != api.ErrVectorNotFound {
			t.Errorf("Expected ErrVectorNotFound, got %v", err)
		}
	})

	t.Run("PutBatch", func(t *testing.T) {
		vectors := []*api.Vector{
			{
				ID:   "batch1",
				Data: []float32{4.0, 5.0, 6.0},
			},
			{
				ID:   "batch2",
				Data: []float32{7.0, 8.0, 9.0},
			},
		}

		err := store.PutBatch(ctx, vectors)
		if err != nil {
			t.Errorf("Failed to put batch: %v", err)
		}

		// Verify both vectors were stored
		count, err := store.Count(ctx)
		if err != nil {
			t.Errorf("Failed to get count: %v", err)
		}
		if count != 3 { // test1 + batch1 + batch2
			t.Errorf("Expected count 3, got %d", count)
		}
	})

	t.Run("Delete", func(t *testing.T) {
		err := store.Delete(ctx, "test1")
		if err != nil {
			t.Errorf("Failed to delete vector: %v", err)
		}

		// Verify deletion
		_, err = store.Get(ctx, "test1")
		if err != api.ErrVectorNotFound {
			t.Errorf("Expected ErrVectorNotFound, got %v", err)
		}

		// Delete non-existent vector
		err = store.Delete(ctx, "nonexistent")
		if err != api.ErrVectorNotFound {
			t.Errorf("Expected ErrVectorNotFound, got %v", err)
		}
	})

	t.Run("DeleteBatch", func(t *testing.T) {
		err := store.DeleteBatch(ctx, []string{"batch1", "batch2"})
		if err != nil {
			t.Errorf("Failed to delete batch: %v", err)
		}

		// Verify all vectors were deleted
		count, err := store.Count(ctx)
		if err != nil {
			t.Errorf("Failed to get count: %v", err)
		}
		if count != 0 {
			t.Errorf("Expected count 0, got %d", count)
		}
	})

	t.Run("Filter", func(t *testing.T) {
		// Add test vectors with different categories
		vectors := []*api.Vector{
			{
				ID:   "v1",
				Data: []float32{1.0, 2.0, 3.0},
				Metadata: map[string]interface{}{
					"category": "A",
				},
			},
			{
				ID:   "v2",
				Data: []float32{4.0, 5.0, 6.0},
				Metadata: map[string]interface{}{
					"category": "B",
				},
			},
			{
				ID:   "v3",
				Data: []float32{7.0, 8.0, 9.0},
				Metadata: map[string]interface{}{
					"category": "A",
				},
			},
		}

		err := store.PutBatch(ctx, vectors)
		if err != nil {
			t.Errorf("Failed to put test vectors: %v", err)
		}

		// Filter for category A using FuncFilter adapter
		filterFunc := func(metadata map[string]interface{}) bool {
			category, ok := metadata["category"]
			return ok && category == "A"
		}
		filter := api.NewFuncFilter(filterFunc)

		filtered, err := store.Filter(ctx, filter, 10)
		if err != nil {
			t.Errorf("Failed to filter vectors: %v", err)
		}

		if len(filtered) != 2 {
			t.Errorf("Expected 2 filtered vectors, got %d", len(filtered))
		}
	})

	t.Run("List", func(t *testing.T) {
		// Test pagination
		vectors, err := store.List(ctx, 2, 0)
		if err != nil {
			t.Errorf("Failed to list vectors: %v", err)
		}
		if len(vectors) != 2 {
			t.Errorf("Expected 2 vectors, got %d", len(vectors))
		}

		// Test offset
		vectors, err = store.List(ctx, 2, 2)
		if err != nil {
			t.Errorf("Failed to list vectors with offset: %v", err)
		}
		if len(vectors) != 1 {
			t.Errorf("Expected 1 vector, got %d", len(vectors))
		}
	})

	t.Run("Clear", func(t *testing.T) {
		err := store.Clear(ctx)
		if err != nil {
			t.Errorf("Failed to clear store: %v", err)
		}

		count, err := store.Count(ctx)
		if err != nil {
			t.Errorf("Failed to get count: %v", err)
		}
		if count != 0 {
			t.Errorf("Expected count 0 after clear, got %d", count)
		}
	})

	t.Run("Stats", func(t *testing.T) {
		stats, err := store.Stats(ctx)
		if err != nil {
			t.Errorf("Failed to get stats: %v", err)
		}

		if stats == nil {
			t.Error("Expected non-nil stats")
		}
	})
}

func TestMemoryStoreEdgeCases(t *testing.T) {
	config := DefaultStoreConfig("test-store")
	store := NewMemoryStore(config)
	ctx := context.Background()

	t.Run("NilVector", func(t *testing.T) {
		err := store.Put(ctx, nil)
		if err == nil {
			t.Error("Expected error for nil vector")
		}
	})

	t.Run("EmptyID", func(t *testing.T) {
		vector := &api.Vector{
			ID:   "",
			Data: []float32{1.0, 2.0, 3.0},
		}
		err := store.Put(ctx, vector)
		if err == nil {
			t.Error("Expected error for empty ID")
		}
	})

	t.Run("NilData", func(t *testing.T) {
		vector := &api.Vector{
			ID:   "test",
			Data: nil,
		}
		err := store.Put(ctx, vector)
		if err == nil {
			t.Error("Expected error for nil data")
		}
	})

	t.Run("EmptyBatch", func(t *testing.T) {
		err := store.PutBatch(ctx, []*api.Vector{})
		if err != nil {
			t.Errorf("Expected no error for empty batch, got %v", err)
		}
	})

	t.Run("NilFilter", func(t *testing.T) {
		_, err := store.Filter(ctx, nil, 10)
		if err != nil {
			t.Errorf("Expected no error for nil filter, got %v", err)
		}
	})

	t.Run("NegativeLimit", func(t *testing.T) {
		_, err := store.List(ctx, -1, 0)
		if err == nil {
			t.Error("Expected error for negative limit")
		}
	})

	t.Run("NegativeOffset", func(t *testing.T) {
		_, err := store.List(ctx, 10, -1)
		if err == nil {
			t.Error("Expected error for negative offset")
		}
	})

	t.Run("ConcurrentAccess", func(t *testing.T) {
		// Test concurrent access to the store
		done := make(chan bool)
		for i := 0; i < 10; i++ {
			go func(i int) {
				vector := &api.Vector{
					ID:   fmt.Sprintf("concurrent-%d", i),
					Data: []float32{1.0, 2.0, 3.0},
				}
				_ = store.Put(ctx, vector)
				done <- true
			}(i)
		}

		// Wait for all goroutines to finish
		for i := 0; i < 10; i++ {
			<-done
		}

		// Verify all vectors were stored
		count, err := store.Count(ctx)
		if err != nil {
			t.Errorf("Failed to get count: %v", err)
		}
		if count != 10 {
			t.Errorf("Expected count 10, got %d", count)
		}
	})
}
