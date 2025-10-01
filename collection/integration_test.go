// Package collection provides integration tests for the complete persistence cycle.
package collection

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// TestPersistentCollectionIntegration tests the complete persistence lifecycle
func TestPersistentCollectionIntegration(t *testing.T) {
	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "persistent_integration_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create configuration
	config := DefaultPersistentCollectionConfig("test-collection", tempDir, 384)
	config.AutoSnapshotEnabled = true
	config.AutoSnapshotThreshold = 100 // Snapshot every 100 operations

	ctx := context.Background()

	t.Run("CompleteLifecycle", func(t *testing.T) {
		// Phase 1: Create and populate collection
		collection, err := NewPersistentVectorCollection(config)
		if err != nil {
			t.Fatalf("Failed to create persistent collection: %v", err)
		}

		if err := collection.Start(ctx); err != nil {
			t.Fatalf("Failed to start collection: %v", err)
		}

		// Add vectors
		vectors := make([]*api.Vector, 250)
		for i := 0; i < 250; i++ {
			vectors[i] = &api.Vector{
				ID:   fmt.Sprintf("vec-%d", i),
				Data: make([]float32, 384),
				Metadata: map[string]interface{}{
					"category": fmt.Sprintf("cat-%d", i%5),
					"value":    i,
				},
			}
			// Fill with test data
			for j := range vectors[i].Data {
				vectors[i].Data[j] = float32(i*j) / 1000.0
			}
		}

		// Add vectors in batches
		batchSize := 50
		for i := 0; i < len(vectors); i += batchSize {
			end := i + batchSize
			if end > len(vectors) {
				end = len(vectors)
			}

			batch := vectors[i:end]
			if err := collection.AddBatch(ctx, batch); err != nil {
				t.Errorf("Failed to add batch %d: %v", i/batchSize, err)
			}
		}

		// Verify vectors were added
		count, err := collection.Count(ctx)
		if err != nil {
			t.Errorf("Failed to get count: %v", err)
		}
		if count != 250 {
			t.Errorf("Expected 250 vectors, got %d", count)
		}

		// Test search functionality
		queryVector := vectors[100].Data
		results, err := collection.Search(ctx, &api.SearchRequest{
			Vector: queryVector,
			K:      5,
		})
		if err != nil {
			t.Errorf("Failed to search: %v", err)
		}
		if len(results) == 0 {
			t.Error("Expected search results")
		}

		// Force a snapshot
		_, err = collection.CreateSnapshot(ctx)
		if err != nil {
			t.Errorf("Failed to create snapshot: %v", err)
		}

		// Close collection (simulating normal shutdown)
		if err := collection.Close(); err != nil {
			t.Errorf("Failed to close collection: %v", err)
		}

		// Phase 2: Reopen collection and verify data persistence
		collection2, err := NewPersistentVectorCollection(config)
		if err != nil {
			t.Fatalf("Failed to create second collection: %v", err)
		}

		if err := collection2.Start(ctx); err != nil {
			t.Fatalf("Failed to start second collection: %v", err)
		}

		// Verify data survived restart
		count2, err := collection2.Count(ctx)
		if err != nil {
			t.Errorf("Failed to get count after restart: %v", err)
		}
		if count2 != 250 {
			t.Errorf("Expected 250 vectors after restart, got %d", count2)
		}

		// Verify search still works
		results2, err := collection2.Search(ctx, &api.SearchRequest{
			Vector: queryVector,
			K:      5,
		})
		if err != nil {
			t.Errorf("Failed to search after restart: %v", err)
		}
		if len(results2) != len(results) {
			t.Errorf("Search results changed after restart: %d vs %d", len(results), len(results2))
		}

		// Phase 3: Add more data and test incremental persistence
		additionalVectors := make([]*api.Vector, 100)
		for i := 0; i < 100; i++ {
			additionalVectors[i] = &api.Vector{
				ID:   fmt.Sprintf("additional-vec-%d", i),
				Data: make([]float32, 384),
				Metadata: map[string]interface{}{
					"category": "additional",
					"value":    i + 1000,
				},
			}
		}

		if err := collection2.AddBatch(ctx, additionalVectors); err != nil {
			t.Errorf("Failed to add additional vectors: %v", err)
		}

		// Delete some vectors
		deleteIDs := []string{"vec-0", "vec-1", "vec-2"}
		if err := collection2.DeleteBatch(ctx, deleteIDs); err != nil {
			t.Errorf("Failed to delete vectors: %v", err)
		}

		// Final count should be 250 + 100 - 3 = 347
		finalCount, err := collection2.Count(ctx)
		if err != nil {
			t.Errorf("Failed to get final count: %v", err)
		}
		if finalCount != 347 {
			t.Errorf("Expected 347 vectors, got %d", finalCount)
		}

		if err := collection2.Close(); err != nil {
			t.Errorf("Failed to close second collection: %v", err)
		}

		// Phase 4: Final verification after all operations
		collection3, err := NewPersistentVectorCollection(config)
		if err != nil {
			t.Fatalf("Failed to create third collection: %v", err)
		}

		if err := collection3.Start(ctx); err != nil {
			t.Fatalf("Failed to start third collection: %v", err)
		}
		defer collection3.Close()

		// Verify final state
		verifyCount, err := collection3.Count(ctx)
		if err != nil {
			t.Errorf("Failed to get verification count: %v", err)
		}
		if verifyCount != 347 {
			t.Errorf("Expected 347 vectors in final verification, got %d", verifyCount)
		}

		// Verify deleted vectors are gone
		for _, id := range deleteIDs {
			_, err := collection3.Get(ctx, id)
			if err != api.ErrVectorNotFound {
				t.Errorf("Deleted vector %s should not exist", id)
			}
		}

		// Verify additional vectors exist
		_, err = collection3.Get(ctx, "additional-vec-0")
		if err != nil {
			t.Errorf("Additional vector should exist: %v", err)
		}
	})
}

// TestCrashRecovery simulates a crash and tests recovery
func TestCrashRecovery(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "crash_recovery_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultPersistentCollectionConfig("crash-test-collection", tempDir, 128)
	config.AutoSnapshotEnabled = false // Disable auto-snapshots for controlled testing

	ctx := context.Background()

	t.Run("RecoveryFromWALOnly", func(t *testing.T) {
		// Phase 1: Create collection and add data
		collection, err := NewPersistentVectorCollection(config)
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		if err := collection.Start(ctx); err != nil {
			t.Fatalf("Failed to start collection: %v", err)
		}

		// Add vectors
		vectors := make([]*api.Vector, 50)
		for i := 0; i < 50; i++ {
			vectors[i] = &api.Vector{
				ID:   fmt.Sprintf("crash-vec-%d", i),
				Data: make([]float32, 128),
			}
		}

		if err := collection.AddBatch(ctx, vectors); err != nil {
			t.Errorf("Failed to add vectors: %v", err)
		}

		// Simulate crash by closing persistence layer directly
		// (not calling collection.Close() which would do clean shutdown)
		collection.persistence.Stop(ctx)
		collection.manifest.Close(ctx)

		// Phase 2: Recover from crash
		collection2, err := NewPersistentVectorCollection(config)
		if err != nil {
			t.Fatalf("Failed to create recovery collection: %v", err)
		}

		if err := collection2.Start(ctx); err != nil {
			t.Fatalf("Failed to start recovery collection: %v", err)
		}
		defer collection2.Close()

		// Verify recovery
		count, err := collection2.Count(ctx)
		if err != nil {
			t.Errorf("Failed to get count after recovery: %v", err)
		}
		if count != 50 {
			t.Errorf("Expected 50 vectors after recovery, got %d", count)
		}

		// Verify specific vectors
		for i := 0; i < 5; i++ {
			_, err := collection2.Get(ctx, fmt.Sprintf("crash-vec-%d", i))
			if err != nil {
				t.Errorf("Failed to get vector after recovery: %v", err)
			}
		}
	})

	t.Run("RecoveryFromSnapshotAndWAL", func(t *testing.T) {
		// Phase 1: Create collection, add data, and create snapshot
		collection, err := NewPersistentVectorCollection(config)
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		if err := collection.Start(ctx); err != nil {
			t.Fatalf("Failed to start collection: %v", err)
		}

		// Add initial vectors
		initialVectors := make([]*api.Vector, 30)
		for i := 0; i < 30; i++ {
			initialVectors[i] = &api.Vector{
				ID:   fmt.Sprintf("snapshot-vec-%d", i),
				Data: make([]float32, 128),
			}
		}

		if err := collection.AddBatch(ctx, initialVectors); err != nil {
			t.Errorf("Failed to add initial vectors: %v", err)
		}

		// Create snapshot
		_, err = collection.CreateSnapshot(ctx)
		if err != nil {
			t.Errorf("Failed to create snapshot: %v", err)
		}

		// Add more vectors after snapshot
		postSnapshotVectors := make([]*api.Vector, 20)
		for i := 0; i < 20; i++ {
			postSnapshotVectors[i] = &api.Vector{
				ID:   fmt.Sprintf("post-snapshot-vec-%d", i),
				Data: make([]float32, 128),
			}
		}

		if err := collection.AddBatch(ctx, postSnapshotVectors); err != nil {
			t.Errorf("Failed to add post-snapshot vectors: %v", err)
		}

		// Delete some initial vectors
		deleteIDs := []string{"snapshot-vec-0", "snapshot-vec-1"}
		if err := collection.DeleteBatch(ctx, deleteIDs); err != nil {
			t.Errorf("Failed to delete vectors: %v", err)
		}

		// Expected count: 30 - 2 + 20 = 48
		preRecoveryCount, err := collection.Count(ctx)
		if err != nil {
			t.Errorf("Failed to get pre-recovery count: %v", err)
		}

		// Simulate crash
		collection.persistence.Stop(ctx)
		collection.manifest.Close(ctx)

		// Phase 2: Recover from snapshot + WAL
		collection2, err := NewPersistentVectorCollection(config)
		if err != nil {
			t.Fatalf("Failed to create recovery collection: %v", err)
		}

		if err := collection2.Start(ctx); err != nil {
			t.Fatalf("Failed to start recovery collection: %v", err)
		}
		defer collection2.Close()

		// Verify recovery
		postRecoveryCount, err := collection2.Count(ctx)
		if err != nil {
			t.Errorf("Failed to get post-recovery count: %v", err)
		}
		if postRecoveryCount != preRecoveryCount {
			t.Errorf("Expected %d vectors after recovery, got %d", preRecoveryCount, postRecoveryCount)
		}

		// Verify post-snapshot vectors exist
		_, err = collection2.Get(ctx, "post-snapshot-vec-0")
		if err != nil {
			t.Errorf("Post-snapshot vector should exist after recovery: %v", err)
		}

		// Verify deleted vectors are still gone
		for _, id := range deleteIDs {
			_, err := collection2.Get(ctx, id)
			if err != api.ErrVectorNotFound {
				t.Errorf("Deleted vector %s should not exist after recovery", id)
			}
		}
	})
}

// TestCorruptionHandling tests handling of corrupted data
func TestCorruptionHandling(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "corruption_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultPersistentCollectionConfig("corruption-test", tempDir, 64)
	ctx := context.Background()

	t.Run("CorruptedManifestRecovery", func(t *testing.T) {
		// Create collection and add some data
		collection, err := NewPersistentVectorCollection(config)
		if err != nil {
			t.Fatalf("Failed to create collection: %v", err)
		}

		if err := collection.Start(ctx); err != nil {
			t.Fatalf("Failed to start collection: %v", err)
		}

		vectors := []*api.Vector{
			{ID: "test-1", Data: make([]float32, 64)},
			{ID: "test-2", Data: make([]float32, 64)},
		}

		if err := collection.AddBatch(ctx, vectors); err != nil {
			t.Errorf("Failed to add vectors: %v", err)
		}

		if err := collection.Close(); err != nil {
			t.Errorf("Failed to close collection: %v", err)
		}

		// Corrupt the manifest file
		manifestPath := config.ManifestPath
		if err := os.WriteFile(manifestPath, []byte("corrupted data"), 0644); err != nil {
			t.Errorf("Failed to corrupt manifest: %v", err)
		}

		// Try to recover - should handle corruption gracefully
		collection2, err := NewPersistentVectorCollection(config)
		if err != nil {
			t.Fatalf("Failed to create collection after corruption: %v", err)
		}

		// Should either recover or fail gracefully
		err = collection2.Start(ctx)
		if err != nil {
			// Corruption detected, which is expected
			t.Logf("Corruption detected as expected: %v", err)
		} else {
			// If it starts, it should have repaired or recreated the manifest
			defer collection2.Close()
		}
	})
}

// TestLongRunningOperations tests persistence under long-running operations
func TestLongRunningOperations(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running test in short mode")
	}

	tempDir, err := os.MkdirTemp("", "long_running_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultPersistentCollectionConfig("long-running-test", tempDir, 256)
	config.AutoSnapshotEnabled = true
	config.AutoSnapshotInterval = 5 * time.Second
	config.AutoSnapshotThreshold = 50

	ctx := context.Background()

	collection, err := NewPersistentVectorCollection(config)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	if err := collection.Start(ctx); err != nil {
		t.Fatalf("Failed to start collection: %v", err)
	}
	defer collection.Close()

	// Continuously add vectors over time
	totalVectors := 500
	batchSize := 10

	for i := 0; i < totalVectors; i += batchSize {
		vectors := make([]*api.Vector, batchSize)
		for j := 0; j < batchSize; j++ {
			vectors[j] = &api.Vector{
				ID:   fmt.Sprintf("long-running-vec-%d", i+j),
				Data: make([]float32, 256),
			}
		}

		if err := collection.AddBatch(ctx, vectors); err != nil {
			t.Errorf("Failed to add batch at iteration %d: %v", i, err)
		}

		// Brief pause to simulate real-world usage
		time.Sleep(10 * time.Millisecond)

		// Periodically verify data integrity
		if i%100 == 0 && i > 0 {
			count, err := collection.Count(ctx)
			if err != nil {
				t.Errorf("Failed to get count at iteration %d: %v", i, err)
			}
			expectedCount := int64(i + batchSize)
			if count != expectedCount {
				t.Errorf("Count mismatch at iteration %d: expected %d, got %d", i, expectedCount, count)
			}
		}
	}

	// Final verification
	finalCount, err := collection.Count(ctx)
	if err != nil {
		t.Errorf("Failed to get final count: %v", err)
	}
	if finalCount != int64(totalVectors) {
		t.Errorf("Final count mismatch: expected %d, got %d", totalVectors, finalCount)
	}

	// Verify search functionality still works
	testVector := make([]float32, 256)
	results, err := collection.Search(ctx, &api.SearchRequest{
		Vector: testVector,
		K:      10,
	})
	if err != nil {
		t.Errorf("Search failed after long-running operations: %v", err)
	}
	if len(results) == 0 {
		t.Error("Expected search results after long-running operations")
	}
}

// TestConcurrentPersistence tests concurrent operations with persistence
func TestConcurrentPersistence(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "concurrent_persistence_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultPersistentCollectionConfig("concurrent-test", tempDir, 128)
	ctx := context.Background()

	collection, err := NewPersistentVectorCollection(config)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	if err := collection.Start(ctx); err != nil {
		t.Fatalf("Failed to start collection: %v", err)
	}
	defer collection.Close()

	// Run concurrent operations
	numWorkers := 4
	vectorsPerWorker := 25

	errChan := make(chan error, numWorkers)
	doneChan := make(chan bool, numWorkers)

	// Concurrent inserts
	for worker := 0; worker < numWorkers; worker++ {
		go func(workerID int) {
			defer func() { doneChan <- true }()

			for i := 0; i < vectorsPerWorker; i++ {
				vector := &api.Vector{
					ID:   fmt.Sprintf("concurrent-vec-%d-%d", workerID, i),
					Data: make([]float32, 128),
					Metadata: map[string]interface{}{
						"worker": workerID,
						"index":  i,
					},
				}

				if err := collection.Add(ctx, vector); err != nil {
					errChan <- fmt.Errorf("worker %d failed to add vector %d: %w", workerID, i, err)
					return
				}

				// Brief pause to increase chance of concurrent operations
				time.Sleep(1 * time.Millisecond)
			}
		}(worker)
	}

	// Wait for all workers to complete
	for i := 0; i < numWorkers; i++ {
		select {
		case err := <-errChan:
			t.Errorf("Concurrent operation error: %v", err)
		case <-doneChan:
			// Worker completed successfully
		case <-time.After(30 * time.Second):
			t.Fatal("Concurrent operations timed out")
		}
	}

	// Verify final state
	expectedCount := int64(numWorkers * vectorsPerWorker)
	actualCount, err := collection.Count(ctx)
	if err != nil {
		t.Errorf("Failed to get count after concurrent operations: %v", err)
	}
	if actualCount != expectedCount {
		t.Errorf("Count mismatch after concurrent operations: expected %d, got %d", expectedCount, actualCount)
	}

	// Verify data integrity by checking a few random vectors
	for worker := 0; worker < numWorkers; worker++ {
		vectorID := fmt.Sprintf("concurrent-vec-%d-0", worker)
		vector, err := collection.Get(ctx, vectorID)
		if err != nil {
			t.Errorf("Failed to get vector %s: %v", vectorID, err)
		}
		if vector == nil || vector.ID != vectorID {
			t.Errorf("Vector %s not found or corrupted", vectorID)
		}
	}
}
