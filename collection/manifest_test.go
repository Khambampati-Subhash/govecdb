package collection

// Package collection provides tests for the manifest management system.

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

func TestManifestManager(t *testing.T) {
	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "manifest_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manifestPath := tempDir + "/manifest.json"
	config := DefaultManifestConfig(manifestPath)

	mm, err := NewManifestManager(config)
	if err != nil {
		t.Fatalf("Failed to create manifest manager: %v", err)
	}
	defer mm.Close(context.Background())

	ctx := context.Background()

	t.Run("CreateCollection", func(t *testing.T) {
		collectionConfig := &api.CollectionConfig{
			Name:           "test-collection",
			Dimension:      384,
			Metric:         api.Cosine,
			M:              16,
			EfConstruction: 200,
			MaxLayer:       16,
			ThreadSafe:     true,
		}

		err := mm.CreateCollection(ctx, collectionConfig)
		if err != nil {
			t.Errorf("Failed to create collection: %v", err)
		}

		// Try to create the same collection again - should fail
		err = mm.CreateCollection(ctx, collectionConfig)
		if err != api.ErrCollectionExists {
			t.Errorf("Expected ErrCollectionExists, got: %v", err)
		}
	})

	t.Run("GetCollection", func(t *testing.T) {
		info, err := mm.GetCollection(ctx, "test-collection")
		if err != nil {
			t.Errorf("Failed to get collection: %v", err)
		}

		if info.Name != "test-collection" {
			t.Errorf("Expected collection name 'test-collection', got '%s'", info.Name)
		}

		if info.Config.Dimension != 384 {
			t.Errorf("Expected dimension 384, got %d", info.Config.Dimension)
		}

		if info.State != CollectionStateCreating {
			t.Errorf("Expected state 'creating', got '%s'", info.State)
		}

		// Try to get non-existent collection
		_, err = mm.GetCollection(ctx, "non-existent")
		if err != api.ErrCollectionNotFound {
			t.Errorf("Expected ErrCollectionNotFound, got: %v", err)
		}
	})

	t.Run("UpdateCollection", func(t *testing.T) {
		updateInfo := &CollectionInfo{
			Name:        "test-collection",
			State:       CollectionStateActive,
			VectorCount: 1000,
			IndexSize:   1024 * 1024,
			DataSize:    2048 * 1024,
			Properties: map[string]interface{}{
				"custom_field": "custom_value",
			},
		}

		err := mm.UpdateCollection(ctx, updateInfo)
		if err != nil {
			t.Errorf("Failed to update collection: %v", err)
		}

		// Verify update
		info, err := mm.GetCollection(ctx, "test-collection")
		if err != nil {
			t.Errorf("Failed to get updated collection: %v", err)
		}

		if info.State != CollectionStateActive {
			t.Errorf("Expected state 'active', got '%s'", info.State)
		}

		if info.VectorCount != 1000 {
			t.Errorf("Expected vector count 1000, got %d", info.VectorCount)
		}

		if info.Properties["custom_field"] != "custom_value" {
			t.Errorf("Custom property not updated correctly")
		}
	})

	t.Run("ListCollections", func(t *testing.T) {
		names, err := mm.ListCollections(ctx)
		if err != nil {
			t.Errorf("Failed to list collections: %v", err)
		}

		if len(names) != 1 {
			t.Errorf("Expected 1 collection, got %d", len(names))
		}

		if names[0] != "test-collection" {
			t.Errorf("Expected 'test-collection', got '%s'", names[0])
		}
	})

	t.Run("SetCollectionState", func(t *testing.T) {
		err := mm.SetCollectionState(ctx, "test-collection", CollectionStateOptimizing)
		if err != nil {
			t.Errorf("Failed to set collection state: %v", err)
		}

		info, err := mm.GetCollection(ctx, "test-collection")
		if err != nil {
			t.Errorf("Failed to get collection after state change: %v", err)
		}

		if info.State != CollectionStateOptimizing {
			t.Errorf("Expected state 'optimizing', got '%s'", info.State)
		}
	})

	t.Run("RecoveryState", func(t *testing.T) {
		// Update recovery state
		err := mm.UpdateRecoveryState(ctx, 12345, "snapshot-123", "recovering")
		if err != nil {
			t.Errorf("Failed to update recovery state: %v", err)
		}

		// Get recovery info
		lastLSN, snapshotID, recoveryState, err := mm.GetRecoveryInfo(ctx)
		if err != nil {
			t.Errorf("Failed to get recovery info: %v", err)
		}

		if lastLSN != 12345 {
			t.Errorf("Expected LSN 12345, got %d", lastLSN)
		}

		if snapshotID != "snapshot-123" {
			t.Errorf("Expected snapshot ID 'snapshot-123', got '%s'", snapshotID)
		}

		if recoveryState != "recovering" {
			t.Errorf("Expected recovery state 'recovering', got '%s'", recoveryState)
		}
	})

	t.Run("Properties", func(t *testing.T) {
		// Set a property
		err := mm.SetProperty(ctx, "test_key", "test_value")
		if err != nil {
			t.Errorf("Failed to set property: %v", err)
		}

		// Get the property
		value, exists := mm.GetProperty(ctx, "test_key")
		if !exists {
			t.Error("Property should exist")
		}

		if value != "test_value" {
			t.Errorf("Expected 'test_value', got '%v'", value)
		}

		// Get non-existent property
		_, exists = mm.GetProperty(ctx, "non_existent")
		if exists {
			t.Error("Non-existent property should not exist")
		}
	})

	t.Run("DropCollection", func(t *testing.T) {
		// First verify the collection exists
		_, err := mm.GetCollection(ctx, "test-collection")
		if err != nil {
			t.Errorf("Collection should exist before drop: %v", err)
		}

		// Drop the collection
		err = mm.DropCollection(ctx, "test-collection")
		if err != nil {
			t.Errorf("Failed to drop collection: %v", err)
		}

		// Verify it's gone
		_, err = mm.GetCollection(ctx, "test-collection")
		if err != api.ErrCollectionNotFound {
			t.Errorf("Expected ErrCollectionNotFound after drop, got: %v", err)
		}

		// List should be empty
		names, err := mm.ListCollections(ctx)
		if err != nil {
			t.Errorf("Failed to list collections after drop: %v", err)
		}

		if len(names) != 0 {
			t.Errorf("Expected 0 collections after drop, got %d", len(names))
		}
	})
}

func TestManifestPersistence(t *testing.T) {
	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "manifest_persistence_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manifestPath := tempDir + "/manifest.json"
	config := DefaultManifestConfig(manifestPath)

	ctx := context.Background()

	// Create first manifest manager and add data
	mm1, err := NewManifestManager(config)
	if err != nil {
		t.Fatalf("Failed to create first manifest manager: %v", err)
	}

	collectionConfig := &api.CollectionConfig{
		Name:      "persistent-collection",
		Dimension: 128,
		Metric:    api.Euclidean,
	}

	err = mm1.CreateCollection(ctx, collectionConfig)
	if err != nil {
		t.Errorf("Failed to create collection: %v", err)
	}

	err = mm1.SetProperty(ctx, "version", "1.0.0")
	if err != nil {
		t.Errorf("Failed to set property: %v", err)
	}

	// Save and close
	err = mm1.Save(ctx)
	if err != nil {
		t.Errorf("Failed to save manifest: %v", err)
	}

	err = mm1.Close(ctx)
	if err != nil {
		t.Errorf("Failed to close manifest manager: %v", err)
	}

	// Create second manifest manager and verify data persisted
	mm2, err := NewManifestManager(config)
	if err != nil {
		t.Fatalf("Failed to create second manifest manager: %v", err)
	}
	defer mm2.Close(ctx)

	// Verify collection exists
	info, err := mm2.GetCollection(ctx, "persistent-collection")
	if err != nil {
		t.Errorf("Failed to get persisted collection: %v", err)
	}

	if info.Config.Dimension != 128 {
		t.Errorf("Expected dimension 128, got %d", info.Config.Dimension)
	}

	if info.Config.Metric != api.Euclidean {
		t.Errorf("Expected Euclidean metric, got %v", info.Config.Metric)
	}

	// Verify property exists
	value, exists := mm2.GetProperty(ctx, "version")
	if !exists {
		t.Error("Property should be persisted")
	}

	if value != "1.0.0" {
		t.Errorf("Expected version '1.0.0', got '%v'", value)
	}
}

func TestManifestAutoSave(t *testing.T) {
	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "manifest_autosave_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manifestPath := tempDir + "/manifest.json"
	config := DefaultManifestConfig(manifestPath)
	config.AutoSave = true
	config.SaveInterval = 100 * time.Millisecond

	mm, err := NewManifestManager(config)
	if err != nil {
		t.Fatalf("Failed to create manifest manager: %v", err)
	}
	defer mm.Close(context.Background())

	ctx := context.Background()

	// Add a collection
	collectionConfig := &api.CollectionConfig{
		Name:      "autosave-collection",
		Dimension: 256,
		Metric:    api.DotProduct,
	}

	err = mm.CreateCollection(ctx, collectionConfig)
	if err != nil {
		t.Errorf("Failed to create collection: %v", err)
	}

	// Wait for auto-save to trigger
	time.Sleep(200 * time.Millisecond)

	// Create new manager to verify auto-save worked
	mm2, err := NewManifestManager(config)
	if err != nil {
		t.Fatalf("Failed to create second manifest manager: %v", err)
	}
	defer mm2.Close(ctx)

	// Verify collection was auto-saved
	_, err = mm2.GetCollection(ctx, "autosave-collection")
	if err != nil {
		t.Errorf("Auto-saved collection not found: %v", err)
	}
}

func TestManifestValidation(t *testing.T) {
	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "manifest_validation_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manifestPath := tempDir + "/manifest.json"
	config := DefaultManifestConfig(manifestPath)

	mm, err := NewManifestManager(config)
	if err != nil {
		t.Fatalf("Failed to create manifest manager: %v", err)
	}
	defer mm.Close(context.Background())

	ctx := context.Background()

	t.Run("ValidManifest", func(t *testing.T) {
		err := mm.ValidateManifest(ctx)
		if err != nil {
			t.Errorf("Valid manifest failed validation: %v", err)
		}
	})

	t.Run("InvalidCollectionConfig", func(t *testing.T) {
		// Create collection with invalid config
		invalidConfig := &api.CollectionConfig{
			Name:      "invalid-collection",
			Dimension: -1, // Invalid dimension
		}

		err := mm.CreateCollection(ctx, invalidConfig)
		if err == nil {
			t.Error("Should have failed to create collection with invalid config")
		}
	})

	t.Run("RepairManifest", func(t *testing.T) {
		// Create a new manifest manager for clean state
		repairPath := filepath.Join(tempDir, "repair_manifest.json")
		repairConfig := &ManifestConfig{
			ManifestPath: repairPath,
			AutoSave:     true,
			SaveInterval: time.Second,
		}
		repairMM, err := NewManifestManager(repairConfig)
		if err != nil {
			t.Fatalf("Failed to create repair manifest manager: %v", err)
		}
		defer repairMM.Close(ctx)

		err = repairMM.RepairManifest(ctx)
		if err != nil {
			t.Errorf("Failed to repair manifest: %v", err)
		}

		// Validate after repair - should pass with clean manifest
		err = repairMM.ValidateManifest(ctx)
		if err != nil {
			t.Errorf("Manifest validation failed after repair: %v", err)
		}
	})
}

func BenchmarkManifestOperations(b *testing.B) {
	// Create temporary directory for benchmark
	tempDir, err := os.MkdirTemp("", "manifest_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manifestPath := tempDir + "/manifest.json"
	config := DefaultManifestConfig(manifestPath)
	config.AutoSave = false // Disable auto-save for benchmark

	mm, err := NewManifestManager(config)
	if err != nil {
		b.Fatalf("Failed to create manifest manager: %v", err)
	}
	defer mm.Close(context.Background())

	ctx := context.Background()

	b.Run("CreateCollection", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			config := &api.CollectionConfig{
				Name:      fmt.Sprintf("bench-collection-%d-%d", time.Now().UnixNano(), i),
				Dimension: 384,
				Metric:    api.Cosine,
			}
			err := mm.CreateCollection(ctx, config)
			if err != nil && !strings.Contains(err.Error(), "already exists") {
				b.Errorf("Failed to create collection: %v", err)
			}
		}
	})

	b.Run("GetCollection", func(b *testing.B) {
		// Create a collection first
		config := &api.CollectionConfig{
			Name:      "get-bench-collection",
			Dimension: 384,
			Metric:    api.Cosine,
		}
		mm.CreateCollection(ctx, config)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := mm.GetCollection(ctx, "get-bench-collection")
			if err != nil {
				b.Errorf("Failed to get collection: %v", err)
			}
		}
	})

	b.Run("UpdateCollection", func(b *testing.B) {
		// Create a collection first
		config := &api.CollectionConfig{
			Name:      "update-bench-collection",
			Dimension: 384,
			Metric:    api.Cosine,
		}
		mm.CreateCollection(ctx, config)

		updateInfo := &CollectionInfo{
			Name:        "update-bench-collection",
			State:       CollectionStateActive,
			VectorCount: 1000,
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			updateInfo.VectorCount = int64(i)
			err := mm.UpdateCollection(ctx, updateInfo)
			if err != nil {
				b.Errorf("Failed to update collection: %v", err)
			}
		}
	})
}

func TestManifestConcurrency(t *testing.T) {
	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "manifest_concurrency_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manifestPath := tempDir + "/manifest.json"
	config := DefaultManifestConfig(manifestPath)
	config.AutoSave = false // Control saves manually

	mm, err := NewManifestManager(config)
	if err != nil {
		t.Fatalf("Failed to create manifest manager: %v", err)
	}
	defer mm.Close(context.Background())

	ctx := context.Background()
	numWorkers := 10
	collectionsPerWorker := 10

	// Test concurrent collection creation
	t.Run("ConcurrentCreate", func(t *testing.T) {
		var wg sync.WaitGroup
		errors := make(chan error, numWorkers)

		for worker := 0; worker < numWorkers; worker++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for i := 0; i < collectionsPerWorker; i++ {
					config := &api.CollectionConfig{
						Name:      fmt.Sprintf("concurrent-collection-%d-%d", workerID, i),
						Dimension: 384,
						Metric:    api.Cosine,
					}
					if err := mm.CreateCollection(ctx, config); err != nil {
						errors <- err
						return
					}
				}
			}(worker)
		}

		wg.Wait()
		close(errors)

		for err := range errors {
			t.Errorf("Concurrent create error: %v", err)
		}

		// Verify all collections were created
		names, err := mm.ListCollections(ctx)
		if err != nil {
			t.Errorf("Failed to list collections: %v", err)
		}

		expectedCount := numWorkers * collectionsPerWorker
		if len(names) != expectedCount {
			t.Errorf("Expected %d collections, got %d", expectedCount, len(names))
		}
	})
}
