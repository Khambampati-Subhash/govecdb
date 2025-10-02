// Package segment provides comprehensive tests for the segment system.
package segment

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// TestSegmentManagerLifecycle tests the basic lifecycle of segment manager
func TestSegmentManagerLifecycle(t *testing.T) {
	config := DefaultSegmentManagerConfig()
	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		t.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()

	// Test start
	if err := manager.Start(ctx); err != nil {
		t.Fatalf("Failed to start segment manager: %v", err)
	}

	if !manager.IsRunning() {
		t.Fatal("Manager should be running after start")
	}

	// Test stop
	if err := manager.Stop(ctx); err != nil {
		t.Fatalf("Failed to stop segment manager: %v", err)
	}

	if manager.IsRunning() {
		t.Fatal("Manager should not be running after stop")
	}
}

// TestSegmentManagerBasicOperations tests basic CRUD operations
func TestSegmentManagerBasicOperations(t *testing.T) {
	config := DefaultSegmentManagerConfig()
	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		t.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		t.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	// Test vector operations
	vector := &api.Vector{
		ID:       "test-vector-1",
		Data:     []float32{1.0, 2.0, 3.0, 4.0},
		Metadata: map[string]interface{}{"category": "test"},
	}

	// Test Put
	if err := manager.Put(ctx, vector); err != nil {
		t.Fatalf("Failed to put vector: %v", err)
	}

	// Test Get
	retrieved, err := manager.Get(ctx, "test-vector-1")
	if err != nil {
		t.Fatalf("Failed to get vector: %v", err)
	}

	if retrieved.ID != vector.ID {
		t.Fatalf("Retrieved vector ID mismatch: expected %s, got %s", vector.ID, retrieved.ID)
	}

	// Test Delete
	if err := manager.Delete(ctx, "test-vector-1"); err != nil {
		t.Fatalf("Failed to delete vector: %v", err)
	}

	// Verify deletion
	_, err = manager.Get(ctx, "test-vector-1")
	if err != api.ErrVectorNotFound {
		t.Fatalf("Expected vector not found error after deletion, got: %v", err)
	}
}

// TestSegmentManagerBatchOperations tests batch operations
func TestSegmentManagerBatchOperations(t *testing.T) {
	config := DefaultSegmentManagerConfig()
	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		t.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		t.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	// Create batch of vectors
	vectors := make([]*api.Vector, 100)
	ids := make([]string, 100)
	for i := 0; i < 100; i++ {
		id := fmt.Sprintf("batch-vector-%d", i)
		vectors[i] = &api.Vector{
			ID:       id,
			Data:     []float32{float32(i), float32(i + 1), float32(i + 2)},
			Metadata: map[string]interface{}{"batch": i},
		}
		ids[i] = id
	}

	// Test PutBatch
	if err := manager.PutBatch(ctx, vectors); err != nil {
		t.Fatalf("Failed to put batch: %v", err)
	}

	// Test GetBatch
	retrieved, err := manager.GetBatch(ctx, ids)
	if err != nil {
		t.Fatalf("Failed to get batch: %v", err)
	}

	if len(retrieved) != len(vectors) {
		t.Fatalf("Retrieved batch size mismatch: expected %d, got %d", len(vectors), len(retrieved))
	}

	// Test DeleteBatch
	if err := manager.DeleteBatch(ctx, ids); err != nil {
		t.Fatalf("Failed to delete batch: %v", err)
	}

	// Verify deletion
	retrieved, err = manager.GetBatch(ctx, ids)
	if err != nil {
		t.Fatalf("Unexpected error getting batch after deletion: %v", err)
	}

	if len(retrieved) != 0 {
		t.Fatalf("Expected empty batch after deletion, got %d vectors", len(retrieved))
	}
}

// TestSegmentManagerMultipleSegments tests operations across multiple segments
func TestSegmentManagerMultipleSegments(t *testing.T) {
	config := DefaultSegmentManagerConfig()
	// Set small segment size to force multiple segments
	config.DefaultSegmentConfig.MaxVectors = 10

	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		t.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		t.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	// Add vectors to fill multiple segments
	vectorCount := 25
	for i := 0; i < vectorCount; i++ {
		vector := &api.Vector{
			ID:       fmt.Sprintf("multi-seg-vector-%d", i),
			Data:     []float32{float32(i), float32(i + 1)},
			Metadata: map[string]interface{}{"index": i},
		}

		if err := manager.Put(ctx, vector); err != nil {
			t.Fatalf("Failed to put vector %d: %v", i, err)
		}
	}

	// Verify we have multiple segments
	segments, err := manager.ListSegments(ctx)
	if err != nil {
		t.Fatalf("Failed to list segments: %v", err)
	}

	if len(segments) < 2 {
		t.Fatalf("Expected multiple segments, got %d", len(segments))
	}

	// Test retrieval from multiple segments
	for i := 0; i < vectorCount; i++ {
		id := fmt.Sprintf("multi-seg-vector-%d", i)
		vector, err := manager.Get(ctx, id)
		if err != nil {
			t.Fatalf("Failed to get vector %s: %v", id, err)
		}

		if vector.ID != id {
			t.Fatalf("Retrieved vector ID mismatch: expected %s, got %s", id, vector.ID)
		}
	}
}

// TestSegmentManagerFiltering tests filtering capabilities
func TestSegmentManagerFiltering(t *testing.T) {
	config := DefaultSegmentManagerConfig()
	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		t.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		t.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	// Add vectors with different categories
	categories := []string{"type_a", "type_b", "type_c"}
	for i, category := range categories {
		for j := 0; j < 5; j++ {
			vector := &api.Vector{
				ID:   fmt.Sprintf("%s-vector-%d", category, j),
				Data: []float32{float32(i), float32(j)},
				Metadata: map[string]interface{}{
					"category": category,
					"value":    i*10 + j,
				},
			}

			if err := manager.Put(ctx, vector); err != nil {
				t.Fatalf("Failed to put vector: %v", err)
			}
		}
	}

	// Test filtering by category
	filter := &api.FieldFilter{Field: "category", Op: api.FilterEq, Value: "type_a"}
	results, err := manager.Filter(ctx, filter)
	if err != nil {
		t.Fatalf("Failed to filter vectors: %v", err)
	}

	if len(results) != 5 {
		t.Fatalf("Expected 5 vectors with category 'type_a', got %d", len(results))
	}

	for _, vector := range results {
		if vector.Metadata["category"] != "type_a" {
			t.Fatalf("Filter returned vector with wrong category: %v", vector.Metadata["category"])
		}
	}
}

// TestSegmentManagerScan tests the scan functionality
func TestSegmentManagerScan(t *testing.T) {
	config := DefaultSegmentManagerConfig()
	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		t.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		t.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	// Add test vectors
	vectorCount := 20
	for i := 0; i < vectorCount; i++ {
		vector := &api.Vector{
			ID:       fmt.Sprintf("scan-vector-%d", i),
			Data:     []float32{float32(i)},
			Metadata: map[string]interface{}{"index": i},
		}

		if err := manager.Put(ctx, vector); err != nil {
			t.Fatalf("Failed to put vector: %v", err)
		}
	}

	// Test scan
	scannedCount := 0
	err = manager.Scan(ctx, func(vector *api.Vector) bool {
		scannedCount++
		return true // Continue scanning
	})

	if err != nil {
		t.Fatalf("Failed to scan vectors: %v", err)
	}

	if scannedCount != vectorCount {
		t.Fatalf("Expected to scan %d vectors, got %d", vectorCount, scannedCount)
	}

	// Test early termination
	scannedCount = 0
	err = manager.Scan(ctx, func(vector *api.Vector) bool {
		scannedCount++
		return scannedCount < 5 // Stop after 5 vectors
	})

	if err != nil {
		t.Fatalf("Failed to scan vectors with early termination: %v", err)
	}

	if scannedCount != 5 {
		t.Fatalf("Expected early termination after 5 vectors, got %d", scannedCount)
	}
}

// TestSegmentManagerStats tests statistics functionality
func TestSegmentManagerStats(t *testing.T) {
	config := DefaultSegmentManagerConfig()
	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		t.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		t.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	// Add some vectors
	for i := 0; i < 10; i++ {
		vector := &api.Vector{
			ID:       fmt.Sprintf("stats-vector-%d", i),
			Data:     []float32{float32(i)},
			Metadata: map[string]interface{}{"index": i},
		}

		if err := manager.Put(ctx, vector); err != nil {
			t.Fatalf("Failed to put vector: %v", err)
		}
	}

	// Get statistics
	stats, err := manager.Stats(ctx)
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.TotalVectors != 10 {
		t.Fatalf("Expected 10 total vectors in stats, got %d", stats.TotalVectors)
	}

	if stats.TotalSegments == 0 {
		t.Fatalf("Expected at least 1 segment in stats, got %d", stats.TotalSegments)
	}

	if stats.ActiveSegments == 0 {
		t.Fatalf("Expected at least 1 active segment in stats, got %d", stats.ActiveSegments)
	}
}

// TestSegmentManagerHealth tests health monitoring
func TestSegmentManagerHealth(t *testing.T) {
	config := DefaultSegmentManagerConfig()
	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		t.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		t.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	// Get health status
	health, err := manager.Health(ctx)
	if err != nil {
		t.Fatalf("Failed to get health: %v", err)
	}

	if health.HealthScore < 0 || health.HealthScore > 1 {
		t.Fatalf("Health score should be between 0 and 1, got %f", health.HealthScore)
	}

	if health.Status == "" {
		t.Fatal("Health status should not be empty")
	}
}

// TestSegmentManagerCompaction tests compaction functionality
func TestSegmentManagerCompaction(t *testing.T) {
	config := DefaultSegmentManagerConfig()
	config.AutoCompactionEnabled = false       // Disable auto compaction for controlled testing
	config.DefaultSegmentConfig.MaxVectors = 5 // Small segments to trigger compaction

	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		t.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		t.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	// Add enough vectors to create multiple segments
	for i := 0; i < 15; i++ {
		vector := &api.Vector{
			ID:       fmt.Sprintf("compact-vector-%d", i),
			Data:     []float32{float32(i)},
			Metadata: map[string]interface{}{"index": i},
		}

		if err := manager.Put(ctx, vector); err != nil {
			t.Fatalf("Failed to put vector: %v", err)
		}
	}

	// Check that we have multiple segments
	segments, err := manager.ListSegments(ctx)
	if err != nil {
		t.Fatalf("Failed to list segments: %v", err)
	}

	if len(segments) < 2 {
		t.Fatalf("Expected multiple segments before compaction, got %d", len(segments))
	}

	// Trigger compaction
	if err := manager.TriggerCompaction(ctx); err != nil {
		t.Fatalf("Failed to trigger compaction: %v", err)
	}

	// Wait a bit for compaction to potentially complete
	time.Sleep(500 * time.Millisecond)

	// Get compaction status
	status, err := manager.GetCompactionStatus(ctx)
	if err != nil {
		t.Fatalf("Failed to get compaction status: %v", err)
	}

	// Status should indicate some compaction activity or completion
	if status == nil {
		t.Fatal("Compaction status should not be nil")
	}
}

// TestSegmentRotation tests active segment rotation
func TestSegmentRotation(t *testing.T) {
	config := DefaultSegmentManagerConfig()
	config.DefaultSegmentConfig.MaxVectors = 5 // Small segments for easy rotation

	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		t.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		t.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	// Get initial active segment
	activeSegment1, err := manager.GetActiveSegment(ctx)
	if err != nil {
		t.Fatalf("Failed to get active segment: %v", err)
	}

	// Fill the active segment
	for i := 0; i < 5; i++ {
		vector := &api.Vector{
			ID:       fmt.Sprintf("rotation-vector-%d", i),
			Data:     []float32{float32(i)},
			Metadata: map[string]interface{}{"index": i},
		}

		if err := manager.Put(ctx, vector); err != nil {
			t.Fatalf("Failed to put vector: %v", err)
		}
	}

	// Add one more vector which should trigger rotation
	vector := &api.Vector{
		ID:       "rotation-vector-trigger",
		Data:     []float32{99.0},
		Metadata: map[string]interface{}{"trigger": true},
	}

	if err := manager.Put(ctx, vector); err != nil {
		t.Fatalf("Failed to put trigger vector: %v", err)
	}

	// Get new active segment
	activeSegment2, err := manager.GetActiveSegment(ctx)
	if err != nil {
		t.Fatalf("Failed to get active segment after rotation: %v", err)
	}

	// Should be different segments
	if activeSegment1.ID() == activeSegment2.ID() {
		t.Fatal("Active segment should have rotated but IDs are the same")
	}

	// Verify all vectors are still accessible
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("rotation-vector-%d", i)
		_, err := manager.Get(ctx, id)
		if err != nil {
			t.Fatalf("Failed to get vector %s after rotation: %v", id, err)
		}
	}

	_, err = manager.Get(ctx, "rotation-vector-trigger")
	if err != nil {
		t.Fatalf("Failed to get trigger vector after rotation: %v", err)
	}
}

// TestCompactionPolicy tests the default compaction policy
func TestCompactionPolicy(t *testing.T) {
	policy := NewDefaultCompactionPolicy()

	// Test with no segments
	shouldCompact := policy.ShouldCompact([]Segment{})
	if shouldCompact {
		t.Fatal("Should not compact with no segments")
	}

	// Test segment selection with no segments
	selected := policy.SelectSegmentsForCompaction([]Segment{})
	if len(selected) != 0 {
		t.Fatalf("Should select no segments when none available, got %d", len(selected))
	}

	// Test benefit estimation with no segments
	benefit := policy.EstimateCompactionBenefit([]Segment{})
	if benefit.NetBenefit != 0 {
		t.Fatalf("Expected zero benefit for no segments, got %f", benefit.NetBenefit)
	}

	// Test thresholds
	thresholds := policy.GetThresholds()
	if thresholds.MinSegmentsToCompact <= 0 {
		t.Fatal("MinSegmentsToCompact should be positive")
	}

	// Test setting thresholds
	newThresholds := thresholds
	newThresholds.MinSegmentsToCompact = 5
	policy.SetThresholds(newThresholds)

	updatedThresholds := policy.GetThresholds()
	if updatedThresholds.MinSegmentsToCompact != 5 {
		t.Fatalf("Expected MinSegmentsToCompact to be 5, got %d", updatedThresholds.MinSegmentsToCompact)
	}
}

// BenchmarkSegmentManagerPut benchmarks put operations
func BenchmarkSegmentManagerPut(b *testing.B) {
	config := DefaultSegmentManagerConfig()
	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		b.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		b.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		vector := &api.Vector{
			ID:       fmt.Sprintf("bench-vector-%d", i),
			Data:     []float32{float32(i), float32(i + 1), float32(i + 2)},
			Metadata: map[string]interface{}{"index": i},
		}

		if err := manager.Put(ctx, vector); err != nil {
			b.Fatalf("Failed to put vector: %v", err)
		}
	}
}

// BenchmarkSegmentManagerGet benchmarks get operations
func BenchmarkSegmentManagerGet(b *testing.B) {
	config := DefaultSegmentManagerConfig()
	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		b.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		b.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	// Pre-populate with test data
	vectorCount := 10000
	for i := 0; i < vectorCount; i++ {
		vector := &api.Vector{
			ID:       fmt.Sprintf("bench-get-vector-%d", i),
			Data:     []float32{float32(i)},
			Metadata: map[string]interface{}{"index": i},
		}

		if err := manager.Put(ctx, vector); err != nil {
			b.Fatalf("Failed to put vector during setup: %v", err)
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		id := fmt.Sprintf("bench-get-vector-%d", i%vectorCount)
		_, err := manager.Get(ctx, id)
		if err != nil {
			b.Fatalf("Failed to get vector: %v", err)
		}
	}
}

// BenchmarkSegmentManagerBatchPut benchmarks batch put operations
func BenchmarkSegmentManagerBatchPut(b *testing.B) {
	config := DefaultSegmentManagerConfig()
	manager, err := NewConcurrentSegmentManager(config)
	if err != nil {
		b.Fatalf("Failed to create segment manager: %v", err)
	}

	ctx := context.Background()
	if err := manager.Start(ctx); err != nil {
		b.Fatalf("Failed to start segment manager: %v", err)
	}
	defer manager.Stop(ctx)

	batchSize := 100
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		vectors := make([]*api.Vector, batchSize)
		for j := 0; j < batchSize; j++ {
			vectors[j] = &api.Vector{
				ID:       fmt.Sprintf("bench-batch-vector-%d-%d", i, j),
				Data:     []float32{float32(j)},
				Metadata: map[string]interface{}{"batch": i, "index": j},
			}
		}

		if err := manager.PutBatch(ctx, vectors); err != nil {
			b.Fatalf("Failed to put batch: %v", err)
		}
	}
}
