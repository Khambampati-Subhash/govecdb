package persist

// Package persist provides tests for the persistence layer components.

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// TestWAL tests the Write Ahead Log functionality
func TestWAL(t *testing.T) {
	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "wal_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create WAL configuration
	config := DefaultWALConfig(tempDir)
	config.SyncWrites = true
	config.FlushInterval = 100 * time.Millisecond

	// Create WAL
	wal, err := NewWAL(config)
	if err != nil {
		t.Fatalf("Failed to create WAL: %v", err)
	}

	ctx := context.Background()

	// Start WAL
	if err := wal.Start(ctx); err != nil {
		t.Fatalf("Failed to start WAL: %v", err)
	}
	defer wal.Stop(ctx)

	t.Run("WriteRecord", func(t *testing.T) {
		// Set shorter timeout for tests
		ctxTimeout, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()

		// Test writing a single record
		record := &WALRecord{
			Type: RecordTypeInsert,
			Vector: &api.Vector{
				ID:   "test-1",
				Data: []float32{1.0, 2.0, 3.0},
				Metadata: map[string]interface{}{
					"category": "test",
				},
			},
			TxnID: "txn-1",
		}

		err := wal.WriteRecord(ctxTimeout, record)
		if err != nil {
			t.Errorf("Failed to write record: %v", err)
		}

		// Verify LSN was assigned
		if record.LSN == 0 {
			t.Error("LSN was not assigned to record")
		}

		// Verify checksum was calculated
		if record.Checksum == 0 {
			t.Error("Checksum was not calculated for record")
		}
	})

	t.Run("WriteBatch", func(t *testing.T) {
		// Test writing multiple records
		records := []*WALRecord{
			{
				Type: RecordTypeInsert,
				Vector: &api.Vector{
					ID:   "test-2",
					Data: []float32{4.0, 5.0, 6.0},
				},
				TxnID: "txn-2",
			},
			{
				Type: RecordTypeInsert,
				Vector: &api.Vector{
					ID:   "test-3",
					Data: []float32{7.0, 8.0, 9.0},
				},
				TxnID: "txn-2",
			},
		}

		err := wal.WriteBatch(ctx, records)
		if err != nil {
			t.Errorf("Failed to write batch: %v", err)
		}

		// Verify LSNs were assigned and are sequential
		if records[0].LSN == 0 || records[1].LSN == 0 {
			t.Error("LSNs were not assigned to batch records")
		}
		if records[1].LSN != records[0].LSN+1 {
			t.Error("Batch LSNs are not sequential")
		}
	})

	t.Run("GetLastLSN", func(t *testing.T) {
		lastLSN := wal.GetLastLSN()
		if lastLSN == 0 {
			t.Error("Last LSN should not be 0 after writing records")
		}
	})

	t.Run("Checkpoint", func(t *testing.T) {
		checkpointLSN, err := wal.Checkpoint(ctx)
		if err != nil {
			t.Errorf("Failed to create checkpoint: %v", err)
		}

		if checkpointLSN == 0 {
			t.Error("Checkpoint LSN should not be 0")
		}
	})

	t.Run("Sync", func(t *testing.T) {
		err := wal.Sync()
		if err != nil {
			t.Errorf("Failed to sync WAL: %v", err)
		}
	})
}

// MockReplayHandler implements ReplayHandler for testing
type MockReplayHandler struct {
	InsertCount      int
	UpdateCount      int
	DeleteCount      int
	BatchInsertCount int
	BatchDeleteCount int
	ClearCount       int
	SnapshotCount    int
	Records          []*WALRecord
}

func (m *MockReplayHandler) HandleInsert(ctx context.Context, record *WALRecord) error {
	m.InsertCount++
	m.Records = append(m.Records, record)
	return nil
}

func (m *MockReplayHandler) HandleUpdate(ctx context.Context, record *WALRecord) error {
	m.UpdateCount++
	m.Records = append(m.Records, record)
	return nil
}

func (m *MockReplayHandler) HandleDelete(ctx context.Context, record *WALRecord) error {
	m.DeleteCount++
	m.Records = append(m.Records, record)
	return nil
}

func (m *MockReplayHandler) HandleBatchInsert(ctx context.Context, record *WALRecord) error {
	m.BatchInsertCount++
	m.Records = append(m.Records, record)
	return nil
}

func (m *MockReplayHandler) HandleBatchDelete(ctx context.Context, record *WALRecord) error {
	m.BatchDeleteCount++
	m.Records = append(m.Records, record)
	return nil
}

func (m *MockReplayHandler) HandleClear(ctx context.Context, record *WALRecord) error {
	m.ClearCount++
	m.Records = append(m.Records, record)
	return nil
}

func (m *MockReplayHandler) HandleSnapshot(ctx context.Context, record *WALRecord) error {
	m.SnapshotCount++
	m.Records = append(m.Records, record)
	return nil
}

func TestWALReplay(t *testing.T) {
	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "wal_replay_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultWALConfig(tempDir)
	ctx := context.Background()

	// Create and write records
	wal, err := NewWAL(config)
	if err != nil {
		t.Fatalf("Failed to create WAL: %v", err)
	}

	if err := wal.Start(ctx); err != nil {
		t.Fatalf("Failed to start WAL: %v", err)
	}

	// Write various types of records
	records := []*WALRecord{
		{
			Type: RecordTypeInsert,
			Vector: &api.Vector{
				ID:   "test-1",
				Data: []float32{1.0, 2.0, 3.0},
			},
		},
		{
			Type:     RecordTypeDelete,
			VectorID: "test-1",
		},
		{
			Type: RecordTypeBatchInsert,
			Vectors: []*api.Vector{
				{ID: "test-2", Data: []float32{4.0, 5.0, 6.0}},
				{ID: "test-3", Data: []float32{7.0, 8.0, 9.0}},
			},
		},
		{
			Type:      RecordTypeBatchDelete,
			VectorIDs: []string{"test-2", "test-3"},
		},
		{
			Type: RecordTypeClear,
		},
	}

	for _, record := range records {
		if err := wal.WriteRecord(ctx, record); err != nil {
			t.Errorf("Failed to write record: %v", err)
		}
	}

	// Sync to ensure all data is written before stopping
	if err := wal.Sync(); err != nil {
		t.Errorf("Failed to sync WAL: %v", err)
	}

	// Stop WAL to flush all data
	if err := wal.Stop(ctx); err != nil {
		t.Errorf("Failed to stop WAL: %v", err)
	}

	// Create new WAL instance for replay (with same config)
	replayConfig := DefaultWALConfig(tempDir)
	wal2, err := NewWAL(replayConfig)
	if err != nil {
		t.Fatalf("Failed to create WAL for replay: %v", err)
	}

	// Start the second WAL instance (it needs to be started to read files)
	if err := wal2.Start(ctx); err != nil {
		t.Fatalf("Failed to start second WAL: %v", err)
	}
	defer wal2.Stop(ctx)

	handler := &MockReplayHandler{}

	// Replay from LSN 0 (start from beginning)
	err = wal2.Replay(ctx, 0, handler)
	if err != nil {
		t.Errorf("Failed to replay WAL: %v", err)
	}

	// Verify replay results
	if handler.InsertCount != 1 {
		t.Errorf("Expected 1 insert, got %d", handler.InsertCount)
	}
	if handler.DeleteCount != 1 {
		t.Errorf("Expected 1 delete, got %d", handler.DeleteCount)
	}
	if handler.BatchInsertCount != 1 {
		t.Errorf("Expected 1 batch insert, got %d", handler.BatchInsertCount)
	}
	if handler.BatchDeleteCount != 1 {
		t.Errorf("Expected 1 batch delete, got %d", handler.BatchDeleteCount)
	}
	if handler.ClearCount != 1 {
		t.Errorf("Expected 1 clear, got %d", handler.ClearCount)
	}

	// Verify total record count
	expectedTotal := len(records)
	if len(handler.Records) != expectedTotal {
		t.Errorf("Expected %d records replayed, got %d", expectedTotal, len(handler.Records))
	}
}

// TestSnapshotManager tests the snapshot management functionality
func TestSnapshotManager(t *testing.T) {
	// Create temporary directory for test
	tempDir, err := os.MkdirTemp("", "snapshot_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultSnapshotConfig(tempDir)
	config.CompressionEnabled = true

	sm, err := NewSnapshotManager(config)
	if err != nil {
		t.Fatalf("Failed to create snapshot manager: %v", err)
	}

	ctx := context.Background()

	if err := sm.Start(ctx); err != nil {
		t.Fatalf("Failed to start snapshot manager: %v", err)
	}
	defer sm.Stop(ctx)

	t.Run("CreateSnapshot", func(t *testing.T) {
		// Create test data
		snapshotData := &CollectionSnapshot{
			Metadata: &api.CollectionConfig{
				Name:      "test-collection",
				Dimension: 3,
				Metric:    api.Cosine,
			},
			Vectors: []*api.Vector{
				{ID: "vec-1", Data: []float32{1.0, 2.0, 3.0}},
				{ID: "vec-2", Data: []float32{4.0, 5.0, 6.0}},
			},
		}

		metadata, err := sm.CreateSnapshot(ctx, 100, snapshotData)
		if err != nil {
			t.Errorf("Failed to create snapshot: %v", err)
		}

		if metadata.ID == "" {
			t.Error("Snapshot ID should not be empty")
		}
		if metadata.LSN != 100 {
			t.Errorf("Expected LSN 100, got %d", metadata.LSN)
		}
		if metadata.VectorCount != 2 {
			t.Errorf("Expected 2 vectors, got %d", metadata.VectorCount)
		}
		if metadata.Checksum == "" {
			t.Error("Checksum should not be empty")
		}
	})

	t.Run("ListSnapshots", func(t *testing.T) {
		snapshots, err := sm.ListSnapshots(ctx)
		if err != nil {
			t.Errorf("Failed to list snapshots: %v", err)
		}

		if len(snapshots) == 0 {
			t.Error("Expected at least one snapshot")
		}
	})

	t.Run("RestoreSnapshot", func(t *testing.T) {
		// Get the first snapshot
		snapshots, err := sm.ListSnapshots(ctx)
		if err != nil {
			t.Fatalf("Failed to list snapshots: %v", err)
		}

		if len(snapshots) == 0 {
			t.Fatal("No snapshots available for restore test")
		}

		snapshotID := snapshots[0].ID

		restoredData, err := sm.RestoreSnapshot(ctx, snapshotID)
		if err != nil {
			t.Errorf("Failed to restore snapshot: %v", err)
		}

		collectionSnapshot, ok := restoredData.(*CollectionSnapshot)
		if !ok {
			t.Error("Restored data is not a CollectionSnapshot")
		}

		if len(collectionSnapshot.Vectors) != 2 {
			t.Errorf("Expected 2 vectors in restored data, got %d", len(collectionSnapshot.Vectors))
		}
	})
}

// TestSnapshotCompression tests snapshot compression
func TestSnapshotCompression(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "compression_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	ctx := context.Background()

	// Test with compression enabled
	t.Run("CompressionEnabled", func(t *testing.T) {
		config := DefaultSnapshotConfig(filepath.Join(tempDir, "compressed"))
		config.CompressionEnabled = true

		sm, err := NewSnapshotManager(config)
		if err != nil {
			t.Fatalf("Failed to create snapshot manager: %v", err)
		}

		if err := sm.Start(ctx); err != nil {
			t.Fatalf("Failed to start snapshot manager: %v", err)
		}
		defer sm.Stop(ctx)

		// Create large test data (should compress well)
		vectors := make([]*api.Vector, 100)
		for i := 0; i < 100; i++ {
			vectors[i] = &api.Vector{
				ID:   fmt.Sprintf("vec-%d", i),
				Data: []float32{1.0, 1.0, 1.0, 1.0, 1.0}, // Repeated values compress well
			}
		}

		snapshotData := &CollectionSnapshot{
			Vectors: vectors,
		}

		metadata, err := sm.CreateSnapshot(ctx, 200, snapshotData)
		if err != nil {
			t.Errorf("Failed to create compressed snapshot: %v", err)
		}

		if !metadata.Compressed {
			t.Error("Expected snapshot to be compressed")
		}

		// Verify we can restore compressed data
		_, err = sm.RestoreSnapshot(ctx, metadata.ID)
		if err != nil {
			t.Errorf("Failed to restore compressed snapshot: %v", err)
		}
	})

	// Test with compression disabled
	t.Run("CompressionDisabled", func(t *testing.T) {
		config := DefaultSnapshotConfig(filepath.Join(tempDir, "uncompressed"))
		config.CompressionEnabled = false

		sm, err := NewSnapshotManager(config)
		if err != nil {
			t.Fatalf("Failed to create snapshot manager: %v", err)
		}

		if err := sm.Start(ctx); err != nil {
			t.Fatalf("Failed to start snapshot manager: %v", err)
		}
		defer sm.Stop(ctx)

		snapshotData := &CollectionSnapshot{
			Vectors: []*api.Vector{
				{ID: "vec-1", Data: []float32{1.0, 2.0, 3.0}},
			},
		}

		metadata, err := sm.CreateSnapshot(ctx, 300, snapshotData)
		if err != nil {
			t.Errorf("Failed to create uncompressed snapshot: %v", err)
		}

		if metadata.Compressed {
			t.Error("Expected snapshot to not be compressed")
		}
	})
}

// TestSnapshotCleanup tests snapshot cleanup functionality
func TestSnapshotCleanup(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "cleanup_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultSnapshotConfig(tempDir)
	config.MaxSnapshots = 3

	sm, err := NewSnapshotManager(config)
	if err != nil {
		t.Fatalf("Failed to create snapshot manager: %v", err)
	}

	ctx := context.Background()

	if err := sm.Start(ctx); err != nil {
		t.Fatalf("Failed to start snapshot manager: %v", err)
	}
	defer sm.Stop(ctx)

	// Create multiple snapshots
	snapshotData := &CollectionSnapshot{
		Vectors: []*api.Vector{
			{ID: "vec-1", Data: []float32{1.0, 2.0, 3.0}},
		},
	}

	for i := 0; i < 5; i++ {
		_, err := sm.CreateSnapshot(ctx, uint64(i+1), snapshotData)
		if err != nil {
			t.Errorf("Failed to create snapshot %d: %v", i, err)
		}
		time.Sleep(10 * time.Millisecond) // Ensure different timestamps
	}

	// Clean up old snapshots, keeping only 2
	err = sm.CleanupOldSnapshots(ctx, 2)
	if err != nil {
		t.Errorf("Failed to cleanup old snapshots: %v", err)
	}

	// Verify only 2 snapshots remain
	remainingSnapshots, err := sm.ListSnapshots(ctx)
	if err != nil {
		t.Errorf("Failed to list snapshots after cleanup: %v", err)
	}

	if len(remainingSnapshots) != 2 {
		t.Errorf("Expected 2 snapshots after cleanup, got %d", len(remainingSnapshots))
	}
}

// BenchmarkWALWrite benchmarks WAL write performance
func BenchmarkWALWrite(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "wal_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultWALConfig(tempDir)
	config.SyncWrites = false       // Disable sync for benchmark
	config.BufferSize = 1024 * 1024 // 1MB buffer

	wal, err := NewWAL(config)
	if err != nil {
		b.Fatalf("Failed to create WAL: %v", err)
	}

	ctx := context.Background()
	if err := wal.Start(ctx); err != nil {
		b.Fatalf("Failed to start WAL: %v", err)
	}
	defer wal.Stop(ctx)

	record := &WALRecord{
		Type: RecordTypeInsert,
		Vector: &api.Vector{
			ID:   "benchmark-vector",
			Data: make([]float32, 384), // Common embedding dimension
		},
		TxnID: "benchmark-txn",
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if err := wal.WriteRecord(ctx, record); err != nil {
				b.Errorf("Failed to write record: %v", err)
			}
		}
	})
}

// BenchmarkSnapshotCreate benchmarks snapshot creation performance
func BenchmarkSnapshotCreate(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "snapshot_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultSnapshotConfig(tempDir)
	sm, err := NewSnapshotManager(config)
	if err != nil {
		b.Fatalf("Failed to create snapshot manager: %v", err)
	}

	ctx := context.Background()
	if err := sm.Start(ctx); err != nil {
		b.Fatalf("Failed to start snapshot manager: %v", err)
	}
	defer sm.Stop(ctx)

	// Create test data
	vectors := make([]*api.Vector, 1000)
	for i := 0; i < 1000; i++ {
		vectors[i] = &api.Vector{
			ID:   fmt.Sprintf("vec-%d", i),
			Data: make([]float32, 384),
		}
	}

	snapshotData := &CollectionSnapshot{
		Vectors: vectors,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := sm.CreateSnapshot(ctx, uint64(i), snapshotData)
		if err != nil {
			b.Errorf("Failed to create snapshot: %v", err)
		}
	}
}
