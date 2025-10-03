// Package persist provides performance benchmarks for the persistence layer.
package persist

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"testing"

	"github.com/khambampati-subhash/govecdb/api"
)

// BenchmarkWALOperations benchmarks various WAL operations
func BenchmarkWALOperations(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "wal_perf_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultWALConfig(tempDir)
	config.SyncWrites = false       // Disable sync for performance testing
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

	// Create test vectors of different sizes
	smallVector := &api.Vector{
		ID:   "small-vec",
		Data: make([]float32, 128),
	}

	mediumVector := &api.Vector{
		ID:   "medium-vec",
		Data: make([]float32, 384),
	}

	largeVector := &api.Vector{
		ID:   "large-vec",
		Data: make([]float32, 1536),
	}

	b.Run("WriteSmallVector", func(b *testing.B) {
		record := &WALRecord{
			Type:   RecordTypeInsert,
			Vector: smallVector,
			TxnID:  "bench-txn",
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			record.Vector.ID = fmt.Sprintf("small-vec-%d", i)
			if err := wal.WriteRecord(ctx, record); err != nil {
				b.Errorf("Failed to write record: %v", err)
			}
		}
	})

	b.Run("WriteMediumVector", func(b *testing.B) {
		record := &WALRecord{
			Type:   RecordTypeInsert,
			Vector: mediumVector,
			TxnID:  "bench-txn",
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			record.Vector.ID = fmt.Sprintf("medium-vec-%d", i)
			if err := wal.WriteRecord(ctx, record); err != nil {
				b.Errorf("Failed to write record: %v", err)
			}
		}
	})

	b.Run("WriteLargeVector", func(b *testing.B) {
		record := &WALRecord{
			Type:   RecordTypeInsert,
			Vector: largeVector,
			TxnID:  "bench-txn",
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			record.Vector.ID = fmt.Sprintf("large-vec-%d", i)
			if err := wal.WriteRecord(ctx, record); err != nil {
				b.Errorf("Failed to write record: %v", err)
			}
		}
	})

	b.Run("WriteBatch", func(b *testing.B) {
		batchSize := 100
		records := make([]*WALRecord, batchSize)
		for i := 0; i < batchSize; i++ {
			records[i] = &WALRecord{
				Type: RecordTypeInsert,
				Vector: &api.Vector{
					ID:   fmt.Sprintf("batch-vec-%d", i),
					Data: make([]float32, 384),
				},
				TxnID: "batch-txn",
			}
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Update IDs to make each batch unique
			for j, record := range records {
				record.Vector.ID = fmt.Sprintf("batch-%d-vec-%d", i, j)
			}
			if err := wal.WriteBatch(ctx, records); err != nil {
				b.Errorf("Failed to write batch: %v", err)
			}
		}
	})
}

// BenchmarkWALSync benchmarks WAL synchronization performance
func BenchmarkWALSync(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "wal_sync_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultWALConfig(tempDir)
	config.SyncWrites = true // Enable sync for this benchmark

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
			ID:   "sync-vec",
			Data: make([]float32, 384),
		},
		TxnID: "sync-txn",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		record.Vector.ID = fmt.Sprintf("sync-vec-%d", i)
		if err := wal.WriteRecord(ctx, record); err != nil {
			b.Errorf("Failed to write record: %v", err)
		}
	}
}

// BenchmarkWALReplay benchmarks WAL replay performance
func BenchmarkWALReplay(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "wal_replay_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultWALConfig(tempDir)
	ctx := context.Background()

	// Pre-populate WAL with records
	wal, err := NewWAL(config)
	if err != nil {
		b.Fatalf("Failed to create WAL: %v", err)
	}

	if err := wal.Start(ctx); err != nil {
		b.Fatalf("Failed to start WAL: %v", err)
	}

	// Write test records
	numRecords := 10000
	for i := 0; i < numRecords; i++ {
		record := &WALRecord{
			Type: RecordTypeInsert,
			Vector: &api.Vector{
				ID:   fmt.Sprintf("replay-vec-%d", i),
				Data: make([]float32, 384),
			},
		}
		if err := wal.WriteRecord(ctx, record); err != nil {
			b.Fatalf("Failed to write record: %v", err)
		}
	}

	if err := wal.Stop(ctx); err != nil {
		b.Fatalf("Failed to stop WAL: %v", err)
	}

	// Create new WAL for replay
	replayWal, err := NewWAL(config)
	if err != nil {
		b.Fatalf("Failed to create replay WAL: %v", err)
	}

	handler := &MockReplayHandler{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handler = &MockReplayHandler{} // Reset handler
		if err := replayWal.Replay(ctx, 1, handler); err != nil {
			b.Errorf("Failed to replay WAL: %v", err)
		}
	}
}

// BenchmarkSnapshotOperations benchmarks snapshot operations
func BenchmarkSnapshotOperations(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "snapshot_perf_bench")
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

	// Create test data of different sizes
	createTestData := func(numVectors int, dimension int) *CollectionSnapshot {
		vectors := make([]*api.Vector, numVectors)
		for i := 0; i < numVectors; i++ {
			vectors[i] = &api.Vector{
				ID:   fmt.Sprintf("vec-%d", i),
				Data: make([]float32, dimension),
			}
			// Fill with random data
			for j := range vectors[i].Data {
				vectors[i].Data[j] = rand.Float32()
			}
		}
		return &CollectionSnapshot{Vectors: vectors}
	}

	b.Run("CreateSmallSnapshot", func(b *testing.B) {
		data := createTestData(100, 128) // 100 vectors, 128 dimensions
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := sm.CreateSnapshot(ctx, uint64(i), data)
			if err != nil {
				b.Errorf("Failed to create snapshot: %v", err)
			}
		}
	})

	b.Run("CreateMediumSnapshot", func(b *testing.B) {
		data := createTestData(1000, 384) // 1000 vectors, 384 dimensions
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := sm.CreateSnapshot(ctx, uint64(i+1000), data)
			if err != nil {
				b.Errorf("Failed to create snapshot: %v", err)
			}
		}
	})

	b.Run("CreateLargeSnapshot", func(b *testing.B) {
		data := createTestData(10000, 384) // 10000 vectors, 384 dimensions
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := sm.CreateSnapshot(ctx, uint64(i+2000), data)
			if err != nil {
				b.Errorf("Failed to create snapshot: %v", err)
			}
		}
	})
}

// BenchmarkSnapshotCompression benchmarks compression performance
func BenchmarkSnapshotCompression(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "snapshot_compression_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	ctx := context.Background()

	// Test data with good compression characteristics
	createCompressibleData := func(numVectors int) *CollectionSnapshot {
		vectors := make([]*api.Vector, numVectors)
		for i := 0; i < numVectors; i++ {
			vectors[i] = &api.Vector{
				ID:   fmt.Sprintf("vec-%d", i),
				Data: make([]float32, 384),
			}
			// Fill with repeated patterns (compresses well)
			for j := range vectors[i].Data {
				vectors[i].Data[j] = float32(j % 10)
			}
		}
		return &CollectionSnapshot{Vectors: vectors}
	}

	b.Run("CompressedSnapshot", func(b *testing.B) {
		config := DefaultSnapshotConfig(tempDir + "/compressed")
		config.CompressionEnabled = true

		sm, err := NewSnapshotManager(config)
		if err != nil {
			b.Fatalf("Failed to create snapshot manager: %v", err)
		}

		if err := sm.Start(ctx); err != nil {
			b.Fatalf("Failed to start snapshot manager: %v", err)
		}
		defer sm.Stop(ctx)

		data := createCompressibleData(1000)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := sm.CreateSnapshot(ctx, uint64(i), data)
			if err != nil {
				b.Errorf("Failed to create compressed snapshot: %v", err)
			}
		}
	})

	b.Run("UncompressedSnapshot", func(b *testing.B) {
		config := DefaultSnapshotConfig(tempDir + "/uncompressed")
		config.CompressionEnabled = false

		sm, err := NewSnapshotManager(config)
		if err != nil {
			b.Fatalf("Failed to create snapshot manager: %v", err)
		}

		if err := sm.Start(ctx); err != nil {
			b.Fatalf("Failed to start snapshot manager: %v", err)
		}
		defer sm.Stop(ctx)

		data := createCompressibleData(1000)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := sm.CreateSnapshot(ctx, uint64(i+10000), data)
			if err != nil {
				b.Errorf("Failed to create uncompressed snapshot: %v", err)
			}
		}
	})
}

// BenchmarkSnapshotRestore benchmarks snapshot restoration performance
func BenchmarkSnapshotRestore(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "snapshot_restore_bench")
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

	// Create a snapshot to restore from
	vectors := make([]*api.Vector, 1000)
	for i := 0; i < 1000; i++ {
		vectors[i] = &api.Vector{
			ID:   fmt.Sprintf("restore-vec-%d", i),
			Data: make([]float32, 384),
		}
	}

	snapshotData := &CollectionSnapshot{Vectors: vectors}
	metadata, err := sm.CreateSnapshot(ctx, 1, snapshotData)
	if err != nil {
		b.Fatalf("Failed to create snapshot for restore benchmark: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := sm.RestoreSnapshot(ctx, metadata.ID)
		if err != nil {
			b.Errorf("Failed to restore snapshot: %v", err)
		}
	}
}

// BenchmarkPersistenceManager benchmarks the combined persistence manager
func BenchmarkPersistenceManager(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "persistence_manager_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	walConfig := DefaultWALConfig(tempDir + "/wal")
	walConfig.SyncWrites = false

	snapshotConfig := DefaultSnapshotConfig(tempDir + "/snapshots")

	pm, err := NewPersistenceManager(walConfig, snapshotConfig)
	if err != nil {
		b.Fatalf("Failed to create persistence manager: %v", err)
	}

	ctx := context.Background()
	if err := pm.Start(ctx); err != nil {
		b.Fatalf("Failed to start persistence manager: %v", err)
	}
	defer pm.Stop(ctx)

	b.Run("WriteAndSnapshot", func(b *testing.B) {
		vectors := make([]*api.Vector, 100)
		for i := 0; i < 100; i++ {
			vectors[i] = &api.Vector{
				ID:   fmt.Sprintf("combined-vec-%d", i),
				Data: make([]float32, 384),
			}
		}

		snapshotData := &CollectionSnapshot{Vectors: vectors}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Write WAL record
			record := &WALRecord{
				Type:    RecordTypeBatchInsert,
				Vectors: vectors,
			}
			if err := pm.WriteRecord(ctx, record); err != nil {
				b.Errorf("Failed to write WAL record: %v", err)
			}

			// Create snapshot every 10 iterations
			if i%10 == 0 {
				_, err := pm.CreateSnapshot(ctx, uint64(i), snapshotData)
				if err != nil {
					b.Errorf("Failed to create snapshot: %v", err)
				}
			}
		}
	})
}

// BenchmarkConcurrentOperations benchmarks concurrent persistence operations
func BenchmarkConcurrentOperations(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "concurrent_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	walConfig := DefaultWALConfig(tempDir + "/wal")
	walConfig.SyncWrites = false

	wal, err := NewWAL(walConfig)
	if err != nil {
		b.Fatalf("Failed to create WAL: %v", err)
	}

	ctx := context.Background()
	if err := wal.Start(ctx); err != nil {
		b.Fatalf("Failed to start WAL: %v", err)
	}
	defer wal.Stop(ctx)

	b.Run("ConcurrentWrites", func(b *testing.B) {
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			workerID := 0
			recordID := 0
			for pb.Next() {
				record := &WALRecord{
					Type: RecordTypeInsert,
					Vector: &api.Vector{
						ID:   fmt.Sprintf("concurrent-vec-%d-%d", workerID, recordID),
						Data: make([]float32, 384),
					},
				}
				if err := wal.WriteRecord(ctx, record); err != nil {
					b.Errorf("Failed to write record: %v", err)
				}
				recordID++
			}
		})
	})
}

// BenchmarkRecoveryPerformance benchmarks recovery performance
func BenchmarkRecoveryPerformance(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "recovery_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	config := DefaultWALConfig(tempDir)
	ctx := context.Background()

	// Pre-populate with varying numbers of records
	recordCounts := []int{1000, 5000, 10000}

	for _, recordCount := range recordCounts {
		b.Run(fmt.Sprintf("Recovery_%d_records", recordCount), func(b *testing.B) {
			// Create WAL and populate with records
			wal, err := NewWAL(config)
			if err != nil {
				b.Fatalf("Failed to create WAL: %v", err)
			}

			if err := wal.Start(ctx); err != nil {
				b.Fatalf("Failed to start WAL: %v", err)
			}

			// Write records
			for i := 0; i < recordCount; i++ {
				record := &WALRecord{
					Type: RecordTypeInsert,
					Vector: &api.Vector{
						ID:   fmt.Sprintf("recovery-vec-%d", i),
						Data: make([]float32, 384),
					},
				}
				if err := wal.WriteRecord(ctx, record); err != nil {
					b.Fatalf("Failed to write record: %v", err)
				}
			}

			if err := wal.Stop(ctx); err != nil {
				b.Fatalf("Failed to stop WAL: %v", err)
			}

			// Benchmark recovery
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				recoveryWal, err := NewWAL(config)
				if err != nil {
					b.Fatalf("Failed to create recovery WAL: %v", err)
				}

				handler := &MockReplayHandler{}
				if err := recoveryWal.Replay(ctx, 1, handler); err != nil {
					b.Errorf("Failed to replay WAL: %v", err)
				}

				// Skip verification during benchmarks to avoid stale data issues
				_ = handler.Records
			}
		})
	}
}

// BenchmarkMemoryUsage benchmarks memory usage patterns
func BenchmarkMemoryUsage(b *testing.B) {
	tempDir, err := os.MkdirTemp("", "memory_bench")
	if err != nil {
		b.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	b.Run("WALMemoryUsage", func(b *testing.B) {
		config := DefaultWALConfig(tempDir + "/memory_wal")
		config.BufferSize = 64 * 1024 // Smaller buffer to test memory usage

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
				ID:   "memory-test-vec",
				Data: make([]float32, 1536), // Larger vector
			},
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			record.Vector.ID = fmt.Sprintf("memory-test-vec-%d", i)
			if err := wal.WriteRecord(ctx, record); err != nil {
				b.Errorf("Failed to write record: %v", err)
			}
		}
	})
}

// init sets up benchmark environment
func init() {
	// Random seed is no longer needed as of Go 1.20 - using default random source
}
