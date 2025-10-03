package persist

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// OptimizedPersistence provides high-performance persistence operations
type OptimizedPersistence struct {
	// Configuration
	dataDir     string
	snapshotDir string
	walDir      string

	// Write-ahead log
	wal   *OptimizedWAL
	walMu sync.RWMutex

	// Snapshot management
	snapshotMu    sync.RWMutex
	lastSnapshot  time.Time
	snapshotCount int64

	// Background operations
	ctx          context.Context
	cancel       context.CancelFunc
	backgroundWG sync.WaitGroup

	// Memory management
	bufferPool   sync.Pool
	compressPool sync.Pool

	// Metrics
	bytesWritten int64
	bytesRead    int64
	snapshots    int64
	walEntries   int64
}

// OptimizedWAL represents a write-ahead log optimized for performance
type OptimizedWAL struct {
	mu      sync.RWMutex
	file    *os.File
	writer  *bufio.Writer
	entries int64
	size    int64
	maxSize int64

	// Buffer management
	writeBuffer []byte
	bufferMu    sync.Mutex
}

// WALEntry represents a write-ahead log entry
type WALEntry struct {
	Timestamp time.Time
	Operation string
	Key       string
	Data      []byte
	Checksum  uint32
}

// SnapshotMetadata contains metadata about a snapshot

// NewOptimizedPersistence creates a new optimized persistence manager
func NewOptimizedPersistence(dataDir string) (*OptimizedPersistence, error) {
	// Create directories
	dirs := []string{
		dataDir,
		filepath.Join(dataDir, "snapshots"),
		filepath.Join(dataDir, "wal"),
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	ctx, cancel := context.WithCancel(context.Background())

	op := &OptimizedPersistence{
		dataDir:     dataDir,
		snapshotDir: filepath.Join(dataDir, "snapshots"),
		walDir:      filepath.Join(dataDir, "wal"),
		ctx:         ctx,
		cancel:      cancel,
		bufferPool: sync.Pool{
			New: func() interface{} {
				return make([]byte, 64*1024) // 64KB buffers
			},
		},
		compressPool: sync.Pool{
			New: func() interface{} {
				return gzip.NewWriter(nil)
			},
		},
	}

	// Initialize WAL
	wal, err := op.initializeWAL()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize WAL: %w", err)
	}
	op.wal = wal

	// Start background operations
	op.startBackgroundTasks()

	return op, nil
}

// initializeWAL initializes the write-ahead log
func (op *OptimizedPersistence) initializeWAL() (*OptimizedWAL, error) {
	walPath := filepath.Join(op.walDir, "wal.log")

	file, err := os.OpenFile(walPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, err
	}

	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, err
	}

	wal := &OptimizedWAL{
		file:        file,
		writer:      bufio.NewWriterSize(file, 64*1024), // 64KB buffer
		size:        stat.Size(),
		maxSize:     100 * 1024 * 1024, // 100MB max WAL size
		writeBuffer: make([]byte, 0, 1024),
	}

	return wal, nil
}

// WriteWALEntry writes an entry to the write-ahead log
func (op *OptimizedPersistence) WriteWALEntry(operation, key string, data []byte) error {
	entry := WALEntry{
		Timestamp: time.Now(),
		Operation: operation,
		Key:       key,
		Data:      data,
	}

	// Calculate checksum
	entry.Checksum = op.calculateChecksum(entry)

	return op.wal.WriteEntry(entry)
}

// WriteEntry writes an entry to the WAL
func (wal *OptimizedWAL) WriteEntry(entry WALEntry) error {
	wal.bufferMu.Lock()
	defer wal.bufferMu.Unlock()

	// Serialize entry to buffer
	wal.writeBuffer = wal.writeBuffer[:0] // Reset buffer

	// Write timestamp
	ts := entry.Timestamp.UnixNano()
	wal.writeBuffer = binary.AppendVarint(wal.writeBuffer, ts)

	// Write operation
	wal.writeBuffer = binary.AppendUvarint(wal.writeBuffer, uint64(len(entry.Operation)))
	wal.writeBuffer = append(wal.writeBuffer, entry.Operation...)

	// Write key
	wal.writeBuffer = binary.AppendUvarint(wal.writeBuffer, uint64(len(entry.Key)))
	wal.writeBuffer = append(wal.writeBuffer, entry.Key...)

	// Write data
	wal.writeBuffer = binary.AppendUvarint(wal.writeBuffer, uint64(len(entry.Data)))
	wal.writeBuffer = append(wal.writeBuffer, entry.Data...)

	// Write checksum
	wal.writeBuffer = binary.BigEndian.AppendUint32(wal.writeBuffer, entry.Checksum)

	// Write to WAL file
	wal.mu.Lock()
	defer wal.mu.Unlock()

	n, err := wal.writer.Write(wal.writeBuffer)
	if err != nil {
		return err
	}

	atomic.AddInt64(&wal.entries, 1)
	atomic.AddInt64(&wal.size, int64(n))

	// Flush periodically
	if wal.entries%100 == 0 {
		if err := wal.writer.Flush(); err != nil {
			return err
		}
		if err := wal.file.Sync(); err != nil {
			return err
		}
	}

	return nil
}

// CreateSnapshot creates an optimized snapshot of the index
func (op *OptimizedPersistence) CreateSnapshot(ctx context.Context, index interface{}) error {
	op.snapshotMu.Lock()
	defer op.snapshotMu.Unlock()

	timestamp := time.Now()
	filename := fmt.Sprintf("snapshot_%d.gz", timestamp.Unix())
	filepath := filepath.Join(op.snapshotDir, filename)

	// Create snapshot file
	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create snapshot file: %w", err)
	}
	defer file.Close()

	// Use compression
	compressor := op.compressPool.Get().(*gzip.Writer)
	defer op.compressPool.Put(compressor)
	compressor.Reset(file)
	defer compressor.Close()

	// Write snapshot data (simplified - in production, serialize the actual index)
	buffer := op.bufferPool.Get().([]byte)
	defer op.bufferPool.Put(buffer)

	// Serialize index data (placeholder implementation)
	data := op.serializeIndex(index)

	// Write in chunks to manage memory
	const chunkSize = 64 * 1024
	for i := 0; i < len(data); i += chunkSize {
		end := i + chunkSize
		if end > len(data) {
			end = len(data)
		}

		if _, err := compressor.Write(data[i:end]); err != nil {
			return fmt.Errorf("failed to write snapshot chunk: %w", err)
		}

		// Check for cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}

	// Update metrics
	atomic.AddInt64(&op.snapshots, 1)
	atomic.AddInt64(&op.bytesWritten, int64(len(data)))
	op.lastSnapshot = timestamp

	// Clean up old snapshots
	go op.cleanupOldSnapshots()

	return nil
}

// LoadSnapshot loads a snapshot from disk
func (op *OptimizedPersistence) LoadSnapshot(ctx context.Context, filename string) ([]byte, error) {
	filepath := filepath.Join(op.snapshotDir, filename)

	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open snapshot file: %w", err)
	}
	defer file.Close()

	// Decompress
	reader, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer reader.Close()

	// Read data
	var result bytes.Buffer
	buffer := op.bufferPool.Get().([]byte)
	defer op.bufferPool.Put(buffer)

	for {
		n, err := reader.Read(buffer)
		if n > 0 {
			result.Write(buffer[:n])
			atomic.AddInt64(&op.bytesRead, int64(n))
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read snapshot data: %w", err)
		}

		// Check for cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}

	return result.Bytes(), nil
}

// startBackgroundTasks starts background maintenance tasks
func (op *OptimizedPersistence) startBackgroundTasks() {
	// WAL rotation task
	op.backgroundWG.Add(1)
	go op.walRotationTask()

	// Snapshot cleanup task
	op.backgroundWG.Add(1)
	go op.snapshotCleanupTask()
}

// walRotationTask handles WAL file rotation
func (op *OptimizedPersistence) walRotationTask() {
	defer op.backgroundWG.Done()

	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-op.ctx.Done():
			return
		case <-ticker.C:
			if atomic.LoadInt64(&op.wal.size) > op.wal.maxSize {
				op.rotateWAL()
			}
		}
	}
}

// snapshotCleanupTask handles old snapshot cleanup
func (op *OptimizedPersistence) snapshotCleanupTask() {
	defer op.backgroundWG.Done()

	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-op.ctx.Done():
			return
		case <-ticker.C:
			op.cleanupOldSnapshots()
		}
	}
}

// rotateWAL rotates the WAL file
func (op *OptimizedPersistence) rotateWAL() error {
	op.walMu.Lock()
	defer op.walMu.Unlock()

	// Close current WAL
	if err := op.wal.writer.Flush(); err != nil {
		return err
	}
	if err := op.wal.file.Sync(); err != nil {
		return err
	}
	if err := op.wal.file.Close(); err != nil {
		return err
	}

	// Rename current WAL
	oldPath := filepath.Join(op.walDir, "wal.log")
	newPath := filepath.Join(op.walDir, fmt.Sprintf("wal_%d.log", time.Now().Unix()))
	if err := os.Rename(oldPath, newPath); err != nil {
		return err
	}

	// Create new WAL
	wal, err := op.initializeWAL()
	if err != nil {
		return err
	}
	op.wal = wal

	return nil
}

// cleanupOldSnapshots removes old snapshot files
func (op *OptimizedPersistence) cleanupOldSnapshots() {
	const maxSnapshots = 10

	files, err := filepath.Glob(filepath.Join(op.snapshotDir, "snapshot_*.gz"))
	if err != nil {
		return
	}

	if len(files) <= maxSnapshots {
		return
	}

	// Sort by modification time and remove oldest
	// (simplified implementation)
	for i := 0; i < len(files)-maxSnapshots; i++ {
		os.Remove(files[i])
	}
}

// calculateChecksum calculates checksum for WAL entry
func (op *OptimizedPersistence) calculateChecksum(entry WALEntry) uint32 {
	// Simple checksum calculation (use CRC32 in production)
	hash := uint32(0)
	for _, b := range []byte(entry.Operation + entry.Key) {
		hash = hash*31 + uint32(b)
	}
	for _, b := range entry.Data {
		hash = hash*31 + uint32(b)
	}
	return hash
}

// serializeIndex serializes index data (placeholder)
func (op *OptimizedPersistence) serializeIndex(index interface{}) []byte {
	// This is a placeholder - in production, implement proper serialization
	return []byte("serialized index data")
}

// GetMetrics returns persistence metrics
func (op *OptimizedPersistence) GetMetrics() PersistenceMetrics {
	return PersistenceMetrics{
		WALEntries:   atomic.LoadInt64(&op.walEntries),
		WALSize:      atomic.LoadInt64(&op.wal.size),
		Snapshots:    atomic.LoadInt64(&op.snapshots),
		BytesWritten: atomic.LoadInt64(&op.bytesWritten),
		BytesRead:    atomic.LoadInt64(&op.bytesRead),
		LastSnapshot: op.lastSnapshot,
	}
}

// Close closes the persistence manager
func (op *OptimizedPersistence) Close() error {
	op.cancel()
	op.backgroundWG.Wait()

	if op.wal != nil {
		op.wal.writer.Flush()
		op.wal.file.Sync()
		op.wal.file.Close()
	}

	return nil
}

// PersistenceMetrics represents persistence metrics
type PersistenceMetrics struct {
	WALEntries   int64
	WALSize      int64
	Snapshots    int64
	BytesWritten int64
	BytesRead    int64
	LastSnapshot time.Time
}
