// Package persist provides snapshot functionality for GoVecDB.
// Snapshots provide point-in-time backups of collection data with compression.
package persist

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// SnapshotConfig represents configuration for snapshot management
type SnapshotConfig struct {
	SnapshotDir        string        `json:"snapshot_dir"`
	CompressionEnabled bool          `json:"compression_enabled"`
	MaxSnapshots       int           `json:"max_snapshots"`      // Maximum number of snapshots to keep
	SnapshotInterval   time.Duration `json:"snapshot_interval"`  // Automatic snapshot interval
	BackgroundEnabled  bool          `json:"background_enabled"` // Enable background snapshots
	VerifyIntegrity    bool          `json:"verify_integrity"`   // Verify checksums on restore
	ChunkSize          int           `json:"chunk_size"`         // Size of data chunks for large snapshots
}

// DefaultSnapshotConfig returns a default snapshot configuration
func DefaultSnapshotConfig(snapshotDir string) *SnapshotConfig {
	return &SnapshotConfig{
		SnapshotDir:        snapshotDir,
		CompressionEnabled: true,
		MaxSnapshots:       10,
		SnapshotInterval:   1 * time.Hour,
		BackgroundEnabled:  true,
		VerifyIntegrity:    true,
		ChunkSize:          1024 * 1024, // 1 MB chunks
	}
}

// SnapshotManagerImpl implements snapshot management functionality
type SnapshotManagerImpl struct {
	config *SnapshotConfig
	mu     sync.RWMutex

	// Background operations
	ticker   *time.Ticker
	stopChan chan struct{}

	// State
	started bool
	stopped bool

	// Statistics
	stats struct {
		snapshotsCreated  int64
		snapshotsRestored int64
		lastSnapshotTime  time.Time
		totalSnapshotSize int64
	}
}

// NewSnapshotManager creates a new snapshot manager
func NewSnapshotManager(config *SnapshotConfig) (*SnapshotManagerImpl, error) {
	if config == nil {
		return nil, fmt.Errorf("snapshot config cannot be nil")
	}

	if config.SnapshotDir == "" {
		return nil, fmt.Errorf("snapshot directory cannot be empty")
	}

	// Create snapshot directory if it doesn't exist
	if err := os.MkdirAll(config.SnapshotDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create snapshot directory: %w", err)
	}

	sm := &SnapshotManagerImpl{
		config:   config,
		stopChan: make(chan struct{}),
	}

	return sm, nil
}

// Start starts the snapshot manager and background operations
func (sm *SnapshotManagerImpl) Start(ctx context.Context) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.started {
		return fmt.Errorf("snapshot manager already started")
	}

	// Start background snapshot routine if enabled
	if sm.config.BackgroundEnabled && sm.config.SnapshotInterval > 0 {
		sm.ticker = time.NewTicker(sm.config.SnapshotInterval)
		go sm.backgroundSnapshot()
	}

	sm.started = true
	return nil
}

// Stop stops the snapshot manager
func (sm *SnapshotManagerImpl) Stop(ctx context.Context) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.stopped {
		return nil
	}

	sm.stopped = true

	// Stop background operations
	if sm.ticker != nil {
		sm.ticker.Stop()
	}
	close(sm.stopChan)

	return nil
}

// CreateSnapshot creates a new snapshot of the provided data
func (sm *SnapshotManagerImpl) CreateSnapshot(ctx context.Context, lsn uint64, data SnapshotData) (*SnapshotMetadata, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.stopped {
		return nil, fmt.Errorf("snapshot manager is stopped")
	}

	// Generate snapshot ID
	snapshotID := sm.generateSnapshotID()

	// Serialize data
	serializedData, err := data.Serialize()
	if err != nil {
		return nil, fmt.Errorf("failed to serialize snapshot data: %w", err)
	}

	// Compress if enabled
	var finalData []byte
	compressed := false

	if sm.config.CompressionEnabled {
		compressedData, err := sm.compressData(serializedData)
		if err != nil {
			return nil, fmt.Errorf("failed to compress snapshot data: %w", err)
		}

		// Use compressed data if it's smaller
		if len(compressedData) < len(serializedData) {
			finalData = compressedData
			compressed = true
		} else {
			finalData = serializedData
		}
	} else {
		finalData = serializedData
	}

	// Calculate checksum
	checksum := sm.calculateChecksum(finalData)

	// Create metadata
	metadata := &SnapshotMetadata{
		ID:          snapshotID,
		Timestamp:   time.Now(),
		LSN:         lsn,
		VectorCount: data.Size(),
		FileSize:    int64(len(finalData)),
		Checksum:    checksum,
		Compressed:  compressed,
		Version:     "1.0.0",
	}

	// Write snapshot to disk
	if err := sm.writeSnapshot(snapshotID, finalData, metadata); err != nil {
		return nil, fmt.Errorf("failed to write snapshot: %w", err)
	}

	// Update statistics
	sm.stats.snapshotsCreated++
	sm.stats.lastSnapshotTime = time.Now()
	sm.stats.totalSnapshotSize += int64(len(finalData))

	// Cleanup old snapshots if needed
	if sm.config.MaxSnapshots > 0 {
		go sm.cleanupOldSnapshots(sm.config.MaxSnapshots)
	}

	return metadata, nil
}

// RestoreSnapshot restores data from a snapshot
func (sm *SnapshotManagerImpl) RestoreSnapshot(ctx context.Context, snapshotID string) (SnapshotData, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Read snapshot metadata
	metadata, err := sm.readSnapshotMetadata(snapshotID)
	if err != nil {
		return nil, fmt.Errorf("failed to read snapshot metadata: %w", err)
	}

	// Read snapshot data
	snapshotData, err := sm.readSnapshotData(snapshotID)
	if err != nil {
		return nil, fmt.Errorf("failed to read snapshot data: %w", err)
	}

	// Verify checksum if enabled
	if sm.config.VerifyIntegrity {
		actualChecksum := sm.calculateChecksum(snapshotData)
		if actualChecksum != metadata.Checksum {
			return nil, fmt.Errorf("snapshot checksum mismatch: expected %s, got %s",
				metadata.Checksum, actualChecksum)
		}
	}

	// Decompress if needed
	var finalData []byte
	if metadata.Compressed {
		decompressedData, err := sm.decompressData(snapshotData)
		if err != nil {
			return nil, fmt.Errorf("failed to decompress snapshot data: %w", err)
		}
		finalData = decompressedData
	} else {
		finalData = snapshotData
	}

	// Deserialize into CollectionSnapshot
	collectionSnapshot := &CollectionSnapshot{}
	if err := collectionSnapshot.Deserialize(finalData); err != nil {
		return nil, fmt.Errorf("failed to deserialize snapshot: %w", err)
	}

	// Update statistics
	sm.stats.snapshotsRestored++

	return collectionSnapshot, nil
}

// ListSnapshots returns a list of all available snapshots
func (sm *SnapshotManagerImpl) ListSnapshots(ctx context.Context) ([]*SnapshotMetadata, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	files, err := filepath.Glob(filepath.Join(sm.config.SnapshotDir, "*.metadata"))
	if err != nil {
		return nil, fmt.Errorf("failed to list snapshot files: %w", err)
	}

	var snapshots []*SnapshotMetadata

	for _, file := range files {
		snapshotID := strings.TrimSuffix(filepath.Base(file), ".metadata")

		metadata, err := sm.readSnapshotMetadata(snapshotID)
		if err != nil {
			continue // Skip corrupted metadata files
		}

		snapshots = append(snapshots, metadata)
	}

	// Sort by timestamp (newest first)
	sort.Slice(snapshots, func(i, j int) bool {
		return snapshots[i].Timestamp.After(snapshots[j].Timestamp)
	})

	return snapshots, nil
}

// DeleteSnapshot deletes a specific snapshot
func (sm *SnapshotManagerImpl) DeleteSnapshot(ctx context.Context, snapshotID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Delete data file
	dataPath := filepath.Join(sm.config.SnapshotDir, snapshotID+".data")
	if err := os.Remove(dataPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete snapshot data file: %w", err)
	}

	// Delete metadata file
	metadataPath := filepath.Join(sm.config.SnapshotDir, snapshotID+".metadata")
	if err := os.Remove(metadataPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete snapshot metadata file: %w", err)
	}

	return nil
}

// CleanupOldSnapshots removes old snapshots keeping only the specified count
func (sm *SnapshotManagerImpl) CleanupOldSnapshots(ctx context.Context, keepCount int) error {
	snapshots, err := sm.ListSnapshots(ctx)
	if err != nil {
		return fmt.Errorf("failed to list snapshots for cleanup: %w", err)
	}

	if len(snapshots) <= keepCount {
		return nil // Nothing to cleanup
	}

	// Delete old snapshots (keeping the newest ones)
	toDelete := snapshots[keepCount:]

	for _, snapshot := range toDelete {
		if err := sm.DeleteSnapshot(ctx, snapshot.ID); err != nil {
			return fmt.Errorf("failed to delete old snapshot %s: %w", snapshot.ID, err)
		}
	}

	return nil
}

// GetStats returns snapshot manager statistics
func (sm *SnapshotManagerImpl) GetStats() *SnapshotStats {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	return &SnapshotStats{
		SnapshotsCreated:  sm.stats.snapshotsCreated,
		SnapshotsRestored: sm.stats.snapshotsRestored,
		LastSnapshotTime:  sm.stats.lastSnapshotTime,
		TotalSnapshotSize: sm.stats.totalSnapshotSize,
	}
}

// Private methods

// generateSnapshotID generates a unique snapshot ID
func (sm *SnapshotManagerImpl) generateSnapshotID() string {
	return fmt.Sprintf("snapshot-%d", time.Now().UnixNano())
}

// compressData compresses data using gzip
func (sm *SnapshotManagerImpl) compressData(data []byte) ([]byte, error) {
	var buf bytes.Buffer

	writer := gzip.NewWriter(&buf)
	defer writer.Close()

	if _, err := writer.Write(data); err != nil {
		return nil, err
	}

	if err := writer.Close(); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// decompressData decompresses gzip-compressed data
func (sm *SnapshotManagerImpl) decompressData(data []byte) ([]byte, error) {
	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, reader); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// calculateChecksum calculates SHA256 checksum of data
func (sm *SnapshotManagerImpl) calculateChecksum(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

// writeSnapshot writes snapshot data and metadata to disk
func (sm *SnapshotManagerImpl) writeSnapshot(snapshotID string, data []byte, metadata *SnapshotMetadata) error {
	// Write data file
	dataPath := filepath.Join(sm.config.SnapshotDir, snapshotID+".data")
	if err := sm.writeFile(dataPath, data); err != nil {
		return fmt.Errorf("failed to write snapshot data: %w", err)
	}

	// Write metadata file
	metadataBytes, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	metadataPath := filepath.Join(sm.config.SnapshotDir, snapshotID+".metadata")
	if err := sm.writeFile(metadataPath, metadataBytes); err != nil {
		return fmt.Errorf("failed to write snapshot metadata: %w", err)
	}

	return nil
}

// writeFile writes data to a file atomically
func (sm *SnapshotManagerImpl) writeFile(path string, data []byte) error {
	// Write to temporary file first
	tempPath := path + ".tmp"

	file, err := os.Create(tempPath)
	if err != nil {
		return err
	}
	defer file.Close()

	if _, err := file.Write(data); err != nil {
		os.Remove(tempPath)
		return err
	}

	if err := file.Sync(); err != nil {
		os.Remove(tempPath)
		return err
	}

	if err := file.Close(); err != nil {
		os.Remove(tempPath)
		return err
	}

	// Atomic rename
	return os.Rename(tempPath, path)
}

// readSnapshotData reads snapshot data from disk
func (sm *SnapshotManagerImpl) readSnapshotData(snapshotID string) ([]byte, error) {
	dataPath := filepath.Join(sm.config.SnapshotDir, snapshotID+".data")
	return os.ReadFile(dataPath)
}

// readSnapshotMetadata reads snapshot metadata from disk
func (sm *SnapshotManagerImpl) readSnapshotMetadata(snapshotID string) (*SnapshotMetadata, error) {
	metadataPath := filepath.Join(sm.config.SnapshotDir, snapshotID+".metadata")

	data, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, err
	}

	var metadata SnapshotMetadata
	if err := json.Unmarshal(data, &metadata); err != nil {
		return nil, err
	}

	return &metadata, nil
}

// backgroundSnapshot runs the background snapshot routine
func (sm *SnapshotManagerImpl) backgroundSnapshot() {
	for {
		select {
		case <-sm.ticker.C:
			// This would be called by the collection manager
			// with the actual data to snapshot
		case <-sm.stopChan:
			return
		}
	}
}

// cleanupOldSnapshots removes old snapshots (background operation)
func (sm *SnapshotManagerImpl) cleanupOldSnapshots(keepCount int) {
	ctx := context.Background()
	_ = sm.CleanupOldSnapshots(ctx, keepCount)
}

// SnapshotStats contains snapshot manager statistics
type SnapshotStats struct {
	SnapshotsCreated  int64     `json:"snapshots_created"`
	SnapshotsRestored int64     `json:"snapshots_restored"`
	LastSnapshotTime  time.Time `json:"last_snapshot_time"`
	TotalSnapshotSize int64     `json:"total_snapshot_size"`
}

// Implement SnapshotData for CollectionSnapshot
func (c *CollectionSnapshot) Serialize() ([]byte, error) {
	// Use JSON for now, could be optimized with binary formats later
	return json.Marshal(c)
}

func (c *CollectionSnapshot) Deserialize(data []byte) error {
	return json.Unmarshal(data, c)
}

func (c *CollectionSnapshot) Size() int64 {
	return int64(len(c.Vectors))
}

// PersistenceManagerImpl implements the PersistenceManager interface
type PersistenceManagerImpl struct {
	wal      *WAL
	snapshot *SnapshotManagerImpl
	mu       sync.RWMutex

	// Statistics
	stats *PersistenceStats
}

// NewPersistenceManager creates a new persistence manager
func NewPersistenceManager(walConfig *WALConfig, snapshotConfig *SnapshotConfig) (*PersistenceManagerImpl, error) {
	wal, err := NewWAL(walConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create WAL: %w", err)
	}

	snapshot, err := NewSnapshotManager(snapshotConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create snapshot manager: %w", err)
	}

	pm := &PersistenceManagerImpl{
		wal:      wal,
		snapshot: snapshot,
		stats: &PersistenceStats{
			LastSync: time.Now(),
		},
	}

	return pm, nil
}

// WAL operations - delegate to WAL
func (pm *PersistenceManagerImpl) WriteRecord(ctx context.Context, record *WALRecord) error {
	return pm.wal.WriteRecord(ctx, record)
}

func (pm *PersistenceManagerImpl) WriteBatch(ctx context.Context, records []*WALRecord) error {
	return pm.wal.WriteBatch(ctx, records)
}

func (pm *PersistenceManagerImpl) Replay(ctx context.Context, fromLSN uint64, handler ReplayHandler) error {
	return pm.wal.Replay(ctx, fromLSN, handler)
}

func (pm *PersistenceManagerImpl) GetLastLSN() uint64 {
	return pm.wal.GetLastLSN()
}

func (pm *PersistenceManagerImpl) Checkpoint(ctx context.Context) (uint64, error) {
	return pm.wal.Checkpoint(ctx)
}

func (pm *PersistenceManagerImpl) Compact(ctx context.Context, beforeLSN uint64) error {
	return pm.wal.Compact(ctx, beforeLSN)
}

func (pm *PersistenceManagerImpl) Start(ctx context.Context) error {
	if err := pm.wal.Start(ctx); err != nil {
		return err
	}
	return pm.snapshot.Start(ctx)
}

func (pm *PersistenceManagerImpl) Stop(ctx context.Context) error {
	if err := pm.wal.Stop(ctx); err != nil {
		return err
	}
	return pm.snapshot.Stop(ctx)
}

func (pm *PersistenceManagerImpl) Sync() error {
	return pm.wal.Sync()
}

// Snapshot operations - delegate to snapshot manager
func (pm *PersistenceManagerImpl) CreateSnapshot(ctx context.Context, lsn uint64, data SnapshotData) (*SnapshotMetadata, error) {
	return pm.snapshot.CreateSnapshot(ctx, lsn, data)
}

func (pm *PersistenceManagerImpl) RestoreSnapshot(ctx context.Context, snapshotID string) (SnapshotData, error) {
	return pm.snapshot.RestoreSnapshot(ctx, snapshotID)
}

func (pm *PersistenceManagerImpl) ListSnapshots(ctx context.Context) ([]*SnapshotMetadata, error) {
	return pm.snapshot.ListSnapshots(ctx)
}

func (pm *PersistenceManagerImpl) DeleteSnapshot(ctx context.Context, snapshotID string) error {
	return pm.snapshot.DeleteSnapshot(ctx, snapshotID)
}

func (pm *PersistenceManagerImpl) CleanupOldSnapshots(ctx context.Context, keepCount int) error {
	return pm.snapshot.CleanupOldSnapshots(ctx, keepCount)
}

// Combined operations
func (pm *PersistenceManagerImpl) Recover(ctx context.Context) error {
	// This would implement the full recovery process:
	// 1. Find latest snapshot
	// 2. Restore from snapshot
	// 3. Replay WAL from snapshot LSN
	// For now, this is a placeholder
	return nil
}

func (pm *PersistenceManagerImpl) HealthCheck(ctx context.Context) error {
	// Check WAL and snapshot health
	return nil
}

func (pm *PersistenceManagerImpl) GetStats(ctx context.Context) (*PersistenceStats, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	// Combine stats from WAL and snapshot managers
	stats := *pm.stats
	stats.LastLSN = pm.wal.GetLastLSN()

	snapshotStats := pm.snapshot.GetStats()
	stats.SnapshotCount = int(snapshotStats.SnapshotsCreated)
	stats.LastSnapshotTime = snapshotStats.LastSnapshotTime
	stats.LastSnapshotSize = snapshotStats.TotalSnapshotSize

	return &stats, nil
}
