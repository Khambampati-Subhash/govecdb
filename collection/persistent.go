// Package collection provides persistent collection implementation for GoVecDB.
// This implementation adds durability through WAL and snapshots.
package collection

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
	"github.com/khambampati-subhash/govecdb/persist"
	"github.com/khambampati-subhash/govecdb/store"
)

// PersistentVectorCollection implements a persistent vector collection with WAL and snapshots
type PersistentVectorCollection struct {
	// Base collection
	*VectorCollection

	// Persistence layer
	persistence persist.PersistenceManager
	manifest    *ManifestManager

	// Configuration
	persistConfig *PersistentCollectionConfig

	// Recovery handler
	recoveryHandler *CollectionRecoveryHandler

	// State
	lastSnapshotLSN uint64
	mu              sync.RWMutex
}

// PersistentCollectionConfig represents configuration for persistent collections
type PersistentCollectionConfig struct {
	// Base collection config
	CollectionConfig *api.CollectionConfig

	// Storage paths
	DataDir      string `json:"data_dir"`
	WALDir       string `json:"wal_dir"`
	SnapshotDir  string `json:"snapshot_dir"`
	ManifestPath string `json:"manifest_path"`

	// Persistence settings
	WALConfig      *persist.WALConfig      `json:"wal_config"`
	SnapshotConfig *persist.SnapshotConfig `json:"snapshot_config"`

	// Recovery settings
	RecoveryTimeout  time.Duration `json:"recovery_timeout"`
	VerifyOnRecovery bool          `json:"verify_on_recovery"`

	// Snapshot settings
	AutoSnapshotEnabled   bool          `json:"auto_snapshot_enabled"`
	AutoSnapshotInterval  time.Duration `json:"auto_snapshot_interval"`
	AutoSnapshotThreshold int64         `json:"auto_snapshot_threshold"` // Operations before snapshot

	// Compaction settings
	AutoCompactionEnabled bool          `json:"auto_compaction_enabled"`
	CompactionInterval    time.Duration `json:"compaction_interval"`
}

// DefaultPersistentCollectionConfig returns a default persistent collection configuration
func DefaultPersistentCollectionConfig(name, dataDir string, dimension int) *PersistentCollectionConfig {
	return &PersistentCollectionConfig{
		CollectionConfig:      api.DefaultCollectionConfig(name, dimension),
		DataDir:               dataDir,
		WALDir:                fmt.Sprintf("%s/wal", dataDir),
		SnapshotDir:           fmt.Sprintf("%s/snapshots", dataDir),
		ManifestPath:          fmt.Sprintf("%s/manifest.json", dataDir),
		WALConfig:             persist.DefaultWALConfig(fmt.Sprintf("%s/wal", dataDir)),
		SnapshotConfig:        persist.DefaultSnapshotConfig(fmt.Sprintf("%s/snapshots", dataDir)),
		RecoveryTimeout:       30 * time.Minute,
		VerifyOnRecovery:      true,
		AutoSnapshotEnabled:   true,
		AutoSnapshotInterval:  1 * time.Hour,
		AutoSnapshotThreshold: 10000,
		AutoCompactionEnabled: true,
		CompactionInterval:    6 * time.Hour,
	}
}

// NewPersistentVectorCollection creates a new persistent vector collection
func NewPersistentVectorCollection(config *PersistentCollectionConfig) (*PersistentVectorCollection, error) {
	if config == nil {
		return nil, fmt.Errorf("persistent collection config cannot be nil")
	}

	if err := config.CollectionConfig.Validate(); err != nil {
		return nil, fmt.Errorf("invalid collection config: %w", err)
	}

	// Create persistence manager
	persistence, err := persist.NewPersistenceManager(config.WALConfig, config.SnapshotConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create persistence manager: %w", err)
	}

	// Create manifest manager
	manifestConfig := DefaultManifestConfig(config.ManifestPath)
	manifest, err := NewManifestManager(manifestConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create manifest manager: %w", err)
	}

	// Create base collection
	storeConfig := store.DefaultStoreConfig(config.CollectionConfig.Name)
	baseCollection, err := NewVectorCollection(config.CollectionConfig, storeConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create base collection: %w", err)
	}

	// Create persistent collection
	persistent := &PersistentVectorCollection{
		VectorCollection: baseCollection,
		persistence:      persistence,
		manifest:         manifest,
		persistConfig:    config,
	}

	// Create recovery handler
	persistent.recoveryHandler = NewCollectionRecoveryHandler(persistent)

	return persistent, nil
}

// Start starts the persistent collection and performs recovery if needed
func (pc *PersistentVectorCollection) Start(ctx context.Context) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	// Start persistence layer
	if err := pc.persistence.Start(ctx); err != nil {
		return fmt.Errorf("failed to start persistence layer: %w", err)
	}

	// Register collection in manifest
	if err := pc.manifest.CreateCollection(ctx, pc.persistConfig.CollectionConfig); err != nil {
		if err != api.ErrCollectionExists {
			return fmt.Errorf("failed to register collection in manifest: %w", err)
		}
	}

	// Set collection state to active
	if err := pc.manifest.SetCollectionState(ctx, pc.persistConfig.CollectionConfig.Name, CollectionStateActive); err != nil {
		return fmt.Errorf("failed to set collection state: %w", err)
	}

	// Perform recovery if needed
	if err := pc.recoverIfNeeded(ctx); err != nil {
		return fmt.Errorf("recovery failed: %w", err)
	}

	return nil
}

// Add adds a vector with WAL logging
func (pc *PersistentVectorCollection) Add(ctx context.Context, vector *api.Vector) error {
	if vector == nil {
		return api.ErrEmptyVector
	}

	if err := vector.Validate(); err != nil {
		return err
	}

	pc.mu.Lock()
	defer pc.mu.Unlock()

	// Create WAL record
	walRecord := &persist.WALRecord{
		Type:   persist.RecordTypeInsert,
		Vector: vector,
		TxnID:  pc.generateTxnID(),
	}

	// Write to WAL first
	if err := pc.persistence.WriteRecord(ctx, walRecord); err != nil {
		return fmt.Errorf("failed to write WAL record: %w", err)
	}

	// Apply the operation
	if err := pc.VectorCollection.Add(ctx, vector); err != nil {
		return fmt.Errorf("failed to add vector after WAL: %w", err)
	}

	// Update collection statistics
	if err := pc.updateCollectionStats(ctx); err != nil {
		// Log error but don't fail the operation
		// In production, you'd use a proper logger
	}

	// Check if we need to create a snapshot
	if pc.shouldCreateSnapshot() {
		go pc.createSnapshotAsync(ctx)
	}

	return nil
}

// AddBatch adds multiple vectors with WAL logging
func (pc *PersistentVectorCollection) AddBatch(ctx context.Context, vectors []*api.Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	// Validate all vectors first
	for _, vector := range vectors {
		if vector == nil {
			return api.ErrEmptyVector
		}
		if err := vector.Validate(); err != nil {
			return err
		}
	}

	pc.mu.Lock()
	defer pc.mu.Unlock()

	// Create WAL record
	walRecord := &persist.WALRecord{
		Type:    persist.RecordTypeBatchInsert,
		Vectors: vectors,
		TxnID:   pc.generateTxnID(),
	}

	// Write to WAL first
	if err := pc.persistence.WriteRecord(ctx, walRecord); err != nil {
		return fmt.Errorf("failed to write WAL record: %w", err)
	}

	// Apply the operation
	if err := pc.VectorCollection.AddBatch(ctx, vectors); err != nil {
		return fmt.Errorf("failed to add vectors after WAL: %w", err)
	}

	// Update collection statistics
	if err := pc.updateCollectionStats(ctx); err != nil {
		// Log error but don't fail the operation
	}

	// Check if we need to create a snapshot
	if pc.shouldCreateSnapshot() {
		go pc.createSnapshotAsync(ctx)
	}

	return nil
}

// Delete deletes a vector with WAL logging
func (pc *PersistentVectorCollection) Delete(ctx context.Context, id string) error {
	if id == "" {
		return api.ErrVectorNotFound
	}

	pc.mu.Lock()
	defer pc.mu.Unlock()

	// Create WAL record
	walRecord := &persist.WALRecord{
		Type:     persist.RecordTypeDelete,
		VectorID: id,
		TxnID:    pc.generateTxnID(),
	}

	// Write to WAL first
	if err := pc.persistence.WriteRecord(ctx, walRecord); err != nil {
		return fmt.Errorf("failed to write WAL record: %w", err)
	}

	// Apply the operation
	if err := pc.VectorCollection.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete vector after WAL: %w", err)
	}

	// Update collection statistics
	if err := pc.updateCollectionStats(ctx); err != nil {
		// Log error but don't fail the operation
	}

	return nil
}

// DeleteBatch deletes multiple vectors with WAL logging
func (pc *PersistentVectorCollection) DeleteBatch(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	pc.mu.Lock()
	defer pc.mu.Unlock()

	// Create WAL record
	walRecord := &persist.WALRecord{
		Type:      persist.RecordTypeBatchDelete,
		VectorIDs: ids,
		TxnID:     pc.generateTxnID(),
	}

	// Write to WAL first
	if err := pc.persistence.WriteRecord(ctx, walRecord); err != nil {
		return fmt.Errorf("failed to write WAL record: %w", err)
	}

	// Apply the operation
	if err := pc.VectorCollection.DeleteBatch(ctx, ids); err != nil {
		return fmt.Errorf("failed to delete vectors after WAL: %w", err)
	}

	// Update collection statistics
	if err := pc.updateCollectionStats(ctx); err != nil {
		// Log error but don't fail the operation
	}

	return nil
}

// Clear clears all vectors with WAL logging
func (pc *PersistentVectorCollection) Clear(ctx context.Context) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	// Create WAL record
	walRecord := &persist.WALRecord{
		Type:  persist.RecordTypeClear,
		TxnID: pc.generateTxnID(),
	}

	// Write to WAL first
	if err := pc.persistence.WriteRecord(ctx, walRecord); err != nil {
		return fmt.Errorf("failed to write WAL record: %w", err)
	}

	// Apply the operation
	if err := pc.VectorCollection.Clear(ctx); err != nil {
		return fmt.Errorf("failed to clear collection after WAL: %w", err)
	}

	// Update collection statistics
	if err := pc.updateCollectionStats(ctx); err != nil {
		// Log error but don't fail the operation
	}

	return nil
}

// CreateSnapshot creates a snapshot of the collection
func (pc *PersistentVectorCollection) CreateSnapshot(ctx context.Context) (*persist.SnapshotMetadata, error) {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	// Get current LSN
	currentLSN := pc.persistence.GetLastLSN()

	// Create snapshot data
	snapshotData, err := pc.createSnapshotData(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create snapshot data: %w", err)
	}

	// Create snapshot
	metadata, err := pc.persistence.CreateSnapshot(ctx, currentLSN, snapshotData)
	if err != nil {
		return nil, fmt.Errorf("failed to create snapshot: %w", err)
	}

	// Update collection info
	collectionInfo := &CollectionInfo{
		Name:            pc.persistConfig.CollectionConfig.Name,
		LastSnapshotID:  metadata.ID,
		LastSnapshotLSN: currentLSN,
		LastSnapshotAt:  time.Now(),
	}

	if err := pc.manifest.UpdateCollection(ctx, collectionInfo); err != nil {
		// Log error but don't fail snapshot creation
	}

	pc.lastSnapshotLSN = currentLSN

	return metadata, nil
}

// RestoreFromSnapshot restores the collection from a snapshot
func (pc *PersistentVectorCollection) RestoreFromSnapshot(ctx context.Context, snapshotID string) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	// Set collection state to recovering
	if err := pc.manifest.SetCollectionState(ctx, pc.persistConfig.CollectionConfig.Name, CollectionStateRecovering); err != nil {
		return fmt.Errorf("failed to set collection state: %w", err)
	}

	// Restore snapshot
	snapshotData, err := pc.persistence.RestoreSnapshot(ctx, snapshotID)
	if err != nil {
		return fmt.Errorf("failed to restore snapshot: %w", err)
	}

	// Apply snapshot data
	collectionSnapshot, ok := snapshotData.(*persist.CollectionSnapshot)
	if !ok {
		return fmt.Errorf("invalid snapshot data type")
	}

	// Clear current data
	if err := pc.VectorCollection.Clear(ctx); err != nil {
		return fmt.Errorf("failed to clear collection for restore: %w", err)
	}

	// Add vectors from snapshot
	if err := pc.VectorCollection.AddBatch(ctx, collectionSnapshot.Vectors); err != nil {
		return fmt.Errorf("failed to restore vectors from snapshot: %w", err)
	}

	// Set collection state back to active
	if err := pc.manifest.SetCollectionState(ctx, pc.persistConfig.CollectionConfig.Name, CollectionStateActive); err != nil {
		return fmt.Errorf("failed to set collection state after restore: %w", err)
	}

	return nil
}

// Close closes the persistent collection
func (pc *PersistentVectorCollection) Close() error {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	ctx := context.Background()

	// Create final snapshot if needed
	if pc.shouldCreateSnapshot() {
		_, _ = pc.CreateSnapshot(ctx) // Best effort
	}

	// Close persistence layer
	if err := pc.persistence.Stop(ctx); err != nil {
		return fmt.Errorf("failed to stop persistence layer: %w", err)
	}

	// Close manifest
	if err := pc.manifest.Close(ctx); err != nil {
		return fmt.Errorf("failed to close manifest: %w", err)
	}

	// Close base collection
	return pc.VectorCollection.Close()
}

// Private methods

// recoverIfNeeded performs recovery if the collection was not cleanly shut down
func (pc *PersistentVectorCollection) recoverIfNeeded(ctx context.Context) error {
	// Get recovery information from manifest
	lastLSN, lastSnapshotID, recoveryState, err := pc.manifest.GetRecoveryInfo(ctx)
	if err != nil {
		return fmt.Errorf("failed to get recovery info: %w", err)
	}

	// If clean shutdown, no recovery needed
	if recoveryState == "clean" {
		return nil
	}

	// Set recovery state
	if err := pc.manifest.UpdateRecoveryState(ctx, lastLSN, lastSnapshotID, "recovering"); err != nil {
		return fmt.Errorf("failed to set recovery state: %w", err)
	}

	// Restore from latest snapshot if available
	if lastSnapshotID != "" {
		if err := pc.RestoreFromSnapshot(ctx, lastSnapshotID); err != nil {
			return fmt.Errorf("failed to restore from snapshot: %w", err)
		}
		pc.lastSnapshotLSN = lastLSN
	}

	// Replay WAL from snapshot LSN
	fromLSN := pc.lastSnapshotLSN + 1
	if err := pc.persistence.Replay(ctx, fromLSN, pc.recoveryHandler); err != nil {
		return fmt.Errorf("failed to replay WAL: %w", err)
	}

	// Mark recovery as complete
	currentLSN := pc.persistence.GetLastLSN()
	if err := pc.manifest.UpdateRecoveryState(ctx, currentLSN, lastSnapshotID, "clean"); err != nil {
		return fmt.Errorf("failed to update recovery state: %w", err)
	}

	return nil
}

// createSnapshotData creates snapshot data for the collection
func (pc *PersistentVectorCollection) createSnapshotData(ctx context.Context) (*persist.CollectionSnapshot, error) {
	// Get all vectors
	count, err := pc.VectorCollection.Count(ctx)
	if err != nil {
		return nil, err
	}

	vectors, err := pc.VectorCollection.List(ctx, int(count), 0)
	if err != nil {
		return nil, err
	}

	// Get collection stats
	stats, err := pc.VectorCollection.Stats(ctx)
	if err != nil {
		return nil, err
	}

	return &persist.CollectionSnapshot{
		Metadata: pc.persistConfig.CollectionConfig,
		Vectors:  vectors,
		Stats:    stats,
	}, nil
}

// shouldCreateSnapshot determines if a snapshot should be created
func (pc *PersistentVectorCollection) shouldCreateSnapshot() bool {
	if !pc.persistConfig.AutoSnapshotEnabled {
		return false
	}

	currentLSN := pc.persistence.GetLastLSN()
	return currentLSN-pc.lastSnapshotLSN >= uint64(pc.persistConfig.AutoSnapshotThreshold)
}

// createSnapshotAsync creates a snapshot in the background
func (pc *PersistentVectorCollection) createSnapshotAsync(ctx context.Context) {
	_, _ = pc.CreateSnapshot(ctx) // Best effort
}

// updateCollectionStats updates collection statistics in the manifest
func (pc *PersistentVectorCollection) updateCollectionStats(ctx context.Context) error {
	stats, err := pc.VectorCollection.Stats(ctx)
	if err != nil {
		return err
	}

	collectionInfo := &CollectionInfo{
		Name:        pc.persistConfig.CollectionConfig.Name,
		VectorCount: stats.VectorCount,
		IndexSize:   0, // Could be calculated from index stats
		DataSize:    stats.SizeBytes,
	}

	return pc.manifest.UpdateCollection(ctx, collectionInfo)
}

// generateTxnID generates a transaction ID
func (pc *PersistentVectorCollection) generateTxnID() string {
	return fmt.Sprintf("txn-%d", time.Now().UnixNano())
}

// CollectionRecoveryHandler handles WAL replay during recovery
type CollectionRecoveryHandler struct {
	collection *PersistentVectorCollection
}

// NewCollectionRecoveryHandler creates a new recovery handler
func NewCollectionRecoveryHandler(collection *PersistentVectorCollection) *CollectionRecoveryHandler {
	return &CollectionRecoveryHandler{
		collection: collection,
	}
}

// HandleInsert handles an insert record during replay
func (h *CollectionRecoveryHandler) HandleInsert(ctx context.Context, record *persist.WALRecord) error {
	return h.collection.VectorCollection.Add(ctx, record.Vector)
}

// HandleUpdate handles an update record during replay
func (h *CollectionRecoveryHandler) HandleUpdate(ctx context.Context, record *persist.WALRecord) error {
	// For now, treat update as delete + insert
	if err := h.collection.VectorCollection.Delete(ctx, record.Vector.ID); err != nil && err != api.ErrVectorNotFound {
		return err
	}
	return h.collection.VectorCollection.Add(ctx, record.Vector)
}

// HandleDelete handles a delete record during replay
func (h *CollectionRecoveryHandler) HandleDelete(ctx context.Context, record *persist.WALRecord) error {
	err := h.collection.VectorCollection.Delete(ctx, record.VectorID)
	if err == api.ErrVectorNotFound {
		return nil // Ignore not found during recovery
	}
	return err
}

// HandleBatchInsert handles a batch insert record during replay
func (h *CollectionRecoveryHandler) HandleBatchInsert(ctx context.Context, record *persist.WALRecord) error {
	return h.collection.VectorCollection.AddBatch(ctx, record.Vectors)
}

// HandleBatchDelete handles a batch delete record during replay
func (h *CollectionRecoveryHandler) HandleBatchDelete(ctx context.Context, record *persist.WALRecord) error {
	err := h.collection.VectorCollection.DeleteBatch(ctx, record.VectorIDs)
	if err == api.ErrVectorNotFound {
		return nil // Ignore not found during recovery
	}
	return err
}

// HandleClear handles a clear record during replay
func (h *CollectionRecoveryHandler) HandleClear(ctx context.Context, record *persist.WALRecord) error {
	return h.collection.VectorCollection.Clear(ctx)
}

// HandleSnapshot handles a snapshot record during replay
func (h *CollectionRecoveryHandler) HandleSnapshot(ctx context.Context, record *persist.WALRecord) error {
	// Snapshot records are informational, no action needed during replay
	return nil
}
