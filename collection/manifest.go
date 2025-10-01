// Package collection provides manifest functionality for GoVecDB.
// The manifest stores database metadata, collection information, and recovery state.
package collection

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// ManifestVersion represents the version of the manifest format
const ManifestVersion = "1.0.0"

// DatabaseManifest represents the top-level database manifest
type DatabaseManifest struct {
	Version     string                     `json:"version"`
	DatabaseID  string                     `json:"database_id"`
	CreatedAt   time.Time                  `json:"created_at"`
	UpdatedAt   time.Time                  `json:"updated_at"`
	Collections map[string]*CollectionInfo `json:"collections"`

	// Recovery information
	LastLSN        uint64 `json:"last_lsn"`
	LastSnapshotID string `json:"last_snapshot_id"`
	RecoveryState  string `json:"recovery_state"` // "clean", "recovering", "dirty"

	// Metadata
	Properties map[string]interface{} `json:"properties,omitempty"`
}

// CollectionInfo represents information about a single collection
type CollectionInfo struct {
	Name      string                `json:"name"`
	Config    *api.CollectionConfig `json:"config"`
	State     CollectionState       `json:"state"`
	CreatedAt time.Time             `json:"created_at"`
	UpdatedAt time.Time             `json:"updated_at"`

	// Persistence information
	LastSnapshotID  string    `json:"last_snapshot_id,omitempty"`
	LastSnapshotLSN uint64    `json:"last_snapshot_lsn,omitempty"`
	LastSnapshotAt  time.Time `json:"last_snapshot_at,omitempty"`

	// Statistics
	VectorCount     int64     `json:"vector_count"`
	IndexSize       int64     `json:"index_size"`
	DataSize        int64     `json:"data_size"`
	LastOptimizedAt time.Time `json:"last_optimized_at,omitempty"`

	// Segments (for future segmented storage)
	Segments []SegmentInfo `json:"segments,omitempty"`

	// Properties
	Properties map[string]interface{} `json:"properties,omitempty"`
}

// CollectionState represents the state of a collection
type CollectionState string

const (
	CollectionStateActive     CollectionState = "active"
	CollectionStateCreating   CollectionState = "creating"
	CollectionStateDropping   CollectionState = "dropping"
	CollectionStateRecovering CollectionState = "recovering"
	CollectionStateOptimizing CollectionState = "optimizing"
	CollectionStateError      CollectionState = "error"
)

// SegmentInfo represents information about a data segment
type SegmentInfo struct {
	ID          string    `json:"id"`
	CreatedAt   time.Time `json:"created_at"`
	VectorCount int64     `json:"vector_count"`
	Size        int64     `json:"size"`
	FilePath    string    `json:"file_path"`
	Compacted   bool      `json:"compacted"`

	// LSN range covered by this segment
	FirstLSN uint64 `json:"first_lsn"`
	LastLSN  uint64 `json:"last_lsn"`
}

// ManifestManager manages the database manifest
type ManifestManager struct {
	mu           sync.RWMutex
	manifestPath string
	manifest     *DatabaseManifest

	// Auto-save configuration
	autoSave     bool
	saveInterval time.Duration
	ticker       *time.Ticker
	stopChan     chan struct{}
}

// ManifestConfig represents configuration for the manifest manager
type ManifestConfig struct {
	ManifestPath string        `json:"manifest_path"`
	AutoSave     bool          `json:"auto_save"`
	SaveInterval time.Duration `json:"save_interval"`
	BackupCount  int           `json:"backup_count"` // Number of manifest backups to keep
}

// DefaultManifestConfig returns a default manifest configuration
func DefaultManifestConfig(manifestPath string) *ManifestConfig {
	return &ManifestConfig{
		ManifestPath: manifestPath,
		AutoSave:     true,
		SaveInterval: 30 * time.Second,
		BackupCount:  5,
	}
}

// NewManifestManager creates a new manifest manager
func NewManifestManager(config *ManifestConfig) (*ManifestManager, error) {
	if config == nil {
		return nil, fmt.Errorf("manifest config cannot be nil")
	}

	if config.ManifestPath == "" {
		return nil, fmt.Errorf("manifest path cannot be empty")
	}

	// Ensure directory exists
	manifestDir := filepath.Dir(config.ManifestPath)
	if err := os.MkdirAll(manifestDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create manifest directory: %w", err)
	}

	mm := &ManifestManager{
		manifestPath: config.ManifestPath,
		autoSave:     config.AutoSave,
		saveInterval: config.SaveInterval,
		stopChan:     make(chan struct{}),
	}

	// Try to load existing manifest
	if err := mm.load(); err != nil {
		// If loading fails, create a new manifest
		mm.manifest = mm.createNewManifest()
	}

	// Start auto-save if enabled
	if config.AutoSave && config.SaveInterval > 0 {
		mm.startAutoSave()
	}

	return mm, nil
}

// CreateCollection registers a new collection in the manifest
func (mm *ManifestManager) CreateCollection(ctx context.Context, config *api.CollectionConfig) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	if mm.manifest.Collections == nil {
		mm.manifest.Collections = make(map[string]*CollectionInfo)
	}

	// Check if collection already exists
	if _, exists := mm.manifest.Collections[config.Name]; exists {
		return api.ErrCollectionExists
	}

	// Create collection info
	collectionInfo := &CollectionInfo{
		Name:        config.Name,
		Config:      config,
		State:       CollectionStateCreating,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		VectorCount: 0,
		IndexSize:   0,
		DataSize:    0,
		Properties:  make(map[string]interface{}),
	}

	mm.manifest.Collections[config.Name] = collectionInfo
	mm.manifest.UpdatedAt = time.Now()

	return mm.saveIfNeeded()
}

// GetCollection retrieves collection information from the manifest
func (mm *ManifestManager) GetCollection(ctx context.Context, name string) (*CollectionInfo, error) {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	collection, exists := mm.manifest.Collections[name]
	if !exists {
		return nil, api.ErrCollectionNotFound
	}

	// Return a copy to prevent external modifications
	collectionCopy := *collection
	return &collectionCopy, nil
}

// UpdateCollection updates collection information in the manifest
func (mm *ManifestManager) UpdateCollection(ctx context.Context, info *CollectionInfo) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	if mm.manifest.Collections == nil {
		return api.ErrCollectionNotFound
	}

	existing, exists := mm.manifest.Collections[info.Name]
	if !exists {
		return api.ErrCollectionNotFound
	}

	// Update the information
	existing.State = info.State
	existing.UpdatedAt = time.Now()
	existing.VectorCount = info.VectorCount
	existing.IndexSize = info.IndexSize
	existing.DataSize = info.DataSize
	existing.LastSnapshotID = info.LastSnapshotID
	existing.LastSnapshotLSN = info.LastSnapshotLSN
	existing.LastSnapshotAt = info.LastSnapshotAt
	existing.LastOptimizedAt = info.LastOptimizedAt

	// Merge properties
	if info.Properties != nil {
		if existing.Properties == nil {
			existing.Properties = make(map[string]interface{})
		}
		for k, v := range info.Properties {
			existing.Properties[k] = v
		}
	}

	mm.manifest.UpdatedAt = time.Now()

	return mm.saveIfNeeded()
}

// DropCollection removes a collection from the manifest
func (mm *ManifestManager) DropCollection(ctx context.Context, name string) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	if mm.manifest.Collections == nil {
		return api.ErrCollectionNotFound
	}

	if _, exists := mm.manifest.Collections[name]; !exists {
		return api.ErrCollectionNotFound
	}

	delete(mm.manifest.Collections, name)
	mm.manifest.UpdatedAt = time.Now()

	return mm.saveIfNeeded()
}

// ListCollections returns a list of all collection names
func (mm *ManifestManager) ListCollections(ctx context.Context) ([]string, error) {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	if mm.manifest.Collections == nil {
		return []string{}, nil
	}

	names := make([]string, 0, len(mm.manifest.Collections))
	for name := range mm.manifest.Collections {
		names = append(names, name)
	}

	return names, nil
}

// SetCollectionState updates the state of a collection
func (mm *ManifestManager) SetCollectionState(ctx context.Context, name string, state CollectionState) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	if mm.manifest.Collections == nil {
		return api.ErrCollectionNotFound
	}

	collection, exists := mm.manifest.Collections[name]
	if !exists {
		return api.ErrCollectionNotFound
	}

	collection.State = state
	collection.UpdatedAt = time.Now()
	mm.manifest.UpdatedAt = time.Now()

	return mm.saveIfNeeded()
}

// UpdateRecoveryState updates the recovery state information
func (mm *ManifestManager) UpdateRecoveryState(ctx context.Context, lastLSN uint64, snapshotID string, state string) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	mm.manifest.LastLSN = lastLSN
	if snapshotID != "" {
		mm.manifest.LastSnapshotID = snapshotID
	}
	mm.manifest.RecoveryState = state
	mm.manifest.UpdatedAt = time.Now()

	return mm.saveIfNeeded()
}

// GetRecoveryInfo returns recovery information
func (mm *ManifestManager) GetRecoveryInfo(ctx context.Context) (uint64, string, string, error) {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	return mm.manifest.LastLSN, mm.manifest.LastSnapshotID, mm.manifest.RecoveryState, nil
}

// Save forces a save of the manifest
func (mm *ManifestManager) Save(ctx context.Context) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	return mm.save()
}

// Close closes the manifest manager and performs final save
func (mm *ManifestManager) Close(ctx context.Context) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	// Stop auto-save
	mm.stopAutoSave()

	// Final save
	return mm.save()
}

// GetManifest returns a copy of the current manifest
func (mm *ManifestManager) GetManifest(ctx context.Context) (*DatabaseManifest, error) {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	// Create a deep copy
	manifestCopy := *mm.manifest

	// Copy collections map
	if mm.manifest.Collections != nil {
		manifestCopy.Collections = make(map[string]*CollectionInfo)
		for name, info := range mm.manifest.Collections {
			infoCopy := *info
			manifestCopy.Collections[name] = &infoCopy
		}
	}

	return &manifestCopy, nil
}

// SetProperty sets a database-level property
func (mm *ManifestManager) SetProperty(ctx context.Context, key string, value interface{}) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	if mm.manifest.Properties == nil {
		mm.manifest.Properties = make(map[string]interface{})
	}

	mm.manifest.Properties[key] = value
	mm.manifest.UpdatedAt = time.Now()

	return mm.saveIfNeeded()
}

// GetProperty gets a database-level property
func (mm *ManifestManager) GetProperty(ctx context.Context, key string) (interface{}, bool) {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	if mm.manifest.Properties == nil {
		return nil, false
	}

	value, exists := mm.manifest.Properties[key]
	return value, exists
}

// Private methods

// createNewManifest creates a new empty manifest
func (mm *ManifestManager) createNewManifest() *DatabaseManifest {
	return &DatabaseManifest{
		Version:       ManifestVersion,
		DatabaseID:    fmt.Sprintf("db-%d", time.Now().UnixNano()),
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
		Collections:   make(map[string]*CollectionInfo),
		RecoveryState: "clean",
		Properties:    make(map[string]interface{}),
	}
}

// load loads the manifest from disk
func (mm *ManifestManager) load() error {
	data, err := os.ReadFile(mm.manifestPath)
	if err != nil {
		if os.IsNotExist(err) {
			// File doesn't exist, will create new
			return fmt.Errorf("manifest file does not exist")
		}
		return fmt.Errorf("failed to read manifest file: %w", err)
	}

	var manifest DatabaseManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return fmt.Errorf("failed to unmarshal manifest: %w", err)
	}

	// Validate manifest version
	if manifest.Version != ManifestVersion {
		return fmt.Errorf("unsupported manifest version: %s (expected %s)",
			manifest.Version, ManifestVersion)
	}

	mm.manifest = &manifest
	return nil
}

// save saves the manifest to disk
func (mm *ManifestManager) save() error {
	if mm.manifest == nil {
		return fmt.Errorf("no manifest to save")
	}

	// Update timestamp
	mm.manifest.UpdatedAt = time.Now()

	// Marshal to JSON with indentation for readability
	data, err := json.MarshalIndent(mm.manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}

	// Write to temporary file first for atomic update
	tempPath := mm.manifestPath + ".tmp"

	if err := os.WriteFile(tempPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write manifest to temp file: %w", err)
	}

	// Atomic rename
	if err := os.Rename(tempPath, mm.manifestPath); err != nil {
		os.Remove(tempPath) // Clean up temp file
		return fmt.Errorf("failed to rename manifest file: %w", err)
	}

	return nil
}

// saveIfNeeded saves the manifest only if auto-save is disabled
func (mm *ManifestManager) saveIfNeeded() error {
	if !mm.autoSave {
		return mm.save()
	}
	return nil
}

// startAutoSave starts the auto-save routine
func (mm *ManifestManager) startAutoSave() {
	if mm.saveInterval <= 0 {
		return
	}

	mm.ticker = time.NewTicker(mm.saveInterval)

	go func() {
		for {
			select {
			case <-mm.ticker.C:
				mm.mu.Lock()
				_ = mm.save() // Ignore errors in background save
				mm.mu.Unlock()
			case <-mm.stopChan:
				return
			}
		}
	}()
}

// stopAutoSave stops the auto-save routine
func (mm *ManifestManager) stopAutoSave() {
	if mm.ticker != nil {
		mm.ticker.Stop()
	}

	select {
	case mm.stopChan <- struct{}{}:
	default:
		// Channel might be full or closed
	}
}

// Validation methods

// ValidateManifest validates the manifest structure and data
func (mm *ManifestManager) ValidateManifest(ctx context.Context) error {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	if mm.manifest == nil {
		return fmt.Errorf("manifest is nil")
	}

	// Validate version
	if mm.manifest.Version != ManifestVersion {
		return fmt.Errorf("invalid manifest version: %s", mm.manifest.Version)
	}

	// Validate database ID
	if mm.manifest.DatabaseID == "" {
		return fmt.Errorf("database ID cannot be empty")
	}

	// Validate collection configurations
	if mm.manifest.Collections != nil {
		for name, info := range mm.manifest.Collections {
			if info.Name != name {
				return fmt.Errorf("collection name mismatch: %s != %s", info.Name, name)
			}

			if info.Config != nil {
				if err := info.Config.Validate(); err != nil {
					return fmt.Errorf("invalid config for collection %s: %w", name, err)
				}
			}
		}
	}

	return nil
}

// RepairManifest attempts to repair a corrupted manifest
func (mm *ManifestManager) RepairManifest(ctx context.Context) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	if mm.manifest == nil {
		mm.manifest = mm.createNewManifest()
		return mm.save()
	}

	// Fix missing fields
	if mm.manifest.Version == "" {
		mm.manifest.Version = ManifestVersion
	}

	if mm.manifest.DatabaseID == "" {
		mm.manifest.DatabaseID = fmt.Sprintf("db-%d", time.Now().UnixNano())
	}

	if mm.manifest.Collections == nil {
		mm.manifest.Collections = make(map[string]*CollectionInfo)
	}

	if mm.manifest.Properties == nil {
		mm.manifest.Properties = make(map[string]interface{})
	}

	if mm.manifest.RecoveryState == "" {
		mm.manifest.RecoveryState = "clean"
	}

	// Fix collection info
	for name, info := range mm.manifest.Collections {
		if info.Name == "" {
			info.Name = name
		}

		if info.Properties == nil {
			info.Properties = make(map[string]interface{})
		}

		if info.State == "" {
			info.State = CollectionStateActive
		}
	}

	mm.manifest.UpdatedAt = time.Now()

	return mm.save()
}
