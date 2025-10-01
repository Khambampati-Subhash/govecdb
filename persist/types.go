package persist

// Package persist provides durability and recovery mechanisms for GoVecDB.
// It implements Write Ahead Logging (WAL) and snapshot-based persistence.

import (
	"context"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// RecordType represents the type of operation in the WAL
type RecordType uint8

const (
	RecordTypeInsert RecordType = iota + 1
	RecordTypeUpdate
	RecordTypeDelete
	RecordTypeBatchInsert
	RecordTypeBatchDelete
	RecordTypeClear
	RecordTypeSnapshot
	RecordTypeCheckpoint
	RecordTypeCompaction
)

// String returns the string representation of the record type
func (r RecordType) String() string {
	switch r {
	case RecordTypeInsert:
		return "INSERT"
	case RecordTypeUpdate:
		return "UPDATE"
	case RecordTypeDelete:
		return "DELETE"
	case RecordTypeBatchInsert:
		return "BATCH_INSERT"
	case RecordTypeBatchDelete:
		return "BATCH_DELETE"
	case RecordTypeClear:
		return "CLEAR"
	case RecordTypeSnapshot:
		return "SNAPSHOT"
	case RecordTypeCheckpoint:
		return "CHECKPOINT"
	case RecordTypeCompaction:
		return "COMPACTION"
	default:
		return "UNKNOWN"
	}
}

// WALRecord represents a single record in the Write Ahead Log
type WALRecord struct {
	// Header information
	LSN       uint64     `json:"lsn"`       // Log Sequence Number
	Type      RecordType `json:"type"`      // Type of operation
	Timestamp time.Time  `json:"timestamp"` // When the operation occurred
	TxnID     string     `json:"txn_id"`    // Transaction ID (for future ACID support)

	// Checksum for integrity
	Checksum uint32 `json:"checksum"`

	// Payload data
	VectorID  string                 `json:"vector_id,omitempty"`
	Vector    *api.Vector            `json:"vector,omitempty"`
	VectorIDs []string               `json:"vector_ids,omitempty"`
	Vectors   []*api.Vector          `json:"vectors,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// WALConfig represents configuration for the WAL
type WALConfig struct {
	// File paths
	WALDir      string `json:"wal_dir"`
	MaxFileSize int64  `json:"max_file_size"` // Maximum size of a single WAL file

	// Behavior settings
	SyncWrites    bool `json:"sync_writes"`    // Force sync after each write
	BufferSize    int  `json:"buffer_size"`    // Buffer size for writes
	RetentionDays int  `json:"retention_days"` // How long to keep WAL files

	// Performance settings
	FlushInterval     time.Duration `json:"flush_interval"`     // How often to flush buffers
	CompactionEnabled bool          `json:"compaction_enabled"` // Enable automatic compaction

	// Recovery settings
	RecoveryTimeout time.Duration `json:"recovery_timeout"` // Max time for recovery
	VerifyChecksums bool          `json:"verify_checksums"` // Verify checksums during recovery
}

// DefaultWALConfig returns a default WAL configuration
func DefaultWALConfig(walDir string) *WALConfig {
	return &WALConfig{
		WALDir:            walDir,
		MaxFileSize:       100 * 1024 * 1024, // 100 MB
		SyncWrites:        true,
		BufferSize:        64 * 1024, // 64 KB
		RetentionDays:     7,
		FlushInterval:     5 * time.Second,
		CompactionEnabled: true,
		RecoveryTimeout:   30 * time.Minute,
		VerifyChecksums:   true,
	}
}

// SnapshotMetadata contains information about a snapshot
type SnapshotMetadata struct {
	ID          string    `json:"id"`
	Timestamp   time.Time `json:"timestamp"`
	LSN         uint64    `json:"lsn"`          // Last LSN included in snapshot
	VectorCount int64     `json:"vector_count"` // Number of vectors at snapshot time
	FileSize    int64     `json:"file_size"`    // Size of snapshot file
	Checksum    string    `json:"checksum"`     // SHA256 checksum of snapshot data
	Compressed  bool      `json:"compressed"`   // Whether snapshot is compressed
	Version     string    `json:"version"`      // Snapshot format version
}

// WALManager defines the interface for Write Ahead Log management
type WALManager interface {
	// Write operations
	WriteRecord(ctx context.Context, record *WALRecord) error
	WriteBatch(ctx context.Context, records []*WALRecord) error

	// Recovery operations
	Replay(ctx context.Context, fromLSN uint64, handler ReplayHandler) error
	GetLastLSN() uint64

	// Maintenance operations
	Checkpoint(ctx context.Context) (uint64, error)
	Compact(ctx context.Context, beforeLSN uint64) error

	// Lifecycle
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	Sync() error
}

// SnapshotManager defines the interface for snapshot management
type SnapshotManager interface {
	// Create snapshot
	CreateSnapshot(ctx context.Context, lsn uint64, data SnapshotData) (*SnapshotMetadata, error)

	// Restore from snapshot
	RestoreSnapshot(ctx context.Context, snapshotID string) (SnapshotData, error)

	// List and manage snapshots
	ListSnapshots(ctx context.Context) ([]*SnapshotMetadata, error)
	DeleteSnapshot(ctx context.Context, snapshotID string) error

	// Cleanup
	CleanupOldSnapshots(ctx context.Context, keepCount int) error
}

// ReplayHandler handles WAL record replay during recovery
type ReplayHandler interface {
	HandleInsert(ctx context.Context, record *WALRecord) error
	HandleUpdate(ctx context.Context, record *WALRecord) error
	HandleDelete(ctx context.Context, record *WALRecord) error
	HandleBatchInsert(ctx context.Context, record *WALRecord) error
	HandleBatchDelete(ctx context.Context, record *WALRecord) error
	HandleClear(ctx context.Context, record *WALRecord) error
	HandleSnapshot(ctx context.Context, record *WALRecord) error
}

// SnapshotData represents the data to be included in a snapshot
type SnapshotData interface {
	// Serialize converts the data to bytes for storage
	Serialize() ([]byte, error)

	// Deserialize loads data from bytes
	Deserialize(data []byte) error

	// Size returns the size of the data in bytes
	Size() int64
}

// PersistenceManager coordinates WAL and snapshot operations
type PersistenceManager interface {
	WALManager
	SnapshotManager

	// Recovery operations
	Recover(ctx context.Context) error

	// Health checks
	HealthCheck(ctx context.Context) error

	// Statistics
	GetStats(ctx context.Context) (*PersistenceStats, error)
}

// PersistenceStats contains statistics about persistence operations
type PersistenceStats struct {
	// WAL stats
	WALSize      int64     `json:"wal_size"`
	WALFileCount int       `json:"wal_file_count"`
	LastLSN      uint64    `json:"last_lsn"`
	WriteRate    float64   `json:"write_rate"` // Records per second
	LastSync     time.Time `json:"last_sync"`

	// Snapshot stats
	SnapshotCount        int       `json:"snapshot_count"`
	LastSnapshotTime     time.Time `json:"last_snapshot_time"`
	LastSnapshotSize     int64     `json:"last_snapshot_size"`
	SnapshotCreationRate float64   `json:"snapshot_creation_rate"` // Snapshots per hour

	// Recovery stats
	LastRecoveryTime     time.Time     `json:"last_recovery_time"`
	LastRecoveryDuration time.Duration `json:"last_recovery_duration"`
	RecoveredRecords     int64         `json:"recovered_records"`

	// Error tracking
	WriteErrors    int64 `json:"write_errors"`
	RecoveryErrors int64 `json:"recovery_errors"`
	ChecksumErrors int64 `json:"checksum_errors"`
}

// CollectionSnapshot represents a snapshot of a collection's data
type CollectionSnapshot struct {
	Metadata *api.CollectionConfig `json:"metadata"`
	Vectors  []*api.Vector         `json:"vectors"`
	Stats    *api.CollectionStats  `json:"stats"`
}

// Constants for WAL file format
const (
	WALMagicNumber = uint32(0x57414C47) // "WALG"
	WALVersion     = uint16(1)
	MaxRecordSize  = 64 * 1024 * 1024 // 64 MB max record size
)

// File header for WAL files
type WALFileHeader struct {
	MagicNumber uint32 `json:"magic_number"`
	Version     uint16 `json:"version"`
	Created     int64  `json:"created"`
	FirstLSN    uint64 `json:"first_lsn"`
	LastLSN     uint64 `json:"last_lsn"`
	RecordCount uint32 `json:"record_count"`
	Checksum    uint32 `json:"checksum"`
}
