// Package segment provides segmented storage capabilities for GoVecDB.
// It implements a segment-based architecture that allows the database to scale
// beyond memory limits and provides efficient compaction and merge operations.
package segment

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// Common errors for segment operations
var (
	ErrSegmentNotFound      = errors.New("segment not found")
	ErrSegmentExists        = errors.New("segment already exists")
	ErrSegmentFull          = errors.New("segment is full")
	ErrSegmentClosed        = errors.New("segment is closed")
	ErrInvalidSegmentID     = errors.New("invalid segment ID")
	ErrCompactionInProgress = errors.New("compaction in progress")
	ErrSegmentCorrupted     = errors.New("segment is corrupted")
	ErrInsufficientSpace    = errors.New("insufficient space for operation")
)

// SegmentType represents the type of segment
type SegmentType string

const (
	ActiveSegment    SegmentType = "active"    // Currently accepting writes
	ImmutableSegment SegmentType = "immutable" // Read-only, pending compaction
	CompactedSegment SegmentType = "compacted" // Result of compaction
	TombstoneSegment SegmentType = "tombstone" // Contains deletion markers
)

// SegmentState represents the current state of a segment
type SegmentState string

const (
	SegmentStateActive     SegmentState = "active"
	SegmentStateImmutable  SegmentState = "immutable"
	SegmentStateCompacting SegmentState = "compacting"
	SegmentStateMerging    SegmentState = "merging"
	SegmentStateDeleted    SegmentState = "deleted"
	SegmentStateCorrupted  SegmentState = "corrupted"
)

// SegmentConfig holds configuration for a segment
type SegmentConfig struct {
	// Segment identification
	ID   string      `json:"id"`
	Type SegmentType `json:"type"`

	// Size limits
	MaxVectors   int64 `json:"max_vectors"`    // Maximum number of vectors
	MaxSizeBytes int64 `json:"max_size_bytes"` // Maximum size in bytes

	// Compaction settings
	CompactionThreshold float64 `json:"compaction_threshold"` // Trigger compaction at this fill ratio
	MergeThreshold      int     `json:"merge_threshold"`      // Merge when this many segments exist
	CompactionInterval  int     `json:"compaction_interval"`  // Seconds between compaction checks

	// Performance settings
	BloomFilterBits    int  `json:"bloom_filter_bits"`   // Bits per element in bloom filter
	IndexCacheSize     int  `json:"index_cache_size"`    // Index cache size in MB
	CompressionEnabled bool `json:"compression_enabled"` // Enable compression
	ChecksumEnabled    bool `json:"checksum_enabled"`    // Enable checksums

	// Storage settings
	BasePath     string `json:"base_path"`     // Base directory for segments
	AsyncWrites  bool   `json:"async_writes"`  // Enable asynchronous writes
	SyncInterval int    `json:"sync_interval"` // Seconds between syncs

	// Advanced settings
	MemoryMapEnabled bool  `json:"memory_map_enabled"` // Use memory-mapped files
	DirectIOEnabled  bool  `json:"direct_io_enabled"`  // Use direct I/O
	ReadAheadSize    int64 `json:"read_ahead_size"`    // Read-ahead buffer size
}

// DefaultSegmentConfig returns default segment configuration
func DefaultSegmentConfig(id string) *SegmentConfig {
	return &SegmentConfig{
		ID:                  id,
		Type:                ActiveSegment,
		MaxVectors:          1000000, // 1M vectors
		MaxSizeBytes:        1 << 30, // 1GB
		CompactionThreshold: 0.8,     // 80% full
		MergeThreshold:      10,      // Merge when 10 segments
		CompactionInterval:  300,     // 5 minutes
		BloomFilterBits:     10,      // 10 bits per element
		IndexCacheSize:      100,     // 100MB cache
		CompressionEnabled:  true,
		ChecksumEnabled:     true,
		BasePath:            "./data",
		AsyncWrites:         true,
		SyncInterval:        30, // 30 seconds
		MemoryMapEnabled:    true,
		DirectIOEnabled:     false,
		ReadAheadSize:       1 << 20, // 1MB
	}
}

// Validate validates the segment configuration
func (c *SegmentConfig) Validate() error {
	if c.ID == "" {
		return fmt.Errorf("segment ID cannot be empty")
	}
	if c.MaxVectors <= 0 {
		return fmt.Errorf("max vectors must be positive")
	}
	if c.MaxSizeBytes <= 0 {
		return fmt.Errorf("max size bytes must be positive")
	}
	if c.CompactionThreshold <= 0 || c.CompactionThreshold > 1 {
		return fmt.Errorf("compaction threshold must be between 0 and 1")
	}
	if c.BasePath == "" {
		return fmt.Errorf("base path cannot be empty")
	}
	return nil
}

// SegmentMetadata holds metadata about a segment
type SegmentMetadata struct {
	// Basic information
	ID         string       `json:"id"`
	Type       SegmentType  `json:"type"`
	State      SegmentState `json:"state"`
	Generation int64        `json:"generation"`

	// Size and capacity
	VectorCount int64 `json:"vector_count"`
	SizeBytes   int64 `json:"size_bytes"`
	IndexSize   int64 `json:"index_size"`
	BloomSize   int64 `json:"bloom_size"`

	// Timestamps
	CreatedAt   time.Time `json:"created_at"`
	ModifiedAt  time.Time `json:"modified_at"`
	CompactedAt time.Time `json:"compacted_at"`
	AccessedAt  time.Time `json:"accessed_at"`

	// File information
	DataFile     string `json:"data_file"`
	IndexFile    string `json:"index_file"`
	BloomFile    string `json:"bloom_file"`
	MetadataFile string `json:"metadata_file"`

	// Statistics
	ReadCount  int64  `json:"read_count"`
	WriteCount int64  `json:"write_count"`
	ErrorCount int64  `json:"error_count"`
	Checksum   string `json:"checksum"`

	// Compaction info
	CompactionLevel    int      `json:"compaction_level"`
	SourceSegments     []string `json:"source_segments"`
	CompressionRatio   float64  `json:"compression_ratio"`
	FragmentationRatio float64  `json:"fragmentation_ratio"`
}

// SegmentStats provides runtime statistics about a segment
type SegmentStats struct {
	// Basic metrics
	VectorCount    int64 `json:"vector_count"`
	UniqueVectors  int64 `json:"unique_vectors"`
	DeletedVectors int64 `json:"deleted_vectors"`
	SizeBytes      int64 `json:"size_bytes"`

	// Performance metrics
	ReadLatency     float64 `json:"read_latency_ms"`
	WriteLatency    float64 `json:"write_latency_ms"`
	ThroughputRead  float64 `json:"throughput_read_ops_sec"`
	ThroughputWrite float64 `json:"throughput_write_ops_sec"`

	// Cache metrics
	CacheHitRate   float64 `json:"cache_hit_rate"`
	CacheMissRate  float64 `json:"cache_miss_rate"`
	IndexCacheSize int64   `json:"index_cache_size"`

	// I/O metrics
	DiskReads    int64 `json:"disk_reads"`
	DiskWrites   int64 `json:"disk_writes"`
	BytesRead    int64 `json:"bytes_read"`
	BytesWritten int64 `json:"bytes_written"`

	// Bloom filter metrics
	BloomQueries        int64 `json:"bloom_queries"`
	BloomHits           int64 `json:"bloom_hits"`
	BloomFalsePositives int64 `json:"bloom_false_positives"`

	// Health metrics
	ErrorRate       float64 `json:"error_rate"`
	CorruptionCount int64   `json:"corruption_count"`
	RecoveryCount   int64   `json:"recovery_count"`
}

// Segment defines the interface for a storage segment
type Segment interface {
	// Basic information
	ID() string
	Type() SegmentType
	State() SegmentState
	Metadata() *SegmentMetadata
	Stats() *SegmentStats

	// Vector operations
	Put(ctx context.Context, vector *api.Vector) error
	Get(ctx context.Context, id string) (*api.Vector, error)
	Delete(ctx context.Context, id string) error
	Contains(ctx context.Context, id string) (bool, error)

	// Batch operations
	PutBatch(ctx context.Context, vectors []*api.Vector) error
	GetBatch(ctx context.Context, ids []string) ([]*api.Vector, error)
	DeleteBatch(ctx context.Context, ids []string) error

	// Scanning and iteration
	Scan(ctx context.Context, callback func(*api.Vector) bool) error
	ScanRange(ctx context.Context, start, end string, callback func(*api.Vector) bool) error
	Keys(ctx context.Context) ([]string, error)

	// Filtering
	Filter(ctx context.Context, filter api.FilterExpr) ([]*api.Vector, error)
	FilterKeys(ctx context.Context, filter api.FilterExpr) ([]string, error)

	// State management
	Freeze() error                                             // Make segment immutable
	Compact(ctx context.Context) error                         // Compact segment
	Merge(ctx context.Context, other Segment) (Segment, error) // Merge with another segment

	// Size and capacity
	Size() int64
	VectorCount() int64
	IsEmpty() bool
	IsFull() bool
	CanAcceptWrites() bool

	// Health and maintenance
	Validate(ctx context.Context) error
	Repair(ctx context.Context) error
	Checkpoint(ctx context.Context) error

	// Lifecycle
	Open(ctx context.Context) error
	Close() error
	IsClosed() bool
}

// SegmentManager manages multiple segments and coordinates operations
type SegmentManager interface {
	// Segment lifecycle
	CreateSegment(ctx context.Context, config *SegmentConfig) (Segment, error)
	OpenSegment(ctx context.Context, id string) (Segment, error)
	CloseSegment(ctx context.Context, id string) error
	DeleteSegment(ctx context.Context, id string) error
	Clear(ctx context.Context) error

	// Segment discovery
	ListSegments(ctx context.Context) ([]string, error)
	GetSegment(ctx context.Context, id string) (Segment, error)
	GetActiveSegment(ctx context.Context) (Segment, error)

	// Multi-segment operations
	Put(ctx context.Context, vector *api.Vector) error
	Get(ctx context.Context, id string) (*api.Vector, error)
	Delete(ctx context.Context, id string) error

	// Batch operations across segments
	PutBatch(ctx context.Context, vectors []*api.Vector) error
	GetBatch(ctx context.Context, ids []string) ([]*api.Vector, error)
	DeleteBatch(ctx context.Context, ids []string) error

	// Filtering across segments
	Filter(ctx context.Context, filter api.FilterExpr) ([]*api.Vector, error)
	Scan(ctx context.Context, callback func(*api.Vector) bool) error

	// Compaction and maintenance
	TriggerCompaction(ctx context.Context) error
	TriggerMajorCompaction(ctx context.Context) error
	GetCompactionStatus(ctx context.Context) (*CompactionStatus, error)

	// Segment rotation
	RotateActiveSegment(ctx context.Context) error
	FreezeSegment(ctx context.Context, id string) error

	// Statistics and monitoring
	Stats(ctx context.Context) (*SegmentManagerStats, error)
	Health(ctx context.Context) (*SegmentHealthStatus, error)

	// Configuration
	UpdateConfig(ctx context.Context, config *SegmentManagerConfig) error
	GetConfig(ctx context.Context) *SegmentManagerConfig

	// Lifecycle
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	IsRunning() bool
}

// SegmentManagerConfig holds configuration for the segment manager
type SegmentManagerConfig struct {
	// Segment settings
	DefaultSegmentConfig *SegmentConfig `json:"default_segment_config"`
	MaxActiveSegments    int            `json:"max_active_segments"`
	MaxTotalSegments     int            `json:"max_total_segments"`
	MergeThreshold       int            `json:"merge_threshold"`

	// Compaction settings
	AutoCompactionEnabled   bool   `json:"auto_compaction_enabled"`
	CompactionThreads       int    `json:"compaction_threads"`
	CompactionSchedule      string `json:"compaction_schedule"`
	MajorCompactionSchedule string `json:"major_compaction_schedule"`

	// Memory management
	SegmentCacheSize  int64 `json:"segment_cache_size"`
	MetadataCacheSize int64 `json:"metadata_cache_size"`
	MaxMemoryUsage    int64 `json:"max_memory_usage"`

	// I/O settings
	MaxConcurrentReads  int `json:"max_concurrent_reads"`
	MaxConcurrentWrites int `json:"max_concurrent_writes"`
	IOTimeout           int `json:"io_timeout"`

	// Monitoring
	MetricsEnabled      bool `json:"metrics_enabled"`
	HealthCheckInterval int  `json:"health_check_interval"`
	StatsUpdateInterval int  `json:"stats_update_interval"`
	StatisticsInterval  int  `json:"statistics_interval"`

	// Advanced settings
	BackgroundTasksEnabled bool `json:"background_tasks_enabled"`
	WalEnabled             bool `json:"wal_enabled"`
	ReplicationEnabled     bool `json:"replication_enabled"`
}

// DefaultSegmentManagerConfig returns default configuration for segment manager
func DefaultSegmentManagerConfig() *SegmentManagerConfig {
	return &SegmentManagerConfig{
		DefaultSegmentConfig:    DefaultSegmentConfig("default"),
		MaxActiveSegments:       5,
		MaxTotalSegments:        100,
		MergeThreshold:          10,
		AutoCompactionEnabled:   true,
		CompactionThreads:       2,
		CompactionSchedule:      "0 */30 * * * *", // Every 30 minutes
		MajorCompactionSchedule: "0 0 2 * * *",    // Daily at 2 AM
		SegmentCacheSize:        500 << 20,        // 500MB
		MetadataCacheSize:       50 << 20,         // 50MB
		MaxMemoryUsage:          2 << 30,          // 2GB
		MaxConcurrentReads:      100,
		MaxConcurrentWrites:     10,
		IOTimeout:               30, // 30 seconds
		MetricsEnabled:          true,
		HealthCheckInterval:     60,  // 60 seconds
		StatsUpdateInterval:     30,  // 30 seconds
		StatisticsInterval:      300, // 5 minutes
		BackgroundTasksEnabled:  true,
		WalEnabled:              true,
		ReplicationEnabled:      false,
	}
}

// SegmentManagerStats provides statistics about the segment manager
type SegmentManagerStats struct {
	// Segment counts
	TotalSegments     int64 `json:"total_segments"`
	ActiveSegments    int64 `json:"active_segments"`
	ImmutableSegments int64 `json:"immutable_segments"`
	CompactedSegments int64 `json:"compacted_segments"`

	// Data statistics
	TotalVectors       int64 `json:"total_vectors"`
	TotalSizeBytes     int64 `json:"total_size_bytes"`
	AverageSegmentSize int64 `json:"average_segment_size"`

	// Performance statistics
	ReadThroughput      float64 `json:"read_throughput"`
	WriteThroughput     float64 `json:"write_throughput"`
	AverageReadLatency  float64 `json:"average_read_latency"`
	AverageWriteLatency float64 `json:"average_write_latency"`

	// Compaction statistics
	CompactionsInProgress int64     `json:"compactions_in_progress"`
	LastCompactionTime    time.Time `json:"last_compaction_time"`
	TotalCompactions      int64     `json:"total_compactions"`
	CompactionErrors      int64     `json:"compaction_errors"`

	// Memory usage
	CacheMemoryUsage int64 `json:"cache_memory_usage"`
	IndexMemoryUsage int64 `json:"index_memory_usage"`
	TotalMemoryUsage int64 `json:"total_memory_usage"`

	// Error statistics
	TotalErrors       int64   `json:"total_errors"`
	ErrorRate         float64 `json:"error_rate"`
	CorruptedSegments int64   `json:"corrupted_segments"`
}

// CompactionStatus provides information about ongoing compactions
type CompactionStatus struct {
	// Overall status
	InProgress          bool      `json:"in_progress"`
	StartTime           time.Time `json:"start_time"`
	EstimatedCompletion time.Time `json:"estimated_completion"`
	Progress            float64   `json:"progress"` // 0.0 to 1.0

	// Current operation
	CurrentSegment    string `json:"current_segment"`
	CurrentOperation  string `json:"current_operation"`
	SegmentsProcessed int    `json:"segments_processed"`
	SegmentsTotal     int    `json:"segments_total"`

	// Performance metrics
	RecordsProcessed int64   `json:"records_processed"`
	RecordsTotal     int64   `json:"records_total"`
	BytesProcessed   int64   `json:"bytes_processed"`
	BytesTotal       int64   `json:"bytes_total"`
	ProcessingRate   float64 `json:"processing_rate"` // records per second

	// Results
	InputSegments    []string `json:"input_segments"`
	OutputSegments   []string `json:"output_segments"`
	SpaceSaved       int64    `json:"space_saved"`
	CompressionRatio float64  `json:"compression_ratio"`

	// Errors
	Errors   []string `json:"errors"`
	Warnings []string `json:"warnings"`
}

// SegmentHealthStatus provides health information about segments
type SegmentHealthStatus struct {
	// Overall health
	HealthScore   float64   `json:"health_score"` // 0.0 to 1.0
	Status        string    `json:"status"`       // "healthy", "degraded", "critical"
	LastCheckTime time.Time `json:"last_check_time"`

	// Segment health details
	HealthySegments     int64 `json:"healthy_segments"`
	DegradedSegments    int64 `json:"degraded_segments"`
	CorruptedSegments   int64 `json:"corrupted_segments"`
	UnreachableSegments int64 `json:"unreachable_segments"`

	// Performance health
	PerformanceScore float64 `json:"performance_score"`
	LatencyIssues    bool    `json:"latency_issues"`
	ThroughputIssues bool    `json:"throughput_issues"`

	// Resource health
	MemoryUsageRatio float64 `json:"memory_usage_ratio"`
	DiskUsageRatio   float64 `json:"disk_usage_ratio"`
	IOQueueDepth     int64   `json:"io_queue_depth"`

	// Maintenance needs
	CompactionNeeded bool `json:"compaction_needed"`
	RepairNeeded     bool `json:"repair_needed"`
	CleanupNeeded    bool `json:"cleanup_needed"`

	// Issues and recommendations
	Issues          []string `json:"issues"`
	Recommendations []string `json:"recommendations"`
}

// CompactionPolicy defines how segments should be compacted
type CompactionPolicy interface {
	// Policy evaluation
	ShouldCompact(segments []Segment) bool
	SelectSegmentsForCompaction(segments []Segment) []Segment
	EstimateCompactionBenefit(segments []Segment) CompactionBenefit

	// Configuration
	GetThresholds() CompactionThresholds
	SetThresholds(thresholds CompactionThresholds)
}

// CompactionThresholds holds thresholds for compaction decisions
type CompactionThresholds struct {
	// Size-based thresholds
	MinSegmentSize     int64   `json:"min_segment_size"`
	MaxSegmentSize     int64   `json:"max_segment_size"`
	FragmentationRatio float64 `json:"fragmentation_ratio"`

	// Count-based thresholds
	MinSegmentsToCompact int `json:"min_segments_to_compact"`
	MaxSegmentsToCompact int `json:"max_segments_to_compact"`

	// Performance thresholds
	ReadAmplificationMax  float64 `json:"read_amplification_max"`
	WriteAmplificationMax float64 `json:"write_amplification_max"`

	// Time-based thresholds
	MaxSegmentAge      int64 `json:"max_segment_age"`
	CompactionCooldown int64 `json:"compaction_cooldown"`
}

// CompactionBenefit represents the estimated benefit of compaction
type CompactionBenefit struct {
	SpaceSavings    int64   `json:"space_savings"`
	IOReduction     float64 `json:"io_reduction"`
	PerformanceGain float64 `json:"performance_gain"`
	MaintenanceCost float64 `json:"maintenance_cost"`
	NetBenefit      float64 `json:"net_benefit"`
}

// SegmentIterator provides iteration over segments
type SegmentIterator interface {
	// Iteration control
	HasNext() bool
	Next() (Segment, error)
	Reset() error
	Close() error

	// Filtering
	WithFilter(filter func(Segment) bool) SegmentIterator
	WithType(segmentType SegmentType) SegmentIterator
	WithState(state SegmentState) SegmentIterator

	// Ordering
	OrderBySize() SegmentIterator
	OrderByAge() SegmentIterator
	OrderByActivity() SegmentIterator
}

// SegmentEvent represents events in segment lifecycle
type SegmentEvent struct {
	Type      SegmentEventType `json:"type"`
	SegmentID string           `json:"segment_id"`
	Timestamp time.Time        `json:"timestamp"`
	Details   interface{}      `json:"details"`
}

// SegmentEventType represents the type of segment event
type SegmentEventType string

const (
	SegmentCreated   SegmentEventType = "created"
	SegmentOpened    SegmentEventType = "opened"
	SegmentClosed    SegmentEventType = "closed"
	SegmentFrozen    SegmentEventType = "frozen"
	SegmentCompacted SegmentEventType = "compacted"
	SegmentMerged    SegmentEventType = "merged"
	SegmentDeleted   SegmentEventType = "deleted"
	SegmentCorrupted SegmentEventType = "corrupted"
	SegmentRepaired  SegmentEventType = "repaired"
)

// SegmentEventListener defines callback for segment events
type SegmentEventListener interface {
	OnSegmentEvent(event *SegmentEvent) error
}

// BloomFilter interface for membership testing
type BloomFilter interface {
	Add(key []byte)
	Contains(key []byte) bool
	Size() int64
	Clear()
	FalsePositiveRate() float64
}

// SegmentReader provides read-only access to segment data
type SegmentReader interface {
	// Vector reading
	Get(ctx context.Context, id string) (*api.Vector, error)
	GetBatch(ctx context.Context, ids []string) ([]*api.Vector, error)
	Contains(ctx context.Context, id string) (bool, error)

	// Scanning
	Scan(ctx context.Context, callback func(*api.Vector) bool) error
	ScanKeys(ctx context.Context, callback func(string) bool) error

	// Statistics
	VectorCount() int64
	Size() int64

	// Lifecycle
	Close() error
}

// SegmentWriter provides write access to segment data
type SegmentWriter interface {
	// Vector writing
	Put(ctx context.Context, vector *api.Vector) error
	PutBatch(ctx context.Context, vectors []*api.Vector) error
	Delete(ctx context.Context, id string) error
	DeleteBatch(ctx context.Context, ids []string) error

	// Lifecycle
	Flush(ctx context.Context) error
	Sync(ctx context.Context) error
	Close() error
}

// SegmentIndex provides indexing capabilities for segments
type SegmentIndex interface {
	// Index operations
	Add(ctx context.Context, vectorID string, metadata map[string]interface{}) error
	Remove(ctx context.Context, vectorID string) error
	Update(ctx context.Context, vectorID string, oldMetadata, newMetadata map[string]interface{}) error

	// Querying
	Filter(ctx context.Context, expr api.FilterExpr) ([]string, error)
	EstimateSelectivity(expr api.FilterExpr) float64

	// Management
	Optimize(ctx context.Context) error
	Stats() map[string]interface{}

	// Lifecycle
	Close() error
}

// KeyValueStore provides simple key-value storage for segment metadata
type KeyValueStore interface {
	// Basic operations
	Get(key []byte) ([]byte, error)
	Put(key, value []byte) error
	Delete(key []byte) error

	// Batch operations
	WriteBatch(batch []BatchOperation) error

	// Iteration
	NewIterator(prefix []byte) Iterator

	// Management
	Compact() error
	Stats() map[string]interface{}

	// Lifecycle
	Close() error
}

// BatchOperation represents a batch operation for KeyValueStore
type BatchOperation struct {
	Type  BatchOpType `json:"type"`
	Key   []byte      `json:"key"`
	Value []byte      `json:"value"`
}

// BatchOpType represents the type of batch operation
type BatchOpType string

const (
	BatchOpPut    BatchOpType = "put"
	BatchOpDelete BatchOpType = "delete"
)

// Iterator provides iteration over key-value pairs
type Iterator interface {
	// Iteration
	Valid() bool
	Next()
	Key() []byte
	Value() []byte

	// Control
	Seek(key []byte)
	SeekToFirst()
	SeekToLast()

	// Lifecycle
	Close() error
}

// ConcurrentSegmentManager provides thread-safe segment management
type ConcurrentSegmentManager struct {
	// Core components
	segments      map[string]Segment
	activeSegment Segment

	// Configuration
	config *SegmentManagerConfig

	// Compaction
	compactor *SegmentCompactor
	policy    CompactionPolicy

	// Monitoring
	stats     *SegmentManagerStats
	health    *SegmentHealthStatus
	events    chan *SegmentEvent
	listeners []SegmentEventListener

	// Synchronization
	mu      sync.RWMutex
	running bool

	// Background tasks
	compactionTicker *time.Ticker
	healthTicker     *time.Ticker
	statsTicker      *time.Ticker
	stopChan         chan struct{}
	wg               sync.WaitGroup

	// Sequence for generating segment IDs
	segmentSequence int64
}

// SegmentCompactor handles compaction operations
type SegmentCompactor struct {
	// Configuration
	config *CompactionConfig
	policy CompactionPolicy

	// State
	inProgress map[string]*CompactionTask
	queue      chan *CompactionTask
	workers    []*CompactionWorker

	// Statistics
	stats *CompactionStats

	// Synchronization
	mu      sync.RWMutex
	wg      sync.WaitGroup
	running bool
}

// CompactionConfig holds configuration for compaction
type CompactionConfig struct {
	// Worker settings
	WorkerCount        int `json:"worker_count"`
	MaxConcurrentTasks int `json:"max_concurrent_tasks"`
	TaskTimeout        int `json:"task_timeout"`

	// Performance settings
	BufferSize           int64 `json:"buffer_size"`
	CompressionLevel     int   `json:"compression_level"`
	ChecksumVerification bool  `json:"checksum_verification"`

	// Resource limits
	MaxMemoryUsage int64    `json:"max_memory_usage"`
	MaxDiskUsage   int64    `json:"max_disk_usage"`
	IOLimits       IOLimits `json:"io_limits"`
}

// IOLimits defines I/O resource limits
type IOLimits struct {
	ReadBytesPerSec  int64 `json:"read_bytes_per_sec"`
	WriteBytesPerSec int64 `json:"write_bytes_per_sec"`
	ReadOpsPerSec    int64 `json:"read_ops_per_sec"`
	WriteOpsPerSec   int64 `json:"write_ops_per_sec"`
}

// CompactionTask represents a compaction task
type CompactionTask struct {
	ID            string         `json:"id"`
	Type          CompactionType `json:"type"`
	InputSegments []string       `json:"input_segments"`
	OutputSegment string         `json:"output_segment"`
	Priority      int            `json:"priority"`
	CreatedAt     time.Time      `json:"created_at"`
	StartedAt     time.Time      `json:"started_at"`
	CompletedAt   time.Time      `json:"completed_at"`
	Status        TaskStatus     `json:"status"`
	Progress      float64        `json:"progress"`
	Error         string         `json:"error"`
}

// CompactionType represents the type of compaction
type CompactionType string

const (
	MinorCompaction CompactionType = "minor"
	MajorCompaction CompactionType = "major"
	LevelCompaction CompactionType = "level"
)

// TaskStatus represents the status of a compaction task
type TaskStatus string

const (
	TaskPending   TaskStatus = "pending"
	TaskRunning   TaskStatus = "running"
	TaskCompleted TaskStatus = "completed"
	TaskFailed    TaskStatus = "failed"
	TaskCancelled TaskStatus = "cancelled"
)

// CompactionWorker executes compaction tasks
type CompactionWorker struct {
	ID          int
	compactor   *SegmentCompactor
	currentTask *CompactionTask
	mu          sync.RWMutex
	stopChan    chan struct{}
}

// CompactionStats provides statistics about compaction operations
type CompactionStats struct {
	// Task counts
	TotalTasks     int64 `json:"total_tasks"`
	CompletedTasks int64 `json:"completed_tasks"`
	FailedTasks    int64 `json:"failed_tasks"`
	PendingTasks   int64 `json:"pending_tasks"`
	RunningTasks   int64 `json:"running_tasks"`

	// Performance
	AverageTaskTime    float64 `json:"average_task_time"`
	TotalDataProcessed int64   `json:"total_data_processed"`
	TotalSpaceSaved    int64   `json:"total_space_saved"`

	// Resource usage
	MemoryUsage int64   `json:"memory_usage"`
	DiskUsage   int64   `json:"disk_usage"`
	CPUUsage    float64 `json:"cpu_usage"`
}

// SegmentFactory creates segments with specific implementations
type SegmentFactory interface {
	CreateSegment(config *SegmentConfig) (Segment, error)
	CreateReader(path string) (SegmentReader, error)
	CreateWriter(path string) (SegmentWriter, error)
	SupportedTypes() []SegmentType
}
