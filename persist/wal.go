// Package persist provides Write Ahead Logging (WAL) functionality for GoVecDB.
// The WAL ensures durability by logging all operations before applying them.
package persist

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// walWriter handles the actual writing of WAL records to files
type walWriter struct {
	file        *os.File
	writer      *bufio.Writer
	fileSize    int64
	recordCount uint32
	firstLSN    uint64
	lastLSN     uint64
	filePath    string
}

// WAL implements the WALManager interface
type WAL struct {
	config *WALConfig
	mu     sync.RWMutex

	// File management
	currentWriter *walWriter
	walDir        string

	// LSN management
	currentLSN uint64

	// Background operations
	flushTicker *time.Ticker
	stopChan    chan struct{}
	flushChan   chan struct{}

	// Statistics
	stats struct {
		writeCount    uint64
		errorCount    uint64
		lastFlushTime time.Time
	}

	// State
	started bool
	stopped bool
}

// NewWAL creates a new Write Ahead Log instance
func NewWAL(config *WALConfig) (*WAL, error) {
	if config == nil {
		return nil, fmt.Errorf("WAL config cannot be nil")
	}

	// Validate configuration
	if config.WALDir == "" {
		return nil, fmt.Errorf("WAL directory cannot be empty")
	}

	if config.MaxFileSize <= 0 {
		config.MaxFileSize = 100 * 1024 * 1024 // 100 MB default
	}

	if config.BufferSize <= 0 {
		config.BufferSize = 64 * 1024 // 64 KB default
	}

	// Create WAL directory if it doesn't exist
	if err := os.MkdirAll(config.WALDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create WAL directory: %w", err)
	}

	wal := &WAL{
		config:    config,
		walDir:    config.WALDir,
		stopChan:  make(chan struct{}),
		flushChan: make(chan struct{}, 1),
	}

	// Initialize LSN from existing files
	if err := wal.initializeLSN(); err != nil {
		return nil, fmt.Errorf("failed to initialize LSN: %w", err)
	}

	return wal, nil
}

// Start starts the WAL and background operations
func (w *WAL) Start(ctx context.Context) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.started {
		return fmt.Errorf("WAL already started")
	}

	// Create initial WAL file
	if err := w.createNewWriter(); err != nil {
		return fmt.Errorf("failed to create initial WAL file: %w", err)
	}

	// Start background flush routine
	w.flushTicker = time.NewTicker(w.config.FlushInterval)
	go w.backgroundFlush()

	w.started = true
	return nil
}

// Stop stops the WAL and flushes all pending data
func (w *WAL) Stop(ctx context.Context) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.stopped {
		return nil
	}

	w.stopped = true

	// Stop background operations
	if w.flushTicker != nil {
		w.flushTicker.Stop()
	}
	close(w.stopChan)

	// Final flush and close
	if w.currentWriter != nil {
		if err := w.flushWriter(); err != nil {
			return fmt.Errorf("failed to flush WAL on stop: %w", err)
		}
		if err := w.closeCurrentWriter(); err != nil {
			return fmt.Errorf("failed to close WAL writer: %w", err)
		}
	}

	return nil
}

// WriteRecord writes a single record to the WAL
func (w *WAL) WriteRecord(ctx context.Context, record *WALRecord) error {
	if record == nil {
		return fmt.Errorf("WAL record cannot be nil")
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if w.stopped {
		return fmt.Errorf("WAL is stopped")
	}

	// Assign LSN and set timestamp
	record.LSN = atomic.AddUint64(&w.currentLSN, 1)
	record.Timestamp = time.Now()

	// Calculate checksum
	record.Checksum = w.calculateChecksum(record)

	// Serialize record
	data, err := w.serializeRecord(record)
	if err != nil {
		atomic.AddUint64(&w.stats.errorCount, 1)
		return fmt.Errorf("failed to serialize WAL record: %w", err)
	}

	// Check if we need to rotate the file
	if w.currentWriter != nil && w.currentWriter.fileSize+int64(len(data)) > w.config.MaxFileSize {
		if err := w.rotateFile(); err != nil {
			atomic.AddUint64(&w.stats.errorCount, 1)
			return fmt.Errorf("failed to rotate WAL file: %w", err)
		}
	}

	// Write record
	if err := w.writeRecordData(data, record.LSN); err != nil {
		atomic.AddUint64(&w.stats.errorCount, 1)
		return fmt.Errorf("failed to write WAL record: %w", err)
	}

	// Sync if configured
	if w.config.SyncWrites {
		if err := w.sync(); err != nil {
			atomic.AddUint64(&w.stats.errorCount, 1)
			return fmt.Errorf("failed to sync WAL: %w", err)
		}
	}

	atomic.AddUint64(&w.stats.writeCount, 1)
	return nil
}

// writeRecordLocked writes a single record to the WAL (assumes mutex is already held)
func (w *WAL) writeRecordLocked(ctx context.Context, record *WALRecord) error {
	if record == nil {
		return fmt.Errorf("WAL record cannot be nil")
	}

	if w.stopped {
		return fmt.Errorf("WAL is stopped")
	}

	// Assign LSN and set timestamp
	record.LSN = atomic.AddUint64(&w.currentLSN, 1)
	record.Timestamp = time.Now()

	// Calculate checksum
	record.Checksum = w.calculateChecksum(record)

	// Serialize record
	data, err := w.serializeRecord(record)
	if err != nil {
		atomic.AddUint64(&w.stats.errorCount, 1)
		return fmt.Errorf("failed to serialize WAL record: %w", err)
	}

	// Check if we need to rotate the file
	if w.currentWriter != nil && w.currentWriter.fileSize+int64(len(data)) > w.config.MaxFileSize {
		if err := w.rotateFile(); err != nil {
			atomic.AddUint64(&w.stats.errorCount, 1)
			return fmt.Errorf("failed to rotate WAL file: %w", err)
		}
	}

	// Write record
	if err := w.writeRecordData(data, record.LSN); err != nil {
		atomic.AddUint64(&w.stats.errorCount, 1)
		return fmt.Errorf("failed to write WAL record: %w", err)
	}

	// Sync if configured
	if w.config.SyncWrites {
		if err := w.sync(); err != nil {
			atomic.AddUint64(&w.stats.errorCount, 1)
			return fmt.Errorf("failed to sync WAL: %w", err)
		}
	}

	atomic.AddUint64(&w.stats.writeCount, 1)
	return nil
}

// WriteBatch writes multiple records as a batch
func (w *WAL) WriteBatch(ctx context.Context, records []*WALRecord) error {
	if len(records) == 0 {
		return nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if w.stopped {
		return fmt.Errorf("WAL is stopped")
	}

	// Process all records
	totalSize := int64(0)
	serializedRecords := make([][]byte, len(records))

	for i, record := range records {
		if record == nil {
			return fmt.Errorf("WAL record %d cannot be nil", i)
		}

		// Assign LSN and timestamp
		record.LSN = atomic.AddUint64(&w.currentLSN, 1)
		record.Timestamp = time.Now()
		record.Checksum = w.calculateChecksum(record)

		// Serialize
		data, err := w.serializeRecord(record)
		if err != nil {
			atomic.AddUint64(&w.stats.errorCount, 1)
			return fmt.Errorf("failed to serialize WAL record %d: %w", i, err)
		}

		serializedRecords[i] = data
		totalSize += int64(len(data))
	}

	// Check if we need to rotate before writing
	if w.currentWriter != nil && w.currentWriter.fileSize+totalSize > w.config.MaxFileSize {
		if err := w.rotateFile(); err != nil {
			atomic.AddUint64(&w.stats.errorCount, 1)
			return fmt.Errorf("failed to rotate WAL file: %w", err)
		}
	}

	// Write all records
	for i, data := range serializedRecords {
		if err := w.writeRecordData(data, records[i].LSN); err != nil {
			atomic.AddUint64(&w.stats.errorCount, 1)
			return fmt.Errorf("failed to write WAL record %d: %w", i, err)
		}
	}

	// Sync if configured
	if w.config.SyncWrites {
		if err := w.sync(); err != nil {
			atomic.AddUint64(&w.stats.errorCount, 1)
			return fmt.Errorf("failed to sync WAL batch: %w", err)
		}
	}

	atomic.AddUint64(&w.stats.writeCount, uint64(len(records)))
	return nil
}

// Replay replays WAL records starting from the specified LSN
func (w *WAL) Replay(ctx context.Context, fromLSN uint64, handler ReplayHandler) error {
	w.mu.RLock()
	defer w.mu.RUnlock()

	// Get all WAL files
	files, err := w.getWALFiles()
	if err != nil {
		return fmt.Errorf("failed to get WAL files: %w", err)
	}

	// Sort files by their first LSN
	sort.Slice(files, func(i, j int) bool {
		return files[i].firstLSN < files[j].firstLSN
	})

	recordsReplayed := int64(0)

	for _, fileInfo := range files {
		// Skip files that are entirely before our target LSN
		if fileInfo.lastLSN < fromLSN {
			continue
		}

		if err := w.replayFile(ctx, fileInfo.path, fromLSN, handler, &recordsReplayed); err != nil {
			return fmt.Errorf("failed to replay file %s: %w", fileInfo.path, err)
		}

		// Check for context cancellation
		if ctx.Err() != nil {
			return ctx.Err()
		}
	}

	return nil
}

// GetLastLSN returns the last assigned LSN
func (w *WAL) GetLastLSN() uint64 {
	return atomic.LoadUint64(&w.currentLSN)
}

// Checkpoint creates a checkpoint and returns the LSN
func (w *WAL) Checkpoint(ctx context.Context) (uint64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Flush current data
	if err := w.flushWriter(); err != nil {
		return 0, fmt.Errorf("failed to flush before checkpoint: %w", err)
	}

	// Get current LSN
	checkpointLSN := atomic.LoadUint64(&w.currentLSN)

	// Write checkpoint record
	checkpointRecord := &WALRecord{
		Type:      RecordTypeCheckpoint,
		Timestamp: time.Now(),
		Metadata: map[string]interface{}{
			"checkpoint_lsn": checkpointLSN,
		},
	}

	if err := w.writeRecordLocked(ctx, checkpointRecord); err != nil {
		return 0, fmt.Errorf("failed to write checkpoint record: %w", err)
	}

	return checkpointLSN, nil
}

// Compact removes WAL files with LSN before the specified value
func (w *WAL) Compact(ctx context.Context, beforeLSN uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	files, err := w.getWALFiles()
	if err != nil {
		return fmt.Errorf("failed to get WAL files for compaction: %w", err)
	}

	removedCount := 0

	for _, fileInfo := range files {
		// Don't remove files that contain LSNs we need
		if fileInfo.lastLSN >= beforeLSN {
			continue
		}

		// Don't remove the current file
		if w.currentWriter != nil && fileInfo.path == w.currentWriter.filePath {
			continue
		}

		if err := os.Remove(fileInfo.path); err != nil {
			return fmt.Errorf("failed to remove WAL file %s: %w", fileInfo.path, err)
		}

		removedCount++
	}

	return nil
}

// Sync forces a sync of the current WAL file
func (w *WAL) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	return w.sync()
}

// Private methods

// initializeLSN initializes the current LSN from existing WAL files
func (w *WAL) initializeLSN() error {
	files, err := w.getWALFiles()
	if err != nil {
		return err
	}

	maxLSN := uint64(0)
	for _, fileInfo := range files {
		if fileInfo.lastLSN > maxLSN {
			maxLSN = fileInfo.lastLSN
		}
	}

	atomic.StoreUint64(&w.currentLSN, maxLSN)
	return nil
}

// createNewWriter creates a new WAL file writer
func (w *WAL) createNewWriter() error {
	// Close existing writer
	if w.currentWriter != nil {
		if err := w.closeCurrentWriter(); err != nil {
			return err
		}
	}

	// Create new file
	timestamp := time.Now().UnixNano()
	filename := fmt.Sprintf("wal-%d.log", timestamp)
	filePath := filepath.Join(w.walDir, filename)

	file, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to create WAL file: %w", err)
	}

	writer := bufio.NewWriterSize(file, w.config.BufferSize)

	w.currentWriter = &walWriter{
		file:     file,
		writer:   writer,
		filePath: filePath,
		firstLSN: atomic.LoadUint64(&w.currentLSN) + 1,
	}

	// Write file header
	return w.writeFileHeader()
}

// writeFileHeader writes the WAL file header
func (w *WAL) writeFileHeader() error {
	header := WALFileHeader{
		MagicNumber: WALMagicNumber,
		Version:     WALVersion,
		Created:     time.Now().UnixNano(),
		FirstLSN:    w.currentWriter.firstLSN,
	}

	headerData, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("failed to marshal header: %w", err)
	}

	// Write header length + header data
	if err := binary.Write(w.currentWriter.writer, binary.LittleEndian, uint32(len(headerData))); err != nil {
		return err
	}

	if _, err := w.currentWriter.writer.Write(headerData); err != nil {
		return err
	}

	w.currentWriter.fileSize += 4 + int64(len(headerData))
	return nil
}

// serializeRecord serializes a WAL record to bytes
func (w *WAL) serializeRecord(record *WALRecord) ([]byte, error) {
	data, err := json.Marshal(record)
	if err != nil {
		return nil, err
	}

	if len(data) > MaxRecordSize {
		return nil, fmt.Errorf("record size %d exceeds maximum %d", len(data), MaxRecordSize)
	}

	return data, nil
}

// writeRecordData writes serialized record data to the current writer
func (w *WAL) writeRecordData(data []byte, lsn uint64) error {
	if w.currentWriter == nil {
		return fmt.Errorf("no active WAL writer")
	}

	// Write record length first
	recordLen := uint32(len(data))
	if err := binary.Write(w.currentWriter.writer, binary.LittleEndian, recordLen); err != nil {
		return fmt.Errorf("failed to write record length: %w", err)
	}

	// Write record data
	n, err := w.currentWriter.writer.Write(data)
	if err != nil {
		return err
	}

	// Update statistics (include the 4 bytes for length prefix)
	totalBytes := int64(4 + n)
	w.currentWriter.fileSize += totalBytes
	w.currentWriter.recordCount++
	w.currentWriter.lastLSN = lsn

	return nil
}

// rotateFile rotates to a new WAL file
func (w *WAL) rotateFile() error {
	if err := w.flushWriter(); err != nil {
		return err
	}

	if err := w.updateFileHeader(); err != nil {
		return err
	}

	return w.createNewWriter()
}

// flushWriter flushes the current writer
func (w *WAL) flushWriter() error {
	if w.currentWriter != nil && w.currentWriter.writer != nil {
		if err := w.currentWriter.writer.Flush(); err != nil {
			return err
		}
	}
	return nil
}

// sync syncs the current file to disk
func (w *WAL) sync() error {
	if w.currentWriter != nil && w.currentWriter.file != nil {
		if err := w.flushWriter(); err != nil {
			return err
		}
		if err := w.currentWriter.file.Sync(); err != nil {
			return err
		}
		w.stats.lastFlushTime = time.Now()
	}
	return nil
}

// closeCurrentWriter closes the current writer
func (w *WAL) closeCurrentWriter() error {
	if w.currentWriter == nil {
		return nil
	}

	if err := w.flushWriter(); err != nil {
		return err
	}

	if err := w.updateFileHeader(); err != nil {
		return err
	}

	if err := w.currentWriter.file.Close(); err != nil {
		return err
	}

	w.currentWriter = nil
	return nil
}

// updateFileHeader updates the file header with final statistics
func (w *WAL) updateFileHeader() error {
	if w.currentWriter == nil {
		return nil
	}

	// We would need to seek to the beginning and update the header
	// For simplicity, we'll skip this in the initial implementation
	return nil
}

// calculateChecksum calculates a checksum for the record
func (w *WAL) calculateChecksum(record *WALRecord) uint32 {
	// Create a copy without checksum for calculation
	temp := *record
	temp.Checksum = 0

	data, err := json.Marshal(temp)
	if err != nil {
		return 0
	}

	return crc32.ChecksumIEEE(data)
}

// backgroundFlush runs the background flush routine
func (w *WAL) backgroundFlush() {
	for {
		select {
		case <-w.flushTicker.C:
			w.performBackgroundFlush()
		case <-w.flushChan:
			w.performBackgroundFlush()
		case <-w.stopChan:
			return
		}
	}
}

func (w *WAL) performBackgroundFlush() {
	// Use TryLock with timeout to avoid deadlocks
	if !w.mu.TryLock() {
		return // Skip flush if can't acquire lock immediately
	}
	defer w.mu.Unlock()
	_ = w.flushWriter()
}

// walFileInfo contains information about a WAL file
type walFileInfo struct {
	path     string
	firstLSN uint64
	lastLSN  uint64
}

// getWALFiles returns information about all WAL files
func (w *WAL) getWALFiles() ([]*walFileInfo, error) {
	files, err := filepath.Glob(filepath.Join(w.walDir, "wal-*.log"))
	if err != nil {
		return nil, err
	}

	var walFiles []*walFileInfo

	for _, file := range files {
		info, err := w.getFileInfo(file)
		if err != nil {
			continue // Skip corrupted files
		}
		walFiles = append(walFiles, info)
	}

	return walFiles, nil
}

// getFileInfo reads basic information from a WAL file
func (w *WAL) getFileInfo(filePath string) (*walFileInfo, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read header length
	var headerLen uint32
	if err := binary.Read(file, binary.LittleEndian, &headerLen); err != nil {
		return nil, err
	}

	// Read header
	headerData := make([]byte, headerLen)
	if _, err := io.ReadFull(file, headerData); err != nil {
		return nil, err
	}

	var header WALFileHeader
	if err := json.Unmarshal(headerData, &header); err != nil {
		return nil, err
	}

	// Scan through the file to find the actual last LSN
	// Note: file position is already past the header after reading headerData
	lastLSN := header.FirstLSN - 1 // Default to before first LSN if no records
	reader := bufio.NewReader(file)
	recordCount := 0

	for {
		// Read record length
		var recordLen uint32
		if err := binary.Read(reader, binary.LittleEndian, &recordLen); err != nil {
			if err == io.EOF {
				break
			}
			// If we can't read more records, that's ok - use what we found
			break
		}

		// Read record data
		recordData := make([]byte, recordLen)
		if _, err := io.ReadFull(reader, recordData); err != nil {
			break
		}

		// Deserialize record to get LSN
		var record WALRecord
		if err := json.Unmarshal(recordData, &record); err != nil {
			continue // Skip corrupted records
		}

		recordCount++
		if record.LSN > lastLSN {
			lastLSN = record.LSN
		}
	}

	// For debug purposes - this will be visible in test logs if we print the struct
	_ = recordCount

	return &walFileInfo{
		path:     filePath,
		firstLSN: header.FirstLSN,
		lastLSN:  lastLSN,
	}, nil
}

// replayFile replays records from a single WAL file
func (w *WAL) replayFile(ctx context.Context, filePath string, fromLSN uint64, handler ReplayHandler, recordCount *int64) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	// Create reader first
	reader := bufio.NewReader(file)

	// Skip header
	var headerLen uint32
	if err := binary.Read(reader, binary.LittleEndian, &headerLen); err != nil {
		return err
	}

	// Skip header data
	headerData := make([]byte, headerLen)
	if _, err := io.ReadFull(reader, headerData); err != nil {
		return err
	}

	for {
		// Check context
		if ctx.Err() != nil {
			return ctx.Err()
		}

		// Read record length
		var recordLen uint32
		if err := binary.Read(reader, binary.LittleEndian, &recordLen); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		// Read record data
		recordData := make([]byte, recordLen)
		if _, err := io.ReadFull(reader, recordData); err != nil {
			return err
		}

		// Deserialize record
		var record WALRecord
		if err := json.Unmarshal(recordData, &record); err != nil {
			return err
		}

		// Skip if before our target LSN
		if record.LSN < fromLSN {
			continue
		}

		// Verify checksum if configured
		if w.config.VerifyChecksums {
			expectedChecksum := w.calculateChecksum(&record)
			if record.Checksum != expectedChecksum {
				return fmt.Errorf("checksum mismatch for record LSN %d", record.LSN)
			}
		}

		// Handle the record
		if err := w.handleReplayRecord(ctx, &record, handler); err != nil {
			return fmt.Errorf("failed to handle record LSN %d: %w", record.LSN, err)
		}

		*recordCount++
	}

	return nil
}

// handleReplayRecord handles a single record during replay
func (w *WAL) handleReplayRecord(ctx context.Context, record *WALRecord, handler ReplayHandler) error {
	switch record.Type {
	case RecordTypeInsert:
		return handler.HandleInsert(ctx, record)
	case RecordTypeUpdate:
		return handler.HandleUpdate(ctx, record)
	case RecordTypeDelete:
		return handler.HandleDelete(ctx, record)
	case RecordTypeBatchInsert:
		return handler.HandleBatchInsert(ctx, record)
	case RecordTypeBatchDelete:
		return handler.HandleBatchDelete(ctx, record)
	case RecordTypeClear:
		return handler.HandleClear(ctx, record)
	case RecordTypeSnapshot:
		return handler.HandleSnapshot(ctx, record)
	case RecordTypeCheckpoint:
		// Checkpoints don't need special handling during replay
		return nil
	default:
		return fmt.Errorf("unknown record type: %v", record.Type)
	}
}
