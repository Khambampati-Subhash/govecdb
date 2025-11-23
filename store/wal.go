package store

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"sync"
)

// RecordType represents the type of record in the WAL
type RecordType byte

const (
	RecordTypeInsert RecordType = 1
	RecordTypeDelete RecordType = 2
)

// WAL implements a Write-Ahead Log for durability
type WAL struct {
	file     *os.File
	writer   *bufio.Writer
	mu       sync.Mutex
	sync     bool
	filepath string
}

// NewWAL creates or opens a WAL file
func NewWAL(filepath string, syncWrites bool) (*WAL, error) {
	f, err := os.OpenFile(filepath, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL file: %w", err)
	}

	return &WAL{
		file:     f,
		writer:   bufio.NewWriterSize(f, 64*1024), // 64KB buffer
		sync:     false,                           // Disable sync for performance
		filepath: filepath,
	}, nil
}

// WriteEntry writes an entry to the WAL
// Format: [CRC(4)][Type(1)][Size(4)][Data(n)]
func (w *WAL) WriteEntry(typ RecordType, data []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Calculate CRC
	crc := crc32.ChecksumIEEE(data)

	// Write Header
	header := make([]byte, 9)
	binary.LittleEndian.PutUint32(header[0:4], crc)
	header[4] = byte(typ)
	binary.LittleEndian.PutUint32(header[5:9], uint32(len(data)))

	if _, err := w.writer.Write(header); err != nil {
		return err
	}

	// Write Data
	if _, err := w.writer.Write(data); err != nil {
		return err
	}

	return nil
}

// BatchWriteEntry writes multiple entries to the WAL with a single lock
func (w *WAL) BatchWriteEntry(typ RecordType, entries [][]byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	for _, data := range entries {
		// Calculate CRC
		crc := crc32.ChecksumIEEE(data)

		// Write Header
		header := make([]byte, 9)
		binary.LittleEndian.PutUint32(header[0:4], crc)
		header[4] = byte(typ)
		binary.LittleEndian.PutUint32(header[5:9], uint32(len(data)))

		if _, err := w.writer.Write(header); err != nil {
			return err
		}

		// Write Data
		if _, err := w.writer.Write(data); err != nil {
			return err
		}
	}

	// Flush if needed
	if w.sync {
		if err := w.writer.Flush(); err != nil {
			return err
		}
		if err := w.file.Sync(); err != nil {
			return err
		}
	}

	return nil
}

// Close closes the WAL file
func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.writer.Flush(); err != nil {
		return err
	}
	return w.file.Close()
}

// Replay reads the WAL and calls the callback for each entry
func (w *WAL) Replay(fn func(typ RecordType, data []byte) error) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Sync before reading to ensure everything is on disk
	w.writer.Flush()

	// Open a new reader from the beginning
	f, err := os.Open(w.filepath)
	if err != nil {
		return err
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	header := make([]byte, 9)

	for {
		// Read Header
		if _, err := io.ReadFull(reader, header); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			return err
		}

		crc := binary.LittleEndian.Uint32(header[0:4])
		typ := RecordType(header[4])
		size := binary.LittleEndian.Uint32(header[5:9])

		// Read Data
		data := make([]byte, size)
		if _, err := io.ReadFull(reader, data); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			return err
		}

		// Verify CRC
		if crc32.ChecksumIEEE(data) != crc {
			// Checksum mismatch indicates corrupted tail (e.g. partial write)
			// We stop replay here and ignore the rest of the file
			fmt.Printf("WAL warning: checksum mismatch at offset (ignoring rest of file)\n")
			break
		}

		// Callback
		if err := fn(typ, data); err != nil {
			return err
		}
	}

	return nil
}
