package store

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/khambampati-subhash/govecdb/index"
)

// Store manages the storage of vectors, including WAL and in-memory index
type Store struct {
	wal   *WAL
	Index *index.HNSWIndex // Exported for Node access
	mu    sync.RWMutex
}

// VectorData represents the data stored in the WAL
type VectorData struct {
	ID     string    `json:"id"`
	Vector []float32 `json:"vector"`
}

// NewStore creates a new store
func NewStore(walPath string, idx *index.HNSWIndex) (*Store, error) {
	wal, err := NewWAL(walPath, false) // Async writes for performance
	if err != nil {
		return nil, err
	}

	s := &Store{
		wal:   wal,
		Index: idx,
	}

	// Replay WAL to restore index
	if err := s.replay(); err != nil {
		return nil, fmt.Errorf("failed to replay WAL: %w", err)
	}

	return s, nil
}

// Insert adds a vector to the store
func (s *Store) Insert(id string, vector []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 1. Write to WAL
	data := VectorData{ID: id, Vector: vector}
	bytes, err := json.Marshal(data)
	if err != nil {
		return err
	}

	if err := s.wal.WriteEntry(RecordTypeInsert, bytes); err != nil {
		return err
	}

	// 2. Update Index
	vec := &index.Vector{ID: id, Data: vector}
	return s.Index.Add(vec)
}

// Delete removes a vector from the store
func (s *Store) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 1. Write to WAL
	if err := s.wal.WriteEntry(RecordTypeDelete, []byte(id)); err != nil {
		return err
	}

	// 2. Update Index
	return s.Index.Delete(id)
}

// Close closes the store
func (s *Store) Close() error {
	return s.wal.Close()
}

// replay restores the index from the WAL
func (s *Store) replay() error {
	return s.wal.Replay(func(typ RecordType, data []byte) error {
		switch typ {
		case RecordTypeInsert:
			var v VectorData
			if err := json.Unmarshal(data, &v); err != nil {
				return err
			}
			vec := &index.Vector{ID: v.ID, Data: v.Vector}
			return s.Index.Add(vec)
		case RecordTypeDelete:
			id := string(data)
			return s.Index.Delete(id)
		}
		return nil
	})
}
