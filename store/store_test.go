package store

import (
	"os"
	"testing"

	"github.com/khambampati-subhash/govecdb/index"
)

func TestStore_Persistence(t *testing.T) {
	walPath := "test_wal.log"
	defer os.Remove(walPath)

	// 1. Create Store and Insert
	cfg := index.DefaultConfig(4)
	idx, _ := index.NewHNSWIndex(cfg)
	s, err := NewStore(walPath, idx)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}

	vec := []float32{1.0, 2.0, 3.0, 4.0}
	if err := s.Insert("vec1", vec); err != nil {
		t.Fatalf("Failed to insert: %v", err)
	}

	if s.Index.Size() != 1 {
		t.Errorf("Expected size 1, got %d", s.Index.Size())
	}

	s.Close()

	// 2. Reopen Store and Verify Recovery
	idx2, _ := index.NewHNSWIndex(cfg)
	s2, err := NewStore(walPath, idx2)
	if err != nil {
		t.Fatalf("Failed to reopen store: %v", err)
	}
	defer s2.Close()

	if s2.Index.Size() != 1 {
		t.Errorf("Expected size 1 after recovery, got %d", s2.Index.Size())
	}

	v, err := s2.Index.Get("vec1")
	if err != nil {
		t.Fatalf("Failed to get vector: %v", err)
	}
	if v.Data[0] != 1.0 {
		t.Errorf("Data mismatch")
	}
}
