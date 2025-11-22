package diskann

import (
	"os"
	"testing"
)

func TestDiskIndex_BuildAndSearch(t *testing.T) {
	filePath := "test_diskann.bin"
	defer os.Remove(filePath)

	config := &Config{
		FilePath:   filePath,
		Dimension:  4,
		MaxDegree:  3,
		SearchList: 10,
	}

	di, err := NewDiskIndex(config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer di.Close()

	// Create dummy vectors
	vectors := [][]float32{
		{1.0, 0.0, 0.0, 0.0}, // 0
		{0.0, 1.0, 0.0, 0.0}, // 1
		{0.0, 0.0, 1.0, 0.0}, // 2
		{0.0, 0.0, 0.0, 1.0}, // 3
		{1.0, 1.0, 0.0, 0.0}, // 4
	}

	if err := di.Build(vectors); err != nil {
		t.Fatalf("Failed to build index: %v", err)
	}

	// Search for vector close to {1, 0, 0, 0}
	query := []float32{0.9, 0.1, 0.0, 0.0}
	results, err := di.Search(query, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatalf("No results found")
	}

	// We expect node 0 or 4 to be closest
	found := false
	for _, id := range results {
		if id == 0 || id == 4 {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("Expected result 0 or 4, got %v", results)
	}
}
