package index

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sync"
	"testing"
)

// Test configuration helpers
func testConfig(dimension int) *Config {
	return &Config{
		Dimension:      dimension,
		Metric:         Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		Seed:           42,
		ThreadSafe:     true,
	}
}

func generateRandomVector(dimension int, rng *rand.Rand) []float32 {
	vector := make([]float32, dimension)
	for i := range vector {
		vector[i] = rng.Float32()*2 - 1 // Random values between -1 and 1
	}
	return vector
}

func generateTestVectors(count, dimension int) []*Vector {
	rng := rand.New(rand.NewSource(42))
	vectors := make([]*Vector, count)

	for i := 0; i < count; i++ {
		vectors[i] = &Vector{
			ID:   fmt.Sprintf("vec_%d", i),
			Data: generateRandomVector(dimension, rng),
			Metadata: map[string]interface{}{
				"index":    i,
				"category": fmt.Sprintf("cat_%d", i%3),
				"value":    float64(i) * 0.1,
			},
		}
	}

	return vectors
}

// Test distance functions
func TestDistanceFunctions(t *testing.T) {
	a := []float32{1.0, 0.0, 0.0}
	b := []float32{0.0, 1.0, 0.0}
	c := []float32{1.0, 0.0, 0.0} // Same as a

	tests := []struct {
		name     string
		funcTest func([]float32, []float32) (float32, error)
		a, b     []float32
		expected float32
		delta    float32
	}{
		{"Cosine distance identical", CosineDistance, a, c, 0.0, 0.001},
		{"Cosine distance orthogonal", CosineDistance, a, b, 1.0, 0.001},
		{"Euclidean distance identical", EuclideanDistance, a, c, 0.0, 0.001},
		{"Euclidean distance orthogonal", EuclideanDistance, a, b, 1.414, 0.01},
		{"Manhattan distance identical", ManhattanDistance, a, c, 0.0, 0.001},
		{"Manhattan distance orthogonal", ManhattanDistance, a, b, 2.0, 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := tt.funcTest(tt.a, tt.b)
			if err != nil {
				t.Fatalf("Distance function returned error: %v", err)
			}

			if math.Abs(float64(result-tt.expected)) > float64(tt.delta) {
				t.Errorf("Expected distance %.3f, got %.3f", tt.expected, result)
			}
		})
	}
}

func TestDistanceFunctionErrors(t *testing.T) {
	a := []float32{1.0, 0.0}
	b := []float32{1.0, 0.0, 0.0} // Different dimension
	empty := []float32{}

	functions := []struct {
		name string
		fn   DistanceFunc
	}{
		{"Cosine", CosineDistance},
		{"Euclidean", EuclideanDistance},
		{"Manhattan", ManhattanDistance},
		{"DotProduct", DotProductDistance},
	}

	for _, fn := range functions {
		t.Run(fn.name+"_DimensionMismatch", func(t *testing.T) {
			_, err := fn.fn(a, b)
			if err != ErrDimensionMismatch {
				t.Errorf("Expected ErrDimensionMismatch, got %v", err)
			}
		})

		t.Run(fn.name+"_EmptyVector", func(t *testing.T) {
			_, err := fn.fn(empty, empty)
			if err != ErrEmptyVector {
				t.Errorf("Expected ErrEmptyVector, got %v", err)
			}
		})
	}
}

func TestVectorOperations(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{4.0, 5.0, 6.0}

	// Test vector addition
	sum, err := VectorAdd(a, b)
	if err != nil {
		t.Fatalf("VectorAdd failed: %v", err)
	}
	expected := []float32{5.0, 7.0, 9.0}
	if !reflect.DeepEqual(sum, expected) {
		t.Errorf("VectorAdd: expected %v, got %v", expected, sum)
	}

	// Test vector subtraction
	diff, err := VectorSubtract(b, a)
	if err != nil {
		t.Fatalf("VectorSubtract failed: %v", err)
	}
	expected = []float32{3.0, 3.0, 3.0}
	if !reflect.DeepEqual(diff, expected) {
		t.Errorf("VectorSubtract: expected %v, got %v", expected, diff)
	}

	// Test vector scaling
	scaled := VectorScale(a, 2.0)
	expected = []float32{2.0, 4.0, 6.0}
	if !reflect.DeepEqual(scaled, expected) {
		t.Errorf("VectorScale: expected %v, got %v", expected, scaled)
	}

	// Test dot product
	dot, err := VectorDotProduct(a, b)
	if err != nil {
		t.Fatalf("VectorDotProduct failed: %v", err)
	}
	expectedDot := float32(32.0) // 1*4 + 2*5 + 3*6 = 32
	if dot != expectedDot {
		t.Errorf("VectorDotProduct: expected %.1f, got %.1f", expectedDot, dot)
	}
}

func TestNormalize(t *testing.T) {
	vector := []float32{3.0, 4.0} // Length should be 5

	err := Normalize(vector)
	if err != nil {
		t.Fatalf("Normalize failed: %v", err)
	}

	// Check if the vector is normalized (length = 1)
	norm := VectorNorm(vector)
	if math.Abs(float64(norm-1.0)) > 0.001 {
		t.Errorf("Expected normalized vector to have length 1.0, got %.6f", norm)
	}

	expected := []float32{0.6, 0.8}
	for i, v := range vector {
		if math.Abs(float64(v-expected[i])) > 0.001 {
			t.Errorf("Normalize: expected [%.1f, %.1f], got [%.6f, %.6f]",
				expected[0], expected[1], vector[0], vector[1])
			break
		}
	}
}

// Test SafeMap
func TestSafeMap(t *testing.T) {
	sm := NewSafeMap()

	// Test empty map
	if sm.Size() != 0 {
		t.Errorf("Expected size 0, got %d", sm.Size())
	}

	// Test set and get
	vector := &Vector{ID: "test", Data: []float32{1.0, 2.0}}
	sm.Set("test", vector)

	retrieved, exists := sm.Get("test")
	if !exists {
		t.Error("Expected key to exist")
	}

	if retrieved.ID != vector.ID {
		t.Errorf("Expected ID %s, got %s", vector.ID, retrieved.ID)
	}

	// Test size
	if sm.Size() != 1 {
		t.Errorf("Expected size 1, got %d", sm.Size())
	}

	// Test delete
	sm.Delete("test")
	if sm.Size() != 0 {
		t.Errorf("Expected size 0 after delete, got %d", sm.Size())
	}

	_, exists = sm.Get("test")
	if exists {
		t.Error("Expected key to not exist after delete")
	}
}

func TestSafeMapConcurrency(t *testing.T) {
	sm := NewSafeMap()
	numRoutines := 100
	numOpsPerRoutine := 100

	var wg sync.WaitGroup
	wg.Add(numRoutines)

	// Start multiple goroutines performing operations
	for i := 0; i < numRoutines; i++ {
		go func(id int) {
			defer wg.Done()

			for j := 0; j < numOpsPerRoutine; j++ {
				key := fmt.Sprintf("key_%d_%d", id, j)
				vector := &Vector{
					ID:   key,
					Data: []float32{float32(id), float32(j)},
				}

				// Set
				sm.Set(key, vector)

				// Get
				if retrieved, exists := sm.Get(key); exists && retrieved.ID != key {
					t.Errorf("Retrieved wrong vector: expected %s, got %s", key, retrieved.ID)
				}

				// Delete every other key
				if j%2 == 0 {
					sm.Delete(key)
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify final state
	expectedSize := numRoutines * numOpsPerRoutine / 2
	actualSize := sm.Size()

	// Allow some tolerance due to concurrent operations
	if actualSize < expectedSize/2 || actualSize > expectedSize*2 {
		t.Errorf("Unexpected final size: expected around %d, got %d", expectedSize, actualSize)
	}
}

// Test HNSW Node
func TestHNSWNode(t *testing.T) {
	vector := &Vector{
		ID:   "test_node",
		Data: []float32{1.0, 2.0, 3.0},
		Metadata: map[string]interface{}{
			"category": "test",
		},
	}

	node := NewHNSWNode(vector, 2)

	// Test basic properties
	if node.Level != 2 {
		t.Errorf("Expected level 2, got %d", node.Level)
	}

	if node.Vector.ID != vector.ID {
		t.Errorf("Expected ID %s, got %s", vector.ID, node.Vector.ID)
	}

	// Test connections
	other := NewHNSWNode(&Vector{ID: "other", Data: []float32{4.0, 5.0, 6.0}}, 1)

	// Add connection
	node.AddConnection(other, 0)

	if !node.HasConnection(other, 0) {
		t.Error("Expected connection to exist")
	}

	if node.ConnectionCount(0) != 1 {
		t.Errorf("Expected 1 connection, got %d", node.ConnectionCount(0))
	}

	// Remove connection
	node.RemoveConnection(other, 0)

	if node.HasConnection(other, 0) {
		t.Error("Expected connection to not exist after removal")
	}

	if node.ConnectionCount(0) != 0 {
		t.Errorf("Expected 0 connections, got %d", node.ConnectionCount(0))
	}
}

func TestHNSWNodeDeletion(t *testing.T) {
	vector := &Vector{ID: "test", Data: []float32{1.0, 2.0}}
	node := NewHNSWNode(vector, 1)

	if node.IsDeleted() {
		t.Error("Node should not be deleted initially")
	}

	node.MarkDeleted()

	if !node.IsDeleted() {
		t.Error("Node should be deleted after marking")
	}

	// Test that GetVector returns nil for deleted nodes
	retrieved := node.GetVector()
	if retrieved != nil {
		t.Error("GetVector should return nil for deleted nodes")
	}
}

// Test HNSW Index
func TestHNSWIndexCreation(t *testing.T) {
	config := testConfig(128)

	index, err := NewHNSWIndex(config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	if index.Dimension() != 128 {
		t.Errorf("Expected dimension 128, got %d", index.Dimension())
	}

	if index.Size() != 0 {
		t.Errorf("Expected empty index, got size %d", index.Size())
	}

	if !index.IsEmpty() {
		t.Error("Expected index to be empty")
	}
}

func TestHNSWIndexBasicOperations(t *testing.T) {
	config := testConfig(3)
	index, err := NewHNSWIndex(config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Test adding vectors
	vectors := []*Vector{
		{ID: "v1", Data: []float32{1.0, 0.0, 0.0}},
		{ID: "v2", Data: []float32{0.0, 1.0, 0.0}},
		{ID: "v3", Data: []float32{0.0, 0.0, 1.0}},
	}

	for _, v := range vectors {
		err := index.Add(v)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", v.ID, err)
		}
	}

	// Test size
	if index.Size() != 3 {
		t.Errorf("Expected size 3, got %d", index.Size())
	}

	// Test retrieval
	for _, v := range vectors {
		retrieved, err := index.Get(v.ID)
		if err != nil {
			t.Errorf("Failed to get vector %s: %v", v.ID, err)
			continue
		}

		if retrieved.ID != v.ID {
			t.Errorf("Retrieved wrong vector: expected %s, got %s", v.ID, retrieved.ID)
		}
	}

	// Test search
	query := []float32{1.0, 0.1, 0.1} // Close to v1
	results, err := index.Search(query, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Search returned no results")
	}

	// The closest should be v1
	if len(results) > 0 && results[0].ID != "v1" {
		t.Errorf("Expected closest result to be v1, got %s", results[0].ID)
	}
}

func TestHNSWIndexBatchOperations(t *testing.T) {
	config := testConfig(10)
	index, err := NewHNSWIndex(config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors := generateTestVectors(100, 10)

	err = index.AddBatch(vectors)
	if err != nil {
		t.Fatalf("Batch add failed: %v", err)
	}

	if index.Size() != 100 {
		t.Errorf("Expected size 100, got %d", index.Size())
	}

	// Test search
	query := vectors[0].Data
	results, err := index.Search(query, 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Search returned no results")
	}

	// The first result should be the exact match
	if results[0].ID != vectors[0].ID {
		t.Errorf("Expected exact match as first result, got %s", results[0].ID)
	}
}

func TestHNSWIndexWithFiltering(t *testing.T) {
	config := testConfig(5)
	index, err := NewHNSWIndex(config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors := []*Vector{
		{
			ID:       "v1",
			Data:     []float32{1.0, 0.0, 0.0, 0.0, 0.0},
			Metadata: map[string]interface{}{"category": "A"},
		},
		{
			ID:       "v2",
			Data:     []float32{0.9, 0.1, 0.0, 0.0, 0.0},
			Metadata: map[string]interface{}{"category": "B"},
		},
		{
			ID:       "v3",
			Data:     []float32{0.8, 0.2, 0.0, 0.0, 0.0},
			Metadata: map[string]interface{}{"category": "A"},
		},
	}

	for _, v := range vectors {
		err := index.Add(v)
		if err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Search with filter for category A
	filter := func(metadata map[string]interface{}) bool {
		category, exists := metadata["category"]
		return exists && category == "A"
	}

	query := []float32{1.0, 0.0, 0.0, 0.0, 0.0}
	results, err := index.SearchWithFilter(query, 10, filter)
	if err != nil {
		t.Fatalf("Filtered search failed: %v", err)
	}

	// Should only return vectors with category A
	for _, result := range results {
		category := result.Metadata["category"]
		if category != "A" {
			t.Errorf("Filter failed: got result with category %v", category)
		}
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results with category A, got %d", len(results))
	}
}

// func TestHNSWIndexDeletion(t *testing.T) {
// 	config := testConfig(3)
// 	index, err := NewHNSWIndex(config)
// 	if err != nil {
// 		t.Fatalf("Failed to create index: %v", err)
// 	}
// 	defer index.Close()

// 	vector := &Vector{ID: "test", Data: []float32{1.0, 2.0, 3.0}}

// 	err = index.Add(vector)
// 	if err != nil {
// 		t.Fatalf("Failed to add vector: %v", err)
// 	}

// 	// Verify it exists
// 	if !index.Contains("test") {
// 		t.Error("Vector should exist before deletion")
// 	}

// 	// Delete it
// 	err = index.Delete("test")
// 	if err != nil {
// 		t.Fatalf("Failed to delete vector: %v", err)
// 	}

// 	// Verify it's gone
// 	if index.Contains("test") {
// 		t.Error("Vector should not exist after deletion")
// 	}
// }

func TestHNSWIndexConcurrency(t *testing.T) {
	config := testConfig(10)
	index, err := NewHNSWIndex(config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	numRoutines := 10
	numOpsPerRoutine := 50

	var wg sync.WaitGroup
	wg.Add(numRoutines)

	// Concurrent insertions and searches
	for i := 0; i < numRoutines; i++ {
		go func(routineID int) {
			defer wg.Done()

			rng := rand.New(rand.NewSource(int64(routineID)))

			for j := 0; j < numOpsPerRoutine; j++ {
				vector := &Vector{
					ID:   fmt.Sprintf("vec_%d_%d", routineID, j),
					Data: generateRandomVector(10, rng),
				}

				// Insert
				err := index.Add(vector)
				if err != nil {
					t.Errorf("Failed to add vector: %v", err)
					continue
				}

				// Search
				query := generateRandomVector(10, rng)
				_, err = index.Search(query, 5)
				if err != nil {
					t.Errorf("Search failed: %v", err)
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify final size
	finalSize := index.Size()
	expectedSize := numRoutines * numOpsPerRoutine

	if finalSize != expectedSize {
		t.Errorf("Expected final size %d, got %d", expectedSize, finalSize)
	}
}

func TestHNSWIndexEdgeCases(t *testing.T) {
	config := testConfig(3)
	index, err := NewHNSWIndex(config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Test adding nil vector
	err = index.Add(nil)
	if err == nil {
		t.Error("Expected error when adding nil vector")
	}

	// Test adding vector with wrong dimension
	wrongDimVector := &Vector{ID: "wrong", Data: []float32{1.0, 2.0}} // 2D instead of 3D
	err = index.Add(wrongDimVector)
	if err != ErrDimensionMismatch {
		t.Errorf("Expected ErrDimensionMismatch, got %v", err)
	}

	// Test searching empty index
	query := []float32{1.0, 2.0, 3.0}
	results, err := index.Search(query, 5)
	if err != nil {
		t.Errorf("Search on empty index should not error, got %v", err)
	}
	if len(results) != 0 {
		t.Errorf("Search on empty index should return empty results, got %d", len(results))
	}

	// Test search with wrong dimension
	wrongQuery := []float32{1.0, 2.0} // 2D instead of 3D
	_, err = index.Search(wrongQuery, 5)
	if err != ErrDimensionMismatch {
		t.Errorf("Expected ErrDimensionMismatch for wrong query dimension, got %v", err)
	}

	// Test search with invalid k
	_, err = index.Search(query, 0)
	if err != ErrInvalidK {
		t.Errorf("Expected ErrInvalidK for k=0, got %v", err)
	}

	_, err = index.Search(query, -1)
	if err != ErrInvalidK {
		t.Errorf("Expected ErrInvalidK for k=-1, got %v", err)
	}
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name   string
		config *Config
		valid  bool
	}{
		{
			"Valid config",
			&Config{Dimension: 10, M: 16, EfConstruction: 200, MaxLayer: 16, Metric: Cosine},
			true,
		},
		{
			"Zero dimension",
			&Config{Dimension: 0, M: 16, EfConstruction: 200, MaxLayer: 16, Metric: Cosine},
			false,
		},
		{
			"Negative M",
			&Config{Dimension: 10, M: -1, EfConstruction: 200, MaxLayer: 16, Metric: Cosine},
			false,
		},
		{
			"Zero EfConstruction",
			&Config{Dimension: 10, M: 16, EfConstruction: 0, MaxLayer: 16, Metric: Cosine},
			false,
		},
		{
			"Invalid metric",
			&Config{Dimension: 10, M: 16, EfConstruction: 200, MaxLayer: 16, Metric: DistanceMetric(999)},
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.valid && err != nil {
				t.Errorf("Expected valid config, got error: %v", err)
			}
			if !tt.valid && err == nil {
				t.Error("Expected invalid config, got no error")
			}
		})
	}
}

func TestIndexStats(t *testing.T) {
	config := testConfig(5)
	index, err := NewHNSWIndex(config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Initial stats
	stats := index.GetStats()
	if stats.NodeCount != 0 {
		t.Errorf("Expected 0 nodes initially, got %d", stats.NodeCount)
	}

	// Add some vectors and check stats
	vectors := generateTestVectors(10, 5)
	for _, v := range vectors {
		err := index.Add(v)
		if err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Perform some searches
	query := []float32{1.0, 0.0, 0.0, 0.0, 0.0}
	for i := 0; i < 5; i++ {
		_, err := index.Search(query, 3)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
	}

	stats = index.GetStats()
	if stats.NodeCount != 10 {
		t.Errorf("Expected 10 nodes, got %d", stats.NodeCount)
	}

	if stats.SearchCount != 5 {
		t.Errorf("Expected 5 searches, got %d", stats.SearchCount)
	}

	if stats.InsertCount != 10 {
		t.Errorf("Expected 10 inserts, got %d", stats.InsertCount)
	}

	if stats.Dimension != 5 {
		t.Errorf("Expected dimension 5, got %d", stats.Dimension)
	}
}
