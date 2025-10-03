package utils

import (
	"testing"
	"time"
)

// TestFloat32Pool tests the float32 pool functionality
func TestFloat32Pool(t *testing.T) {
	pool := NewFloat32Pool(100)

	// Get a slice
	slice1 := pool.Get()
	if cap(slice1) < 100 {
		t.Errorf("Expected capacity >= 100, got %d", cap(slice1))
	}

	// Use the slice
	slice1 = append(slice1, 1.0, 2.0, 3.0)
	if len(slice1) != 3 {
		t.Errorf("Expected length 3, got %d", len(slice1))
	}

	// Return to pool
	pool.Put(slice1)

	// Get another slice (should be reused)
	slice2 := pool.Get()
	if len(slice2) != 0 {
		t.Errorf("Expected empty slice from pool, got length %d", len(slice2))
	}
}

// TestBufferPool tests the buffer pool functionality
func TestBufferPool(t *testing.T) {
	pool := NewBufferPool(1024)

	buf1 := pool.Get()
	if cap(buf1) < 1024 {
		t.Errorf("Expected capacity >= 1024, got %d", cap(buf1))
	}

	buf1 = append(buf1, []byte("test data")...)
	pool.Put(buf1)

	buf2 := pool.Get()
	if len(buf2) != 0 {
		t.Errorf("Expected empty buffer from pool, got length %d", len(buf2))
	}
}

// TestDistanceFunctions tests distance calculation functions
func TestDistanceFunctions(t *testing.T) {
	vec1 := []float32{1.0, 2.0, 3.0}
	vec2 := []float32{4.0, 5.0, 6.0}

	// Test Euclidean distance
	euclidean := EuclideanDistanceOptimized(vec1, vec2)
	if euclidean <= 0 {
		t.Error("Euclidean distance should be positive")
	}

	// Test Cosine distance
	cosine := CosineDistanceOptimized(vec1, vec2)
	if cosine < 0 || cosine > 2 {
		t.Errorf("Cosine distance should be between 0 and 2, got %f", cosine)
	}

	// Test Dot Product
	dotProduct := DotProductOptimized(vec1, vec2)
	if dotProduct >= 0 {
		t.Error("Dot product distance should be negative (inverted)")
	}
}

// TestWorkerPool tests worker pool functionality
func TestWorkerPool(t *testing.T) {
	wp := NewWorkerPool(2)
	defer wp.Close()

	taskCount := 10
	results := make(chan int, taskCount)

	for i := 0; i < taskCount; i++ {
		taskID := i
		wp.Submit(func() {
			results <- taskID
		})
	}

	// Collect results
	completed := make(map[int]bool)
	for i := 0; i < taskCount; i++ {
		select {
		case result := <-results:
			completed[result] = true
		case <-time.After(5 * time.Second):
			t.Fatal("Tasks did not complete within timeout")
		}
	}

	if len(completed) != taskCount {
		t.Errorf("Expected %d completed tasks, got %d", taskCount, len(completed))
	}
}

// TestCache tests LRU cache functionality
func TestCache(t *testing.T) {
	cache := NewCache(3)

	// Test Put and Get
	cache.Put("key1", "value1")
	cache.Put("key2", "value2")
	cache.Put("key3", "value3")

	if val, found := cache.Get("key1"); !found || val != "value1" {
		t.Error("Failed to get cached value")
	}

	// Test eviction
	cache.Put("key4", "value4") // Should evict key2 (least recently used)

	if _, found := cache.Get("key2"); found {
		t.Error("LRU item should have been evicted")
	}

	if val, found := cache.Get("key1"); !found || val != "value1" {
		t.Error("Recently accessed item should still be in cache")
	}
}

// TestBatchDistanceCalculator tests batch distance calculations
func TestBatchDistanceCalculator(t *testing.T) {
	calc := NewBatchDistanceCalculator(EuclideanDistance, 2)

	query := []float32{1.0, 2.0, 3.0}
	vectors := [][]float32{
		{4.0, 5.0, 6.0},
		{1.0, 1.0, 1.0},
		{7.0, 8.0, 9.0},
	}

	distances := calc.CalculateDistances(query, vectors)
	if len(distances) != len(vectors) {
		t.Errorf("Expected %d distances, got %d", len(vectors), len(distances))
	}

	// All distances should be positive
	for i, distance := range distances {
		if distance <= 0 {
			t.Errorf("Distance %d should be positive, got %f", i, distance)
		}
	}
}

// TestVectorQuantizer tests vector quantization
func TestVectorQuantizer(t *testing.T) {
	quantizer := NewVectorQuantizer(8) // 8-bit quantization

	// Test vector with values in [0, 1] range
	vector := []float32{0.0, 0.5, 1.0, 0.25, 0.75}

	quantized := quantizer.Quantize(vector)
	if len(quantized) != len(vector) {
		t.Errorf("Expected %d quantized values, got %d", len(vector), len(quantized))
	}

	// Dequantize and check
	dequantized := quantizer.Dequantize(quantized)
	if len(dequantized) != len(vector) {
		t.Errorf("Expected %d dequantized values, got %d", len(vector), len(dequantized))
	}

	// Check that values are approximately preserved
	for i, original := range vector {
		recovered := dequantized[i]
		diff := original - recovered
		if diff < 0 {
			diff = -diff
		}
		if diff > 0.1 { // Allow some quantization error
			t.Errorf("Quantization error too large for index %d: original=%f, recovered=%f",
				i, original, recovered)
		}
	}
}

// BenchmarkEuclideanDistance benchmarks Euclidean distance calculation
func BenchmarkEuclideanDistance(b *testing.B) {
	vec1 := make([]float32, 256)
	vec2 := make([]float32, 256)

	for i := 0; i < 256; i++ {
		vec1[i] = float32(i) / 256.0
		vec2[i] = float32(i+1) / 256.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EuclideanDistanceOptimized(vec1, vec2)
	}
}

// BenchmarkFloat32Pool benchmarks pool performance vs direct allocation
func BenchmarkFloat32Pool(b *testing.B) {
	pool := NewFloat32Pool(1024)

	b.Run("Pooled", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			slice := pool.Get()
			for j := 0; j < 100; j++ {
				slice = append(slice, float32(j))
			}
			pool.Put(slice)
		}
	})

	b.Run("Direct", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			slice := make([]float32, 0, 1024)
			for j := 0; j < 100; j++ {
				slice = append(slice, float32(j))
			}
			_ = slice // Use the slice to avoid SA4010
		}
	})
}
