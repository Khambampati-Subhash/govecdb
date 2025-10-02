package index

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// Benchmark helpers
func benchmarkConfig(dimension int) *Config {
	return &Config{
		Dimension:      dimension,
		Metric:         Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		Seed:           42,
		ThreadSafe:     false, // Disable for benchmarks to reduce overhead
	}
}

func generateBenchmarkVectors(count, dimension int, seed int64) []*Vector {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([]*Vector, count)

	for i := 0; i < count; i++ {
		data := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			data[j] = rng.Float32()*2 - 1
		}

		vectors[i] = &Vector{
			ID:   fmt.Sprintf("bench_vec_%d", i),
			Data: data,
			Metadata: map[string]interface{}{
				"index": i,
				"group": i % 10,
			},
		}
	}

	return vectors
}

// Distance function benchmarks
func BenchmarkCosineDistance(b *testing.B) {
	dimensions := []int{128, 384, 768, 1536}

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("dim_%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			a := generateRandomVector(dim, rng)
			bVec := generateRandomVector(dim, rng)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, _ = CosineDistance(a, bVec)
			}
		})
	}
}

func BenchmarkEuclideanDistance(b *testing.B) {
	dimensions := []int{128, 384, 768, 1536}

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("dim_%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			a := generateRandomVector(dim, rng)
			bVec := generateRandomVector(dim, rng)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, _ = EuclideanDistance(a, bVec)
			}
		})
	}
}

func BenchmarkEuclideanDistanceSquared(b *testing.B) {
	dimensions := []int{128, 384, 768, 1536}

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("dim_%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			a := generateRandomVector(dim, rng)
			bVec := generateRandomVector(dim, rng)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, _ = EuclideanDistanceSquared(a, bVec)
			}
		})
	}
}

func BenchmarkDotProductDistance(b *testing.B) {
	dimensions := []int{128, 384, 768, 1536}

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("dim_%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			a := generateRandomVector(dim, rng)
			bVec := generateRandomVector(dim, rng)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, _ = DotProductDistance(a, bVec)
			}
		})
	}
}

// Vector operation benchmarks
func BenchmarkVectorNormalize(b *testing.B) {
	dimensions := []int{128, 384, 768, 1536}

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("dim_%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				vector := generateRandomVector(dim, rng)
				_ = Normalize(vector)
			}
		})
	}
}

func BenchmarkVectorDotProduct(b *testing.B) {
	dimensions := []int{128, 384, 768, 1536}

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("dim_%d", dim), func(b *testing.B) {
			rng := rand.New(rand.NewSource(42))
			a := generateRandomVector(dim, rng)
			bVec := generateRandomVector(dim, rng)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, _ = VectorDotProduct(a, bVec)
			}
		})
	}
}

// SafeMap benchmarks
func BenchmarkSafeMapOperations(b *testing.B) {
	sm := NewSafeMap()
	vectors := generateBenchmarkVectors(1000, 128, 42)

	// Populate the map
	for _, v := range vectors {
		sm.Set(v.ID, v)
	}

	b.Run("Get", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			key := vectors[i%len(vectors)].ID
			_, _ = sm.Get(key)
		}
	})

	b.Run("Set", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("bench_key_%d", i)
			vector := &Vector{ID: key, Data: []float32{1.0, 2.0, 3.0}}
			sm.Set(key, vector)
		}
	})
}

// HNSW Index benchmarks
func BenchmarkHNSWIndexInsertion(b *testing.B) {
	testCases := []struct {
		dimension int
		count     int
	}{
		{128, 100},
		{128, 1000},
		{384, 100},
		{384, 1000},
		{768, 100},
		{1536, 100},
	}

	for _, tc := range testCases {
		b.Run(fmt.Sprintf("dim_%d_count_%d", tc.dimension, tc.count), func(b *testing.B) {
			config := benchmarkConfig(tc.dimension)
			vectors := generateBenchmarkVectors(tc.count, tc.dimension, 42)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				b.StopTimer()
				index, err := NewHNSWIndex(config)
				if err != nil {
					b.Fatalf("Failed to create index: %v", err)
				}
				b.StartTimer()

				for _, v := range vectors {
					err := index.Add(v)
					if err != nil {
						b.Fatalf("Failed to add vector: %v", err)
					}
				}

				b.StopTimer()
				index.Close()
			}
		})
	}
}

func BenchmarkHNSWIndexSearch(b *testing.B) {
	testCases := []struct {
		dimension int
		count     int
		k         int
	}{
		{128, 100, 10},
		{128, 1000, 10},
		{128, 1000, 100},
		{384, 100, 10},
		{384, 1000, 10},
		{768, 100, 10},
		{1536, 100, 10},
	}

	for _, tc := range testCases {
		b.Run(fmt.Sprintf("dim_%d_count_%d_k_%d", tc.dimension, tc.count, tc.k), func(b *testing.B) {
			// Setup
			config := benchmarkConfig(tc.dimension)
			index, err := NewHNSWIndex(config)
			if err != nil {
				b.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			vectors := generateBenchmarkVectors(tc.count, tc.dimension, 42)
			for _, v := range vectors {
				err := index.Add(v)
				if err != nil {
					b.Fatalf("Failed to add vector: %v", err)
				}
			}

			// Generate query vectors
			queryVectors := generateBenchmarkVectors(100, tc.dimension, 123)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				query := queryVectors[i%len(queryVectors)].Data
				_, err := index.Search(query, tc.k)
				if err != nil {
					b.Fatalf("Search failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkHNSWIndexSearchWithFilter(b *testing.B) {
	dimension := 128
	count := 10000
	k := 10

	config := benchmarkConfig(dimension)
	index, err := NewHNSWIndex(config)
	if err != nil {
		b.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors := generateBenchmarkVectors(count, dimension, 42)
	for _, v := range vectors {
		err := index.Add(v)
		if err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}

	// Filter function that accepts ~50% of vectors
	filter := func(metadata map[string]interface{}) bool {
		if group, exists := metadata["group"]; exists {
			if groupInt, ok := group.(int); ok {
				return groupInt < 5
			}
		}
		return false
	}

	queryVectors := generateBenchmarkVectors(100, dimension, 123)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		query := queryVectors[i%len(queryVectors)].Data
		_, err := index.SearchWithFilter(query, k, filter)
		if err != nil {
			b.Fatalf("Filtered search failed: %v", err)
		}
	}
}

func BenchmarkHNSWIndexBatchInsert(b *testing.B) {
	testCases := []struct {
		dimension int
		batchSize int
	}{
		{128, 100},
		{128, 1000},
		{384, 100},
		{384, 1000},
		{768, 100},
	}

	for _, tc := range testCases {
		b.Run(fmt.Sprintf("dim_%d_batch_%d", tc.dimension, tc.batchSize), func(b *testing.B) {
			config := benchmarkConfig(tc.dimension)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				b.StopTimer()
				index, err := NewHNSWIndex(config)
				if err != nil {
					b.Fatalf("Failed to create index: %v", err)
				}

				vectors := generateBenchmarkVectors(tc.batchSize, tc.dimension, int64(i))
				b.StartTimer()

				err = index.AddBatch(vectors)
				if err != nil {
					b.Fatalf("Batch insert failed: %v", err)
				}

				b.StopTimer()
				index.Close()
			}
		})
	}
}

// Memory usage benchmarks
func BenchmarkHNSWIndexMemoryUsage(b *testing.B) {
	testCases := []struct {
		dimension int
		count     int
	}{
		{128, 1000},
		{128, 10000},
		{384, 1000},
		{384, 10000},
		{768, 1000},
		{1536, 1000},
	}

	for _, tc := range testCases {
		b.Run(fmt.Sprintf("dim_%d_count_%d", tc.dimension, tc.count), func(b *testing.B) {
			config := benchmarkConfig(tc.dimension)
			vectors := generateBenchmarkVectors(tc.count, tc.dimension, 42)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				index, err := NewHNSWIndex(config)
				if err != nil {
					b.Fatalf("Failed to create index: %v", err)
				}

				for _, v := range vectors {
					err := index.Add(v)
					if err != nil {
						b.Fatalf("Failed to add vector: %v", err)
					}
				}

				// Perform some searches to exercise the index
				rng := rand.New(rand.NewSource(42))
				for j := 0; j < 10; j++ {
					query := generateRandomVector(tc.dimension, rng)
					_, err := index.Search(query, 10)
					if err != nil {
						b.Fatalf("Search failed: %v", err)
					}
				}

				index.Close()
			}
		})
	}
}

// Concurrent operation benchmarks
func BenchmarkHNSWIndexConcurrentSearch(b *testing.B) {
	dimension := 128
	count := 10000
	k := 10

	config := benchmarkConfig(dimension)
	config.ThreadSafe = true // Enable thread safety for concurrent benchmarks

	index, err := NewHNSWIndex(config)
	if err != nil {
		b.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors := generateBenchmarkVectors(count, dimension, 42)
	for _, v := range vectors {
		err := index.Add(v)
		if err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}

	queryVectors := generateBenchmarkVectors(100, dimension, 123)

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			query := queryVectors[i%len(queryVectors)].Data
			_, err := index.Search(query, k)
			if err != nil {
				b.Fatalf("Search failed: %v", err)
			}
			i++
		}
	})
}

func BenchmarkHNSWIndexConcurrentInsert(b *testing.B) {
	dimension := 128

	config := benchmarkConfig(dimension)
	config.ThreadSafe = true

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		index, err := NewHNSWIndex(config)
		if err != nil {
			b.Fatalf("Failed to create index: %v", err)
		}
		defer index.Close()

		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		i := 0

		for pb.Next() {
			vector := &Vector{
				ID:   fmt.Sprintf("vec_%d_%d", b.N, i),
				Data: generateRandomVector(dimension, rng),
			}

			err := index.Add(vector)
			if err != nil {
				b.Fatalf("Failed to add vector: %v", err)
			}
			i++
		}
	})
}

// Comparative benchmarks between different distance metrics
func BenchmarkDistanceMetricsComparison(b *testing.B) {
	dimension := 384

	metrics := []struct {
		name string
		fn   DistanceFunc
	}{
		{"Cosine", CosineDistance},
		{"Euclidean", EuclideanDistance},
		{"EuclideanSquared", EuclideanDistanceSquared},
		{"Manhattan", ManhattanDistance},
		{"DotProduct", DotProductDistance},
	}

	rng := rand.New(rand.NewSource(42))
	vecA := generateRandomVector(dimension, rng)
	vecB := generateRandomVector(dimension, rng)

	for _, metric := range metrics {
		b.Run(metric.name, func(b *testing.B) {
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_, _ = metric.fn(vecA, vecB)
			}
		})
	}
}

// Search accuracy vs performance benchmark
func BenchmarkSearchAccuracyVsPerformance(b *testing.B) {
	dimension := 128
	count := 10000

	efValues := []int{50, 100, 200, 400}

	for _, ef := range efValues {
		b.Run(fmt.Sprintf("ef_%d", ef), func(b *testing.B) {
			config := benchmarkConfig(dimension)
			config.EfConstruction = ef

			index, err := NewHNSWIndex(config)
			if err != nil {
				b.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			vectors := generateBenchmarkVectors(count, dimension, 42)
			for _, v := range vectors {
				err := index.Add(v)
				if err != nil {
					b.Fatalf("Failed to add vector: %v", err)
				}
			}

			queryVectors := generateBenchmarkVectors(100, dimension, 123)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				query := queryVectors[i%len(queryVectors)].Data
				_, err := index.Search(query, 10)
				if err != nil {
					b.Fatalf("Search failed: %v", err)
				}
			}
		})
	}
}
