// Package bench provides simple performance benchmarks for GoVecDB Phase 5 utilities.
package bench

import (
	"fmt"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/utils"
)

// SimpleBenchmarkSuite provides basic benchmarking for Phase 5 features
type SimpleBenchmarkSuite struct {
	VectorPools        map[int]*utils.VectorPool
	DistanceCalculator *utils.BatchDistanceCalculator
	Cache              *utils.Cache
}

// NewSimpleBenchmarkSuite creates a new simple benchmark suite
func NewSimpleBenchmarkSuite() *SimpleBenchmarkSuite {
	return &SimpleBenchmarkSuite{
		VectorPools:        make(map[int]*utils.VectorPool),
		DistanceCalculator: utils.NewBatchDistanceCalculator(utils.EuclideanDistance, 4),
		Cache:              utils.NewCache(1000),
	}
}

// GetVectorPool gets or creates a vector pool for the given dimension
func (sbs *SimpleBenchmarkSuite) GetVectorPool(dimension int) *utils.VectorPool {
	if pool, exists := sbs.VectorPools[dimension]; exists {
		return pool
	}

	pool := utils.NewVectorPool(dimension)
	sbs.VectorPools[dimension] = pool
	return pool
}

// BenchmarkDistanceCalculations benchmarks different distance calculations
func BenchmarkDistanceCalculations(b *testing.B) {
	dimensions := []int{128, 256, 512, 1024}

	for _, dim := range dimensions {
		// Generate test vectors
		vec1 := make([]float32, dim)
		vec2 := make([]float32, dim)
		for i := 0; i < dim; i++ {
			vec1[i] = float32(i) / float32(dim)
			vec2[i] = float32(i+1) / float32(dim)
		}

		// Benchmark different distance functions
		b.Run(fmt.Sprintf("Euclidean_%dd", dim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = utils.EuclideanDistanceOptimized(vec1, vec2)
			}
		})

		b.Run(fmt.Sprintf("Cosine_%dd", dim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = utils.CosineDistanceOptimized(vec1, vec2)
			}
		})

		b.Run(fmt.Sprintf("DotProduct_%dd", dim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = utils.DotProductOptimized(vec1, vec2)
			}
		})

		b.Run(fmt.Sprintf("Manhattan_%dd", dim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = utils.ManhattanDistanceOptimized(vec1, vec2)
			}
		})
	}
}

// BenchmarkBatchDistanceCalculation benchmarks batch distance calculations
func BenchmarkBatchDistanceCalculation(b *testing.B) {
	dimension := 256
	batchSizes := []int{10, 100, 1000}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("Batch_%d_vectors", batchSize), func(b *testing.B) {
			// Setup test data
			query := make([]float32, dimension)
			vectors := make([][]float32, batchSize)

			for i := 0; i < dimension; i++ {
				query[i] = float32(i) / float32(dimension)
			}

			for i := 0; i < batchSize; i++ {
				vectors[i] = make([]float32, dimension)
				for j := 0; j < dimension; j++ {
					vectors[i][j] = float32(i+j) / float32(dimension)
				}
			}

			calc := utils.NewBatchDistanceCalculator(utils.EuclideanDistance, 4)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = calc.CalculateDistances(query, vectors)
			}
		})
	}
}

// BenchmarkMemoryPooling benchmarks memory pool performance vs direct allocation
func BenchmarkMemoryPooling(b *testing.B) {
	poolSizes := []int{128, 512, 1024, 4096}

	for _, poolSize := range poolSizes {
		b.Run(fmt.Sprintf("Float32Pool_%d", poolSize), func(b *testing.B) {
			pool := utils.NewFloat32Pool(poolSize)

			b.Run("Pooled", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					slice := pool.Get()
					// Simulate usage
					for j := 0; j < 100 && j < cap(slice); j++ {
						slice = append(slice, float32(j))
					}
					pool.Put(slice)
				}
			})

			b.Run("Direct", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					slice := make([]float32, 0, poolSize)
					// Simulate usage
					for j := 0; j < 100; j++ {
						slice = append(slice, float32(j))
					}
					// No pooling, let GC handle it
				}
			})
		})
	}
}

// BenchmarkBufferPooling benchmarks buffer pool performance
func BenchmarkBufferPooling(b *testing.B) {
	bufferSizes := []int{1024, 4096, 16384}

	for _, bufferSize := range bufferSizes {
		b.Run(fmt.Sprintf("BufferPool_%d", bufferSize), func(b *testing.B) {
			pool := utils.NewBufferPool(bufferSize)

			b.Run("Pooled", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					buf := pool.Get()
					// Simulate usage
					data := []byte("test data for buffer pool benchmarking")
					for len(buf)+len(data) < cap(buf) {
						buf = append(buf, data...)
					}
					pool.Put(buf)
				}
			})

			b.Run("Direct", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					buf := make([]byte, 0, bufferSize)
					// Simulate usage
					data := []byte("test data for buffer pool benchmarking")
					for len(buf)+len(data) < cap(buf) {
						buf = append(buf, data...)
					}
					// No pooling, let GC handle it
				}
			})
		})
	}
}

// BenchmarkWorkerPool benchmarks worker pool performance
func BenchmarkWorkerPool(b *testing.B) {
	workerCounts := []int{1, 2, 4, 8, 16}

	for _, workerCount := range workerCounts {
		b.Run(fmt.Sprintf("Workers_%d", workerCount), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				wp := utils.NewWorkerPool(workerCount)

				taskCount := 100
				done := make(chan bool, taskCount)

				// Submit tasks
				for j := 0; j < taskCount; j++ {
					wp.Submit(func() {
						// Simulate some work
						sum := 0
						for k := 0; k < 1000; k++ {
							sum += k
						}
						done <- true
					})
				}

				// Wait for completion
				for j := 0; j < taskCount; j++ {
					<-done
				}

				wp.Close()
			}
		})
	}
}

// BenchmarkLRUCache benchmarks LRU cache performance
func BenchmarkLRUCache(b *testing.B) {
	cacheSizes := []int{100, 500, 1000, 5000}

	for _, cacheSize := range cacheSizes {
		b.Run(fmt.Sprintf("LRUCache_%d", cacheSize), func(b *testing.B) {
			cache := utils.NewCache(cacheSize)

			// Pre-populate cache
			for i := 0; i < cacheSize/2; i++ {
				cache.Put(fmt.Sprintf("key_%d", i), fmt.Sprintf("value_%d", i))
			}

			b.Run("Get", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					key := fmt.Sprintf("key_%d", i%(cacheSize/2))
					_, _ = cache.Get(key)
				}
			})

			b.Run("Put", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					key := fmt.Sprintf("new_key_%d", i)
					value := fmt.Sprintf("new_value_%d", i)
					cache.Put(key, value)
				}
			})
		})
	}
}

// BenchmarkVectorNormalization benchmarks vector normalization
func BenchmarkVectorNormalization(b *testing.B) {
	dimensions := []int{128, 256, 512, 1024}
	normalizer := utils.NewVectorNormalizer()

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("L2Normalize_%dd", dim), func(b *testing.B) {
			// Generate test vector
			vector := make([]float32, dim)
			for i := 0; i < dim; i++ {
				vector[i] = float32(i) / float32(dim)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				normalized := normalizer.L2Normalize(vector)
				normalizer.Release(normalized) // Return to pool
			}
		})
	}
}

// BenchmarkVectorQuantization benchmarks vector quantization
func BenchmarkVectorQuantization(b *testing.B) {
	dimensions := []int{128, 256, 512, 1024}
	bitLevels := []int{4, 8, 16}

	for _, dim := range dimensions {
		for _, bits := range bitLevels {
			b.Run(fmt.Sprintf("Quantize_%dd_%dbit", dim, bits), func(b *testing.B) {
				quantizer := utils.NewVectorQuantizer(bits)

				// Generate test vector with values in [0, 1] range
				vector := make([]float32, dim)
				for i := 0; i < dim; i++ {
					vector[i] = float32(i%256) / 256.0 // Ensure [0, 1] range
				}

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					quantized := quantizer.Quantize(vector)
					_ = quantizer.Dequantize(quantized) // Also test dequantization
				}
			})
		}
	}
}

// TestPerformanceRegression runs basic performance regression tests
func TestPerformanceRegression(t *testing.T) {
	// Test distance calculation performance
	vec1 := make([]float32, 256)
	vec2 := make([]float32, 256)
	for i := 0; i < 256; i++ {
		vec1[i] = float32(i) / 256.0
		vec2[i] = float32(i+1) / 256.0
	}

	start := time.Now()
	iterations := 10000
	for i := 0; i < iterations; i++ {
		_ = utils.EuclideanDistanceOptimized(vec1, vec2)
	}
	duration := time.Since(start)

	avgDuration := duration / time.Duration(iterations)
	t.Logf("Euclidean distance (256d): %d iterations in %v (avg: %v)",
		iterations, duration, avgDuration)

	// Should be faster than 1 microsecond per calculation
	if avgDuration > time.Microsecond {
		t.Logf("Warning: Distance calculation may be slower than expected: %v per calculation", avgDuration)
	}

	// Test memory pool performance
	pool := utils.NewFloat32Pool(1024)

	start = time.Now()
	for i := 0; i < iterations; i++ {
		slice := pool.Get()
		// Simulate usage
		for j := 0; j < 100; j++ {
			slice = append(slice, float32(j))
		}
		pool.Put(slice)
	}
	poolDuration := time.Since(start)

	// Direct allocation test
	start = time.Now()
	for i := 0; i < iterations; i++ {
		slice := make([]float32, 0, 1024)
		// Simulate usage
		for j := 0; j < 100; j++ {
			slice = append(slice, float32(j))
		}
	}
	directDuration := time.Since(start)

	t.Logf("Pool allocation: %v, Direct allocation: %v", poolDuration, directDuration)
	t.Logf("Pool performance ratio: %.2fx", float64(directDuration)/float64(poolDuration))
}

// RunAllBenchmarks runs all Phase 5 benchmarks
func RunAllBenchmarks(b *testing.B) {
	b.Run("DistanceCalculations", BenchmarkDistanceCalculations)
	b.Run("BatchDistanceCalculation", BenchmarkBatchDistanceCalculation)
	b.Run("MemoryPooling", BenchmarkMemoryPooling)
	b.Run("BufferPooling", BenchmarkBufferPooling)
	b.Run("WorkerPool", BenchmarkWorkerPool)
	b.Run("LRUCache", BenchmarkLRUCache)
	b.Run("VectorNormalization", BenchmarkVectorNormalization)
	b.Run("VectorQuantization", BenchmarkVectorQuantization)
}
