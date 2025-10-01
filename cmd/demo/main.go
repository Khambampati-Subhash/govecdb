package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/khambampati-subhash/govecdb/index"
)

func generateRandomVector(dimension int, rng *rand.Rand) []float32 {
	vector := make([]float32, dimension)
	for i := range vector {
		vector[i] = rng.Float32()*2 - 1 // Random values between -1 and 1
	}
	return vector
}

func main() {
	fmt.Println("🚀 GoVecDB HNSW Implementation Demo")
	fmt.Println("===================================")

	// Create configuration
	config := &index.Config{
		Dimension:      384, // Common embedding dimension (e.g., sentence-transformers)
		Metric:         index.Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		Seed:           42,
		ThreadSafe:     true,
	}

	// Create index
	fmt.Println("📦 Creating HNSW index...")
	hnswIndex, err := index.NewHNSWIndex(config)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer hnswIndex.Close()

	// Generate sample vectors with metadata
	fmt.Println("🎲 Generating sample vectors...")
	rng := rand.New(rand.NewSource(42))
	vectors := make([]*index.Vector, 1000)

	categories := []string{"technology", "science", "arts", "sports", "music"}

	for i := 0; i < 1000; i++ {
		vectors[i] = &index.Vector{
			ID:   fmt.Sprintf("doc_%d", i),
			Data: generateRandomVector(384, rng),
			Metadata: map[string]interface{}{
				"title":      fmt.Sprintf("Document %d", i),
				"category":   categories[i%len(categories)],
				"created_at": time.Now().Add(-time.Duration(i) * time.Hour),
				"importance": rng.Float64(),
				"word_count": rng.Intn(1000) + 100,
			},
		}
	}

	// Batch insert vectors
	fmt.Println("📤 Inserting vectors into index...")
	start := time.Now()
	err = hnswIndex.AddBatch(vectors)
	if err != nil {
		log.Fatalf("Failed to insert vectors: %v", err)
	}
	insertTime := time.Since(start)

	// Display statistics
	stats := hnswIndex.GetStats()
	fmt.Printf("✅ Inserted %d vectors in %v\n", stats.NodeCount, insertTime)
	fmt.Printf("📊 Index Statistics:\n")
	fmt.Printf("   - Nodes: %d\n", stats.NodeCount)
	fmt.Printf("   - Edges: %d\n", stats.EdgeCount)
	fmt.Printf("   - Max Layer: %d\n", stats.MaxLayer)
	fmt.Printf("   - Avg Degree: %.2f\n", stats.AvgDegree)
	fmt.Printf("   - Dimension: %d\n", stats.Dimension)
	fmt.Printf("   - Metric: %s\n", stats.Metric)

	// Perform searches
	fmt.Println("\n🔍 Performing sample searches...")

	// Search 1: Basic similarity search
	fmt.Println("\n--- Search 1: Basic Similarity ---")
	queryVector := vectors[42].Data // Use one of our vectors as query
	start = time.Now()
	results, err := hnswIndex.Search(queryVector, 5)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	searchTime := time.Since(start)

	fmt.Printf("Query took %v, found %d results:\n", searchTime, len(results))
	for i, result := range results {
		title := result.Metadata["title"]
		category := result.Metadata["category"]
		fmt.Printf("  %d. ID: %s, Score: %.4f, Title: %s, Category: %s\n",
			i+1, result.ID, result.Score, title, category)
	}

	// Search 2: Filtered search
	fmt.Println("\n--- Search 2: Filtered Search (Technology category) ---")
	techFilter := func(metadata map[string]interface{}) bool {
		category, exists := metadata["category"]
		return exists && category == "technology"
	}

	start = time.Now()
	filteredResults, err := hnswIndex.SearchWithFilter(queryVector, 5, techFilter)
	if err != nil {
		log.Fatalf("Filtered search failed: %v", err)
	}
	filteredSearchTime := time.Since(start)

	fmt.Printf("Filtered query took %v, found %d results:\n", filteredSearchTime, len(filteredResults))
	for i, result := range filteredResults {
		title := result.Metadata["title"]
		importance := result.Metadata["importance"]
		fmt.Printf("  %d. ID: %s, Score: %.4f, Title: %s, Importance: %.3f\n",
			i+1, result.ID, result.Score, title, importance)
	}

	// Search 3: Different distance metrics comparison
	fmt.Println("\n--- Search 3: Distance Metrics Comparison ---")
	metrics := []index.DistanceMetric{
		index.Cosine,
		index.Euclidean,
		index.Manhattan,
		index.DotProduct,
	}

	testQuery := generateRandomVector(384, rng)

	for _, metric := range metrics {
		// Create a new index with different metric
		metricConfig := *config
		metricConfig.Metric = metric

		metricIndex, err := index.NewHNSWIndex(&metricConfig)
		if err != nil {
			log.Printf("Failed to create index with %s metric: %v", metric.String(), err)
			continue
		}

		// Add a subset of vectors for quick demo
		err = metricIndex.AddBatch(vectors[:100])
		if err != nil {
			log.Printf("Failed to add vectors to %s index: %v", metric.String(), err)
			metricIndex.Close()
			continue
		}

		start = time.Now()
		metricResults, err := metricIndex.Search(testQuery, 3)
		if err != nil {
			log.Printf("Search failed for %s metric: %v", metric.String(), err)
		} else {
			searchTime = time.Since(start)
			fmt.Printf("  %s: %v, Top result score: %.4f\n",
				metric.String(), searchTime, metricResults[0].Score)
		}

		metricIndex.Close()
	}

	// Performance benchmark
	fmt.Println("\n⚡ Performance Benchmark")
	fmt.Println("--- Search Performance ---")

	numQueries := 100
	queryVectors := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queryVectors[i] = generateRandomVector(384, rng)
	}

	start = time.Now()
	totalResults := 0
	for _, query := range queryVectors {
		results, err := hnswIndex.Search(query, 10)
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}
		totalResults += len(results)
	}
	benchmarkTime := time.Since(start)

	fmt.Printf("Executed %d searches in %v\n", numQueries, benchmarkTime)
	fmt.Printf("Average search time: %v\n", benchmarkTime/time.Duration(numQueries))
	fmt.Printf("Searches per second: %.0f\n", float64(numQueries)/benchmarkTime.Seconds())
	fmt.Printf("Total results returned: %d\n", totalResults)

	// Demonstrate vector operations
	fmt.Println("\n🧮 Vector Operations Demo")
	fmt.Println("--- Distance Calculations ---")

	vec1 := []float32{1.0, 0.0, 0.0}
	vec2 := []float32{0.0, 1.0, 0.0}
	vec3 := []float32{1.0, 0.0, 0.0} // Same as vec1

	cosine12, _ := index.CosineDistance(vec1, vec2)
	cosine13, _ := index.CosineDistance(vec1, vec3)
	euclidean12, _ := index.EuclideanDistance(vec1, vec2)
	manhattan12, _ := index.ManhattanDistance(vec1, vec2)

	fmt.Printf("Vector 1: [%.1f, %.1f, %.1f]\n", vec1[0], vec1[1], vec1[2])
	fmt.Printf("Vector 2: [%.1f, %.1f, %.1f]\n", vec2[0], vec2[1], vec2[2])
	fmt.Printf("Vector 3: [%.1f, %.1f, %.1f]\n", vec3[0], vec3[1], vec3[2])
	fmt.Printf("\nDistances between vec1 and vec2:\n")
	fmt.Printf("  Cosine: %.4f\n", cosine12)
	fmt.Printf("  Euclidean: %.4f\n", euclidean12)
	fmt.Printf("  Manhattan: %.4f\n", manhattan12)
	fmt.Printf("\nCosine distance between identical vectors (vec1, vec3): %.4f\n", cosine13)

	// Final statistics
	fmt.Println("\n📈 Final Index Statistics")
	finalStats := hnswIndex.GetStats()
	fmt.Printf("Final search count: %d\n", finalStats.SearchCount)
	fmt.Printf("Index created at: %s\n", finalStats.CreatedAt.Format(time.RFC3339))
	fmt.Printf("Last updated at: %s\n", finalStats.LastUpdateAt.Format(time.RFC3339))

	fmt.Println("\n🎉 Demo completed successfully!")
}
