package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/khambampati-subhash/govecdb/index"
)

func main() {
	fmt.Println("GoVecDB Search Accuracy Test")
	fmt.Println("============================")

	// Create index
	config := &index.Config{
		Dimension:      128,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		Metric:         index.Cosine,
		ThreadSafe:     true,
	}

	idx, err := index.NewHNSWIndex(config)
	if err != nil {
		fmt.Printf("Error creating index: %v\n", err)
		return
	}
	defer idx.Close()

	// Generate test vectors
	numVectors := 1000
	vectors := make([]*index.Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		data := make([]float32, 128)
		for j := 0; j < 128; j++ {
			data[j] = rand.Float32()*2 - 1
		}
		
		// Normalize
		var norm float32
		for _, val := range data {
			norm += val * val
		}
		norm = float32(1.0 / (float64(norm) + 1e-8))
		for j := range data {
			data[j] *= norm
		}

		vectors[i] = &index.Vector{
			ID:   fmt.Sprintf("vec_%d", i),
			Data: data,
		}
	}

	// Insert vectors using our optimized batch insert
	start := time.Now()
	err = idx.AddBatch(vectors)
	if err != nil {
		fmt.Printf("Error inserting batch: %v\n", err)
		return
	}
	elapsed := time.Since(start)
	
	fmt.Printf("Inserted %d vectors in %v (%.2f vec/sec)\n", 
		numVectors, elapsed, float64(numVectors)/(float64(elapsed.Nanoseconds())/1e9))

	// Test search accuracy
	fmt.Println("\nTesting search accuracy...")
	
	correctResults := 0
	totalTests := 10
	
	for i := 0; i < totalTests; i++ {
		// Use an existing vector as query
		queryVector := vectors[i].Data
		expectedID := vectors[i].ID
		
		// Search for k=10 neighbors
		results, err := idx.Search(queryVector, 10)
		if err != nil {
			fmt.Printf("Search error: %v\n", err)
			continue
		}
		
		// Check if the exact vector is in top results (should be #1 with distance ~0)
		if len(results) > 0 && results[0].ID == expectedID {
			correctResults++
		}
		
		fmt.Printf("Query %d: Found %d results, exact match at position ", i+1, len(results))
		exactPosition := -1
		for j, result := range results {
			if result.ID == expectedID {
				exactPosition = j
				break
			}
		}
		if exactPosition >= 0 {
			fmt.Printf("%d (distance: %.6f)\n", exactPosition+1, results[exactPosition].Score)
		} else {
			fmt.Printf("not found\n")
		}
	}
	
	accuracy := float64(correctResults) / float64(totalTests) * 100
	fmt.Printf("\nSearch Accuracy: %d/%d (%.1f%%) exact matches found as #1 result\n", 
		correctResults, totalTests, accuracy)
}