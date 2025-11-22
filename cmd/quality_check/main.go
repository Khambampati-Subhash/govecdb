package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/khambampati-subhash/govecdb/client"
)

var (
	addr = flag.String("addr", "localhost:8080", "Server address")
	dim  = flag.Int("dim", 128, "Vector dimension")
	n    = flag.Int("n", 100, "Number of vectors to insert and check")
)

func main() {
	flag.Parse()

	// Connect to server
	c, err := client.NewClient(*addr)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	fmt.Printf("Generating %d vectors of dimension %d...\n", *n, *dim)
	vectors := make([][]float32, *n)
	ids := make([]string, *n)
	for i := 0; i < *n; i++ {
		vectors[i] = make([]float32, *dim)
		for j := 0; j < *dim; j++ {
			vectors[i][j] = rand.Float32()
		}
		ids[i] = fmt.Sprintf("vec-%d", i)
	}

	// Insert
	fmt.Println("Inserting vectors...")
	start := time.Now()
	for i := 0; i < *n; i++ {
		if err := c.Put(context.Background(), ids[i], vectors[i]); err != nil {
			log.Fatalf("Insert failed: %v", err)
		}
	}
	fmt.Printf("Insertion took %v\n", time.Since(start))

	// Verify Data Integrity (Get by ID and compare values)
	fmt.Println("Verifying data integrity (Get by ID and compare values)...")
	integritySuccess := 0
	for i := 0; i < *n; i++ {
		retrievedVec, err := c.Get(context.Background(), ids[i])
		if err != nil {
			log.Printf("Get failed for %s: %v", ids[i], err)
			continue
		}

		// Compare values
		if len(retrievedVec) != len(vectors[i]) {
			log.Printf("Dimension mismatch for %s: expected %d, got %d", ids[i], len(vectors[i]), len(retrievedVec))
			continue
		}

		match := true
		for j := 0; j < *dim; j++ {
			// Exact float comparison because we expect exact storage
			if retrievedVec[j] != vectors[i][j] {
				log.Printf("Data corruption for %s at index %d: expected %f, got %f", ids[i], j, vectors[i][j], retrievedVec[j])
				match = false
				break
			}
		}

		if match {
			integritySuccess++
		}
	}
	fmt.Printf("Data Integrity: %d/%d vectors match exactly.\n", integritySuccess, *n)

	// Verify (Recall@1)
	fmt.Println("Verifying retrieval (searching for exact inserted vectors)...")
	successCount := 0
	for i := 0; i < *n; i++ {
		// Search for the exact vector we inserted
		results, err := c.Search(context.Background(), vectors[i], 10) // Request top 10 to be safe
		if err != nil {
			log.Printf("Search failed for %s: %v", ids[i], err)
			continue
		}

		found := false
		if len(results) > 0 {
			// Check if the top result is the correct ID
			if results[0].ID == ids[i] {
				found = true
			} else {
				// Check if it's in the top k (just in case of floating point quirks, though dist should be 0)
				for _, res := range results {
					if res.ID == ids[i] {
						// Found but not top? That's weird for exact match unless duplicates.
						// But for "Recall@1" we strictly want it at the top.
						// Let's count it as success if it's the top result.
						break
					}
				}
			}
		}

		if found {
			successCount++
		} else {
			// log.Printf("Failed to retrieve %s. Top result: %v", ids[i], results[0].ID)
		}
	}

	recall := float64(successCount) / float64(*n)
	fmt.Printf("Recall@1 (Exact Match): %.2f%% (%d/%d)\n", recall*100, successCount, *n)

	if successCount == *n {
		fmt.Println("SUCCESS: Perfect retrieval quality!")
	} else {
		fmt.Println("WARNING: Retrieval is not 100%. HNSW is an approximate index.")
	}
}
