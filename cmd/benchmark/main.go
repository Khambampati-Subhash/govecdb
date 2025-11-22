package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/khambampati-subhash/govecdb/client"
)

var (
	addr        = flag.String("addr", "localhost:8080", "Server address")
	numVectors  = flag.Int("n", 10000, "Number of vectors to insert")
	dim         = flag.Int("dim", 128, "Vector dimension")
	concurrency = flag.Int("c", 10, "Concurrency level")
	searchOnly  = flag.Bool("search-only", false, "Skip insertion, only search")
)

func main() {
	flag.Parse()

	c, err := client.NewClient(*addr)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	ctx := context.Background()

	// Insertion Phase
	if !*searchOnly {
		fmt.Printf("Inserting %d vectors (dim=%d, concurrency=%d)...\n", *numVectors, *dim, *concurrency)
		start := time.Now()

		var wg sync.WaitGroup
		sem := make(chan struct{}, *concurrency)
		var count int64

		for i := 0; i < *numVectors; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(id int) {
				defer wg.Done()
				defer func() { <-sem }()

				vec := randomVector(*dim)
				vecID := fmt.Sprintf("vec-%d", id)

				if err := c.Put(ctx, vecID, vec); err != nil {
					log.Printf("Put failed: %v", err)
					return
				}
				atomic.AddInt64(&count, 1)
			}(i)
		}
		wg.Wait()

		duration := time.Since(start)
		fmt.Printf("Inserted %d vectors in %v (%.2f ops/s)\n", count, duration, float64(count)/duration.Seconds())
	}

	// Search Phase
	fmt.Printf("Searching (concurrency=%d)...\n", *concurrency)
	start := time.Now()
	var searchCount int64

	// Run searches for 5 seconds
	done := make(chan struct{})
	time.AfterFunc(5*time.Second, func() { close(done) })

	var wg sync.WaitGroup
	for i := 0; i < *concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-done:
					return
				default:
					query := randomVector(*dim)
					_, err := c.Search(ctx, query, 10)
					if err != nil {
						log.Printf("Search failed: %v", err)
					}
					atomic.AddInt64(&searchCount, 1)
				}
			}
		}()
	}
	wg.Wait()

	duration := time.Since(start)
	fmt.Printf("Performed %d searches in %v (%.2f QPS)\n", searchCount, duration, float64(searchCount)/duration.Seconds())
}

func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}
