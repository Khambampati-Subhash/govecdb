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
	addr = flag.String("addr", "localhost:8080", "Server address")
	n    = flag.Int("n", 1000, "Number of vectors")
	dim  = flag.Int("dim", 128, "Vector dimension")
)

func main() {
	flag.Parse()

	// Connect to server
	c, err := client.NewClient(*addr)
	if err != nil {
		log.Fatalf("Did not connect: %v", err)
	}
	defer c.Close()
	fmt.Printf("Benchmarking: N=%d, Dim=%d\n", *n, *dim)

	// Generate Data
	vectors := make([][]float32, *n)
	for i := 0; i < *n; i++ {
		vectors[i] = randomVector(*dim)
	}

	// 1. Benchmark Insertion (Batch)
	start := time.Now()
	batchSize := 100
	concurrency := 10
	var wg sync.WaitGroup
	sem := make(chan struct{}, concurrency)

	for i := 0; i < *n; i += batchSize {
		end := i + batchSize
		if end > *n {
			end = *n
		}

		wg.Add(1)
		sem <- struct{}{}
		go func(startIdx, endIdx int) {
			defer wg.Done()
			defer func() { <-sem }()

			for j := startIdx; j < endIdx; j++ {
				id := fmt.Sprintf("vec-%d", j)
				err := c.Put(context.Background(), id, vectors[j])
				if err != nil {
					log.Printf("Put failed: %v", err)
				}
			}
		}(i, end)
	}
	wg.Wait()
	insertDuration := time.Since(start)
	fmt.Printf("Insertion Time: %v\n", insertDuration)
	fmt.Printf("Insertion Rate: %.2f ops/s\n", float64(*n)/insertDuration.Seconds())

	// 2. Benchmark Search Quality (Recall) - Sequential
	// We measure recall on a subset to avoid taking too long
	numRecallQueries := 100
	k := 10
	recallSum := 0.0

	for i := 0; i < numRecallQueries; i++ {
		queryIdx := rand.Intn(*n)
		queryVec := vectors[queryIdx]

		// Ground Truth (Brute Force)
		groundTruth := bruteForceSearch(vectors, queryVec, k)

		// ANN Search
		results, err := c.Search(context.Background(), queryVec, k)
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}

		// Calculate Recall
		found := 0
		for _, res := range results {
			for _, gt := range groundTruth {
				if res.ID == gt.ID {
					found++
					break
				}
			}
		}
		recallSum += float64(found) / float64(k)
	}
	fmt.Printf("Recall@%d: %.4f\n", k, recallSum/float64(numRecallQueries))

	// 3. Benchmark Search Performance (QPS) - Concurrent
	// Run for 5 seconds or fixed number of queries
	searchDuration := 5 * time.Second
	fmt.Printf("Benchmarking Search QPS (duration=%v, concurrency=%d)...\n", searchDuration, concurrency)

	var searchOps int64
	searchStart := time.Now()
	done := make(chan struct{})
	time.AfterFunc(searchDuration, func() { close(done) })

	var searchWg sync.WaitGroup
	for i := 0; i < concurrency; i++ {
		searchWg.Add(1)
		go func() {
			defer searchWg.Done()
			for {
				select {
				case <-done:
					return
				default:
					queryIdx := rand.Intn(*n)
					queryVec := vectors[queryIdx]
					_, err := c.Search(context.Background(), queryVec, k)
					if err != nil {
						log.Printf("Search failed: %v", err)
					}
					atomic.AddInt64(&searchOps, 1)
				}
			}
		}()
	}
	searchWg.Wait()
	actualDuration := time.Since(searchStart)

	fmt.Printf("Search QPS: %.2f\n", float64(searchOps)/actualDuration.Seconds())
	fmt.Printf("Avg Latency: %v\n", time.Duration(float64(actualDuration.Nanoseconds())/float64(searchOps)))
}

func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()
	}
	return vec
}

type Result struct {
	ID    string
	Score float32
}

func bruteForceSearch(vectors [][]float32, query []float32, k int) []Result {
	// Simple brute force
	// Note: This might be slow for large N and Dim, but for N=10000 it's okay-ish (10k comparisons)
	// For 10k * 4096, it's 40M floats.

	// Use a heap for top-k
	// ... implementing a simple sort for now

	type item struct {
		id   string
		dist float32
	}

	items := make([]item, len(vectors))
	for i, vec := range vectors {
		dist := euclideanDistance(query, vec)
		items[i] = item{id: fmt.Sprintf("vec-%d", i), dist: dist}
	}

	// Sort
	// (Optimization: use partial sort or heap)
	// Since this is just ground truth generation, simple sort is fine for small N
	// For larger N, we might need optimization.

	// Bubble sort top K? No, just full sort for simplicity
	// Or just iterate and keep top K

	topK := make([]item, 0, k)

	for _, it := range items {
		if len(topK) < k {
			topK = append(topK, it)
		} else {
			// Find max dist in topK
			maxIdx := 0
			for j := 1; j < len(topK); j++ {
				if topK[j].dist > topK[maxIdx].dist {
					maxIdx = j
				}
			}

			if it.dist < topK[maxIdx].dist {
				topK[maxIdx] = it
			}
		}
	}

	res := make([]Result, len(topK))
	for i, it := range topK {
		res[i] = Result{ID: it.id, Score: it.dist}
	}
	return res
}

func euclideanDistance(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum // Squared Euclidean for comparison
}
