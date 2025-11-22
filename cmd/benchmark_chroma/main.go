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

	chroma "github.com/amikos-tech/chroma-go"
	"github.com/amikos-tech/chroma-go/types"
)

var (
	addr = flag.String("addr", "http://localhost:8000", "ChromaDB address")
	n    = flag.Int("n", 1000, "Number of vectors")
	dim  = flag.Int("dim", 128, "Vector dimension")
)

func main() {
	flag.Parse()

	// Connect to ChromaDB
	client, err := chroma.NewClient(chroma.WithBasePath(*addr))
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	fmt.Printf("Benchmarking ChromaDB: N=%d, Dim=%d\n", *n, *dim)

	// Create Collection
	collectionName := fmt.Sprintf("bench-%d-%d-%d", *n, *dim, time.Now().Unix())
	// Metadata for HNSW (optional, Chroma defaults are usually fine but we can tune if needed)
	metadata := map[string]interface{}{
		"hnsw:space": "l2", // Euclidean distance
	}

	// Reset/Delete if exists (not easy in Chroma client without error checking, so we use unique name)
	// Actually, let's try to get or create.

	collection, err := client.CreateCollection(context.Background(), collectionName, metadata, true, nil, types.L2)
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}

	// Ensure cleanup
	defer func() {
		_, err := client.DeleteCollection(context.Background(), collectionName)
		if err != nil {
			log.Printf("Failed to delete collection: %v", err)
		}
	}()

	// Generate Data
	vectors := make([][]float32, *n)
	ids := make([]string, *n)
	for i := 0; i < *n; i++ {
		vectors[i] = randomVector(*dim)
		ids[i] = fmt.Sprintf("vec-%d", i)
	}

	// 1. Benchmark Insertion (Batch)
	start := time.Now()
	batchSize := 100 // Chroma recommends smaller batches usually, but 100 is fine

	// Chroma client might not be thread-safe for the SAME collection object?
	// Documentation says "Client is thread safe". Collection object?
	// Usually it's better to batch in main thread for Chroma as it's HTTP based.
	// Parallel HTTP requests might help.

	var wg sync.WaitGroup
	sem := make(chan struct{}, 10) // Concurrency 10

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

			batchIDs := ids[startIdx:endIdx]
			batchVecs := vectors[startIdx:endIdx]

			// Convert to Chroma types if needed.
			// Add expects embeddings as []*types.Embedding
			embeddings := make([]*types.Embedding, len(batchVecs))
			for k, v := range batchVecs {
				embeddings[k] = types.NewEmbeddingFromFloat32(v)
			}

			_, err := collection.Add(context.Background(), embeddings, nil, batchIDs, nil)
			if err != nil {
				log.Printf("Add failed: %v", err)
			}
		}(i, end)
	}
	wg.Wait()
	insertDuration := time.Since(start)
	fmt.Printf("Insertion Time: %v\n", insertDuration)
	fmt.Printf("Insertion Rate: %.2f ops/s\n", float64(*n)/insertDuration.Seconds())

	// 2. Benchmark Search Quality (Recall)
	numQueries := 100
	k := 10
	recallSum := 0.0
	var totalSearchDuration time.Duration

	for i := 0; i < numQueries; i++ {
		queryIdx := rand.Intn(*n)
		queryVec := vectors[queryIdx]
		queryEmbedding := types.NewEmbeddingFromFloat32(queryVec)

		// Ground Truth
		groundTruth := bruteForceSearch(vectors, queryVec, k)

		// ANN Search
		start := time.Now()
		results, err := collection.QueryWithOptions(context.Background(),
			types.WithQueryEmbedding(queryEmbedding),
			types.WithNResults(int32(k)),
		)
		totalSearchDuration += time.Since(start)

		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}

		// Calculate Recall
		// Results structure: results.Ids is [][]string (batch results)
		if len(results.Ids) > 0 {
			resultIDs := results.Ids[0]
			found := 0
			for _, resID := range resultIDs {
				for _, gt := range groundTruth {
					if resID == gt.ID {
						found++
						break
					}
				}
			}
			recallSum += float64(found) / float64(k)
		}
	}

	fmt.Printf("Search Time (Total): %v\n", totalSearchDuration)
	fmt.Printf("Search Latency (Avg): %v\n", totalSearchDuration/time.Duration(numQueries))
	fmt.Printf("Search QPS: %.2f\n", float64(numQueries)/totalSearchDuration.Seconds())
	fmt.Printf("Recall@%d: %.4f\n", k, recallSum/float64(numQueries))

	// 3. Benchmark Search QPS (Concurrent)
	searchDuration := 5 * time.Second
	concurrency := 10
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
					queryEmbedding := types.NewEmbeddingFromFloat32(queryVec)

					_, err := collection.QueryWithOptions(context.Background(),
						types.WithQueryEmbedding(queryEmbedding),
						types.WithNResults(int32(k)),
					)
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
	type item struct {
		id   string
		dist float32
	}

	items := make([]item, len(vectors))
	for i, vec := range vectors {
		dist := euclideanDistance(query, vec)
		items[i] = item{id: fmt.Sprintf("vec-%d", i), dist: dist}
	}

	topK := make([]item, 0, k)
	for _, it := range items {
		if len(topK) < k {
			topK = append(topK, it)
		} else {
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
	return sum
}
