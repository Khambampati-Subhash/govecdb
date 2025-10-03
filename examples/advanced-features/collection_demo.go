// Package main demonstrates comprehensive GoVecDB collection features
// This example shows collection-level operations and real-world scenarios
// Run with: go run examples/advanced-features/collection_demo.go
package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
	"github.com/khambampati-subhash/govecdb/collection"
	"github.com/khambampati-subhash/govecdb/store"
)

func main() {
	fmt.Println("🚀 GoVecDB Collection Demo Suite")
	fmt.Println("================================")

	// Demo 1: Basic Operations
	basicOperationsDemo()

	// Demo 2: Advanced Filtering
	advancedFilteringDemo()

	// Demo 3: Performance Testing
	performanceDemo()

	// Demo 4: Real-world Scenarios
	realWorldScenariosDemo()

	fmt.Println("\n🎉 Collection demos completed successfully!")
	fmt.Println("💡 GoVecDB collections are ready for production use!")
}

func basicOperationsDemo() {
	fmt.Println("\n📊 Demo 1: Basic Vector Operations")
	fmt.Println("==================================")

	// Create configuration
	config := &api.CollectionConfig{
		Name:           "basic-demo",
		Dimension:      4,
		Metric:         api.Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		ThreadSafe:     true,
	}

	storeConfig := &store.StoreConfig{
		Name:         "basic-store",
		PreallocSize: 100,
		EnableStats:  true,
	}

	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		log.Fatal(err)
	}
	defer coll.Close()

	ctx := context.Background()

	// Add vectors
	vectors := []*api.Vector{
		{
			ID:       "vec1",
			Data:     []float32{1.0, 0.0, 0.0, 0.0},
			Metadata: map[string]interface{}{"label": "x-axis", "type": "unit"},
		},
		{
			ID:       "vec2",
			Data:     []float32{0.0, 1.0, 0.0, 0.0},
			Metadata: map[string]interface{}{"label": "y-axis", "type": "unit"},
		},
		{
			ID:       "vec3",
			Data:     []float32{0.0, 0.0, 1.0, 0.0},
			Metadata: map[string]interface{}{"label": "z-axis", "type": "unit"},
		},
		{
			ID:       "vec4",
			Data:     []float32{0.7, 0.7, 0.0, 0.0},
			Metadata: map[string]interface{}{"label": "diagonal", "type": "composite"},
		},
	}

	fmt.Printf("📤 Adding %d vectors...\n", len(vectors))
	for _, vec := range vectors {
		err = coll.Add(ctx, vec)
		if err != nil {
			log.Fatal(err)
		}
	}

	// Search
	query := []float32{0.8, 0.6, 0.0, 0.0}
	request := &api.SearchRequest{
		Vector: query,
		K:      3,
	}

	results, err := coll.Search(ctx, request)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("🔍 Query: [%.1f, %.1f, %.1f, %.1f]\n", query[0], query[1], query[2], query[3])
	fmt.Printf("📝 Found %d results:\n", len(results))
	for i, result := range results {
		fmt.Printf("  %d. %s (%.4f) - %s\n",
			i+1, result.Vector.ID, result.Score, result.Vector.Metadata["label"])
	}

	// Stats
	stats, err := coll.Stats(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("📊 Collection: %d vectors\n", stats.VectorCount)
}

func advancedFilteringDemo() {
	fmt.Println("\n🔍 Demo 2: Advanced Filtering")
	fmt.Println("=============================")

	config := &api.CollectionConfig{
		Name:           "filtering-demo",
		Dimension:      3,
		Metric:         api.Euclidean,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		ThreadSafe:     true,
	}

	storeConfig := &store.StoreConfig{
		Name:         "filtering-store",
		PreallocSize: 100,
		EnableStats:  true,
	}

	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		log.Fatal(err)
	}
	defer coll.Close()

	ctx := context.Background()

	// Add documents with rich metadata
	documents := []*api.Vector{
		{
			ID:   "doc1",
			Data: []float32{1.0, 0.0, 0.0},
			Metadata: map[string]interface{}{
				"title":    "AI Revolution",
				"category": "technology",
				"rating":   4.5,
				"year":     2023,
				"tags":     []interface{}{"ai", "machine-learning"},
			},
		},
		{
			ID:   "doc2",
			Data: []float32{0.0, 1.0, 0.0},
			Metadata: map[string]interface{}{
				"title":    "Go Programming",
				"category": "technology",
				"rating":   4.2,
				"year":     2022,
				"tags":     []interface{}{"programming", "go"},
			},
		},
		{
			ID:   "doc3",
			Data: []float32{0.0, 0.0, 1.0},
			Metadata: map[string]interface{}{
				"title":    "Physics Today",
				"category": "science",
				"rating":   4.0,
				"year":     2023,
				"tags":     []interface{}{"physics", "research"},
			},
		},
		{
			ID:   "doc4",
			Data: []float32{0.5, 0.5, 0.0},
			Metadata: map[string]interface{}{
				"title":    "Deep Learning",
				"category": "technology",
				"rating":   4.8,
				"year":     2023,
				"tags":     []interface{}{"ai", "deep-learning"},
			},
		},
	}

	fmt.Printf("📤 Adding %d documents...\n", len(documents))
	for _, doc := range documents {
		err = coll.Add(ctx, doc)
		if err != nil {
			log.Fatal(err)
		}
	}

	query := []float32{0.6, 0.4, 0.0}

	// Test 1: Simple filter
	fmt.Println("\n🎯 Test 1: Simple category filter")
	categoryFilter := &api.FieldFilter{
		Field: "category",
		Op:    api.FilterEq,
		Value: "technology",
	}

	request := &api.SearchRequest{
		Vector: query,
		K:      10,
		Filter: categoryFilter,
	}

	results, err := coll.Search(ctx, request)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   Found %d technology documents:\n", len(results))
	for _, result := range results {
		fmt.Printf("   - %s (%.3f)\n", result.Vector.Metadata["title"], result.Score)
	}

	// Test 2: Complex AND filter
	fmt.Println("\n⚡ Test 2: Complex AND filter (technology + rating > 4.0 + year = 2023)")
	complexFilter := &api.LogicalFilter{
		Op: api.FilterAnd,
		Filters: []api.FilterExpr{
			&api.FieldFilter{Field: "category", Op: api.FilterEq, Value: "technology"},
			&api.FieldFilter{Field: "rating", Op: api.FilterGt, Value: 4.0},
			&api.FieldFilter{Field: "year", Op: api.FilterEq, Value: 2023},
		},
	}

	request.Filter = complexFilter
	results, err = coll.Search(ctx, request)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   Found %d matching documents:\n", len(results))
	for _, result := range results {
		fmt.Printf("   - %s (%.3f) - Rating: %.1f, Year: %d\n",
			result.Vector.Metadata["title"], result.Score,
			result.Vector.Metadata["rating"], result.Vector.Metadata["year"])
	}

	// Test 3: OR filter
	fmt.Println("\n🌟 Test 3: OR filter (category = technology OR science)")
	orFilter := &api.LogicalFilter{
		Op: api.FilterOr,
		Filters: []api.FilterExpr{
			&api.FieldFilter{Field: "category", Op: api.FilterEq, Value: "technology"},
			&api.FieldFilter{Field: "category", Op: api.FilterEq, Value: "science"},
		},
	}

	request.Filter = orFilter
	results, err = coll.Search(ctx, request)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   Found %d documents:\n", len(results))
	for _, result := range results {
		fmt.Printf("   - %s (%s) - %.3f\n",
			result.Vector.Metadata["title"], result.Vector.Metadata["category"], result.Score)
	}
}

func performanceDemo() {
	fmt.Println("\n⚡ Demo 3: Performance Testing")
	fmt.Println("=============================")

	config := &api.CollectionConfig{
		Name:           "perf-demo",
		Dimension:      256,
		Metric:         api.Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		ThreadSafe:     true,
	}

	storeConfig := &store.StoreConfig{
		Name:         "perf-store",
		PreallocSize: 10000,
		EnableStats:  true,
	}

	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		log.Fatal(err)
	}
	defer coll.Close()

	ctx := context.Background()

	// Test batch insertion
	fmt.Println("📦 Testing batch insertion...")
	batchSize := 5000
	vectors := make([]*api.Vector, batchSize)

	start := time.Now()
	for i := 0; i < batchSize; i++ {
		data := make([]float32, 256)
		for j := range data {
			data[j] = rand.Float32()*2 - 1
		}

		vectors[i] = &api.Vector{
			ID:   fmt.Sprintf("vec_%05d", i),
			Data: data,
			Metadata: map[string]interface{}{
				"batch":     i / 1000,
				"id":        i,
				"timestamp": time.Now().Unix(),
			},
		}
	}
	generationTime := time.Since(start)

	fmt.Printf("   Generated %d vectors in %v\n", batchSize, generationTime)

	start = time.Now()
	err = coll.AddBatch(ctx, vectors)
	if err != nil {
		log.Fatal(err)
	}
	insertTime := time.Since(start)

	fmt.Printf("   Inserted %d vectors in %v\n", batchSize, insertTime)
	fmt.Printf("   Insert rate: %.0f vectors/sec\n", float64(batchSize)/insertTime.Seconds())

	// Test search performance
	fmt.Println("🔍 Testing search performance...")
	numQueries := 1000
	queries := make([][]float32, numQueries)

	for i := 0; i < numQueries; i++ {
		query := make([]float32, 256)
		for j := range query {
			query[j] = rand.Float32()*2 - 1
		}
		queries[i] = query
	}

	start = time.Now()
	totalResults := 0

	for i, query := range queries {
		request := &api.SearchRequest{
			Vector: query,
			K:      10,
		}

		results, err := coll.Search(ctx, request)
		if err != nil {
			log.Printf("   Search %d failed: %v", i, err)
			continue
		}
		totalResults += len(results)
	}

	searchTime := time.Since(start)
	avgSearchTime := searchTime / time.Duration(numQueries)

	fmt.Printf("   Executed %d searches in %v\n", numQueries, searchTime)
	fmt.Printf("   Average search time: %v\n", avgSearchTime)
	fmt.Printf("   Search rate: %.0f queries/sec\n", float64(numQueries)/searchTime.Seconds())
	fmt.Printf("   Total results: %d\n", totalResults)

	// Memory usage
	stats, err := coll.Stats(ctx)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("📊 Final stats: %d vectors in collection\n", stats.VectorCount)
}

func realWorldScenariosDemo() {
	fmt.Println("\n🌟 Demo 4: Real-world Scenarios")
	fmt.Println("==============================")

	// Scenario 1: Document Search Engine
	fmt.Println("📚 Scenario 1: Document Search Engine")
	documentSearchScenario()

	// Scenario 2: Product Recommendation
	fmt.Println("\n🛍️  Scenario 2: Product Recommendation")
	productRecommendationScenario()

	// Scenario 3: Content Moderation
	fmt.Println("\n🛡️  Scenario 3: Content Moderation")
	contentModerationScenario()
}

func documentSearchScenario() {
	config := &api.CollectionConfig{
		Name:           "documents",
		Dimension:      128,
		Metric:         api.Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		ThreadSafe:     true,
	}

	storeConfig := &store.StoreConfig{
		Name:         "doc-store",
		PreallocSize: 1000,
		EnableStats:  true,
	}

	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		log.Fatal(err)
	}
	defer coll.Close()

	ctx := context.Background()

	// Simulate document embeddings
	documents := []*api.Vector{
		{
			ID:   "paper1",
			Data: generateSimilarVector([]float32{0.8, 0.6, 0.2}, 128),
			Metadata: map[string]interface{}{
				"title":     "Machine Learning Fundamentals",
				"authors":   []string{"Smith, J.", "Doe, A."},
				"journal":   "AI Review",
				"year":      2023,
				"citations": 156,
				"topic":     "machine-learning",
			},
		},
		{
			ID:   "paper2",
			Data: generateSimilarVector([]float32{0.7, 0.8, 0.1}, 128),
			Metadata: map[string]interface{}{
				"title":     "Deep Neural Networks",
				"authors":   []string{"Johnson, B."},
				"journal":   "Neural Computing",
				"year":      2023,
				"citations": 89,
				"topic":     "deep-learning",
			},
		},
		{
			ID:   "paper3",
			Data: generateSimilarVector([]float32{0.3, 0.2, 0.9}, 128),
			Metadata: map[string]interface{}{
				"title":     "Quantum Computing Principles",
				"authors":   []string{"Wilson, C.", "Brown, D."},
				"journal":   "Quantum Journal",
				"year":      2022,
				"citations": 234,
				"topic":     "quantum-computing",
			},
		},
	}

	for _, doc := range documents {
		err = coll.Add(ctx, doc)
		if err != nil {
			log.Fatal(err)
		}
	}

	// Search for ML papers
	mlQuery := generateSimilarVector([]float32{0.8, 0.6, 0.2}, 128)
	request := &api.SearchRequest{
		Vector: mlQuery,
		K:      5,
		Filter: &api.FieldFilter{
			Field: "topic",
			Op:    api.FilterEq,
			Value: "machine-learning",
		},
	}

	results, err := coll.Search(ctx, request)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   Found %d ML papers:\n", len(results))
	for _, result := range results {
		fmt.Printf("   - %s (%.3f) - %d citations\n",
			result.Vector.Metadata["title"], result.Score,
			int(result.Vector.Metadata["citations"].(int)))
	}
}

func productRecommendationScenario() {
	config := &api.CollectionConfig{
		Name:           "products",
		Dimension:      64,
		Metric:         api.Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		ThreadSafe:     true,
	}

	storeConfig := &store.StoreConfig{
		Name:         "product-store",
		PreallocSize: 1000,
		EnableStats:  true,
	}

	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		log.Fatal(err)
	}
	defer coll.Close()

	ctx := context.Background()

	// Product catalog
	products := []*api.Vector{
		{
			ID:   "laptop1",
			Data: generateSimilarVector([]float32{0.9, 0.7, 0.5}, 64),
			Metadata: map[string]interface{}{
				"name":        "Gaming Laptop Pro",
				"category":    "electronics",
				"subcategory": "laptops",
				"price":       1299.99,
				"rating":      4.5,
				"brand":       "TechCorp",
			},
		},
		{
			ID:   "laptop2",
			Data: generateSimilarVector([]float32{0.8, 0.8, 0.4}, 64),
			Metadata: map[string]interface{}{
				"name":        "Business Laptop",
				"category":    "electronics",
				"subcategory": "laptops",
				"price":       899.99,
				"rating":      4.2,
				"brand":       "OfficeTech",
			},
		},
		{
			ID:   "phone1",
			Data: generateSimilarVector([]float32{0.6, 0.9, 0.7}, 64),
			Metadata: map[string]interface{}{
				"name":        "Smartphone X",
				"category":    "electronics",
				"subcategory": "phones",
				"price":       799.99,
				"rating":      4.3,
				"brand":       "PhoneCorp",
			},
		},
	}

	for _, product := range products {
		err = coll.Add(ctx, product)
		if err != nil {
			log.Fatal(err)
		}
	}

	// User interested in laptops - find similar products
	userInterest := generateSimilarVector([]float32{0.85, 0.75, 0.45}, 64)
	request := &api.SearchRequest{
		Vector: userInterest,
		K:      3,
		Filter: &api.LogicalFilter{
			Op: api.FilterAnd,
			Filters: []api.FilterExpr{
				&api.FieldFilter{Field: "category", Op: api.FilterEq, Value: "electronics"},
				&api.FieldFilter{Field: "rating", Op: api.FilterGte, Value: 4.0},
			},
		},
	}

	results, err := coll.Search(ctx, request)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   Recommended products:\n")
	for i, result := range results {
		price := result.Vector.Metadata["price"].(float64)
		rating := result.Vector.Metadata["rating"].(float64)
		fmt.Printf("   %d. %s - $%.2f (%.1f⭐) - %.3f similarity\n",
			i+1, result.Vector.Metadata["name"], price, rating, result.Score)
	}
}

func contentModerationScenario() {
	config := &api.CollectionConfig{
		Name:           "content",
		Dimension:      32,
		Metric:         api.Cosine,
		M:              16,
		EfConstruction: 200,
		MaxLayer:       16,
		ThreadSafe:     true,
	}

	storeConfig := &store.StoreConfig{
		Name:         "content-store",
		PreallocSize: 1000,
		EnableStats:  true,
	}

	coll, err := collection.NewVectorCollection(config, storeConfig)
	if err != nil {
		log.Fatal(err)
	}
	defer coll.Close()

	ctx := context.Background()

	// Content samples with safety ratings
	contents := []*api.Vector{
		{
			ID:   "post1",
			Data: generateSimilarVector([]float32{0.1, 0.9, 0.8}, 32), // Safe content
			Metadata: map[string]interface{}{
				"text":       "Great weather today! Perfect for a walk.",
				"safety":     "safe",
				"confidence": 0.95,
				"timestamp":  time.Now().Unix(),
			},
		},
		{
			ID:   "post2",
			Data: generateSimilarVector([]float32{0.8, 0.2, 0.1}, 32), // Unsafe content
			Metadata: map[string]interface{}{
				"text":       "[inappropriate content example]",
				"safety":     "unsafe",
				"confidence": 0.89,
				"timestamp":  time.Now().Unix(),
			},
		},
		{
			ID:   "post3",
			Data: generateSimilarVector([]float32{0.2, 0.8, 0.7}, 32), // Safe content
			Metadata: map[string]interface{}{
				"text":       "Excited about the new book release!",
				"safety":     "safe",
				"confidence": 0.92,
				"timestamp":  time.Now().Unix(),
			},
		},
	}

	for _, content := range contents {
		err = coll.Add(ctx, content)
		if err != nil {
			log.Fatal(err)
		}
	}

	// Check new content against known unsafe patterns
	newContent := generateSimilarVector([]float32{0.7, 0.3, 0.2}, 32) // Similar to unsafe
	request := &api.SearchRequest{
		Vector: newContent,
		K:      3,
		Filter: &api.LogicalFilter{
			Op: api.FilterAnd,
			Filters: []api.FilterExpr{
				&api.FieldFilter{Field: "safety", Op: api.FilterEq, Value: "unsafe"},
				&api.FieldFilter{Field: "confidence", Op: api.FilterGt, Value: 0.8},
			},
		},
	}

	results, err := coll.Search(ctx, request)
	if err != nil {
		log.Fatal(err)
	}

	if len(results) > 0 {
		fmt.Printf("   ⚠️  Content flagged as potentially unsafe:\n")
		for _, result := range results {
			confidence := result.Vector.Metadata["confidence"].(float64)
			fmt.Printf("   - Similarity: %.3f, Reference confidence: %.2f\n",
				result.Score, confidence)
		}
	} else {
		fmt.Printf("   ✅ Content appears safe\n")
	}
}

// Helper function to generate vectors similar to a base vector
func generateSimilarVector(base []float32, dimension int) []float32 {
	if dimension < len(base) {
		dimension = len(base)
	}

	result := make([]float32, dimension)

	// Copy base values
	for i := 0; i < len(base) && i < dimension; i++ {
		// Add small random variation
		noise := (rand.Float32() - 0.5) * 0.2
		result[i] = base[i] + noise
	}

	// Fill remaining dimensions with small random values
	for i := len(base); i < dimension; i++ {
		result[i] = (rand.Float32() - 0.5) * 0.1
	}

	return result
}
