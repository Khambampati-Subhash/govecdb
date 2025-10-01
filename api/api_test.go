package api_test

import (
	"context"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb/api"
)

// TestDistanceMetric tests the distance metric enum
func TestDistanceMetric(t *testing.T) {
	tests := []struct {
		metric   api.DistanceMetric
		expected string
	}{
		{api.Cosine, "cosine"},
		{api.Euclidean, "euclidean"},
		{api.Manhattan, "manhattan"},
		{api.DotProduct, "dot_product"},
		{api.DistanceMetric(999), "unknown"},
	}

	for _, test := range tests {
		if got := test.metric.String(); got != test.expected {
			t.Errorf("DistanceMetric.String() = %v, want %v", got, test.expected)
		}
	}
}

// TestVector tests vector operations
func TestVector(t *testing.T) {
	t.Run("ValidVector", func(t *testing.T) {
		vector := &api.Vector{
			ID:   "test-1",
			Data: []float32{1.0, 2.0, 3.0},
			Metadata: map[string]interface{}{
				"category": "test",
				"score":    0.95,
			},
		}

		if err := vector.Validate(); err != nil {
			t.Errorf("valid vector failed validation: %v", err)
		}

		if dim := vector.Dimension(); dim != 3 {
			t.Errorf("vector dimension = %d, want 3", dim)
		}
	})

	t.Run("InvalidVector", func(t *testing.T) {
		// Empty ID
		vector := &api.Vector{
			ID:   "",
			Data: []float32{1.0, 2.0, 3.0},
		}
		if err := vector.Validate(); err == nil {
			t.Error("empty ID should fail validation")
		}

		// Empty data
		vector = &api.Vector{
			ID:   "test-1",
			Data: []float32{},
		}
		if err := vector.Validate(); err == nil {
			t.Error("empty data should fail validation")
		}
	})

	t.Run("CloneVector", func(t *testing.T) {
		original := &api.Vector{
			ID:   "test-1",
			Data: []float32{1.0, 2.0, 3.0},
			Metadata: map[string]interface{}{
				"category": "test",
			},
		}

		clone := original.Clone()
		if clone == nil {
			t.Fatal("clone should not be nil")
		}

		if clone.ID != original.ID {
			t.Error("clone ID should match original")
		}

		if len(clone.Data) != len(original.Data) {
			t.Error("clone data length should match original")
		}

		// Modify clone and ensure original is unchanged
		clone.Data[0] = 999.0
		if original.Data[0] == 999.0 {
			t.Error("modifying clone should not affect original")
		}

		clone.Metadata["new"] = "value"
		if _, exists := original.Metadata["new"]; exists {
			t.Error("modifying clone metadata should not affect original")
		}
	})

	t.Run("NilClone", func(t *testing.T) {
		var vector *api.Vector
		clone := vector.Clone()
		if clone != nil {
			t.Error("cloning nil vector should return nil")
		}
	})
}

// TestSearchRequest tests search request validation
func TestSearchRequest(t *testing.T) {
	t.Run("ValidRequest", func(t *testing.T) {
		req := &api.SearchRequest{
			Vector:      []float32{1.0, 2.0, 3.0},
			K:           10,
			IncludeData: true,
		}

		if err := req.Validate(); err != nil {
			t.Errorf("valid request failed validation: %v", err)
		}
	})

	t.Run("InvalidRequests", func(t *testing.T) {
		// Empty vector
		req := &api.SearchRequest{
			Vector: []float32{},
			K:      10,
		}
		if err := req.Validate(); err == nil {
			t.Error("empty vector should fail validation")
		}

		// Invalid K
		req = &api.SearchRequest{
			Vector: []float32{1.0, 2.0, 3.0},
			K:      0,
		}
		if err := req.Validate(); err == nil {
			t.Error("zero K should fail validation")
		}

		// Invalid max distance
		maxDist := float32(-1.0)
		req = &api.SearchRequest{
			Vector:      []float32{1.0, 2.0, 3.0},
			K:           10,
			MaxDistance: &maxDist,
		}
		if err := req.Validate(); err == nil {
			t.Error("negative max distance should fail validation")
		}

		// Invalid min score
		minScore := float32(1.5)
		req = &api.SearchRequest{
			Vector:   []float32{1.0, 2.0, 3.0},
			K:        10,
			MinScore: &minScore,
		}
		if err := req.Validate(); err == nil {
			t.Error("min score > 1.0 should fail validation")
		}
	})
}

// TestCollectionConfig tests collection configuration
func TestCollectionConfig(t *testing.T) {
	t.Run("ValidConfig", func(t *testing.T) {
		config := &api.CollectionConfig{
			Name:           "test-collection",
			Dimension:      128,
			Metric:         api.Cosine,
			M:              16,
			EfConstruction: 200,
			MaxLayer:       16,
			Seed:           12345,
			ThreadSafe:     true,
		}

		if err := config.Validate(); err != nil {
			t.Errorf("valid config failed validation: %v", err)
		}
	})

	t.Run("InvalidConfigs", func(t *testing.T) {
		// Empty name
		config := &api.CollectionConfig{
			Name:           "",
			Dimension:      128,
			Metric:         api.Cosine,
			M:              16,
			EfConstruction: 200,
			MaxLayer:       16,
		}
		if err := config.Validate(); err == nil {
			t.Error("empty name should fail validation")
		}

		// Invalid dimension
		config = &api.CollectionConfig{
			Name:           "test",
			Dimension:      0,
			Metric:         api.Cosine,
			M:              16,
			EfConstruction: 200,
			MaxLayer:       16,
		}
		if err := config.Validate(); err == nil {
			t.Error("zero dimension should fail validation")
		}

		// Invalid metric
		config = &api.CollectionConfig{
			Name:           "test",
			Dimension:      128,
			Metric:         api.DistanceMetric(999),
			M:              16,
			EfConstruction: 200,
			MaxLayer:       16,
		}
		if err := config.Validate(); err == nil {
			t.Error("invalid metric should fail validation")
		}
	})

	t.Run("DefaultConfig", func(t *testing.T) {
		config := api.DefaultCollectionConfig("test", 128)
		if config.Name != "test" {
			t.Error("default config name mismatch")
		}
		if config.Dimension != 128 {
			t.Error("default config dimension mismatch")
		}
		if config.Metric != api.Cosine {
			t.Error("default config should use cosine metric")
		}
		if !config.ThreadSafe {
			t.Error("default config should be thread safe")
		}
	})
}

// TestFieldFilter tests field-based filtering
func TestFieldFilter(t *testing.T) {
	metadata := map[string]interface{}{
		"category": "electronics",
		"price":    99.99,
		"rating":   4.5,
		"name":     "test product",
	}

	t.Run("EqualityFilter", func(t *testing.T) {
		filter := &api.FieldFilter{
			Field: "category",
			Op:    api.FilterEq,
			Value: "electronics",
		}

		if !filter.Evaluate(metadata) {
			t.Error("equality filter should match")
		}

		filter.Value = "books"
		if filter.Evaluate(metadata) {
			t.Error("equality filter should not match")
		}
	})

	t.Run("NumericFilters", func(t *testing.T) {
		// Greater than
		filter := &api.FieldFilter{
			Field: "price",
			Op:    api.FilterGt,
			Value: 90.0,
		}
		if !filter.Evaluate(metadata) {
			t.Error("greater than filter should match")
		}

		// Less than
		filter = &api.FieldFilter{
			Field: "rating",
			Op:    api.FilterLt,
			Value: 5.0,
		}
		if !filter.Evaluate(metadata) {
			t.Error("less than filter should match")
		}
	})

	t.Run("InFilter", func(t *testing.T) {
		filter := &api.FieldFilter{
			Field: "category",
			Op:    api.FilterIn,
			Value: []interface{}{"electronics", "books", "toys"},
		}
		if !filter.Evaluate(metadata) {
			t.Error("in filter should match")
		}

		filter.Value = []interface{}{"books", "toys"}
		if filter.Evaluate(metadata) {
			t.Error("in filter should not match")
		}
	})

	t.Run("MatchFilter", func(t *testing.T) {
		filter := &api.FieldFilter{
			Field: "name",
			Op:    api.FilterMatch,
			Value: "test",
		}
		if !filter.Evaluate(metadata) {
			t.Error("match filter should match")
		}
	})

	t.Run("MissingField", func(t *testing.T) {
		filter := &api.FieldFilter{
			Field: "nonexistent",
			Op:    api.FilterEq,
			Value: "anything",
		}
		if filter.Evaluate(metadata) {
			t.Error("filter on missing field should not match")
		}
	})
}

// TestLogicalFilter tests logical filtering operations
func TestLogicalFilter(t *testing.T) {
	metadata := map[string]interface{}{
		"category": "electronics",
		"price":    99.99,
		"rating":   4.5,
	}

	t.Run("AndFilter", func(t *testing.T) {
		filter1 := &api.FieldFilter{Field: "category", Op: api.FilterEq, Value: "electronics"}
		filter2 := &api.FieldFilter{Field: "price", Op: api.FilterGt, Value: 50.0}

		andFilter := &api.LogicalFilter{
			Op:      api.FilterAnd,
			Filters: []api.FilterExpr{filter1, filter2},
		}

		if !andFilter.Evaluate(metadata) {
			t.Error("AND filter should match when both conditions are true")
		}

		filter2.Value = 150.0 // Make second condition false
		if andFilter.Evaluate(metadata) {
			t.Error("AND filter should not match when one condition is false")
		}
	})

	t.Run("OrFilter", func(t *testing.T) {
		filter1 := &api.FieldFilter{Field: "category", Op: api.FilterEq, Value: "books"}
		filter2 := &api.FieldFilter{Field: "price", Op: api.FilterGt, Value: 50.0}

		orFilter := &api.LogicalFilter{
			Op:      api.FilterOr,
			Filters: []api.FilterExpr{filter1, filter2},
		}

		if !orFilter.Evaluate(metadata) {
			t.Error("OR filter should match when one condition is true")
		}

		filter2.Value = 150.0 // Make second condition false
		if orFilter.Evaluate(metadata) {
			t.Error("OR filter should not match when both conditions are false")
		}
	})

	t.Run("NotFilter", func(t *testing.T) {
		filter := &api.FieldFilter{Field: "category", Op: api.FilterEq, Value: "books"}

		notFilter := &api.LogicalFilter{
			Op:      api.FilterNot,
			Filters: []api.FilterExpr{filter},
		}

		if !notFilter.Evaluate(metadata) {
			t.Error("NOT filter should match when condition is false")
		}

		filter.Value = "electronics"
		if notFilter.Evaluate(metadata) {
			t.Error("NOT filter should not match when condition is true")
		}
	})
}

// BenchmarkFieldFilterEvaluation benchmarks field filter evaluation
func BenchmarkFieldFilterEvaluation(b *testing.B) {
	metadata := map[string]interface{}{
		"category": "electronics",
		"price":    99.99,
		"rating":   4.5,
	}

	filter := &api.FieldFilter{
		Field: "category",
		Op:    api.FilterEq,
		Value: "electronics",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		filter.Evaluate(metadata)
	}
}

// TestConcurrentFilterEvaluation tests concurrent filter evaluation safety
func TestConcurrentFilterEvaluation(t *testing.T) {
	metadata := map[string]interface{}{
		"category": "electronics",
		"price":    99.99,
		"rating":   4.5,
	}

	filter := &api.FieldFilter{
		Field: "category",
		Op:    api.FilterEq,
		Value: "electronics",
	}

	// Run concurrent evaluations
	const numGoroutines = 10
	const numEvaluations = 100

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer func() { done <- true }()
			for j := 0; j < numEvaluations; j++ {
				select {
				case <-ctx.Done():
					return
				default:
					if !filter.Evaluate(metadata) {
						t.Errorf("filter evaluation failed in goroutine")
						return
					}
				}
			}
		}()
	}

	// Wait for all goroutines to complete
	for i := 0; i < numGoroutines; i++ {
		select {
		case <-done:
		case <-ctx.Done():
			t.Fatal("test timed out")
		}
	}
}
