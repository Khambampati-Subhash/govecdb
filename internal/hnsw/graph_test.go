package hnsw

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"
)

func randomVector(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()
	}
	return v
}

// bruteForceNearest returns the ids of the true k nearest vectors to query.
func bruteForceNearest(data map[string][]float32, dist DistanceFunc, query []float32, k int) []string {
	type item struct {
		id string
		d  float32
	}
	items := make([]item, 0, len(data))
	for id, v := range data {
		items = append(items, item{id, dist(v, query)})
	}
	sort.Slice(items, func(i, j int) bool { return items[i].d < items[j].d })
	if len(items) > k {
		items = items[:k]
	}
	ids := make([]string, len(items))
	for i, it := range items {
		ids[i] = it.id
	}
	return ids
}

func TestEmptyGraph(t *testing.T) {
	g, err := New(DefaultConfig(8, Cosine))
	if err != nil {
		t.Fatal(err)
	}
	if g.Len() != 0 {
		t.Fatalf("expected empty graph, got %d", g.Len())
	}
	res, err := g.Search(make([]float32, 8), 5, 10)
	if err != nil {
		t.Fatal(err)
	}
	if res != nil {
		t.Fatalf("expected nil results on empty graph, got %v", res)
	}
}

func TestValidation(t *testing.T) {
	g, _ := New(DefaultConfig(4, Euclidean))
	if err := g.Insert("a", nil); err != ErrEmptyVector {
		t.Fatalf("want ErrEmptyVector, got %v", err)
	}
	if err := g.Insert("a", []float32{1, 2, 3}); err != ErrDimensionMismatch {
		t.Fatalf("want ErrDimensionMismatch, got %v", err)
	}
}

func TestExactMatchFound(t *testing.T) {
	g, _ := New(DefaultConfig(16, Euclidean))
	rng := rand.New(rand.NewSource(42))
	var target []float32
	for i := range 500 {
		v := randomVector(rng, 16)
		id := fmt.Sprintf("v%d", i)
		if i == 250 {
			target = v
		}
		if err := g.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}
	res, err := g.Search(target, 1, 32)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != "v250" {
		t.Fatalf("expected to find exact match v250, got %+v", res)
	}
}

func TestRecallVsBruteForce(t *testing.T) {
	const (
		n   = 2000
		dim = 32
		k   = 10
		ef  = 64
	)
	rng := rand.New(rand.NewSource(7))
	cfg := DefaultConfig(dim, Cosine)
	g, _ := New(cfg)

	data := make(map[string][]float32, n)
	for i := range n {
		v := randomVector(rng, dim)
		id := fmt.Sprintf("v%d", i)
		data[id] = v
		if err := g.Insert(id, v); err != nil {
			t.Fatal(err)
		}
	}
	if g.Len() != n {
		t.Fatalf("expected %d nodes, got %d", n, g.Len())
	}

	dist := cfg.Metric.Func()
	const queries = 100
	var hits, total int
	for range queries {
		query := randomVector(rng, dim)
		got, err := g.Search(query, k, ef)
		if err != nil {
			t.Fatal(err)
		}
		want := bruteForceNearest(data, dist, query, k)
		wantSet := make(map[string]struct{}, len(want))
		for _, id := range want {
			wantSet[id] = struct{}{}
		}
		for _, r := range got {
			if _, ok := wantSet[r.ID]; ok {
				hits++
			}
		}
		total += len(want)
	}

	recall := float64(hits) / float64(total)
	t.Logf("recall@%d over %d queries: %.3f", k, queries, recall)
	if recall < 0.90 {
		t.Fatalf("recall too low: %.3f (want >= 0.90)", recall)
	}
}
