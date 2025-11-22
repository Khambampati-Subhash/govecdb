package index

import (
	"sync"
)

// SearchContext holds reusable resources for search operations
type SearchContext struct {
	visited    map[string]bool
	candidates *MinNodeHeap
	dynamic    *MaxNodeHeap
}

var searchContextPool = sync.Pool{
	New: func() interface{} {
		return &SearchContext{
			visited:    make(map[string]bool),
			candidates: NewMinNodeHeap(100),
			dynamic:    NewMaxNodeHeap(100),
		}
	},
}

// GetSearchContext retrieves a context from the pool
func GetSearchContext() *SearchContext {
	ctx := searchContextPool.Get().(*SearchContext)
	// Reset maps and heaps
	for k := range ctx.visited {
		delete(ctx.visited, k)
	}
	ctx.candidates.Reset()
	ctx.dynamic.Reset()
	return ctx
}

// PutSearchContext returns a context to the pool
func PutSearchContext(ctx *SearchContext) {
	searchContextPool.Put(ctx)
}
