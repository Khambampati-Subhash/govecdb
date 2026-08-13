package hnsw

import "container/heap"

// candidate is a (node index, distance-to-query) pair used while searching.
type candidate struct {
	idx  int
	dist float32
}

// minHeap pops the CLOSEST candidate first (smallest distance). Used as the
// frontier of nodes still worth exploring during a layer search.
type minHeap []candidate

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)         { *h = append(*h, x.(candidate)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// maxHeap pops the FARTHEST candidate first (largest distance). Used to hold
// the current best results: when it overflows ef, we drop the farthest.
type maxHeap []candidate

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)         { *h = append(*h, x.(candidate)) }
func (h *maxHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// Compile-time checks that our heaps satisfy heap.Interface.
var (
	_ heap.Interface = (*minHeap)(nil)
	_ heap.Interface = (*maxHeap)(nil)
)
