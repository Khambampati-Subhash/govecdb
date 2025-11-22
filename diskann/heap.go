package diskann

import "container/heap"

// Item represents an item in the priority queue
type Item struct {
	ID    uint32
	Dist  float32
	Index int // The index of the item in the heap
}

// MinHeap implements a min-heap for Items
type MinHeap []*Item

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].Dist < h[j].Dist }
func (h MinHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].Index = i
	h[j].Index = j
}

func (h *MinHeap) Push(x interface{}) {
	n := len(*h)
	item := x.(*Item)
	item.Index = n
	*h = append(*h, item)
}

func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil  // avoid memory leak
	item.Index = -1 // for safety
	*h = old[0 : n-1]
	return item
}

// NewMinHeap creates a new min heap
func NewMinHeap() *MinHeapWrapper {
	h := &MinHeap{}
	heap.Init(h)
	return &MinHeapWrapper{h}
}

// MinHeapWrapper wraps the heap interface for easier usage
type MinHeapWrapper struct {
	h *MinHeap
}

func (w *MinHeapWrapper) Push(id uint32, dist float32) {
	heap.Push(w.h, &Item{ID: id, Dist: dist})
}

func (w *MinHeapWrapper) Pop() (uint32, float32) {
	if w.h.Len() == 0 {
		return 0, -1
	}
	item := heap.Pop(w.h).(*Item)
	return item.ID, item.Dist
}

func (w *MinHeapWrapper) Len() int {
	return w.h.Len()
}
