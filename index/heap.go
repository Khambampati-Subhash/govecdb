package index

// MinNodeHeap is a specialized min-heap for NodeCandidate
type MinNodeHeap struct {
	items []*NodeCandidate
}

// NewMinNodeHeap creates a new min-heap with optional initial capacity
func NewMinNodeHeap(capacity int) *MinNodeHeap {
	return &MinNodeHeap{
		items: make([]*NodeCandidate, 0, capacity),
	}
}

// Len returns the number of items in the heap
func (h *MinNodeHeap) Len() int {
	return len(h.items)
}

// Push adds an item to the heap
func (h *MinNodeHeap) Push(item *NodeCandidate) {
	h.items = append(h.items, item)
	h.up(len(h.items) - 1)
}

// Pop removes and returns the minimum item from the heap
func (h *MinNodeHeap) Pop() *NodeCandidate {
	n := len(h.items) - 1
	if n < 0 {
		return nil
	}
	h.swap(0, n)
	item := h.items[n]
	h.items[n] = nil // Avoid memory leak
	h.items = h.items[:n]
	h.down(0, n)
	return item
}

// Peek returns the minimum item without removing it
func (h *MinNodeHeap) Peek() *NodeCandidate {
	if len(h.items) == 0 {
		return nil
	}
	return h.items[0]
}

// Reset clears the heap
func (h *MinNodeHeap) Reset() {
	// Clear items to allow GC
	for i := range h.items {
		h.items[i] = nil
	}
	h.items = h.items[:0]
}

func (h *MinNodeHeap) swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
}

func (h *MinNodeHeap) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || h.items[j].Distance >= h.items[i].Distance {
			break
		}
		h.swap(i, j)
		j = i
	}
}

func (h *MinNodeHeap) down(i0, n int) bool {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := j1 // left child
		j2 := j1 + 1
		if j2 < n && h.items[j2].Distance < h.items[j1].Distance {
			j = j2 // = 2*i + 2  // right child
		}
		if h.items[j].Distance >= h.items[i].Distance {
			break
		}
		h.swap(i, j)
		i = j
	}
	return i > i0
}

// MaxNodeHeap is a specialized max-heap for NodeCandidate
type MaxNodeHeap struct {
	items []*NodeCandidate
}

// NewMaxNodeHeap creates a new max-heap with optional initial capacity
func NewMaxNodeHeap(capacity int) *MaxNodeHeap {
	return &MaxNodeHeap{
		items: make([]*NodeCandidate, 0, capacity),
	}
}

// Len returns the number of items in the heap
func (h *MaxNodeHeap) Len() int {
	return len(h.items)
}

// Push adds an item to the heap
func (h *MaxNodeHeap) Push(item *NodeCandidate) {
	h.items = append(h.items, item)
	h.up(len(h.items) - 1)
}

// Pop removes and returns the maximum item from the heap
func (h *MaxNodeHeap) Pop() *NodeCandidate {
	n := len(h.items) - 1
	if n < 0 {
		return nil
	}
	h.swap(0, n)
	item := h.items[n]
	h.items[n] = nil
	h.items = h.items[:n]
	h.down(0, n)
	return item
}

// Peek returns the maximum item without removing it
func (h *MaxNodeHeap) Peek() *NodeCandidate {
	if len(h.items) == 0 {
		return nil
	}
	return h.items[0]
}

// Reset clears the heap
func (h *MaxNodeHeap) Reset() {
	for i := range h.items {
		h.items[i] = nil
	}
	h.items = h.items[:0]
}

func (h *MaxNodeHeap) swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
}

func (h *MaxNodeHeap) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || h.items[j].Distance <= h.items[i].Distance {
			break
		}
		h.swap(i, j)
		j = i
	}
}

func (h *MaxNodeHeap) down(i0, n int) bool {
	i := i0
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 {
			break
		}
		j := j1 // left child
		j2 := j1 + 1
		if j2 < n && h.items[j2].Distance > h.items[j1].Distance {
			j = j2 // right child
		}
		if h.items[j].Distance <= h.items[i].Distance {
			break
		}
		h.swap(i, j)
		i = j
	}
	return i > i0
}
