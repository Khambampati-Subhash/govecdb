package hnsw

// candidate is a (node index, distance-to-query) pair used while searching.
type candidate struct {
	idx  int
	dist float32
}

// The two priority queues below are written out by hand instead of using
// container/heap. That package works through the heap.Interface methods, so
// every Push/Pop passes a candidate as `any` — which boxes the struct on the
// heap and cost ~765 allocations per search in benchmarks. These operate on a
// []candidate directly: no interfaces, no boxing, no allocation beyond the
// slice's own amortized growth.
//
// Both return the (possibly reallocated) slice, so callers must reassign:
//
//	h = minPush(h, c)
//	c, h = minPop(h)

// minPush inserts c into a min-heap ordered by ascending distance.
func minPush(h []candidate, c candidate) []candidate {
	h = append(h, c)
	i := len(h) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if h[parent].dist <= h[i].dist {
			break
		}
		h[parent], h[i] = h[i], h[parent]
		i = parent
	}
	return h
}

// minPop removes and returns the closest candidate. The heap must be non-empty.
func minPop(h []candidate) (candidate, []candidate) {
	top := h[0]
	n := len(h) - 1
	h[0] = h[n]
	h = h[:n]

	i := 0
	for {
		l, r := 2*i+1, 2*i+2
		smallest := i
		if l < n && h[l].dist < h[smallest].dist {
			smallest = l
		}
		if r < n && h[r].dist < h[smallest].dist {
			smallest = r
		}
		if smallest == i {
			break
		}
		h[i], h[smallest] = h[smallest], h[i]
		i = smallest
	}
	return top, h
}

// maxPush inserts c into a max-heap ordered by descending distance, so h[0] is
// always the farthest result held so far.
func maxPush(h []candidate, c candidate) []candidate {
	h = append(h, c)
	i := len(h) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if h[parent].dist >= h[i].dist {
			break
		}
		h[parent], h[i] = h[i], h[parent]
		i = parent
	}
	return h
}

// maxPop removes and returns the farthest candidate. The heap must be non-empty.
func maxPop(h []candidate) (candidate, []candidate) {
	top := h[0]
	n := len(h) - 1
	h[0] = h[n]
	h = h[:n]

	i := 0
	for {
		l, r := 2*i+1, 2*i+2
		largest := i
		if l < n && h[l].dist > h[largest].dist {
			largest = l
		}
		if r < n && h[r].dist > h[largest].dist {
			largest = r
		}
		if largest == i {
			break
		}
		h[i], h[largest] = h[largest], h[i]
		i = largest
	}
	return top, h
}
