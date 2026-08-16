package hnsw

// visitedList tracks "have I already seen this node during this search?".
//
// The obvious implementation is a map[int]struct{} allocated per search, which
// costs an allocation plus a hash on every probe — and searchLayer runs on
// every layer of every insert and every query, so it dominates the profile.
//
// Instead we keep one reusable array of generation stamps. Marking a node
// writes the current generation; a node counts as visited only if its stamp
// equals the current generation. Starting a new search just bumps the counter,
// so "clearing" is O(1) instead of O(n) — no allocation, no hashing.
type visitedList struct {
	marks []uint32
	gen   uint32
}

// reset prepares the list for a search over n nodes.
func (v *visitedList) reset(n int) {
	if cap(v.marks) < n {
		// Grow generously; graphs only get bigger.
		newCap := max(n, 2*cap(v.marks))
		v.marks = make([]uint32, n, newCap)
	} else {
		v.marks = v.marks[:n]
	}

	v.gen++
	// On wraparound every stale stamp could collide with the new generation,
	// so wipe once and restart. This happens after 4 billion searches.
	if v.gen == 0 {
		clear(v.marks)
		v.gen = 1
	}
}

// visit marks node idx as seen and reports whether it was already seen.
func (v *visitedList) visit(idx int) (alreadySeen bool) {
	if v.marks[idx] == v.gen {
		return true
	}
	v.marks[idx] = v.gen
	return false
}
