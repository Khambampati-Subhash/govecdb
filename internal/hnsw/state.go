package hnsw

// searchState is the scratch space one traversal needs: a visited set, the two
// heaps searchLayer runs on, the reject list selectNeighbors backfills from, and
// a buffer for the normalized query.
//
// These used to live on Graph, which made Search a *writer*: every call stamped
// the visited array and rewrote the heap slices. An RWMutex would have been a
// lie — two readers holding RLock would have raced over the same scratch. Moving
// it here is what makes "Search only reads the graph" true, and therefore what
// makes concurrent search possible at all.
//
// The state is pooled rather than allocated per call because it is exactly the
// thing the zero-allocation search path was built to avoid: a 40 KB visited
// array plus three heaps would dwarf the 2 allocations a search costs today.
type searchState struct {
	visited  visitedList
	cands    []candidate // min-heap: the frontier still worth expanding
	results  []candidate // max-heap: the best ef found so far, worst at [0]
	found    []candidate // searchLayer's sorted output; valid until its next call
	rejected []candidate // selectNeighbors' backfill list
	pruneBuf []candidate // pruneConnections' distance-annotated neighbors
	selected []int       // insert's chosen neighbors for the current layer
	queryBuf []float32   // normalized copy of the caller's query
	nbrBuf   []int       // a neighbor list copied out under its lock

	// locks is non-nil only for a worker of a parallel batch (batch.go), and
	// it is what switches neighbor-list access to locked copies. Nil for every
	// search and every serial insert, so those pay one branch per expansion
	// and nothing else. Cleared before the state goes back to the pool.
	locks *nodeLocks

	// hide is 1 + the slot searchLayer must treat as absent, or 0 for none.
	// Only Calibrate sets it — see searchWithout — and it is cleared before
	// the state goes back to the pool.
	hide int
}

// acquireState borrows scratch space for one traversal. Every path out must
// release it, so callers pair this with a deferred releaseState.
func (g *Graph) acquireState() *searchState {
	return g.pool.Get().(*searchState)
}

// releaseState returns scratch space to the pool. The buffers keep their
// capacity, which is the whole point: the next search reuses them.
func (g *Graph) releaseState(st *searchState) {
	g.pool.Put(st)
}
