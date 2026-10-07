package hnsw

import (
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
)

// InsertBatch inserts many vectors, linking them on up to workers goroutines at
// once. Zero workers means GOMAXPROCS; one is exactly a loop over Insert, which
// is what a caller wanting a reproducible graph asks for.
//
// # Why it exists
//
// Insert is the whole cost of a build, it is linear in dimension — about 2–4 ms
// a vector at 512 — and the write lock makes it one core per graph. Building a
// million vectors is minutes; rebuilding a collection to change its spec is
// tens of them. Searches parallelize because they only read; this is the same
// for the writes inside one batch.
//
// # How it stays correct
//
// The batch runs in chunks, and each chunk holds the write lock throughout, so
// no search ever sees a node half linked. Readers get the lock back between
// chunks, which bounds how long a large batch can stall them: a chunk is a few
// inserts per worker, so tens of milliseconds at high dimension rather than the
// whole batch.
//
// Inside a chunk:
//
//  1. Serially, everything that touches shared state other than edges: upserts
//     resolved (an unchanged vector skipped, a changed one tombstoned exactly as
//     Insert would), levels drawn from the graph's RNG in batch order, and
//     every node placed in its slot. g.nodes and g.ids do not change again
//     until the chunk ends, so workers read them without a lock.
//  2. If a new node reaches above the graph's top layer, it is linked alone
//     first and becomes the entry. After that no node in the chunk can promote
//     itself, so entry and maxLevel are constant and read without a lock too.
//  3. In parallel, link every other node. Neighbor lists are the only thing
//     two workers can touch at once, and those go through striped locks: a
//     traversal copies a list out under its lock, and a write — appending an
//     edge and pruning the list back to its cap — holds it for the duration. No
//     worker holds two locks, so there is no order to get wrong.
//
// Vectors are immutable once placed, and so are tombstone flags for the life of
// the chunk, so every distance and every prune ranking reads them freely.
//
// # A worker never builds on an unfinished node
//
// Every node of the chunk is marked in-flight until its own link returns, and
// parallel traversals step around in-flight nodes. That rule is what makes the
// graph as connected as a serial one, and it was found by losing vectors
// without it. A node is linked top layer first, so for a while it is findable
// on layer 1 with an empty layer-0 list; a worker whose descent lands there
// searches layer 0 from a dead end, links to one or two nodes, and the graph
// grows islands nothing points into. Measured without the rule: 17 of 10,000
// vectors unreachable at dimension 8 and 16 workers, 7 at dimension 64. With
// it: none of 710,000, across uniform and clustered data and batches into a
// live graph with deletes — the same as a serial build. TestInsertBatchStrandsNothing
// walks the graph to hold that.
//
// A repair pass — find nodes that lost their last inbound edge, re-link them —
// was written first, and deleted once this rule left it nothing to repair.
//
// # What it does not promise
//
// The graph is not the one serial inserts would build: which worker links first
// changes which edges each one sees. It is an equally good one — recall is
// measured against the serial build (TestInsertBatchRecallMatchesSerial). An id
// named twice ends with its last vector, as replaying the batch would leave it;
// the earlier one is never inserted, so it costs no tombstone either.
func (g *Graph) InsertBatch(ids []string, vectors [][]float32, workers int) error {
	if len(ids) != len(vectors) {
		return ErrBatchMismatch
	}
	for _, v := range vectors {
		if len(v) == 0 {
			return ErrEmptyVector
		}
		if len(v) != g.cfg.Dimension {
			return ErrDimensionMismatch
		}
	}
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers == 1 || len(ids) < minParallelBatch {
		for i := range ids {
			if err := g.Insert(ids[i], vectors[i]); err != nil {
				return err
			}
		}
		return nil
	}

	// Last occurrence wins, decided once for the whole batch so a duplicate
	// split across two chunks resolves the same way as one inside a chunk.
	last := make(map[string]int, len(ids))
	for i, id := range ids {
		last[id] = i
	}
	order := make([]int, 0, len(last))
	for i, id := range ids {
		if last[id] == i {
			order = append(order, i)
		}
	}

	// Copied and normalized before any lock, as Insert does.
	prepared := make([][]float32, len(vectors))
	for _, i := range order {
		prepared[i] = g.prepare(vectors[i])
	}

	for lo := 0; lo < len(order); {
		lo += g.insertChunk(ids, prepared, order[lo:], workers)
	}
	return nil
}

const (
	// minParallelBatch is the batch below which spinning up workers costs more
	// than it saves.
	minParallelBatch = 32

	// chunkPerWorker sizes a chunk: enough inserts per worker that one slow
	// insert does not leave the rest idle at the chunk's end, few enough that
	// the write lock is not held for long. Eight is ~30 ms at dimension 512.
	chunkPerWorker = 8

	// nodesPerWorker is how much live graph each concurrent linker needs:
	// below it, nodes in flight are too large a share of the graph for their
	// mutual blindness not to show in its edges. Measured as self-search
	// misses at the suggested width, building 50 to 5,000 vectors with 16
	// workers and 40% deleted: 4 missed 0.1%, 16 and 64 missed nothing in
	// 12,000 searches per cell. 32 is the smallest clean value with margin,
	// and full concurrency still arrives by about 500 live vectors.
	nodesPerWorker = 32

	// lockStripes is how many mutexes guard neighbor lists during a batch.
	// Workers lock one node at a time and rarely the same one, so collisions
	// are what a stripe count buys down; 4,096 is 32 KiB, allocated once.
	lockStripes = 4096
)

// nodeLocks guards neighbor lists while a batch links in parallel. Striped
// rather than one mutex per node, so a graph that never batches pays nothing
// and one that does pays 32 KiB rather than 8 bytes a vector.
type nodeLocks [lockStripes]sync.Mutex

func (l *nodeLocks) of(idx int) *sync.Mutex { return &l[idx%lockStripes] }

// insertChunk inserts ids[i], prepared[i] for a prefix of rest, under one hold
// of the write lock, and reports how many it consumed. See InsertBatch for the
// three phases.
func (g *Graph) insertChunk(ids []string, prepared [][]float32, rest []int, workers int) int {
	g.mu.Lock()
	defer g.mu.Unlock()

	// How many nodes may link at once is bounded by how much graph there is
	// to link into, because nodes in flight cannot see each other. Into a
	// graph of 50, sixteen at once is a third of it linking blind, and the
	// edges show it: self-searches at the suggested width missed 0.8% of the
	// time, against none serially. So a small graph grows serially, and
	// workers join as it grows (see nodesPerWorker).
	workers = min(workers, (len(g.nodes)-g.numDeleted)/nodesPerWorker)
	if workers < 2 {
		n := min(len(rest), chunkPerWorker)
		for _, i := range rest[:n] {
			if g.resolve(ids[i], prepared[i]) {
				g.insertPrepared(ids[i], prepared[i])
			}
		}
		return n
	}
	which := rest[:min(len(rest), workers*chunkPerWorker)]

	if g.locks == nil {
		g.locks = new(nodeLocks)
	}

	// Phase 1: everything but edges, serially and in batch order.
	todo := make([]int, 0, len(which))
	for _, i := range which {
		if g.resolve(ids[i], prepared[i]) {
			todo = append(todo, g.place(ids[i], prepared[i], g.randomLevel()))
		}
	}
	if len(todo) == 0 {
		return len(which)
	}

	// Phase 2: the one node that could move the entry goes first and alone.
	top := 0
	for i, idx := range todo {
		if g.nodes[idx].topLevel() > g.nodes[todo[top]].topLevel() {
			top = i
		}
	}
	if idx, level := todo[top], g.nodes[todo[top]].topLevel(); g.entry == -1 || level > g.maxLevel {
		if g.entry == -1 {
			g.entry, g.maxLevel = idx, level
		} else {
			st := g.acquireState()
			g.link(st, idx)
			g.releaseState(st)
			g.entry, g.maxLevel = idx, level
		}
		todo = slices.Delete(todo, top, top+1)
	}

	// Phase 3: link the rest in parallel. Every node starts in-flight, and
	// becomes visible to other workers' traversals once its own link is done.
	for _, idx := range todo {
		g.nodes[idx].linking.Store(true)
	}
	var (
		next atomic.Int64
		wg   sync.WaitGroup
	)
	for range min(workers, len(todo)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			st := g.acquireState()
			st.locks = g.locks
			for {
				i := int(next.Add(1)) - 1
				if i >= len(todo) {
					break
				}
				g.link(st, todo[i])
				g.nodes[todo[i]].linking.Store(false)
			}
			st.locks = nil
			g.releaseState(st)
		}()
	}
	wg.Wait()
	return len(which)
}

// resolve applies Insert's upsert rule to one id ahead of placing it: false
// for a vector the graph already holds unchanged, which costs nothing; a
// changed one has its old slot tombstoned. Callers hold the write lock.
func (g *Graph) resolve(id string, vec []float32) bool {
	if prev, exists := g.ids[id]; exists {
		if slices.Equal(g.nodes[prev].vector, vec) {
			return false
		}
		g.tombstone(id)
	}
	return true
}
