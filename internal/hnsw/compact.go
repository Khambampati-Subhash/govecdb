package hnsw

import (
	"runtime"
	"slices"
	"strings"
)

// Compact rebuilds the graph over its live vectors and reports how many slots
// were reclaimed. Calling it on a graph with no tombstones is free.
//
// Tombstones never release memory on their own — that is the bargain Delete
// makes, and Insert makes it too, once per replaced vector. A workload that
// re-embeds the same corpus therefore grows without bound while its live
// population stays flat: three passes over 300 vectors leave 300 live vectors
// carried by 1,200 slots. Compaction is what pays that debt back, and it is the
// *only* thing that does.
//
// # Why a rebuild and not a repair
//
// Every neighbor list in the graph is a list of indices into g.nodes, so a slot
// cannot be removed in place: dropping one shifts every index above it and
// invalidates every list in the graph at once. Repairing them in place is the
// same O(N·M) scan that made in-place updates unavailable, except now it has to
// run for every dead slot rather than one.
//
// So compaction does not renumber the live graph — it builds a *new* graph from
// the live vectors and swaps it in whole. The slot-index invariant is never
// violated; it is retired along with the graph that held it.
//
// # What the rebuild produces
//
// Exactly the graph you would have built if the dead vectors had never existed:
// the replacement draws its levels from a fresh RNG seeded from the same config,
// and the live vectors are re-inserted in slot order, so the result is
// deterministic and testable by equality rather than by sampling
// (TestCompactMatchesAFreshBuild). That holds when no write lands during the
// compaction; one that does is caught up afterwards, below, and the graph is
// then an equally good one rather than the identical one.
//
// It is also a *better* graph than the one it replaces, not merely a smaller
// one. Edges that pointed at tombstones become edges between live nodes, and
// the pruning bound in searchLayer tightens again because results fill at full
// speed. Recall goes up and search gets faster; only the memory is the headline.
//
// # Searches and writes keep running
//
// The replacement is built holding no lock on this graph at all. That is safe
// because what it reads — a node's id and stored vector — is written once in
// newNode and never again: an update builds a new slot rather than editing the
// old one. The live nodes are captured under a short read lock, and the build
// runs from that list while searches and writes carry on against the old graph.
//
// A write the build cannot see is not lost. While a compaction runs, place and
// tombstone record the id they touched (dirty), and the compaction catches up:
// it drains those ids under a short read lock, and sets each one in the
// replacement to whatever the old graph holds for it *now* — its vector, or
// nothing. Applying the final state rather than replaying the writes is what
// makes order irrelevant: an id written five times is caught up once, and an
// id inserted and deleted again is a no-op. The catch-up repeats while writes
// keep arriving, each round shorter than the last, since it runs on several
// workers against a writer that runs on one.
//
// Everything up to the swap runs under the read lock, never the write lock, and
// that is not an economy. A database snapshot holds the read lock for seconds
// while it streams the graph out; a compaction waiting for the write lock
// behind it would hold every new search until the snapshot ended, because Go's
// RWMutex blocks new readers once a writer waits. The read lock excludes
// exactly what the drain needs excluded — writers — and nothing else.
//
// The swap is the only step that must see no further write. It takes the write
// lock, drains whatever arrived since the last round, applies it, and swaps —
// so a standalone caller's writers wait for at most one round's worth of
// catch-up rather than the whole rebuild. CompactOnline lets a caller that
// controls its own writers make even that a constant.
//
// So the index does not decide *when*. It exposes Stats().DeadRatio() and lets
// the caller pick a moment that tolerates the cost.
func (g *Graph) Compact() int {
	return g.CompactOnline(nil)
}

// CompactOnline is Compact for a caller that can hold its own writers off.
// Just before the final catch-up and the swap it calls quiesce, which must
// return only once no further write will reach the graph; the func quiesce
// returns is called once the swap is done. Writers then wait for one short
// catch-up instead of a round of it under the graph's write lock — and, unlike
// waiting on that lock, they do not hold searches up behind them while they
// wait. quiesce is not called when there is nothing to compact.
//
// A nil quiesce is Compact.
func (g *Graph) CompactOnline(quiesce func() (resume func())) int {
	g.compactMu.Lock()
	defer g.compactMu.Unlock()

	live := g.beginCompaction()
	if live == nil {
		return 0
	}
	if g.onCompactBuild != nil {
		g.onCompactBuild()
	}
	fresh := newGraph(g.cfg)
	for _, n := range live {
		// The vector moves across as-is rather than being re-prepared: it is
		// already this graph's own normalized copy, and re-normalizing a unit
		// vector drifts it by an ulp. Dead vectors are the memory being
		// reclaimed — nothing references them once the swap lands.
		fresh.insertPrepared(n.id, n.vector)
	}

	workers := runtime.GOMAXPROCS(0)
	for range catchUpRounds {
		if g.catchUp(fresh, workers) <= catchUpQuiet {
			break
		}
	}
	if quiesce != nil {
		resume := quiesce()
		defer resume()
		g.catchUp(fresh, workers)
	}
	return g.swap(fresh, workers)
}

const (
	// catchUpQuiet is a drain small enough to finish at the swap. The swap
	// holds the write lock, so this bounds how long a standalone compaction
	// stops searches for — a few hundred inserts rather than the rebuild.
	catchUpQuiet = 256

	// catchUpRounds bounds the rounds spent waiting for writers to go quiet.
	// Each round runs on every core against writers that serialize on the
	// graph's lock, so they shrink fast; a writer that somehow keeps pace is
	// cut off here rather than chased forever, and its last round lands at the
	// swap.
	catchUpRounds = 8
)

// beginCompaction starts recording writes and captures the live nodes in slot
// order, or returns nil when there is nothing to reclaim.
//
// Both happen under one read lock, so no write falls between them: every
// write is either in the capture or recorded after it. Recording after the
// capture would lose the writes in between.
func (g *Graph) beginCompaction() []*node {
	g.mu.RLock()
	defer g.mu.RUnlock()
	// Cheap enough to call on a timer: a clean graph is not worth rebuilding,
	// and rebuilding it would throw away a perfectly good index for nothing.
	if g.numDeleted == 0 {
		return nil
	}
	g.dirty = make(map[string]struct{})
	live := make([]*node, 0, len(g.nodes)-g.numDeleted)
	for _, n := range g.nodes {
		if !n.deleted {
			live = append(live, n)
		}
	}
	return live
}

// touch records that a write changed id, for a compaction in flight. place and
// tombstone are the two places a write reaches the id map, so between them
// they see every one. Callers hold the write lock — which is what lets the
// compaction drain dirty under the read lock.
func (g *Graph) touch(id string) {
	if g.dirty != nil {
		g.dirty[id] = struct{}{}
	}
}

// pendingWrite is an id's state in the graph being compacted: its stored
// vector, or nil when the id is no longer bound.
type pendingWrite struct {
	id  string
	vec []float32
}

// catchUp applies to fresh every write the graph has taken since the last
// drain, and reports how many ids that was. Only the drain holds the lock, and
// only the read lock (see Compact for why not the write lock).
func (g *Graph) catchUp(fresh *Graph, workers int) int {
	g.mu.RLock()
	writes := g.drain()
	g.mu.RUnlock()
	fresh.applyCompacted(writes, workers)
	return len(writes)
}

// swap brings fresh fully up to date and makes it this graph, reporting how
// many slots that freed.
func (g *Graph) swap(fresh *Graph, workers int) int {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Empty when the caller quiesced; at most one round's worth otherwise.
	fresh.applyCompacted(g.drain(), workers)

	// fresh may hold tombstones of its own, from updates and deletes it was
	// caught up with, and those are not reclaimed — the old graph made the
	// same ones.
	reclaimed := g.numDeleted - fresh.numDeleted
	g.nodes, g.ids, g.layer0 = fresh.nodes, fresh.ids, fresh.layer0
	g.entry, g.maxLevel = fresh.entry, fresh.maxLevel
	g.numDeleted = fresh.numDeleted
	g.dirty = nil
	g.layout++

	// g.pool is deliberately not swapped. Its pooled scratch was sized for the
	// larger graph, which is safe — visitedList.reset re-slices down and its
	// generation counter only ever increases, so a stale stamp can never match
	// the current generation — and the extra capacity dies with the graph.
	// g.locks is not swapped either: a stripe guards whatever slot hashes to
	// it, in this graph or the next.
	return reclaimed
}

// drain empties dirty and returns each id's current state, sorted by id so
// that catching up is reproducible rather than map-ordered. Callers hold
// either lock.
func (g *Graph) drain() []pendingWrite {
	if len(g.dirty) == 0 {
		return nil
	}
	writes := make([]pendingWrite, 0, len(g.dirty))
	for id := range g.dirty {
		w := pendingWrite{id: id}
		if idx, ok := g.ids[id]; ok {
			w.vec = g.nodes[idx].vector
		}
		writes = append(writes, w)
	}
	clear(g.dirty)
	slices.SortFunc(writes, func(a, b pendingWrite) int { return strings.Compare(a.id, b.id) })
	return writes
}

// applyCompacted sets each id in a replacement graph to the state given for
// it. The ids are distinct, so deletes and puts commute and the puts can go
// through the batch path. Nothing else can see the replacement yet, so its
// lock is uncontended.
func (g *Graph) applyCompacted(writes []pendingWrite, workers int) {
	ids := make([]string, 0, len(writes))
	vecs := make([][]float32, 0, len(writes))
	order := make([]int, 0, len(writes))
	for _, w := range writes {
		if w.vec == nil {
			g.Delete(w.id)
			continue
		}
		order = append(order, len(ids))
		ids = append(ids, w.id)
		vecs = append(vecs, w.vec)
	}
	if len(ids) < minParallelBatch {
		workers = 1
	}
	g.insertPreparedBatch(ids, vecs, order, workers)
}
