package hnsw

// Delete tombstones the vector stored under id and reports whether anything was
// removed. The slot keeps its index and its edges — it stays a waypoint — but it
// can never appear in a search result again.
//
// Deleting an unknown id is a no-op, not an error, which makes Delete
// idempotent. That is deliberate rather than lenient: WAL recovery replays
// records, and an operation that failed the second time it was applied would
// make replay order-sensitive.
//
// # Why a tombstone rather than a real removal
//
// Two reasons, and the second is the one that actually forces the design:
//
//  1. Slots are addressed by index. Removing one shifts every index above it,
//     invalidating every neighbor list in the graph — an O(N·M) repair for a
//     single delete.
//  2. A node is a *bridge*. HNSW reaches a region by hopping through whatever
//     nodes lie between, and those hops do not care whether the waypoint is
//     still wanted. Cut one out and a whole neighborhood can become
//     unreachable — vectors nobody deleted, lost.
//
// The price is that dead slots keep their memory and keep being traversed.
// Compaction reclaims them; see the note in searchLayer for how the cost shows
// up in the meantime.
//
// Safe to call concurrently; it takes the write lock.
func (g *Graph) Delete(id string) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.tombstone(id)
}

// tombstone unbinds id and marks its slot dead, reporting whether there was
// anything to kill. Delete is the standalone form; Insert calls this directly
// when it replaces an existing id, so that the tombstone and the new slot
// commit under a *single* write lock. Going through Delete would open a window
// in which the id belongs to nobody, and a concurrent reader would see an
// update as a disappearance.
//
// Callers hold the write lock.
func (g *Graph) tombstone(id string) bool {
	idx, ok := g.ids[id]
	if !ok {
		return false
	}

	// Drop the id binding but keep the slot. Re-inserting this id afterwards
	// therefore allocates a *new* slot rather than resurrecting this one — the
	// old edges were chosen for the old vector and would be wrong for a new one.
	delete(g.ids, id)
	g.nodes[idx].deleted = true
	g.numDeleted++

	if g.entry == idx {
		g.reelectEntry()
	}
	return true
}

// reelectEntry restores the two invariants that a deleted entry point breaks:
// the entry must be live, and it must sit at maxLevel (Insert's descent walks
// down from maxLevel starting at entry, so a shorter entry would index past the
// end of its own neighbor slice).
//
// Both are restored by the same choice: the live node with the highest top
// level. Layers above it are left populated but unreachable, which is harmless
// — only tombstones live up there now.
//
// The O(N) scan is affordable because it is rare: it costs a pass only when the
// one specific node that happens to be the entry point is deleted. Maintaining
// a level-indexed structure to avoid it would add a permanent write-path cost
// to make a rare path cheap.
//
// Callers hold the write lock.
func (g *Graph) reelectEntry() {
	best, bestLevel := -1, -1
	for i, n := range g.nodes {
		if n.deleted {
			continue
		}
		if lvl := n.topLevel(); lvl > bestLevel {
			best, bestLevel = i, lvl
		}
	}

	if best == -1 {
		// Every slot is a tombstone. Reset to the empty-graph state so the next
		// Insert takes the first-node path and starts a fresh graph; the orphaned
		// tombstones stay addressable until compaction clears them.
		g.entry, g.maxLevel = -1, 0
		return
	}
	g.entry, g.maxLevel = best, bestLevel
}
