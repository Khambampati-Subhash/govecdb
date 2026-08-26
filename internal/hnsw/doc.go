// Package hnsw is a small, readable implementation of the Hierarchical
// Navigable Small World (HNSW) graph for approximate nearest-neighbor search.
//
// Every distance function returns a value where SMALLER MEANS CLOSER, so the
// graph can reason about "nearest" without ever branching on the metric.
//
// A Graph is safe for concurrent use: searches run in parallel under a read
// lock, inserts take the write lock, and all per-traversal scratch comes from a
// pool so no two callers share mutable state.
//
// Deletion is by tombstone. A deleted slot keeps its index and its edges, so
// searches still route *through* it, but it never appears in a result. That
// keeps every neighbor list valid and keeps the graph connected; the cost is
// memory and traversal work that only compaction reclaims.
//
// Insert is an upsert: a second Insert under a live id tombstones the old slot
// and builds a new one, because the old slot's inbound edges were chosen for the
// old vector and would misroute searches if the vector underneath them changed.
// So updates pay into the same tombstone debt that deletes do.
//
// The package is split one responsibility per file:
//
//	config.go     Config knobs, defaults, and the sentinel errors callers see.
//	graph.go      The Graph type: state, construction, and shared helpers.
//	insert.go     Insert — building the graph, and replacing an existing id.
//	delete.go     Delete — tombstoning, and re-electing the entry point.
//	search.go     Search and the layer-walking primitives it is built from.
//	neighbors.go  Edge management: selection, pruning, adjacency lookups.
//	node.go       A single vector and its per-layer neighbor lists.
//	distance.go   Metrics and their hand-unrolled kernels.
//	pq.go         Allocation-free min/max heaps over []candidate.
//	visited.go    Generation-stamped visited set, reused across searches.
//	state.go      Pooled per-traversal scratch; why Search can hold RLock.
package hnsw
