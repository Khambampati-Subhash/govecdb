// Package hnsw is a small, readable implementation of the Hierarchical
// Navigable Small World (HNSW) graph for approximate nearest-neighbor search.
//
// Every distance function returns a value where SMALLER MEANS CLOSER, so the
// graph can reason about "nearest" without ever branching on the metric.
//
// The package is split one responsibility per file:
//
//	config.go     Config knobs, defaults, and the sentinel errors callers see.
//	graph.go      The Graph type: state, construction, and shared helpers.
//	insert.go     Insert — building the graph.
//	search.go     Search and the layer-walking primitives it is built from.
//	neighbors.go  Edge management: selection, pruning, adjacency lookups.
//	node.go       A single vector and its per-layer neighbor lists.
//	distance.go   Metrics and their hand-unrolled kernels.
//	pq.go         Allocation-free min/max heaps over []candidate.
//	visited.go    Generation-stamped visited set, reused across searches.
package hnsw
