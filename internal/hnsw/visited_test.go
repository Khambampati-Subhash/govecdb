package hnsw

import (
	"math"
	"testing"
)

// visitedList trades an allocation per search for a generation counter, which
// moves the correctness argument from "the map is fresh" to "a stale stamp can
// never equal the current generation". These tests are that argument, including
// the wraparound branch that real traffic reaches once every four billion
// searches and no other test can reach at all.

func TestVisitedMarksAndClears(t *testing.T) {
	var v visitedList
	v.reset(10)

	for i := range 10 {
		if v.visit(i) {
			t.Fatalf("node %d reported as already seen in a fresh generation", i)
		}
	}
	for i := range 10 {
		if !v.visit(i) {
			t.Fatalf("node %d was marked but does not report as seen", i)
		}
	}

	// A new generation clears everything without touching the array.
	v.reset(10)
	for i := range 10 {
		if v.visit(i) {
			t.Fatalf("node %d survived a reset", i)
		}
	}
}

// TestVisitedReusesCapacityAcrossSizes covers the pooled path: one searchState
// can be handed to graphs of different sizes, so reset must both grow and shrink
// without ever exposing a stale mark.
func TestVisitedReusesCapacityAcrossSizes(t *testing.T) {
	var v visitedList

	v.reset(1000)
	for i := range 1000 {
		v.visit(i)
	}
	grown := cap(v.marks)

	// Shrink: a smaller graph must not see marks written by the larger one.
	v.reset(10)
	if len(v.marks) != 10 {
		t.Fatalf("reset(10) left len %d", len(v.marks))
	}
	if cap(v.marks) != grown {
		t.Fatalf("reset shed capacity: %d, want %d retained", cap(v.marks), grown)
	}
	for i := range 10 {
		if v.visit(i) {
			t.Fatalf("node %d carried a mark across a shrink", i)
		}
	}

	// Grow back into the same array: still no stale marks.
	v.reset(1000)
	for i := range 1000 {
		if v.visit(i) {
			t.Fatalf("node %d carried a mark across a regrow", i)
		}
	}
}

// TestVisitedGenerationWraparound is the once-in-four-billion branch. When gen
// laps back to zero, every stale stamp in the array would suddenly match the
// current generation and the search would treat the entire graph as already
// visited — returning almost nothing, silently. Reaching it honestly would take
// 2^32 searches, so the counter is driven to the edge directly.
func TestVisitedGenerationWraparound(t *testing.T) {
	var v visitedList
	v.reset(100)

	// Stamp the array with the generation that is about to be reused.
	v.gen = math.MaxUint32
	for i := range 100 {
		v.marks[i] = math.MaxUint32
	}

	// This reset increments to 0, detects the lap, and must wipe.
	v.reset(100)
	if v.gen == 0 {
		t.Fatal("generation left at 0 after wraparound; every stamp would be ambiguous")
	}
	for i := range 100 {
		if v.visit(i) {
			t.Fatalf("node %d looked visited after wraparound — the wipe did not happen", i)
		}
	}

	// And the list keeps working normally afterwards.
	for i := range 100 {
		if !v.visit(i) {
			t.Fatalf("node %d lost its mark after wraparound", i)
		}
	}
	v.reset(100)
	if v.visit(0) {
		t.Fatal("reset after wraparound failed to clear")
	}
}

// TestVisitedZeroSize guards the empty-graph path: Search on a graph with no
// nodes resets to zero length, which must not panic on the slicing.
func TestVisitedZeroSize(t *testing.T) {
	var v visitedList
	v.reset(0)
	if len(v.marks) != 0 {
		t.Fatalf("reset(0) left len %d", len(v.marks))
	}
	v.reset(4)
	if v.visit(3) {
		t.Fatal("mark survived from a zero-length reset")
	}
}
