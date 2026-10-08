package govecdb

import (
	"container/heap"
	"fmt"
	"slices"
	"strings"
)

// Reading many vectors at once. Get answers for an id the caller already has;
// these exist because everything rebuild-shaped — resharding, re-embedding,
// copying a collection under a new spec — starts from "what is in here?", and
// without them the only answer was a ledger kept outside the database.
//
// # Weakly consistent, on purpose
//
// None of these holds a lock across the whole walk. Enumerating a million ids
// while excluding writers would stall every one of them for the length of the
// scan, and Range in particular hands control back to the caller between pages,
// which may be far longer. So each page is read under its own short hold of
// applyMu: a vector deleted while a scan is in progress is skipped if the scan
// had not reached it yet, and one added mid-scan may or may not appear. What a
// page does promise is that every vector in it existed, with exactly those
// values and that metadata, at one instant — values and metadata are read under
// one hold, the same rule Get follows.

// rangePage is how many vectors Range reads per lock acquisition. Large enough
// to amortize the lock, small enough that a writer waiting behind a page waits
// for microseconds, not milliseconds.
const rangePage = 256

// GetBatch returns the vectors stored under ids, in the order asked for. Ids
// that are not in the database are skipped rather than reported as an error —
// the result is shorter than ids by exactly the number that were missing, and a
// caller that needs to know which compares ids against the result.
//
// The batch is read under one hold of applyMu, so it is a consistent picture: a
// concurrent Add or Delete lands entirely before it or entirely after. An
// AddBatch is applied in groups (see applyGroup), so a GetBatch running beside
// one can see some of its vectors and not others — but never a vector without
// its own metadata.
func (db *DB) GetBatch(ids []string) ([]Vector, error) {
	if len(ids) > db.opts.maxBatch {
		return nil, fmt.Errorf("%w: batch of %d ids, max %d", ErrInvalidRequest, len(ids), db.opts.maxBatch)
	}
	if db.closed.Load() {
		return nil, ErrClosed
	}
	return db.lookup(ids), nil
}

// Scan returns up to limit vectors whose ids sort strictly after `after`, in
// ascending byte order of id. Pass "" to start at the beginning and the last id
// of each page to continue; a page shorter than limit is the last one.
//
// It is the paging primitive — what an HTTP listing is built on, where a client
// holds the cursor between requests and nothing can be held open for it. Each
// call costs a pass over the ids, O(N log limit), because the index keeps no
// sorted order to seek into: keeping one would put a cost on every insert to
// make an occasional enumeration cheaper. A caller walking everything in one
// process should use Range, which pays the sort once.
func (db *DB) Scan(after string, limit int) ([]Vector, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("%w: limit is %d, want at least 1", ErrInvalidRequest, limit)
	}
	if limit > db.opts.maxBatch {
		return nil, fmt.Errorf("%w: limit is %d, max %d", ErrInvalidRequest, limit, db.opts.maxBatch)
	}

	idx, err := db.indexHandle()
	if err != nil {
		return nil, err
	}

	// A max-heap of the smallest `limit` ids past the cursor: the root is the
	// largest kept, so a new id only displaces it when it sorts lower.
	h := &idHeap{}
	idx.IDs(func(id string) bool {
		if id <= after {
			return true
		}
		if h.Len() < limit {
			heap.Push(h, id)
		} else if id < (*h)[0] {
			(*h)[0] = id
			heap.Fix(h, 0)
		}
		return true
	})
	ids := []string(*h)
	slices.Sort(ids)

	if db.closed.Load() {
		return nil, ErrClosed
	}
	return db.lookup(ids), nil
}

// Range calls fn for every vector in the database, in ascending byte order of
// id, stopping early if fn returns false. It reports ErrClosed if the database
// was closed before or during the walk.
//
// fn is called without any lock held, so it may do anything — including write
// to this database. A vector fn writes is not guaranteed to be visited again;
// see the note on consistency at the top of this file.
//
// Memory is one string header per live id, for the sorted list it walks; the
// vectors themselves are read a page at a time.
func (db *DB) Range(fn func(Vector) bool) error {
	idx, err := db.indexHandle()
	if err != nil {
		return err
	}
	ids := make([]string, 0, idx.Len())
	idx.IDs(func(id string) bool {
		ids = append(ids, id)
		return true
	})
	slices.Sort(ids)

	for page := range slices.Chunk(ids, rangePage) {
		if db.closed.Load() {
			return ErrClosed
		}
		vs := db.lookup(page)

		for _, v := range vs {
			if !fn(v) {
				return nil
			}
		}
	}
	return nil
}

// indexHandle returns the index of an open database. The index is never
// replaced after Open, so the handle needs no lock; the closed check is what
// it is for.
func (db *DB) indexHandle() (Index, error) {
	if db.closed.Load() {
		return nil, ErrClosed
	}
	return db.index, nil
}

// lookup reads each id's values and metadata, skipping ids that are no longer
// live. One shared hold of applyMu covers them all, which is what makes values
// and metadata agree.
func (db *DB) lookup(ids []string) []Vector {
	db.applyMu.RLock()
	defer db.applyMu.RUnlock()

	out := make([]Vector, 0, len(ids))
	for _, id := range ids {
		values, ok := db.index.Lookup(id)
		if !ok {
			continue
		}
		md, _ := db.store.Get(id)
		out = append(out, Vector{ID: id, Values: values, Metadata: md})
	}
	return out
}

// idHeap is a max-heap of ids, for keeping the lowest k of a stream.
type idHeap []string

func (h idHeap) Len() int           { return len(h) }
func (h idHeap) Less(i, j int) bool { return strings.Compare(h[i], h[j]) > 0 }
func (h idHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *idHeap) Push(x any)        { *h = append(*h, x.(string)) }
func (h *idHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}
