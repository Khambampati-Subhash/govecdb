// Package store holds the metadata attached to each vector.
//
// # Why metadata only, and not the vectors
//
// The obvious design is a store that owns whole records — id, values, metadata —
// with the index as a derived structure over it. That would mean two copies of
// every vector in memory, one here and one in the graph, and vectors are the
// largest thing the process holds: a million 768-dimension embeddings is 3 GB per
// copy. Paying that to keep a tidier ownership diagram is not a good trade.
//
// So values live once, in the index, and this package holds only what the index
// has no place for. The cost of that choice is real and belongs in the open: for
// a normalizing metric the index keeps the unit vector, so the magnitude a caller
// passed in is not recoverable. A cosine index is already a statement that
// magnitude is not meaningful, which is what makes the trade defensible rather
// than merely cheap.
//
// # The value types are a closed set
//
// Metadata comes off a disk that a process does not control, so decoding it is
// the one place in this database where untrusted bytes become live objects. A
// reflection-based decoder — gob, or anything that reconstructs arbitrary types
// from type names on the wire — turns that into a much larger attack surface for
// no benefit anybody asked for. Four types cover what filters need, and a decoder
// over four tags is a total function that can be read in one sitting.
package store

import (
	"maps"
	"sync"
)

// Metadata is the arbitrary key/value data attached to a vector.
//
// Values must be one of string, bool, int64 or float64. Validate enforces it at
// the boundary so a rejection is a clear error at Put rather than a surprise at
// encode time, and so nothing unencodable is ever admitted to the store.
type Metadata = map[string]any

// Store is what the database depends on, so metadata storage can be swapped —
// for something disk-backed, or for a stub in tests — without the layer above
// knowing. The methods are the whole contract: no iteration order is promised
// and none should be relied on.
type Store interface {
	// Put stores md under id, replacing anything already there. A nil or empty
	// md removes the entry rather than storing emptiness, so "no metadata" has
	// exactly one representation.
	Put(id string, md Metadata)

	// Get returns the metadata for id and whether there was any.
	Get(id string) (Metadata, bool)

	// Match evaluates pred against id's metadata and returns its answer. An id
	// with no metadata is passed a nil map rather than skipped.
	Match(id string, pred func(Metadata) bool) bool

	// Delete removes id's metadata, reporting whether there was any.
	Delete(id string) bool

	// Len is how many ids carry metadata. Not the number of vectors — a vector
	// with no metadata is not counted here.
	Len() int

	// All calls fn for each id and its metadata, in unspecified order. It stops
	// early if fn returns false.
	All(fn func(id string, md Metadata) bool)
}

// Map is an in-memory Store, safe for concurrent use.
type Map struct {
	mu sync.RWMutex
	m  map[string]Metadata
}

var _ Store = (*Map)(nil)

// New returns an empty Map.
func New() *Map {
	return &Map{m: make(map[string]Metadata)}
}

// Put stores a copy of md.
//
// The copy matters: a caller who reuses or mutates the map they passed would
// otherwise be editing stored state from outside, with no lock and no way for
// this package to notice. It is the same reason the index copies vectors on
// insert.
func (s *Map) Put(id string, md Metadata) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(md) == 0 {
		// Storing an empty map would make "absent" and "present but empty" two
		// spellings of the same thing, which every caller would then have to
		// handle. One spelling.
		delete(s.m, id)
		return
	}
	cp := make(Metadata, len(md))
	maps.Copy(cp, md)
	s.m[id] = cp
}

// Get returns a copy of the metadata under id, for the same reason Put stores
// one: a shared map would be mutable state escaping the lock.
func (s *Map) Get(id string) (Metadata, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	md, ok := s.m[id]
	if !ok {
		return nil, false
	}
	cp := make(Metadata, len(md))
	maps.Copy(cp, md)
	return cp, true
}

// Match evaluates pred against the metadata under id without copying it.
//
// Get copies because a caller keeps what it is given. This one does not, and the
// difference is the reason it exists: it is called once per candidate node
// inside a search, and copying a map per candidate would cost more than the
// distance arithmetic the search is actually there to do. The price is that pred
// runs under the read lock and borrows a map it must not retain or mutate —
// which is a contract a predicate can keep, unlike an arbitrary caller.
//
// An id with no metadata is passed a nil map rather than being skipped. "Has no
// metadata" is a thing a filter can legitimately ask about, and a nil map reads
// as empty for every operation a predicate performs on one.
func (s *Map) Match(id string, pred func(Metadata) bool) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return pred(s.m[id])
}

func (s *Map) Delete(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.m[id]; !ok {
		return false
	}
	delete(s.m, id)
	return true
}

func (s *Map) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.m)
}

// All iterates under the read lock, so fn must not call back into the store —
// doing so deadlocks. It is documented rather than defended against, because the
// alternative is copying the whole map to iterate it.
func (s *Map) All(fn func(id string, md Metadata) bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for id, md := range s.m {
		if !fn(id, md) {
			return
		}
	}
}
