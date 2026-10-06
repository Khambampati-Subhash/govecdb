package store

// Interning: one copy of each distinct metadata string, shared by every entry
// that holds it.
//
// Metadata repeats. Every vector carries the same key names, and every chunk of
// one document carries the same document id, source path and title — dozens to
// thousands of identical strings, each its own allocation, plus a 16-byte box for
// each one stored in an `any`. Measured over 200,000 vectors with eleven keys
// and fifty chunks to a document, the store held 1,141 bytes a vector; interned,
// 754 — a third less, on a structure that was a fifth of a large collection's
// memory. Where every value is unique the key names alone pay for the table:
// 1,143 against 1,147.
//
// Both the string and its box are shared, so storing an interned value in a map
// allocates nothing. Entries are reference-counted and dropped when the last
// metadata map using them goes, so deleting a document's vectors frees its
// strings rather than leaking them into a table that only grows.

// interned is one shared string, boxed once.
type interned struct {
	box  any // holds the canonical string
	refs int
}

// internTable maps a string's contents to its shared copy. Not safe for
// concurrent use: Map holds its write lock around every call.
type internTable map[string]*interned

// ref returns the shared copy of s, taking a reference to it.
func (t internTable) ref(s string) *interned {
	if e, ok := t[s]; ok {
		e.refs++
		return e
	}
	e := &interned{box: s, refs: 1}
	t[s] = e
	return e
}

// unref drops a reference to s, forgetting it when nothing holds it.
func (t internTable) unref(s string) {
	e, ok := t[s]
	if !ok {
		return
	}
	if e.refs--; e.refs == 0 {
		delete(t, s)
	}
}

// adopt returns md rebuilt from shared strings — keys always, values when they
// are strings — taking a reference for each. md itself is not modified, and the
// result is a fresh map: what the caller passed is still theirs.
func (t internTable) adopt(md Metadata) Metadata {
	out := make(Metadata, len(md))
	for k, v := range md {
		key := t.ref(k).box.(string)
		if s, ok := v.(string); ok {
			out[key] = t.ref(s).box
			continue
		}
		out[key] = v
	}
	return out
}

// release drops the references adopt took for md.
func (t internTable) release(md Metadata) {
	for k, v := range md {
		t.unref(k)
		if s, ok := v.(string); ok {
			t.unref(s)
		}
	}
}
