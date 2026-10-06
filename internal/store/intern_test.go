package store

import (
	"bufio"
	"bytes"
	"fmt"
	"strings"
	"testing"
	"unsafe"
)

// sameBytes reports whether two strings share one backing array — which is the
// whole claim interning makes.
func sameBytes(a, b string) bool { return unsafe.StringData(a) == unsafe.StringData(b) }

func keyOf(t *testing.T, md Metadata, want string) string {
	t.Helper()
	for k := range md {
		if k == want {
			return k
		}
	}
	t.Fatalf("no key %q in %v", want, md)
	return ""
}

func TestRepeatedStringsAreStoredOnce(t *testing.T) {
	s := New()
	// strings.Clone, so the two inputs genuinely do not share bytes — as two
	// decoded records would not.
	s.Put("a", Metadata{strings.Clone("doc"): strings.Clone("handbook.pdf"), "n": int64(1)})
	s.Put("b", Metadata{strings.Clone("doc"): strings.Clone("handbook.pdf"), "n": int64(2)})

	a, b := s.m["a"], s.m["b"]
	if !sameBytes(a["doc"].(string), b["doc"].(string)) {
		t.Error("equal values are stored twice")
	}
	if !sameBytes(keyOf(t, a, "doc"), keyOf(t, b, "doc")) {
		t.Error("equal keys are stored twice")
	}
	if got := s.strs["handbook.pdf"].refs; got != 2 {
		t.Errorf("refs = %d, want 2", got)
	}
}

// TestInternedStringsAreReleased: a table that only grew would turn interning
// into a leak the moment documents are deleted or rewritten.
func TestInternedStringsAreReleased(t *testing.T) {
	s := New()
	for i := range 100 {
		s.Put(fmt.Sprintf("v%d", i), Metadata{"doc": fmt.Sprintf("d%d", i/10), "page": int64(i)})
	}
	// Replacing drops the old value's reference.
	s.Put("v0", Metadata{"doc": "elsewhere"})
	if e := s.strs["d0"]; e == nil || e.refs != 9 {
		t.Fatalf("after one replace, d0 refs = %+v, want 9", e)
	}
	// An empty Put is a delete, and releases too.
	s.Put("v1", nil)
	for i := range 100 {
		s.Delete(fmt.Sprintf("v%d", i))
	}
	if len(s.strs) != 0 {
		t.Fatalf("%d strings still interned after every entry was deleted: %v", len(s.strs), s.strs)
	}
}

func TestSnapshotLoadInterns(t *testing.T) {
	src := New()
	for i := range 20 {
		src.Put(fmt.Sprintf("v%d", i), Metadata{"doc": "same", "i": int64(i)})
	}
	var buf bytes.Buffer
	w := bufio.NewWriter(&buf)
	if err := src.WriteTo(w); err != nil {
		t.Fatal(err)
	}
	w.Flush()

	dst := New()
	if err := dst.ReadFrom(bufio.NewReader(&buf), 64); err != nil {
		t.Fatal(err)
	}
	if !sameBytes(dst.m["v0"]["doc"].(string), dst.m["v19"]["doc"].(string)) {
		t.Error("a loaded snapshot holds a copy of the value per entry")
	}
	if e := dst.strs["same"]; e == nil || e.refs != 20 {
		t.Fatalf("refs after load = %+v, want 20", e)
	}
	// And the loaded store keeps counting correctly from there.
	for i := range 20 {
		dst.Delete(fmt.Sprintf("v%d", i))
	}
	if len(dst.strs) != 0 {
		t.Fatalf("%d strings left after deleting a loaded store", len(dst.strs))
	}
}

// BenchmarkPutRepeatedMetadata is the write-path price of interning: two map
// lookups per entry under the lock the copy already took.
func BenchmarkPutRepeatedMetadata(b *testing.B) {
	s := New()
	md := Metadata{"doc_id": "doc-1", "source": "/corpus/a.pdf", "title": "A Title", "page": int64(3), "lang": "en"}
	b.ReportAllocs()
	for i := range b.N {
		s.Put(fmt.Sprintf("v%d", i%10000), md)
	}
}
