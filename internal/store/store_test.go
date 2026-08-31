package store

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"
)

func TestPutGetDelete(t *testing.T) {
	s := New()

	if _, ok := s.Get("missing"); ok {
		t.Fatal("an empty store returned metadata")
	}
	s.Put("a", Metadata{"k": "v"})
	md, ok := s.Get("a")
	if !ok || md["k"] != "v" {
		t.Fatalf("Get = %+v, %v", md, ok)
	}
	if s.Len() != 1 {
		t.Fatalf("Len = %d", s.Len())
	}

	s.Put("a", Metadata{"k": "w"})
	if md, _ := s.Get("a"); md["k"] != "w" {
		t.Fatalf("Put did not replace: %+v", md)
	}

	if !s.Delete("a") {
		t.Fatal("Delete reported nothing to remove")
	}
	if s.Delete("a") {
		t.Fatal("Delete removed the same id twice")
	}
	if s.Len() != 0 {
		t.Fatalf("Len = %d after deleting everything", s.Len())
	}
}

// TestEmptyMetadataIsAbsence pins the decision that "absent" and "present but
// empty" are one thing, so no caller has to handle both.
func TestEmptyMetadataIsAbsence(t *testing.T) {
	s := New()
	s.Put("a", Metadata{"k": "v"})

	for _, md := range []Metadata{nil, {}} {
		s.Put("a", md)
		if _, ok := s.Get("a"); ok {
			t.Fatalf("storing %v left an entry behind", md)
		}
		if s.Len() != 0 {
			t.Fatalf("Len = %d", s.Len())
		}
		s.Put("a", Metadata{"k": "v"})
	}
}

// TestStoreCopiesOnTheWayInAndOut: a shared map would be mutable state escaping
// the lock, with no way for this package to notice it changed.
func TestStoreCopiesOnTheWayInAndOut(t *testing.T) {
	s := New()

	given := Metadata{"k": "original"}
	s.Put("a", given)
	given["k"] = "mutated after Put"
	if md, _ := s.Get("a"); md["k"] != "original" {
		t.Fatalf("mutating the caller's map changed stored state: %+v", md)
	}

	got, _ := s.Get("a")
	got["k"] = "mutated after Get"
	if md, _ := s.Get("a"); md["k"] != "original" {
		t.Fatalf("mutating a returned map changed stored state: %+v", md)
	}
}

func TestAllStopsEarly(t *testing.T) {
	s := New()
	for i := range 10 {
		s.Put(fmt.Sprintf("v%d", i), Metadata{"n": int64(i)})
	}

	seen := 0
	s.All(func(string, Metadata) bool {
		seen++
		return seen < 3
	})
	if seen != 3 {
		t.Fatalf("All visited %d entries after being told to stop at 3", seen)
	}

	seen = 0
	s.All(func(string, Metadata) bool { seen++; return true })
	if seen != 10 {
		t.Fatalf("All visited %d of 10", seen)
	}
}

func TestConcurrentAccess(t *testing.T) {
	s := New()
	var wg sync.WaitGroup

	for g := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 200 {
				id := fmt.Sprintf("g%d-%d", g, i)
				s.Put(id, Metadata{"g": int64(g)})
				s.Get(id)
				if i%3 == 0 {
					s.Delete(id)
				}
				s.Len()
			}
		}()
	}
	wg.Wait()
}

// --- Validation -------------------------------------------------------------

func TestValidate(t *testing.T) {
	if err := Validate(nil); err != nil {
		t.Fatalf("nil metadata: %v", err)
	}
	if err := Validate(Metadata{"s": "x", "b": true, "i": int64(1), "f": 1.5}); err != nil {
		t.Fatalf("the four supported types were rejected: %v", err)
	}

	for _, tc := range []struct {
		name string
		md   Metadata
		want error
	}{
		// int is not int64. Go's untyped constants make this the easy mistake,
		// and guessing would be a silent widening here and a silent narrowing
		// somewhere else.
		{"int is not int64", Metadata{"n": 1}, ErrMetadataValueType},
		{"float32", Metadata{"f": float32(1)}, ErrMetadataValueType},
		{"nil value", Metadata{"n": nil}, ErrMetadataValueType},
		{"slice", Metadata{"s": []string{"a"}}, ErrMetadataValueType},
		{"nested map", Metadata{"m": map[string]any{}}, ErrMetadataValueType},
		{"struct", Metadata{"s": struct{}{}}, ErrMetadataValueType},
		{"empty key", Metadata{"": "v"}, ErrMetadataKey},
		{"invalid utf-8 key", Metadata{"\xff": "v"}, ErrMetadataKey},
		{"invalid utf-8 value", Metadata{"k": "\xff\xfe"}, ErrMetadataValueType},
		{"oversized key", Metadata{strings.Repeat("k", MaxKeyBytes+1): "v"}, ErrMetadataTooLarge},
		{"oversized value", Metadata{"k": strings.Repeat("v", MaxStringBytes+1)}, ErrMetadataTooLarge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := Validate(tc.md); !errors.Is(err, tc.want) {
				t.Fatalf("Validate = %v, want %v", err, tc.want)
			}
		})
	}

	t.Run("too many keys", func(t *testing.T) {
		md := make(Metadata, MaxKeys+1)
		for i := range MaxKeys + 1 {
			md[fmt.Sprintf("k%d", i)] = true
		}
		if err := Validate(md); !errors.Is(err, ErrMetadataTooLarge) {
			t.Fatalf("Validate = %v", err)
		}
	})

	t.Run("too large encoded", func(t *testing.T) {
		// Each value is under its own cap; together they are over the total. The
		// total is the bound that actually protects memory.
		md := make(Metadata, 32)
		for i := range 32 {
			md[fmt.Sprintf("k%d", i)] = strings.Repeat("v", MaxStringBytes)
		}
		if err := Validate(md); !errors.Is(err, ErrMetadataTooLarge) {
			t.Fatalf("Validate = %v", err)
		}
	})
}

// --- Encoding ---------------------------------------------------------------

func TestMetadataRoundTrip(t *testing.T) {
	for _, md := range []Metadata{
		nil,
		{},
		{"s": "hello"},
		{"unicode": "héllo → 世界"},
		{"empty": ""},
		{"t": true, "f": false},
		{"min": int64(math.MinInt64), "max": int64(math.MaxInt64), "zero": int64(0)},
		{"pi": math.Pi, "neg": -1e300, "small": 5e-324},
		{"s": "x", "b": true, "i": int64(7), "f": 1.5},
	} {
		t.Run(fmt.Sprintf("%d keys", len(md)), func(t *testing.T) {
			if err := Validate(md); err != nil {
				t.Fatal(err)
			}
			encoded := Encode(nil, md)
			if len(encoded) != EncodedLen(md) {
				t.Fatalf("EncodedLen said %d, Encode produced %d", EncodedLen(md), len(encoded))
			}

			got, n, err := Decode(encoded)
			if err != nil {
				t.Fatal(err)
			}
			if n != len(encoded) {
				t.Fatalf("Decode consumed %d of %d bytes", n, len(encoded))
			}
			if len(got) != len(md) {
				t.Fatalf("got %d keys, want %d", len(got), len(md))
			}
			for k, w := range md {
				if got[k] != w {
					t.Fatalf("[%q] = %v (%T), want %v (%T)", k, got[k], got[k], w, w)
				}
			}
		})
	}
}

// TestDecodeReportsItsLength is what lets metadata sit at the end of a larger
// record without a length prefix of its own.
func TestDecodeReportsItsLength(t *testing.T) {
	md := Metadata{"k": "v"}
	buf := Encode([]byte("prefix"), md)
	buf = append(buf, "suffix"...)

	got, n, err := Decode(buf[len("prefix"):])
	if err != nil {
		t.Fatal(err)
	}
	if got["k"] != "v" {
		t.Fatalf("got %+v", got)
	}
	if n != EncodedLen(md) {
		t.Fatalf("consumed %d bytes, want %d — trailing data was swallowed", n, EncodedLen(md))
	}
}

func TestDecodeRejects(t *testing.T) {
	valid := Encode(nil, Metadata{"key": "value"})

	for _, tc := range []struct {
		name  string
		build func() []byte
		want  error
	}{
		{"empty", func() []byte { return nil }, ErrMetadataCorrupt},
		{"truncated count", func() []byte { return []byte{1} }, ErrMetadataCorrupt},
		{"truncated mid-entry", func() []byte { return valid[:len(valid)-3] }, ErrMetadataCorrupt},
		{
			// Zero-filled space must never decode as an entry, which is why tag
			// zero is not a type.
			name:  "zero tag",
			build: func() []byte { b := append([]byte(nil), valid...); b[7] = tagInvalid; return b },
			want:  ErrMetadataCorrupt,
		},
		{
			name:  "unknown tag",
			build: func() []byte { b := append([]byte(nil), valid...); b[7] = 99; return b },
			want:  ErrMetadataCorrupt,
		},
		{
			name:  "zero-length key",
			build: func() []byte { b := append([]byte(nil), valid...); byteOrder.PutUint16(b[2:], 0); return b },
			want:  ErrMetadataCorrupt,
		},
		{
			name: "key longer than the buffer",
			build: func() []byte {
				b := append([]byte(nil), valid...)
				byteOrder.PutUint16(b[2:], 5000)
				return b
			},
			want: ErrMetadataCorrupt,
		},
		{
			name: "string longer than the cap",
			build: func() []byte {
				b := append([]byte(nil), valid...)
				byteOrder.PutUint32(b[8:], MaxStringBytes+1)
				return b
			},
			want: ErrMetadataTooLarge,
		},
		{
			name: "key count over the cap",
			build: func() []byte {
				b := append([]byte(nil), valid...)
				byteOrder.PutUint16(b, MaxKeys+1)
				return b
			},
			want: ErrMetadataTooLarge,
		},
		{
			// Two entries with one name did not come from a Go map, so they did
			// not come from Encode.
			name: "duplicate key",
			build: func() []byte {
				one := Encode(nil, Metadata{"k": true})
				body := one[2:]
				b := byteOrder.AppendUint16(nil, 2)
				b = append(b, body...)
				return append(b, body...)
			},
			want: ErrMetadataCorrupt,
		},
		{
			// Only 0 and 1 are booleans; anything else would let two byte
			// strings mean one value.
			name: "bool that is neither 0 nor 1",
			build: func() []byte {
				b := Encode(nil, Metadata{"k": true})
				b[len(b)-1] = 2
				return b
			},
			want: ErrMetadataCorrupt,
		},
		{
			name: "invalid utf-8 value",
			build: func() []byte {
				b := Encode(nil, Metadata{"k": "ab"})
				b[len(b)-2], b[len(b)-1] = 0xff, 0xfe
				return b
			},
			want: ErrMetadataCorrupt,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := Decode(tc.build()); !errors.Is(err, tc.want) {
				t.Fatalf("Decode = %v, want %v", err, tc.want)
			}
		})
	}
}

// --- The snapshot section ---------------------------------------------------

func TestSnapshotSectionRoundTrip(t *testing.T) {
	s := New()
	for i := range 500 {
		s.Put(fmt.Sprintf("id-%d", i), Metadata{
			"n":    int64(i),
			"name": fmt.Sprintf("value-%d", i),
			"flag": i%2 == 0,
		})
	}

	var buf bytes.Buffer
	bw := bufio.NewWriterSize(&buf, 64<<10)
	if err := s.WriteTo(bw); err != nil {
		t.Fatal(err)
	}
	if err := bw.Flush(); err != nil {
		t.Fatal(err)
	}

	got := New()
	if err := got.ReadFrom(bufio.NewReaderSize(&buf, 64<<10), 512); err != nil {
		t.Fatal(err)
	}
	if got.Len() != 500 {
		t.Fatalf("read %d entries, wrote 500", got.Len())
	}
	for i := range 500 {
		md, ok := got.Get(fmt.Sprintf("id-%d", i))
		if !ok || md["n"] != int64(i) || md["name"] != fmt.Sprintf("value-%d", i) || md["flag"] != (i%2 == 0) {
			t.Fatalf("id-%d = %+v", i, md)
		}
	}
}

func TestSnapshotSectionOfAnEmptyStore(t *testing.T) {
	var buf bytes.Buffer
	bw := bufio.NewWriterSize(&buf, 4096)
	if err := New().WriteTo(bw); err != nil {
		t.Fatal(err)
	}
	if err := bw.Flush(); err != nil {
		t.Fatal(err)
	}

	got := New()
	if err := got.ReadFrom(bufio.NewReaderSize(&buf, 4096), 512); err != nil {
		t.Fatal(err)
	}
	if got.Len() != 0 {
		t.Fatalf("an empty section restored %d entries", got.Len())
	}
}

// TestSnapshotSectionLeavesTheReaderPositioned is what lets this section share a
// stream with others: it must consume exactly its own bytes and no more.
func TestSnapshotSectionLeavesTheReaderPositioned(t *testing.T) {
	s := New()
	for i := range 200 {
		s.Put(fmt.Sprintf("id-%d", i), Metadata{"n": int64(i)})
	}

	var buf bytes.Buffer
	bw := bufio.NewWriterSize(&buf, 64<<10)
	if err := s.WriteTo(bw); err != nil {
		t.Fatal(err)
	}
	const trailer = "-----AFTER-THE-SECTION-----"
	if _, err := bw.WriteString(trailer); err != nil {
		t.Fatal(err)
	}
	if err := bw.Flush(); err != nil {
		t.Fatal(err)
	}

	br := bufio.NewReaderSize(&buf, 64<<10)
	got := New()
	if err := got.ReadFrom(br, 512); err != nil {
		t.Fatal(err)
	}
	rest := make([]byte, len(trailer))
	if _, err := br.Read(rest); err != nil {
		t.Fatal(err)
	}
	if string(rest) != trailer {
		t.Fatalf("after the section the reader is at %q, want %q — bytes were over-read",
			rest, trailer)
	}
}

func TestSnapshotSectionRejects(t *testing.T) {
	s := New()
	s.Put("a", Metadata{"k": "v"})

	var buf bytes.Buffer
	bw := bufio.NewWriterSize(&buf, 4096)
	if err := s.WriteTo(bw); err != nil {
		t.Fatal(err)
	}
	if err := bw.Flush(); err != nil {
		t.Fatal(err)
	}
	valid := buf.Bytes()

	for _, tc := range []struct {
		name  string
		build func() []byte
		max   int
	}{
		{"empty", func() []byte { return nil }, 512},
		{"unknown version", func() []byte { b := append([]byte(nil), valid...); b[0] = 9; return b }, 512},
		{"truncated count", func() []byte { return valid[:4] }, 512},
		{"truncated entry", func() []byte { return valid[:len(valid)-3] }, 512},
		{
			// The id length is read off a disk this process does not control, so
			// it is refused against the caller's cap before any allocation. The
			// section is written with a six-byte id and read back with a cap of
			// three, so the length genuinely exceeds it.
			name: "id longer than the cap",
			build: func() []byte {
				long := New()
				long.Put("abcdef", Metadata{"k": "v"})
				var b bytes.Buffer
				w := bufio.NewWriterSize(&b, 4096)
				if err := long.WriteTo(w); err != nil {
					t.Fatal(err)
				}
				if err := w.Flush(); err != nil {
					t.Fatal(err)
				}
				return b.Bytes()
			},
			max: 3,
		},
		{
			name: "zero-length id",
			build: func() []byte {
				b := append([]byte(nil), valid...)
				byteOrder.PutUint32(b[9:], 0)
				return b
			},
			max: 512,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := New()
			got.Put("survivor", Metadata{"k": "v"})
			err := got.ReadFrom(bufio.NewReaderSize(bytes.NewReader(tc.build()), 4096), tc.max)
			if err == nil {
				t.Fatal("a broken section was accepted")
			}
			// A failed read must leave the store as it was rather than half
			// replaced, which is why the map is built aside and swapped in.
			if _, ok := got.Get("survivor"); !ok {
				t.Fatal("a failed ReadFrom destroyed the existing contents")
			}
		})
	}
}
