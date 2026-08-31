package govecdb

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/khambampati-subhash/govecdb/internal/store"
	"github.com/khambampati-subhash/govecdb/internal/wal"
)

func testOptions() options {
	o := defaultOptions()
	o.dimension = testDim
	return o
}

func TestRecordRoundTrip(t *testing.T) {
	o := testOptions()
	rng := rand.New(rand.NewSource(300))

	for _, tc := range []struct {
		name string
		v    Vector
	}{
		{"no metadata", Vector{ID: "a", Values: vec(rng, testDim)}},
		{"every type", Vector{ID: "b", Values: vec(rng, testDim), Metadata: Metadata{
			"s": "text", "b": true, "i": int64(-5), "f": 1.5,
		}}},
		{"unicode id", Vector{ID: "文書-1", Values: vec(rng, testDim)}},
		{"long id", Vector{ID: strings.Repeat("x", 512), Values: vec(rng, testDim)}},
		{"empty string value", Vector{ID: "c", Values: vec(rng, testDim), Metadata: Metadata{"e": ""}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			payload := encodePut(nil, tc.v)
			got, isPut, err := decodeRecord(wal.TypePut, payload, &o)
			if err != nil {
				t.Fatal(err)
			}
			if !isPut {
				t.Fatal("a PUT decoded as something else")
			}
			if got.ID != tc.v.ID {
				t.Fatalf("id = %q, want %q", got.ID, tc.v.ID)
			}
			if !reflect.DeepEqual(got.Values, tc.v.Values) {
				t.Fatal("values did not survive the round trip bit for bit")
			}
			if len(got.Metadata) != len(tc.v.Metadata) {
				t.Fatalf("metadata = %+v, want %+v", got.Metadata, tc.v.Metadata)
			}
			for k, w := range tc.v.Metadata {
				if got.Metadata[k] != w {
					t.Fatalf("metadata[%q] = %v (%T), want %v (%T)", k, got.Metadata[k], got.Metadata[k], w, w)
				}
			}
		})
	}
}

func TestDeleteRecordRoundTrip(t *testing.T) {
	o := testOptions()

	payload := encodeDelete(nil, "gone")
	got, isPut, err := decodeRecord(wal.TypeDelete, payload, &o)
	if err != nil {
		t.Fatal(err)
	}
	if isPut {
		t.Fatal("a DELETE decoded as a PUT")
	}
	if got.ID != "gone" {
		t.Fatalf("id = %q", got.ID)
	}
	if got.Values != nil {
		t.Fatal("a DELETE carried values")
	}
}

// TestEncodeReusesTheBuffer pins the property that keeps the write path free of
// per-record allocation.
func TestEncodeReusesTheBuffer(t *testing.T) {
	rng := rand.New(rand.NewSource(301))
	v := Vector{ID: "a", Values: vec(rng, testDim)}

	buf := encodePut(nil, v)
	grown := buf[:0]
	for range 100 {
		grown = encodePut(grown[:0], v)
	}
	if len(grown) != len(buf) {
		t.Fatalf("re-encoding the same vector produced %d bytes, first time %d", len(grown), len(buf))
	}
	if cap(grown) < len(buf) {
		t.Fatal("the buffer was not reused")
	}
}

// TestDecodeRecordRejects covers a payload that passes the log's checksum and
// still does not describe a record this database can hold — a log from a
// differently configured database, or a file somebody edited.
func TestDecodeRecordRejects(t *testing.T) {
	o := testOptions()
	rng := rand.New(rand.NewSource(302))
	good := Vector{ID: "a", Values: vec(rng, testDim), Metadata: Metadata{"k": "v"}}

	for _, tc := range []struct {
		name  string
		typ   wal.RecordType
		build func() []byte
	}{
		{
			name:  "empty payload",
			typ:   wal.TypePut,
			build: func() []byte { return nil },
		},
		{
			name: "unknown version",
			typ:  wal.TypePut,
			build: func() []byte {
				p := encodePut(nil, good)
				p[0] = 99
				return p
			},
		},
		{
			name: "zero-length id",
			typ:  wal.TypePut,
			build: func() []byte {
				p := encodePut(nil, good)
				byteOrder.PutUint16(p[1:], 0)
				return p
			},
		},
		{
			name: "id longer than the payload",
			typ:  wal.TypePut,
			build: func() []byte {
				p := encodePut(nil, good)
				byteOrder.PutUint16(p[1:], 500)
				return p
			},
		},
		{
			name: "invalid utf-8 id",
			typ:  wal.TypePut,
			build: func() []byte {
				p := encodePut(nil, Vector{ID: "ab", Values: good.Values})
				p[3], p[4] = 0xff, 0xfe
				return p
			},
		},
		{
			// A record from a database of a different shape. Applying it would
			// mean inserting a wrong-length vector into this index.
			name: "wrong dimension",
			typ:  wal.TypePut,
			build: func() []byte {
				return encodePut(nil, Vector{ID: "a", Values: make([]float32, testDim+1)})
			},
		},
		{
			// Cannot get in through Add, so finding one here means the bytes did
			// not come from this database — and it would poison every comparison
			// the index makes.
			name: "NaN value",
			typ:  wal.TypePut,
			build: func() []byte {
				v := Vector{ID: "a", Values: append([]float32(nil), good.Values...)}
				v.Values[2] = float32(math.NaN())
				return encodePut(nil, v)
			},
		},
		{
			name: "truncated mid-values",
			typ:  wal.TypePut,
			build: func() []byte {
				p := encodePut(nil, good)
				return p[:len(p)-20]
			},
		},
		{
			// Ignoring trailing bytes would let a record smuggle content past
			// every length this decoder checked.
			name:  "trailing bytes on a PUT",
			typ:   wal.TypePut,
			build: func() []byte { return append(encodePut(nil, good), 0, 0, 0) },
		},
		{
			name:  "trailing bytes on a DELETE",
			typ:   wal.TypeDelete,
			build: func() []byte { return append(encodeDelete(nil, "a"), 1, 2) },
		},
		{
			name: "unknown metadata tag",
			typ:  wal.TypePut,
			build: func() []byte {
				p := encodePut(nil, good)
				// The tag sits just after the single key's length and bytes, at
				// the end of the payload: count(2) key(2+1) tag(1) len(4) "v".
				p[len(p)-6] = 77
				return p
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := decodeRecord(tc.typ, tc.build(), &o); !errors.Is(err, ErrCorrupt) {
				t.Fatalf("decodeRecord = %v, want ErrCorrupt", err)
			}
		})
	}
}

// TestSnapshotPayloadRoundTrip covers the composed payload: an index and a
// metadata store, one after the other.
func TestSnapshotPayloadRoundTrip(t *testing.T) {
	o := testOptions()
	idx, err := newHNSWIndex(o)
	if err != nil {
		t.Fatal(err)
	}
	st := store.New()

	rng := rand.New(rand.NewSource(303))
	for i := range 200 {
		id := fmt.Sprintf("v%d", i)
		if err := idx.Insert(id, vec(rng, testDim)); err != nil {
			t.Fatal(err)
		}
		if i%2 == 0 {
			st.Put(id, Metadata{"n": int64(i), "tag": "even"})
		}
	}

	var buf bytes.Buffer
	if err := writeSnapshot(&buf, idx, st); err != nil {
		t.Fatal(err)
	}
	gotIdx, gotStore, err := readSnapshot(&buf, o.maxIDBytes)
	if err != nil {
		t.Fatal(err)
	}

	if gotIdx.Len() != idx.Len() {
		t.Fatalf("index holds %d, want %d", gotIdx.Len(), idx.Len())
	}
	if gotStore.Len() != st.Len() {
		t.Fatalf("store holds %d, want %d", gotStore.Len(), st.Len())
	}
	for i := range 200 {
		id := fmt.Sprintf("v%d", i)
		if _, ok := gotIdx.Lookup(id); !ok {
			t.Fatalf("%s is missing from the index", id)
		}
		md, ok := gotStore.Get(id)
		if i%2 == 0 {
			if !ok || md["n"] != int64(i) || md["tag"] != "even" {
				t.Fatalf("%s metadata = %+v", id, md)
			}
		} else if ok {
			t.Fatalf("%s has metadata it was never given: %+v", id, md)
		}
	}
}

// TestSnapshotPayloadSectionsDoNotOverread is the guard on the one thing in this
// file that would fail silently rather than loudly.
//
// The two sections are self-delimiting and follow one another with no length
// prefix, which only works because they share a single buffered reader — a
// section that wrapped the stream in its own bufio would read ahead into the
// next section's bytes and lose them. Nothing about that is visible at compile
// time, and the symptom would be a corrupted restore rather than an error.
//
// A metadata store big enough to span several buffer refills is what makes this
// a real check: the index section has to leave the reader positioned exactly at
// the store's first byte, however much either one buffers.
func TestSnapshotPayloadSectionsDoNotOverread(t *testing.T) {
	o := testOptions()
	idx, err := newHNSWIndex(o)
	if err != nil {
		t.Fatal(err)
	}
	st := store.New()

	rng := rand.New(rand.NewSource(304))
	const n = 400
	for i := range n {
		id := fmt.Sprintf("id-%06d", i)
		if err := idx.Insert(id, vec(rng, testDim)); err != nil {
			t.Fatal(err)
		}
		st.Put(id, Metadata{
			"index": int64(i),
			"blob":  strings.Repeat("m", 200),
		})
	}

	var buf bytes.Buffer
	if err := writeSnapshot(&buf, idx, st); err != nil {
		t.Fatal(err)
	}
	if buf.Len() < payloadBufSize {
		t.Fatalf("payload is only %d bytes; it must exceed one buffer (%d) to test anything",
			buf.Len(), payloadBufSize)
	}

	gotIdx, gotStore, err := readSnapshot(&buf, o.maxIDBytes)
	if err != nil {
		t.Fatalf("reading a payload larger than one buffer: %v", err)
	}
	if gotIdx.Len() != n {
		t.Fatalf("index holds %d, want %d", gotIdx.Len(), n)
	}
	if gotStore.Len() != n {
		t.Fatalf("store holds %d of %d — the index section swallowed metadata bytes",
			gotStore.Len(), n)
	}
	for i := range n {
		id := fmt.Sprintf("id-%06d", i)
		md, ok := gotStore.Get(id)
		if !ok || md["index"] != int64(i) || md["blob"] != strings.Repeat("m", 200) {
			t.Fatalf("%s metadata did not survive: %+v", id, md)
		}
	}
}

func TestSnapshotPayloadRejects(t *testing.T) {
	o := testOptions()
	idx, err := newHNSWIndex(o)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Insert("a", vec(rand.New(rand.NewSource(305)), testDim)); err != nil {
		t.Fatal(err)
	}
	st := store.New()
	st.Put("a", Metadata{"k": "v"})

	var good bytes.Buffer
	if err := writeSnapshot(&good, idx, st); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name  string
		patch func(b []byte) []byte
	}{
		{"not a govecdb snapshot", func(b []byte) []byte { copy(b, "XXXX"); return b }},
		{"future version", func(b []byte) []byte {
			byteOrder.PutUint16(b[payloadMagicSize:], payloadVersion+1)
			return b
		}},
		{"truncated header", func(b []byte) []byte { return b[:4] }},
		{"truncated index", func(b []byte) []byte { return b[:payloadHeaderSize+10] }},
		{"truncated store", func(b []byte) []byte { return b[:len(b)-4] }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := tc.patch(append([]byte(nil), good.Bytes()...))
			if _, _, err := readSnapshot(bytes.NewReader(b), o.maxIDBytes); !errors.Is(err, ErrCorrupt) {
				t.Fatalf("readSnapshot = %v, want ErrCorrupt", err)
			}
		})
	}
}

// TestSnapshotPayloadOfAnEmptyDatabase: the state every database starts in has
// to serialize and come back.
func TestSnapshotPayloadOfAnEmptyDatabase(t *testing.T) {
	o := testOptions()
	idx, err := newHNSWIndex(o)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := writeSnapshot(&buf, idx, store.New()); err != nil {
		t.Fatal(err)
	}
	gotIdx, gotStore, err := readSnapshot(&buf, o.maxIDBytes)
	if err != nil {
		t.Fatal(err)
	}
	if gotIdx.Len() != 0 || gotStore.Len() != 0 {
		t.Fatalf("empty payload restored %d vectors and %d metadata entries",
			gotIdx.Len(), gotStore.Len())
	}
}
