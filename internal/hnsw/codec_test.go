package hnsw

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/khambampati-subhash/govecdb/internal/snapshot"
)

// The codec is the one place the graph's internal representation becomes bytes
// on somebody's disk, so these tests care about two things: that a round trip
// returns the *same graph* rather than an equivalent-looking one, and that a
// file which does not describe a walkable graph is refused at load rather than
// panicking inside a search days later.

// roundTrip serializes a graph and reads it back.
func roundTrip(t *testing.T, g *Graph) *Graph {
	t.Helper()

	var buf bytes.Buffer
	n, err := g.WriteTo(&buf)
	if err != nil {
		t.Fatalf("WriteTo: %v", err)
	}
	if n != int64(buf.Len()) {
		t.Fatalf("WriteTo reported %d bytes, wrote %d", n, buf.Len())
	}

	got, err := Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	return got
}

// assertSameGraph compares every piece of state the index actually uses.
func assertSameGraph(t *testing.T, want, got *Graph) {
	t.Helper()

	if want.cfg != got.cfg {
		t.Fatalf("config: got %+v, want %+v", got.cfg, want.cfg)
	}
	if got.entry != want.entry {
		t.Fatalf("entry = %d, want %d", got.entry, want.entry)
	}
	if got.maxLevel != want.maxLevel {
		t.Fatalf("maxLevel = %d, want %d", got.maxLevel, want.maxLevel)
	}
	if got.numDeleted != want.numDeleted {
		t.Fatalf("numDeleted = %d, want %d — the tombstone count is derived, not stored", got.numDeleted, want.numDeleted)
	}
	if len(got.nodes) != len(want.nodes) {
		t.Fatalf("%d nodes, want %d", len(got.nodes), len(want.nodes))
	}
	if !reflect.DeepEqual(got.ids, want.ids) {
		t.Fatalf("id index differs: %d entries, want %d", len(got.ids), len(want.ids))
	}

	for i := range want.nodes {
		w, g := want.nodes[i], got.nodes[i]
		if g.id != w.id {
			t.Fatalf("node %d id = %q, want %q", i, g.id, w.id)
		}
		if g.deleted != w.deleted {
			t.Fatalf("node %d deleted = %v, want %v", i, g.deleted, w.deleted)
		}
		// Bit-exact: float32 goes out as its bits and comes back as its bits, so
		// a round trip that merely got close would be a bug.
		if !reflect.DeepEqual(g.vector, w.vector) {
			t.Fatalf("node %d vector differs", i)
		}
		if !reflect.DeepEqual(g.neighbors, w.neighbors) {
			t.Fatalf("node %d neighbors differ:\n got %v\nwant %v", i, g.neighbors, w.neighbors)
		}
	}

	// The derived fields newGraph rebuilds have to land where they were, or a
	// loaded graph would search differently from the one it came from.
	if got.mMax != want.mMax || got.mMax0 != want.mMax0 {
		t.Fatalf("layer caps = %d/%d, want %d/%d", got.mMax, got.mMax0, want.mMax, want.mMax0)
	}
	if got.ml != want.ml || got.alpha != want.alpha || got.normalized != want.normalized {
		t.Fatal("derived construction knobs differ")
	}
}

// buildCodecGraph makes a graph with n vectors, deleting every `deleteEvery`th.
// Separate from delete_test.go's buildGraph because the codec needs to cover
// every metric and to leave tombstones behind on purpose.
func buildCodecGraph(t *testing.T, cfg Config, n, deleteEvery int) *Graph {
	t.Helper()

	g, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	for i := range n {
		if err := g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, cfg.Dimension)); err != nil {
			t.Fatal(err)
		}
	}
	if deleteEvery > 0 {
		for i := 0; i < n; i += deleteEvery {
			g.Delete(fmt.Sprintf("v%d", i))
		}
	}
	return g
}

func TestCodecRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name        string
		metric      Metric
		n           int
		deleteEvery int
	}{
		{"cosine", Cosine, 200, 0},
		{"euclidean", Euclidean, 200, 0},
		{"dotproduct", DotProduct, 200, 0},
		{"with tombstones", Cosine, 200, 4},
		{"single vector", Cosine, 1, 0},
		{"every vector deleted", Cosine, 20, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultConfig(16, tc.metric)
			want := buildCodecGraph(t, cfg, tc.n, tc.deleteEvery)
			assertSameGraph(t, want, roundTrip(t, want))
		})
	}
}

// TestCodecEmptyGraph covers the state every database starts in. A round trip
// must not be the thing that gives an empty graph memory.
func TestCodecEmptyGraph(t *testing.T) {
	g, err := New(DefaultConfig(8, Cosine))
	if err != nil {
		t.Fatal(err)
	}
	got := roundTrip(t, g)

	assertSameGraph(t, g, got)
	if got.nodes != nil {
		t.Fatalf("a loaded empty graph allocated %d slots", len(got.nodes))
	}
	if got.entry != -1 || got.Len() != 0 {
		t.Fatalf("entry = %d, Len = %d", got.entry, got.Len())
	}

	// And it must still be usable, not just equal.
	if err := got.Insert("first", make([]float32, 8)); err != nil {
		t.Fatal(err)
	}
	if got.Len() != 1 {
		t.Fatalf("Len after inserting into a loaded empty graph = %d", got.Len())
	}
}

// TestCodecPreservesSearchResults is the property that actually matters. Equal
// structure is the mechanism; identical answers are the point.
func TestCodecPreservesSearchResults(t *testing.T) {
	cfg := DefaultConfig(32, Cosine)
	want := buildCodecGraph(t, cfg, 500, 7)
	got := roundTrip(t, want)

	rng := rand.New(rand.NewSource(99))
	for q := range 100 {
		query := randomVector(rng, cfg.Dimension)

		a, err := want.Search(query, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		b, err := got.Search(query, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		if len(a) != len(b) {
			t.Fatalf("query %d returned %d results, want %d", q, len(b), len(a))
		}
		for i := range a {
			if a[i].ID != b[i].ID || a[i].Distance != b[i].Distance {
				t.Fatalf("query %d result %d: got %s/%v, want %s/%v",
					q, i, b[i].ID, b[i].Distance, a[i].ID, a[i].Distance)
			}
		}
	}
}

// TestCodecRoundTripsACompactedGraph covers the other graph shape that exists:
// one produced by a rebuild rather than by inserts.
func TestCodecRoundTripsACompactedGraph(t *testing.T) {
	cfg := DefaultConfig(16, Cosine)
	g := buildCodecGraph(t, cfg, 300, 3)
	if reclaimed := g.Compact(); reclaimed == 0 {
		t.Fatal("nothing was reclaimed; the fixture has no tombstones")
	}
	assertSameGraph(t, g, roundTrip(t, g))
}

// TestCodecSurvivesFurtherWrites checks that a loaded graph is a working graph,
// not just a matching one: inserting, deleting and compacting must all behave.
func TestCodecSurvivesFurtherWrites(t *testing.T) {
	cfg := DefaultConfig(16, Cosine)
	g := roundTrip(t, buildCodecGraph(t, cfg, 200, 5))

	before := g.Len()
	rng := rand.New(rand.NewSource(21))
	for i := range 50 {
		if err := g.Insert(fmt.Sprintf("new%d", i), randomVector(rng, cfg.Dimension)); err != nil {
			t.Fatal(err)
		}
	}
	if g.Len() != before+50 {
		t.Fatalf("Len = %d, want %d", g.Len(), before+50)
	}

	// The new vectors have to be findable, which is the real test that the
	// restored entry point and neighbor lists are coherent.
	for i := range 50 {
		id := fmt.Sprintf("new%d", i)
		idx := g.ids[id]
		res, err := g.Search(g.nodes[idx].vector, 1, 64)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != id {
			t.Fatalf("%s could not find itself after a round trip: %+v", id, res)
		}
	}

	if !g.Delete("new0") {
		t.Fatal("Delete on a loaded graph found nothing")
	}
	g.Compact()
	if g.Stats().Deleted != 0 {
		t.Fatalf("Compact on a loaded graph left %d tombstones", g.Stats().Deleted)
	}
}

// --- Rejections -------------------------------------------------------------

// rawGraph writes the format by hand, so a test can produce a file the real
// encoder would never emit. An independent writer is the point: corruption tests
// that went through the encoder could only ever prove it agrees with itself.
type rawGraph struct {
	cfg      Config
	entry    int
	maxLevel int
	nodes    []rawNode
}

type rawNode struct {
	deleted  bool
	id       string
	topLevel int
	vector   []float32
	layers   [][]int
}

func (r rawGraph) encode() []byte {
	var b bytes.Buffer
	u32 := func(v uint32) {
		var x [4]byte
		byteOrder.PutUint32(x[:], v)
		b.Write(x[:])
	}
	u64 := func(v uint64) {
		var x [8]byte
		byteOrder.PutUint64(x[:], v)
		b.Write(x[:])
	}

	b.Write(codecMagicBytes[:])
	u32(uint32(codecVersion)) // version (2) + reserved (2)

	u32(uint32(r.cfg.Dimension))
	u32(uint32(r.cfg.Metric))
	u32(uint32(r.cfg.M))
	u32(uint32(r.cfg.EfConstruction))
	u32(math.Float32bits(r.cfg.Alpha))
	u64(uint64(r.cfg.Seed))

	u64(uint64(r.entry + 1))
	u32(uint32(r.maxLevel))
	u64(uint64(len(r.nodes)))

	for _, n := range r.nodes {
		if n.deleted {
			b.WriteByte(1)
		} else {
			b.WriteByte(0)
		}
		u32(uint32(len(n.id)))
		b.WriteString(n.id)
		u32(uint32(n.topLevel))
		for _, f := range n.vector {
			u32(math.Float32bits(f))
		}
		for _, layer := range n.layers {
			u32(uint32(len(layer)))
			for _, ix := range layer {
				u32(uint32(ix))
			}
		}
	}
	return b.Bytes()
}

// validRaw is a minimal well-formed graph: two live nodes pointing at each other
// on layer 0. Every rejection case below is this with one thing broken.
func validRaw() rawGraph {
	return rawGraph{
		cfg:      Config{Dimension: 2, Metric: Cosine, M: 2, EfConstruction: 10, Alpha: 1, Seed: 1},
		entry:    0,
		maxLevel: 0,
		nodes: []rawNode{
			{id: "a", topLevel: 0, vector: []float32{1, 0}, layers: [][]int{{1}}},
			{id: "b", topLevel: 0, vector: []float32{0, 1}, layers: [][]int{{0}}},
		},
	}
}

func TestCodecRejects(t *testing.T) {
	// The baseline has to load, or every case below would pass for the wrong
	// reason.
	if _, err := Read(bytes.NewReader(validRaw().encode())); err != nil {
		t.Fatalf("the valid fixture does not load: %v", err)
	}

	for _, tc := range []struct {
		name  string
		build func(r *rawGraph)
		patch func(b []byte) []byte
		want  error
	}{
		{
			name:  "not a graph",
			patch: func(b []byte) []byte { copy(b, "XXXX"); return b },
			want:  ErrBadCodecMagic,
		},
		{
			name: "future version",
			patch: func(b []byte) []byte {
				byteOrder.PutUint16(b[codecMagicSize:], codecVersion+1)
				return b
			},
			want: ErrUnsupportedCodecVersion,
		},
		{
			name:  "truncated mid-node",
			patch: func(b []byte) []byte { return b[:len(b)-6] },
			want:  io.ErrUnexpectedEOF,
		},
		{
			name:  "truncated header",
			patch: func(b []byte) []byte { return b[:5] },
			want:  io.ErrUnexpectedEOF,
		},
		{
			name:  "zero dimension",
			build: func(r *rawGraph) { r.cfg.Dimension = 0 },
			want:  ErrCorruptGraph,
		},
		{
			// The allocation guard: a corrupt dimension multiplies by node count.
			name:  "absurd dimension",
			build: func(r *rawGraph) { r.cfg.Dimension = maxCodecDimension + 1 },
			want:  ErrCorruptGraph,
		},
		{
			name:  "unknown metric",
			build: func(r *rawGraph) { r.cfg.Metric = Metric(99) },
			want:  ErrCorruptGraph,
		},
		{
			name:  "zero M",
			build: func(r *rawGraph) { r.cfg.M = 0 },
			want:  ErrCorruptGraph,
		},
		{
			name:  "zero EfConstruction",
			build: func(r *rawGraph) { r.cfg.EfConstruction = 0 },
			want:  ErrCorruptGraph,
		},
		{
			name:  "entry past the end",
			build: func(r *rawGraph) { r.entry = 5 },
			want:  ErrCorruptGraph,
		},
		{
			name:  "entry on an empty graph",
			build: func(r *rawGraph) { r.nodes = nil; r.entry = 0 },
			want:  ErrCorruptGraph,
		},
		{
			name:  "absurd maxLevel",
			build: func(r *rawGraph) { r.maxLevel = maxCodecLevel + 1 },
			want:  ErrCorruptGraph,
		},
		{
			// Insert descends from maxLevel starting at entry and indexes into
			// that node's neighbor slice, so this is a panic waiting for the next
			// write rather than a cosmetic disagreement.
			name:  "entry shorter than maxLevel",
			build: func(r *rawGraph) { r.maxLevel = 1 },
			want:  ErrCorruptGraph,
		},
		{
			name: "entry is a tombstone",
			build: func(r *rawGraph) {
				r.nodes[0].deleted = true
			},
			want: ErrCorruptGraph,
		},
		{
			name: "two live nodes share an id",
			build: func(r *rawGraph) {
				r.nodes[1].id = "a"
			},
			want: ErrCorruptGraph,
		},
		{
			// The corruption that would otherwise surface as a panic deep inside
			// a search, long after the file that caused it is forgotten.
			name: "neighbor index past the end",
			build: func(r *rawGraph) {
				r.nodes[0].layers[0] = []int{7}
			},
			want: ErrCorruptGraph,
		},
		{
			// mMax0 is 2*M = 4 here, so five neighbors on layer 0 describes a
			// graph pruneConnections could not have produced.
			name: "more neighbors than the layer cap",
			build: func(r *rawGraph) {
				r.nodes[0].layers[0] = []int{1, 1, 1, 1, 1}
			},
			want: ErrCorruptGraph,
		},
		{
			name: "absurd top level",
			build: func(r *rawGraph) {
				r.nodes[0].topLevel = maxCodecLevel + 1
			},
			want: ErrCorruptGraph,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := validRaw()
			if tc.build != nil {
				tc.build(&raw)
			}
			b := raw.encode()
			if tc.patch != nil {
				b = tc.patch(b)
			}

			g, err := Read(bytes.NewReader(b))
			if err == nil {
				t.Fatalf("a broken graph loaded: %d nodes", len(g.nodes))
			}
			if !errors.Is(err, tc.want) {
				t.Fatalf("Read = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestCodecReadsNothingFromAnEmptyReader is the degenerate input a caller hits
// when a file exists but holds no payload.
func TestCodecReadsNothingFromAnEmptyReader(t *testing.T) {
	if _, err := Read(bytes.NewReader(nil)); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("Read on empty input = %v, want ErrUnexpectedEOF", err)
	}
}

// TestCodecLayoutIsFrozen guards the numbers an on-disk format lives or dies by.
// This file is the graph's internal representation made durable, so a field that
// changes size here makes every existing snapshot unreadable.
func TestCodecLayoutIsFrozen(t *testing.T) {
	if codecHeaderSize != 8 {
		t.Fatalf("header is %d bytes, format says 8 (magic4+version2+reserved2)", codecHeaderSize)
	}
	if codecConfigSize != 28 {
		t.Fatalf("config block is %d bytes, format says 28", codecConfigSize)
	}
	if codecGraphSize != 20 {
		t.Fatalf("graph block is %d bytes, format says 20 (entry8+maxLevel4+count8)", codecGraphSize)
	}
	if codecVersion != 1 {
		t.Fatalf("codec version is %d; bumping it is a migration, not an edit", codecVersion)
	}

	// The fixed prefix has to sit where the format says, because everything
	// after it is variable-length and unfindable if this drifts.
	g, err := New(DefaultConfig(4, Cosine))
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := g.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	if got, want := buf.Len(), codecHeaderSize+codecConfigSize+codecGraphSize; got != want {
		t.Fatalf("an empty graph encodes to %d bytes, want exactly the %d-byte prefix", got, want)
	}
}

// TestCodecThroughTheSnapshotStore is the whole persistence path end to end:
// index → codec → atomic checksummed file → back to a working index. Each half
// is tested on its own; this is the first thing that proves they compose.
func TestCodecThroughTheSnapshotStore(t *testing.T) {
	dir := t.TempDir()
	cfg := DefaultConfig(24, Cosine)
	want := buildCodecGraph(t, cfg, 400, 6)

	// Sequence 42 stands in for "the WAL had reached 42 when this was taken".
	snap, err := snapshot.Create(dir, 42, func(w io.Writer) error {
		_, err := want.WriteTo(w)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if snap.Seq != 42 {
		t.Fatalf("snapshot seq = %d", snap.Seq)
	}

	var got *Graph
	res, err := snapshot.Load(dir, func(r io.Reader) error {
		var err error
		got, err = Read(r)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Found || res.Snapshot.Seq != 42 {
		t.Fatalf("Load: %+v", res)
	}

	assertSameGraph(t, want, got)

	// And the sequence recovery would resume the log from.
	if res.Snapshot.Seq+1 != 43 {
		t.Fatalf("would resume the WAL at %d, want 43", res.Snapshot.Seq+1)
	}
}

func BenchmarkGraphWriteTo(b *testing.B) {
	g, _ := buildBenchGraph(b)

	b.SetBytes(encodedSize(b, g))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := g.WriteTo(io.Discard); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGraphRead(b *testing.B) {
	g, _ := buildBenchGraph(b)
	var buf bytes.Buffer
	if _, err := g.WriteTo(&buf); err != nil {
		b.Fatal(err)
	}
	encoded := buf.Bytes()

	b.SetBytes(int64(len(encoded)))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := Read(bytes.NewReader(encoded)); err != nil {
			b.Fatal(err)
		}
	}
}

// buildBenchGraph makes a graph big enough that the codec rather than the
// fixture is what is being measured.
func buildBenchGraph(b *testing.B) (*Graph, int) {
	b.Helper()

	const n = 10000
	g, err := New(DefaultConfig(128, Cosine))
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewSource(5))
	for i := range n {
		if err := g.Insert(fmt.Sprintf("v%d", i), randomVector(rng, 128)); err != nil {
			b.Fatal(err)
		}
	}
	return g, n
}

func encodedSize(b *testing.B, g *Graph) int64 {
	b.Helper()

	n, err := g.WriteTo(io.Discard)
	if err != nil {
		b.Fatal(err)
	}
	return n
}
