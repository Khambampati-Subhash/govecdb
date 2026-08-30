package hnsw

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// Serializing the graph, so recovery does not have to rebuild it.
//
// # Why the graph and not just the vectors
//
// A snapshot holding only the live vectors would be a much simpler format, and
// it would be half a snapshot. Replay reads a log at 368 ns/record but *applying*
// a record costs 703 µs, so reading is 0.05% of recovery and the index build is
// all of it. A graph of dim 128 at M=16 encodes to 662 bytes per vector, so a
// million vectors is ~662 MB — ~0.37 s to verify and decode against ~703 s to
// rebuild, about 1,900×. Bounding the log's size while leaving recovery time
// untouched is not what a snapshot is for. Measured in docs/DURABILITY.md.
//
// The price is that this file freezes the graph's internal representation on
// disk — neighbor lists are slot indices, so what is written here is exactly the
// structure the index depends on. That is why the format is versioned from the
// first byte and why TestLayoutIsFrozen pins its sizes.
//
// # There is no checksum here, on purpose
//
// This format carries no integrity check of its own because it is designed to
// live inside a snapshot, and internal/snapshot already hashes the whole payload
// end to end before handing back a byte of it. Checksumming twice would cost a
// second pass over gigabytes to learn the same thing. The consequence is a
// contract worth stating plainly: **Read trusts its input to have been verified
// already**. Writing this format somewhere without integrity checking is a
// mistake the compiler cannot catch.
//
// It still validates *structure* — indices in range, counts within their caps,
// the entry point live and at the right level. That is not redundancy with the
// checksum; a checksum proves the bytes are the bytes that were written, not
// that they describe a graph a search can safely walk.

const (
	// codecMagic spells GVGR — GoVecDB GRaph. Four ASCII bytes so `head -c 4` on
	// a payload answers "what is this".
	codecMagicSize    = 4
	codecVersionSize  = 2
	codecReservedSize = 2
	// codecHeaderSize is 8, which keeps the config that follows 8-byte aligned.
	codecHeaderSize = codecMagicSize + codecVersionSize + codecReservedSize

	// codecConfigSize: dimension, metric, m, efConstruction (4 each), alpha (4),
	// seed (8).
	codecConfigSize = 4 + 4 + 4 + 4 + 4 + 8 // 28

	// codecGraphSize: entry (8, signed because -1 means empty), maxLevel (4),
	// nodeCount (8).
	codecGraphSize = 8 + 4 + 8 // 20

	// codecVersion is bumped when the layout changes incompatibly.
	codecVersion uint16 = 1
)

// codecMagicBytes spells GVGR.
var codecMagicBytes = [codecMagicSize]byte{'G', 'V', 'G', 'R'}

// Little-endian throughout, matching the WAL and the snapshot store, so a graph
// written on one machine loads on another.
var byteOrder = binary.LittleEndian

// Bounds on what a header is allowed to make Read allocate.
//
// Every one of these is read off the wire before there is any way to know it is
// sane, and each one multiplies an allocation. This is the same rule the WAL
// reader follows for record lengths: nothing is allocated on the strength of a
// number that has not been checked. They are deliberately far above anything
// real — they are not limits on what can be indexed, they are limits on what a
// corrupt byte can ask for.
const (
	// maxCodecDimension is ~1M. The largest embedding models in use are ~16k.
	maxCodecDimension = 1 << 20

	// maxCodecIDBytes caps one external id at 64 KiB.
	maxCodecIDBytes = 1 << 16

	// maxCodecLevel caps a node's top layer. Levels are drawn from
	// int(-ln(u)/ln(M)), so reaching layer 64 has probability M^-64 — below
	// 2^-64 even at M=2. A file claiming more than this is corrupt, not lucky.
	maxCodecLevel = 64

	// maxCodecNodes is what a uint32 slot index can address. Neighbor lists are
	// written as uint32 because they are the bulk of the file and a graph of four
	// billion vectors does not fit in memory anyway.
	maxCodecNodes = math.MaxUint32
)

var (
	// ErrBadCodecMagic means the bytes are not a serialized graph.
	ErrBadCodecMagic = errors.New("hnsw: not a serialized graph")

	// ErrUnsupportedCodecVersion means the graph was written by a format version
	// this build does not know how to read. Refusing is the point of carrying the
	// version; guessing at an unknown layout would build a broken index.
	ErrUnsupportedCodecVersion = errors.New("hnsw: unsupported graph format version")

	// ErrCorruptGraph means the bytes decoded but do not describe a usable graph.
	// Structure, not integrity — see the note at the top of this file.
	ErrCorruptGraph = errors.New("hnsw: serialized graph is structurally invalid")
)

// WriteTo serializes the graph to w. It implements io.WriterTo.
//
// # It holds the read lock for the whole write
//
// Searches continue to run; inserts wait. That is the honest cost of a
// consistent point in time, and it is bounded by throughput rather than by graph
// size in any surprising way — roughly 2 GB/s, so a gigabyte of graph blocks
// writers for about half a second.
//
// Doing better means letting writes land while the snapshot streams, which needs
// the same change log and double-buffered swap that online compaction needs. The
// two should be solved together, once, rather than half-solved twice.
//
// # What is written, and what is left to be recomputed
//
// The node array, the entry point, and maxLevel. Not the id index and not the
// tombstone count: both are pure functions of the node array, and writing them
// down would create a second source of truth that a corrupt file could put in
// disagreement with the first. Anything derivable is derived on load.
func (g *Graph) WriteTo(w io.Writer) (int64, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// Buffered unconditionally. The fields below are small and numerous, so an
	// unbuffered writer would turn one snapshot into millions of syscalls. When
	// the caller already buffers — snapshot.Create does — the cost is one memcpy
	// through 64 KiB, which is nothing against the disk behind it.
	e := &encoder{w: bufio.NewWriterSize(w, 64<<10)}

	var hdr [codecHeaderSize]byte
	copy(hdr[:codecMagicSize], codecMagicBytes[:])
	byteOrder.PutUint16(hdr[codecMagicSize:], codecVersion)
	e.raw(hdr[:])

	e.u32(uint32(g.cfg.Dimension))
	e.u32(uint32(g.cfg.Metric))
	e.u32(uint32(g.cfg.M))
	e.u32(uint32(g.cfg.EfConstruction))
	e.u32(math.Float32bits(g.cfg.Alpha))
	e.u64(uint64(g.cfg.Seed))

	e.u64(uint64(g.entry + 1)) // shifted so -1 encodes as 0; see decode
	e.u32(uint32(g.maxLevel))
	e.u64(uint64(len(g.nodes)))

	for _, n := range g.nodes {
		var flags byte
		if n.deleted {
			flags = 1
		}
		e.u8(flags)

		e.u32(uint32(len(n.id)))
		e.str(n.id)

		e.u32(uint32(n.topLevel()))
		e.vector(n.vector)

		for _, layer := range n.neighbors {
			e.u32(uint32(len(layer)))
			e.indices(layer)
		}
		if e.err != nil {
			return e.n, e.err
		}
	}

	if err := e.w.Flush(); err != nil {
		return e.n, fmt.Errorf("hnsw: flush graph: %w", err)
	}
	return e.n, e.err
}

// Read reconstructs a graph written by WriteTo.
//
// It is the counterpart to New: both produce a graph ready to search, one from a
// config and one from bytes. Every derived field — the distance kernel, the layer
// caps, the level factor, the scratch pool — is rebuilt by newGraph from the
// config that was stored, so there is exactly one place that decides what a
// config implies and no chance of a loaded graph disagreeing with a fresh one.
//
// # The RNG is not restored
//
// math/rand's source cannot be marshaled, so a loaded graph draws future levels
// from a fresh sequence seeded by the stored Seed rather than continuing the one
// that was interrupted. Levels stay correctly distributed, so nothing about
// recall or correctness changes; what differs is that building 1,000 vectors,
// saving, loading and inserting 1,000 more no longer produces byte-identical
// results to building 2,000 in one go. Persisting the state would mean either
// reimplementing the source or carrying a draw counter on every insert, and a
// counter would not survive Compact — which replaces the node array but not the
// RNG — so it would be a field that is subtly wrong rather than absent.
func Read(r io.Reader) (*Graph, error) {
	d := &decoder{r: bufio.NewReaderSize(r, 64<<10)}

	var hdr [codecHeaderSize]byte
	d.raw(hdr[:])
	if d.err != nil {
		return nil, d.err
	}
	if [codecMagicSize]byte(hdr[:codecMagicSize]) != codecMagicBytes {
		return nil, ErrBadCodecMagic
	}
	if v := byteOrder.Uint16(hdr[codecMagicSize:]); v != codecVersion {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedCodecVersion, v)
	}

	cfg := Config{
		Dimension:      int(d.u32()),
		Metric:         Metric(d.u32()),
		M:              int(d.u32()),
		EfConstruction: int(d.u32()),
		Alpha:          math.Float32frombits(d.u32()),
		Seed:           int64(d.u64()),
	}
	entry := int(d.u64()) - 1 // undo the shift; 0 decodes back to -1
	maxLevel := int(d.u32())
	nodeCount := d.u64()
	if d.err != nil {
		return nil, d.err
	}

	if err := validateHeader(cfg, entry, maxLevel, nodeCount); err != nil {
		return nil, err
	}

	// newGraph applies the same defaults and derives the same kernels a fresh
	// graph would, so a loaded graph and a rebuilt one are the same object.
	g := newGraph(cfg)
	// Left nil when there is nothing to hold, so a loaded empty graph is the
	// same object as a fresh one — an empty graph is an empty container, and a
	// round trip must not be the thing that gives it memory.
	if nodeCount > 0 {
		g.nodes = make([]*node, nodeCount)
	}
	g.entry = entry
	g.maxLevel = maxLevel

	for i := range g.nodes {
		n, err := d.node(g, uint32(nodeCount))
		if err != nil {
			return nil, fmt.Errorf("hnsw: node %d: %w", i, err)
		}
		g.nodes[i] = n
	}
	if d.err != nil {
		return nil, d.err
	}

	if err := g.rebuildDerived(); err != nil {
		return nil, err
	}
	return g, nil
}

// validateHeader checks everything decidable before a node is read.
func validateHeader(cfg Config, entry, maxLevel int, nodeCount uint64) error {
	switch {
	case cfg.Dimension <= 0 || cfg.Dimension > maxCodecDimension:
		return fmt.Errorf("%w: dimension %d", ErrCorruptGraph, cfg.Dimension)
	case cfg.Metric != Cosine && cfg.Metric != Euclidean && cfg.Metric != DotProduct:
		return fmt.Errorf("%w: unknown metric %d", ErrCorruptGraph, cfg.Metric)
	case cfg.M <= 0:
		return fmt.Errorf("%w: M %d", ErrCorruptGraph, cfg.M)
	case cfg.EfConstruction <= 0:
		return fmt.Errorf("%w: EfConstruction %d", ErrCorruptGraph, cfg.EfConstruction)
	case nodeCount > maxCodecNodes:
		return fmt.Errorf("%w: %d nodes exceeds what a slot index can address", ErrCorruptGraph, nodeCount)
	case maxLevel < 0 || maxLevel > maxCodecLevel:
		return fmt.Errorf("%w: maxLevel %d", ErrCorruptGraph, maxLevel)
	case entry < -1 || (entry >= 0 && uint64(entry) >= nodeCount):
		return fmt.Errorf("%w: entry %d with %d nodes", ErrCorruptGraph, entry, nodeCount)
	case nodeCount == 0 && entry != -1:
		return fmt.Errorf("%w: empty graph with entry %d", ErrCorruptGraph, entry)
	}
	return nil
}

// rebuildDerived recomputes the id index and the tombstone count, then checks
// the invariants the rest of the package relies on.
//
// The entry check is the one that matters. Insert descends from maxLevel
// starting at entry and indexes straight into that node's neighbor slice, so an
// entry point shorter than maxLevel panics on the next write — a long way from
// the file that caused it. reelectEntry maintains "entry is live and sits at
// maxLevel"; asserting it here turns a corrupt file into a load error instead of
// a crash during someone's insert.
func (g *Graph) rebuildDerived() error {
	g.ids = make(map[string]int, len(g.nodes))
	g.numDeleted = 0

	for i, n := range g.nodes {
		if n.deleted {
			g.numDeleted++
			continue
		}
		if prev, dup := g.ids[n.id]; dup {
			return fmt.Errorf("%w: id %q is live at both slot %d and %d", ErrCorruptGraph, n.id, prev, i)
		}
		g.ids[n.id] = i
	}

	if g.entry == -1 {
		return nil
	}
	if e := g.nodes[g.entry]; e.deleted {
		return fmt.Errorf("%w: entry %d is a tombstone", ErrCorruptGraph, g.entry)
	} else if e.topLevel() != g.maxLevel {
		return fmt.Errorf("%w: entry %d reaches level %d, maxLevel is %d",
			ErrCorruptGraph, g.entry, e.topLevel(), g.maxLevel)
	}
	return nil
}

// encoder writes the format with a sticky error, so the caller checks once
// rather than after every field. The first failure ends the encoding and every
// later call is a no-op — the same shape as the WAL writer, and for the same
// reason: continuing after a failed write produces a file with a hole in it.
type encoder struct {
	w   *bufio.Writer
	n   int64
	err error
	buf []byte
	// num is reused for the fixed-width fields. A local array here would escape
	// into Write and cost a heap allocation per field, which across a few
	// million nodes is millions of allocations to write eight bytes at a time.
	num [8]byte
}

func (e *encoder) raw(p []byte) {
	if e.err != nil {
		return
	}
	n, err := e.w.Write(p)
	e.n += int64(n)
	if err != nil {
		e.err = fmt.Errorf("hnsw: write graph: %w", err)
	}
}

func (e *encoder) u8(v byte) {
	if e.err != nil {
		return
	}
	if err := e.w.WriteByte(v); err != nil {
		e.err = fmt.Errorf("hnsw: write graph: %w", err)
		return
	}
	e.n++
}

func (e *encoder) u32(v uint32) {
	byteOrder.PutUint32(e.num[:4], v)
	e.raw(e.num[:4])
}

func (e *encoder) u64(v uint64) {
	byteOrder.PutUint64(e.num[:8], v)
	e.raw(e.num[:8])
}

// str writes a string without the []byte conversion that would copy it onto the
// heap once per node.
func (e *encoder) str(s string) {
	if e.err != nil {
		return
	}
	n, err := e.w.WriteString(s)
	e.n += int64(n)
	if err != nil {
		e.err = fmt.Errorf("hnsw: write graph: %w", err)
	}
}

// vector writes dim float32s as one contiguous run. Encoding through a reused
// scratch buffer rather than four bytes at a time is what keeps this at disk
// speed instead of function-call speed: vectors are the bulk of the file.
func (e *encoder) vector(v []float32) {
	b := e.scratch(len(v) * 4)
	for i, f := range v {
		byteOrder.PutUint32(b[i*4:], math.Float32bits(f))
	}
	e.raw(b)
}

// indices writes a neighbor list as uint32s.
func (e *encoder) indices(ix []int) {
	b := e.scratch(len(ix) * 4)
	for i, v := range ix {
		byteOrder.PutUint32(b[i*4:], uint32(v))
	}
	e.raw(b)
}

func (e *encoder) scratch(n int) []byte {
	if cap(e.buf) < n {
		e.buf = make([]byte, n)
	}
	return e.buf[:n]
}

// decoder mirrors encoder: sticky error, and every read returns a zero value
// once something has failed. Zeros are safe to carry — they make every later
// length zero rather than arbitrary — but callers still check err before acting
// on anything, so a failure never gets as far as building a graph.
type decoder struct {
	r    *bufio.Reader
	err  error
	buf  []byte
	buf2 []byte
	// num is reused for fixed-width fields, for the same reason as the encoder's.
	num [8]byte
}

func (d *decoder) raw(p []byte) {
	if d.err != nil {
		return
	}
	if _, err := io.ReadFull(d.r, p); err != nil {
		d.err = fmt.Errorf("hnsw: read graph: %w", shortOr(err))
	}
}

func (d *decoder) u8() byte {
	if d.err != nil {
		return 0
	}
	v, err := d.r.ReadByte()
	if err != nil {
		d.err = fmt.Errorf("hnsw: read graph: %w", shortOr(err))
		return 0
	}
	return v
}

func (d *decoder) u32() uint32 {
	d.raw(d.num[:4])
	if d.err != nil {
		return 0
	}
	return byteOrder.Uint32(d.num[:4])
}

func (d *decoder) u64() uint64 {
	d.raw(d.num[:8])
	if d.err != nil {
		return 0
	}
	return byteOrder.Uint64(d.num[:8])
}

// node reads one node, refusing every count that would make it allocate on a
// number it has not checked.
func (d *decoder) node(g *Graph, nodeCount uint32) (*node, error) {
	flags := d.u8()

	idLen := d.u32()
	if d.err != nil {
		return nil, d.err
	}
	if idLen > maxCodecIDBytes {
		return nil, fmt.Errorf("%w: id of %d bytes", ErrCorruptGraph, idLen)
	}
	id := d.scratch(int(idLen))
	d.raw(id)

	topLevel := d.u32()
	if d.err != nil {
		return nil, d.err
	}
	if topLevel > maxCodecLevel {
		return nil, fmt.Errorf("%w: top level %d", ErrCorruptGraph, topLevel)
	}

	vec := make([]float32, g.cfg.Dimension)
	raw := d.scratch2(g.cfg.Dimension * 4)
	d.raw(raw)
	if d.err != nil {
		return nil, d.err
	}
	for i := range vec {
		vec[i] = math.Float32frombits(byteOrder.Uint32(raw[i*4:]))
	}

	n := &node{
		id:        string(id), // copies; scratch is reused by the next node
		vector:    vec,
		neighbors: make([][]int, topLevel+1),
		deleted:   flags&1 != 0,
	}

	for lc := range n.neighbors {
		count := d.u32()
		if d.err != nil {
			return nil, d.err
		}
		// The cap is a real invariant, not a guess: pruneConnections trims every
		// list back to maxConn, and selectNeighbors never returns more. A file
		// claiming more than that describes a graph this package cannot have
		// produced, and loading it would silently degrade search.
		if limit := g.maxConn(lc); int64(count) > int64(limit) {
			return nil, fmt.Errorf("%w: %d neighbors on layer %d, cap is %d",
				ErrCorruptGraph, count, lc, limit)
		}
		if count == 0 {
			continue
		}
		raw := d.scratch2(int(count) * 4)
		d.raw(raw)
		if d.err != nil {
			return nil, d.err
		}
		layer := make([]int, count)
		for i := range layer {
			idx := byteOrder.Uint32(raw[i*4:])
			// An out-of-range neighbor is the one corruption that would not
			// surface here at all — it panics deep inside a search, long after
			// the file that caused it is forgotten.
			if idx >= nodeCount {
				return nil, fmt.Errorf("%w: neighbor %d on layer %d, %d nodes exist",
					ErrCorruptGraph, idx, lc, nodeCount)
			}
			layer[i] = int(idx)
		}
		n.neighbors[lc] = layer
	}
	return n, nil
}

func (d *decoder) scratch(n int) []byte {
	if cap(d.buf) < n {
		d.buf = make([]byte, n)
	}
	return d.buf[:n]
}

// scratch2 is a second reusable buffer, because a node needs its id and its
// payload live at the same time and one buffer cannot hold both.
func (d *decoder) scratch2(n int) []byte {
	if cap(d.buf2) < n {
		d.buf2 = make([]byte, n)
	}
	return d.buf2[:n]
}

// shortOr maps running out of input onto io.ErrUnexpectedEOF, so a truncated
// payload reads as truncation rather than as a bare EOF from somewhere deep in
// the node loop.
func shortOr(err error) error {
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}
