package govecdb

import (
	"bufio"
	"io"

	"github.com/khambampati-subhash/govecdb/internal/hnsw"
)

// Index is the vector index a database is built on.
//
// It exists so DB depends on an abstraction rather than on HNSW specifically:
// a flat brute-force index for small corpora, or a quantized one, is a different
// implementation of the same contract and not a different database. That is the
// open/closed part of the design, and it is only real if the interface is stated
// in this package's own types — which is why nothing here mentions hnsw.
//
// Every method must be safe for concurrent use. DB serializes nothing on the
// index's behalf; it relies on searches running in parallel, which is the whole
// reason reads scale.
type Index interface {
	// Insert stores a vector under id, replacing any vector already there.
	// Implementations must copy: the caller's slice is not theirs to keep.
	Insert(id string, values []float32) error

	// Delete removes id, reporting whether it was there.
	Delete(id string) bool

	// Lookup returns the stored form of id's vector. For a normalizing metric
	// that is the unit vector, not what the caller passed in.
	Lookup(id string) ([]float32, bool)

	// Search returns the k nearest vectors to query, nearest first, with ef
	// controlling how wide the traversal keeps its candidate set.
	Search(query []float32, k, ef int) ([]Match, error)

	// SuggestedEf proposes a search width for k results at a target recall,
	// given how much data the index currently holds.
	SuggestedEf(k int, targetRecall float64) int

	// Len is how many vectors a search can return.
	Len() int

	// Stats reports live and tombstoned occupancy.
	Stats() (live, deleted, slots int)

	// Compact reclaims tombstoned slots and reports how many were freed. It is
	// allowed to stop the world.
	Compact() int
}

// hnswIndex adapts the HNSW graph to Index.
//
// The adapter is thin on purpose — it translates types and nothing else. Any
// logic that crept in here would be logic that a different Index implementation
// silently does not get, which is how an interface stops meaning anything.
type hnswIndex struct {
	g *hnsw.Graph
}

var _ Index = (*hnswIndex)(nil)

func newHNSWIndex(o options) (*hnswIndex, error) {
	g, err := hnsw.New(hnsw.Config{
		Dimension:      o.dimension,
		Metric:         o.metric.internal(),
		M:              o.m,
		EfConstruction: o.efConstruction,
		Alpha:          1.2,
		Seed:           o.seed,
	})
	if err != nil {
		return nil, err
	}
	return &hnswIndex{g: g}, nil
}

func (h *hnswIndex) Insert(id string, values []float32) error { return h.g.Insert(id, values) }
func (h *hnswIndex) Delete(id string) bool                    { return h.g.Delete(id) }
func (h *hnswIndex) Lookup(id string) ([]float32, bool)       { return h.g.Vector(id) }
func (h *hnswIndex) Len() int                                 { return h.g.Len() }
func (h *hnswIndex) Compact() int                             { return h.g.Compact() }

func (h *hnswIndex) SuggestedEf(k int, targetRecall float64) int {
	return h.g.SuggestedEf(k, targetRecall)
}

func (h *hnswIndex) Stats() (live, deleted, slots int) {
	s := h.g.Stats()
	return s.Live, s.Deleted, s.Slots
}

func (h *hnswIndex) Search(query []float32, k, ef int) ([]Match, error) {
	res, err := h.g.Search(query, k, ef)
	if err != nil {
		return nil, err
	}
	// Metadata is attached by the caller, which owns the store. Doing it here
	// would give the index a dependency on metadata it has no reason to have.
	out := make([]Match, len(res))
	for i, r := range res {
		out[i] = Match{ID: r.ID, Distance: r.Distance}
	}
	return out, nil
}

// writeTo and readFrom are how the index takes part in a snapshot. They are not
// on the Index interface: an implementation that cannot serialize itself is
// still a usable index, and demanding it would make the interface bigger than
// what most of the database needs. DB checks for them at snapshot time and says
// so plainly when they are absent.
type indexSerializer interface {
	WriteTo(w io.Writer) (int64, error)
}

func (h *hnswIndex) WriteTo(w io.Writer) (int64, error) { return h.g.WriteTo(w) }

// readIndex reconstructs an index from a snapshot section.
//
// It takes the shared *bufio.Reader so this section consumes exactly its own
// bytes: bufio.NewReaderSize hands back a reader it is given when the buffer is
// already large enough, so hnsw.Read reuses this one instead of wrapping it and
// swallowing bytes that belong to the section after.
func readIndex(br *bufio.Reader) (*hnswIndex, error) {
	g, err := hnsw.Read(br)
	if err != nil {
		return nil, err
	}
	return &hnswIndex{g: g}, nil
}

// config reports what the loaded index was built with, so recovery can refuse a
// database whose options no longer match what is on disk.
func (h *hnswIndex) config() (dimension int, metric Metric, m int) {
	c := h.g.Config()
	switch c.Metric {
	case hnsw.Euclidean:
		metric = Euclidean
	case hnsw.DotProduct:
		metric = DotProduct
	default:
		metric = Cosine
	}
	return c.Dimension, metric, c.M
}
