package govecdb

import "github.com/khambampati-subhash/govecdb/internal/store"

// Metadata is the key/value data attached to a vector.
//
// Values must be one of string, bool, int64 or float64. That is a closed set on
// purpose: metadata is decoded from a disk this process does not control, and a
// decoder that reconstructs arbitrary types from names on the wire is a far
// larger surface than filtering needs. An unsupported type is rejected by Add
// with the offending key named, rather than failing later during a write.
//
// Note that int is not int64. Go's untyped constants make `map[string]any{"n": 1}`
// an int, which is a different type; Add says so rather than guessing, because a
// silent widening here would become a silent narrowing on some other platform.
type Metadata = store.Metadata

// Vector is one record: an identifier, its embedding, and optional metadata.
type Vector struct {
	// ID identifies the vector. Adding an existing ID replaces it.
	ID string

	// Values is the embedding. Its length must equal the configured Dimension,
	// and every element must be finite — see Add.
	Values []float32

	// Metadata is optional. Nil and empty mean the same thing.
	Metadata Metadata
}

// SearchRequest asks for the nearest vectors to a query.
type SearchRequest struct {
	// Query is the vector to search near. Same dimension rules as Vector.Values.
	Query []float32

	// K is how many results to return. Fewer come back if the database holds
	// fewer live vectors.
	K int

	// Ef is the search width: how many candidates the traversal keeps in flight.
	// Higher means better recall and a slower query, and it is clamped up to at
	// least K.
	//
	// Leave it zero and the width is chosen for you from the corpus size and
	// TargetRecall, which is the right default: recall at a fixed Ef *falls* as
	// a corpus grows, so any constant a caller picks today is wrong later.
	Ef int

	// TargetRecall is the recall a zero Ef aims for, treated as a floor.
	// Defaults to the database's SearchTargetRecall option when zero.
	TargetRecall float64
}

// Match is one search result.
type Match struct {
	// ID of the matching vector.
	ID string

	// Distance to the query, where smaller always means closer whatever the
	// metric. Not a similarity score: nothing here ever needs to branch on
	// whether bigger or smaller is better.
	//
	// The scale is the metric's own — squared distance for Euclidean, a negated
	// dot product for DotProduct — so distances are comparable within one query
	// and meaningless across metrics.
	Distance float32

	// Metadata attached to the vector, nil if it has none.
	Metadata Metadata
}

// Stats describes what the database is holding.
type Stats struct {
	// Live is how many vectors a search can return.
	Live int

	// Deleted is how many tombstoned slots the index still carries. Deletes and
	// replacements both create them, and only Compact reclaims them.
	Deleted int

	// Slots is Live + Deleted: what the index is actually paying for.
	Slots int

	// WithMetadata is how many live ids carry metadata.
	WithMetadata int

	// LastSeq is the highest write-ahead log sequence assigned so far. It is the
	// point recovery would resume from, and it advances on every write.
	LastSeq uint64

	// SnapshotSeq is the sequence covered by the newest snapshot on disk, or
	// zero if there is none. The gap between it and LastSeq is how much log a
	// restart would have to replay.
	SnapshotSeq uint64
}

// DeadRatio is the fraction of slots that are tombstones, in [0,1].
//
// It is the signal for when to Compact, and it is exposed rather than acted on
// because compaction stops the world: only the caller knows which moment can
// afford the pause. Around 0.5 is the point where compacting pays — the pause
// tracks survivors rather than garbage, so compacting early costs more and
// reclaims less.
func (s Stats) DeadRatio() float64 {
	if s.Slots == 0 {
		return 0
	}
	return float64(s.Deleted) / float64(s.Slots)
}
