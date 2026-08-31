package govecdb_test

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/khambampati-subhash/govecdb"
)

// The whole of the API in one place: open a directory, add some vectors, ask
// what is nearest.
func Example() {
	dir, err := os.MkdirTemp("", "govecdb-example")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	db, err := govecdb.Open(filepath.Join(dir, "db"),
		govecdb.WithDimension(4),
		govecdb.WithMetric(govecdb.Cosine),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	docs := []govecdb.Vector{
		{ID: "cat", Values: []float32{1, 0, 0, 0}, Metadata: govecdb.Metadata{"kind": "animal"}},
		{ID: "dog", Values: []float32{0.9, 0.1, 0, 0}, Metadata: govecdb.Metadata{"kind": "animal"}},
		{ID: "car", Values: []float32{0, 0, 1, 0}, Metadata: govecdb.Metadata{"kind": "vehicle"}},
	}
	if err := db.AddBatch(docs); err != nil {
		log.Fatal(err)
	}

	// Leaving Ef zero lets the search width be chosen from the corpus size,
	// which is what keeps recall steady as the database grows.
	matches, err := db.Search(govecdb.SearchRequest{
		Query: []float32{1, 0, 0, 0},
		K:     2,
	})
	if err != nil {
		log.Fatal(err)
	}
	for _, m := range matches {
		fmt.Printf("%s (%s)\n", m.ID, m.Metadata["kind"])
	}

	// Output:
	// cat (animal)
	// dog (animal)
}

// Durability is a knob. The zero value is the safe one, so choosing anything
// else should be deliberate — and this is what the choice looks like.
func ExampleWithSyncPolicy() {
	dir, err := os.MkdirTemp("", "govecdb-example")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	// SyncInterval batches fsyncs onto a timer: roughly a thousand times the
	// write throughput, at the cost of losing up to one interval of
	// acknowledged writes to a crash.
	db, err := govecdb.Open(filepath.Join(dir, "db"),
		govecdb.WithDimension(8),
		govecdb.WithSyncPolicy(govecdb.SyncInterval),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	if err := db.Add(govecdb.Vector{ID: "a", Values: make([]float32, 8)}); err != nil {
		log.Fatal(err)
	}
	// Sync draws a line: after it returns, everything written so far is durable
	// whatever the policy.
	if err := db.Sync(); err != nil {
		log.Fatal(err)
	}
	fmt.Println(db.Len())

	// Output:
	// 1
}

// A snapshot is what bounds how long a restart takes. Without one, recovery
// replays the whole log and rebuilds the index; with one, it loads a graph.
func ExampleDB_Snapshot() {
	dir, err := os.MkdirTemp("", "govecdb-example")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "db")

	db, err := govecdb.Open(path, govecdb.WithDimension(4))
	if err != nil {
		log.Fatal(err)
	}
	for i := range 100 {
		if err := db.Add(govecdb.Vector{
			ID:     fmt.Sprintf("v%d", i),
			Values: []float32{float32(i), 1, 0, 0},
		}); err != nil {
			log.Fatal(err)
		}
	}
	if err := db.Snapshot(); err != nil {
		log.Fatal(err)
	}
	if err := db.Close(); err != nil {
		log.Fatal(err)
	}

	// Reopening loads the snapshot and replays only what came after it.
	db, err = govecdb.Open(path, govecdb.WithDimension(4))
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	stats := db.Stats()
	fmt.Println(stats.Live, stats.LastSeq-stats.SnapshotSeq)

	// Output:
	// 100 0
}

// Tombstones are the price of deletes and replacements alike, and compaction is
// the only thing that pays it back. The database exposes the signal and leaves
// the timing to you, because compaction stops the world.
func ExampleDB_Compact() {
	dir, err := os.MkdirTemp("", "govecdb-example")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	db, err := govecdb.Open(filepath.Join(dir, "db"), govecdb.WithDimension(4))
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	for i := range 10 {
		if err := db.Add(govecdb.Vector{
			ID:     fmt.Sprintf("v%d", i),
			Values: []float32{float32(i), 1, 0, 0},
		}); err != nil {
			log.Fatal(err)
		}
	}
	for i := range 6 {
		if err := db.Delete(fmt.Sprintf("v%d", i)); err != nil {
			log.Fatal(err)
		}
	}

	if db.Stats().DeadRatio() > 0.5 {
		reclaimed, err := db.Compact()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("reclaimed", reclaimed)
	}
	fmt.Println("live", db.Stats().Live, "dead", db.Stats().Deleted)

	// Output:
	// reclaimed 6
	// live 4 dead 0
}

// Errors are sentinels, so callers match the category and print the detail.
func ExampleDB_Add_validation() {
	dir, err := os.MkdirTemp("", "govecdb-example")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	db, err := govecdb.Open(filepath.Join(dir, "db"), govecdb.WithDimension(4))
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// int is not int64, and metadata values are a closed set of four types.
	err = db.Add(govecdb.Vector{
		ID:       "a",
		Values:   []float32{1, 0, 0, 0},
		Metadata: govecdb.Metadata{"count": 1},
	})
	fmt.Println(errors.Is(err, govecdb.ErrInvalidMetadata))

	// Vector values must be finite: a NaN compares false against everything and
	// would silently corrupt the ordering the index rests on.
	err = db.Add(govecdb.Vector{ID: "b", Values: []float32{1, 0, 0, float32(nan())}})
	fmt.Println(errors.Is(err, govecdb.ErrInvalidVector))

	// Output:
	// true
	// true
}

func nan() float64 {
	zero := 0.0
	return zero / zero
}
