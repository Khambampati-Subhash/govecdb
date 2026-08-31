package govecdb

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const testDim = 16

func vec(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()
	}
	return v
}

// openDB opens a database in a fresh directory with fast, test-appropriate
// durability.
func openDB(t *testing.T, opts ...Option) (*DB, string) {
	t.Helper()

	dir := t.TempDir()
	db := openDBAt(t, dir, opts...)
	return db, dir
}

func openDBAt(t *testing.T, dir string, opts ...Option) *DB {
	t.Helper()

	all := append([]Option{WithDimension(testDim), WithSyncPolicy(SyncNever)}, opts...)
	db, err := Open(dir, all...)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

// fill adds n vectors and returns what was added.
func fill(t *testing.T, db *DB, n int, seed int64) []Vector {
	t.Helper()

	rng := rand.New(rand.NewSource(seed))
	out := make([]Vector, n)
	for i := range n {
		out[i] = Vector{
			ID:       fmt.Sprintf("v%d", i),
			Values:   vec(rng, testDim),
			Metadata: Metadata{"n": int64(i), "even": i%2 == 0},
		}
		if err := db.Add(out[i]); err != nil {
			t.Fatalf("Add %d: %v", i, err)
		}
	}
	return out
}

func TestAddGetSearch(t *testing.T) {
	db, _ := openDB(t)
	want := fill(t, db, 100, 1)

	if db.Len() != 100 {
		t.Fatalf("Len = %d, want 100", db.Len())
	}

	got, err := db.Get("v42")
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != "v42" {
		t.Fatalf("Get returned %q", got.ID)
	}
	if got.Metadata["n"] != int64(42) || got.Metadata["even"] != true {
		t.Fatalf("metadata = %+v", got.Metadata)
	}

	// A vector must find itself, which is the end-to-end check that the index,
	// the store and the API all agree about what was stored.
	for _, w := range []Vector{want[0], want[57], want[99]} {
		res, err := db.Search(SearchRequest{Query: w.Values, K: 1})
		if err != nil {
			t.Fatal(err)
		}
		if len(res) != 1 || res[0].ID != w.ID {
			t.Fatalf("searching for %s returned %+v", w.ID, res)
		}
		if res[0].Metadata["n"] == nil {
			t.Fatalf("result carries no metadata: %+v", res[0])
		}
	}
}

func TestSearchOrdersByDistance(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 200, 2)

	rng := rand.New(rand.NewSource(9))
	res, err := db.Search(SearchRequest{Query: vec(rng, testDim), K: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 10 {
		t.Fatalf("got %d results, want 10", len(res))
	}
	for i := 1; i < len(res); i++ {
		if res[i].Distance < res[i-1].Distance {
			t.Fatalf("result %d is closer than %d — results are not sorted", i, i-1)
		}
	}
}

func TestSearchFewerVectorsThanK(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 3, 3)

	rng := rand.New(rand.NewSource(4))
	res, err := db.Search(SearchRequest{Query: vec(rng, testDim), K: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 3 {
		t.Fatalf("got %d results from a 3-vector database, want 3", len(res))
	}
}

func TestReplaceAndDelete(t *testing.T) {
	db, _ := openDB(t)
	rng := rand.New(rand.NewSource(5))

	first := vec(rng, testDim)
	if err := db.Add(Vector{ID: "a", Values: first, Metadata: Metadata{"v": int64(1)}}); err != nil {
		t.Fatal(err)
	}
	second := vec(rng, testDim)
	if err := db.Add(Vector{ID: "a", Values: second, Metadata: Metadata{"v": int64(2)}}); err != nil {
		t.Fatal(err)
	}

	if db.Len() != 1 {
		t.Fatalf("Len = %d after replacing one id, want 1", db.Len())
	}
	got, err := db.Get("a")
	if err != nil {
		t.Fatal(err)
	}
	if got.Metadata["v"] != int64(2) {
		t.Fatalf("metadata was not replaced: %+v", got.Metadata)
	}
	// A replacement tombstones the old slot, which is the cost compaction pays
	// back later.
	if s := db.Stats(); s.Deleted != 1 {
		t.Fatalf("replacing left %d tombstones, want 1", s.Deleted)
	}

	if err := db.Delete("a"); err != nil {
		t.Fatal(err)
	}
	if db.Len() != 0 {
		t.Fatalf("Len = %d after deleting the only vector", db.Len())
	}
	if _, err := db.Get("a"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Get after Delete = %v, want ErrNotFound", err)
	}

	// Idempotent: replay applies records more than once across a snapshot
	// boundary, so a second delete must not be an error.
	if err := db.Delete("a"); err != nil {
		t.Fatalf("deleting a missing id = %v, want nil", err)
	}
}

func TestCompactReclaimsSlots(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 100, 6)
	for i := range 40 {
		if err := db.Delete(fmt.Sprintf("v%d", i)); err != nil {
			t.Fatal(err)
		}
	}

	if s := db.Stats(); s.Deleted != 40 || s.DeadRatio() < 0.39 {
		t.Fatalf("before compaction: %+v, ratio %v", s, s.DeadRatio())
	}
	n, err := db.Compact()
	if err != nil {
		t.Fatal(err)
	}
	if n != 40 {
		t.Fatalf("Compact reclaimed %d, want 40", n)
	}
	if s := db.Stats(); s.Deleted != 0 || s.Live != 60 {
		t.Fatalf("after compaction: %+v", s)
	}

	// Compaction rebuilds the index; the surviving vectors have to remain
	// findable, and their metadata has to still be attached.
	res, err := db.Search(SearchRequest{Query: mustGet(t, db, "v50").Values, K: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].ID != "v50" || res[0].Metadata["n"] != int64(50) {
		t.Fatalf("after compaction: %+v", res)
	}
}

func mustGet(t *testing.T, db *DB, id string) Vector {
	t.Helper()
	v, err := db.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestAddBatchIsValidatedBeforeAnythingIsWritten(t *testing.T) {
	db, _ := openDB(t)
	rng := rand.New(rand.NewSource(7))

	batch := []Vector{
		{ID: "ok1", Values: vec(rng, testDim)},
		{ID: "ok2", Values: vec(rng, testDim)},
		{ID: "bad", Values: vec(rng, testDim+1)}, // wrong dimension
	}
	if err := db.AddBatch(batch); !errors.Is(err, ErrInvalidVector) {
		t.Fatalf("AddBatch = %v, want ErrInvalidVector", err)
	}
	// The good vectors ahead of the bad one must not have landed: validating the
	// whole batch first is the only reason to have a batch method at all.
	if db.Len() != 0 {
		t.Fatalf("a rejected batch wrote %d vectors", db.Len())
	}

	batch[2].Values = vec(rng, testDim)
	if err := db.AddBatch(batch); err != nil {
		t.Fatal(err)
	}
	if db.Len() != 3 {
		t.Fatalf("Len = %d after a valid batch of 3", db.Len())
	}
}

// --- Validation -------------------------------------------------------------

func TestAddRejects(t *testing.T) {
	db, _ := openDB(t)
	good := vec(rand.New(rand.NewSource(8)), testDim)

	for _, tc := range []struct {
		name string
		v    Vector
		want error
	}{
		{"empty id", Vector{ID: "", Values: good}, ErrInvalidVector},
		{"oversized id", Vector{ID: strings.Repeat("x", 513), Values: good}, ErrInvalidVector},
		{"invalid utf-8 id", Vector{ID: "\xff\xfe", Values: good}, ErrInvalidVector},
		{"no values", Vector{ID: "a"}, ErrInvalidVector},
		{"wrong dimension", Vector{ID: "a", Values: make([]float32, testDim+1)}, ErrInvalidVector},
		{
			// The one that would otherwise be silent: NaN compares false against
			// everything, so it corrupts the ordering the index rests on and no
			// error is ever raised.
			name: "NaN value",
			v:    Vector{ID: "a", Values: withValue(good, 3, float32(math.NaN()))},
			want: ErrInvalidVector,
		},
		{
			name: "infinite value",
			v:    Vector{ID: "a", Values: withValue(good, 0, float32(math.Inf(1)))},
			want: ErrInvalidVector,
		},
		{
			// int is not int64, and saying so beats guessing: a silent widening
			// here becomes a silent narrowing somewhere else.
			name: "unsupported metadata type",
			v:    Vector{ID: "a", Values: good, Metadata: Metadata{"n": 1}},
			want: ErrInvalidMetadata,
		},
		{
			name: "nested metadata",
			v:    Vector{ID: "a", Values: good, Metadata: Metadata{"m": map[string]any{}}},
			want: ErrInvalidMetadata,
		},
		{
			name: "empty metadata key",
			v:    Vector{ID: "a", Values: good, Metadata: Metadata{"": "x"}},
			want: ErrInvalidMetadata,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := db.Add(tc.v); !errors.Is(err, tc.want) {
				t.Fatalf("Add = %v, want %v", err, tc.want)
			}
			if db.Len() != 0 {
				t.Fatalf("a rejected Add stored something: Len = %d", db.Len())
			}
		})
	}
}

func withValue(v []float32, i int, f float32) []float32 {
	out := append([]float32(nil), v...)
	out[i] = f
	return out
}

func TestSearchRejects(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 10, 10)
	good := vec(rand.New(rand.NewSource(11)), testDim)

	for _, tc := range []struct {
		name string
		req  SearchRequest
		want error
	}{
		{"zero K", SearchRequest{Query: good, K: 0}, ErrInvalidRequest},
		{"negative K", SearchRequest{Query: good, K: -1}, ErrInvalidRequest},
		{"K over the limit", SearchRequest{Query: good, K: 10_001}, ErrInvalidRequest},
		{"negative Ef", SearchRequest{Query: good, K: 1, Ef: -1}, ErrInvalidRequest},
		{"Ef over the limit", SearchRequest{Query: good, K: 1, Ef: 100_001}, ErrInvalidRequest},
		{"target recall of 1", SearchRequest{Query: good, K: 1, TargetRecall: 1}, ErrInvalidRequest},
		{"wrong query dimension", SearchRequest{Query: make([]float32, 3), K: 1}, ErrInvalidVector},
		{"NaN in query", SearchRequest{Query: withValue(good, 1, float32(math.NaN())), K: 1}, ErrInvalidVector},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := db.Search(tc.req); !errors.Is(err, tc.want) {
				t.Fatalf("Search = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestSearchAutoEfGrowsWithTheCorpus pins why Ef defaults to zero rather than to
// a constant: recall at a fixed width falls as a corpus grows, so any number
// chosen today is wrong later.
func TestSearchAutoEfGrowsWithTheCorpus(t *testing.T) {
	db, _ := openDB(t)
	rng := rand.New(rand.NewSource(12))

	fill(t, db, 50, 12)
	small, err := db.opts.validateSearch(SearchRequest{Query: vec(rng, testDim), K: 10}, db.index.SuggestedEf)
	if err != nil {
		t.Fatal(err)
	}

	for i := range 3000 {
		if err := db.Add(Vector{ID: fmt.Sprintf("x%d", i), Values: vec(rng, testDim)}); err != nil {
			t.Fatal(err)
		}
	}
	large, err := db.opts.validateSearch(SearchRequest{Query: vec(rng, testDim), K: 10}, db.index.SuggestedEf)
	if err != nil {
		t.Fatal(err)
	}
	if large <= small {
		t.Fatalf("suggested ef did not grow with the corpus: %d then %d", small, large)
	}
}

func TestOpenRejects(t *testing.T) {
	t.Run("no dimension", func(t *testing.T) {
		if _, err := Open(t.TempDir()); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("Open without a dimension = %v", err)
		}
	})
	t.Run("empty directory path", func(t *testing.T) {
		if _, err := Open("", WithDimension(4)); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("Open(\"\") = %v", err)
		}
	})
	t.Run("M of one", func(t *testing.T) {
		if _, err := Open(t.TempDir(), WithDimension(4), WithM(1)); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("WithM(1) = %v", err)
		}
	})
	t.Run("unknown metric", func(t *testing.T) {
		if _, err := Open(t.TempDir(), WithDimension(4), WithMetric(Metric(9))); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("bad metric = %v", err)
		}
	})
	t.Run("nil option", func(t *testing.T) {
		if _, err := Open(t.TempDir(), WithDimension(4), nil); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("nil option = %v", err)
		}
	})
}

// TestOpenIsExclusiveWithinTheProcess covers the assumption the log and the
// snapshot store both make: one writer per directory. Two would interleave
// segment numbering and delete each other's temporary files.
func TestOpenIsExclusiveWithinTheProcess(t *testing.T) {
	dir := t.TempDir()
	db := openDBAt(t, dir)

	if _, err := Open(dir, WithDimension(testDim)); !errors.Is(err, ErrAlreadyOpen) {
		t.Fatalf("second Open = %v, want ErrAlreadyOpen", err)
	}
	// A different spelling of the same path must be caught too, or the guard is
	// trivially bypassed.
	if _, err := Open(dir+string(os.PathSeparator)+".", WithDimension(testDim)); !errors.Is(err, ErrAlreadyOpen) {
		t.Fatalf("second Open via a different path spelling = %v", err)
	}

	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	// And closing has to release the claim, or a restart would be impossible.
	again, err := Open(dir, WithDimension(testDim))
	if err != nil {
		t.Fatalf("Open after Close = %v", err)
	}
	if err := again.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestFailedOpenReleasesTheDirectory: a failed Open must not leave the path
// claimed, or one bad attempt would make the directory unopenable for the life
// of the process.
func TestFailedOpenReleasesTheDirectory(t *testing.T) {
	dir := t.TempDir()

	if _, err := Open(dir, WithDimension(testDim), WithM(1)); err == nil {
		t.Fatal("expected the bad config to be refused")
	}
	db, err := Open(dir, WithDimension(testDim))
	if err != nil {
		t.Fatalf("Open after a failed Open = %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestClosedDatabaseRefusesWork(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 5, 13)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	// Idempotent: Close belongs in a defer.
	if err := db.Close(); err != nil {
		t.Fatalf("second Close = %v", err)
	}

	good := vec(rand.New(rand.NewSource(14)), testDim)
	if err := db.Add(Vector{ID: "a", Values: good}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Add after Close = %v", err)
	}
	if err := db.Delete("a"); !errors.Is(err, ErrClosed) {
		t.Fatalf("Delete after Close = %v", err)
	}
	if _, err := db.Search(SearchRequest{Query: good, K: 1}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Search after Close = %v", err)
	}
	if _, err := db.Get("a"); !errors.Is(err, ErrClosed) {
		t.Fatalf("Get after Close = %v", err)
	}
	if err := db.Snapshot(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Snapshot after Close = %v", err)
	}
	if _, err := db.Compact(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Compact after Close = %v", err)
	}
}

// TestDirectoriesAreNotWorldReadable: embeddings are derived from whatever was
// embedded, so the directories this package creates are 0700. That is what
// protects the files inside them — another user cannot traverse a directory they
// cannot read, whatever the individual file modes are.
func TestDirectoriesAreNotWorldReadable(t *testing.T) {
	// A path Open has to create, since MkdirAll leaves an existing directory's
	// mode alone and t.TempDir() has already made one at 0700-or-looser.
	dir := filepath.Join(t.TempDir(), "db")
	openDBAt(t, dir)

	for _, p := range []string{dir, filepath.Join(dir, walSubdir), filepath.Join(dir, snapshotSubdir)} {
		info, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}
		if perm := info.Mode().Perm(); perm&0o077 != 0 {
			t.Fatalf("%s is %04o, want no group or other access", p, perm)
		}
	}
}

// TestExistingDirectoryModeIsLeftAlone is the other half of that decision. If an
// operator created the directory themselves, its mode is their call — silently
// tightening it would revoke access somebody granted on purpose, with no message
// anywhere. The subdirectories this package creates are still locked down.
func TestExistingDirectoryModeIsLeftAlone(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "db")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	openDBAt(t, dir)

	info, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o755 {
		t.Fatalf("Open changed an existing directory from 0755 to %04o", perm)
	}
	sub, err := os.Stat(filepath.Join(dir, walSubdir))
	if err != nil {
		t.Fatal(err)
	}
	if perm := sub.Mode().Perm(); perm&0o077 != 0 {
		t.Fatalf("log directory is %04o inside a permissive parent", perm)
	}
}

// TestConcurrentUse is worth having under -race. Searches must run in parallel
// with writes without either producing a wrong answer.
func TestConcurrentUse(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 200, 15)

	var wg sync.WaitGroup
	for g := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(g)))
			for i := range 50 {
				switch i % 4 {
				case 0:
					if err := db.Add(Vector{
						ID:       fmt.Sprintf("g%d-%d", g, i),
						Values:   vec(rng, testDim),
						Metadata: Metadata{"g": int64(g)},
					}); err != nil {
						t.Error(err)
						return
					}
				case 1:
					_ = db.Delete(fmt.Sprintf("v%d", i))
				case 2:
					if _, err := db.Search(SearchRequest{Query: vec(rng, testDim), K: 5}); err != nil {
						t.Error(err)
						return
					}
				case 3:
					_ = db.Stats()
					_, _ = db.Get(fmt.Sprintf("v%d", i))
				}
			}
		}()
	}
	wg.Wait()

	if s := db.Stats(); s.Live <= 0 {
		t.Fatalf("stats after concurrent use: %+v", s)
	}
}

func TestSnapshotIntervalRunsOnItsOwn(t *testing.T) {
	db, dir := openDB(t, WithSnapshotInterval(15*time.Millisecond))
	fill(t, db, 20, 16)

	// Polled on Stats rather than on the directory. The file lands before
	// snapSeq is published, so watching the filesystem races with the very
	// bookkeeping the test is checking — and it is Stats that a caller would
	// actually use to see that a snapshot happened.
	deadline := time.Now().Add(3 * time.Second)
	for db.Stats().SnapshotSeq == 0 {
		if time.Now().After(deadline) {
			t.Fatal("the snapshot interval never produced a snapshot")
		}
		time.Sleep(5 * time.Millisecond)
	}

	if entries, err := os.ReadDir(filepath.Join(dir, snapshotSubdir)); err != nil || len(entries) == 0 {
		t.Fatalf("SnapshotSeq advanced but no snapshot is on disk: %v, %d entries", err, len(entries))
	}
}

func TestSnapshotRetention(t *testing.T) {
	db, dir := openDB(t, WithSnapshotsKept(2))
	for i := range 5 {
		if err := db.Add(Vector{ID: fmt.Sprintf("v%d", i), Values: make([]float32, testDim)}); err != nil {
			t.Fatal(err)
		}
		if err := db.Snapshot(); err != nil {
			t.Fatal(err)
		}
	}

	entries, err := os.ReadDir(filepath.Join(dir, snapshotSubdir))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("%d snapshots on disk, want the 2 retained", len(entries))
	}
}

func TestStatsReportsTheReplayGap(t *testing.T) {
	db, _ := openDB(t)
	fill(t, db, 10, 17)

	if s := db.Stats(); s.SnapshotSeq != 0 || s.LastSeq != 10 {
		t.Fatalf("before any snapshot: %+v", s)
	}
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	s := db.Stats()
	if s.SnapshotSeq != 10 || s.LastSeq != 10 {
		t.Fatalf("after a snapshot: %+v", s)
	}

	fill(t, db, 3, 18)
	// The gap between the two is how much log a restart would have to replay,
	// which is the number this is here to make visible.
	if s := db.Stats(); s.LastSeq-s.SnapshotSeq != 3 {
		t.Fatalf("replay gap = %d, want 3 (%+v)", s.LastSeq-s.SnapshotSeq, s)
	}
}
