package govecdb

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// Recovery is the reason the log and the snapshots exist, so these tests care
// about one question: after a restart, does the database hold exactly what it
// held before, whether or not a snapshot was taken and whether or not the log
// was cut short.

// reopen closes a database and opens the same directory again.
func reopen(t *testing.T, db *DB, dir string, opts ...Option) *DB {
	t.Helper()

	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return openDBAt(t, dir, opts...)
}

// assertHolds checks that every vector is present with its metadata intact and
// findable by search.
func assertHolds(t *testing.T, db *DB, want []Vector) {
	t.Helper()

	if db.Len() != len(want) {
		t.Fatalf("Len = %d, want %d", db.Len(), len(want))
	}
	for _, w := range want {
		got, err := db.Get(w.ID)
		if err != nil {
			t.Fatalf("Get %s: %v", w.ID, err)
		}
		if len(got.Metadata) != len(w.Metadata) {
			t.Fatalf("%s metadata = %+v, want %+v", w.ID, got.Metadata, w.Metadata)
		}
		for k, v := range w.Metadata {
			if got.Metadata[k] != v {
				t.Fatalf("%s metadata[%q] = %v (%T), want %v (%T)",
					w.ID, k, got.Metadata[k], got.Metadata[k], v, v)
			}
		}
		res, err := db.Search(SearchRequest{Query: w.Values, K: 1})
		if err != nil {
			t.Fatalf("Search for %s: %v", w.ID, err)
		}
		if len(res) == 0 || res[0].ID != w.ID {
			t.Fatalf("%s is no longer findable after recovery: %+v", w.ID, res)
		}
	}
}

// TestRecoveryFromTheLogAlone is the path with no snapshot: everything is
// rebuilt by replaying every record.
func TestRecoveryFromTheLogAlone(t *testing.T) {
	db, dir := openDB(t)
	want := fill(t, db, 150, 100)

	db = reopen(t, db, dir)
	assertHolds(t, db, want)

	if s := db.Stats(); s.SnapshotSeq != 0 {
		t.Fatalf("no snapshot was taken but SnapshotSeq is %d", s.SnapshotSeq)
	}
}

// TestRecoveryFromASnapshot is the path a snapshot exists for: the index is
// loaded rather than rebuilt.
func TestRecoveryFromASnapshot(t *testing.T) {
	db, dir := openDB(t)
	want := fill(t, db, 150, 101)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}

	db = reopen(t, db, dir)
	assertHolds(t, db, want)

	if s := db.Stats(); s.SnapshotSeq != 150 {
		t.Fatalf("SnapshotSeq = %d after recovering from a snapshot at 150", s.SnapshotSeq)
	}
}

// TestRecoveryAcrossASnapshotBoundary is the interesting one: some records are
// inside the snapshot and some were written after it, so recovery has to load
// one and replay the other without applying anything twice.
func TestRecoveryAcrossASnapshotBoundary(t *testing.T) {
	db, dir := openDB(t)
	before := fill(t, db, 100, 102)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(103))
	after := make([]Vector, 50)
	for i := range after {
		after[i] = Vector{
			ID:       fmt.Sprintf("after%d", i),
			Values:   vec(rng, testDim),
			Metadata: Metadata{"phase": "after"},
		}
		if err := db.Add(after[i]); err != nil {
			t.Fatal(err)
		}
	}

	db = reopen(t, db, dir)
	assertHolds(t, db, append(append([]Vector{}, before...), after...))

	// Nothing may be applied twice. A PUT replayed over a vector the snapshot
	// already holds is an early return in the index and costs nothing, but one
	// that replaced a vector would tombstone a slot on every single start — so a
	// clean run must leave the index with no tombstones at all.
	if s := db.Stats(); s.Deleted != 0 {
		t.Fatalf("recovery created %d tombstones; records inside the snapshot were replayed", s.Deleted)
	}
	if s := db.Stats(); s.LastSeq != 150 {
		t.Fatalf("LastSeq = %d after recovering 150 records", s.LastSeq)
	}
}

// TestRecoveryContinuesTheSequence: writes after a restart must not reuse
// sequence numbers, or replay would refuse the log for going backwards.
func TestRecoveryContinuesTheSequence(t *testing.T) {
	db, dir := openDB(t)

	var all []Vector
	for run := range 3 {
		rng := rand.New(rand.NewSource(int64(200 + run)))
		for i := range 10 {
			v := Vector{ID: fmt.Sprintf("r%d-%d", run, i), Values: vec(rng, testDim)}
			if err := db.Add(v); err != nil {
				t.Fatal(err)
			}
			all = append(all, v)
		}
		if want := uint64((run + 1) * 10); db.Stats().LastSeq != want {
			t.Fatalf("run %d: LastSeq = %d, want %d", run, db.Stats().LastSeq, want)
		}
		db = reopen(t, db, dir)
	}

	assertHolds(t, db, all)
	if s := db.Stats(); s.LastSeq != 30 {
		t.Fatalf("LastSeq = %d after three runs of ten writes", s.LastSeq)
	}
}

// TestRecoveryReplaysDeletes: a delete has to survive a restart, or a vector
// somebody removed comes back.
func TestRecoveryReplaysDeletes(t *testing.T) {
	db, dir := openDB(t)
	want := fill(t, db, 50, 104)

	for i := range 20 {
		if err := db.Delete(fmt.Sprintf("v%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	db = reopen(t, db, dir)

	if db.Len() != 30 {
		t.Fatalf("Len = %d after recovering 50 adds and 20 deletes", db.Len())
	}
	for i := range 20 {
		if _, err := db.Get(fmt.Sprintf("v%d", i)); !errors.Is(err, ErrNotFound) {
			t.Fatalf("v%d came back from the dead: %v", i, err)
		}
	}
	assertHolds(t, db, want[20:])
}

// TestRecoveryReplaysReplacements: the last write for an id wins, and only the
// last one.
func TestRecoveryReplaysReplacements(t *testing.T) {
	db, dir := openDB(t)
	rng := rand.New(rand.NewSource(105))

	var last Vector
	for i := range 5 {
		last = Vector{ID: "a", Values: vec(rng, testDim), Metadata: Metadata{"gen": int64(i)}}
		if err := db.Add(last); err != nil {
			t.Fatal(err)
		}
	}

	db = reopen(t, db, dir)
	got, err := db.Get("a")
	if err != nil {
		t.Fatal(err)
	}
	if got.Metadata["gen"] != int64(4) {
		t.Fatalf("recovered generation %v, want the last one", got.Metadata["gen"])
	}
	if db.Len() != 1 {
		t.Fatalf("Len = %d after replaying five writes to one id", db.Len())
	}
}

// TestRecoveryFromATornLog is what a crash actually leaves: a record cut off
// mid-write. The prefix has to survive and the database has to open.
func TestRecoveryFromATornLog(t *testing.T) {
	db, dir := openDB(t)
	want := fill(t, db, 40, 106)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	// Cut the tail off the segment, exactly as power loss during a buffered
	// write would.
	segments, err := filepath.Glob(filepath.Join(dir, walSubdir, "*.log"))
	if err != nil || len(segments) == 0 {
		t.Fatalf("no segments: %v", err)
	}
	last := segments[len(segments)-1]
	info, err := os.Stat(last)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(last, info.Size()-12); err != nil {
		t.Fatal(err)
	}

	db = openDBAt(t, dir)
	// The last record is gone; every one before it must be intact. A torn tail
	// is a write that was never acknowledged, so losing exactly one is correct.
	if db.Len() != 39 {
		t.Fatalf("Len = %d after a torn tail, want 39 of 40", db.Len())
	}
	assertHolds(t, db, want[:39])

	// And the database has to be writable afterwards, at a sequence that does
	// not collide with what survived.
	if err := db.Add(Vector{ID: "after-tear", Values: want[0].Values}); err != nil {
		t.Fatalf("writing after recovering from a tear: %v", err)
	}
}

// TestRecoveryIgnoresACorruptSnapshot: a snapshot is a cache, and the log is the
// source of truth. A corrupt one costs a longer replay, not the database.
func TestRecoveryIgnoresACorruptSnapshot(t *testing.T) {
	db, dir := openDB(t)
	want := fill(t, db, 60, 107)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	snaps, err := filepath.Glob(filepath.Join(dir, snapshotSubdir, "*.snap"))
	if err != nil || len(snaps) == 0 {
		t.Fatalf("no snapshots: %v", err)
	}
	for _, p := range snaps {
		data, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		data[len(data)/2] ^= 0xff
		if err := os.WriteFile(p, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}

	db = openDBAt(t, dir)
	assertHolds(t, db, want)
	if s := db.Stats(); s.SnapshotSeq != 0 {
		t.Fatalf("a corrupt snapshot was accepted: SnapshotSeq = %d", s.SnapshotSeq)
	}
}

// TestRecoveryFallsBackToAnOlderSnapshot is why retention keeps more than one.
func TestRecoveryFallsBackToAnOlderSnapshot(t *testing.T) {
	db, dir := openDB(t, WithSnapshotsKept(3))
	want := fill(t, db, 30, 108)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	more := fill2(t, db, 30, 20, 109)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	snaps, err := filepath.Glob(filepath.Join(dir, snapshotSubdir, "*.snap"))
	if err != nil || len(snaps) != 2 {
		t.Fatalf("want 2 snapshots, got %d (%v)", len(snaps), err)
	}
	// Glob sorts, and snapshot names are zero-padded by sequence, so the last is
	// the newest. Corrupt only that one.
	newest := snaps[len(snaps)-1]
	data, err := os.ReadFile(newest)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)/2] ^= 0xff
	if err := os.WriteFile(newest, data, 0o600); err != nil {
		t.Fatal(err)
	}

	db = openDBAt(t, dir)
	assertHolds(t, db, append(append([]Vector{}, want...), more...))
	// Fell back to the older snapshot and replayed the rest of the log on top.
	if s := db.Stats(); s.SnapshotSeq != 30 {
		t.Fatalf("SnapshotSeq = %d, want the older snapshot at 30", s.SnapshotSeq)
	}
}

// fill2 adds n vectors numbered from `from`.
func fill2(t *testing.T, db *DB, from, n int, seed int64) []Vector {
	t.Helper()

	rng := rand.New(rand.NewSource(seed))
	out := make([]Vector, n)
	for i := range n {
		out[i] = Vector{
			ID:       fmt.Sprintf("v%d", from+i),
			Values:   vec(rng, testDim),
			Metadata: Metadata{"n": int64(from + i), "even": (from+i)%2 == 0},
		}
		if err := db.Add(out[i]); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

// TestReopenRefusesAChangedShape: dimension, metric and M are structural. A
// graph's edges were chosen under one set of rules, and searching it under
// another returns quietly wrong answers rather than failing — so reopening with
// different ones has to be refused.
func TestReopenRefusesAChangedShape(t *testing.T) {
	dir := t.TempDir()
	db := openDBAt(t, dir)
	fill(t, db, 20, 110)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name string
		opts []Option
	}{
		{"different dimension", []Option{WithDimension(testDim + 8)}},
		{"different metric", []Option{WithDimension(testDim), WithMetric(Euclidean)}},
		{"different M", []Option{WithDimension(testDim), WithM(32)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			all := append([]Option{WithSyncPolicy(SyncNever)}, tc.opts...)
			got, err := Open(dir, all...)
			if err == nil {
				got.Close()
				t.Fatal("reopening with a different shape was allowed")
			}
			if !errors.Is(err, ErrInvalidConfig) {
				t.Fatalf("Open = %v, want ErrInvalidConfig", err)
			}
		})
	}

	// Non-structural settings may change freely: they affect how new vectors are
	// inserted, not how existing ones are read.
	again, err := Open(dir, WithDimension(testDim), WithSyncPolicy(SyncNever), WithEfConstruction(64))
	if err != nil {
		t.Fatalf("changing EfConstruction between runs was refused: %v", err)
	}
	if err := again.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestRecoveryRefusesALogFromAnotherDatabase: the log is opaque bytes, so a
// payload of the wrong shape passes its checksum and still cannot be applied.
// Refusing beats loading a partial index and calling it recovered.
func TestRecoveryRefusesALogFromAnotherDatabase(t *testing.T) {
	dir := t.TempDir()
	db := openDBAt(t, dir)
	fill(t, db, 10, 111)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	// Same directory, different dimension, and no snapshot to catch it first —
	// so the mismatch has to be caught by the record decoder.
	got, err := Open(dir, WithDimension(testDim+8), WithSyncPolicy(SyncNever))
	if err == nil {
		got.Close()
		t.Fatal("a log from a differently shaped database was replayed")
	}
	if !errors.Is(err, ErrCorrupt) {
		t.Fatalf("Open = %v, want ErrCorrupt", err)
	}
}

// TestEmptyDatabaseRoundTrips: opening, closing and reopening a database that
// was never written to must work, and must not invent state.
func TestEmptyDatabaseRoundTrips(t *testing.T) {
	dir := t.TempDir()
	db := openDBAt(t, dir)
	if err := db.Snapshot(); err != nil {
		t.Fatalf("snapshotting an empty database: %v", err)
	}
	db = reopen(t, db, dir)

	if db.Len() != 0 {
		t.Fatalf("Len = %d on a recovered empty database", db.Len())
	}
	// And it has to be usable afterwards.
	if err := db.Add(Vector{ID: "first", Values: make([]float32, testDim)}); err != nil {
		t.Fatal(err)
	}
	if db.Len() != 1 {
		t.Fatalf("Len = %d after the first write", db.Len())
	}
}

// TestRecoveryPreservesEveryMetadataType exercises the closed value set across a
// restart, since encoding and decoding are separate code paths.
func TestRecoveryPreservesEveryMetadataType(t *testing.T) {
	db, dir := openDB(t)
	md := Metadata{
		"str":      "hello world",
		"unicode":  "héllo → 世界",
		"true":     true,
		"false":    false,
		"int":      int64(-9007199254740993),
		"maxint":   int64(math.MaxInt64),
		"float":    float64(3.141592653589793),
		"negfloat": float64(-1e300),
		"zero":     int64(0),
		"empty":    "",
	}
	v := Vector{ID: "kitchen-sink", Values: vec(rand.New(rand.NewSource(112)), testDim), Metadata: md}
	if err := db.Add(v); err != nil {
		t.Fatal(err)
	}

	// Once through the log, once through a snapshot: two different encoders.
	db = reopen(t, db, dir)
	assertMetadata(t, db, "kitchen-sink", md)

	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	db = reopen(t, db, dir)
	assertMetadata(t, db, "kitchen-sink", md)
}

func assertMetadata(t *testing.T, db *DB, id string, want Metadata) {
	t.Helper()

	got, err := db.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Metadata) != len(want) {
		t.Fatalf("%d keys, want %d: %+v", len(got.Metadata), len(want), got.Metadata)
	}
	for k, w := range want {
		g := got.Metadata[k]
		if g != w {
			t.Fatalf("metadata[%q] = %v (%T), want %v (%T)", k, g, g, w, w)
		}
	}
}
