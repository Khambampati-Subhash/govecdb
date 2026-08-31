package govecdb

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// Truncation is the only thing this database does that destroys data on purpose,
// so these tests are about what must survive it rather than about what it frees.

func logSegments(t *testing.T, dir string) []string {
	t.Helper()

	segs, err := filepath.Glob(filepath.Join(dir, walSubdir, "*.log"))
	if err != nil {
		t.Fatal(err)
	}
	return segs
}

func snapshotFiles(t *testing.T, dir string) []string {
	t.Helper()

	snaps, err := filepath.Glob(filepath.Join(dir, snapshotSubdir, "*.snap"))
	if err != nil {
		t.Fatal(err)
	}
	return snaps
}

// TestLogDoesNotGrowForever is the defect this phase exists to fix. Writing far
// more than one segment's worth, snapshotting as it goes, must leave a bounded
// number of segments rather than one per batch forever.
func TestLogDoesNotGrowForever(t *testing.T) {
	dir := t.TempDir()
	db := openDBAt(t, dir, WithMaxSegmentBytes(2048), WithSnapshotsKept(2))

	rng := rand.New(rand.NewSource(400))
	var peak int
	for round := range 12 {
		for i := range 40 {
			if err := db.Add(Vector{
				ID:     fmt.Sprintf("r%d-%d", round, i),
				Values: vec(rng, testDim),
			}); err != nil {
				t.Fatal(err)
			}
		}
		if err := db.Snapshot(); err != nil {
			t.Fatal(err)
		}
		peak = max(peak, len(logSegments(t, dir)))
	}

	final := len(logSegments(t, dir))
	// 480 records at ~90 bytes each across 2 KiB segments is on the order of 20
	// segments if nothing is ever deleted. The exact number truncation settles
	// at depends on where snapshots land relative to segment boundaries, so the
	// assertion is on the shape: it stays small and does not track the total.
	if final > 6 {
		t.Fatalf("%d segments remain after 12 snapshots — the log is still growing", final)
	}
	if peak > 8 {
		t.Fatalf("the log peaked at %d segments; truncation is not keeping up", peak)
	}

	// And the data is all still there.
	if db.Len() != 480 {
		t.Fatalf("Len = %d, want 480", db.Len())
	}
	db = reopen(t, db, dir)
	if db.Len() != 480 {
		t.Fatalf("Len = %d after reopening a truncated log, want 480", db.Len())
	}
}

// TestTruncationLeavesTheOlderSnapshotUsable is the whole reason truncation
// follows the *oldest* retained snapshot rather than the newest.
//
// Keeping two snapshots is what makes a corrupt one survivable. That only works
// if the log still reaches back far enough for the older one to be replayed on
// top of — truncating to the newest would delete exactly those records, leaving
// a second copy that is paid for and cannot be used. So: take two snapshots, let
// truncation run, destroy the newest, and require full recovery from the older
// one plus what is left of the log.
func TestTruncationLeavesTheOlderSnapshotUsable(t *testing.T) {
	dir := t.TempDir()
	db := openDBAt(t, dir, WithMaxSegmentBytes(2048), WithSnapshotsKept(2))

	first := fill(t, db, 60, 401)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	second := fill2(t, db, 60, 60, 402)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	snaps := snapshotFiles(t, dir)
	if len(snaps) != 2 {
		t.Fatalf("%d snapshots retained, want 2", len(snaps))
	}
	// Glob sorts, and snapshots are named by zero-padded sequence, so the last
	// is the newest.
	newest := snaps[len(snaps)-1]
	data, err := os.ReadFile(newest)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)/2] ^= 0xff
	if err := os.WriteFile(newest, data, 0o600); err != nil {
		t.Fatal(err)
	}

	db = openDBAt(t, dir, WithMaxSegmentBytes(2048), WithSnapshotsKept(2))
	assertHolds(t, db, append(append([]Vector{}, first...), second...))
	if s := db.Stats(); s.SnapshotSeq != 60 {
		t.Fatalf("recovered from snapshot %d, want the older one at 60", s.SnapshotSeq)
	}
}

// TestTruncationIsSkippedWhenTheOldestSnapshotIsUnreadable: the question
// truncation asks is "may I delete the records this snapshot stands in for?",
// and a snapshot nobody can read cannot stand in for anything. A growing log is
// a disk problem; deleting those records would be a data problem.
func TestTruncationIsSkippedWhenTheOldestSnapshotIsUnreadable(t *testing.T) {
	dir := t.TempDir()
	db := openDBAt(t, dir, WithMaxSegmentBytes(2048), WithSnapshotsKept(2))

	want := fill(t, db, 60, 403)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	// Corrupt the only snapshot, then take another so there are two retained
	// with the oldest unreadable.
	snaps := snapshotFiles(t, dir)
	if len(snaps) != 1 {
		t.Fatalf("%d snapshots, want 1", len(snaps))
	}
	data, err := os.ReadFile(snaps[0])
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)/2] ^= 0xff
	if err := os.WriteFile(snaps[0], data, 0o600); err != nil {
		t.Fatal(err)
	}

	before := len(logSegments(t, dir))
	more := fill2(t, db, 60, 60, 404)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	after := len(logSegments(t, dir))

	if after < before {
		t.Fatalf("segments went from %d to %d — the log was truncated against a snapshot that cannot be read",
			before, after)
	}

	// And everything still recovers, since nothing was deleted.
	db = reopen(t, db, dir, WithMaxSegmentBytes(2048), WithSnapshotsKept(2))
	assertHolds(t, db, append(append([]Vector{}, want...), more...))
}

// TestTruncationSurvivesDeletesAndReplacements: a truncated log plus a snapshot
// has to reproduce the same state, including the records that removed things.
func TestTruncationSurvivesDeletesAndReplacements(t *testing.T) {
	dir := t.TempDir()
	db := openDBAt(t, dir, WithMaxSegmentBytes(1024), WithSnapshotsKept(2))

	rng := rand.New(rand.NewSource(405))
	for i := range 80 {
		if err := db.Add(Vector{ID: fmt.Sprintf("v%d", i), Values: vec(rng, testDim)}); err != nil {
			t.Fatal(err)
		}
	}
	for i := range 30 {
		if err := db.Delete(fmt.Sprintf("v%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	// Replacements after the deletes, so the log holds all three record shapes.
	for i := 30; i < 50; i++ {
		if err := db.Add(Vector{
			ID:       fmt.Sprintf("v%d", i),
			Values:   vec(rng, testDim),
			Metadata: Metadata{"replaced": true},
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}

	wantLen := db.Len()
	db = reopen(t, db, dir, WithMaxSegmentBytes(1024), WithSnapshotsKept(2))

	if db.Len() != wantLen {
		t.Fatalf("Len = %d after recovering a truncated log, want %d", db.Len(), wantLen)
	}
	for i := range 30 {
		if _, err := db.Get(fmt.Sprintf("v%d", i)); err == nil {
			t.Fatalf("v%d came back after a truncated recovery", i)
		}
	}
	for i := 30; i < 50; i++ {
		got, err := db.Get(fmt.Sprintf("v%d", i))
		if err != nil {
			t.Fatal(err)
		}
		if got.Metadata["replaced"] != true {
			t.Fatalf("v%d lost its replacement: %+v", i, got.Metadata)
		}
	}
}

// TestTruncationDoesNotBreakSequenceNumbering: after the head of the log is
// gone, a restart must still resume above every sequence ever handed out, or
// replay would refuse the log for going backwards.
func TestTruncationDoesNotBreakSequenceNumbering(t *testing.T) {
	dir := t.TempDir()
	db := openDBAt(t, dir, WithMaxSegmentBytes(1024), WithSnapshotsKept(1))

	rng := rand.New(rand.NewSource(406))
	for round := range 4 {
		for i := range 30 {
			if err := db.Add(Vector{
				ID:     fmt.Sprintf("r%d-%d", round, i),
				Values: vec(rng, testDim),
			}); err != nil {
				t.Fatal(err)
			}
		}
		if err := db.Snapshot(); err != nil {
			t.Fatal(err)
		}
		db = reopen(t, db, dir, WithMaxSegmentBytes(1024), WithSnapshotsKept(1))

		if want := uint64((round + 1) * 30); db.Stats().LastSeq != want {
			t.Fatalf("round %d: LastSeq = %d, want %d", round, db.Stats().LastSeq, want)
		}
	}
	if db.Len() != 120 {
		t.Fatalf("Len = %d, want 120", db.Len())
	}
}
