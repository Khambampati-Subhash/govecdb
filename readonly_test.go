package govecdb

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

func TestReadOnlyServesReadsAndRefusesWrites(t *testing.T) {
	dir := t.TempDir()
	w := openDBAt(t, dir)
	added := fill(t, w, 40, 11)
	if err := w.Snapshot(); err != nil {
		t.Fatal(err)
	}
	// One after the snapshot, so the read-only open has to replay the log too.
	if err := w.Add(Vector{ID: "late", Values: added[0].Values}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	before := listTree(t, dir)

	db, err := Open(dir, WithDimension(testDim), WithReadOnly())
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if db.Len() != 41 {
		t.Fatalf("Len = %d, want 41", db.Len())
	}
	if _, err := db.Get("late"); err != nil {
		t.Fatalf("a record only in the log was not replayed: %v", err)
	}
	if got, err := db.Search(SearchRequest{Query: added[3].Values, K: 1}); err != nil || got[0].ID != "v3" {
		t.Fatalf("Search = %v, %v", got, err)
	}
	if st := db.Stats(); st.LastSeq != 41 {
		t.Fatalf("Stats.LastSeq = %d, want 41", st.LastSeq)
	}
	if err := db.Sync(); err != nil {
		t.Fatalf("Sync on a read-only database = %v, want nil", err)
	}

	for name, err := range map[string]error{
		"Add":      db.Add(added[0]),
		"AddBatch": db.AddBatch(added[:2]),
		"Delete":   db.Delete("v1"),
		"Snapshot": db.Snapshot(),
	} {
		if !errors.Is(err, ErrReadOnly) {
			t.Errorf("%s = %v, want ErrReadOnly", name, err)
		}
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	// The promise is that nothing in the directory was written — not a segment,
	// not a temp file.
	if after := listTree(t, dir); !slices.Equal(before, after) {
		t.Fatalf("a read-only open changed the directory:\nbefore %v\nafter  %v", before, after)
	}
}

func TestReadOnlyRefusesAMissingDirectory(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nope")
	if _, err := Open(dir, WithDimension(testDim), WithReadOnly()); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("Open = %v, want ErrInvalidConfig", err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("a read-only open created %s", dir)
	}
}

func TestReadOnlyRefusesASnapshotInterval(t *testing.T) {
	_, err := Open(t.TempDir(), WithDimension(testDim), WithReadOnly(), WithSnapshotInterval(time.Minute))
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("Open = %v, want ErrInvalidConfig", err)
	}
}

// TestFailedOpenReleasesTheLock: restore runs after the flock is taken, so an
// Open refused there must drop it — otherwise the directory would report itself
// locked by "another process" for as long as this one lives.
func TestFailedOpenReleasesTheLock(t *testing.T) {
	dir := t.TempDir()
	db := openDBAt(t, dir)
	fill(t, db, 3, 1)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := Open(dir, WithDimension(testDim*2)); !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("Open at the wrong dimension = %v, want ErrInvalidConfig", err)
	}
	again, err := Open(dir, WithDimension(testDim))
	if err != nil {
		t.Fatalf("Open after a refused Open = %v", err)
	}
	again.Close()
}

func listTree(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		out = append(out, fmt.Sprintf("%s %d %v", rel, info.Size(), info.ModTime()))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}
