package snapshot

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// Snapshots are named for the WAL sequence they cover rather than numbered in
// their own sequence, because that number is what a caller actually needs: it
// says where to resume replaying the log and which segments are safe to delete.
// A separate counter would mean opening a file to answer either question.

const (
	filePrefix = "snap-"
	fileSuffix = ".snap"
	// fileDigits is 20, which is every value a uint64 can take. Fixed width so
	// lexical order matches numeric order — `ls` and a naive sort both give the
	// newest-last ordering for free, which is worth the wide names.
	fileDigits = 20

	// tempPattern is what a snapshot is called while it is being written.
	// os.CreateTemp fills in the star, so two writers cannot collide, and the
	// name cannot parse as a finished snapshot.
	tempPattern = filePrefix + "*.tmp"
	tempSuffix  = ".tmp"
)

// Snapshot describes one snapshot file on disk.
type Snapshot struct {
	// Seq is the WAL sequence this snapshot includes through. Recovery replays
	// the log from Seq+1; WAL segments holding only sequences at or below it are
	// deletable.
	Seq uint64

	// Path is the file's full path.
	Path string

	// Bytes is the size of the file on disk, framing included.
	Bytes int64
}

// fileName renders the file name for a sequence.
func fileName(seq uint64) string {
	return fmt.Sprintf("%s%0*d%s", filePrefix, fileDigits, seq, fileSuffix)
}

// parseFileName extracts the sequence from a file name, reporting whether the
// name is one of ours at all. Anything else in the directory is ignored rather
// than being an error: a stray .DS_Store should not stop a database from opening.
func parseFileName(name string) (uint64, bool) {
	if !strings.HasPrefix(name, filePrefix) || !strings.HasSuffix(name, fileSuffix) {
		return 0, false
	}
	digits := name[len(filePrefix) : len(name)-len(fileSuffix)]
	if len(digits) == 0 {
		return 0, false
	}
	seq, err := strconv.ParseUint(digits, 10, 64)
	if err != nil {
		return 0, false
	}
	return seq, true
}

// filePath joins a directory and a sequence.
func filePath(dir string, seq uint64) string {
	return filepath.Join(dir, fileName(seq))
}

// List returns the snapshots in dir, newest first.
//
// It reads names and sizes only — nothing here is opened or verified, because
// listing is what a retention policy and a status line call, and neither should
// pay to hash gigabytes. Verification belongs to Load, which is the only caller
// about to trust the contents.
//
// A missing directory is not an error. It is a database that has never
// snapshotted, which is exactly what the first startup finds.
func List(dir string) ([]Snapshot, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("snapshot: read dir: %w", err)
	}

	var out []Snapshot
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		seq, ok := parseFileName(e.Name())
		if !ok {
			continue
		}
		info, err := e.Info()
		if err != nil {
			// The file went away between the listing and the stat. Something
			// else is managing this directory, which is worth reporting rather
			// than papering over — see the single-writer note on Prune.
			return nil, fmt.Errorf("snapshot: stat %s: %w", e.Name(), err)
		}
		out = append(out, Snapshot{Seq: seq, Path: filepath.Join(dir, e.Name()), Bytes: info.Size()})
	}

	// Newest first, which is the order both callers want: Load tries them in
	// this order, and Prune keeps a prefix of it.
	slices.Reverse(out)
	return out, nil
}

// Latest returns the highest-sequence snapshot in dir, if there is one.
//
// It does not verify the file. A caller that is about to *use* a snapshot wants
// Load, which verifies and falls back; Latest answers the cheaper question of
// what the newest one claims to be.
func Latest(dir string) (Snapshot, bool, error) {
	all, err := List(dir)
	if err != nil {
		return Snapshot{}, false, err
	}
	if len(all) == 0 {
		return Snapshot{}, false, nil
	}
	return all[0], true, nil
}
