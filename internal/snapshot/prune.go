package snapshot

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Prune deletes all but the newest keep snapshots, and any temporary files left
// behind by a crash. It returns how many files it removed.
//
// # Why keep more than one
//
// One snapshot is one copy, and a copy that fails its checksum is no copy at
// all. Retaining a second means a corrupt newest snapshot costs a longer WAL
// replay instead of the database — which is the whole reason Load falls back
// rather than failing. Keeping one is supported and is a choice to make
// deliberately, not by default.
//
// # The ordering constraint that makes the fallback real
//
// Falling back to an older snapshot only works if the WAL still holds records
// from that snapshot's sequence. So WAL truncation must be driven by the
// *oldest retained* snapshot, never the newest, and it must run after this — a
// caller that prunes to the newest snapshot and then truncates the log below it
// has just made the second copy unusable and kept paying to store it.
//
// This package does not enforce that, because the WAL is not its business. It is
// stated here because it is the part that is easy to get backwards.
//
// # Single writer
//
// Prune removes every temporary file it finds, so it must not run while another
// Create is in flight. That is not a limitation being papered over: a snapshot
// directory belongs to one database instance, the same way a WAL directory does
// — the WAL refuses a second writer outright with O_EXCL.
func Prune(dir string, keep int) (int, error) {
	if keep < 1 {
		return 0, fmt.Errorf("%w: asked to keep %d", ErrKeepTooFew, keep)
	}

	all, err := List(dir)
	if err != nil {
		return 0, err
	}

	removed := 0
	// List is newest first, so everything past keep is what to drop. Deleting
	// oldest first means an interrupted prune leaves a contiguous run of the
	// newest snapshots rather than holes in the middle of the history.
	for i := len(all) - 1; i >= keep; i-- {
		if err := os.Remove(all[i].Path); err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return removed, fmt.Errorf("snapshot: remove %s: %w", all[i].Path, err)
		}
		removed++
	}

	n, err := removeTemps(dir)
	removed += n
	if err != nil {
		return removed, err
	}

	// The removals are only durable once the directory is. Without this a crash
	// can resurrect a snapshot that Prune reported as gone, which is confusing
	// rather than dangerous — but a retention policy that does not retain is
	// worth getting right.
	if removed > 0 {
		if err := syncDir(dir); err != nil {
			return removed, err
		}
	}
	return removed, nil
}

// removeTemps deletes leftover temporary files. A crash between creating one and
// renaming it leaves a file nothing else will ever clean up, and nothing else
// will ever read either — Create's own failure path removes its temporary, so
// anything still here outlived the process that made it.
func removeTemps(dir string) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, fmt.Errorf("snapshot: read dir: %w", err)
	}

	removed := 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasPrefix(name, filePrefix) || !strings.HasSuffix(name, tempSuffix) {
			continue
		}
		if err := os.Remove(filepath.Join(dir, name)); err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return removed, fmt.Errorf("snapshot: remove %s: %w", name, err)
		}
		removed++
	}
	return removed, nil
}
