package wal

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// The log is a directory of numbered segments rather than one growing file,
// because truncation is the operation that has to work: once a snapshot at
// sequence N is durable, every record below N is dead weight. Deleting whole
// files is something every filesystem does instantly; punching a hole in the
// front of a large file is not.

const (
	segmentPrefix = "wal-"
	segmentSuffix = ".log"
	// segmentDigits is the zero padding, which keeps `ls` in replay order for
	// the first million segments. It is not a limit: a larger index simply
	// renders wider, and listSegments sorts by number rather than trusting the
	// names to sort.
	segmentDigits = 6
)

// segmentName renders the file name for a segment index.
func segmentName(index uint32) string {
	return fmt.Sprintf("%s%0*d%s", segmentPrefix, segmentDigits, index, segmentSuffix)
}

// parseSegmentName extracts the index from a file name, reporting whether the
// name is one of ours at all. Anything else in the directory is ignored rather
// than being an error: a stray editor swapfile or .DS_Store should not stop a
// database from opening.
func parseSegmentName(name string) (uint32, bool) {
	if !strings.HasPrefix(name, segmentPrefix) || !strings.HasSuffix(name, segmentSuffix) {
		return 0, false
	}
	digits := name[len(segmentPrefix) : len(name)-len(segmentSuffix)]
	if len(digits) == 0 {
		return 0, false
	}
	n, err := strconv.ParseUint(digits, 10, 32)
	if err != nil {
		return 0, false
	}
	return uint32(n), true
}

// listSegments returns the segment indexes present in dir, ascending.
// A missing directory is not an error — it is an empty log.
func listSegments(dir string) ([]uint32, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var out []uint32
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if idx, ok := parseSegmentName(e.Name()); ok {
			out = append(out, idx)
		}
	}
	// Sorted by number, not by name. The zero padding makes the two agree only
	// up to segmentDigits: wal-1000000.log sorts before wal-999999.log, and
	// replay would read segments out of order, refuse the rewound sequence as
	// ErrOutOfOrder, and the database would never open again. A restart costs
	// a segment, so a collection reopened often gets there.
	slices.Sort(out)
	return out, nil
}

// segmentPath joins a directory and a segment index.
func segmentPath(dir string, index uint32) string {
	return filepath.Join(dir, segmentName(index))
}
