package wal

import (
	"fmt"
	"os"
)

// Truncate deletes segments that hold no record at or after keepFromSeq, and
// reports how many files it removed.
//
// It is what stops a log growing forever. Once a snapshot covers sequence N,
// every record at or below N can be reconstructed from it, so the segments
// holding only those records are dead weight — and deleting whole files is
// something every filesystem does instantly, which is the reason the log is a
// directory of segments rather than one growing file.
//
// # How a segment is judged disposable
//
// A segment does not record the range of sequences it holds, and finding its
// *last* one means scanning it to the end. So the decision is made from the
// other direction: sequences increase across the whole log, so every record in
// segment i comes before every record in any later segment j. If some later j
// starts at or below keepFromSeq, then segment i ends below it too, and segment
// i is disposable.
//
// That needs one record read per segment instead of a full scan. It is also
// conservative — a segment whose last record happens to sit just below the line
// may survive one extra round because the next segment starts above it — and
// conservative is the right direction for a deletion.
//
// # What it refuses to reason about
//
// The newest segment is never deleted: nothing follows it, so nothing can vouch
// for where it ends, and it is also the one a writer is appending to. Neither is
// a segment whose successor cannot be read — an empty one left by a restart, or
// a torn first record. Truncation is an optimization, and a segment it cannot
// read is a segment it declines to reason about. Declining costs disk; guessing
// costs data.
//
// # The caller's constraint
//
// keepFromSeq must come from the **oldest retained** snapshot, not the newest.
// Retaining two snapshots is what makes a corrupt one survivable, and truncating
// to the newest would delete the records the older one still needs — leaving a
// second copy that is paid for and cannot be used.
//
// Safe to call while a Writer is appending: it only ever removes segments below
// the one being written.
func Truncate(dir string, keepFromSeq uint64, opts Options) (int, error) {
	opts = opts.withDefaults()

	// Sequences start at 1, so nothing can lie below 1 and there is nothing a
	// snapshot covering nothing could authorize.
	if keepFromSeq <= 1 {
		return 0, nil
	}

	indexes, err := listSegments(dir)
	if err != nil {
		return 0, fmt.Errorf("wal: list segments: %w", err)
	}
	if len(indexes) < 2 {
		// One segment is the active one, and there is nothing before it.
		return 0, nil
	}

	starts := make([]uint64, len(indexes))
	known := make([]bool, len(indexes))
	for i, idx := range indexes {
		starts[i], known[i] = firstSeq(segmentPath(dir, idx), opts.MaxRecordBytes)
	}

	// Oldest first, so an interrupted truncation leaves a contiguous run of the
	// newest segments rather than holes in the middle of the log.
	removed := 0
	for i := 0; i < len(indexes)-1; i++ {
		j, ok := nextKnown(known, i)
		// Sequences increase across segments, so the nearest later segment with
		// a readable start has the smallest one. If that is already past the
		// line, every later segment is too, and so is every later i.
		if !ok || starts[j] > keepFromSeq {
			break
		}
		if err := os.Remove(segmentPath(dir, indexes[i])); err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return removed, fmt.Errorf("wal: remove %s: %w", segmentName(indexes[i]), err)
		}
		removed++
	}

	// The removals are only durable once the directory is. Without this a crash
	// can resurrect a segment Truncate reported as gone — harmless for
	// correctness, since replay skips what a snapshot already covers, but a
	// retention policy that does not retain is worth getting right.
	if removed > 0 {
		if err := syncDir(dir); err != nil {
			return removed, err
		}
	}
	return removed, nil
}

// nextKnown finds the nearest segment after i whose first sequence could be read.
func nextKnown(known []bool, i int) (int, bool) {
	for j := i + 1; j < len(known); j++ {
		if known[j] {
			return j, true
		}
	}
	return 0, false
}

// firstSeq reads the sequence of a segment's first record, reporting whether it
// could be determined at all.
//
// It goes through the ordinary reader rather than pulling the header apart
// directly, which costs reading one record's payload and buys every check the
// reader already makes. That matters here more than anywhere: the sequence is
// covered by the record's checksum, and this number decides whether other files
// get deleted. Trusting an unverified one would let a single flipped bit
// authorize destroying a segment.
//
// Anything that is not a clean first record — an empty segment, a torn one, a
// file that is not a segment at all — comes back as "unknown" rather than an
// error. Truncation must never be the thing that fails a database.
func firstSeq(path string, maxRecord int) (uint64, bool) {
	r, err := openSegmentReader(path, maxRecord)
	if err != nil {
		return 0, false
	}
	defer r.close()

	ok, err := r.next()
	if err != nil || !ok {
		return 0, false
	}
	return r.rec.Seq, true
}
