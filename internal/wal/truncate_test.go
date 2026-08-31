package wal

import (
	"fmt"
	"os"
	"slices"
	"testing"
)

// Truncation is the only operation here that destroys data on purpose, so these
// tests care about two things in this order: that it never removes a record
// something still needs, and only then that it removes anything at all.

// writeSegments fills a log with count records, small enough that it rotates
// often, and returns the segment indexes on disk.
func writeSegments(t *testing.T, dir string, count int) []uint32 {
	t.Helper()

	w, err := Open(dir, Options{SyncPolicy: SyncNever, MaxSegmentBytes: 128})
	if err != nil {
		t.Fatal(err)
	}
	for i := range count {
		if _, err := w.Append(TypePut, fmt.Appendf(nil, "record-%04d", i)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	segs, err := listSegments(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) < 5 {
		t.Fatalf("fixture produced %d segments, need several to test truncation", len(segs))
	}
	return segs
}

// replaySeqs returns the sequences still readable from a log.
func replaySeqs(t *testing.T, dir string) []uint64 {
	t.Helper()

	var got []uint64
	if _, err := Replay(dir, Options{}, func(r Record) error {
		got = append(got, r.Seq)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return got
}

// TestTruncateKeepsEverythingAtOrAfterTheLine is the property that matters: a
// record at or after keepFromSeq must still be there afterwards.
func TestTruncateKeepsEverythingAtOrAfterTheLine(t *testing.T) {
	for _, keepFrom := range []uint64{2, 10, 25, 40} {
		t.Run(fmt.Sprintf("keepFrom=%d", keepFrom), func(t *testing.T) {
			dir := t.TempDir()
			writeSegments(t, dir, 50)

			if _, err := Truncate(dir, keepFrom, Options{}); err != nil {
				t.Fatal(err)
			}

			got := replaySeqs(t, dir)
			if len(got) == 0 {
				t.Fatal("truncation emptied the log")
			}
			// Nothing at or after the line may have been dropped.
			for want := keepFrom; want <= 50; want++ {
				if !slices.Contains(got, want) {
					t.Fatalf("sequence %d was deleted but is at or after keepFrom=%d", want, keepFrom)
				}
			}
			// What survives must still be contiguous and increasing: truncation
			// removes a prefix, never a hole from the middle.
			for i := 1; i < len(got); i++ {
				if got[i] != got[i-1]+1 {
					t.Fatalf("gap in the surviving log: %d then %d", got[i-1], got[i])
				}
			}
		})
	}
}

func TestTruncateActuallyRemovesSegments(t *testing.T) {
	dir := t.TempDir()
	before := writeSegments(t, dir, 50)

	removed, err := Truncate(dir, 40, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if removed == 0 {
		t.Fatal("truncating a 50-record log at 40 removed nothing")
	}

	after, err := listSegments(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != len(before)-removed {
		t.Fatalf("%d segments left, expected %d - %d", len(after), len(before), removed)
	}
	// A prefix was removed, so what remains is the tail of the original list.
	for i, idx := range after {
		if want := before[len(before)-len(after)+i]; idx != want {
			t.Fatalf("segment %d is %d, want %d — the wrong files were removed", i, idx, want)
		}
	}
}

// TestTruncateNeverRemovesTheNewestSegment: nothing follows it, so nothing can
// vouch for where it ends — and it is the one a writer appends to.
func TestTruncateNeverRemovesTheNewestSegment(t *testing.T) {
	dir := t.TempDir()
	before := writeSegments(t, dir, 50)

	// Far past every record in the log.
	if _, err := Truncate(dir, 1_000_000, Options{}); err != nil {
		t.Fatal(err)
	}
	after, err := listSegments(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != 1 {
		t.Fatalf("%d segments survived an unbounded truncation, want 1", len(after))
	}
	if after[0] != before[len(before)-1] {
		t.Fatalf("kept segment %d, want the newest (%d)", after[0], before[len(before)-1])
	}
}

func TestTruncateNoOps(t *testing.T) {
	t.Run("sequence 0 or 1", func(t *testing.T) {
		// Sequences start at 1, so nothing can lie below it.
		dir := t.TempDir()
		before := writeSegments(t, dir, 50)
		for _, keep := range []uint64{0, 1} {
			removed, err := Truncate(dir, keep, Options{})
			if err != nil || removed != 0 {
				t.Fatalf("Truncate(keepFrom=%d) removed %d, %v", keep, removed, err)
			}
		}
		if after, _ := listSegments(dir); len(after) != len(before) {
			t.Fatalf("%d segments, want %d untouched", len(after), len(before))
		}
	})

	t.Run("single segment", func(t *testing.T) {
		dir := t.TempDir()
		w, err := Open(dir, Options{SyncPolicy: SyncNever})
		if err != nil {
			t.Fatal(err)
		}
		for range 5 {
			if _, err := w.Append(TypePut, []byte("x")); err != nil {
				t.Fatal(err)
			}
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}

		removed, err := Truncate(dir, 100, Options{})
		if err != nil || removed != 0 {
			t.Fatalf("removed %d from a single-segment log, %v", removed, err)
		}
	})

	t.Run("missing directory", func(t *testing.T) {
		removed, err := Truncate(t.TempDir()+"/nope", 100, Options{})
		if err != nil || removed != 0 {
			t.Fatalf("removed %d, %v", removed, err)
		}
	})

	t.Run("idempotent", func(t *testing.T) {
		dir := t.TempDir()
		writeSegments(t, dir, 50)

		first, err := Truncate(dir, 30, Options{})
		if err != nil {
			t.Fatal(err)
		}
		second, err := Truncate(dir, 30, Options{})
		if err != nil {
			t.Fatal(err)
		}
		if first == 0 {
			t.Fatal("the first truncation removed nothing")
		}
		if second != 0 {
			t.Fatalf("a repeated truncation removed %d more segments", second)
		}
	})
}

// TestTruncateDeclinesWhatItCannotRead: a segment whose successor cannot be read
// is one it must not reason about. Declining costs disk; guessing costs data.
func TestTruncateDeclinesWhatItCannotRead(t *testing.T) {
	dir := t.TempDir()
	segs := writeSegments(t, dir, 50)

	// Blank the second segment's first record, so its start sequence cannot be
	// determined. Segment one can then no longer be judged from it.
	path := segmentPath(dir, segs[1])
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for i := fileHeaderSize; i < len(data); i++ {
		data[i] = 0
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	// The third segment is still readable, so segments one and two are both
	// judged from it and may go — but nothing may be deleted on the strength of
	// the unreadable one.
	removed, err := Truncate(dir, 1_000_000, Options{})
	if err != nil {
		t.Fatal(err)
	}
	after, err := listSegments(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(after)+removed != len(segs) {
		t.Fatalf("accounting is off: %d left, %d removed, %d before", len(after), removed, len(segs))
	}
	if len(after) < 1 {
		t.Fatal("everything was deleted")
	}
}

// TestTruncateStopsAtTheFirstSegmentItMustKeep: once a segment has to stay,
// every later one does too, because sequences only increase.
func TestTruncateStopsAtTheFirstSegmentItMustKeep(t *testing.T) {
	dir := t.TempDir()
	writeSegments(t, dir, 50)

	// Truncate in two rounds and check the second removes only what the first
	// left behind, never anything past the line.
	if _, err := Truncate(dir, 20, Options{}); err != nil {
		t.Fatal(err)
	}
	mid := replaySeqs(t, dir)
	if _, err := Truncate(dir, 45, Options{}); err != nil {
		t.Fatal(err)
	}
	last := replaySeqs(t, dir)

	if len(last) > len(mid) {
		t.Fatalf("the log grew from %d records to %d", len(mid), len(last))
	}
	for want := uint64(45); want <= 50; want++ {
		found := false
		for _, s := range last {
			if s == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("sequence %d was lost across two truncations", want)
		}
	}
}

// TestWriterContinuesAfterTruncation: truncation runs while a database is live,
// so a writer must be unaffected by it.
func TestWriterContinuesAfterTruncation(t *testing.T) {
	dir := t.TempDir()
	writeSegments(t, dir, 50)

	w, err := Open(dir, Options{SyncPolicy: SyncNever, MaxSegmentBytes: 128, FirstSeq: 51})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Append(TypePut, []byte("before truncation")); err != nil {
		t.Fatal(err)
	}

	// Truncating below the writer's own segment, while it is open.
	if _, err := Truncate(dir, 40, Options{}); err != nil {
		t.Fatal(err)
	}

	if _, err := w.Append(TypePut, []byte("after truncation")); err != nil {
		t.Fatalf("appending after a truncation: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	got := replaySeqs(t, dir)
	if len(got) < 2 {
		t.Fatalf("only %d records survived", len(got))
	}
	if got[len(got)-1] != 52 {
		t.Fatalf("last sequence is %d, want 52", got[len(got)-1])
	}
}

// TestTruncateRefusesAnUnverifiedSequence is the reason firstSeq goes through
// the ordinary reader rather than pulling the header apart: the sequence is
// covered by the record's checksum, and this number authorizes deleting files.
// A flipped bit in it must not be trusted.
func TestTruncateRefusesAnUnverifiedSequence(t *testing.T) {
	dir := t.TempDir()
	segs := writeSegments(t, dir, 50)

	// Corrupt the sequence field of the second segment's first record. The
	// checksum now fails, so its start is unknown rather than believed.
	path := segmentPath(dir, segs[1])
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	byteOrder.PutUint64(data[fileHeaderSize+crcSize+typeSize:], 1)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, ok := firstSeq(path, defaultMaxRecordBytes); ok {
		t.Fatal("a record whose sequence fails its checksum was accepted as a segment's start")
	}
}
