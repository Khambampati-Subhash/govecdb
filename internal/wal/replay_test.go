package wal

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// These tests are about the half of durability the writer cannot demonstrate.
// Appending is only useful if the bytes come back, and they have to come back
// from files that power loss cut off mid-record — so most of what follows
// deliberately damages a log and asserts on what survives.

// replayAll collects a whole log, cloning as it goes because the payload a
// callback receives is only valid until the next record is read.
func replayAll(t *testing.T, dir string, opts Options) ([]Record, Result) {
	t.Helper()

	var got []Record
	res, err := Replay(dir, opts, func(r Record) error {
		got = append(got, r.Clone())
		return nil
	})
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	return got, res
}

// writeLog appends count records and closes cleanly, returning the payloads.
func writeLog(t *testing.T, dir string, opts Options, count int) [][]byte {
	t.Helper()

	w, err := Open(dir, opts)
	if err != nil {
		t.Fatal(err)
	}
	payloads := make([][]byte, count)
	for i := range count {
		payloads[i] = fmt.Appendf(nil, "record-%04d", i)
		if _, err := w.Append(TypePut, payloads[i]); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return payloads
}

func TestReplayRoundTrip(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(dir, Options{SyncPolicy: SyncNever})
	if err != nil {
		t.Fatal(err)
	}
	want := []Record{
		{Type: TypePut, Seq: 1, Payload: []byte("first")},
		{Type: TypeDelete, Seq: 2, Payload: []byte("first")},
		{Type: TypePut, Seq: 3, Payload: nil},
		{Type: TypeCheckpoint, Seq: 4, Payload: bytes.Repeat([]byte("c"), 300)},
	}
	for _, r := range want {
		if _, err := w.Append(r.Type, r.Payload); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	got, res := replayAll(t, dir, Options{})
	if len(got) != len(want) {
		t.Fatalf("replayed %d records, wrote %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Type != want[i].Type || got[i].Seq != want[i].Seq {
			t.Fatalf("record %d: %v/%d, want %v/%d", i, got[i].Type, got[i].Seq, want[i].Type, want[i].Seq)
		}
		if !bytes.Equal(got[i].Payload, want[i].Payload) {
			t.Fatalf("record %d payload: %q, want %q", i, got[i].Payload, want[i].Payload)
		}
	}
	if len(res.Tears) != 0 {
		t.Fatalf("a cleanly closed log reported damage: %v", res.Tears)
	}
	if res.LastSeq != 4 || res.NextSeq() != 5 {
		t.Fatalf("LastSeq %d, NextSeq %d, want 4 and 5", res.LastSeq, res.NextSeq())
	}
	if res.Segments != 1 {
		t.Fatalf("read %d segments, want 1", res.Segments)
	}
}

func TestReplayEmpty(t *testing.T) {
	t.Run("missing directory", func(t *testing.T) {
		// A database that has never been written to is not a broken one — this
		// is what the very first Open finds.
		res, err := Replay(filepath.Join(t.TempDir(), "nope"), Options{}, func(Record) error {
			t.Fatal("a missing log produced a record")
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if res.Records != 0 || res.NextSeq() != 1 {
			t.Fatalf("%+v", res)
		}
	})

	t.Run("segment with only a header", func(t *testing.T) {
		// What Open leaves behind when nothing is appended. Every restart makes
		// one of these, so it must not read as damage.
		dir := t.TempDir()
		w, err := Open(dir, Options{SyncPolicy: SyncNever})
		if err != nil {
			t.Fatal(err)
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}

		got, res := replayAll(t, dir, Options{})
		if len(got) != 0 {
			t.Fatalf("an empty segment produced %d records", len(got))
		}
		if res.Segments != 1 || len(res.Tears) != 0 {
			t.Fatalf("%+v", res)
		}
	})
}

func TestReplayAcrossSegments(t *testing.T) {
	dir := t.TempDir()
	const count = 40
	// Small segments so the log crosses several boundaries.
	want := writeLog(t, dir, Options{MaxSegmentBytes: 128, SyncPolicy: SyncNever}, count)

	segments, err := listSegments(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) < 4 {
		t.Fatalf("expected repeated rotation, got %d segments", len(segments))
	}

	got, res := replayAll(t, dir, Options{})
	if len(got) != count {
		t.Fatalf("replayed %d records across %d segments, wrote %d", len(got), len(segments), count)
	}
	for i, r := range got {
		if r.Seq != uint64(i+1) {
			t.Fatalf("record %d has seq %d — segments were replayed out of order", i, r.Seq)
		}
		if !bytes.Equal(r.Payload, want[i]) {
			t.Fatalf("record %d payload: %q, want %q", i, r.Payload, want[i])
		}
	}
	if res.Segments != len(segments) {
		t.Fatalf("Result reports %d segments, directory has %d", res.Segments, len(segments))
	}
}

// truncateSegment cuts bytes off the end of a segment, which is what power loss
// during a buffered write leaves behind: a record whose header made it to the
// platter and whose payload did not.
func truncateSegment(t *testing.T, dir string, idx uint32, drop int64) {
	t.Helper()

	path := segmentPath(dir, idx)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(path, info.Size()-drop); err != nil {
		t.Fatal(err)
	}
}

func TestReplayTornTail(t *testing.T) {
	// A record here is 17 bytes of header plus an 11-byte payload, so these cut
	// in four distinct places. The 28-byte case is the interesting one: it lands
	// exactly on a record boundary, which is not damage at all — the file ends
	// where a file legitimately ends, and there is nothing for recovery to
	// repair. Losing a record and detecting a tear are separate things.
	for _, tc := range []struct {
		name  string
		drop  int64
		lost  int // records the truncation destroys
		tears int
	}{
		{"payload cut short", 5, 1, 1},
		{"header cut short", 20, 1, 1},
		{"exactly on a record boundary", 28, 1, 0},
		{"two records lost", 40, 2, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			const count = 10
			want := writeLog(t, dir, Options{SyncPolicy: SyncNever, MaxSegmentBytes: 1 << 20}, count)
			truncateSegment(t, dir, 1, tc.drop)

			got, res := replayAll(t, dir, Options{})

			// The prefix must survive intact. A torn tail costs the records that
			// were being written, never the ones that were already there.
			if len(got) != count-tc.lost {
				t.Fatalf("replayed %d records, want the %d that were fully written", len(got), count-tc.lost)
			}
			for i, r := range got {
				if r.Seq != uint64(i+1) || !bytes.Equal(r.Payload, want[i]) {
					t.Fatalf("record %d came back as seq %d / %q", i, r.Seq, r.Payload)
				}
			}
			if res.NextSeq() != uint64(count-tc.lost)+1 {
				t.Fatalf("NextSeq = %d after losing %d records", res.NextSeq(), tc.lost)
			}

			if len(res.Tears) != tc.tears {
				t.Fatalf("got %d tears, want %d: %v", len(res.Tears), tc.tears, res.Tears)
			}
			if tc.tears == 0 {
				return
			}
			tear := res.Tears[0]
			if tear.Segment != 1 {
				t.Fatalf("tear reported in segment %d", tear.Segment)
			}
			if !errors.Is(tear.Cause, ErrShortRecord) {
				t.Fatalf("tear cause = %v, want ErrShortRecord", tear.Cause)
			}
			// The offset has to name the start of the damaged record, since that
			// is the point the log is logically truncated at.
			if tear.Offset+tear.Discarded != fileSize(t, segmentPath(dir, 1)) {
				t.Fatalf("tear %+v does not account for the whole file", tear)
			}
		})
	}
}

func fileSize(t *testing.T, path string) int64 {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info.Size()
}

// TestReplayStopsAtACorruptRecord covers rot rather than truncation: the file is
// the right length and a byte inside it changed. The checksum is the only thing
// standing between that and a wrong answer delivered confidently.
func TestReplayStopsAtACorruptRecord(t *testing.T) {
	dir := t.TempDir()
	const count = 10
	writeLog(t, dir, Options{SyncPolicy: SyncNever, MaxSegmentBytes: 1 << 20}, count)

	path := segmentPath(dir, 1)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Record 5 begins after the file header and four whole records.
	const recordSize = recordHeaderSize + 11
	corruptAt := fileHeaderSize + 4*recordSize + recordHeaderSize + 2
	data[corruptAt] ^= 0x01
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	got, res := replayAll(t, dir, Options{})
	if len(got) != 4 {
		t.Fatalf("replayed %d records, want the 4 before the corruption", len(got))
	}
	if len(res.Tears) != 1 || !errors.Is(res.Tears[0].Cause, ErrChecksum) {
		t.Fatalf("tears = %v, want one checksum mismatch", res.Tears)
	}
	if want := int64(fileHeaderSize + 4*recordSize); res.Tears[0].Offset != want {
		t.Fatalf("tear at offset %d, want %d", res.Tears[0].Offset, want)
	}
}

// TestReplayStopsAtZeroFilledSpace is the case the zero record type exists for.
// A filesystem that zero-fills a block after a crash hands back something that
// parses as a record of type 0 with a zero length — and nothing about that is a
// record, so nothing about it may decode as one.
func TestReplayStopsAtZeroFilledSpace(t *testing.T) {
	dir := t.TempDir()
	writeLog(t, dir, Options{SyncPolicy: SyncNever, MaxSegmentBytes: 1 << 20}, 5)

	f, err := os.OpenFile(segmentPath(dir, 1), os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(make([]byte, 4096)); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	got, res := replayAll(t, dir, Options{})
	if len(got) != 5 {
		t.Fatalf("replayed %d records, want 5 — zero-filled space decoded as data", len(got))
	}
	if len(res.Tears) != 1 {
		t.Fatalf("tears = %v, want one", res.Tears)
	}
	// The checksum catches it before the type does, which is the honest
	// diagnosis: those bytes are corruption, not an unknown record type.
	if !errors.Is(res.Tears[0].Cause, ErrChecksum) {
		t.Fatalf("tear cause = %v, want ErrChecksum", res.Tears[0].Cause)
	}
	if res.Tears[0].Discarded != 4096 {
		t.Fatalf("discarded %d bytes, want the 4096 that were zeroed", res.Tears[0].Discarded)
	}
}

// TestReplayContinuesPastATearInAnEarlierSegment is the test the whole tolerance
// rule exists for.
//
// Open always starts a new segment, so a second crash leaves exactly this shape:
// segment 1 torn by the first crash, segment 2 full of good records written
// after the restart. A reader that only forgave damage in the *last* segment
// would make a database unrecoverable from the situation the writer is designed
// to create.
func TestReplayContinuesPastATearInAnEarlierSegment(t *testing.T) {
	dir := t.TempDir()
	opts := Options{SyncPolicy: SyncNever, MaxSegmentBytes: 1 << 20}

	// First run, then a crash that cuts the tail off.
	writeLog(t, dir, opts, 6)
	truncateSegment(t, dir, 1, 10)

	// Restart. Recovery replays what survived and hands the sequence forward.
	_, first := replayAll(t, dir, Options{})
	if first.Records != 5 {
		t.Fatalf("first recovery replayed %d records, want 5", first.Records)
	}

	w, err := Open(dir, Options{SyncPolicy: SyncNever, FirstSeq: first.NextSeq()})
	if err != nil {
		t.Fatal(err)
	}
	if w.Segment() != 2 {
		t.Fatalf("reopened into segment %d, want a fresh 2", w.Segment())
	}
	for i := range 4 {
		if _, err := w.Append(TypePut, fmt.Appendf(nil, "after-%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	got, res := replayAll(t, dir, Options{})
	if len(got) != 9 {
		t.Fatalf("replayed %d records, want 5 survivors plus 4 written after the restart", len(got))
	}
	for i, r := range got {
		if r.Seq != uint64(i+1) {
			t.Fatalf("record %d has seq %d — the log is not contiguous across the tear", i, r.Seq)
		}
	}
	if !bytes.Equal(got[8].Payload, []byte("after-3")) {
		t.Fatalf("last record is %q — records written after the tear were lost", got[8].Payload)
	}
	if len(res.Tears) != 1 || res.Tears[0].Segment != 1 {
		t.Fatalf("tears = %v, want one in segment 1", res.Tears)
	}
}

// TestReplayToleratesRepeatedTears extends that to the shape a machine with a
// flaky power supply produces: crash, restart, crash, restart.
func TestReplayToleratesRepeatedTears(t *testing.T) {
	dir := t.TempDir()

	next := uint64(1)
	for run := range 3 {
		w, err := Open(dir, Options{SyncPolicy: SyncNever, MaxSegmentBytes: 1 << 20, FirstSeq: next})
		if err != nil {
			t.Fatal(err)
		}
		for i := range 4 {
			if _, err := w.Append(TypePut, fmt.Appendf(nil, "run%d-%d", run, i)); err != nil {
				t.Fatal(err)
			}
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
		// Cut the last record off, as a crash mid-append would.
		truncateSegment(t, dir, uint32(run+1), 10)

		_, res := replayAll(t, dir, Options{})
		next = res.NextSeq()
	}

	got, res := replayAll(t, dir, Options{})
	if len(got) != 9 {
		t.Fatalf("replayed %d records, want 3 runs of 3 survivors", len(got))
	}
	if len(res.Tears) != 3 {
		t.Fatalf("tears = %v, want one per crash", res.Tears)
	}
	for i, r := range got {
		if r.Seq != uint64(i+1) {
			t.Fatalf("record %d has seq %d — sequences did not stay contiguous across three crashes", i, r.Seq)
		}
	}
}

// TestReplayRefusesAnUnverifiedLength is the guard that separates a bit-flip
// from a wild allocation. A corrupt length arrives *before* the checksum that
// would disprove it, so it has to be refused on its face — against what the file
// actually holds, and against the configured cap.
func TestReplayRefusesAnUnverifiedLength(t *testing.T) {
	t.Run("longer than the file", func(t *testing.T) {
		dir := t.TempDir()
		writeLog(t, dir, Options{SyncPolicy: SyncNever, MaxSegmentBytes: 1 << 20}, 5)

		path := segmentPath(dir, 1)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		const recordSize = recordHeaderSize + 11
		lengthAt := fileHeaderSize + 3*recordSize + crcSize + typeSize + seqSize
		byteOrder.PutUint32(data[lengthAt:], 1<<30) // a gigabyte that is not there
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}

		got, res := replayAll(t, dir, Options{})
		if len(got) != 3 {
			t.Fatalf("replayed %d records, want the 3 before the corrupt length", len(got))
		}
		if len(res.Tears) != 1 || !errors.Is(res.Tears[0].Cause, ErrShortRecord) {
			t.Fatalf("tears = %v, want one short record", res.Tears)
		}
	})

	t.Run("longer than the cap", func(t *testing.T) {
		// A length that does fit inside the file, so only MaxRecordBytes stands
		// between it and a read of arbitrary size.
		dir := t.TempDir()
		writeLog(t, dir, Options{SyncPolicy: SyncNever, MaxSegmentBytes: 1 << 20}, 200)

		path := segmentPath(dir, 1)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		const recordSize = recordHeaderSize + 11
		lengthAt := fileHeaderSize + 3*recordSize + crcSize + typeSize + seqSize
		byteOrder.PutUint32(data[lengthAt:], 2000)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}

		got, res := replayAll(t, dir, Options{MaxRecordBytes: 64})
		if len(got) != 3 {
			t.Fatalf("replayed %d records, want the 3 before the corrupt length", len(got))
		}
		if len(res.Tears) != 1 || !errors.Is(res.Tears[0].Cause, ErrRecordTooLarge) {
			t.Fatalf("tears = %v, want one oversized record", res.Tears)
		}
	})
}

func TestReplayRefusesAForeignSegment(t *testing.T) {
	t.Run("not a wal", func(t *testing.T) {
		dir := t.TempDir()
		// Named like a segment, so it cannot be ignored the way a stray file is.
		if err := os.WriteFile(segmentPath(dir, 1), []byte("this is not a log at all"), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := Replay(dir, Options{}, func(Record) error { return nil })
		if !errors.Is(err, ErrBadMagic) {
			t.Fatalf("Replay = %v, want ErrBadMagic", err)
		}
	})

	t.Run("future version", func(t *testing.T) {
		// Refusing is the whole point of carrying a version field. Guessing at a
		// layout this build does not know is worse than failing.
		dir := t.TempDir()
		writeLog(t, dir, Options{SyncPolicy: SyncNever}, 3)

		path := segmentPath(dir, 1)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		byteOrder.PutUint16(data[magicSize:], formatVersion+1)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}

		_, err = Replay(dir, Options{}, func(Record) error { return nil })
		if !errors.Is(err, ErrUnsupportedVersion) {
			t.Fatalf("Replay = %v, want ErrUnsupportedVersion", err)
		}
	})

	t.Run("truncated file header", func(t *testing.T) {
		// A crash between creating a segment and writing its header. There is
		// nothing in the file to lose, so it is damage rather than a refusal.
		dir := t.TempDir()
		writeLog(t, dir, Options{SyncPolicy: SyncNever}, 3)
		if err := os.WriteFile(segmentPath(dir, 2), nil, 0o644); err != nil {
			t.Fatal(err)
		}

		got, res := replayAll(t, dir, Options{})
		if len(got) != 3 {
			t.Fatalf("replayed %d records, want 3", len(got))
		}
		if len(res.Tears) != 1 || res.Tears[0].Segment != 2 || res.Tears[0].Offset != 0 {
			t.Fatalf("tears = %v, want one at the start of segment 2", res.Tears)
		}
	})
}

// TestReplayDetectsRewoundSequences catches the mistake that recovery exists to
// prevent: reopening a log without carrying FirstSeq forward, so the numbering
// starts again on top of records that already used it.
func TestReplayDetectsRewoundSequences(t *testing.T) {
	dir := t.TempDir()
	writeLog(t, dir, Options{SyncPolicy: SyncNever}, 5)
	// The bug: FirstSeq left at its default rather than set to res.NextSeq().
	writeLog(t, dir, Options{SyncPolicy: SyncNever}, 5)

	res, err := Replay(dir, Options{}, func(Record) error { return nil })
	if !errors.Is(err, ErrOutOfOrder) {
		t.Fatalf("Replay = %v, want ErrOutOfOrder", err)
	}
	// The partial Result still has to be usable: a caller reporting the failure
	// needs to say how far recovery got.
	if res.Records != 5 || res.LastSeq != 5 {
		t.Fatalf("partial result = %+v, want the 5 records before the rewind", res)
	}
}

func TestReplayStopsWhenTheCallbackFails(t *testing.T) {
	dir := t.TempDir()
	writeLog(t, dir, Options{SyncPolicy: SyncNever}, 10)

	boom := errors.New("apply failed")
	seen := 0
	res, err := Replay(dir, Options{}, func(Record) error {
		seen++
		if seen == 4 {
			return boom
		}
		return nil
	})
	if !errors.Is(err, boom) {
		t.Fatalf("Replay = %v, want the callback's error", err)
	}
	if seen != 4 || res.Records != 4 {
		t.Fatalf("replay continued past a failing callback: seen %d, %+v", seen, res)
	}
}

// TestReplayPayloadsAreReused documents the contract that keeps replay free of
// per-record allocation: the bytes belong to the reader until the next record.
// Holding one without Clone is a bug, and it is a quiet one, so it is worth a
// test that shows the difference rather than a comment that claims it.
func TestReplayPayloadsAreReused(t *testing.T) {
	dir := t.TempDir()
	writeLog(t, dir, Options{SyncPolicy: SyncNever}, 20)

	var held, cloned []Record
	if _, err := Replay(dir, Options{}, func(r Record) error {
		held = append(held, r)
		cloned = append(cloned, r.Clone())
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	// Every clone still says what it said when it was handed over.
	for i, r := range cloned {
		if want := fmt.Sprintf("record-%04d", i); string(r.Payload) != want {
			t.Fatalf("clone %d = %q, want %q", i, r.Payload, want)
		}
	}
	// The uncloned ones do not, because they all point at the same buffer. If
	// this ever stops being true the contract has changed, and Replay's doc
	// comment and Record.Clone need to change with it.
	same := 0
	for _, r := range held {
		if bytes.Equal(r.Payload, held[len(held)-1].Payload) {
			same++
		}
	}
	if same != len(held) {
		t.Fatalf("%d of %d retained payloads still differ — the reader stopped reusing its buffer", len(held)-same, len(held))
	}
}

// TestRecoveryRoundTrip is the loop a database actually runs on startup: replay
// what is there, open the log at the sequence recovery reached, keep writing.
// Doing it twice is what proves the numbering survives a restart rather than
// merely surviving a single recovery.
func TestRecoveryRoundTrip(t *testing.T) {
	dir := t.TempDir()

	var applied []string
	for run := range 3 {
		applied = applied[:0]
		res, err := Replay(dir, Options{}, func(r Record) error {
			applied = append(applied, string(r.Payload))
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if want := run * 3; len(applied) != want {
			t.Fatalf("run %d replayed %d records, want %d", run, len(applied), want)
		}

		w, err := Open(dir, Options{SyncPolicy: SyncNever, FirstSeq: res.NextSeq()})
		if err != nil {
			t.Fatal(err)
		}
		for i := range 3 {
			seq, err := w.Append(TypePut, fmt.Appendf(nil, "run%d-%d", run, i))
			if err != nil {
				t.Fatal(err)
			}
			if want := uint64(run*3+i) + 1; seq != want {
				t.Fatalf("run %d append %d got seq %d, want %d", run, i, seq, want)
			}
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
	}

	got, res := replayAll(t, dir, Options{})
	if len(got) != 9 {
		t.Fatalf("final replay has %d records, want 9", len(got))
	}
	if res.LastSeq != 9 {
		t.Fatalf("LastSeq = %d, want 9", res.LastSeq)
	}
	for i, r := range got {
		if want := fmt.Sprintf("run%d-%d", i/3, i%3); string(r.Payload) != want {
			t.Fatalf("record %d = %q, want %q", i, r.Payload, want)
		}
	}
}

func TestTearString(t *testing.T) {
	// The string is what lands in a startup log line, so it has to name the
	// segment and the offset an operator would go looking at.
	got := Tear{Segment: 7, Offset: 128, Discarded: 44, Cause: ErrChecksum}.String()
	for _, want := range []string{"wal-000007.log", "128", "44", "checksum"} {
		if !bytes.Contains([]byte(got), []byte(want)) {
			t.Fatalf("Tear.String() = %q, missing %q", got, want)
		}
	}
}
