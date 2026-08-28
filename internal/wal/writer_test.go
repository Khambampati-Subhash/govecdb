package wal

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// scanSegment reads a segment file back into records, validating the file header
// and every checksum. It is a deliberately dumb reader living in the test: the
// real one, with torn-tail repair, is the next task, and a writer nobody can
// read back is a writer nobody has verified.
func scanSegment(t *testing.T, path string) []Record {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := decodeFileHeader(data); err != nil {
		t.Fatalf("%s: %v", filepath.Base(path), err)
	}

	var out []Record
	for off := fileHeaderSize; off < len(data); {
		if off+recordHeaderSize > len(data) {
			t.Fatalf("%s: header runs past end of file at offset %d", filepath.Base(path), off)
		}
		hdr := data[off : off+recordHeaderSize]
		_, typ, seq, length, err := decodeRecordHeader(hdr)
		if err != nil {
			t.Fatal(err)
		}
		start := off + recordHeaderSize
		if start+int(length) > len(data) {
			t.Fatalf("%s: payload runs past end of file at offset %d", filepath.Base(path), off)
		}
		payload := data[start : start+int(length)]
		if !verifyChecksum(hdr, payload) {
			t.Fatalf("%s: checksum mismatch at offset %d", filepath.Base(path), off)
		}
		out = append(out, Record{Type: typ, Seq: seq, Payload: append([]byte(nil), payload...)})
		off = start + int(length)
	}
	return out
}

// scanAll reads every segment in a directory, in order.
func scanAll(t *testing.T, dir string) []Record {
	t.Helper()

	indexes, err := listSegments(dir)
	if err != nil {
		t.Fatal(err)
	}
	var out []Record
	for _, idx := range indexes {
		out = append(out, scanSegment(t, segmentPath(dir, idx))...)
	}
	return out
}

func TestAppendRoundTrip(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}

	want := []Record{
		{Type: TypePut, Seq: 1, Payload: []byte("first")},
		{Type: TypePut, Seq: 2, Payload: []byte("second")},
		{Type: TypeDelete, Seq: 3, Payload: []byte("first")},
		{Type: TypePut, Seq: 4, Payload: nil},
	}
	for _, r := range want {
		seq, err := w.Append(r.Type, r.Payload)
		if err != nil {
			t.Fatal(err)
		}
		if seq != r.Seq {
			t.Fatalf("Append returned seq %d, want %d", seq, r.Seq)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	got := scanAll(t, dir)
	if len(got) != len(want) {
		t.Fatalf("read %d records, wrote %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Type != want[i].Type || got[i].Seq != want[i].Seq {
			t.Fatalf("record %d: got %v/%d, want %v/%d", i, got[i].Type, got[i].Seq, want[i].Type, want[i].Seq)
		}
		if !bytes.Equal(got[i].Payload, want[i].Payload) {
			t.Fatalf("record %d payload: got %q, want %q", i, got[i].Payload, want[i].Payload)
		}
	}
}

// TestSyncAlwaysIsDurableOnReturn is the promise SyncAlways makes: when Append
// returns, the bytes are on disk — not sitting in a bufio buffer waiting for a
// flush that a crash would cancel. Reading through a second handle while the
// writer is still open is what makes this a real check rather than a
// tautological one.
func TestSyncAlwaysIsDurableOnReturn(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(dir, Options{SyncPolicy: SyncAlways})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	if _, err := w.Append(TypePut, []byte("durable")); err != nil {
		t.Fatal(err)
	}

	// Deliberately no Sync and no Close before reading.
	got := scanAll(t, dir)
	if len(got) != 1 || !bytes.Equal(got[0].Payload, []byte("durable")) {
		t.Fatalf("record was not on disk when Append returned: %+v", got)
	}
}

// TestSyncNeverStillLandsOnClose covers the other end of the policy: buffering
// is allowed to delay the write, but never to lose it on a clean shutdown.
func TestSyncNeverStillLandsOnClose(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(dir, Options{SyncPolicy: SyncNever})
	if err != nil {
		t.Fatal(err)
	}
	for i := range 50 {
		if _, err := w.Append(TypePut, fmt.Appendf(nil, "record-%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	if got := scanAll(t, dir); len(got) != 50 {
		t.Fatalf("read %d records after Close, wrote 50", len(got))
	}
}

// TestSyncIntervalFlushesWithoutAnAppend is why the interval policy needs a
// timer rather than a check inside Append: the write that most needs flushing is
// the last one before traffic stops, and no later Append is coming to trigger it.
func TestSyncIntervalFlushesWithoutAnAppend(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(dir, Options{SyncPolicy: SyncInterval, SyncInterval: 10 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	if _, err := w.Append(TypePut, []byte("eventually")); err != nil {
		t.Fatal(err)
	}

	// Poll rather than sleep a fixed span, so a slow machine is not a failure.
	deadline := time.Now().Add(2 * time.Second)
	for {
		if data, err := os.ReadFile(segmentPath(dir, 1)); err == nil && len(data) > fileHeaderSize {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the interval flusher never reached the disk")
		}
		time.Sleep(2 * time.Millisecond)
	}
}

func TestSegmentRotation(t *testing.T) {
	dir := t.TempDir()
	// Room for a handful of small records per segment, so the test crosses
	// several boundaries rather than exactly one.
	w, err := Open(dir, Options{MaxSegmentBytes: 128, SyncPolicy: SyncNever})
	if err != nil {
		t.Fatal(err)
	}

	const count = 40
	for i := range count {
		if _, err := w.Append(TypePut, fmt.Appendf(nil, "payload-%02d", i)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	segments, err := listSegments(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) < 4 {
		t.Fatalf("expected the log to rotate repeatedly, got %d segments", len(segments))
	}

	// Rotation must not lose or reorder anything: the records read back across
	// every segment have to be the ones written, in order.
	got := scanAll(t, dir)
	if len(got) != count {
		t.Fatalf("read %d records across %d segments, wrote %d", len(got), len(segments), count)
	}
	for i, r := range got {
		if r.Seq != uint64(i+1) {
			t.Fatalf("record %d has seq %d — rotation reordered the log", i, r.Seq)
		}
		if want := fmt.Sprintf("payload-%02d", i); string(r.Payload) != want {
			t.Fatalf("record %d payload = %q, want %q", i, r.Payload, want)
		}
	}

	// No segment may hold only a header: rotating on an empty segment would
	// leave empty files behind forever.
	for _, idx := range segments {
		info, err := os.Stat(segmentPath(dir, idx))
		if err != nil {
			t.Fatal(err)
		}
		if info.Size() <= fileHeaderSize {
			t.Fatalf("%s contains nothing but a header", segmentName(idx))
		}
	}
}

// TestOversizedRecordGetsItsOwnSegment pins the decision that MaxSegmentBytes is
// a truncation granularity, not a limit on what can be stored. Refusing a record
// bigger than a segment would make the two settings secretly coupled.
func TestOversizedRecordGetsItsOwnSegment(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(dir, Options{MaxSegmentBytes: 64, SyncPolicy: SyncNever})
	if err != nil {
		t.Fatal(err)
	}

	big := bytes.Repeat([]byte("x"), 500)
	if _, err := w.Append(TypePut, []byte("small")); err != nil {
		t.Fatal(err)
	}
	if _, err := w.Append(TypePut, big); err != nil {
		t.Fatalf("a record larger than a segment must still be writable: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	got := scanAll(t, dir)
	if len(got) != 2 {
		t.Fatalf("read %d records, wrote 2", len(got))
	}
	if !bytes.Equal(got[1].Payload, big) {
		t.Fatalf("the oversized payload came back wrong: %d bytes", len(got[1].Payload))
	}
}

func TestAppendRejects(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(dir, Options{MaxRecordBytes: 32})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	t.Run("invalid type", func(t *testing.T) {
		if _, err := w.Append(TypeInvalid, []byte("x")); err == nil {
			t.Fatal("a zero record type was accepted")
		}
	})

	t.Run("too large", func(t *testing.T) {
		// The writer has to enforce the same cap the reader does, or a record
		// could be written that can never be read back.
		if _, err := w.Append(TypePut, bytes.Repeat([]byte("x"), 33)); err == nil {
			t.Fatal("an oversized record was accepted")
		}
	})

	t.Run("rejections leave no trace", func(t *testing.T) {
		// A refused append must not consume a sequence number or write bytes;
		// otherwise the log grows a gap that replay would stop at.
		if got := scanAll(t, dir); len(got) != 0 {
			t.Fatalf("refused appends wrote %d records", len(got))
		}
		seq, err := w.Append(TypePut, []byte("ok"))
		if err != nil {
			t.Fatal(err)
		}
		if seq != 1 {
			t.Fatalf("first successful append got seq %d, want 1 — a rejection burned a sequence number", seq)
		}
	})
}

// TestReopenStartsANewSegment is the property that keeps recovery simple. A
// segment whose tail was cut off by power loss ends in a partial record, and
// replay stops at the first record that fails its checksum — so appending after
// that point would bury good writes behind a permanent stopping point.
func TestReopenStartsANewSegment(t *testing.T) {
	dir := t.TempDir()

	w, err := Open(dir, Options{SyncPolicy: SyncNever})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Append(TypePut, []byte("before restart")); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	before, err := os.ReadFile(segmentPath(dir, 1))
	if err != nil {
		t.Fatal(err)
	}

	// Recovery would pass the sequence it replayed up to; here that is 1.
	w2, err := Open(dir, Options{SyncPolicy: SyncNever, FirstSeq: 2})
	if err != nil {
		t.Fatal(err)
	}
	if got := w2.Segment(); got != 2 {
		t.Fatalf("reopened into segment %d, want a fresh 2", got)
	}
	seq, err := w2.Append(TypePut, []byte("after restart"))
	if err != nil {
		t.Fatal(err)
	}
	if seq != 2 {
		t.Fatalf("sequence restarted at %d instead of continuing at 2", seq)
	}
	if err := w2.Close(); err != nil {
		t.Fatal(err)
	}

	after, err := os.ReadFile(segmentPath(dir, 1))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("reopening modified the previous segment")
	}

	got := scanAll(t, dir)
	if len(got) != 2 || got[0].Seq != 1 || got[1].Seq != 2 {
		t.Fatalf("log across a restart: %+v", got)
	}
}

func TestClosedWriterRefusesWork(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// Idempotent: Close lands in defers, and a shutdown path should not have to
	// track whether it already ran.
	if err := w.Close(); err != nil {
		t.Fatalf("second Close returned %v", err)
	}
	if _, err := w.Append(TypePut, []byte("x")); err != ErrClosed {
		t.Fatalf("Append after Close: %v, want ErrClosed", err)
	}
	if err := w.Sync(); err != ErrClosed {
		t.Fatalf("Sync after Close: %v, want ErrClosed", err)
	}
}

// TestConcurrentAppends is worth having under -race, and worth having at all
// because sequence assignment is the log's ordering invariant: whatever order
// goroutines arrive in, the numbers handed out must be unique and contiguous.
func TestConcurrentAppends(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(dir, Options{SyncPolicy: SyncNever})
	if err != nil {
		t.Fatal(err)
	}

	const (
		writers = 8
		each    = 100
	)
	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		seen = map[uint64]bool{}
	)
	for g := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range each {
				seq, err := w.Append(TypePut, fmt.Appendf(nil, "g%d-%d", g, i))
				if err != nil {
					t.Error(err)
					return
				}
				mu.Lock()
				if seen[seq] {
					t.Errorf("sequence %d handed out twice", seq)
				}
				seen[seq] = true
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if len(seen) != writers*each {
		t.Fatalf("got %d distinct sequences, want %d", len(seen), writers*each)
	}
	// Contiguous from 1: a gap would mean a record was lost, a repeat that two
	// records share an identity.
	for i := uint64(1); i <= writers*each; i++ {
		if !seen[i] {
			t.Fatalf("sequence %d was never assigned", i)
		}
	}
	if got := scanAll(t, dir); len(got) != writers*each {
		t.Fatalf("log holds %d records, %d were appended", len(got), writers*each)
	}
}

func TestLastSeqTracksAppends(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(dir, Options{SyncPolicy: SyncNever, FirstSeq: 100})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	// Before anything is written, the last assigned sequence is one below the
	// first that will be — recovery hands FirstSeq forward, so this has to line
	// up rather than reporting zero.
	if got := w.LastSeq(); got != 99 {
		t.Fatalf("LastSeq on an empty log = %d, want 99", got)
	}
	for range 5 {
		if _, err := w.Append(TypePut, []byte("x")); err != nil {
			t.Fatal(err)
		}
	}
	if got := w.LastSeq(); got != 104 {
		t.Fatalf("LastSeq = %d, want 104", got)
	}
}

func TestOpenIgnoresForeignFiles(t *testing.T) {
	dir := t.TempDir()
	// A stray file should not stop a database from opening.
	for _, name := range []string{".DS_Store", "notes.txt", "wal-.log", "wal-abc.log"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("junk"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	w, err := Open(dir, Options{SyncPolicy: SyncNever})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	if got := w.Segment(); got != 1 {
		t.Fatalf("foreign files influenced segment numbering: got %d, want 1", got)
	}
}

func TestSegmentNameRoundTrip(t *testing.T) {
	for _, idx := range []uint32{0, 1, 42, 999999, 1 << 20} {
		name := segmentName(idx)
		got, ok := parseSegmentName(name)
		if !ok {
			t.Fatalf("%q did not parse back", name)
		}
		if got != idx {
			t.Fatalf("%q parsed to %d, want %d", name, got, idx)
		}
	}
	for _, name := range []string{"", "wal-.log", "wal-x.log", "log-000001.log", "wal-000001.txt", "wal-000001"} {
		if _, ok := parseSegmentName(name); ok {
			t.Fatalf("%q was accepted as a segment name", name)
		}
	}
}

func TestNopWAL(t *testing.T) {
	// The stub still hands out monotonic sequences: one that returned zero every
	// time would hide exactly the ordering bugs it is standing in for.
	var n Nop
	for i := uint64(1); i <= 3; i++ {
		seq, err := n.Append(TypePut, []byte("x"))
		if err != nil {
			t.Fatal(err)
		}
		if seq != i {
			t.Fatalf("Nop handed out seq %d, want %d", seq, i)
		}
	}
	if err := n.Sync(); err != nil {
		t.Fatal(err)
	}
	if err := n.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestOptionDefaults(t *testing.T) {
	got := Options{}.withDefaults()
	if got.SyncPolicy != SyncAlways {
		t.Fatalf("zero SyncPolicy = %v, want always — the safe answer, not the fast one", got.SyncPolicy)
	}
	if got.MaxSegmentBytes != defaultMaxSegmentBytes || got.MaxRecordBytes != defaultMaxRecordBytes {
		t.Fatalf("size defaults not applied: %+v", got)
	}
	if got.FirstSeq != 1 || got.SyncInterval != defaultSyncInterval {
		t.Fatalf("defaults not applied: %+v", got)
	}

	// Explicit values must survive.
	set := Options{SyncPolicy: SyncNever, MaxSegmentBytes: 7, MaxRecordBytes: 9, FirstSeq: 11, SyncInterval: time.Second}
	if got := set.withDefaults(); got != set {
		t.Fatalf("withDefaults overwrote explicit options: %+v", got)
	}
}
