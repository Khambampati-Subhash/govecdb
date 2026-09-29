package wal

import (
	"bytes"
	"errors"
	"fmt"
	"testing"
)

// TestAppendBatchIsDurableOnReturn is TestSyncAlwaysIsDurableOnReturn for a
// batch: the single trailing fsync must cover every record, not just the last.
func TestAppendBatchIsDurableOnReturn(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(dir, Options{SyncPolicy: SyncAlways, FirstSeq: 10})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	// One reused buffer, as the database passes it: each record must be copied
	// out before the next call overwrites it.
	var buf []byte
	first, err := w.AppendBatch(TypePut, 5, func(i int) []byte {
		buf = fmt.Appendf(buf[:0], "rec-%d", i)
		return buf
	})
	if err != nil {
		t.Fatal(err)
	}
	if first != 10 {
		t.Fatalf("first seq = %d, want 10", first)
	}

	// Deliberately no Sync and no Close before reading.
	got := scanAll(t, dir)
	if len(got) != 5 {
		t.Fatalf("%d records on disk when AppendBatch returned, want 5", len(got))
	}
	for i, r := range got {
		if r.Seq != uint64(10+i) || !bytes.Equal(r.Payload, fmt.Appendf(nil, "rec-%d", i)) {
			t.Fatalf("record %d = seq %d %q", i, r.Seq, r.Payload)
		}
	}
	if seq, _ := w.Append(TypePut, []byte("after")); seq != 15 {
		t.Fatalf("Append after a batch got seq %d, want 15", seq)
	}
}

// TestAppendBatchRotates checks a batch is not one giant segment: rotation, and
// the fsync that precedes it, still happen per record.
func TestAppendBatchRotates(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(dir, Options{MaxSegmentBytes: 128, SyncPolicy: SyncAlways})
	if err != nil {
		t.Fatal(err)
	}
	const count = 40
	if _, err := w.AppendBatch(TypePut, count, func(i int) []byte {
		return fmt.Appendf(nil, "payload-%02d", i)
	}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	segments, err := listSegments(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) < 2 {
		t.Fatalf("%d segments, want rotation to have happened", len(segments))
	}
	if got := scanAll(t, dir); len(got) != count {
		t.Fatalf("%d records, want %d", len(got), count)
	}
}

// TestAppendBatchOversizedFailsTheWriter pins the one way AppendBatch differs
// from Append: a record refused mid-batch has predecessors already buffered, and
// letting a later Sync make them durable would persist a batch the caller was
// told had failed.
func TestAppendBatchOversizedFailsTheWriter(t *testing.T) {
	w, err := Open(t.TempDir(), Options{MaxRecordBytes: 16, SyncPolicy: SyncNever})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	_, err = w.AppendBatch(TypePut, 3, func(i int) []byte {
		if i == 1 {
			return make([]byte, 17)
		}
		return []byte("ok")
	})
	if !errors.Is(err, ErrRecordTooLarge) {
		t.Fatalf("err = %v, want ErrRecordTooLarge", err)
	}
	if _, err := w.Append(TypePut, []byte("x")); !errors.Is(err, ErrRecordTooLarge) {
		t.Fatalf("writer accepted an append after a failed batch: %v", err)
	}
}

func TestAppendBatchRejectsInvalidType(t *testing.T) {
	w, err := Open(t.TempDir(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	called := false
	if _, err := w.AppendBatch(0, 1, func(int) []byte { called = true; return nil }); !errors.Is(err, ErrInvalidType) {
		t.Fatalf("err = %v, want ErrInvalidType", err)
	}
	if called {
		t.Fatal("payload was produced for a batch refused up front")
	}
}
