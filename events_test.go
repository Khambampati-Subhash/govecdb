package govecdb

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

// recorder collects events. It locks because events arrive from the snapshot
// timer and the calibrator as well as the test's own goroutine.
type recorder struct {
	mu     sync.Mutex
	events []Event
}

func (r *recorder) observe(e Event) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
}

// all returns every recorded event of type T.
func all[T Event](r *recorder) []T {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []T
	for _, e := range r.events {
		if t, ok := e.(T); ok {
			out = append(out, t)
		}
	}
	return out
}

func one[T Event](t *testing.T, r *recorder) T {
	t.Helper()
	got := all[T](r)
	if len(got) != 1 {
		var zero T
		t.Fatalf("%d %T events, want 1 (all: %v)", len(got), zero, r.events)
	}
	return got[0]
}

func TestRecoveredReportsWhatOpenRebuiltFrom(t *testing.T) {
	db, dir := openDB(t)
	fill(t, db, 30, 701)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	fill2(t, db, 30, 12, 702)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	var r recorder
	openDBAt(t, dir, WithObserver(r.observe))
	got := one[Recovered](t, &r)
	if got.SnapshotSeq != 30 || got.Replayed != 12 {
		t.Fatalf("Recovered = %+v, want the snapshot at 30 and the 12 records after it", got)
	}
	if got.Segments == 0 || got.Took <= 0 {
		t.Fatalf("Recovered = %+v, want segments read and a duration", got)
	}
}

func TestTornLogIsReported(t *testing.T) {
	db, dir := openDB(t)
	fill(t, db, 20, 703)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	segments, err := filepath.Glob(filepath.Join(dir, walSubdir, "*.log"))
	if err != nil || len(segments) == 0 {
		t.Fatalf("no segments: %v", err)
	}
	last := segments[len(segments)-1]
	info, err := os.Stat(last)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(last, info.Size()-12); err != nil {
		t.Fatal(err)
	}

	var r recorder
	openDBAt(t, dir, WithObserver(r.observe))
	tear := one[TornLog](t, &r)
	if tear.Segment != filepath.Base(last) || tear.Cause == nil || tear.Offset <= 0 || tear.Discarded <= 0 {
		t.Fatalf("TornLog = %+v, want %s with an offset, a size and a cause", tear, filepath.Base(last))
	}
	if got := one[Recovered](t, &r); got.Replayed != 19 {
		t.Fatalf("Replayed = %d, want 19 of 20 after the tear", got.Replayed)
	}
}

func TestRejectedSnapshotIsReported(t *testing.T) {
	db, dir := openDB(t, WithSnapshotsKept(3))
	fill(t, db, 20, 704)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	fill2(t, db, 20, 10, 705)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	snaps := snapshotFiles(t, dir)
	newest := snaps[len(snaps)-1]
	corrupt(t, newest)

	var r recorder
	openDBAt(t, dir, WithObserver(r.observe), WithSnapshotsKept(3))
	rej := one[SnapshotRejected](t, &r)
	if rej.Path != newest || rej.Seq != 30 || rej.Cause == nil {
		t.Fatalf("SnapshotRejected = %+v, want %s at seq 30", rej, newest)
	}
	if got := one[Recovered](t, &r); got.SnapshotSeq != 20 {
		t.Fatalf("recovered from seq %d, want the fallback at 20", got.SnapshotSeq)
	}
}

func TestSnapshotTakenIsReported(t *testing.T) {
	var r recorder
	db, _ := openDB(t, WithObserver(r.observe), WithMaxSegmentBytes(2048), WithSnapshotsKept(1))
	fill(t, db, 40, 706)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	fill2(t, db, 40, 40, 707)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}

	taken := all[SnapshotTaken](&r)
	if len(taken) != 2 {
		t.Fatalf("%d SnapshotTaken events, want 2", len(taken))
	}
	second := taken[1]
	if second.Seq != 80 || second.Bytes <= 0 || second.Took <= 0 {
		t.Fatalf("SnapshotTaken = %+v, want seq 80 with a size and a duration", second)
	}
	// Keeping one means the second snapshot prunes the first, and the log
	// behind it is truncated.
	if second.Pruned != 1 || second.SegmentsRemoved == 0 {
		t.Fatalf("SnapshotTaken = %+v, want 1 pruned and some segments removed", second)
	}
}

func TestSnapshotFailedIsReported(t *testing.T) {
	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("needs a directory the process cannot write to")
	}
	var r recorder
	db, dir := openDB(t, WithObserver(r.observe))
	fill(t, db, 5, 708)

	snapDir := filepath.Join(dir, snapshotSubdir)
	if err := os.Chmod(snapDir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(snapDir, 0o700) })

	err := db.Snapshot()
	if err == nil {
		t.Fatal("Snapshot into an unwritable directory succeeded")
	}
	failed := one[SnapshotFailed](t, &r)
	if failed.Cause != err {
		t.Fatalf("event cause %v, returned %v", failed.Cause, err)
	}
	if len(all[SnapshotTaken](&r)) != 0 {
		t.Fatal("a failed snapshot was also reported as taken")
	}
	// A failed snapshot is a slow next start, not a durability failure.
	if err := db.Add(Vector{ID: "still-writable", Values: batchOf(1, 709)[0].Values}); err != nil {
		t.Fatalf("Add after a failed snapshot: %v", err)
	}
}

func TestTruncationSkippedIsReported(t *testing.T) {
	var r recorder
	db, dir := openDB(t, WithObserver(r.observe), WithMaxSegmentBytes(2048), WithSnapshotsKept(2))
	fill(t, db, 40, 710)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	oldest := snapshotFiles(t, dir)[0]
	corrupt(t, oldest)

	fill2(t, db, 40, 40, 711)
	if err := db.Snapshot(); err != nil {
		t.Fatal(err)
	}
	skip := one[TruncationSkipped](t, &r)
	if skip.Path != oldest || skip.Seq != 40 || skip.Cause == nil {
		t.Fatalf("TruncationSkipped = %+v, want %s at seq 40", skip, oldest)
	}
	if got := all[SnapshotTaken](&r); got[len(got)-1].SegmentsRemoved != 0 {
		t.Fatalf("a skipped truncation still removed %d segments", got[len(got)-1].SegmentsRemoved)
	}
}

// TestDurabilityFailureFiresOnce drives fail directly: a real fsync failure is
// not something a test can arrange portably, and what matters here is that the
// transition is reported once however many writes hit it afterwards.
func TestDurabilityFailureFiresOnce(t *testing.T) {
	var r recorder
	db, _ := openDB(t, WithObserver(r.observe))
	cause := errors.New("disk on fire")

	db.mu.Lock()
	db.fail(cause)
	db.fail(errors.New("a later symptom"))
	db.mu.Unlock()

	if got := one[DurabilityFailure](t, &r); !errors.Is(got.Cause, cause) {
		t.Fatalf("DurabilityFailure cause = %v, want the first one", got.Cause)
	}
	if err := db.Add(Vector{ID: "x", Values: batchOf(1, 713)[0].Values}); !errors.Is(err, ErrReadOnly) {
		t.Fatalf("Add after failure = %v, want ErrReadOnly", err)
	}
	if n := len(all[DurabilityFailure](&r)); n != 1 {
		t.Fatalf("%d DurabilityFailure events after a refused write, want still 1", n)
	}
}

func TestCalibratedIsReported(t *testing.T) {
	var r recorder
	db, _ := openDB(t, WithObserver(r.observe), WithEfCalibration(false))
	if err := db.AddBatch(batchOf(2000, 712)); err != nil {
		t.Fatal(err)
	}
	if err := db.Calibrate(); err != nil {
		t.Fatal(err)
	}
	got := one[Calibrated](t, &r)
	if got.Live != 2000 || got.Scale != db.Stats().EfScale || got.Took <= 0 {
		t.Fatalf("Calibrated = %+v, want 2000 live at scale %v", got, db.Stats().EfScale)
	}
}

// TestEveryEventPrints: an observer that logs e.String() is the common case,
// and a zero-valued event must not panic on the way to a log line.
func TestEveryEventPrints(t *testing.T) {
	for _, e := range []Event{
		Recovered{}, TornLog{}, SnapshotRejected{}, SnapshotTaken{}, SnapshotFailed{},
		TruncationSkipped{}, DurabilityFailure{}, Calibrated{}, CalibrationFailed{},
	} {
		if e.String() == "" {
			t.Errorf("%T prints as an empty string", e)
		}
	}
}

func corrupt(t *testing.T, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)/2] ^= 0xff
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
