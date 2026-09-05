package wal

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// The crash harness: a real process, killed mid-write, with no chance to clean
// up after itself.
//
// Every other recovery test in this package damages a log by truncating and
// rewriting it. That reproduces the *shapes* power loss leaves behind, and it is
// how the interesting cases get covered deterministically — but it cannot
// reproduce the *timing*, because the damage is applied by code that ran after
// the writer stopped. This one does not simulate anything: it kills a process
// that is in the middle of appending and then asks the log what survived.
//
// SIGKILL specifically. It cannot be caught, blocked or handled, so no deferred
// Close runs, no buffer is flushed on the way out, and no signal handler gets to
// tidy up. A clean shutdown would prove nothing about power loss.
//
// The assertion is the durability contract stated exactly: under SyncAlways,
// Append returns only after fsync, so **every record the child acknowledged must
// be readable afterwards**. The child prints a line only after Append returns,
// which makes each line a promise the log has already committed to. Records
// written after the last line the parent read are unconstrained — they may or
// may not be there, and either is correct.

const (
	crashDirEnv     = "GOVECDB_WAL_CRASH_DIR"
	ackPrefix       = "ACK "
	crashPayloadFmt = "crash-record-%08d"

	// Enough acknowledgements to be past the first segment rotation below, so
	// the kill lands in a directory that already has several segments rather
	// than only ever at the end of the first.
	wantAcks = 24
)

// TestCrashChild is the other half of TestCrashMidWriteKeepsEveryAcknowledged-
// Record, and does nothing unless the parent invokes it with the environment
// variable set. It never returns: the parent kills it.
func TestCrashChild(t *testing.T) {
	dir := os.Getenv(crashDirEnv)
	if dir == "" {
		t.Skip("child half of the crash harness; not meaningful on its own")
	}

	w, err := Open(dir, Options{
		SyncPolicy: SyncAlways,
		// A record here is 17 bytes of header plus a 20-byte payload, so 256
		// bytes is a rotation every 7 records — several before the kill. The
		// dangerous crash is the one that leaves damage in the *middle* of a
		// multi-segment log, and a single-segment test cannot produce it.
		MaxSegmentBytes: 256,
	})
	if err != nil {
		// os.Exit rather than t.Fatal throughout: this process is designed to be
		// killed, and the test framework's own reporting is not what the parent
		// is reading.
		fmt.Fprintln(os.Stdout, "CHILD-OPEN-FAILED", err)
		os.Exit(2)
	}

	for i := 0; ; i++ {
		seq, err := w.Append(TypePut, fmt.Appendf(nil, crashPayloadFmt, i))
		if err != nil {
			fmt.Fprintln(os.Stdout, "CHILD-APPEND-FAILED", err)
			os.Exit(2)
		}
		// Printed strictly after Append returns. Under SyncAlways that return is
		// the durability promise, so a line the parent has read is a record that
		// must survive whatever happens next.
		fmt.Fprintf(os.Stdout, "%s%d %d\n", ackPrefix, seq, i)
	}
}

func TestCrashMidWriteKeepsEveryAcknowledgedRecord(t *testing.T) {
	if testing.Short() {
		t.Skip("spawns a subprocess and fsyncs on every append")
	}

	dir := t.TempDir()

	// os.Args[0] under `go test` is this test binary, so the child is the same
	// code with the same build flags — including -race when the suite runs
	// under it.
	cmd := exec.Command(os.Args[0], "-test.run=^TestCrashChild$", "-test.timeout=120s")
	cmd.Env = append(os.Environ(), crashDirEnv+"="+dir)
	cmd.Stderr = os.Stderr

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting the child: %v", err)
	}
	// If an assertion below fails before the kill, the child would otherwise
	// outlive the test and keep writing.
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()

	type ack struct {
		seq uint64
		i   int
	}
	acks := make(chan ack, 4096)
	go func() {
		defer close(acks)
		sc := bufio.NewScanner(stdout)
		for sc.Scan() {
			line := sc.Text()
			if !strings.HasPrefix(line, ackPrefix) {
				continue
			}
			var a ack
			if _, err := fmt.Sscanf(line[len(ackPrefix):], "%d %d", &a.seq, &a.i); err != nil {
				continue
			}
			acks <- a
		}
	}()

	acked := make([]ack, 0, wantAcks)
	deadline := time.After(120 * time.Second)
	for len(acked) < wantAcks {
		select {
		case a, ok := <-acks:
			if !ok {
				t.Fatalf("the child stopped on its own after %d acknowledgements; "+
					"it should only ever stop by being killed", len(acked))
			}
			acked = append(acked, a)
		case <-deadline:
			t.Fatalf("timed out with %d of %d acknowledgements", len(acked), wantAcks)
		}
	}

	if err := cmd.Process.Kill(); err != nil {
		t.Fatalf("killing the child: %v", err)
	}
	_ = cmd.Wait()

	// Recovery reads a directory whose writer died mid-append.
	seen := make(map[uint64]string, len(acked))
	res, err := Replay(dir, Options{}, func(r Record) error {
		// The payload aliases a reused buffer, so it is copied rather than kept.
		seen[r.Seq] = string(r.Payload)
		return nil
	})
	if err != nil {
		t.Fatalf("replay after a real crash: %v", err)
	}

	for _, a := range acked {
		payload, ok := seen[a.seq]
		if !ok {
			t.Fatalf("seq %d was acknowledged under SyncAlways and did not survive "+
				"the kill; fsync-before-return is not holding", a.seq)
		}
		if want := fmt.Sprintf(crashPayloadFmt, a.i); payload != want {
			t.Fatalf("seq %d payload = %q, want %q", a.seq, payload, want)
		}
	}

	last := acked[len(acked)-1].seq
	if res.LastSeq < last {
		t.Fatalf("replay reported LastSeq %d, below the acknowledged %d", res.LastSeq, last)
	}
	if res.NextSeq() <= last {
		t.Fatalf("NextSeq %d would reuse an acknowledged sequence", res.NextSeq())
	}

	// Not asserted, only reported: whether the kill landed inside a record.
	// Requiring a tear would make this test depend on the scheduler, and a
	// crash between two appends is just as real a crash.
	//
	// Measured, it reliably reports *zero* tears, and that is worth knowing
	// rather than treating as a weak test. Under SyncAlways the record's bytes
	// reach the file before fsync is called, so the wide window — the
	// milliseconds spent in fsync — is one where the log already ends at a
	// record boundary. Tearing needs the kill to land inside the write itself,
	// which is a far narrower target.
	//
	// So the two halves of recovery testing divide cleanly: torn tails are
	// covered deterministically in replay_test.go by damaging a log directly,
	// and this test covers the thing that cannot be simulated — real timing,
	// with no cleanup on the way out.
	t.Logf("survived: %d records across %d segments, %d acknowledged, %d torn segments",
		res.Records, res.Segments, len(acked), len(res.Tears))
}

// The other half of the contract, and the one a crash makes tempting to forget:
// after recovering from a torn log the writer has to be reopened at NextSeq, or
// it reuses sequence numbers that already exist on disk.
func TestCrashedLogReopensAndKeepsAppending(t *testing.T) {
	if testing.Short() {
		t.Skip("spawns a subprocess and fsyncs on every append")
	}

	dir := t.TempDir()

	cmd := exec.Command(os.Args[0], "-test.run=^TestCrashChild$", "-test.timeout=120s")
	cmd.Env = append(os.Environ(), crashDirEnv+"="+dir)
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting the child: %v", err)
	}
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()

	sc := bufio.NewScanner(stdout)
	var lastAcked uint64
	for n := 0; n < wantAcks && sc.Scan(); {
		line := sc.Text()
		if !strings.HasPrefix(line, ackPrefix) {
			continue
		}
		var seq uint64
		var i int
		if _, err := fmt.Sscanf(line[len(ackPrefix):], "%d %d", &seq, &i); err != nil {
			continue
		}
		lastAcked, n = seq, n+1
	}
	if lastAcked == 0 {
		t.Fatal("the child acknowledged nothing")
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatalf("killing the child: %v", err)
	}
	_ = cmd.Wait()

	res, err := Replay(dir, Options{}, func(Record) error { return nil })
	if err != nil {
		t.Fatalf("replay: %v", err)
	}

	w, err := Open(dir, Options{SyncPolicy: SyncAlways, FirstSeq: res.NextSeq()})
	if err != nil {
		t.Fatalf("reopening after a crash: %v", err)
	}
	seq, err := w.Append(TypePut, []byte("after-the-crash"))
	if err != nil {
		t.Fatalf("append after a crash: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if seq <= res.LastSeq {
		t.Fatalf("the reopened writer assigned %d, at or below the recovered %d — "+
			"a sequence that already exists on disk", seq, res.LastSeq)
	}

	// And the whole log still reads back, old records and new, strictly
	// increasing throughout.
	var prev uint64
	var found bool
	final, err := Replay(dir, Options{}, func(r Record) error {
		if r.Seq <= prev {
			t.Errorf("sequence went backwards: %d after %d", r.Seq, prev)
		}
		prev = r.Seq
		if r.Seq == seq {
			found = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("final replay: %v", err)
	}
	if !found {
		t.Fatal("the record written after recovery is missing")
	}
	t.Logf("recovered %d records across %d segments, then appended seq %d",
		final.Records, final.Segments, seq)
}
