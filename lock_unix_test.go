//go:build unix

package govecdb

import (
	"bufio"
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// The claims worth testing about the flock are about *other processes* — this
// one is already covered by openDirs — so these re-run the test binary as a
// child that opens the directory and then waits to be killed.

const lockChildEnv = "GOVECDB_LOCK_CHILD"

// TestLockChild is not a test. It is the child process: it opens the directory
// named in the environment, says so on stdout, and blocks until killed.
func TestLockChild(t *testing.T) {
	spec := os.Getenv(lockChildEnv)
	if spec == "" {
		t.Skip("helper process for the cross-process lock tests")
	}
	mode, dir, _ := strings.Cut(spec, ":")
	opts := []Option{WithDimension(testDim), WithSyncPolicy(SyncNever)}
	if mode == "ro" {
		opts = append(opts, WithReadOnly())
	}
	db, err := Open(dir, opts...)
	if err != nil {
		os.Stdout.WriteString("error " + err.Error() + "\n")
		os.Exit(1)
	}
	os.Stdout.WriteString("ready\n")
	time.Sleep(time.Hour)
	db.Close()
}

// startHolder starts a child holding dir and waits until it has it.
func startHolder(t *testing.T, mode, dir string) *exec.Cmd {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe, "-test.run=^TestLockChild$")
	cmd.Env = append(os.Environ(), lockChildEnv+"="+mode+":"+dir)
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cmd.Process.Kill(); _ = cmd.Wait() })

	line, err := bufio.NewReader(out).ReadString('\n')
	if err != nil || strings.TrimSpace(line) != "ready" {
		t.Fatalf("child did not open %s: %q, %v", dir, line, err)
	}
	return cmd
}

// TestAnotherProcessIsRefused, and — the whole argument for flock over a lock
// file — a process that dies without closing leaves nothing behind.
func TestAnotherProcessIsRefused(t *testing.T) {
	dir := t.TempDir()
	child := startHolder(t, "rw", dir)

	if _, err := Open(dir, WithDimension(testDim)); !errors.Is(err, ErrAlreadyOpen) {
		t.Fatalf("Open while another process writes = %v, want ErrAlreadyOpen", err)
	}
	if _, err := Open(dir, WithDimension(testDim), WithReadOnly()); !errors.Is(err, ErrAlreadyOpen) {
		t.Fatalf("read-only Open while another process writes = %v, want ErrAlreadyOpen", err)
	}

	// SIGKILL: no deferred Close, no cleanup of any kind.
	if err := child.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = child.Wait()

	db, err := Open(dir, WithDimension(testDim))
	if err != nil {
		t.Fatalf("Open after the holder was killed = %v; a crash must not leave a stale lock", err)
	}
	db.Close()
}

func TestReadersShareButExcludeAWriter(t *testing.T) {
	dir := t.TempDir()
	seed := openDBAt(t, dir)
	fill(t, seed, 5, 1)
	if err := seed.Close(); err != nil {
		t.Fatal(err)
	}

	startHolder(t, "ro", dir)

	ro, err := Open(dir, WithDimension(testDim), WithReadOnly())
	if err != nil {
		t.Fatalf("second reader = %v, want it to share the directory", err)
	}
	defer ro.Close()
	if ro.Len() != 5 {
		t.Fatalf("reader sees %d vectors", ro.Len())
	}
	if err := ro.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := Open(dir, WithDimension(testDim)); !errors.Is(err, ErrAlreadyOpen) {
		t.Fatalf("writer alongside a reader = %v, want ErrAlreadyOpen", err)
	}
}
