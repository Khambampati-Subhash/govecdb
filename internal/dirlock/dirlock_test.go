//go:build unix

package dirlock

import (
	"errors"
	"testing"
)

// Within one process the conflicts show up exactly as they would across two,
// because flock belongs to the open file description, not the process. That is
// what lets these run without a child; the database's own tests cover the
// cross-process case, including a holder killed with SIGKILL.
func TestExclusiveExcludesEverything(t *testing.T) {
	dir := t.TempDir()
	h, err := Lock(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, shared := range []bool{false, true} {
		if _, err := Lock(dir, shared); !errors.Is(err, ErrLocked) {
			t.Fatalf("Lock(shared=%v) under an exclusive lock = %v, want ErrLocked", shared, err)
		}
	}
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
	again, err := Lock(dir, false)
	if err != nil {
		t.Fatalf("Lock after Release = %v", err)
	}
	again.Release()
}

func TestSharedAdmitsSharedOnly(t *testing.T) {
	dir := t.TempDir()
	a, err := Lock(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Release()
	b, err := Lock(dir, true)
	if err != nil {
		t.Fatalf("second shared lock = %v", err)
	}
	defer b.Release()
	if _, err := Lock(dir, false); !errors.Is(err, ErrLocked) {
		t.Fatalf("exclusive under shared = %v, want ErrLocked", err)
	}
}

func TestLockOfAMissingDirectory(t *testing.T) {
	if _, err := Lock(t.TempDir()+"/nope", false); err == nil || errors.Is(err, ErrLocked) {
		t.Fatalf("Lock of a missing directory = %v, want an open error", err)
	}
}

func TestNilHandleReleases(t *testing.T) {
	var h *Handle
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
}
