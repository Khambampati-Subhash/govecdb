package service

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/khambampati-subhash/govecdb"
)

// A collection the filesystem will not let open is ErrUnavailable — a 503 a
// client can retry — rather than an anonymous error that became a 500. The
// shape here is the one that happens in production: a reopen on a disk that
// refuses the new log segment every open creates.
func TestAnOpenRefusedByTheFilesystemIsUnavailable(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root ignores directory permissions")
	}
	root := t.TempDir()
	first := newManagerAt(t, root, Options{})
	mustCreate(t, first, "docs", testSpec())
	first.Close()

	walDir := filepath.Join(root, "docs", dataSubdir, "wal")
	if err := os.Chmod(walDir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(walDir, 0o700) })

	m := newManagerAt(t, root, Options{})
	err := m.Use("docs", func(*govecdb.DB) error { return nil })
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("Use = %v, want ErrUnavailable", err)
	}
}

// L2: a collection still inside its first open holds no index yet.
func TestLoadedDoesNotCountAnOpenInProgress(t *testing.T) {
	m := newManager(t, Options{})
	mustCreate(t, m, "docs", testSpec())

	m.mu.Lock()
	m.cols["opening"] = &collection{name: "opening", loading: true}
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		delete(m.cols, "opening")
		m.mu.Unlock()
	}()

	if n := m.Loaded(); n != 1 {
		t.Fatalf("Loaded = %d with one open collection and one opening, want 1", n)
	}
}
