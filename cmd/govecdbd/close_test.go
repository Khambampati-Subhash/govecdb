package main

import (
	"errors"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb"
	"github.com/khambampati-subhash/govecdb/service"
)

// After a Shutdown that timed out, a handler can still hold a collection. The
// daemon must still exit — with every idle collection flushed — rather than
// wait on that one for as long as it takes.
func TestCloseWithinIsBoundedByABusyCollection(t *testing.T) {
	root := t.TempDir()
	mgr, err := service.NewManager(root, service.Options{})
	if err != nil {
		t.Fatal(err)
	}
	spec := service.Spec{Dimension: 4, SyncPolicy: govecdb.SyncNever}
	for _, name := range []string{"busy", "idle"} {
		if err := mgr.Create(name, spec); err != nil {
			t.Fatal(err)
		}
	}
	if err := mgr.Use("idle", func(db *govecdb.DB) error {
		return db.Add(govecdb.Vector{ID: "a", Values: []float32{1, 0, 0, 0}})
	}); err != nil {
		t.Fatal(err)
	}

	inUse, finish := make(chan struct{}), make(chan struct{})
	go mgr.Use("busy", func(*govecdb.DB) error {
		close(inUse)
		<-finish
		return nil
	})
	<-inUse
	defer close(finish)

	start := time.Now()
	err = closeWithin(mgr, 100*time.Millisecond)
	if !errors.Is(err, errCloseTimeout) {
		t.Fatalf("closeWithin = %v, want errCloseTimeout", err)
	}
	if took := time.Since(start); took > 2*time.Second {
		t.Fatalf("closeWithin took %v with a 100ms bound", took)
	}

	// The idle collection was closed — so flushed — before the bound bit:
	// its directory can be opened by somebody else, which a live writer's
	// lock would refuse.
	db, err := govecdb.Open(root+"/idle/data", govecdb.WithDimension(4), govecdb.WithReadOnly())
	if err != nil {
		t.Fatalf("idle collection still open after closeWithin: %v", err)
	}
	defer db.Close()
	if _, err := db.Get("a"); err != nil {
		t.Fatalf("idle collection lost its write: %v", err)
	}
}

func TestCloseWithinReturnsTheCloseError(t *testing.T) {
	want := errors.New("boom")
	if err := closeWithin(closerFunc(func() error { return want }), time.Second); err != want {
		t.Fatalf("closeWithin = %v, want %v", err, want)
	}
}

type closerFunc func() error

func (f closerFunc) Close() error { return f() }
