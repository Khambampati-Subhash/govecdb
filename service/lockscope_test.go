package service

import (
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb"
)

// These tests pin the rule on Manager.mu: nothing slow happens under it. Each
// makes one call slow on demand through the Options seams, then shows that a
// request for a *different* collection still completes while that call is
// stuck. Before the rule, every one of them hung until the gate opened.

// prompt is how long a request on an unrelated collection may take while
// another collection is stuck. Generous, so a loaded CI machine does not flake;
// the failure it guards against is "until the gate opens", which is forever.
const prompt = 2 * time.Second

// gate blocks the first call through it until opened, and reports when that
// call has arrived.
type gate struct {
	arrived chan struct{}
	open    chan struct{}
	once    sync.Once
	first   atomic.Bool
}

func newGate() *gate {
	return &gate{arrived: make(chan struct{}), open: make(chan struct{})}
}

// wait blocks iff this is the first call.
func (g *gate) wait() {
	if !g.first.CompareAndSwap(false, true) {
		return
	}
	close(g.arrived)
	<-g.open
}

func (g *gate) release() { g.once.Do(func() { close(g.open) }) }

func within(t *testing.T, what string, fn func() error) {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- fn() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	case <-time.After(prompt):
		t.Fatalf("%s did not finish within %v: blocked behind another collection", what, prompt)
	}
}

func search(m *Manager, name string) func() error {
	return func() error {
		return m.Use(name, func(db *govecdb.DB) error {
			_, err := db.Search(govecdb.SearchRequest{Query: []float32{1, 0, 0, 0}, K: 1})
			return err
		})
	}
}

// The H1 scenario: a /metrics scrape (List) reaches a collection whose Stats
// is stuck behind its own write lock — a Compact, a large AddBatch. Every
// other collection must keep serving.
func TestSlowStatsDoesNotBlockOtherCollections(t *testing.T) {
	g := newGate()
	defer g.release()
	m := newManager(t, Options{stats: func(db *govecdb.DB) govecdb.Stats {
		g.wait()
		return db.Stats()
	}})
	mustCreate(t, m, "a", testSpec())
	mustCreate(t, m, "b", testSpec())
	mustCreate(t, m, "cold", testSpec())
	mustAdd(t, m, "b", govecdb.Vector{ID: "x", Values: []float32{1, 0, 0, 0}})

	listed := make(chan []Info, 1)
	go func() {
		infos, _ := m.List()
		listed <- infos
	}()
	<-g.arrived // List is inside a's Stats

	within(t, "search on b", search(m, "b"))
	within(t, "Get(b)", func() error { _, err := m.Get("b"); return err })
	within(t, "Create(c)", func() error { return m.Create("c", testSpec()) })
	within(t, "Use of a missing collection", func() error {
		err := m.Use("nope", func(*govecdb.DB) error { return nil })
		if err == nil {
			t.Error("Use(nope) succeeded")
		}
		return nil
	})

	g.release()
	infos := <-listed
	if len(infos) != 3 { // a, b, cold — c was created after the directory was read
		t.Fatalf("List = %d entries, want 3: %+v", len(infos), infos)
	}
	for _, info := range infos {
		if !info.Loaded {
			t.Errorf("%s reported not loaded", info.Name)
		}
	}
}

// Eviction for MaxOpen closes the victim with the lock released. While it
// closes, the victim's own callers wait and then reopen it; everyone else is
// untouched.
func TestEvictionClosesOutsideTheLock(t *testing.T) {
	g := newGate()
	defer g.release()
	m := newManager(t, Options{MaxOpen: 2, closeDB: func(db *govecdb.DB) error {
		g.wait()
		return db.Close()
	}})
	mustCreate(t, m, "a", testSpec())
	mustAdd(t, m, "a", govecdb.Vector{ID: "x", Values: []float32{1, 0, 0, 0}})
	mustCreate(t, m, "b", testSpec())
	mustAdd(t, m, "b", govecdb.Vector{ID: "x", Values: []float32{1, 0, 0, 0}}) // a is now the LRU

	created := make(chan error, 1)
	go func() { created <- m.Create("c", testSpec()) }() // evicts a
	<-g.arrived

	within(t, "search on b", search(m, "b"))
	within(t, "List", func() error { _, err := m.List(); return err })

	// a is mid-close: a caller for it waits for the close, then reopens.
	reopened := make(chan error, 1)
	go func() { reopened <- search(m, "a")() }()
	select {
	case err := <-reopened:
		t.Fatalf("search on a returned (%v) while a was still closing", err)
	case <-time.After(50 * time.Millisecond):
	}

	g.release()
	if err := <-created; err != nil {
		t.Fatalf("Create(c): %v", err)
	}
	// Reopening a evicts another collection; with MaxOpen 2 that must still
	// leave a holding its vector.
	if err := <-reopened; err != nil {
		t.Fatalf("search on a after its eviction: %v", err)
	}
}

// Drop's close and RemoveAll run with the lock released, and the collection
// being dropped is already gone to anyone who asks.
func TestDropRemovesOutsideTheLock(t *testing.T) {
	g := newGate()
	defer g.release()
	m := newManager(t, Options{closeDB: func(db *govecdb.DB) error {
		g.wait()
		return db.Close()
	}})
	mustCreate(t, m, "a", testSpec())
	mustCreate(t, m, "b", testSpec())
	mustAdd(t, m, "b", govecdb.Vector{ID: "x", Values: []float32{1, 0, 0, 0}})

	dropped := make(chan error, 1)
	go func() { dropped <- m.Drop("a") }()
	<-g.arrived

	within(t, "search on b", search(m, "b"))
	within(t, "Use(a) during its drop", func() error {
		if err := search(m, "a")(); err == nil {
			t.Error("Use(a) succeeded while a was being dropped")
		}
		return nil
	})
	within(t, "Create(a) during its drop", func() error {
		if err := m.Create("a", testSpec()); err == nil {
			t.Error("Create(a) succeeded while a was being dropped")
		}
		return nil
	})

	g.release()
	if err := <-dropped; err != nil {
		t.Fatalf("Drop: %v", err)
	}
	if _, err := os.Stat(filepath.Join(m.Root(), "a")); !os.IsNotExist(err) {
		t.Fatalf("a's directory survived the drop: %v", err)
	}
	// And the spec cache forgot it, or a recreate would be refused or, worse,
	// a lookup would open the deleted directory.
	if _, err := m.Get("a"); err == nil {
		t.Fatal("Get(a) found the dropped collection")
	}
	mustCreate(t, m, "a", testSpec())
}

// List reads each spec from disk once. The spec is immutable after Create, and
// re-reading it is what made a scrape over thousands of collections hold the
// lock for hundreds of milliseconds.
func TestListCachesSpecs(t *testing.T) {
	root := t.TempDir()
	first := newManagerAt(t, root, Options{})
	mustCreate(t, first, "a", testSpec())
	first.Close()

	m := newManagerAt(t, root, Options{})
	if _, err := m.List(); err != nil {
		t.Fatal(err)
	}
	// Damage the file. A cached spec does not notice; one re-read per call
	// would report ErrCorruptSpec.
	if err := os.WriteFile(filepath.Join(root, "a", specFileName), []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	infos, err := m.List()
	if err != nil || len(infos) != 1 || infos[0].Spec.Dimension != 4 {
		t.Fatalf("List = %+v, %v; want the cached spec", infos, err)
	}
}

// M2: Close flushes every idle collection before it waits for a busy one.
// Under SyncInterval or SyncNever an idle collection's acknowledged writes sit
// in a user-space buffer until its Close, and a shutdown that queued them
// behind one long Compact lost them to the SIGKILL at the end of the grace
// period.
func TestCloseClosesIdleCollectionsBeforeWaitingForBusyOnes(t *testing.T) {
	var closed atomic.Int32
	root := t.TempDir()
	m, err := NewManager(root, Options{closeDB: func(db *govecdb.DB) error {
		closed.Add(1)
		return db.Close()
	}})
	if err != nil {
		t.Fatal(err)
	}
	names := []string{"busy", "i1", "i2", "i3", "i4", "i5", "i6"}
	for _, n := range names {
		mustCreate(t, m, n, testSpec())
		mustAdd(t, m, n, govecdb.Vector{ID: "x", Values: []float32{1, 0, 0, 0}})
	}

	inUse, finish := make(chan struct{}), make(chan struct{})
	go m.Use("busy", func(*govecdb.DB) error {
		close(inUse)
		<-finish
		return nil
	})
	<-inUse

	closeDone := make(chan error, 1)
	go func() { closeDone <- m.Close() }()

	deadline := time.Now().Add(prompt)
	for closed.Load() < int32(len(names)-1) {
		if time.Now().After(deadline) {
			t.Fatalf("%d of %d idle collections closed while one was busy", closed.Load(), len(names)-1)
		}
		time.Sleep(time.Millisecond)
	}
	select {
	case err := <-closeDone:
		t.Fatalf("Close returned (%v) with a borrow in flight", err)
	default:
	}

	close(finish)
	if err := <-closeDone; err != nil {
		t.Fatalf("Close: %v", err)
	}
	if n := closed.Load(); n != int32(len(names)) {
		t.Fatalf("closed %d databases, want %d", n, len(names))
	}
}
