package service

import (
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb"
)

// Tests run with SyncNever throughout. SyncAlways costs ~4 ms per append, which
// this package is not testing: what is under test here is lifecycle, and every
// durability guarantee has its own tests a layer down.
func testSpec() Spec {
	return Spec{Dimension: 4, SyncPolicy: govecdb.SyncNever}
}

func newManagerAt(t *testing.T, root string, opts Options) *Manager {
	t.Helper()
	m, err := NewManager(root, opts)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func newManager(t *testing.T, opts Options) *Manager {
	t.Helper()
	return newManagerAt(t, t.TempDir(), opts)
}

func mustCreate(t *testing.T, m *Manager, name string, spec Spec) {
	t.Helper()
	if err := m.Create(name, spec); err != nil {
		t.Fatalf("Create(%q): %v", name, err)
	}
}

func mustAdd(t *testing.T, m *Manager, name string, v govecdb.Vector) {
	t.Helper()
	err := m.Use(name, func(db *govecdb.DB) error { return db.Add(v) })
	if err != nil {
		t.Fatalf("Add to %q: %v", name, err)
	}
}

func TestCreateThenUse(t *testing.T) {
	m := newManager(t, Options{})
	mustCreate(t, m, "docs", testSpec())
	mustAdd(t, m, "docs", govecdb.Vector{
		ID:       "a",
		Values:   []float32{1, 0, 0, 0},
		Metadata: govecdb.Metadata{"source": "handbook"},
	})

	var got []govecdb.Match
	err := m.Use("docs", func(db *govecdb.DB) error {
		var err error
		got, err = db.Search(govecdb.SearchRequest{Query: []float32{1, 0, 0, 0}, K: 1})
		return err
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("Search = %+v, want one match on \"a\"", got)
	}
	if got[0].Metadata["source"] != "handbook" {
		t.Errorf("metadata = %v, want source=handbook", got[0].Metadata)
	}
}

func TestCreateRefusesAnExistingCollection(t *testing.T) {
	m := newManager(t, Options{})
	mustCreate(t, m, "docs", testSpec())

	if err := m.Create("docs", testSpec()); !errors.Is(err, ErrExists) {
		t.Fatalf("second Create = %v, want ErrExists", err)
	}
}

func TestCreateRefusesAnInvalidName(t *testing.T) {
	m := newManager(t, Options{})
	if err := m.Create("../escape", testSpec()); !errors.Is(err, ErrInvalidName) {
		t.Fatalf("Create = %v, want ErrInvalidName", err)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(m.Root()), "escape")); !os.IsNotExist(err) {
		t.Fatal("a directory was created outside the root")
	}
}

func TestCreateWithoutADimensionTouchesNothing(t *testing.T) {
	m := newManager(t, Options{})

	err := m.Create("docs", Spec{})
	if !errors.Is(err, ErrInvalidSpec) {
		t.Fatalf("Create = %v, want ErrInvalidSpec", err)
	}
	entries, err := os.ReadDir(m.Root())
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("root holds %d entries, want none", len(entries))
	}
}

// A spec the database refuses is only discovered by opening it, which happens
// after the directory exists. What must not survive is the directory: a
// collection that lists but cannot load turns a bad request into a permanent
// operational puzzle.
func TestCreateCleansUpAfterARefusedSpec(t *testing.T) {
	m := newManager(t, Options{})

	spec := testSpec()
	spec.M = 1 // undefined for HNSW, and refused by the root package.
	if err := m.Create("docs", spec); !errors.Is(err, ErrInvalidSpec) {
		t.Fatalf("Create = %v, want ErrInvalidSpec", err)
	}
	if _, err := os.Stat(filepath.Join(m.Root(), "docs")); !os.IsNotExist(err) {
		t.Fatalf("the collection directory survived a failed Create: %v", err)
	}
	if n := m.Loaded(); n != 0 {
		t.Fatalf("Loaded = %d, want 0", n)
	}
}

// The point of the spec file: a second process knows nothing about how the
// collection was built, and the structural options are the ones a database
// refuses to be reopened under if they are wrong.
func TestReopenUsesTheStoredSpec(t *testing.T) {
	root := t.TempDir()

	spec := testSpec()
	spec.Metric = govecdb.Euclidean
	spec.M = 8
	spec.EfConstruction = 64

	first := newManagerAt(t, root, Options{})
	mustCreate(t, first, "docs", spec)
	mustAdd(t, first, "docs", govecdb.Vector{ID: "a", Values: []float32{1, 2, 3, 4}})
	if err := first.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	second := newManagerAt(t, root, Options{})
	info, err := second.Get("docs")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if info.Spec.Metric != govecdb.Euclidean || info.Spec.M != 8 || info.Spec.EfConstruction != 64 {
		t.Fatalf("spec = %+v, want euclidean/M=8/ef=64", info.Spec)
	}

	var got []govecdb.Match
	err = second.Use("docs", func(db *govecdb.DB) error {
		var err error
		got, err = db.Search(govecdb.SearchRequest{Query: []float32{1, 2, 3, 4}, K: 1})
		return err
	})
	if err != nil {
		t.Fatalf("Search after reopen: %v", err)
	}
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("Search = %+v, want the vector written before the restart", got)
	}
}

func TestCreateRecordsEffectiveDefaults(t *testing.T) {
	m := newManager(t, Options{})
	mustCreate(t, m, "docs", testSpec())

	spec, err := readSpec(filepath.Join(m.Root(), "docs"))
	if err != nil {
		t.Fatalf("readSpec: %v", err)
	}
	// Nothing may be left as "0, meaning whatever this build's default is": a
	// later change to that default would silently reshape an existing index.
	if spec.M == 0 || spec.EfConstruction == 0 || spec.Seed == 0 ||
		spec.SnapshotsKept == 0 || spec.TargetRecall == 0 || spec.SyncInterval == 0 {
		t.Fatalf("spec = %+v, want every default resolved to its effective value", spec)
	}
	if spec.M != 16 {
		t.Errorf("M = %d, want the library default of 16", spec.M)
	}
}

func TestUseReportsAMissingCollection(t *testing.T) {
	m := newManager(t, Options{})
	err := m.Use("nope", func(*govecdb.DB) error { return nil })
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("Use = %v, want ErrNotFound", err)
	}
}

func TestUseReturnsTheCallbackError(t *testing.T) {
	m := newManager(t, Options{})
	mustCreate(t, m, "docs", testSpec())

	sentinel := errors.New("from the callback")
	if err := m.Use("docs", func(*govecdb.DB) error { return sentinel }); !errors.Is(err, sentinel) {
		t.Fatalf("Use = %v, want the callback's own error", err)
	}
}

func TestListAndGetDoNotLoad(t *testing.T) {
	root := t.TempDir()

	first := newManagerAt(t, root, Options{})
	mustCreate(t, first, "alpha", testSpec())
	mustCreate(t, first, "beta", testSpec())
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}

	m := newManagerAt(t, root, Options{})
	infos, err := m.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(infos) != 2 || infos[0].Name != "alpha" || infos[1].Name != "beta" {
		t.Fatalf("List = %+v, want alpha and beta in order", infos)
	}
	for _, info := range infos {
		if info.Loaded {
			t.Errorf("%q reports Loaded, but listing must not load anything", info.Name)
		}
	}
	if n := m.Loaded(); n != 0 {
		t.Fatalf("Loaded = %d, want 0", n)
	}

	mustAdd(t, m, "alpha", govecdb.Vector{ID: "a", Values: []float32{1, 0, 0, 0}})

	info, err := m.Get("alpha")
	if err != nil {
		t.Fatal(err)
	}
	if !info.Loaded || info.Stats.Live != 1 {
		t.Fatalf("Get = %+v, want loaded with one live vector", info)
	}
}

func TestListIgnoresForeignDirectories(t *testing.T) {
	m := newManager(t, Options{})
	mustCreate(t, m, "docs", testSpec())

	// A directory with a legal name but no spec file, and one whose name this
	// package would never have written. Neither is a collection.
	if err := os.Mkdir(filepath.Join(m.Root(), "leftover"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(m.Root(), "not.a.collection"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(m.Root(), "stray.tar"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}

	infos, err := m.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(infos) != 1 || infos[0].Name != "docs" {
		t.Fatalf("List = %+v, want only docs", infos)
	}
}

func TestDropRemovesEverything(t *testing.T) {
	m := newManager(t, Options{})
	mustCreate(t, m, "docs", testSpec())
	mustAdd(t, m, "docs", govecdb.Vector{ID: "a", Values: []float32{1, 0, 0, 0}})

	if err := m.Drop("docs"); err != nil {
		t.Fatalf("Drop: %v", err)
	}
	if _, err := os.Stat(filepath.Join(m.Root(), "docs")); !os.IsNotExist(err) {
		t.Fatalf("the directory survived Drop: %v", err)
	}
	if err := m.Use("docs", func(*govecdb.DB) error { return nil }); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Use after Drop = %v, want ErrNotFound", err)
	}
	if n := m.Loaded(); n != 0 {
		t.Fatalf("Loaded = %d, want 0", n)
	}

	// And the name is free again, which it would not be if anything were left.
	mustCreate(t, m, "docs", testSpec())
}

func TestDropReportsAMissingCollection(t *testing.T) {
	m := newManager(t, Options{})
	if err := m.Drop("nope"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Drop = %v, want ErrNotFound", err)
	}
}

// Drop waits rather than failing, because a search takes microseconds and a drop
// is deliberate: losing the race is not something the operator could have done
// anything about.
func TestDropWaitsForABorrowInFlight(t *testing.T) {
	m := newManager(t, Options{})
	mustCreate(t, m, "docs", testSpec())

	inUse := make(chan struct{})
	finish := make(chan struct{})
	go func() {
		m.Use("docs", func(*govecdb.DB) error {
			close(inUse)
			<-finish
			return nil
		})
	}()
	<-inUse

	dropped := make(chan error, 1)
	go func() { dropped <- m.Drop("docs") }()

	select {
	case err := <-dropped:
		t.Fatalf("Drop returned %v while a borrow was in flight", err)
	case <-time.After(50 * time.Millisecond):
	}

	close(finish)
	if err := <-dropped; err != nil {
		t.Fatalf("Drop: %v", err)
	}
}

// The race this exists for: govecdb refuses a second writer on one directory, so
// two goroutines opening the same collection would surface ErrAlreadyOpen on an
// ordinary request. One of them opens; the rest wait.
func TestConcurrentUseOpensOnce(t *testing.T) {
	root := t.TempDir()

	first := newManagerAt(t, root, Options{})
	mustCreate(t, first, "docs", testSpec())
	mustAdd(t, first, "docs", govecdb.Vector{ID: "a", Values: []float32{1, 0, 0, 0}})
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}

	m := newManagerAt(t, root, Options{})

	const n = 32
	start := make(chan struct{})
	errs := make(chan error, n)
	var wg sync.WaitGroup
	for range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			errs <- m.Use("docs", func(db *govecdb.DB) error {
				_, err := db.Search(govecdb.SearchRequest{Query: []float32{1, 0, 0, 0}, K: 1})
				return err
			})
		}()
	}
	close(start)
	wg.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent Use: %v", err)
		}
	}
	if n := m.Loaded(); n != 1 {
		t.Fatalf("Loaded = %d, want exactly 1", n)
	}
}

func TestMaxOpenEvictsTheLeastRecentlyUsed(t *testing.T) {
	m := newManager(t, Options{MaxOpen: 1})
	mustCreate(t, m, "alpha", testSpec())
	mustCreate(t, m, "beta", testSpec()) // evicts alpha

	if n := m.Loaded(); n != 1 {
		t.Fatalf("Loaded = %d, want 1", n)
	}
	info, err := m.Get("beta")
	if err != nil {
		t.Fatal(err)
	}
	if !info.Loaded {
		t.Fatal("beta should be the loaded one")
	}

	// Reaching for alpha again evicts beta, and alpha's data survived being
	// closed and reopened.
	mustAdd(t, m, "alpha", govecdb.Vector{ID: "a", Values: []float32{1, 0, 0, 0}})
	if n := m.Loaded(); n != 1 {
		t.Fatalf("Loaded = %d, want 1", n)
	}
	if info, err := m.Get("alpha"); err != nil || !info.Loaded || info.Stats.Live != 1 {
		t.Fatalf("Get(alpha) = %+v, %v; want loaded with one live vector", info, err)
	}
}

func TestMaxOpenRefusesWhenEverythingIsBusy(t *testing.T) {
	m := newManager(t, Options{MaxOpen: 1})
	mustCreate(t, m, "alpha", testSpec())
	mustCreate(t, m, "beta", testSpec()) // alpha is evicted; beta is loaded

	inUse := make(chan struct{})
	finish := make(chan struct{})
	go func() {
		m.Use("beta", func(*govecdb.DB) error {
			close(inUse)
			<-finish
			return nil
		})
	}()
	<-inUse
	defer close(finish)

	err := m.Use("alpha", func(*govecdb.DB) error { return nil })
	if !errors.Is(err, ErrTooManyOpen) {
		t.Fatalf("Use = %v, want ErrTooManyOpen", err)
	}
}

func TestIdleEvictionClosesAQuietCollection(t *testing.T) {
	clock := time.Now()
	var mu sync.Mutex
	now := func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		return clock
	}
	advance := func(d time.Duration) {
		mu.Lock()
		defer mu.Unlock()
		clock = clock.Add(d)
	}

	// The sweeper is left off (IdleTimeout drives it, and evictIdle is called
	// directly below) so the test controls exactly when eviction is considered.
	m := newManager(t, Options{now: now})
	m.opts.IdleTimeout = time.Minute

	mustCreate(t, m, "docs", testSpec())
	if n := m.Loaded(); n != 1 {
		t.Fatalf("Loaded = %d, want 1", n)
	}

	advance(30 * time.Second)
	m.evictIdle()
	if n := m.Loaded(); n != 1 {
		t.Fatalf("Loaded = %d after 30s of a 60s timeout, want 1", n)
	}

	advance(31 * time.Second)
	m.evictIdle()
	if n := m.Loaded(); n != 0 {
		t.Fatalf("Loaded = %d after the timeout, want 0", n)
	}

	// Evicted, not lost.
	if _, err := m.Get("docs"); err != nil {
		t.Fatalf("Get after eviction: %v", err)
	}
}

func TestCloseIsIdempotentAndFinal(t *testing.T) {
	m := newManager(t, Options{})
	mustCreate(t, m, "docs", testSpec())

	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}

	if err := m.Use("docs", func(*govecdb.DB) error { return nil }); !errors.Is(err, ErrClosed) {
		t.Errorf("Use = %v, want ErrClosed", err)
	}
	if err := m.Create("other", testSpec()); !errors.Is(err, ErrClosed) {
		t.Errorf("Create = %v, want ErrClosed", err)
	}
	if _, err := m.List(); !errors.Is(err, ErrClosed) {
		t.Errorf("List = %v, want ErrClosed", err)
	}
	if _, err := m.Get("docs"); !errors.Is(err, ErrClosed) {
		t.Errorf("Get = %v, want ErrClosed", err)
	}
	if err := m.Drop("docs"); !errors.Is(err, ErrClosed) {
		t.Errorf("Drop = %v, want ErrClosed", err)
	}
}

// The whole lifecycle under -race: creating, using, dropping and evicting the
// same names from many goroutines at once. It asserts nothing beyond "no
// unexpected error and no data race", which is what the detector is here to add.
func TestConcurrentLifecycle(t *testing.T) {
	m := newManager(t, Options{MaxOpen: 2})

	names := []string{"alpha", "beta", "gamma"}
	for _, n := range names {
		mustCreate(t, m, n, testSpec())
	}

	var wg sync.WaitGroup
	for i, name := range names {
		for range 4 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := range 10 {
					err := m.Use(name, func(db *govecdb.DB) error {
						if j%2 == 0 {
							return db.Add(govecdb.Vector{
								ID:     name,
								Values: []float32{float32(i), 0, 0, 1},
							})
						}
						_, err := db.Search(govecdb.SearchRequest{Query: []float32{1, 0, 0, 0}, K: 1})
						return err
					})
					// Everything here is legitimate: a collection may be evicted
					// by MaxOpen and reopened, or every slot may be busy.
					if err != nil && !errors.Is(err, ErrTooManyOpen) && !errors.Is(err, ErrNotFound) {
						t.Errorf("Use(%q): %v", name, err)
						return
					}
				}
			}()
		}
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for range 3 {
			if _, err := m.List(); err != nil {
				t.Errorf("List: %v", err)
				return
			}
		}
	}()

	wg.Wait()

	if err := m.Drop("gamma"); err != nil {
		t.Fatalf("Drop: %v", err)
	}
}
