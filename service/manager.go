package service

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb"
	"github.com/khambampati-subhash/govecdb/internal/dirlock"
)

// Options configures a Manager. The zero value is usable: no cap on how many
// collections are loaded, and none of them ever evicted.
type Options struct {
	// MaxOpen caps how many collections are loaded at once. Zero means no cap.
	//
	// It bounds memory, which is the resource a vector database runs out of
	// first: a loaded collection holds its whole index. When the cap is reached
	// the least recently used *idle* collection is closed to make room; if every
	// loaded collection is in use, the open fails with ErrTooManyOpen rather than
	// queueing — see that error for why.
	MaxOpen int

	// IdleTimeout closes a collection that nothing has used for this long. Zero
	// leaves collections loaded until the Manager is closed.
	//
	// This is the knob that makes "many collections, most of them quiet" cost
	// what it should. Reopening one costs a snapshot load and a log replay, so
	// the timeout wants to be long relative to how bursty the traffic is —
	// minutes, not seconds.
	IdleTimeout time.Duration

	// SweepInterval is how often idle collections are looked for. Zero means a
	// quarter of IdleTimeout, which bounds the overshoot at 25% without making a
	// mostly-idle process wake up constantly.
	SweepInterval time.Duration

	// DefaultSnapshotInterval is the SnapshotInterval a collection gets when
	// Create is handed a spec that leaves it zero. Zero keeps the old
	// behaviour: no automatic snapshots unless a spec asks.
	//
	// It exists because the library's default — off — is right for a
	// short-lived program and wrong for a server. Nobody calls Snapshot before
	// a daemon restarts, and a collection with no snapshot rebuilds its whole
	// index from the log on the next open: about 23 minutes at 250,000
	// vectors of dimension 512. The resolved value is written into the spec,
	// so changing this later never alters an existing collection, and a spec
	// can still opt out with SnapshotOff.
	DefaultSnapshotInterval time.Duration

	// Observer receives every collection's events, tagged with the collection's
	// name. Nil discards them. It is called on the collection's own goroutines,
	// sometimes under its write lock, so it must be safe for concurrent use and
	// must not call back into the Manager or the collection — see
	// govecdb.WithObserver.
	Observer func(collection string, e govecdb.Event)

	// now is the clock, unexported because only this package's tests replace it.
	// Eviction is defined in elapsed time, and a test that demonstrates it by
	// sleeping demonstrates it slowly and then flakily.
	now func() time.Time

	// stats and closeDB stand in for (*govecdb.DB).Stats and Close, for the
	// same reason: the tests that prove no slow call happens under the
	// manager's lock need a call that is slow on demand. A real slow Stats is a
	// Compact on another goroutine, which is a race to set up rather than a
	// test.
	stats   func(*govecdb.DB) govecdb.Stats
	closeDB func(*govecdb.DB) error
}

// Manager owns a directory of collections.
//
// It is safe for concurrent use. Reads and writes within a collection are the
// database's own business and run in parallel exactly as they do without a
// Manager; what this serializes is the lifecycle — which collections are loaded,
// and when one may be closed.
type Manager struct {
	root string
	opts Options

	// now is time.Now except in tests, where idle eviction has to be provoked
	// without waiting for it. stats and closeDB are the database's methods
	// except in tests; see Options.
	now     func() time.Time
	stats   func(*govecdb.DB) govecdb.Stats
	closeDB func(*govecdb.DB) error

	// mu guards everything below, and cond signals the three things a caller
	// may have to wait for: a collection finishing its open, a collection
	// finishing its close, and a collection's last borrower going away.
	//
	// # Nothing slow happens under mu
	//
	// It is one lock for every collection on the server, so whatever it is held
	// across, every request waits for. Opening, closing, deleting, reading a
	// spec and asking a database for its Stats all happen with it released;
	// the entry in cols carries a state (loading, closing, dropping) that tells
	// a concurrent caller what is under way and whether to wait for it. Stats
	// is the subtle one: it takes the database's read lock, which a Compact or
	// a large AddBatch holds for seconds, so a /metrics scrape that called it
	// under mu froze the whole server behind one busy collection.
	mu     sync.Mutex
	cond   *sync.Cond
	cols   map[string]*collection
	closed bool

	// specs caches each collection's spec, which is immutable once Create has
	// written it. Filled by Create and by the first read from disk, emptied by
	// Drop. It is what keeps List — run by every /metrics scrape — from
	// reading one file per collection, and cold lookups from reading one at
	// all once a collection has been seen.
	//
	// dropGen counts completed Drops. A spec read from disk with the lock
	// released is cached only if no Drop finished meanwhile; otherwise it may
	// describe a directory that is already gone, and a cached spec is what
	// acquire opens a database from.
	specs   map[string]Spec
	dropGen uint64

	stop chan struct{}
	done chan struct{}

	// lock keeps a second Manager off this root for as long as this one is
	// open. Each collection's database also locks its own directory, but that
	// does not cover the root's own operations: a second manager could Drop a
	// collection this one has loaded, and Drop is a RemoveAll.
	lock *dirlock.Handle
}

// collection is one loaded database plus the bookkeeping that decides when it
// may be closed. Every field is guarded by the Manager's mutex.
type collection struct {
	name string
	dir  string
	spec Spec
	db   *govecdb.DB

	// loading is set while another goroutine is inside govecdb.Open for this
	// collection, with the Manager's lock released. It is what stops a second
	// caller from opening the same directory — which the database would refuse —
	// and what they wait on instead.
	loading bool

	// dropping is set once Drop has claimed this collection. New borrowers are
	// refused from that moment, so the reference count can actually reach zero.
	dropping bool

	// closing is set while the database is being closed — by eviction, the
	// idle sweep, Drop or Close — with the Manager's lock released. The entry
	// stays in the map until the close returns, so a caller asking for the
	// collection meanwhile waits and then reopens it, rather than opening a
	// directory whose database is still shutting down (which it would refuse).
	closing bool

	// refs is how many callers are inside Use. A collection with refs > 0 is
	// never closed: eviction and Drop both wait, because closing a database out
	// from under a search in flight turns a capacity policy into an error the
	// client did nothing to deserve.
	refs     int
	lastUsed time.Time
}

// NewManager opens a collection directory. The directory is created if it does
// not exist; existing collections inside it are found but not loaded, since
// loading is what NewManager is careful not to do — a process with a thousand
// collections should start in milliseconds and pay for the ones it is asked for.
func NewManager(root string, opts Options) (*Manager, error) {
	if root == "" {
		return nil, fmt.Errorf("%w: root directory is empty", ErrInvalidSpec)
	}
	root = filepath.Clean(root)
	if err := os.MkdirAll(root, dirPerm); err != nil {
		return nil, fmt.Errorf("service: create %q: %w", root, err)
	}
	lock, err := dirlock.Lock(root, false)
	if errors.Is(err, dirlock.ErrLocked) {
		return nil, fmt.Errorf("%w: %s", ErrRootInUse, root)
	}
	if err != nil {
		return nil, fmt.Errorf("service: %w", err)
	}

	m := &Manager{
		root:    root,
		opts:    opts,
		now:     opts.now,
		stats:   opts.stats,
		closeDB: opts.closeDB,
		cols:    make(map[string]*collection),
		specs:   make(map[string]Spec),
		lock:    lock,
	}
	if m.now == nil {
		m.now = time.Now
	}
	if m.stats == nil {
		m.stats = (*govecdb.DB).Stats
	}
	if m.closeDB == nil {
		m.closeDB = (*govecdb.DB).Close
	}
	m.cond = sync.NewCond(&m.mu)

	if opts.IdleTimeout > 0 {
		m.stop = make(chan struct{})
		m.done = make(chan struct{})
		go m.sweep()
	}
	return m, nil
}

// Root is the directory the manager owns.
func (m *Manager) Root() string { return m.root }

// Create makes a new collection and leaves it loaded.
//
// The spec is completed with Defaults and written before the database is opened,
// so what is on disk is what the collection was actually built with. If the open
// then fails — an out-of-range M, an unwritable disk — the directory is removed
// again: a half-created collection that lists but cannot load is worse than none,
// because it turns a bad request into a permanent operational puzzle.
//
// The disk work — the directory, two fsyncs, the open — runs with the manager's
// lock released, behind a loading placeholder, exactly as a cold open does.
func (m *Manager) Create(name string, spec Spec) error {
	if err := ValidateName(name); err != nil {
		return err
	}
	// The one check made before touching the disk. Every other bound belongs to
	// the root package's options and is enforced by Open below, which is the
	// single place that knows them; this one is here because "you forgot the
	// dimension" should not be reported as a failed directory creation.
	if spec.Dimension <= 0 {
		return fmt.Errorf("%w: dimension is required", ErrInvalidSpec)
	}
	switch {
	case spec.SnapshotInterval < 0: // SnapshotOff
		spec.SnapshotInterval = 0
	case spec.SnapshotInterval == 0:
		spec.SnapshotInterval = m.opts.DefaultSnapshotInterval
	}
	spec = spec.Defaults()

	m.mu.Lock()
	for {
		if m.closed {
			m.mu.Unlock()
			return ErrClosed
		}
		if _, ok := m.cols[name]; ok {
			m.mu.Unlock()
			return fmt.Errorf("%w: %q", ErrExists, name)
		}
		evicted, err := m.makeRoomLocked()
		if err != nil {
			m.mu.Unlock()
			return err
		}
		if !evicted {
			break
		}
		// The lock was released for the eviction; everything above may have
		// changed, including someone else creating this very name.
	}
	c := &collection{name: name, dir: m.dir(name), spec: spec, loading: true}
	m.cols[name] = c
	m.mu.Unlock()

	db, created, err := m.createOnDisk(c.dir, name, spec)

	m.mu.Lock()
	defer m.mu.Unlock()
	if err == nil && m.closed {
		// Close ran while this was on disk and could not see the database. The
		// client is told the server is shutting down, so the collection must
		// not exist afterwards either.
		err = ErrClosed
		m.mu.Unlock()
		m.closeDB(db)
		m.mu.Lock()
	}
	if err != nil {
		if created {
			m.mu.Unlock()
			os.RemoveAll(c.dir)
			m.mu.Lock()
		}
		delete(m.cols, name)
		m.cond.Broadcast()
		return err
	}
	c.loading = false
	c.db = db
	c.lastUsed = m.now()
	m.specs[name] = spec
	m.cond.Broadcast()
	return nil
}

// createOnDisk is Create's disk work. created reports whether the directory was
// made by this call, which is what decides whether a failure removes it: a
// directory that was already there belongs to somebody else.
func (m *Manager) createOnDisk(dir, name string, spec Spec) (db *govecdb.DB, created bool, err error) {
	if _, err := os.Stat(dir); err == nil {
		return nil, false, fmt.Errorf("%w: %q", ErrExists, name)
	} else if !os.IsNotExist(err) {
		return nil, false, fmt.Errorf("service: stat %q: %w", dir, err)
	}
	if err := os.Mkdir(dir, dirPerm); err != nil {
		return nil, false, fmt.Errorf("service: create %q: %w", dir, err)
	}
	// The directory entry itself has to be durable before the spec inside it is
	// worth anything.
	if err := syncDir(m.root); err != nil {
		return nil, true, err
	}
	if err := writeSpec(dir, spec); err != nil {
		return nil, true, err
	}
	db, err = govecdb.Open(filepath.Join(dir, dataSubdir), m.openOptions(name, spec)...)
	if err != nil {
		return nil, true, wrapOpen(name, err)
	}
	return db, true, nil
}

// Use borrows a collection's database for the duration of fn.
//
// The callback is the API rather than an Acquire/Release pair because the
// reference count is what keeps a database from being closed underneath a search
// — and a borrow that can be leaked by an early return is a collection that is
// never evicted again. fn's error is returned unchanged, so a handler can pass
// the database's own errors straight up.
//
// fn must not retain the database. Once it returns, the collection may be closed
// at any moment.
func (m *Manager) Use(name string, fn func(*govecdb.DB) error) error {
	c, err := m.acquire(context.Background(), false, name)
	if err != nil {
		return err
	}
	defer m.release(c)
	return fn(c.db)
}

// UseWait is Use for a caller that would rather wait for a slot than be told
// there is none: when MaxOpen collections are loaded and every one is borrowed,
// it waits until one is released or ctx is done, instead of returning
// ErrTooManyOpen at once.
//
// Use stays fail-fast, and request paths should keep calling it — a request
// queued behind a capacity limit becomes a timeout somewhere less visible. This
// is for background work that holds two sets of collections at once, such as a
// rebuild loading replacements while the live ones are still serving, where
// failing part way is worse than waiting. When ctx ends first the error is
// ErrTooManyOpen wrapping ctx's error, so both questions — "why did it fail"
// and "was it capacity" — answer with errors.Is.
func (m *Manager) UseWait(ctx context.Context, name string, fn func(*govecdb.DB) error) error {
	c, err := m.acquire(ctx, true, name)
	if err != nil {
		return err
	}
	defer m.release(c)
	return fn(c.db)
}

// acquire loads the collection if necessary and takes a reference to it. With
// wait false it refuses at once when there is no room; with wait true it waits
// for room until ctx is done.
//
// It is one loop because every step that releases the lock — reading a spec,
// evicting a victim, waiting — invalidates what was established before it, and
// starting again from the top is the only re-check that cannot forget a case.
func (m *Manager) acquire(ctx context.Context, wait bool, name string) (*collection, error) {
	if wait {
		// sync.Cond has no timed wait, so ctx ending is turned into a broadcast:
		// every waiter wakes, and the one whose ctx this was sees it is done.
		stop := context.AfterFunc(ctx, func() {
			m.mu.Lock()
			m.cond.Broadcast()
			m.mu.Unlock()
		})
		defer stop()
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	validated := false
	for {
		if m.closed {
			return nil, ErrClosed
		}
		if c, ok := m.cols[name]; ok {
			if c.dropping {
				return nil, fmt.Errorf("%w: %q", ErrNotFound, name)
			}
			if c.loading || c.closing {
				// Somebody else is inside Open or Close for this directory.
				// Waiting is not merely more efficient than opening it again: the
				// database refuses a second writer on one directory, so the race
				// this avoids would surface as ErrAlreadyOpen on an ordinary
				// request.
				m.cond.Wait()
				continue
			}
			c.refs++
			c.lastUsed = m.now()
			return c, nil
		}

		// Only a miss needs the name checked: the map only ever holds valid ones.
		if !validated {
			if err := ValidateName(name); err != nil {
				return nil, err
			}
			validated = true
		}
		spec, cached, err := m.specLocked(name)
		if err != nil {
			return nil, err
		}
		if !cached {
			continue // the lock was released to read it
		}

		evicted, err := m.makeRoomLocked()
		if err != nil {
			if !wait || !errors.Is(err, ErrTooManyOpen) {
				return nil, err
			}
			if ctx.Err() != nil {
				return nil, fmt.Errorf("%w: %w", err, ctx.Err())
			}
			// Every release that frees a slot broadcasts, as does ctx ending. The
			// whole lookup starts again afterwards: while this waited, someone
			// else may have loaded this very collection, or dropped it.
			m.cond.Wait()
			continue
		}
		if evicted {
			continue
		}
		return m.openLocked(name, spec)
	}
}

// openLocked opens a collection that is not in the map, behind a loading
// placeholder so concurrent callers wait on this open rather than starting
// another. Called with the lock held; it is released for the open itself.
func (m *Manager) openLocked(name string, spec Spec) (*collection, error) {
	c := &collection{name: name, dir: m.dir(name), spec: spec, loading: true}
	m.cols[name] = c

	m.mu.Unlock()
	db, err := govecdb.Open(filepath.Join(c.dir, dataSubdir), m.openOptions(name, spec)...)
	m.mu.Lock()

	if err != nil {
		delete(m.cols, name)
		m.cond.Broadcast()
		return nil, wrapOpen(name, err)
	}
	// Close and Drop can both have run while the lock was released. Neither could
	// see this database, so neither closed it, and it is ours to clean up.
	if m.closed || c.dropping {
		c.loading, c.closing = false, true
		m.mu.Unlock()
		m.closeDB(db)
		m.mu.Lock()
		delete(m.cols, name)
		m.cond.Broadcast()
		if m.closed {
			return nil, ErrClosed
		}
		return nil, fmt.Errorf("%w: %q", ErrNotFound, name)
	}

	c.loading = false
	c.db = db
	c.refs++
	c.lastUsed = m.now()
	m.cond.Broadcast()
	return c, nil
}

// specLocked returns a collection's spec. On a cache hit it returns with the
// lock never released and cached true. On a miss it reads the file with the
// lock released, caches what it read, and returns cached false: the caller's
// view of the map is stale and it must look again before acting on anything.
//
// A missing spec file is ErrNotFound and is not cached. Caching misses would
// make a collection created behind the manager's back invisible; and a lookup
// of a name that does not exist now costs one failed open with the lock
// released, which no longer slows anyone else down.
func (m *Manager) specLocked(name string) (spec Spec, cached bool, err error) {
	if s, ok := m.specs[name]; ok {
		return s, true, nil
	}
	gen := m.dropGen
	m.mu.Unlock()
	s, err := readSpec(m.dir(name))
	m.mu.Lock()
	if err != nil {
		return Spec{}, false, err
	}
	m.cacheSpecLocked(name, s, gen)
	return s, false, nil
}

// cacheSpecLocked records a spec read from disk while the lock was released,
// unless a Drop finished in the meantime — see Manager.dropGen.
func (m *Manager) cacheSpecLocked(name string, s Spec, gen uint64) {
	if m.dropGen == gen {
		m.specs[name] = s
	}
}

func (m *Manager) release(c *collection) {
	m.mu.Lock()
	defer m.mu.Unlock()

	c.refs--
	c.lastUsed = m.now()
	// Drop and Close both wait for this to reach zero.
	if c.refs == 0 {
		m.cond.Broadcast()
	}
}

// Drop closes a collection and deletes its directory.
//
// It waits for callers already inside Use to finish rather than refusing while
// one is in flight: a search takes microseconds and a drop is a deliberate
// administrative act, so failing it because a request happened to overlap would
// be a race the operator has no way to win. New borrowers are refused from the
// moment the drop is claimed, which is what lets the wait terminate.
//
// The close and the RemoveAll run with the manager's lock released. The entry
// stays in the map, marked dropping, until the directory is gone — that is what
// stops a concurrent lookup from reopening a directory halfway through being
// deleted. A collection that was not loaded gets such an entry too, for the
// same reason.
//
// The data is gone when this returns. There is no tombstone and no recycle bin.
func (m *Manager) Drop(name string) error {
	if err := ValidateName(name); err != nil {
		return err
	}

	m.mu.Lock()
	var c *collection
	for {
		if m.closed {
			m.mu.Unlock()
			return ErrClosed
		}
		existing, ok := m.cols[name]
		if !ok {
			c = &collection{name: name, dir: m.dir(name), dropping: true, closing: true}
			m.cols[name] = c
			break
		}
		if existing.dropping {
			// Another Drop got there first. Whichever finishes second reports the
			// collection as gone, which is what it is.
			m.mu.Unlock()
			return fmt.Errorf("%w: %q", ErrNotFound, name)
		}
		if existing.loading || existing.closing {
			m.cond.Wait()
			continue
		}
		c = existing
		c.dropping = true
		for c.refs > 0 {
			m.cond.Wait()
			if m.closed {
				// Left marked dropping, so nothing new can borrow it; Close
				// closes it once it is idle.
				m.mu.Unlock()
				return ErrClosed
			}
		}
		c.closing = true
		break
	}
	db := c.db
	m.mu.Unlock()

	if db != nil {
		m.closeDB(db)
	}
	err := m.removeDir(name)

	m.mu.Lock()
	delete(m.cols, name)
	delete(m.specs, name)
	m.dropGen++
	m.cond.Broadcast()
	m.mu.Unlock()
	return err
}

// removeDir is Drop's disk work.
func (m *Manager) removeDir(name string) error {
	dir := m.dir(name)
	if _, err := os.Stat(filepath.Join(dir, specFileName)); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%w: %q", ErrNotFound, name)
		}
		return fmt.Errorf("service: stat %q: %w", dir, err)
	}
	if err := os.RemoveAll(dir); err != nil {
		return fmt.Errorf("service: remove %q: %w", dir, err)
	}
	return syncDir(m.root)
}

// Info describes one collection.
type Info struct {
	// Name is the collection's name, which is also its directory.
	Name string

	// Spec is what it was created with, read from disk.
	Spec Spec

	// Loaded reports whether the collection currently holds an index in memory.
	// A collection that is not loaded is not idle in some lesser sense — it costs
	// nothing, and the first request for it pays a snapshot load and a replay.
	Loaded bool

	// Stats is the database's own report, and is the zero value when the
	// collection is not loaded. Nothing here loads a collection to fill it in:
	// listing a directory must not have the side effect of reading every index in
	// it into memory.
	Stats govecdb.Stats
}

// List reports every collection in the root directory, loaded or not, sorted by
// name.
//
// The directory is the source of truth rather than the in-memory map, because
// the map holds only what has been asked for since start-up. A spec that cannot
// be decoded is reported as an error rather than skipped: a collection quietly
// missing from a listing is how an operator concludes their data is gone.
//
// The manager's lock is held only to copy state in and out. The directory read,
// any spec not yet cached and every Stats call happen without it, because a
// listing is what /metrics runs on every scrape, and each of those can wait —
// Stats for as long as a Compact on that collection takes.
func (m *Manager) List() ([]Info, error) {
	m.mu.Lock()
	closed := m.closed
	m.mu.Unlock()
	if closed {
		return nil, ErrClosed
	}

	entries, err := os.ReadDir(m.root)
	if err != nil {
		return nil, fmt.Errorf("service: read %q: %w", m.root, err)
	}
	out := make([]Info, 0, len(entries))
	for _, e := range entries {
		// Not ours unless the name is valid. Something else put it here, and
		// refusing to guess is the same rule the spec-file marker exists for.
		if e.IsDir() && ValidateName(e.Name()) == nil {
			out = append(out, Info{Name: e.Name()})
		}
	}

	// Specs from the cache under the lock; whatever it lacks from disk without.
	known := make([]bool, len(out))
	m.mu.Lock()
	gen := m.dropGen
	for i := range out {
		out[i].Spec, known[i] = m.specs[out[i].Name]
	}
	m.mu.Unlock()

	read := 0
	for i := range out {
		if known[i] {
			continue
		}
		spec, err := readSpec(m.dir(out[i].Name))
		if errors.Is(err, ErrNotFound) {
			out[i].Name = "" // a directory that is not a collection, or no longer
			continue
		}
		if err != nil {
			return nil, err
		}
		out[i].Spec = spec
		read++
	}
	if read > 0 {
		m.mu.Lock()
		for i := range out {
			if !known[i] && out[i].Name != "" {
				m.cacheSpecLocked(out[i].Name, out[i].Spec, gen)
			}
		}
		m.mu.Unlock()
	}
	out = slices.DeleteFunc(out, func(i Info) bool { return i.Name == "" })
	slices.SortFunc(out, func(a, b Info) int { return strings.Compare(a.Name, b.Name) })

	if err := m.fillStats(out); err != nil {
		return nil, err
	}
	return out, nil
}

// fillStats sets Loaded and Stats on every entry whose collection is loaded,
// calling Stats with the lock released.
//
// A collection can be evicted while its Stats is being read. It is reported as
// loaded only if the entry that was read is still in the map afterwards and was
// never marked closing: Close is only ever called on an entry so marked, and the
// entry is removed only after Close returns, so "still there, not closing"
// proves the database was open for the whole call.
func (m *Manager) fillStats(infos []Info) error {
	type loaded struct {
		i int
		c *collection
		s govecdb.Stats
	}
	var ls []loaded

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return ErrClosed
	}
	for i := range infos {
		if c, ok := m.cols[infos[i].Name]; ok && servable(c) {
			ls = append(ls, loaded{i: i, c: c})
		}
	}
	m.mu.Unlock()

	for k := range ls {
		ls[k].s = m.stats(ls[k].c.db)
	}

	m.mu.Lock()
	for _, l := range ls {
		if m.cols[l.c.name] == l.c && servable(l.c) {
			infos[l.i].Loaded, infos[l.i].Stats = true, l.s
		}
	}
	m.mu.Unlock()
	return nil
}

// servable reports whether a collection's database is open and staying open:
// loaded, and neither being closed nor dropped.
func servable(c *collection) bool {
	return c.db != nil && !c.loading && !c.closing && !c.dropping
}

// Get reports one collection, loading nothing.
func (m *Manager) Get(name string) (Info, error) {
	if err := ValidateName(name); err != nil {
		return Info{}, err
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return Info{}, ErrClosed
	}
	spec, _, err := m.specLocked(name)
	m.mu.Unlock()
	if err != nil {
		return Info{}, err
	}
	infos := []Info{{Name: name, Spec: spec}}
	if err := m.fillStats(infos); err != nil {
		return Info{}, err
	}
	return infos[0], nil
}

// Loaded is how many collections currently hold an index in memory. A
// collection still inside its first open does not, yet.
func (m *Manager) Loaded() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	n := 0
	for _, c := range m.cols {
		if c.db != nil {
			n++
		}
	}
	return n
}

// Close closes every loaded collection and releases the manager.
//
// It waits for in-flight borrows to finish, for the same reason Drop does, and
// it is idempotent so it can sit in a defer next to a server's shutdown.
//
// # Idle collections first
//
// Every collection nobody is using is closed at once, and only then does Close
// wait for the busy ones. The order is the point: a collection under
// SyncInterval or SyncNever holds acknowledged writes in a user-space buffer
// until it is closed, and a shutdown that waited on one long Compact before
// flushing the others would lose those writes to the SIGKILL that ends an
// overrunning grace period.
//
// It does not snapshot. A shutdown that takes seconds per collection and fails
// on a full disk is exactly the shutdown an operator cannot afford; a caller who
// wants the next start to be fast asks for the snapshot explicitly.
func (m *Manager) Close() error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	m.closed = true
	// Wakes anything waiting on a load or a reference count, so it can see the
	// manager is closed and give up rather than wait for a signal that a stopped
	// sweeper will never send.
	m.cond.Broadcast()
	stop, done := m.stop, m.done
	m.mu.Unlock()

	// Stopped outside the lock: the sweeper takes it, so signalling and waiting
	// while holding it would deadlock.
	if stop != nil {
		close(stop)
		<-done
	}

	var firstErr error
	m.mu.Lock()
	for len(m.cols) > 0 {
		var ready []*collection
		for _, c := range m.cols {
			// A loading entry is finished by its opener, which sees the manager
			// closed and disposes of it; a closing one by whoever is closing it.
			if c.refs == 0 && !c.loading && !c.closing {
				c.closing = true
				ready = append(ready, c)
			}
		}
		if len(ready) == 0 {
			m.cond.Wait()
			continue
		}
		m.mu.Unlock()
		errs := m.closeAll(ready)
		m.mu.Lock()
		for i, c := range ready {
			if errs[i] != nil && firstErr == nil {
				firstErr = fmt.Errorf("service: close %q: %w", c.name, errs[i])
			}
			delete(m.cols, c.name)
		}
		m.cond.Broadcast()
	}
	m.mu.Unlock()

	// Last, once every collection is closed, so a manager that starts the
	// moment the root is free finds no database still open inside it.
	if err := m.lock.Release(); err != nil && firstErr == nil {
		firstErr = fmt.Errorf("service: release root: %w", err)
	}
	return firstErr
}

// closeAll closes the collections' databases concurrently, reporting each
// one's error at its index. Concurrently because each Close is a flush and an
// fsync, and a shutdown with a deadline should pay for the slowest rather than
// for the sum.
func (m *Manager) closeAll(cs []*collection) []error {
	errs := make([]error, len(cs))
	var wg sync.WaitGroup
	for i, c := range cs {
		if c.db == nil {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = m.closeDB(c.db)
		}()
	}
	wg.Wait()
	return errs
}

// makeRoomLocked enforces MaxOpen. If there is room it returns at once with
// the lock held throughout. If not, it closes the least recently used idle
// collection — releasing the lock while it does — and reports evicted, telling
// the caller its view of the map is stale. When everything loaded is busy it
// returns ErrTooManyOpen.
//
// One victim per call, and the caller loops: after the lock has been released
// there is no telling whether more room is still needed.
func (m *Manager) makeRoomLocked() (evicted bool, err error) {
	if m.opts.MaxOpen <= 0 {
		return false, nil
	}
	// Collections already on their way out do not count against the cap. If
	// they did, two openers racing for one slot would each evict a victim.
	occupied := 0
	for _, c := range m.cols {
		if !c.closing && !c.dropping {
			occupied++
		}
	}
	if occupied < m.opts.MaxOpen {
		return false, nil
	}
	victim := m.lruLocked()
	if victim == nil {
		return false, fmt.Errorf("%w: %d loaded, all in use", ErrTooManyOpen, occupied)
	}
	m.evictLocked([]*collection{victim})
	return true, nil
}

// lruLocked returns the evictable collection used longest ago, or nil if every
// loaded collection is busy.
func (m *Manager) lruLocked() *collection {
	var victim *collection
	for _, c := range m.cols {
		if !evictable(c) {
			continue
		}
		if victim == nil || c.lastUsed.Before(victim.lastUsed) {
			victim = c
		}
	}
	return victim
}

// evictable reports whether a collection may be closed for capacity or
// idleness: loaded, unborrowed, and not already being opened, closed or
// dropped by someone else.
func evictable(c *collection) bool {
	return c.refs == 0 && servable(c)
}

// evictLocked closes and unregisters collections the caller has established
// are evictable. Called with the lock held; it is released while the databases
// close — a Close waits out a snapshot in flight, which at millions of vectors
// is seconds — and held again on return.
//
// The Close error is dropped on purpose. Eviction is a memory decision, and the
// alternative — failing the request that happened to trigger the sweep, for a
// different collection's flush error — reports the problem to the one caller
// least able to act on it. The database itself is fail-closed, so a write that
// could not be made durable has already been refused at the write.
func (m *Manager) evictLocked(cs []*collection) {
	for _, c := range cs {
		c.closing = true
	}
	m.mu.Unlock()
	m.closeAll(cs)
	m.mu.Lock()
	for _, c := range cs {
		delete(m.cols, c.name)
	}
	m.cond.Broadcast()
}

// sweep closes collections nothing has touched for IdleTimeout.
func (m *Manager) sweep() {
	defer close(m.done)

	interval := m.opts.SweepInterval
	if interval <= 0 {
		// A quarter of the timeout bounds how long past it a collection can
		// linger, without waking a mostly idle process more often than that costs.
		interval = m.opts.IdleTimeout / 4
	}
	if interval <= 0 {
		interval = time.Second
	}

	t := time.NewTicker(interval)
	defer t.Stop()

	for {
		select {
		case <-m.stop:
			return
		case <-t.C:
			m.evictIdle()
		}
	}
}

func (m *Manager) evictIdle() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return
	}
	cutoff := m.now().Add(-m.opts.IdleTimeout)
	var victims []*collection
	for _, c := range m.cols {
		if evictable(c) && c.lastUsed.Before(cutoff) {
			victims = append(victims, c)
		}
	}
	if len(victims) > 0 {
		m.evictLocked(victims)
	}
}

func (m *Manager) dir(name string) string { return filepath.Join(m.root, name) }

// wrapOpen translates the database's configuration error into this package's, so
// a caller matches one sentinel whether the spec was rejected before it reached
// the disk or after. Everything else passes through with the collection named.
func wrapOpen(name string, err error) error {
	if errors.Is(err, govecdb.ErrInvalidConfig) {
		return fmt.Errorf("%w: %q: %w", ErrInvalidSpec, name, err)
	}
	return fmt.Errorf("service: open %q: %w", name, err)
}

// openOptions is the spec's options plus this manager's observer, bound to the
// collection's name. The observer is not in the spec because it is not a
// property of the collection: it is where this process sends its reports.
func (m *Manager) openOptions(name string, spec Spec) []govecdb.Option {
	opts := spec.options()
	if obs := m.opts.Observer; obs != nil {
		opts = append(opts, govecdb.WithObserver(func(e govecdb.Event) { obs(name, e) }))
	}
	return opts
}
