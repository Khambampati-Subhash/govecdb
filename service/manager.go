package service

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/khambampati-subhash/govecdb"
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

	// now is the clock, unexported because only this package's tests replace it.
	// Eviction is defined in elapsed time, and a test that demonstrates it by
	// sleeping demonstrates it slowly and then flakily.
	now func() time.Time
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
	// without waiting for it.
	now func() time.Time

	// mu guards everything below, and cond signals the two things a caller may
	// have to wait for: a collection finishing its open, and a collection's last
	// borrower going away.
	mu     sync.Mutex
	cond   *sync.Cond
	cols   map[string]*collection
	closed bool

	stop chan struct{}
	done chan struct{}
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

	m := &Manager{
		root: root,
		opts: opts,
		now:  opts.now,
		cols: make(map[string]*collection),
	}
	if m.now == nil {
		m.now = time.Now
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
func (m *Manager) Create(name string, spec Spec) (err error) {
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
	spec = spec.Defaults()

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return ErrClosed
	}
	if _, ok := m.cols[name]; ok {
		return fmt.Errorf("%w: %q", ErrExists, name)
	}

	dir := m.dir(name)
	if _, err := os.Stat(dir); err == nil {
		return fmt.Errorf("%w: %q", ErrExists, name)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("service: stat %q: %w", dir, err)
	}

	if err := m.makeRoomLocked(); err != nil {
		return err
	}

	if err := os.Mkdir(dir, dirPerm); err != nil {
		return fmt.Errorf("service: create %q: %w", dir, err)
	}
	defer func() {
		if err != nil {
			os.RemoveAll(dir)
		}
	}()
	// The directory entry itself has to be durable before the spec inside it is
	// worth anything.
	if err := syncDir(m.root); err != nil {
		return err
	}
	if err := writeSpec(dir, spec); err != nil {
		return err
	}

	db, err := govecdb.Open(filepath.Join(dir, dataSubdir), spec.options()...)
	if err != nil {
		return wrapOpen(name, err)
	}
	m.cols[name] = &collection{
		name:     name,
		dir:      dir,
		spec:     spec,
		db:       db,
		lastUsed: m.now(),
	}
	return nil
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
	c, err := m.acquire(name)
	if err != nil {
		return err
	}
	defer m.release(c)
	return fn(c.db)
}

// acquire loads the collection if necessary and takes a reference to it.
func (m *Manager) acquire(name string) (*collection, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for {
		if m.closed {
			return nil, ErrClosed
		}
		c, ok := m.cols[name]
		if !ok {
			break
		}
		if c.loading {
			// Somebody else is already inside Open for this directory. Waiting is
			// not merely more efficient than opening it a second time: the
			// database refuses a second writer on one directory, so the race this
			// avoids would surface as ErrAlreadyOpen on an ordinary request.
			m.cond.Wait()
			continue
		}
		if c.dropping {
			return nil, fmt.Errorf("%w: %q", ErrNotFound, name)
		}
		c.refs++
		c.lastUsed = m.now()
		return c, nil
	}

	if err := ValidateName(name); err != nil {
		return nil, err
	}
	dir := m.dir(name)
	spec, err := readSpec(dir)
	if err != nil {
		return nil, err
	}
	if err := m.makeRoomLocked(); err != nil {
		return nil, err
	}

	// Registered before the lock is released, so concurrent callers wait on this
	// open rather than starting another.
	c := &collection{name: name, dir: dir, spec: spec, loading: true}
	m.cols[name] = c

	m.mu.Unlock()
	db, err := govecdb.Open(filepath.Join(dir, dataSubdir), spec.options()...)
	m.mu.Lock()

	c.loading = false
	if err != nil {
		delete(m.cols, name)
		m.cond.Broadcast()
		return nil, wrapOpen(name, err)
	}
	// Close and Drop can both have run while the lock was released. Neither could
	// see this database, so neither closed it, and it is ours to clean up.
	if m.closed || c.dropping {
		delete(m.cols, name)
		m.cond.Broadcast()
		db.Close()
		if m.closed {
			return nil, ErrClosed
		}
		return nil, fmt.Errorf("%w: %q", ErrNotFound, name)
	}

	c.db = db
	c.refs++
	c.lastUsed = m.now()
	m.cond.Broadcast()
	return c, nil
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
// The data is gone when this returns. There is no tombstone and no recycle bin.
func (m *Manager) Drop(name string) error {
	if err := ValidateName(name); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for {
		if m.closed {
			return ErrClosed
		}
		c, ok := m.cols[name]
		if !ok {
			break
		}
		if c.loading {
			m.cond.Wait()
			continue
		}
		if c.dropping {
			// Another Drop got there first. Whichever finishes second reports the
			// collection as gone, which is what it is.
			return fmt.Errorf("%w: %q", ErrNotFound, name)
		}
		c.dropping = true
		for c.refs > 0 {
			m.cond.Wait()
			if m.closed {
				return ErrClosed
			}
		}
		delete(m.cols, name)
		if c.db != nil {
			c.db.Close()
		}
		m.cond.Broadcast()
		break
	}

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
func (m *Manager) List() ([]Info, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return nil, ErrClosed
	}

	entries, err := os.ReadDir(m.root)
	if err != nil {
		return nil, fmt.Errorf("service: read %q: %w", m.root, err)
	}

	out := make([]Info, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if ValidateName(name) != nil {
			// Not ours. Something else put it here, and refusing to guess is the
			// same rule the spec-file marker exists for.
			continue
		}
		spec, err := readSpec(m.dir(name))
		if err != nil {
			if errors.Is(err, ErrNotFound) {
				continue
			}
			return nil, err
		}
		info := Info{Name: name, Spec: spec}
		if c, ok := m.cols[name]; ok && c.db != nil && !c.dropping {
			info.Loaded = true
			info.Stats = c.db.Stats()
		}
		out = append(out, info)
	}
	slices.SortFunc(out, func(a, b Info) int { return strings.Compare(a.Name, b.Name) })
	return out, nil
}

// Get reports one collection, loading nothing.
func (m *Manager) Get(name string) (Info, error) {
	if err := ValidateName(name); err != nil {
		return Info{}, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return Info{}, ErrClosed
	}
	spec, err := readSpec(m.dir(name))
	if err != nil {
		return Info{}, err
	}
	info := Info{Name: name, Spec: spec}
	if c, ok := m.cols[name]; ok && c.db != nil && !c.dropping {
		info.Loaded = true
		info.Stats = c.db.Stats()
	}
	return info, nil
}

// Loaded is how many collections currently hold an index in memory.
func (m *Manager) Loaded() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.cols)
}

// Close closes every loaded collection and releases the manager.
//
// It waits for in-flight borrows to finish, for the same reason Drop does, and
// it is idempotent so it can sit in a defer next to a server's shutdown.
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

	m.mu.Lock()
	defer m.mu.Unlock()

	var firstErr error
	for name, c := range m.cols {
		for c.refs > 0 || c.loading {
			m.cond.Wait()
		}
		if c.db != nil {
			if err := c.db.Close(); err != nil && firstErr == nil {
				firstErr = fmt.Errorf("service: close %q: %w", name, err)
			}
		}
		delete(m.cols, name)
	}
	return firstErr
}

// makeRoomLocked enforces MaxOpen by closing the least recently used idle
// collection. Callers hold the lock.
func (m *Manager) makeRoomLocked() error {
	if m.opts.MaxOpen <= 0 {
		return nil
	}
	for len(m.cols) >= m.opts.MaxOpen {
		victim := m.lruLocked()
		if victim == nil {
			return fmt.Errorf("%w: %d loaded, all in use", ErrTooManyOpen, len(m.cols))
		}
		m.closeLocked(victim)
	}
	return nil
}

// lruLocked returns the evictable collection used longest ago, or nil if every
// loaded collection is busy. A loading or dropping collection is never a victim:
// one is mid-open with the lock released, the other is already being disposed of.
func (m *Manager) lruLocked() *collection {
	var victim *collection
	for _, c := range m.cols {
		if c.refs > 0 || c.loading || c.dropping || c.db == nil {
			continue
		}
		if victim == nil || c.lastUsed.Before(victim.lastUsed) {
			victim = c
		}
	}
	return victim
}

// closeLocked closes and unregisters a collection. Callers hold the lock and
// must have established that nothing is using it.
//
// The Close error is dropped on purpose. Eviction is a memory decision, and the
// alternative — failing the request that happened to trigger the sweep, for a
// different collection's flush error — reports the problem to the one caller
// least able to act on it. The database itself is fail-closed, so a write that
// could not be made durable has already been refused at the write.
func (m *Manager) closeLocked(c *collection) {
	if c.db != nil {
		c.db.Close()
	}
	delete(m.cols, c.name)
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
	for _, c := range m.cols {
		if c.refs > 0 || c.loading || c.dropping || c.db == nil {
			continue
		}
		if c.lastUsed.Before(cutoff) {
			m.closeLocked(c)
		}
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
