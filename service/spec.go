package service

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/khambampati-subhash/govecdb"
)

// specFileName is the marker that makes a directory a collection.
//
// List uses it rather than "every subdirectory of the root", so an operator's
// stray tarball or a half-removed directory is not reported as a database.
const specFileName = "collection.json"

// dataSubdir holds the database itself, one level below the spec.
//
// Separate so the collection directory has exactly one file this package writes
// and one directory it does not touch. A future format that adds a second file
// here cannot then collide with something govecdb decided to name.
const dataSubdir = "data"

// specVersion is the spec file's format version. It is written so that a future
// reader can tell an old file from a corrupt one, which is the whole job of a
// version field; nothing reads it yet beyond refusing what it does not know.
const specVersion = 1

// dirPerm matches the root package: a collection holds embeddings, which are
// derived from whatever was embedded and are frequently reversible enough to
// matter, so nothing here is world-readable.
const dirPerm os.FileMode = 0o700

// Spec is everything about a collection that survives a restart.
//
// The structural fields — Dimension, Metric, M — are why this file exists.
// Reopening a database under a different dimension, metric or M is refused,
// correctly, so those three cannot be supplied at start-up by whoever wrote the
// command line: they are decided once, when the collection is created, and read
// back every time after.
//
// The rest are durability and search policy, which a restart may legitimately
// change. They are recorded anyway, so that "what is this collection actually
// doing" is answered by looking at it rather than by reconstructing the flags of
// whichever process created it.
type Spec struct {
	// Dimension is the vector length. Required; there is no useful default.
	Dimension int

	// Metric is the distance function. Structural.
	Metric govecdb.Metric

	// M is the index's neighbours-per-node. Structural. Zero means the library
	// default, which Defaults resolves before anything is written down.
	M int

	// EfConstruction is the build-time search width. Zero means the default.
	EfConstruction int

	// Seed makes index construction reproducible. Zero means the default.
	Seed int64

	// SyncPolicy decides when writes reach stable storage. The zero value is
	// govecdb.SyncAlways, which is the safe answer rather than the fast one.
	SyncPolicy govecdb.SyncPolicy

	// SyncInterval is how often SyncInterval fsyncs. Zero means the default.
	SyncInterval time.Duration

	// SnapshotInterval takes a snapshot on a timer. Zero leaves it off.
	//
	// Worth setting for a service in a way it is not for a library: nobody is
	// standing over a server to call Snapshot before a restart, and without one
	// start-up replays the whole log.
	SnapshotInterval time.Duration

	// SnapshotsKept is how many snapshots to retain. Zero means the default of 2,
	// which is what makes a corrupt newest snapshot survivable.
	SnapshotsKept int

	// TargetRecall is the recall a search aims for when it does not name its own
	// width. Zero means the default.
	TargetRecall float64
}

// Defaults returns the spec with every zero replaced by the value the library
// would have chosen.
//
// It is called before a spec is written, never after it is read. Recording the
// effective value is the point: a spec that stored "0, meaning whatever the
// default is" would let an upgrade of this module silently rebuild an existing
// collection's index under a different M — the one class of change the format is
// explicitly not allowed to make.
func (s Spec) Defaults() Spec {
	if s.M == 0 {
		s.M = 16
	}
	if s.EfConstruction == 0 {
		s.EfConstruction = 200
	}
	if s.Seed == 0 {
		s.Seed = 1
	}
	if s.SyncInterval == 0 {
		s.SyncInterval = 50 * time.Millisecond
	}
	if s.SnapshotsKept == 0 {
		s.SnapshotsKept = 2
	}
	if s.TargetRecall == 0 {
		s.TargetRecall = 0.95
	}
	return s
}

// options renders the spec as the functional options govecdb.Open takes.
//
// Note what is not here: no validation of the values. The bounds live in the
// root package's options and are enforced when Open runs them, so restating them
// here would be a second copy to drift. What this package owns is that Dimension
// was supplied at all, because that error is worth reporting before a directory
// is created for a collection that cannot be built.
func (s Spec) options() []govecdb.Option {
	opts := []govecdb.Option{
		govecdb.WithDimension(s.Dimension),
		govecdb.WithMetric(s.Metric),
		govecdb.WithM(s.M),
		govecdb.WithEfConstruction(s.EfConstruction),
		govecdb.WithSeed(s.Seed),
		govecdb.WithSyncPolicy(s.SyncPolicy),
		govecdb.WithSyncInterval(s.SyncInterval),
		govecdb.WithSnapshotsKept(s.SnapshotsKept),
		govecdb.WithSearchTargetRecall(s.TargetRecall),
	}
	if s.SnapshotInterval > 0 {
		opts = append(opts, govecdb.WithSnapshotInterval(s.SnapshotInterval))
	}
	return opts
}

// specFile is the on-disk shape, kept separate from Spec so the wire format is
// visible in one place and does not move when a Go field is renamed.
//
// Enums are strings and durations are strings: a spec file is something an
// operator reads while working out why a collection behaves the way it does, and
// "cosine" and "5m" answer that where 0 and 300000000000 do not.
type specFile struct {
	Version          int     `json:"version"`
	Dimension        int     `json:"dimension"`
	Metric           string  `json:"metric"`
	M                int     `json:"m"`
	EfConstruction   int     `json:"ef_construction"`
	Seed             int64   `json:"seed"`
	SyncPolicy       string  `json:"sync_policy"`
	SyncInterval     string  `json:"sync_interval"`
	SnapshotInterval string  `json:"snapshot_interval,omitempty"`
	SnapshotsKept    int     `json:"snapshots_kept"`
	TargetRecall     float64 `json:"target_recall"`
}

func (s Spec) file() specFile {
	f := specFile{
		Version:        specVersion,
		Dimension:      s.Dimension,
		Metric:         s.Metric.String(),
		M:              s.M,
		EfConstruction: s.EfConstruction,
		Seed:           s.Seed,
		SyncPolicy:     s.SyncPolicy.String(),
		SyncInterval:   s.SyncInterval.String(),
		SnapshotsKept:  s.SnapshotsKept,
		TargetRecall:   s.TargetRecall,
	}
	if s.SnapshotInterval > 0 {
		f.SnapshotInterval = s.SnapshotInterval.String()
	}
	return f
}

func (f specFile) spec() (Spec, error) {
	if f.Version != specVersion {
		return Spec{}, fmt.Errorf("%w: version %d, this build understands %d",
			ErrInvalidSpec, f.Version, specVersion)
	}
	metric, err := ParseMetric(f.Metric)
	if err != nil {
		return Spec{}, err
	}
	policy, err := ParseSyncPolicy(f.SyncPolicy)
	if err != nil {
		return Spec{}, err
	}
	sync, err := parseDuration(f.SyncInterval, "sync_interval")
	if err != nil {
		return Spec{}, err
	}
	snap, err := parseDuration(f.SnapshotInterval, "snapshot_interval")
	if err != nil {
		return Spec{}, err
	}
	return Spec{
		Dimension:        f.Dimension,
		Metric:           metric,
		M:                f.M,
		EfConstruction:   f.EfConstruction,
		Seed:             f.Seed,
		SyncPolicy:       policy,
		SyncInterval:     sync,
		SnapshotInterval: snap,
		SnapshotsKept:    f.SnapshotsKept,
		TargetRecall:     f.TargetRecall,
	}, nil
}

func parseDuration(s, field string) (time.Duration, error) {
	if s == "" {
		return 0, nil
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return 0, fmt.Errorf("%w: %s: %s", ErrInvalidSpec, field, err)
	}
	if d < 0 {
		return 0, fmt.Errorf("%w: %s is negative", ErrInvalidSpec, field)
	}
	return d, nil
}

// ParseMetric is the inverse of govecdb.Metric.String.
//
// It lives here rather than in the root package because the root package has no
// reason to parse: a Go caller writes govecdb.Cosine. Everything that needs a
// string came from a config file or a request body, which is this layer.
func ParseMetric(s string) (govecdb.Metric, error) {
	switch s {
	case "cosine", "":
		return govecdb.Cosine, nil
	case "euclidean":
		return govecdb.Euclidean, nil
	case "dotproduct":
		return govecdb.DotProduct, nil
	default:
		return 0, fmt.Errorf("%w: unknown metric %q, want cosine, euclidean or dotproduct",
			ErrInvalidSpec, s)
	}
}

// ParseSyncPolicy is the inverse of govecdb.SyncPolicy.String.
//
// The empty string is SyncAlways, matching the zero value: a policy omitted from
// a request body must not be the fast one, for the same reason the library's zero
// value is not.
func ParseSyncPolicy(s string) (govecdb.SyncPolicy, error) {
	switch s {
	case "always", "":
		return govecdb.SyncAlways, nil
	case "interval":
		return govecdb.SyncInterval, nil
	case "never":
		return govecdb.SyncNever, nil
	default:
		return 0, fmt.Errorf("%w: unknown sync policy %q, want always, interval or never",
			ErrInvalidSpec, s)
	}
}

// readSpec loads a collection's spec file.
//
// A missing file is ErrNotFound rather than an I/O error: it is how "there is no
// collection here" is spelled on disk, and every caller asks that question first.
func readSpec(dir string) (Spec, error) {
	b, err := os.ReadFile(filepath.Join(dir, specFileName))
	if err != nil {
		if os.IsNotExist(err) {
			return Spec{}, fmt.Errorf("%w: %s", ErrNotFound, filepath.Base(dir))
		}
		return Spec{}, fmt.Errorf("%w: %s", ErrCorruptSpec, err)
	}
	var f specFile
	if err := json.Unmarshal(b, &f); err != nil {
		return Spec{}, fmt.Errorf("%w: %s: %s", ErrCorruptSpec, filepath.Base(dir), err)
	}
	return f.spec()
}

// writeSpec writes a collection's spec file atomically.
//
// Temp, fsync, rename, fsync the directory — the same sequence snapshots use,
// for the same reason. This file is small and written once, but it is also the
// only record of how to reopen the index sitting next to it: a half-written spec
// after a crash would be a collection nothing can load, with the data intact and
// unreachable.
func writeSpec(dir string, s Spec) (err error) {
	b, err := json.MarshalIndent(s.file(), "", "  ")
	if err != nil {
		return fmt.Errorf("service: encode spec: %w", err)
	}
	b = append(b, '\n')

	tmp, err := os.CreateTemp(dir, specFileName+".tmp-*")
	if err != nil {
		return fmt.Errorf("service: create temp spec: %w", err)
	}
	defer func() {
		if err != nil {
			os.Remove(tmp.Name())
		}
	}()

	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		return fmt.Errorf("service: write spec: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("service: sync spec: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("service: close spec: %w", err)
	}
	if err := os.Rename(tmp.Name(), filepath.Join(dir, specFileName)); err != nil {
		return fmt.Errorf("service: rename spec: %w", err)
	}
	return syncDir(dir)
}

// syncDir makes a directory entry durable.
//
// fsync on a file makes its contents durable and says nothing about the name
// pointing at it, so without this a crash can take the renamed spec away and
// leave neither name in place. The error is returned rather than swallowed: a
// silent failure here downgrades the guarantee to nothing while looking exactly
// like success.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return fmt.Errorf("service: open %q: %w", dir, err)
	}
	defer d.Close()
	if err := d.Sync(); err != nil {
		return fmt.Errorf("service: sync %q: %w", dir, err)
	}
	return nil
}
