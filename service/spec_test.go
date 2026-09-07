package service

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/khambampati-subhash/govecdb"
)

func TestSpecRoundTrips(t *testing.T) {
	dir := t.TempDir()

	want := Spec{
		Dimension:        768,
		Metric:           govecdb.DotProduct,
		M:                32,
		EfConstruction:   150,
		Seed:             42,
		SyncPolicy:       govecdb.SyncInterval,
		SyncInterval:     25 * time.Millisecond,
		SnapshotInterval: 5 * time.Minute,
		SnapshotsKept:    3,
		TargetRecall:     0.9,
	}
	if err := writeSpec(dir, want); err != nil {
		t.Fatalf("writeSpec: %v", err)
	}
	got, err := readSpec(dir)
	if err != nil {
		t.Fatalf("readSpec: %v", err)
	}
	if got != want {
		t.Fatalf("round trip:\n got %+v\nwant %+v", got, want)
	}
}

// The file is something an operator reads while working out why a collection
// behaves the way it does, so the enums and durations stay legible rather than
// becoming integers.
func TestSpecFileIsReadable(t *testing.T) {
	dir := t.TempDir()
	spec := Spec{
		Dimension:        4,
		Metric:           govecdb.Euclidean,
		SyncPolicy:       govecdb.SyncNever,
		SnapshotInterval: 5 * time.Minute,
	}.Defaults()
	if err := writeSpec(dir, spec); err != nil {
		t.Fatal(err)
	}

	b, err := os.ReadFile(filepath.Join(dir, specFileName))
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatalf("the spec file is not valid JSON: %v", err)
	}
	for field, want := range map[string]any{
		"version":           float64(specVersion),
		"metric":            "euclidean",
		"sync_policy":       "never",
		"snapshot_interval": "5m0s",
	} {
		if raw[field] != want {
			t.Errorf("%s = %v, want %v", field, raw[field], want)
		}
	}
}

func TestReadSpecReportsAMissingFileAsNotFound(t *testing.T) {
	if _, err := readSpec(t.TempDir()); !errors.Is(err, ErrNotFound) {
		t.Fatalf("readSpec on an empty directory = %v, want ErrNotFound", err)
	}
}

// A corrupt spec and an invalid one have different fixes — a disk to look at
// versus a request to correct — so they are different errors.
func TestReadSpecSeparatesCorruptFromInvalid(t *testing.T) {
	corrupt := t.TempDir()
	if err := os.WriteFile(filepath.Join(corrupt, specFileName), []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readSpec(corrupt); !errors.Is(err, ErrCorruptSpec) {
		t.Fatalf("readSpec on a truncated file = %v, want ErrCorruptSpec", err)
	}

	future := t.TempDir()
	body := []byte(`{"version": 99, "dimension": 4, "metric": "cosine", "sync_policy": "always"}`)
	if err := os.WriteFile(filepath.Join(future, specFileName), body, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readSpec(future); !errors.Is(err, ErrInvalidSpec) {
		t.Fatalf("readSpec on a future version = %v, want ErrInvalidSpec", err)
	}
}

func TestDefaultsLeaveSetValuesAlone(t *testing.T) {
	spec := Spec{Dimension: 4, M: 8, EfConstruction: 64, Seed: 7, SnapshotsKept: 1, TargetRecall: 0.8}
	got := spec.Defaults()
	if got.M != 8 || got.EfConstruction != 64 || got.Seed != 7 ||
		got.SnapshotsKept != 1 || got.TargetRecall != 0.8 {
		t.Fatalf("Defaults overwrote a value that was set: %+v", got)
	}
}

func TestParseMetric(t *testing.T) {
	for in, want := range map[string]govecdb.Metric{
		"":           govecdb.Cosine,
		"cosine":     govecdb.Cosine,
		"euclidean":  govecdb.Euclidean,
		"dotproduct": govecdb.DotProduct,
	} {
		got, err := ParseMetric(in)
		if err != nil || got != want {
			t.Errorf("ParseMetric(%q) = %v, %v; want %v, nil", in, got, err, want)
		}
		// The two directions have to agree, or a spec written by one release
		// stops being readable by the next.
		if in != "" && got.String() != in {
			t.Errorf("%q.String() = %q, want %q", in, got.String(), in)
		}
	}
	if _, err := ParseMetric("manhattan"); !errors.Is(err, ErrInvalidSpec) {
		t.Errorf("ParseMetric on an unknown metric = %v, want ErrInvalidSpec", err)
	}
}

func TestParseSyncPolicy(t *testing.T) {
	for in, want := range map[string]govecdb.SyncPolicy{
		"":         govecdb.SyncAlways, // omitted must not mean the fast one
		"always":   govecdb.SyncAlways,
		"interval": govecdb.SyncInterval,
		"never":    govecdb.SyncNever,
	} {
		got, err := ParseSyncPolicy(in)
		if err != nil || got != want {
			t.Errorf("ParseSyncPolicy(%q) = %v, %v; want %v, nil", in, got, err, want)
		}
		if in != "" && got.String() != in {
			t.Errorf("%q.String() = %q, want %q", in, got.String(), in)
		}
	}
	if _, err := ParseSyncPolicy("sometimes"); !errors.Is(err, ErrInvalidSpec) {
		t.Errorf("ParseSyncPolicy on an unknown policy = %v, want ErrInvalidSpec", err)
	}
}
