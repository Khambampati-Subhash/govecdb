package snapshot

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// create writes a snapshot whose payload is the given bytes.
func create(t *testing.T, dir string, seq uint64, payload []byte) Snapshot {
	t.Helper()

	snap, err := Create(dir, seq, func(w io.Writer) error {
		_, err := w.Write(payload)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return snap
}

// load reads a snapshot back, returning the payload and the result.
func load(t *testing.T, dir string) ([]byte, Result) {
	t.Helper()

	var got []byte
	res, err := Load(dir, func(r io.Reader) error {
		var err error
		got, err = io.ReadAll(r)
		return err
	})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return got, res
}

func TestCreateLoadRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name    string
		seq     uint64
		payload []byte
	}{
		{"small", 1, []byte("the state of the world")},
		// An empty payload is legitimate: a database with nothing in it still
		// has a state, and where the log had got to is still worth recording.
		{"empty", 2, nil},
		{"binary", 3, []byte{0x00, 0xff, 0x00, 0xff}},
		{"large", 4, bytes.Repeat([]byte("v"), 3<<20)},
		{"max sequence", ^uint64(0), []byte("x")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			snap := create(t, dir, tc.seq, tc.payload)

			if snap.Seq != tc.seq {
				t.Fatalf("Seq = %d, want %d", snap.Seq, tc.seq)
			}
			if want := int64(headerSize + len(tc.payload) + trailerSize); snap.Bytes != want {
				t.Fatalf("Bytes = %d, want %d", snap.Bytes, want)
			}

			got, res := load(t, dir)
			if !res.Found {
				t.Fatal("the snapshot just written was not found")
			}
			if res.Snapshot.Seq != tc.seq {
				t.Fatalf("loaded seq %d, want %d", res.Snapshot.Seq, tc.seq)
			}
			if !bytes.Equal(got, tc.payload) {
				t.Fatalf("payload came back %d bytes, wrote %d", len(got), len(tc.payload))
			}
			if len(res.Rejected) != 0 {
				t.Fatalf("a snapshot just written was rejected: %v", res.Rejected)
			}
		})
	}
}

func TestLoadWithNoSnapshots(t *testing.T) {
	// Neither of these is an error. A database that has never snapshotted
	// replays the whole WAL, which is exactly what the first startup does.
	t.Run("missing directory", func(t *testing.T) {
		got, res := load(t, filepath.Join(t.TempDir(), "nope"))
		if res.Found || got != nil {
			t.Fatalf("%+v", res)
		}
	})

	t.Run("empty directory", func(t *testing.T) {
		if _, res := load(t, t.TempDir()); res.Found {
			t.Fatal("an empty directory produced a snapshot")
		}
	})

	t.Run("foreign files only", func(t *testing.T) {
		dir := t.TempDir()
		for _, name := range []string{".DS_Store", "notes.txt", "snap-.snap"} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte("junk"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if _, res := load(t, dir); res.Found {
			t.Fatal("a stray file was read as a snapshot")
		}
	})
}

func TestLatestPicksTheHighestSequence(t *testing.T) {
	dir := t.TempDir()
	// Written out of order on purpose: the sequence decides, not the mtime.
	for _, seq := range []uint64{5, 100, 9, 2} {
		create(t, dir, seq, fmt.Appendf(nil, "state at %d", seq))
	}

	snap, ok, err := Latest(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || snap.Seq != 100 {
		t.Fatalf("Latest = %+v/%v, want seq 100", snap, ok)
	}

	all, err := List(dir)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint64{100, 9, 5, 2}
	if len(all) != len(want) {
		t.Fatalf("List returned %d snapshots, want %d", len(all), len(want))
	}
	for i, s := range all {
		if s.Seq != want[i] {
			t.Fatalf("List[%d] = %d, want %d — not newest first", i, s.Seq, want[i])
		}
	}

	got, res := load(t, dir)
	if res.Snapshot.Seq != 100 || string(got) != "state at 100" {
		t.Fatalf("Load took seq %d (%q), want the newest", res.Snapshot.Seq, got)
	}
}

func TestCreateIsAtomic(t *testing.T) {
	t.Run("no temporary survives success", func(t *testing.T) {
		dir := t.TempDir()
		create(t, dir, 1, []byte("done"))

		if n := countTemps(t, dir); n != 0 {
			t.Fatalf("%d temporary files left behind by a successful snapshot", n)
		}
	})

	t.Run("a failed payload leaves nothing", func(t *testing.T) {
		// The point of writing to a temporary: a payload that could not be
		// produced must not become a snapshot that cannot be trusted.
		dir := t.TempDir()
		boom := errors.New("could not serialize")
		_, err := Create(dir, 1, func(w io.Writer) error {
			if _, err := w.Write([]byte("half of the state")); err != nil {
				return err
			}
			return boom
		})
		if !errors.Is(err, boom) {
			t.Fatalf("Create = %v, want the callback's error", err)
		}

		if n := countTemps(t, dir); n != 0 {
			t.Fatalf("a failed snapshot left %d temporary files", n)
		}
		if _, res := load(t, dir); res.Found {
			t.Fatal("a failed snapshot became loadable")
		}
	})
}

func countTemps(t *testing.T, dir string) int {
	t.Helper()

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), tempSuffix) {
			n++
		}
	}
	return n
}

// corrupt flips a byte at the given offset in a snapshot file.
func corrupt(t *testing.T, path string, offset int64) {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[offset] ^= 0x01
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestLoadRejects covers every way a snapshot can be wrong. In each case the
// callback must never run: the whole reason verification is a separate pass is
// that a caller cannot un-apply half a bad snapshot.
func TestLoadRejects(t *testing.T) {
	payload := bytes.Repeat([]byte("state"), 100)

	for _, tc := range []struct {
		name   string
		damage func(t *testing.T, dir string, snap Snapshot)
		want   error
	}{
		{
			name: "corrupt payload",
			damage: func(t *testing.T, _ string, snap Snapshot) {
				corrupt(t, snap.Path, headerSize+10)
			},
			want: ErrChecksum,
		},
		{
			// The dangerous one. A snapshot whose sequence is wrong by one
			// replays the log from the wrong place, which silently loses or
			// duplicates a write — so the sequence is checksummed, not trusted.
			name: "corrupt sequence in the header",
			damage: func(t *testing.T, _ string, snap Snapshot) {
				corrupt(t, snap.Path, magicSize+versionSize+reservedSize)
			},
			want: ErrSeqMismatch,
		},
		{
			name: "corrupt version",
			damage: func(t *testing.T, _ string, snap Snapshot) {
				corrupt(t, snap.Path, magicSize)
			},
			want: ErrUnsupportedVersion,
		},
		{
			name: "corrupt magic",
			damage: func(t *testing.T, _ string, snap Snapshot) {
				corrupt(t, snap.Path, 0)
			},
			want: ErrBadMagic,
		},
		{
			name: "truncated",
			damage: func(t *testing.T, _ string, snap Snapshot) {
				if err := os.Truncate(snap.Path, snap.Bytes-20); err != nil {
					t.Fatal(err)
				}
			},
			want: ErrLengthMismatch,
		},
		{
			name: "smaller than its own framing",
			damage: func(t *testing.T, _ string, snap Snapshot) {
				if err := os.Truncate(snap.Path, 4); err != nil {
					t.Fatal(err)
				}
			},
			want: ErrShortSnapshot,
		},
		{
			name: "extended with garbage",
			damage: func(t *testing.T, _ string, snap Snapshot) {
				f, err := os.OpenFile(snap.Path, os.O_WRONLY|os.O_APPEND, 0o644)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := f.Write([]byte("extra")); err != nil {
					t.Fatal(err)
				}
				if err := f.Close(); err != nil {
					t.Fatal(err)
				}
			},
			want: ErrLengthMismatch,
		},
		{
			// The name is not covered by the checksum and the header is, so a
			// renamed file now claims a sequence its contents do not support.
			name: "renamed to a different sequence",
			damage: func(t *testing.T, dir string, snap Snapshot) {
				if err := os.Rename(snap.Path, filePath(dir, snap.Seq+1)); err != nil {
					t.Fatal(err)
				}
			},
			want: ErrSeqMismatch,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			snap := create(t, dir, 7, payload)
			tc.damage(t, dir, snap)

			applied := false
			res, err := Load(dir, func(io.Reader) error {
				applied = true
				return nil
			})
			if err != nil {
				t.Fatalf("Load returned an error rather than rejecting: %v", err)
			}
			if applied {
				t.Fatal("the callback was given bytes from a snapshot that failed verification")
			}
			if res.Found {
				t.Fatal("a damaged snapshot was reported as loaded")
			}
			if len(res.Rejected) != 1 {
				t.Fatalf("rejected = %v, want exactly one", res.Rejected)
			}
			if !errors.Is(res.Rejected[0].Cause, tc.want) {
				t.Fatalf("rejected because %v, want %v", res.Rejected[0].Cause, tc.want)
			}
		})
	}
}

// TestLoadFallsBackToAnOlderSnapshot is the reason retention keeps more than
// one. A corrupt newest snapshot should cost a longer WAL replay, not the
// database.
func TestLoadFallsBackToAnOlderSnapshot(t *testing.T) {
	dir := t.TempDir()
	create(t, dir, 10, []byte("state at 10"))
	create(t, dir, 20, []byte("state at 20"))
	newest := create(t, dir, 30, []byte("state at 30"))

	corrupt(t, newest.Path, headerSize+2)

	got, res := load(t, dir)
	if !res.Found {
		t.Fatal("a corrupt newest snapshot took the whole directory with it")
	}
	if res.Snapshot.Seq != 20 || string(got) != "state at 20" {
		t.Fatalf("fell back to seq %d (%q), want 20", res.Snapshot.Seq, got)
	}
	if len(res.Rejected) != 1 || res.Rejected[0].Snapshot.Seq != 30 {
		t.Fatalf("rejected = %v, want just seq 30", res.Rejected)
	}
	// The rejection is the only place a failing disk shows up, so it has to name
	// the file an operator would go and look at.
	if !strings.Contains(res.Rejected[0].String(), fileName(30)) {
		t.Fatalf("Rejection.String() = %q, does not name the file", res.Rejected[0])
	}
}

func TestLoadWhenEverySnapshotIsCorrupt(t *testing.T) {
	dir := t.TempDir()
	for _, seq := range []uint64{1, 2, 3} {
		snap := create(t, dir, seq, []byte("state"))
		corrupt(t, snap.Path, headerSize+1)
	}

	// Not an error: replaying the whole WAL is slow, but it is correct, and it
	// is strictly better than refusing to start.
	got, res := load(t, dir)
	if res.Found || got != nil {
		t.Fatalf("%+v", res)
	}
	if len(res.Rejected) != 3 {
		t.Fatalf("rejected = %v, want all three", res.Rejected)
	}
}

// TestLoadReturnsTheCallbacksError pins the other half of the fallback rule. A
// snapshot that verified but could not be decoded is a bug in the decoder, and
// quietly falling back to an older file would hide it behind a slow startup.
func TestLoadReturnsTheCallbacksError(t *testing.T) {
	dir := t.TempDir()
	create(t, dir, 1, []byte("older"))
	create(t, dir, 2, []byte("newer"))

	boom := errors.New("cannot decode")
	res, err := Load(dir, func(io.Reader) error { return boom })
	if !errors.Is(err, boom) {
		t.Fatalf("Load = %v, want the callback's error", err)
	}
	if res.Found {
		t.Fatal("a snapshot the callback rejected was reported as loaded")
	}
}

// TestLoadBoundsTheCallbackToThePayload keeps a greedy decoder out of the
// trailer: asking for more than the payload has to end, not leak framing.
func TestLoadBoundsTheCallbackToThePayload(t *testing.T) {
	dir := t.TempDir()
	payload := []byte("exactly this much")
	create(t, dir, 1, payload)

	var got []byte
	if _, err := Load(dir, func(r io.Reader) error {
		buf := make([]byte, len(payload)+64)
		n, err := io.ReadFull(r, buf)
		if !errors.Is(err, io.ErrUnexpectedEOF) {
			return fmt.Errorf("reading past the payload returned %v, want ErrUnexpectedEOF", err)
		}
		got = buf[:n]
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("read %q, want %q", got, payload)
	}
}

func TestPrune(t *testing.T) {
	t.Run("keeps the newest", func(t *testing.T) {
		dir := t.TempDir()
		for _, seq := range []uint64{1, 2, 3, 4, 5} {
			create(t, dir, seq, fmt.Appendf(nil, "state %d", seq))
		}

		removed, err := Prune(dir, 2)
		if err != nil {
			t.Fatal(err)
		}
		if removed != 3 {
			t.Fatalf("removed %d snapshots, want 3", removed)
		}

		all, err := List(dir)
		if err != nil {
			t.Fatal(err)
		}
		if len(all) != 2 || all[0].Seq != 5 || all[1].Seq != 4 {
			t.Fatalf("kept %+v, want seq 5 and 4", all)
		}
		// What survives has to still be loadable — retention that leaves an
		// unusable file is worse than none.
		got, res := load(t, dir)
		if !res.Found || string(got) != "state 5" {
			t.Fatalf("after pruning, Load gave %q (%+v)", got, res)
		}
	})

	t.Run("fewer snapshots than the retention", func(t *testing.T) {
		dir := t.TempDir()
		create(t, dir, 1, []byte("only"))

		removed, err := Prune(dir, 5)
		if err != nil {
			t.Fatal(err)
		}
		if removed != 0 {
			t.Fatalf("removed %d from a directory with one snapshot", removed)
		}
	})

	t.Run("empty directory", func(t *testing.T) {
		removed, err := Prune(t.TempDir(), 1)
		if err != nil || removed != 0 {
			t.Fatalf("Prune on an empty directory = %d, %v", removed, err)
		}
	})

	t.Run("refuses to keep none", func(t *testing.T) {
		// Deleting the last snapshot is not retention. It would leave recovery
		// with nothing but a log nobody is allowed to truncate.
		dir := t.TempDir()
		create(t, dir, 1, []byte("state"))

		for _, keep := range []int{0, -1} {
			if _, err := Prune(dir, keep); !errors.Is(err, ErrKeepTooFew) {
				t.Fatalf("Prune(keep=%d) = %v, want ErrKeepTooFew", keep, err)
			}
		}
		if all, _ := List(dir); len(all) != 1 {
			t.Fatal("a refused Prune deleted something anyway")
		}
	})

	t.Run("clears temporaries a crash left behind", func(t *testing.T) {
		// Create removes its own temporary on failure, so anything still here
		// outlived the process that made it and nothing will ever read it.
		dir := t.TempDir()
		create(t, dir, 1, []byte("state"))
		for i := range 3 {
			name := filepath.Join(dir, fmt.Sprintf("%s%d.tmp", filePrefix, i))
			if err := os.WriteFile(name, []byte("abandoned"), 0o644); err != nil {
				t.Fatal(err)
			}
		}

		removed, err := Prune(dir, 1)
		if err != nil {
			t.Fatal(err)
		}
		if removed != 3 {
			t.Fatalf("removed %d files, want the 3 orphaned temporaries", removed)
		}
		if n := countTemps(t, dir); n != 0 {
			t.Fatalf("%d temporaries survived Prune", n)
		}
		if all, _ := List(dir); len(all) != 1 {
			t.Fatal("Prune took the snapshot along with the temporaries")
		}
	})
}

// TestCreateOverwritesTheSameSequence covers snapshotting twice at one sequence
// — a retry after a failure upstream. The rename makes the second one replace
// the first rather than the directory growing a duplicate.
func TestCreateOverwritesTheSameSequence(t *testing.T) {
	dir := t.TempDir()
	create(t, dir, 5, []byte("first attempt"))
	create(t, dir, 5, []byte("second attempt"))

	all, err := List(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 1 {
		t.Fatalf("%d files for one sequence, want 1", len(all))
	}
	if got, _ := load(t, dir); string(got) != "second attempt" {
		t.Fatalf("payload = %q, want the second attempt", got)
	}
}

// TestCreateStreamsWithoutBufferingItAll writes a payload in many small chunks,
// which is how a real encoder emits one: node by node rather than as a single
// slice the caller had to assemble in memory first.
func TestCreateStreamsWithoutBufferingItAll(t *testing.T) {
	dir := t.TempDir()
	const chunks = 5000

	snap, err := Create(dir, 1, func(w io.Writer) error {
		for i := range chunks {
			if _, err := fmt.Fprintf(w, "%06d", i); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := int64(headerSize + chunks*6 + trailerSize); snap.Bytes != want {
		t.Fatalf("Bytes = %d, want %d", snap.Bytes, want)
	}

	got, res := load(t, dir)
	if !res.Found || len(got) != chunks*6 {
		t.Fatalf("read %d bytes (%+v)", len(got), res)
	}
	for i := range chunks {
		if want := fmt.Sprintf("%06d", i); string(got[i*6:(i+1)*6]) != want {
			t.Fatalf("chunk %d = %q, want %q", i, got[i*6:(i+1)*6], want)
		}
	}
}
