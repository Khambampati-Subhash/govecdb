package snapshot

import (
	"bytes"
	"testing"
)

// The format is the one thing here that can never change silently: bytes written
// by an older build have to keep decoding. These tests pin the layout and the
// refusals.

func TestHeaderRoundTrip(t *testing.T) {
	for _, seq := range []uint64{0, 1, 42, 1 << 40, ^uint64(0)} {
		var buf [headerSize]byte
		encodeHeader(buf[:], seq)

		if got := buf[:magicSize]; !bytes.Equal(got, magicBytes[:]) {
			t.Fatalf("magic = %q, want %q", got, magicBytes)
		}
		got, err := decodeHeader(buf[:])
		if err != nil {
			t.Fatal(err)
		}
		if got != seq {
			t.Fatalf("seq = %d, want %d", got, seq)
		}
	}
}

func TestHeaderRejects(t *testing.T) {
	t.Run("foreign file", func(t *testing.T) {
		if _, err := decodeHeader([]byte("not a snapshot!!")); err != ErrBadMagic {
			t.Fatalf("want ErrBadMagic, got %v", err)
		}
	})

	t.Run("future version", func(t *testing.T) {
		// Refusing is the entire point of carrying a version. A build that
		// guessed at a layout it does not know would apply nonsense.
		var buf [headerSize]byte
		encodeHeader(buf[:], 7)
		byteOrder.PutUint16(buf[magicSize:], formatVersion+1)
		if _, err := decodeHeader(buf[:]); err != ErrUnsupportedVersion {
			t.Fatalf("want ErrUnsupportedVersion, got %v", err)
		}
	})

	t.Run("truncated", func(t *testing.T) {
		if _, err := decodeHeader([]byte{'G', 'V'}); err != ErrShortSnapshot {
			t.Fatalf("want ErrShortSnapshot, got %v", err)
		}
	})
}

func TestTrailerRoundTrip(t *testing.T) {
	var buf [trailerSize]byte
	encodeTrailer(buf[:], 0xDEADBEEF, 1<<40)

	sum, length, err := decodeTrailer(buf[:])
	if err != nil {
		t.Fatal(err)
	}
	if sum != 0xDEADBEEF || length != 1<<40 {
		t.Fatalf("trailer = %#x/%d", sum, length)
	}
	if _, _, err := decodeTrailer(buf[:trailerSize-1]); err != ErrShortSnapshot {
		t.Fatalf("want ErrShortSnapshot, got %v", err)
	}
}

func TestPayloadLen(t *testing.T) {
	if _, ok := payloadLen(headerSize + trailerSize - 1); ok {
		t.Fatal("a file too small for its own framing was accepted")
	}
	got, ok := payloadLen(headerSize + trailerSize + 500)
	if !ok || got != 500 {
		t.Fatalf("payloadLen = %d/%v, want 500/true", got, ok)
	}
	// An empty payload is a legitimate snapshot: a database with nothing in it
	// still has a state, and it is still worth recording where the log had got to.
	if got, ok := payloadLen(headerSize + trailerSize); !ok || got != 0 {
		t.Fatalf("payloadLen on an empty payload = %d/%v", got, ok)
	}
}

// TestLayoutIsFrozen guards the numbers an on-disk format lives or dies by. If a
// field is resized these fail loudly here, rather than quietly making every
// existing snapshot unreadable.
func TestLayoutIsFrozen(t *testing.T) {
	if headerSize != 16 {
		t.Fatalf("header is %d bytes, format says 16 (magic4+version2+reserved2+seq8)", headerSize)
	}
	if trailerSize != 12 {
		t.Fatalf("trailer is %d bytes, format says 12 (crc4+len8)", trailerSize)
	}
	if formatVersion != 1 {
		t.Fatalf("format version is %d; bumping it is a migration, not an edit", formatVersion)
	}
}

func TestFileNameRoundTrip(t *testing.T) {
	for _, seq := range []uint64{0, 1, 42, 999999, ^uint64(0)} {
		name := fileName(seq)
		got, ok := parseFileName(name)
		if !ok {
			t.Fatalf("%q did not parse back", name)
		}
		if got != seq {
			t.Fatalf("%q parsed to %d, want %d", name, got, seq)
		}
	}
	for _, name := range []string{
		"", "snap-.snap", "snap-x.snap", "shot-00000000000000000001.snap",
		"snap-00000000000000000001.txt", "snap-00000000000000000001",
		// A temporary must never parse as a finished snapshot.
		"snap-123456.tmp",
	} {
		if _, ok := parseFileName(name); ok {
			t.Fatalf("%q was accepted as a snapshot name", name)
		}
	}
}

// TestFileNamesSortNewestLast pins the reason the names are 20 digits wide: the
// padding is what makes lexical order the same as numeric order, so listing a
// directory gives the ordering for free.
func TestFileNamesSortNewestLast(t *testing.T) {
	if a, b := fileName(9), fileName(10); !(a < b) {
		t.Fatalf("%q should sort before %q — the padding is not doing its job", a, b)
	}
	if a, b := fileName(1), fileName(^uint64(0)); !(a < b) {
		t.Fatalf("%q should sort before %q", a, b)
	}
}
