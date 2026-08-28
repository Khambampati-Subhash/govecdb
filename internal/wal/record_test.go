package wal

import (
	"bytes"
	"testing"
)

// The record format is the one thing here that can never change silently: bytes
// written by an older build have to keep decoding. These tests pin the layout,
// the checksum's coverage, and the refusals.

func TestFileHeaderRoundTrip(t *testing.T) {
	var buf [fileHeaderSize]byte
	encodeFileHeader(buf[:])

	if got := buf[:magicSize]; !bytes.Equal(got, magicBytes[:]) {
		t.Fatalf("magic = %q, want %q", got, magicBytes)
	}
	version, err := decodeFileHeader(buf[:])
	if err != nil {
		t.Fatal(err)
	}
	if version != formatVersion {
		t.Fatalf("version = %d, want %d", version, formatVersion)
	}
}

func TestFileHeaderRejects(t *testing.T) {
	t.Run("foreign file", func(t *testing.T) {
		// The common case by far: a path pointing at something that is not a
		// log. Saying so beats trying to parse it as records.
		if _, err := decodeFileHeader([]byte("not a wal at all")); err != ErrBadMagic {
			t.Fatalf("want ErrBadMagic, got %v", err)
		}
	})

	t.Run("future version", func(t *testing.T) {
		var buf [fileHeaderSize]byte
		encodeFileHeader(buf[:])
		byteOrder.PutUint16(buf[magicSize:], formatVersion+1)

		// Refusing is the entire point of carrying a version. A build that
		// guessed at a layout it does not know would corrupt on write.
		if _, err := decodeFileHeader(buf[:]); err != ErrUnsupportedVersion {
			t.Fatalf("want ErrUnsupportedVersion, got %v", err)
		}
	})

	t.Run("truncated", func(t *testing.T) {
		if _, err := decodeFileHeader([]byte{'G', 'V'}); err != ErrShortRecord {
			t.Fatalf("want ErrShortRecord, got %v", err)
		}
	})
}

func TestRecordHeaderRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name    string
		typ     RecordType
		seq     uint64
		payload []byte
	}{
		{"put", TypePut, 1, []byte("hello")},
		{"delete", TypeDelete, 42, []byte("id-to-remove")},
		{"checkpoint", TypeCheckpoint, 1 << 40, []byte{0x00, 0xff, 0x00}},
		{"empty payload", TypePut, 7, nil},
		{"max sequence", TypePut, ^uint64(0), []byte("x")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var hdr [recordHeaderSize]byte
			encodeRecordHeader(hdr[:], tc.typ, tc.seq, tc.payload)

			_, typ, seq, length, err := decodeRecordHeader(hdr[:])
			if err != nil {
				t.Fatal(err)
			}
			if typ != tc.typ {
				t.Fatalf("type = %v, want %v", typ, tc.typ)
			}
			if seq != tc.seq {
				t.Fatalf("seq = %d, want %d", seq, tc.seq)
			}
			if int(length) != len(tc.payload) {
				t.Fatalf("length = %d, want %d", length, len(tc.payload))
			}
			if !verifyChecksum(hdr[:], tc.payload) {
				t.Fatal("checksum failed on an untouched record")
			}
		})
	}
}

// TestChecksumCoversEveryField is the test that matters for recovery. A checksum
// over the payload alone would leave the fields that decide *how* the payload is
// read unprotected — and a corrupt length is far more dangerous than a corrupt
// payload, because it turns a wrong answer into a wild read.
func TestChecksumCoversEveryField(t *testing.T) {
	payload := []byte("the quick brown fox")

	build := func() []byte {
		hdr := make([]byte, recordHeaderSize)
		encodeRecordHeader(hdr, TypePut, 99, payload)
		return hdr
	}

	t.Run("payload corruption", func(t *testing.T) {
		hdr := build()
		bad := append([]byte(nil), payload...)
		bad[3] ^= 0x01
		if verifyChecksum(hdr, bad) {
			t.Fatal("a flipped payload bit passed the checksum")
		}
	})

	t.Run("type corruption", func(t *testing.T) {
		hdr := build()
		hdr[crcSize] = byte(TypeDelete) // was TypePut
		if verifyChecksum(hdr, payload) {
			t.Fatal("a rewritten type passed the checksum")
		}
	})

	t.Run("sequence corruption", func(t *testing.T) {
		hdr := build()
		hdr[crcSize+typeSize] ^= 0x01
		if verifyChecksum(hdr, payload) {
			t.Fatal("a flipped sequence bit passed the checksum")
		}
	})

	t.Run("length corruption", func(t *testing.T) {
		// The dangerous one: this is what would make a reader try to allocate
		// and read an arbitrary number of bytes.
		hdr := build()
		byteOrder.PutUint32(hdr[crcSize+typeSize+seqSize:], 1<<30)
		if verifyChecksum(hdr, payload) {
			t.Fatal("a rewritten length passed the checksum")
		}
	})

	t.Run("checksum corruption", func(t *testing.T) {
		hdr := build()
		hdr[0] ^= 0xff
		if verifyChecksum(hdr, payload) {
			t.Fatal("a corrupt checksum field still validated")
		}
	})
}

func TestDecodeRecordHeaderShort(t *testing.T) {
	// A file ending mid-header is what power loss leaves behind, and it must be
	// reported as truncation rather than read past the end of the slice.
	if _, _, _, _, err := decodeRecordHeader(make([]byte, recordHeaderSize-1)); err != ErrShortRecord {
		t.Fatalf("want ErrShortRecord, got %v", err)
	}
}

func TestRecordTypeValidity(t *testing.T) {
	// Zero must never be a valid type: unwritten space, a filesystem hole and a
	// zero-filled block all read as zeros, and none of them is a record.
	if TypeInvalid.valid() {
		t.Fatal("zero decoded as a valid record type")
	}
	for _, typ := range []RecordType{TypePut, TypeDelete, TypeCheckpoint} {
		if !typ.valid() {
			t.Fatalf("%v should be valid", typ)
		}
	}
	if RecordType(200).valid() {
		t.Fatal("an unknown type was accepted")
	}
	if got := TypeInvalid.String(); got != "INVALID" {
		t.Fatalf("String() = %q", got)
	}
}

// TestLayoutIsFrozen guards the numbers an on-disk format lives or dies by. If a
// field is resized, these fail loudly here rather than quietly making every
// existing log unreadable.
func TestLayoutIsFrozen(t *testing.T) {
	if fileHeaderSize != 8 {
		t.Fatalf("file header is %d bytes, format says 8", fileHeaderSize)
	}
	if recordHeaderSize != 17 {
		t.Fatalf("record header is %d bytes, format says 17 (crc4+type1+seq8+len4)", recordHeaderSize)
	}
	if formatVersion != 1 {
		t.Fatalf("format version is %d; bumping it is a migration, not an edit", formatVersion)
	}
}
