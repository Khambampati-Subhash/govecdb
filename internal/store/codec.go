package store

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"unicode/utf8"
)

// The metadata wire format, used both for a WAL record's payload and for the
// metadata section of a snapshot.
//
//	entry count      uint16
//	per entry:       keyLen uint16 | key | tag uint8 | value
//	  tag 1 string:  len uint32 | bytes
//	  tag 2 bool:    uint8, 0 or 1
//	  tag 3 int64:   8 bytes
//	  tag 4 float64: 8 bytes
//
// # Decoding is the security boundary
//
// These bytes come off a disk this process does not control. Everything below
// therefore refuses before it allocates: every length is checked against a cap
// before a byte is read for it, every tag must be one of four known values, and
// there is no path by which the wire decides what *type* to construct beyond
// those four. That is the whole reason the value set is closed — a decoder that
// can be talked into building arbitrary types from type names on the wire is a
// much larger surface than anything metadata filtering needs.
//
// Tag zero is invalid on purpose, so a run of zero bytes cannot decode as an
// entry. Same rule as the WAL's record types, for the same reason.

const (
	tagInvalid uint8 = 0
	tagString  uint8 = 1
	tagBool    uint8 = 2
	tagInt64   uint8 = 3
	tagFloat64 uint8 = 4
)

// Limits on what encoded metadata may claim, enforced on the way in *and* on the
// way out. A cap the writer does not honour is a record that can be written and
// never read back, which is worse than either limit alone.
const (
	// MaxKeys caps entries in one metadata map.
	MaxKeys = 256
	// MaxKeyBytes caps one key.
	MaxKeyBytes = 256
	// MaxStringBytes caps one string value.
	MaxStringBytes = 64 << 10
	// MaxEncodedBytes caps a whole encoded map, which is the bound that actually
	// protects memory: the per-field caps multiplied together are far larger.
	MaxEncodedBytes = 1 << 20
)

var byteOrder = binary.LittleEndian

var (
	// ErrMetadataValueType means a value is not one of the four supported types.
	ErrMetadataValueType = errors.New("store: unsupported metadata value type")
	// ErrMetadataTooLarge means a key, value, or map exceeds its cap.
	ErrMetadataTooLarge = errors.New("store: metadata exceeds limit")
	// ErrMetadataKey means a key is empty or not valid UTF-8.
	ErrMetadataKey = errors.New("store: invalid metadata key")
	// ErrMetadataCorrupt means encoded metadata could not be decoded.
	ErrMetadataCorrupt = errors.New("store: corrupt metadata")
)

// Validate reports whether md can be stored and encoded.
//
// It runs at the API boundary rather than at encode time so a caller learns that
// their value is unsupported when they pass it, with the key named, instead of
// discovering it later from a write that failed halfway.
func Validate(md Metadata) error {
	if len(md) > MaxKeys {
		return fmt.Errorf("%w: %d keys, max %d", ErrMetadataTooLarge, len(md), MaxKeys)
	}
	for k, v := range md {
		if k == "" || !utf8.ValidString(k) {
			return fmt.Errorf("%w: %q", ErrMetadataKey, k)
		}
		if len(k) > MaxKeyBytes {
			return fmt.Errorf("%w: key %q is %d bytes, max %d", ErrMetadataTooLarge, k, len(k), MaxKeyBytes)
		}
		switch val := v.(type) {
		case string:
			if len(val) > MaxStringBytes {
				return fmt.Errorf("%w: value for %q is %d bytes, max %d",
					ErrMetadataTooLarge, k, len(val), MaxStringBytes)
			}
			if !utf8.ValidString(val) {
				return fmt.Errorf("%w: value for %q is not valid UTF-8", ErrMetadataValueType, k)
			}
		case bool, int64, float64:
			// Fine.
		default:
			return fmt.Errorf("%w: %q is %T, want string, bool, int64 or float64", ErrMetadataValueType, k, v)
		}
	}
	if n := EncodedLen(md); n > MaxEncodedBytes {
		return fmt.Errorf("%w: %d encoded bytes, max %d", ErrMetadataTooLarge, n, MaxEncodedBytes)
	}
	return nil
}

// EncodedLen is exactly how many bytes Encode will produce, so a caller can
// check a limit without building the encoding to measure it.
func EncodedLen(md Metadata) int {
	n := 2 // entry count
	for k, v := range md {
		n += 2 + len(k) + 1 // key length, key, tag
		switch val := v.(type) {
		case string:
			n += 4 + len(val)
		case bool:
			n++
		case int64, float64:
			n += 8
		}
	}
	return n
}

// Encode appends md to dst. Callers pass a reused buffer to keep the write path
// free of per-record allocation.
//
// Map iteration order is random, so two encodings of the same map differ byte
// for byte. That is fine here — nothing compares encoded metadata for equality,
// and imposing a sort would cost every write to make no caller's life better.
func Encode(dst []byte, md Metadata) []byte {
	dst = byteOrder.AppendUint16(dst, uint16(len(md)))
	for k, v := range md {
		dst = byteOrder.AppendUint16(dst, uint16(len(k)))
		dst = append(dst, k...)
		switch val := v.(type) {
		case string:
			dst = append(dst, tagString)
			dst = byteOrder.AppendUint32(dst, uint32(len(val)))
			dst = append(dst, val...)
		case bool:
			dst = append(dst, tagBool)
			if val {
				dst = append(dst, 1)
			} else {
				dst = append(dst, 0)
			}
		case int64:
			dst = append(dst, tagInt64)
			dst = byteOrder.AppendUint64(dst, uint64(val))
		case float64:
			dst = append(dst, tagFloat64)
			dst = byteOrder.AppendUint64(dst, math.Float64bits(val))
		default:
			// Unreachable: Validate ran at the boundary. Encoding a tag of zero
			// rather than panicking means a bug here produces a record that
			// fails to decode loudly instead of one that decodes wrongly.
			dst = append(dst, tagInvalid)
		}
	}
	return dst
}

// Decode reads metadata from src and returns the number of bytes consumed, so a
// caller can continue reading whatever follows it in the same buffer.
func Decode(src []byte) (Metadata, int, error) {
	d := &decoder{src: src}
	md, err := d.metadata()
	if err != nil {
		return nil, 0, err
	}
	return md, d.off, nil
}

type decoder struct {
	src []byte
	off int
}

// take returns the next n bytes, refusing rather than slicing past the end.
// Every read in this file goes through it, which is what makes "nothing is
// allocated on an unchecked length" a property of the decoder rather than a
// habit that has to hold at each call site.
func (d *decoder) take(n int) ([]byte, error) {
	if n < 0 || n > len(d.src)-d.off {
		return nil, fmt.Errorf("%w: wanted %d bytes, %d remain", ErrMetadataCorrupt, n, len(d.src)-d.off)
	}
	b := d.src[d.off : d.off+n]
	d.off += n
	return b, nil
}

func (d *decoder) metadata() (Metadata, error) {
	b, err := d.take(2)
	if err != nil {
		return nil, err
	}
	count := int(byteOrder.Uint16(b))
	if count > MaxKeys {
		return nil, fmt.Errorf("%w: %d keys, max %d", ErrMetadataTooLarge, count, MaxKeys)
	}
	if count == 0 {
		return nil, nil
	}

	md := make(Metadata, count)
	for range count {
		b, err := d.take(2)
		if err != nil {
			return nil, err
		}
		keyLen := int(byteOrder.Uint16(b))
		if keyLen == 0 || keyLen > MaxKeyBytes {
			return nil, fmt.Errorf("%w: key length %d", ErrMetadataCorrupt, keyLen)
		}
		kb, err := d.take(keyLen)
		if err != nil {
			return nil, err
		}
		if !utf8.Valid(kb) {
			return nil, fmt.Errorf("%w: key is not valid UTF-8", ErrMetadataCorrupt)
		}
		key := string(kb)

		val, err := d.value(key)
		if err != nil {
			return nil, err
		}
		// A duplicate key means the encoding did not come from a Go map, which
		// means it did not come from Encode. Refusing beats silently keeping
		// whichever copy happened to be last.
		if _, dup := md[key]; dup {
			return nil, fmt.Errorf("%w: duplicate key %q", ErrMetadataCorrupt, key)
		}
		md[key] = val
	}
	return md, nil
}

func (d *decoder) value(key string) (any, error) {
	b, err := d.take(1)
	if err != nil {
		return nil, err
	}
	switch tag := b[0]; tag {
	case tagString:
		b, err := d.take(4)
		if err != nil {
			return nil, err
		}
		n := int64(byteOrder.Uint32(b))
		if n > MaxStringBytes {
			return nil, fmt.Errorf("%w: string for %q is %d bytes, max %d",
				ErrMetadataTooLarge, key, n, MaxStringBytes)
		}
		sb, err := d.take(int(n))
		if err != nil {
			return nil, err
		}
		if !utf8.Valid(sb) {
			return nil, fmt.Errorf("%w: value for %q is not valid UTF-8", ErrMetadataCorrupt, key)
		}
		return string(sb), nil

	case tagBool:
		b, err := d.take(1)
		if err != nil {
			return nil, err
		}
		// Only 0 and 1 are booleans. Accepting anything non-zero as true would
		// let two different byte strings mean the same value, which is exactly
		// the kind of slack that makes a format hard to reason about later.
		switch b[0] {
		case 0:
			return false, nil
		case 1:
			return true, nil
		default:
			return nil, fmt.Errorf("%w: bool for %q is %d", ErrMetadataCorrupt, key, b[0])
		}

	case tagInt64:
		b, err := d.take(8)
		if err != nil {
			return nil, err
		}
		return int64(byteOrder.Uint64(b)), nil

	case tagFloat64:
		b, err := d.take(8)
		if err != nil {
			return nil, err
		}
		return math.Float64frombits(byteOrder.Uint64(b)), nil

	default:
		return nil, fmt.Errorf("%w: tag %d for key %q", ErrMetadataCorrupt, tag, key)
	}
}

// The snapshot section. A store is written as a count and then that many
// id/metadata pairs, self-delimiting so it can sit next to other sections in one
// payload without a length prefix — which matters because a length prefix would
// mean buffering the whole thing to measure it.

const (
	// snapshotVersion is bumped when the section layout changes incompatibly.
	snapshotVersion uint8 = 1
	// maxEntries caps how many ids one snapshot section may claim, so a corrupt
	// count cannot ask for an unbounded map. Well beyond any real corpus.
	maxEntries = 1 << 32
)

// WriteTo writes the store as a snapshot section.
//
// It takes a *bufio.Writer rather than an io.Writer so it can share one buffer
// with the sections around it. Wrapping separately would let this section's
// buffer swallow bytes belonging to the next one.
func (s *Map) WriteTo(w *bufio.Writer) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if err := w.WriteByte(snapshotVersion); err != nil {
		return fmt.Errorf("store: write version: %w", err)
	}
	var num [8]byte
	byteOrder.PutUint64(num[:], uint64(len(s.m)))
	if _, err := w.Write(num[:]); err != nil {
		return fmt.Errorf("store: write count: %w", err)
	}

	buf := make([]byte, 0, 1024)
	for id, md := range s.m {
		buf = buf[:0]
		buf = byteOrder.AppendUint32(buf, uint32(len(id)))
		buf = append(buf, id...)
		buf = Encode(buf, md)
		if _, err := w.Write(buf); err != nil {
			return fmt.Errorf("store: write entry: %w", err)
		}
	}
	return nil
}

// ReadFrom replaces the store's contents with a snapshot section.
//
// Like WriteTo it takes the shared *bufio.Reader, so it consumes exactly its own
// bytes and leaves the reader positioned for whatever follows.
func (s *Map) ReadFrom(r *bufio.Reader, maxIDBytes int) error {
	version, err := r.ReadByte()
	if err != nil {
		return fmt.Errorf("store: read version: %w", shortOr(err))
	}
	if version != snapshotVersion {
		return fmt.Errorf("%w: snapshot section version %d", ErrMetadataCorrupt, version)
	}

	var num [8]byte
	if _, err := io.ReadFull(r, num[:]); err != nil {
		return fmt.Errorf("store: read count: %w", shortOr(err))
	}
	count := byteOrder.Uint64(num[:])
	if count > maxEntries {
		return fmt.Errorf("%w: %d entries", ErrMetadataCorrupt, count)
	}

	// Built aside and swapped in, so a section that fails halfway leaves the
	// store as it was rather than half-replaced.
	m := make(map[string]Metadata, min(count, 1<<16))

	var idLen [4]byte
	body := make([]byte, 0, 1024)
	for range count {
		if _, err := io.ReadFull(r, idLen[:]); err != nil {
			return fmt.Errorf("store: read id length: %w", shortOr(err))
		}
		n := int64(byteOrder.Uint32(idLen[:]))
		if n == 0 || n > int64(maxIDBytes) {
			return fmt.Errorf("%w: id length %d, max %d", ErrMetadataCorrupt, n, maxIDBytes)
		}
		if cap(body) < int(n) {
			body = make([]byte, n)
		}
		idb := body[:n]
		if _, err := io.ReadFull(r, idb); err != nil {
			return fmt.Errorf("store: read id: %w", shortOr(err))
		}
		if !utf8.Valid(idb) {
			return fmt.Errorf("%w: id is not valid UTF-8", ErrMetadataCorrupt)
		}
		id := string(idb)

		md, err := readMetadata(r)
		if err != nil {
			return err
		}
		if _, dup := m[id]; dup {
			return fmt.Errorf("%w: duplicate id %q", ErrMetadataCorrupt, id)
		}
		m[id] = md
	}

	s.mu.Lock()
	s.m = m
	s.mu.Unlock()
	return nil
}

// readMetadata pulls one encoded map off a stream. It reads the entry count,
// then the exact number of bytes those entries need — which is why every
// variable-length field is length-prefixed rather than terminated.
func readMetadata(r *bufio.Reader) (Metadata, error) {
	var count [2]byte
	if _, err := io.ReadFull(r, count[:]); err != nil {
		return nil, fmt.Errorf("store: read metadata count: %w", shortOr(err))
	}
	n := int(byteOrder.Uint16(count[:]))
	if n > MaxKeys {
		return nil, fmt.Errorf("%w: %d keys", ErrMetadataTooLarge, n)
	}
	if n == 0 {
		return nil, nil
	}

	// Reassemble the entries into the buffer Decode understands, so there is one
	// implementation of the entry format rather than two that can drift.
	buf := make([]byte, 0, 256)
	buf = byteOrder.AppendUint16(buf, uint16(n))
	for range n {
		var err error
		if buf, err = copyLenPrefixed(r, buf, 2, MaxKeyBytes); err != nil {
			return nil, err
		}
		tag, err := r.ReadByte()
		if err != nil {
			return nil, fmt.Errorf("store: read tag: %w", shortOr(err))
		}
		buf = append(buf, tag)
		switch tag {
		case tagString:
			if buf, err = copyLenPrefixed(r, buf, 4, MaxStringBytes); err != nil {
				return nil, err
			}
		case tagBool:
			buf, err = copyFixed(r, buf, 1)
		case tagInt64, tagFloat64:
			buf, err = copyFixed(r, buf, 8)
		default:
			return nil, fmt.Errorf("%w: tag %d", ErrMetadataCorrupt, tag)
		}
		if err != nil {
			return nil, err
		}
	}

	md, _, err := Decode(buf)
	return md, err
}

// copyLenPrefixed moves one length-prefixed field from r into buf, refusing a
// length above max before reading a byte of the body.
func copyLenPrefixed(r *bufio.Reader, buf []byte, lenSize, max int) ([]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:lenSize]); err != nil {
		return nil, fmt.Errorf("store: read length: %w", shortOr(err))
	}
	var n int64
	if lenSize == 2 {
		n = int64(byteOrder.Uint16(hdr[:2]))
	} else {
		n = int64(byteOrder.Uint32(hdr[:4]))
	}
	if n > int64(max) {
		return nil, fmt.Errorf("%w: field of %d bytes, max %d", ErrMetadataTooLarge, n, max)
	}
	buf = append(buf, hdr[:lenSize]...)
	return copyFixed(r, buf, int(n))
}

func copyFixed(r *bufio.Reader, buf []byte, n int) ([]byte, error) {
	start := len(buf)
	buf = append(buf, make([]byte, n)...)
	if _, err := io.ReadFull(r, buf[start:]); err != nil {
		return nil, fmt.Errorf("store: read field: %w", shortOr(err))
	}
	return buf, nil
}

func shortOr(err error) error {
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}
