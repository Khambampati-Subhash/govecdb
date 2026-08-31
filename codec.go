package govecdb

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"unicode/utf8"

	"github.com/khambampati-subhash/govecdb/internal/store"
	"github.com/khambampati-subhash/govecdb/internal/wal"
)

// Domain encoding: turning an operation into the opaque bytes the log carries,
// and a snapshot into the bytes the snapshot store carries.
//
// The log knows nothing about vectors on purpose — it moves bytes durably and
// stays testable without an index — so this is the layer that decides what a PUT
// actually contains. Which means this is also where bytes from a disk become
// live objects again, and every decode below refuses before it allocates.
//
//	PUT payload:     version(1) | idLen(2) | id | dim(4) | values | metadata
//	DELETE payload:  version(1) | idLen(2) | id
//
// Metadata is encoded by internal/store, which owns that format because it owns
// the closed set of types that makes it safe to decode.

const (
	// recordVersion is bumped when a payload layout changes incompatibly. It is
	// per-record rather than per-file so a future format can be introduced
	// without rewriting logs that already exist.
	recordVersion uint8 = 1

	recordVersionSize = 1
	recordIDLenSize   = 2
	recordDimSize     = 4
)

var byteOrder = binary.LittleEndian

// encodePut appends a PUT payload to dst, which callers reuse so the write path
// does not allocate per record.
func encodePut(dst []byte, v Vector) []byte {
	dst = append(dst, recordVersion)
	dst = byteOrder.AppendUint16(dst, uint16(len(v.ID)))
	dst = append(dst, v.ID...)
	dst = byteOrder.AppendUint32(dst, uint32(len(v.Values)))
	for _, f := range v.Values {
		dst = byteOrder.AppendUint32(dst, math.Float32bits(f))
	}
	return store.Encode(dst, v.Metadata)
}

// encodeDelete appends a DELETE payload to dst.
func encodeDelete(dst []byte, id string) []byte {
	dst = append(dst, recordVersion)
	dst = byteOrder.AppendUint16(dst, uint16(len(id)))
	return append(dst, id...)
}

// decodeRecord turns a log payload back into an operation.
//
// The limits are passed in rather than read from a package constant because they
// are the *database's* configured limits: a record that could not have been
// written by this database should not be applied by it either. In particular a
// dimension is checked against the configured one before any slice is made for
// it, so a corrupt length cannot become an arbitrary allocation.
//
// The record's checksum has already been verified by the log, so this is not
// guarding against random corruption — that is caught upstream. It is guarding
// against a payload that passes its checksum and still does not describe a
// record this database can hold: a log from a differently configured database,
// or a file somebody edited.
func decodeRecord(typ wal.RecordType, payload []byte, o *options) (Vector, bool, error) {
	d := &payloadDecoder{src: payload}

	version, err := d.u8()
	if err != nil {
		return Vector{}, false, err
	}
	if version != recordVersion {
		return Vector{}, false, fmt.Errorf("%w: record version %d, this build writes %d",
			ErrCorrupt, version, recordVersion)
	}

	idLen, err := d.u16()
	if err != nil {
		return Vector{}, false, err
	}
	if idLen == 0 || int(idLen) > o.maxIDBytes {
		return Vector{}, false, fmt.Errorf("%w: id length %d, max %d", ErrCorrupt, idLen, o.maxIDBytes)
	}
	idb, err := d.take(int(idLen))
	if err != nil {
		return Vector{}, false, err
	}
	if !utf8.Valid(idb) {
		return Vector{}, false, fmt.Errorf("%w: id is not valid UTF-8", ErrCorrupt)
	}
	v := Vector{ID: string(idb)}

	if typ == wal.TypeDelete {
		if d.off != len(d.src) {
			return Vector{}, false, fmt.Errorf("%w: %d trailing bytes on a DELETE",
				ErrCorrupt, len(d.src)-d.off)
		}
		return v, false, nil
	}

	dim, err := d.u32()
	if err != nil {
		return Vector{}, false, err
	}
	// Checked against the configured dimension, not against a ceiling. A log
	// written by a database of a different shape must not be silently applied to
	// this one, and the check happens before the slice is made.
	if int64(dim) != int64(o.dimension) {
		return Vector{}, false, fmt.Errorf("%w: record has dimension %d, database is %d",
			ErrCorrupt, dim, o.dimension)
	}
	raw, err := d.take(int(dim) * 4)
	if err != nil {
		return Vector{}, false, err
	}
	v.Values = make([]float32, dim)
	for i := range v.Values {
		f := math.Float32frombits(byteOrder.Uint32(raw[i*4:]))
		// A non-finite value poisons every comparison the index makes, and
		// nothing downstream would report it. It cannot get in through Add, so
		// finding one here means the bytes did not come from this database.
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return Vector{}, false, fmt.Errorf("%w: value at index %d is %v", ErrCorrupt, i, f)
		}
		v.Values[i] = f
	}

	md, n, err := store.Decode(d.src[d.off:])
	if err != nil {
		return Vector{}, false, fmt.Errorf("%w: %s", ErrCorrupt, err)
	}
	d.off += n
	if d.off != len(d.src) {
		// Trailing bytes mean the payload is not what it claims. Ignoring them
		// would let a record smuggle content past every length this decoder
		// checked.
		return Vector{}, false, fmt.Errorf("%w: %d trailing bytes on a PUT", ErrCorrupt, len(d.src)-d.off)
	}
	v.Metadata = md
	return v, true, nil
}

// payloadDecoder reads a bounded byte slice. Every read goes through take, which
// is what makes "never read past the end" a property of the decoder rather than
// a habit each call site has to remember.
type payloadDecoder struct {
	src []byte
	off int
}

func (d *payloadDecoder) take(n int) ([]byte, error) {
	if n < 0 || n > len(d.src)-d.off {
		return nil, fmt.Errorf("%w: wanted %d bytes, %d remain", ErrCorrupt, n, len(d.src)-d.off)
	}
	b := d.src[d.off : d.off+n]
	d.off += n
	return b, nil
}

func (d *payloadDecoder) u8() (uint8, error) {
	b, err := d.take(1)
	if err != nil {
		return 0, err
	}
	return b[0], nil
}

func (d *payloadDecoder) u16() (uint16, error) {
	b, err := d.take(2)
	if err != nil {
		return 0, err
	}
	return byteOrder.Uint16(b), nil
}

func (d *payloadDecoder) u32() (uint32, error) {
	b, err := d.take(4)
	if err != nil {
		return 0, err
	}
	return byteOrder.Uint32(b), nil
}

// The snapshot payload: an index and a metadata store, one after the other.
//
//	magic "GVDB" (4) | version (2) | reserved (2)
//	index section    (self-delimiting; hnsw's own format)
//	store section    (self-delimiting)
//
// # Why there is no length prefix between the sections
//
// A length prefix would have to be known before the section is written, which
// means buffering a multi-gigabyte index in memory purely to measure it. Both
// sections are self-delimiting instead — each reads exactly the bytes it wrote —
// so they can simply follow one another.
//
// That only works if they share one buffered reader. A section that wrapped the
// stream in its own bufio.Reader would read ahead into the next section's bytes
// and lose them. bufio.NewReaderSize returns the reader it was given when that
// reader is already buffered at least as large, which is documented behaviour
// and is what makes the sharing work — TestSnapshotPayloadSectionsDoNotOverread
// is the guard, because getting this wrong would corrupt on restore rather than
// fail to compile.

const (
	payloadMagicSize    = 4
	payloadVersionSize  = 2
	payloadReservedSize = 2
	payloadHeaderSize   = payloadMagicSize + payloadVersionSize + payloadReservedSize

	payloadVersion uint16 = 1

	// payloadBufSize must be at least the buffer hnsw.Read asks for, or it would
	// wrap this reader instead of reusing it and the sharing above would break.
	payloadBufSize = 64 << 10
)

var payloadMagic = [payloadMagicSize]byte{'G', 'V', 'D', 'B'}

// writeSnapshot streams the whole database state into w.
func writeSnapshot(w io.Writer, idx indexSerializer, st *store.Map) error {
	bw := bufio.NewWriterSize(w, payloadBufSize)

	var hdr [payloadHeaderSize]byte
	copy(hdr[:payloadMagicSize], payloadMagic[:])
	byteOrder.PutUint16(hdr[payloadMagicSize:], payloadVersion)
	if _, err := bw.Write(hdr[:]); err != nil {
		return fmt.Errorf("govecdb: write snapshot header: %w", err)
	}

	if _, err := idx.WriteTo(bw); err != nil {
		return fmt.Errorf("govecdb: write index: %w", err)
	}
	if err := st.WriteTo(bw); err != nil {
		return fmt.Errorf("govecdb: write metadata: %w", err)
	}
	return bw.Flush()
}

// readSnapshot reconstructs an index and a store from a snapshot payload.
func readSnapshot(r io.Reader, maxIDBytes int) (*hnswIndex, *store.Map, error) {
	br := bufio.NewReaderSize(r, payloadBufSize)

	var hdr [payloadHeaderSize]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil {
		return nil, nil, fmt.Errorf("%w: reading snapshot header: %w", ErrCorrupt, err)
	}
	if [payloadMagicSize]byte(hdr[:payloadMagicSize]) != payloadMagic {
		return nil, nil, fmt.Errorf("%w: not a govecdb snapshot", ErrCorrupt)
	}
	if v := byteOrder.Uint16(hdr[payloadMagicSize:]); v != payloadVersion {
		return nil, nil, fmt.Errorf("%w: snapshot version %d, this build reads %d",
			ErrCorrupt, v, payloadVersion)
	}

	idx, err := readIndex(br)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: reading index: %w", ErrCorrupt, err)
	}
	st := store.New()
	if err := st.ReadFrom(br, maxIDBytes); err != nil {
		return nil, nil, fmt.Errorf("%w: reading metadata: %w", ErrCorrupt, err)
	}
	return idx, st, nil
}
