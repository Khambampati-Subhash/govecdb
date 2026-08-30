package wal

import (
	"encoding/binary"
	"hash/crc32"
)

// RecordType tags what happened, so replay can dispatch without decoding the
// payload. The payload's *encoding* stays the domain layer's business; the log
// only needs to know which kind of thing it is holding.
type RecordType uint8

const (
	// TypeInvalid is zero on purpose. Unwritten space, a hole in a file, and a
	// zero-filled block all read as zeros, and none of them should ever decode
	// as a valid record.
	TypeInvalid RecordType = 0

	// TypePut records a vector being stored under an id. It covers creation and
	// replacement alike, because Insert is an upsert — one state transition, one
	// record type, and a meaning that does not depend on what came before it.
	TypePut RecordType = 1

	// TypeDelete records an id being removed.
	TypeDelete RecordType = 2

	// TypeCheckpoint marks a point whose state is durable in a snapshot, so every
	// segment below it can be dropped. Reserved now, written by the checkpointer
	// later; the constant exists so the numbering is not rearranged afterwards.
	TypeCheckpoint RecordType = 3
)

func (t RecordType) valid() bool {
	return t == TypePut || t == TypeDelete || t == TypeCheckpoint
}

func (t RecordType) String() string {
	switch t {
	case TypePut:
		return "PUT"
	case TypeDelete:
		return "DELETE"
	case TypeCheckpoint:
		return "CHECKPOINT"
	default:
		return "INVALID"
	}
}

// Record is one logical operation as it comes back out of the log.
type Record struct {
	Type    RecordType
	Seq     uint64
	Payload []byte
}

// Clone returns a copy that owns its payload.
//
// Replay hands out payloads that point into a buffer it reuses, so they are only
// valid until the next record is read. A callback that keeps a record past its
// return needs this; one that decodes and discards does not, and should not pay
// for a copy it will not use.
func (r Record) Clone() Record {
	r.Payload = append([]byte(nil), r.Payload...)
	return r
}

// The wire layout. Sizes are spelled out as constants rather than derived with
// unsafe.Sizeof, because this is an on-disk format: it must not change because a
// Go type did.
const (
	// magic opens every segment file. Four ASCII bytes so `head -c 4` on a
	// segment answers "what is this", which matters more than the four bytes cost.
	magicSize      = 4
	versionSize    = 2
	reservedSize   = 2
	fileHeaderSize = magicSize + versionSize + reservedSize // 8

	crcSize    = 4
	typeSize   = 1
	seqSize    = 8
	lengthSize = 4
	// recordHeaderSize is everything before the payload: 17 bytes.
	recordHeaderSize = crcSize + typeSize + seqSize + lengthSize

	// formatVersion is bumped when the layout above changes incompatibly.
	formatVersion uint16 = 1
)

// magicBytes spells GVWL — GoVecDB Write-ahead Log.
var magicBytes = [magicSize]byte{'G', 'V', 'W', 'L'}

// crcTable uses Castagnoli rather than the IEEE default: it is the polynomial
// with SSE4.2 / ARMv8 hardware support, so on any machine this runs on the
// checksum costs a few cycles per cache line instead of a table walk.
var crcTable = crc32.MakeTable(crc32.Castagnoli)

// Everything on disk is little-endian. Fixing it here rather than using the host
// order means a log written on one machine reads on another.
var byteOrder = binary.LittleEndian

// encodeFileHeader writes the 8-byte segment header into dst.
func encodeFileHeader(dst []byte) {
	copy(dst[:magicSize], magicBytes[:])
	byteOrder.PutUint16(dst[magicSize:], formatVersion)
	// The reserved bytes stay zero. They exist so the header is 8 bytes wide,
	// which keeps every record that follows 8-byte aligned from the file start.
	byteOrder.PutUint16(dst[magicSize+versionSize:], 0)
}

// decodeFileHeader validates a segment header and returns its format version.
func decodeFileHeader(src []byte) (uint16, error) {
	if len(src) < fileHeaderSize {
		return 0, ErrShortRecord
	}
	if [magicSize]byte(src[:magicSize]) != magicBytes {
		return 0, ErrBadMagic
	}
	version := byteOrder.Uint16(src[magicSize:])
	if version != formatVersion {
		return 0, ErrUnsupportedVersion
	}
	return version, nil
}

// encodeRecordHeader fills a 17-byte header for a record and returns it, with
// the checksum computed over the header's own tail plus the payload.
//
// The checksum deliberately covers the length field. A flipped bit in a payload
// gives a wrong answer; a flipped bit in a length gives a read of arbitrary size
// at an arbitrary offset, which is the failure that actually hurts.
func encodeRecordHeader(dst []byte, typ RecordType, seq uint64, payload []byte) {
	dst[crcSize] = byte(typ)
	byteOrder.PutUint64(dst[crcSize+typeSize:], seq)
	byteOrder.PutUint32(dst[crcSize+typeSize+seqSize:], uint32(len(payload)))

	sum := crc32.Checksum(dst[crcSize:recordHeaderSize], crcTable)
	sum = crc32.Update(sum, crcTable, payload)
	byteOrder.PutUint32(dst[:crcSize], sum)
}

// decodeRecordHeader reads the fixed part of a record. It does not verify the
// checksum, because the payload has not been read yet — the length it returns is
// what tells the caller how much more to read. Nothing may be allocated on the
// strength of that length until verifyChecksum has passed.
func decodeRecordHeader(src []byte) (sum uint32, typ RecordType, seq uint64, length uint32, err error) {
	if len(src) < recordHeaderSize {
		return 0, 0, 0, 0, ErrShortRecord
	}
	sum = byteOrder.Uint32(src[:crcSize])
	typ = RecordType(src[crcSize])
	seq = byteOrder.Uint64(src[crcSize+typeSize:])
	length = byteOrder.Uint32(src[crcSize+typeSize+seqSize:])
	return sum, typ, seq, length, nil
}

// verifyChecksum recomputes a record's checksum from its header and payload.
//
// The header slice must be the record's own 17 bytes: the stored checksum is
// read from it and the rest is fed back through the hash, which is what makes a
// corrupt type or sequence number as detectable as a corrupt payload.
func verifyChecksum(header, payload []byte) bool {
	want := byteOrder.Uint32(header[:crcSize])
	got := crc32.Checksum(header[crcSize:recordHeaderSize], crcTable)
	got = crc32.Update(got, crcTable, payload)
	return want == got
}
