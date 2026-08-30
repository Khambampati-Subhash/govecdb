package snapshot

import (
	"encoding/binary"
	"hash/crc32"
)

// The wire layout. Sizes are spelled out as constants rather than derived with
// unsafe.Sizeof, because this is an on-disk format: it must not change because a
// Go type did.
const (
	// magic opens every snapshot. Four ASCII bytes so `head -c 4` on a file
	// answers "what is this", which is worth more than the four bytes cost.
	magicSize    = 4
	versionSize  = 2
	reservedSize = 2
	seqSize      = 8
	// headerSize is 16, which also keeps the payload 16-byte aligned from the
	// start of the file.
	headerSize = magicSize + versionSize + reservedSize + seqSize

	crcSize = 4
	// lengthSize holds the payload length as a uint64. A snapshot is the whole
	// state, so unlike a WAL record it has no business being capped at 4 GiB.
	lengthSize = 8
	// trailerSize is 12.
	trailerSize = crcSize + lengthSize

	// formatVersion is bumped when the layout above changes incompatibly.
	formatVersion uint16 = 1
)

// magicBytes spells GVSS — GoVecDB SnapShot.
var magicBytes = [magicSize]byte{'G', 'V', 'S', 'S'}

// crcTable uses Castagnoli rather than the IEEE default, matching the WAL: it is
// the polynomial with SSE4.2 / ARMv8 hardware support, which matters far more
// here than there. A snapshot is hashed end to end before it is applied, so this
// choice is the difference between verification being free and it being the
// slowest part of starting up.
var crcTable = crc32.MakeTable(crc32.Castagnoli)

// Everything on disk is little-endian, so a snapshot written on one machine
// reads on another.
var byteOrder = binary.LittleEndian

// encodeHeader writes the 16-byte header into dst.
func encodeHeader(dst []byte, seq uint64) {
	copy(dst[:magicSize], magicBytes[:])
	byteOrder.PutUint16(dst[magicSize:], formatVersion)
	// The reserved bytes stay zero; they are what make the header 16 wide.
	byteOrder.PutUint16(dst[magicSize+versionSize:], 0)
	byteOrder.PutUint64(dst[magicSize+versionSize+reservedSize:], seq)
}

// decodeHeader validates a header and returns the sequence it carries.
func decodeHeader(src []byte) (seq uint64, err error) {
	if len(src) < headerSize {
		return 0, ErrShortSnapshot
	}
	if [magicSize]byte(src[:magicSize]) != magicBytes {
		return 0, ErrBadMagic
	}
	if version := byteOrder.Uint16(src[magicSize:]); version != formatVersion {
		return 0, ErrUnsupportedVersion
	}
	return byteOrder.Uint64(src[magicSize+versionSize+reservedSize:]), nil
}

// encodeTrailer writes the 12-byte trailer into dst.
func encodeTrailer(dst []byte, sum uint32, length uint64) {
	byteOrder.PutUint32(dst[:crcSize], sum)
	byteOrder.PutUint64(dst[crcSize:], length)
}

// decodeTrailer reads the checksum and the payload length back.
//
// The length is not covered by the checksum, and does not need to be: it is
// cross-checked against the size of the file that contains it, which is a
// stronger statement than a hash of the field on its own. What the checksum
// covers is the header — the sequence number especially — and the payload.
func decodeTrailer(src []byte) (sum uint32, length uint64, err error) {
	if len(src) < trailerSize {
		return 0, 0, ErrShortSnapshot
	}
	return byteOrder.Uint32(src[:crcSize]), byteOrder.Uint64(src[crcSize:]), nil
}

// payloadLen reports how many payload bytes a file of this size must hold, and
// whether the size can hold a snapshot's framing at all.
func payloadLen(fileSize int64) (int64, bool) {
	if fileSize < headerSize+trailerSize {
		return 0, false
	}
	return fileSize - headerSize - trailerSize, true
}
