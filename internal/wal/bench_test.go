package wal

import (
	"bytes"
	"fmt"
	"testing"
	"time"
)

// BenchmarkAppend puts a number on the only knob this package exposes. The three
// policies are not variations on a theme — they are three different answers to
// "what does an acknowledged write mean", and the gap between them is what makes
// choosing one a decision rather than a preference.
func BenchmarkAppend(b *testing.B) {
	// 512 floats plus an id: roughly what a PUT for a real embedding costs.
	payload := bytes.Repeat([]byte("v"), 2048+32)

	for _, tc := range []struct {
		name   string
		policy SyncPolicy
	}{
		{"always", SyncAlways},
		{"interval", SyncInterval},
		{"never", SyncNever},
	} {
		b.Run(tc.name, func(b *testing.B) {
			w, err := Open(b.TempDir(), Options{
				SyncPolicy:   tc.policy,
				SyncInterval: 50 * time.Millisecond,
				// Large enough that rotation is not what is being measured.
				MaxSegmentBytes: 1 << 30,
			})
			if err != nil {
				b.Fatal(err)
			}
			defer w.Close()

			b.SetBytes(int64(len(payload) + recordHeaderSize))
			b.ReportAllocs()
			b.ResetTimer()

			for range b.N {
				if _, err := w.Append(TypePut, payload); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkAppendSmall isolates the per-record overhead — header, checksum,
// bookkeeping — from the cost of moving the payload.
func BenchmarkAppendSmall(b *testing.B) {
	w, err := Open(b.TempDir(), Options{SyncPolicy: SyncNever, MaxSegmentBytes: 1 << 30})
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()

	payload := []byte("doc-00000001")
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := w.Append(TypeDelete, payload); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkChecksum measures the hash alone, since it is the one cost every
// record pays and the reason the table is Castagnoli rather than the IEEE
// default: that polynomial has hardware support on every CPU this runs on.
func BenchmarkChecksum(b *testing.B) {
	for _, size := range []int{64, 1024, 16384} {
		b.Run(fmt.Sprintf("%dB", size), func(b *testing.B) {
			payload := bytes.Repeat([]byte("x"), size)
			var hdr [recordHeaderSize]byte

			b.SetBytes(int64(size))
			b.ResetTimer()
			for i := range b.N {
				encodeRecordHeader(hdr[:], TypePut, uint64(i), payload)
			}
		})
	}
}

// BenchmarkSegmentRotation measures what a rotation costs, because it lands
// inside a caller's Append: it fsyncs and closes one file and creates another,
// which is the most expensive thing an append can do and the reason
// MaxSegmentBytes should not be set small.
func BenchmarkSegmentRotation(b *testing.B) {
	payload := bytes.Repeat([]byte("x"), 256)

	w, err := Open(b.TempDir(), Options{
		SyncPolicy: SyncNever,
		// Small enough that nearly every append rotates.
		MaxSegmentBytes: 300,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := w.Append(TypePut, payload); err != nil {
			b.Fatal(err)
		}
	}
}
