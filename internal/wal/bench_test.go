package wal

import (
	"bytes"
	"fmt"
	"sync"
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

// BenchmarkAppendBatch is BenchmarkAppend for 100 records per call: under
// SyncAlways the difference between them is the fsync that batching amortizes.
// ns/op is per batch, so divide by 100 to compare with BenchmarkAppend.
func BenchmarkAppendBatch(b *testing.B) {
	payload := bytes.Repeat([]byte("v"), 2048+32)
	const batch = 100

	for _, tc := range []struct {
		name   string
		policy SyncPolicy
	}{
		{"always", SyncAlways},
		{"never", SyncNever},
	} {
		b.Run(tc.name, func(b *testing.B) {
			w, err := Open(b.TempDir(), Options{SyncPolicy: tc.policy, MaxSegmentBytes: 1 << 30})
			if err != nil {
				b.Fatal(err)
			}
			defer w.Close()

			b.SetBytes(batch * int64(len(payload)+recordHeaderSize))
			b.ReportAllocs()
			b.ResetTimer()

			for range b.N {
				if _, err := w.AppendBatch(TypePut, batch, func(int) []byte { return payload }); err != nil {
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

// BenchmarkReplay measures recovery, which is the number that decides how long
// a crashed process takes to come back. It is reported per record rather than
// per replay, since that is the unit the log grows in.
//
// The allocation count is the load-bearing part: the reader hands out payloads
// pointing into a buffer it reuses, so replaying a million records should not
// allocate a million times. A non-zero number here means that contract broke.
func BenchmarkReplay(b *testing.B) {
	const records = 10_000
	payload := bytes.Repeat([]byte("v"), 2048+32)

	dir := b.TempDir()
	w, err := Open(dir, Options{SyncPolicy: SyncNever, MaxSegmentBytes: 8 << 20})
	if err != nil {
		b.Fatal(err)
	}
	for range records {
		if _, err := w.Append(TypePut, payload); err != nil {
			b.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		b.Fatal(err)
	}

	var sink uint64
	b.SetBytes(int64(records * (len(payload) + recordHeaderSize)))
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		res, err := Replay(dir, Options{}, func(r Record) error {
			sink += r.Seq + uint64(len(r.Payload))
			return nil
		})
		if err != nil {
			b.Fatal(err)
		}
		if res.Records != records {
			b.Fatalf("replayed %d records, want %d", res.Records, records)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*records), "ns/record")
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

// BenchmarkAppendDuringFlush is SyncInterval with appenders on several
// goroutines and a short interval, so a flush is nearly always in progress.
// It measures what an append waits for while the flusher fsyncs: the flush
// holds the writer's lock only to move the buffer into the kernel, and the
// fsync runs outside it, so the max-µs metric is an append's worst wait.
func BenchmarkAppendDuringFlush(b *testing.B) {
	payload := bytes.Repeat([]byte("v"), 2048+32)
	w, err := Open(b.TempDir(), Options{
		SyncPolicy:      SyncInterval,
		SyncInterval:    20 * time.Millisecond,
		MaxSegmentBytes: 1 << 30,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()

	var mu sync.Mutex
	var worst time.Duration
	b.SetBytes(int64(len(payload) + recordHeaderSize))
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var mine time.Duration
		for pb.Next() {
			start := time.Now()
			if _, err := w.Append(TypePut, payload); err != nil {
				b.Error(err)
				return
			}
			mine = max(mine, time.Since(start))
		}
		mu.Lock()
		worst = max(worst, mine)
		mu.Unlock()
	})
	b.ReportMetric(float64(worst.Microseconds()), "max-µs")
}
