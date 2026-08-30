package snapshot

import (
	"bytes"
	"fmt"
	"io"
	"testing"
)

// payloadSizes span the range a real snapshot covers: a small index, and one
// with enough vectors that the I/O rather than the framing is what is measured.
var payloadSizes = []int{1 << 20, 64 << 20}

func sizeName(n int) string { return fmt.Sprintf("%dMiB", n>>20) }

func BenchmarkCreate(b *testing.B) {
	for _, size := range payloadSizes {
		b.Run(sizeName(size), func(b *testing.B) {
			payload := bytes.Repeat([]byte("v"), size)
			dir := b.TempDir()

			b.SetBytes(int64(size))
			b.ReportAllocs()
			b.ResetTimer()

			for i := range b.N {
				if _, err := Create(dir, uint64(i), func(w io.Writer) error {
					_, err := w.Write(payload)
					return err
				}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkLoad is the number that justifies verifying before applying.
//
// Load reads the file twice: once to check the checksum, once to hand the bytes
// to the callback. The cost of that decision is the gap between this and
// BenchmarkLoadVerifyOnly plus BenchmarkLoadApplyOnly measured separately — and
// it is small, because the verify pass leaves the file in the page cache, so the
// apply pass is usually memory rather than disk.
func BenchmarkLoad(b *testing.B) {
	for _, size := range payloadSizes {
		b.Run(sizeName(size), func(b *testing.B) {
			dir := benchDir(b, size)

			b.SetBytes(int64(size))
			b.ReportAllocs()
			b.ResetTimer()

			for range b.N {
				res, err := Load(dir, func(r io.Reader) error {
					_, err := io.Copy(io.Discard, r)
					return err
				})
				if err != nil {
					b.Fatal(err)
				}
				if !res.Found {
					b.Fatal("snapshot not found")
				}
			}
		})
	}
}

// BenchmarkLoadVerifyOnly isolates the checksum pass — the price of not trusting
// what is on the disk.
func BenchmarkLoadVerifyOnly(b *testing.B) {
	for _, size := range payloadSizes {
		b.Run(sizeName(size), func(b *testing.B) {
			dir := benchDir(b, size)
			snap, ok, err := Latest(dir)
			if err != nil || !ok {
				b.Fatal(err)
			}

			b.SetBytes(int64(size))
			b.ReportAllocs()
			b.ResetTimer()

			for range b.N {
				if _, err := verify(snap); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkLoadApplyOnly isolates the pass that actually feeds the caller, so
// the two halves of Load can be read against each other.
func BenchmarkLoadApplyOnly(b *testing.B) {
	for _, size := range payloadSizes {
		b.Run(sizeName(size), func(b *testing.B) {
			dir := benchDir(b, size)
			snap, ok, err := Latest(dir)
			if err != nil || !ok {
				b.Fatal(err)
			}

			b.SetBytes(int64(size))
			b.ReportAllocs()
			b.ResetTimer()

			for range b.N {
				if err := apply(snap, int64(size), func(r io.Reader) error {
					_, err := io.Copy(io.Discard, r)
					return err
				}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// benchDir writes one snapshot of the given payload size and returns its
// directory.
func benchDir(b *testing.B, size int) string {
	b.Helper()

	dir := b.TempDir()
	payload := bytes.Repeat([]byte("v"), size)
	if _, err := Create(dir, 1, func(w io.Writer) error {
		_, err := w.Write(payload)
		return err
	}); err != nil {
		b.Fatal(err)
	}
	return dir
}
