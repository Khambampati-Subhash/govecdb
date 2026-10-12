package govecdb

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

// BenchmarkWritePath measures a 100-vector write through the whole stack — log,
// index, metadata — as one AddBatch and as 100 Adds. Under SyncAlways the gap
// between them is the fsyncs a batch amortizes; the "adds" case is also exactly
// what AddBatch cost before it logged the batch with one sync. ns/op is per 100
// vectors.
func BenchmarkWritePath(b *testing.B) {
	const batch, dim = 100, 128
	for _, policy := range []SyncPolicy{SyncAlways, SyncNever} {
		for _, mode := range []string{"batch", "adds"} {
			b.Run(fmt.Sprintf("%s/%s", policy, mode), func(b *testing.B) {
				db, err := Open(b.TempDir(), WithDimension(dim), WithSyncPolicy(policy))
				if err != nil {
					b.Fatal(err)
				}
				defer db.Close()

				rng := rand.New(rand.NewSource(1))
				vs := make([]Vector, batch)
				b.ReportAllocs()
				b.ResetTimer()
				for n := range b.N {
					b.StopTimer()
					for i := range vs {
						vs[i] = Vector{ID: fmt.Sprintf("v%d-%d", n, i), Values: vec(rng, dim)}
					}
					b.StartTimer()
					if mode == "batch" {
						if err := db.AddBatch(vs); err != nil {
							b.Fatal(err)
						}
						continue
					}
					for i := range vs {
						if err := db.Add(vs[i]); err != nil {
							b.Fatal(err)
						}
					}
				}
			})
		}
	}
}

// BenchmarkConcurrentAdds measures single Adds from 1 and 16 goroutines into a
// 10,000-vector database. With one writer it is BenchmarkWritePath's "adds";
// with sixteen it is what group commit buys — concurrent Adds used to
// serialize and ran no faster than one. ns/op is per vector; vec/s is the
// rate across all writers.
func BenchmarkConcurrentAdds(b *testing.B) {
	const dim, preload = 128, 10_000
	for _, policy := range []SyncPolicy{SyncAlways, SyncNever} {
		for _, writers := range []int{1, 16} {
			b.Run(fmt.Sprintf("%s/writers=%d", policy, writers), func(b *testing.B) {
				db, err := Open(b.TempDir(), WithDimension(dim), WithSyncPolicy(policy), WithEfCalibration(false))
				if err != nil {
					b.Fatal(err)
				}
				defer db.Close()
				rng := rand.New(rand.NewSource(1))
				for lo := 0; lo < preload; lo += 5000 {
					vs := make([]Vector, 5000)
					for i := range vs {
						vs[i] = Vector{ID: fmt.Sprintf("p%d", lo+i), Values: vec(rng, dim)}
					}
					if err := db.AddBatch(vs); err != nil {
						b.Fatal(err)
					}
				}
				vs := make([]Vector, b.N)
				for i := range vs {
					vs[i] = Vector{ID: fmt.Sprintf("w%d", i), Values: vec(rng, dim)}
				}

				b.ResetTimer()
				var wg sync.WaitGroup
				for w := range writers {
					wg.Add(1)
					go func() {
						defer wg.Done()
						for i := w; i < len(vs); i += writers {
							if err := db.Add(vs[i]); err != nil {
								b.Error(err)
								return
							}
						}
					}()
				}
				wg.Wait()
				b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "vec/s")
			})
		}
	}
}
