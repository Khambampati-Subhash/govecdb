package govecdb

import (
	"fmt"
	"math/rand"
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
