package hnsw

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

// TestCalibrateNarrowsAnEasyCorpus is the case calibration exists for: data the
// formula over-searches. The suggestion must fall well below the prior *and*
// still hold the target on queries the calibration never saw.
func TestCalibrateNarrowsAnEasyCorpus(t *testing.T) {
	skipUnderRace(t)
	const dim, k, target = 64, 10, 0.95

	rng := rand.New(rand.NewSource(91))
	c, queries := separatedClusters(rng, 8000, dim, 512)
	g, _ := buildIndex(t, DefaultConfig(dim, Cosine), c)

	prior := g.SuggestedEf(k, target)
	cal, err := g.Calibrate(CalibrateOptions{Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	ef := g.SuggestedEf(k, target)
	recall, _ := evaluate(t, g, c, Cosine, queries(100), k, ef)
	t.Logf("calibration %+v: ef %d -> %d, held-out recall %.3f", cal, prior, ef, recall)

	if !cal.Reached || cal.Scale >= 1 {
		t.Fatalf("calibration did not narrow an easy corpus: %+v", cal)
	}
	if ef*2 > prior {
		t.Fatalf("calibrated ef %d is not well below the prior %d", ef, prior)
	}
	if recall < target-0.02 {
		t.Fatalf("calibrated ef %d reached only %.3f on held-out queries", ef, recall)
	}
}

// TestCalibrateHoldsTheTargetOnHardData: uniform random data is the case the
// formula was fitted to and the hard one for a graph. Whatever calibration
// does to the width there, it must not trade away the target.
func TestCalibrateHoldsTheTargetOnHardData(t *testing.T) {
	skipUnderRace(t)
	const dim, k, target = 128, 10, 0.95

	rng := rand.New(rand.NewSource(92))
	c := uniformCorpus(rng, 5000, dim)
	held := uniformCorpus(rand.New(rand.NewSource(93)), 100, dim)
	g, _ := buildIndex(t, DefaultConfig(dim, Cosine), c)

	cal, err := g.Calibrate(CalibrateOptions{Seed: 2})
	if err != nil {
		t.Fatal(err)
	}
	queries := make([][]float32, 0, len(held.order))
	for _, id := range held.order {
		queries = append(queries, held.vecs[id])
	}
	ef := g.SuggestedEf(k, target)
	recall, _ := evaluate(t, g, c, Cosine, queries, k, ef)
	t.Logf("calibration %+v: ef %d, held-out recall %.3f", cal, ef, recall)

	if recall < target-0.02 {
		t.Fatalf("calibrated ef %d reached only %.3f on held-out queries", ef, recall)
	}
}

func TestCalibrateSkipsASmallGraph(t *testing.T) {
	g, _ := buildGraph(t, MinCalibrationSize-1, 8, 3)
	g.efScale.Store(0x4000000000000000) // a stale 2.0 from when it was larger
	cal, err := g.Calibrate(CalibrateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if cal.Live != 0 || g.EfScale() != 1 {
		t.Fatalf("small graph: %+v, scale %v; want no measurement and the scale reset", cal, g.EfScale())
	}
}

func TestCalibrateStops(t *testing.T) {
	g, _ := buildGraph(t, 2000, 8, 4)
	stop := make(chan struct{})
	close(stop)
	if _, err := g.Calibrate(CalibrateOptions{Stop: stop}); !errors.Is(err, ErrCalibrationStopped) {
		t.Fatalf("Calibrate with a closed Stop = %v", err)
	}
	if g.EfScale() != 1 {
		t.Fatalf("a stopped calibration changed the scale to %v", g.EfScale())
	}
}

// TestConcurrentCalibrateAndInsert is for the race detector: the brute force
// runs outside the lock over references copied under it, while writers go on.
func TestConcurrentCalibrateAndInsert(t *testing.T) {
	g, _ := buildGraph(t, 1500, 8, 5)
	rng := rand.New(rand.NewSource(6))
	extra := make([][]float32, 300)
	for i := range extra {
		extra[i] = randomVector(rng, 8)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i, v := range extra {
			if err := g.Insert(fmt.Sprintf("x%d", i), v); err != nil {
				t.Error(err)
			}
			if i%3 == 0 {
				g.Delete(fmt.Sprintf("v%d", i))
			}
		}
	}()
	if _, err := g.Calibrate(CalibrateOptions{Queries: 16}); err != nil {
		t.Fatal(err)
	}
	_ = g.SuggestedEf(10, 0.95)
	wg.Wait()
}
