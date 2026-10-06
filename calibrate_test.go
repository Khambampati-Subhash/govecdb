package govecdb

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func batchOf(n int, seed int64) []Vector {
	rng := rand.New(rand.NewSource(seed))
	vs := make([]Vector, n)
	for i := range vs {
		vs[i] = Vector{ID: fmt.Sprintf("c%d", i), Values: vec(rng, testDim)}
	}
	return vs
}

// TestCalibrationRunsInTheBackground: crossing the minimum size is enough to
// get a measured width, with nobody calling anything.
func TestCalibrationRunsInTheBackground(t *testing.T) {
	db, _ := openDB(t)
	if s := db.Stats().EfScale; s != 1 {
		t.Fatalf("EfScale before any data = %v, want 1", s)
	}
	if err := db.AddBatch(batchOf(2000, 1)); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(20 * time.Second)
	for db.Stats().EfScale == 1 {
		if time.Now().After(deadline) {
			t.Fatal("no background calibration within 20s of crossing the minimum size")
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Logf("calibrated scale %.3f", db.Stats().EfScale)
}

func TestCalibrateByHand(t *testing.T) {
	db, _ := openDB(t, WithEfCalibration(false))
	if err := db.AddBatch(batchOf(2000, 2)); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)
	if s := db.Stats().EfScale; s != 1 {
		t.Fatalf("EfScale = %v with calibration off, want 1", s)
	}
	if err := db.Calibrate(); err != nil {
		t.Fatal(err)
	}
	if s := db.Stats().EfScale; s == 1 {
		t.Fatal("Calibrate did not change the scale")
	}
	// An explicit Ef is the caller's, calibrated or not.
	q := batchOf(1, 3)[0].Values
	if _, err := db.Search(SearchRequest{Query: q, K: 5, Ef: 7}); err != nil {
		t.Fatal(err)
	}
}

// TestCloseStopsACalibrationInFlight: Close must not wait out a brute-force
// scan, and must not leave the goroutine running (the race detector and the
// test's own exit would both notice a calibrator touching a closed database).
func TestCloseStopsACalibrationInFlight(t *testing.T) {
	db, _ := openDB(t)
	if err := db.AddBatch(batchOf(3000, 4)); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if took := time.Since(start); took > 5*time.Second {
		t.Fatalf("Close took %v with a calibration pending", took)
	}
	if err := db.Calibrate(); err == nil {
		t.Fatal("Calibrate on a closed database succeeded")
	}
}
