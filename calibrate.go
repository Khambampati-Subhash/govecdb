package govecdb

import (
	"errors"
	"fmt"
	"time"

	"github.com/khambampati-subhash/govecdb/internal/hnsw"
)

// Search-width calibration, driven from here because only the database knows
// when the data has changed enough to be worth measuring again. The measuring
// itself is the index's business — see internal/hnsw/calibrate.go for what is
// measured and why a formula alone is not enough.
//
// # When
//
// Whenever the live count has doubled or halved since the last calibration,
// and the index is at least hnsw.MinCalibrationSize. Doubling, because the
// formula already carries how ef grows with n; what a calibration corrects is
// the data's difficulty, which does not change every insert. So a database
// that grows from nothing to a million is measured about ten times in its life.
//
// # Where
//
// In one background goroutine per database. Writers only kick it, through a
// channel that holds at most one pending kick, so a write never waits on a
// calibration and a burst of writes costs one. Close stops it, and a
// calibration in flight gives up between sample queries rather than holding a
// shutdown for the length of a brute-force scan.

// Calibrate measures this database's search width now, regardless of how much
// the data has changed, and applies it to every later search that leaves Ef
// zero. It blocks until done: under a second at a million vectors on a laptop,
// most of it an exact scan spread across GOMAXPROCS goroutines.
//
// Below hnsw.MinCalibrationSize live vectors it does nothing and resets the
// scale to 1 — the formula's widths are tiny there either way.
func (db *DB) Calibrate() error {
	idx, err := db.indexHandle()
	if err != nil {
		return err
	}
	cal, ok := idx.(indexCalibrator)
	if !ok {
		return fmt.Errorf("govecdb: index of type %T cannot be calibrated", idx)
	}
	if err := db.calibrate(idx, cal, nil); err != nil {
		return fmt.Errorf("govecdb: calibrate: %w", err)
	}
	return nil
}

// kickCalibration asks the background calibrator to run if the live count has
// moved far enough from the last calibration. Callers may hold the write lock:
// this never blocks.
func (db *DB) kickCalibration(live int) {
	if db.calKick == nil || live < hnsw.MinCalibrationSize {
		return
	}
	last := db.calAt.Load()
	if last != 0 && int64(live) < 2*last && int64(live) > last/2 {
		return
	}
	select {
	case db.calKick <- struct{}{}:
	default: // one is already pending, and it will see the latest count
	}
}

// calibrationLoop is the background calibrator. See the note at the top of
// this file.
func (db *DB) calibrationLoop() {
	defer close(db.doneCal)
	for {
		select {
		case <-db.stopCal:
			return
		case <-db.calKick:
		}
		idx, err := db.indexHandle()
		if err != nil {
			return
		}
		cal, ok := idx.(indexCalibrator)
		if !ok {
			return
		}
		// Errors go no further than an event: a calibration that fails leaves
		// the previous scale in place, which is a search width, not a
		// correctness problem.
		if err := db.calibrate(idx, cal, db.stopCal); errors.Is(err, hnsw.ErrCalibrationStopped) {
			return
		}
	}
}

// calibrate runs one calibration and reports it. Being stopped by Close is not
// a failure and is not reported.
func (db *DB) calibrate(idx Index, cal indexCalibrator, stop <-chan struct{}) error {
	// Recorded before measuring, so the writes that land during a calibration
	// are compared against the count it measured.
	live := idx.Len()
	db.calAt.Store(int64(live))

	start := time.Now()
	err := cal.Calibrate(db.opts.targetRecall, stop)
	switch {
	case errors.Is(err, hnsw.ErrCalibrationStopped):
	case err != nil:
		db.emit(CalibrationFailed{Cause: err})
	default:
		db.emit(Calibrated{Live: live, Scale: cal.EfScale(), Took: time.Since(start)})
	}
	return err
}

func efScaleOf(idx Index) float64 {
	if cal, ok := idx.(indexCalibrator); ok {
		return cal.EfScale()
	}
	return 1
}
