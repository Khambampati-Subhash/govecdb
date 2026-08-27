//go:build race

package hnsw

// raceDetectorEnabled lets a test ask whether it is running under -race. Go has
// no runtime predicate for this, so it comes from the `race` build tag via this
// file and its !race twin.
const raceDetectorEnabled = true
