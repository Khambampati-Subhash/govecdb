//go:build race

package govecdb

// raceDetectorEnabled lets a timing test relax what -race distorts. Go has no
// runtime predicate for it, so it comes from the build tag via this file and
// its !race twin.
const raceDetectorEnabled = true
