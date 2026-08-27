//go:build !race

package hnsw

// raceDetectorEnabled is the default build's answer; see the //go:build race twin.
const raceDetectorEnabled = false
