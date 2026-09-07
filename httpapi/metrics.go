package httpapi

import (
	"bytes"
	"fmt"
	"net/http"
	"runtime"
	"strconv"
	"sync/atomic"
	"time"
)

// metrics is the whole of this server's instrumentation.
//
// # Counters and gauges, no histogram
//
// A latency histogram is the thing you actually want and it is not worth what it
// costs here: bucket boundaries chosen without a dependency are boundaries
// chosen badly, and hand-rolling one produces a number that looks like a
// quantile and is not. So there is a request count and a total duration, whose
// ratio is a mean — a statistic that is honest about being a mean rather than a
// p99 wearing one's coat.
//
// The gap is deliberate and bounded: percentile latency belongs to the
// observability seam in the library (v2 item 1), where it can be measured at the
// operation rather than at the socket, and where a caller can plug in whatever
// their metrics stack already does well.
type metrics struct {
	// requests is indexed by status class: index 2 is 2xx, 5 is 5xx. Index 0
	// catches anything outside 1xx-5xx, which should be nothing.
	requests [6]atomic.Int64

	// nanos is the summed service time of every request. Paired with the counts
	// above it gives a mean; see the type comment for why that is all it gives.
	nanos atomic.Int64
}

func (m *metrics) observe(status int, d time.Duration) {
	class := status / 100
	if class < 1 || class > 5 {
		class = 0
	}
	m.requests[class].Add(1)
	m.nanos.Add(int64(d))
}

// handleMetrics writes the Prometheus text exposition format.
//
// Written by hand because the module has no third-party dependencies and this
// format is a dozen lines of strconv. It is also why collection names are
// interpolated without escaping: ValidateName allows only ASCII alphanumerics,
// '-' and '_', none of which mean anything to a Prometheus label parser. That is
// a real dependency between the two packages, and the reason it is written down
// in both.
//
// The endpoint requires the bearer token when one is configured. It names every
// collection and reports its size, which is more than a health probe should hand
// out and exactly what an unauthenticated scrape would publish.
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	infos, err := s.mgr.List()
	if err != nil {
		// A scrape that quietly returns process metrics and silently omits the
		// collections would hide the failure behind a graph that still moves.
		s.fail(w, r, err)
		return
	}

	var b bytes.Buffer

	metric(&b, "govecdb_build_info", "gauge", "Version of the running server.")
	fmt.Fprintf(&b, "govecdb_build_info{version=%q,go=%q} 1\n", s.versionLabel(), runtime.Version())

	metric(&b, "govecdb_uptime_seconds", "gauge", "Seconds since the server started.")
	fmt.Fprintf(&b, "govecdb_uptime_seconds %s\n", float(time.Since(s.started).Seconds()))

	metric(&b, "govecdb_http_requests_total", "counter", "Requests served, by status class.")
	for class, label := range map[int]string{1: "1xx", 2: "2xx", 3: "3xx", 4: "4xx", 5: "5xx", 0: "other"} {
		fmt.Fprintf(&b, "govecdb_http_requests_total{class=%q} %d\n", label, s.metrics.requests[class].Load())
	}

	metric(&b, "govecdb_http_request_duration_seconds_total", "counter",
		"Total time spent serving requests. Divide by the request count for a mean.")
	fmt.Fprintf(&b, "govecdb_http_request_duration_seconds_total %s\n",
		float(time.Duration(s.metrics.nanos.Load()).Seconds()))

	metric(&b, "govecdb_collections", "gauge", "Collections in the root directory.")
	fmt.Fprintf(&b, "govecdb_collections %d\n", len(infos))

	metric(&b, "govecdb_collections_loaded", "gauge", "Collections holding an index in memory.")
	fmt.Fprintf(&b, "govecdb_collections_loaded %d\n", s.mgr.Loaded())

	metric(&b, "govecdb_collection_loaded", "gauge", "1 when the collection's index is in memory.")
	for _, info := range infos {
		fmt.Fprintf(&b, "govecdb_collection_loaded{collection=%q} %d\n", info.Name, boolValue(info.Loaded))
	}

	// Only loaded collections report contents. A collection that is not loaded
	// has an unknown size, and publishing 0 for it would draw a graph of vectors
	// disappearing every time one is evicted.
	gauges := []struct {
		name, help string
		value      func(info infoLike) string
	}{
		{"govecdb_collection_live_vectors", "Vectors a search can return.",
			func(i infoLike) string { return strconv.Itoa(i.live) }},
		{"govecdb_collection_deleted_vectors", "Tombstoned slots awaiting compaction.",
			func(i infoLike) string { return strconv.Itoa(i.deleted) }},
		{"govecdb_collection_dead_ratio", "Fraction of slots that are tombstones; compaction pays around 0.5.",
			func(i infoLike) string { return float(i.deadRatio) }},
		{"govecdb_collection_wal_sequence", "Highest write-ahead log sequence assigned.",
			func(i infoLike) string { return strconv.FormatUint(i.lastSeq, 10) }},
		{"govecdb_collection_snapshot_sequence", "Log sequence covered by the newest snapshot.",
			func(i infoLike) string { return strconv.FormatUint(i.snapSeq, 10) }},
	}
	for _, g := range gauges {
		metric(&b, g.name, "gauge", g.help)
		for _, info := range infos {
			if !info.Loaded {
				continue
			}
			fmt.Fprintf(&b, "%s{collection=%q} %s\n", g.name, info.Name, g.value(infoLike{
				live:      info.Stats.Live,
				deleted:   info.Stats.Deleted,
				deadRatio: info.Stats.DeadRatio(),
				lastSeq:   info.Stats.LastSeq,
				snapSeq:   info.Stats.SnapshotSeq,
			}))
		}
	}

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	w.Write(b.Bytes())
}

// infoLike flattens the fields the gauge table reads, so the table stays a table
// rather than five nearly identical loops.
type infoLike struct {
	live, deleted    int
	deadRatio        float64
	lastSeq, snapSeq uint64
}

func metric(b *bytes.Buffer, name, kind, help string) {
	fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, kind)
}

// float formats without an exponent where possible and without trailing zeros,
// which is what Prometheus's parser and a human reading a curl both want.
func float(f float64) string { return strconv.FormatFloat(f, 'g', -1, 64) }

func boolValue(b bool) int {
	if b {
		return 1
	}
	return 0
}

func (s *Server) versionLabel() string {
	if s.version == "" {
		return "unknown"
	}
	return s.version
}
