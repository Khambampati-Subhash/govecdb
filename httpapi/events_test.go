package httpapi

import (
	"bytes"
	"errors"
	"log/slog"
	"net/http"
	"strings"
	"testing"

	"github.com/khambampati-subhash/govecdb"
	"github.com/khambampati-subhash/govecdb/service"
)

// TestEventsReachMetrics wires Events the way the daemon does — into the
// Manager as its observer and into the Server — and checks a database's own
// events come out of /metrics against the right collection.
func TestEventsReachMetrics(t *testing.T) {
	ev := NewEvents(nil)
	mgr, err := service.NewManager(t.TempDir(), service.Options{Observer: ev.Observe})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { mgr.Close() })

	a := newAPI(t, Config{Manager: mgr, Events: ev})
	a.createCollection("docs", 4)
	a.expect(a.do("POST", "/v1/collections/docs/vectors", map[string]any{
		"vectors": []any{map[string]any{"id": "a", "values": []float32{1, 0, 0, 0}}},
	}), http.StatusOK)
	a.expect(a.do("POST", "/v1/collections/docs/snapshot", nil), http.StatusOK)

	body := a.do("GET", "/metrics", nil).Body.String()
	for _, want := range []string{
		"# TYPE govecdb_events_total counter",
		`govecdb_events_total{collection="docs",event="recovered"} 1`,
		`govecdb_events_total{collection="docs",event="snapshot_taken"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics missing %q:\n%s", want, body)
		}
	}
}

// TestEventsWithoutObserverOmitTheCounter: a Server built without Events must
// not publish a counter family that would read as "nothing ever happened".
func TestEventsWithoutObserverOmitTheCounter(t *testing.T) {
	a := newAPI(t, Config{})
	if body := a.do("GET", "/metrics", nil).Body.String(); strings.Contains(body, "govecdb_events_total") {
		t.Fatalf("events counter published with no Events wired:\n%s", body)
	}
}

func TestEventSeverity(t *testing.T) {
	var buf bytes.Buffer
	ev := NewEvents(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))

	cause := errors.New("boom")
	for _, tc := range []struct {
		e     govecdb.Event
		name  string
		level string
	}{
		{govecdb.Recovered{}, "recovered", "INFO"},
		{govecdb.SnapshotTaken{}, "snapshot_taken", "INFO"},
		{govecdb.Calibrated{}, "calibrated", "INFO"},
		{govecdb.TornLog{Cause: cause}, "torn_log", "WARN"},
		{govecdb.SnapshotRejected{Cause: cause}, "snapshot_rejected", "WARN"},
		{govecdb.SnapshotFailed{Cause: cause}, "snapshot_failed", "WARN"},
		{govecdb.TruncationSkipped{Cause: cause}, "truncation_skipped", "WARN"},
		{govecdb.CalibrationFailed{Cause: cause}, "calibration_failed", "WARN"},
		{govecdb.DurabilityFailure{Cause: cause}, "durability_failure", "ERROR"},
	} {
		buf.Reset()
		ev.Observe("docs", tc.e)
		line := buf.String()
		if !strings.Contains(line, "level="+tc.level) || !strings.Contains(line, "msg="+tc.name) ||
			!strings.Contains(line, "collection=docs") {
			t.Errorf("%T logged %q, want level=%s msg=%s collection=docs", tc.e, line, tc.level, tc.name)
		}
	}
}
