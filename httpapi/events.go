package httpapi

import (
	"bytes"
	"cmp"
	"context"
	"fmt"
	"log/slog"
	"slices"
	"sync"

	"github.com/khambampati-subhash/govecdb"
)

// Events turns the database's events into log lines and Prometheus counters.
//
// It exists apart from Server because of construction order: a Manager needs
// its observer when it is built, and a Server needs the Manager. So Events is
// built first, handed to both — as service.Options.Observer and as
// Config.Events — and neither has to be finished before the other starts.
//
// # The severity is decided here, not by the library
//
// What counts as a warning is the operator's policy, so it lives at the edge
// with the rest of the policy. Ordinary work — a recovery, a snapshot, a
// calibration — is Info; anything the database repaired or declined on the
// operator's behalf is Warn; going read-only is Error, because no write will
// succeed again until a restart.
type Events struct {
	log *slog.Logger

	mu     sync.Mutex
	counts map[eventKey]int64
}

type eventKey struct{ collection, event string }

// NewEvents builds an Events that logs to log. A nil log discards, and the
// counters still count.
func NewEvents(log *slog.Logger) *Events {
	if log == nil {
		log = slog.New(slog.DiscardHandler)
	}
	return &Events{log: log, counts: make(map[eventKey]int64)}
}

// Observe records one event. Its signature is service.Options.Observer's.
//
// The lock is held only for the increment. Logging happens after it, so a
// slow log sink stalls the one collection reporting, never every collection
// at once.
func (ev *Events) Observe(collection string, e govecdb.Event) {
	name, level := classifyEvent(e)

	ev.mu.Lock()
	ev.counts[eventKey{collection, name}]++
	ev.mu.Unlock()

	ev.log.Log(context.Background(), level, name, "collection", collection, "detail", e.String())
}

// classifyEvent names an event for a metric label and picks its log level.
//
// The names are a compatibility surface — dashboards and alerts are written
// against them — so they are spelled out rather than derived from Go type
// names, which a refactor would change.
func classifyEvent(e govecdb.Event) (string, slog.Level) {
	switch e.(type) {
	case govecdb.Recovered:
		return "recovered", slog.LevelInfo
	case govecdb.SnapshotTaken:
		return "snapshot_taken", slog.LevelInfo
	case govecdb.Calibrated:
		return "calibrated", slog.LevelInfo
	case govecdb.TornLog:
		return "torn_log", slog.LevelWarn
	case govecdb.SnapshotRejected:
		return "snapshot_rejected", slog.LevelWarn
	case govecdb.SnapshotFailed:
		return "snapshot_failed", slog.LevelWarn
	case govecdb.TruncationSkipped:
		return "truncation_skipped", slog.LevelWarn
	case govecdb.CalibrationFailed:
		return "calibration_failed", slog.LevelWarn
	case govecdb.DurabilityFailure:
		return "durability_failure", slog.LevelError
	default:
		// An event added to the library after this was written. Warn rather
		// than Info, because an unreviewed event is one nobody has decided is
		// harmless yet.
		return "other", slog.LevelWarn
	}
}

// writeMetrics appends govecdb_events_total in the exposition format, sorted so
// a scrape is stable and diffable.
//
// Collection names go into a label unescaped, for the reason handleMetrics
// gives: ValidateName admits nothing a label parser treats specially.
func (ev *Events) writeMetrics(b *bytes.Buffer) {
	ev.mu.Lock()
	keys := make([]eventKey, 0, len(ev.counts))
	for k := range ev.counts {
		keys = append(keys, k)
	}
	values := make([]int64, len(keys))
	slices.SortFunc(keys, func(a, b eventKey) int {
		return cmp.Or(cmp.Compare(a.collection, b.collection), cmp.Compare(a.event, b.event))
	})
	for i, k := range keys {
		values[i] = ev.counts[k]
	}
	ev.mu.Unlock()

	metric(b, "govecdb_events_total", "counter",
		"Database events by collection and kind: recoveries, snapshots, repairs, failures.")
	for i, k := range keys {
		fmt.Fprintf(b, "govecdb_events_total{collection=%q,event=%q} %d\n", k.collection, k.event, values[i])
	}
}
