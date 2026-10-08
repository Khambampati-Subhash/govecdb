package httpapi

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// deadlineWriter is a ResponseWriter that, like a real connection's, accepts a
// write deadline, and records the last one set.
type deadlineWriter struct {
	*httptest.ResponseRecorder
	set      bool
	deadline time.Time
}

func (d *deadlineWriter) SetWriteDeadline(t time.Time) error {
	d.set, d.deadline = true, t
	return nil
}

// The middleware wrappers must not hide the connection from
// http.ResponseController — that is what let a Compact outlive WriteTimeout
// only as a reset connection.
func TestWrappersReachTheConnection(t *testing.T) {
	inner := httptest.NewRecorder()
	var w http.ResponseWriter = &shaper{ResponseWriter: &recorder{ResponseWriter: inner}}
	if err := http.NewResponseController(w).Flush(); err != nil {
		t.Fatalf("Flush through the wrappers: %v", err)
	}
	if !inner.Flushed {
		t.Fatal("Flush did not reach the underlying writer")
	}
}

// Snapshot and compact lift their write deadline; an ordinary route does not.
func TestLongAdminCallsLiftTheirWriteDeadline(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	for _, tc := range []struct {
		method, path string
		lifts        bool
	}{
		{"POST", "/v1/collections/docs/compact", true},
		{"POST", "/v1/collections/docs/snapshot", true},
		{"GET", "/v1/collections/docs", false},
	} {
		w := &deadlineWriter{ResponseRecorder: httptest.NewRecorder()}
		a.server.ServeHTTP(w, httptest.NewRequest(tc.method, tc.path, nil))
		if w.Code != http.StatusOK {
			t.Fatalf("%s %s = %d: %s", tc.method, tc.path, w.Code, w.Body)
		}
		if w.set != tc.lifts || (w.set && !w.deadline.IsZero()) {
			t.Errorf("%s %s: deadline set=%v to %v, want lifted=%v", tc.method, tc.path, w.set, w.deadline, tc.lifts)
		}
	}
}
