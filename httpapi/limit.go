package httpapi

import (
	"errors"
	"net/http"
	"runtime"
	"strings"
	"sync/atomic"
)

// Backpressure.
//
// # Why there is any
//
// Nothing used to bound how many requests ran at once. Searches are CPU-bound
// and ingests are memory-bound — a maximum-size add body allocates about 3.7x
// its size while decoding, ~120 MB at the 32 MiB cap — so at ten times the load
// a server could take, every request slowed down together, clients timed out and
// retried, and the server kept finishing work nobody was waiting for. That is
// how congestion collapse starts, and the only cure is to refuse early.
//
// # Two pools, refusing rather than queueing
//
// Reads (searches, gets, scans, listings) and writes (adds, deletes, lifecycle,
// snapshot, sync, compact) get separate pools, so a burst of ingest cannot
// starve searches of slots and the reverse. A full pool answers 503
// "overloaded" with Retry-After at once. It does not queue, for the reason
// ErrTooManyOpen does not: a request waiting for a slot is a capacity problem
// turned into a timeout somewhere less visible, and the client — or the load
// balancer in front of it — has a better answer available when told now.
//
// The probes and /metrics are exempt. A health check that fails because the
// server is busy reports the wrong thing, and the scrape is how an operator
// sees that it is.

// errOverloaded is a full pool. It is an error rather than a direct write so it
// takes the one path every failure takes, and keeps the one shape.
var errOverloaded = errors.New("httpapi: too many requests in flight; retry shortly")

// errClientGone means the client disconnected before the work started. Nobody
// reads the response; it exists so the log and the status counters say what
// happened instead of recording a 200 for work that was skipped.
var errClientGone = errors.New("httpapi: the client went away before the request was served")

// DefaultMaxInFlightReads and DefaultMaxInFlightWrites size the pools when
// Config leaves them zero: four reads per CPU, because a search is CPU-bound
// and a little more than one per core keeps the cores busy across the parts of
// a request that are not; one write per CPU, at least two, because a write's
// cost is its decoded body as much as its CPU, and writes to one collection
// serialize on its lock regardless.
func DefaultMaxInFlightReads() int  { return 4 * runtime.GOMAXPROCS(0) }
func DefaultMaxInFlightWrites() int { return max(2, runtime.GOMAXPROCS(0)) }

// pool is a counting semaphore that never waits.
type pool struct {
	name     string
	slots    chan struct{} // nil: unlimited
	inflight atomic.Int64
	rejected atomic.Int64
}

func newPool(name string, size, def int) *pool {
	p := &pool{name: name}
	switch {
	case size < 0:
		// Unlimited, for an embedder whose own server already sheds load.
	case size == 0:
		p.slots = make(chan struct{}, def)
	default:
		p.slots = make(chan struct{}, size)
	}
	return p
}

func (p *pool) tryAcquire() bool {
	if p.slots != nil {
		select {
		case p.slots <- struct{}{}:
		default:
			p.rejected.Add(1)
			return false
		}
	}
	p.inflight.Add(1)
	return true
}

func (p *pool) release() {
	p.inflight.Add(-1)
	if p.slots != nil {
		<-p.slots
	}
}

// poolFor picks the pool a request draws from, or nil for the exempt paths.
// Reads are GET and HEAD, plus the two POSTs that only read: a search and a
// batch get carry a body because their arguments do not fit in a URL, not
// because they change anything.
func (s *Server) poolFor(r *http.Request) *pool {
	if isUnauthenticatedPath(r.URL.Path) || r.URL.Path == "/metrics" {
		return nil
	}
	switch r.Method {
	case http.MethodGet, http.MethodHead:
		return s.reads
	case http.MethodPost:
		if strings.HasSuffix(r.URL.Path, "/search") || strings.HasSuffix(r.URL.Path, "/vectors/get") {
			return s.reads
		}
	}
	return s.writes
}

// limiter enforces the pools. It sits inside the authenticator, so a flood of
// unauthenticated requests is refused without occupying a slot.
func (s *Server) limiter(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := s.poolFor(r)
		if p == nil {
			next.ServeHTTP(w, r)
			return
		}
		if !p.tryAcquire() {
			s.fail(w, r, errOverloaded)
			return
		}
		defer p.release()
		// The connection can be gone already — a client that gave up on a slow
		// server is exactly the one this layer exists for. The library takes no
		// context, so this and the check after decoding are the two places work
		// can still be declined.
		if r.Context().Err() != nil {
			s.fail(w, r, errClientGone)
			return
		}
		next.ServeHTTP(w, r)
	})
}
