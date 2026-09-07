package httpapi

import (
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/khambampati-subhash/govecdb/service"
)

// DefaultMaxBodyBytes is the request body cap when Config leaves it zero.
//
// 32 MiB is chosen from the arithmetic rather than from taste: a batch of 10,000
// vectors at 768 dimensions is about 90 MB of JSON, so this deliberately does
// *not* admit the largest batch the database accepts. A caller who wants that
// raises the limit knowingly, which is the right way round — the default should
// not let one request decide how much memory this process spends.
const DefaultMaxBodyBytes int64 = 32 << 20

// Config builds a Server.
type Config struct {
	// Manager is the collections this server exposes. Required.
	Manager *service.Manager

	// Logger receives one line per request plus anything unexpected. Nil
	// discards, which is what tests want and what a caller embedding this handler
	// in their own server may prefer.
	Logger *slog.Logger

	// MaxBodyBytes caps a request body. Zero means DefaultMaxBodyBytes.
	MaxBodyBytes int64

	// AuthToken, when set, is required as "Authorization: Bearer <token>" on
	// every route except /healthz and /readyz.
	//
	// One shared token rather than users and roles. That is the honest shape of
	// what this is — a database process on a private network — and inventing an
	// identity model would imply an authorization story this does not have.
	// Anything richer belongs in a proxy in front, which is also where TLS
	// termination, rate limiting and audit belong.
	AuthToken string

	// Version is reported by /metrics and in the Server header. Cosmetic.
	Version string
}

// Server is an http.Handler over a collection manager.
type Server struct {
	mgr     *service.Manager
	log     *slog.Logger
	maxBody int64
	token   []byte
	version string

	handler http.Handler
	started time.Time
	metrics metrics

	// ready backs /readyz. It starts true and a daemon clears it at the top of a
	// graceful shutdown, so a load balancer stops sending new work while the
	// requests already in flight finish. Liveness is a different question and
	// /healthz keeps answering it.
	ready atomic.Bool
}

// New builds a Server.
func New(cfg Config) (*Server, error) {
	if cfg.Manager == nil {
		return nil, errors.New("httpapi: Config.Manager is required")
	}
	s := &Server{
		mgr:     cfg.Manager,
		log:     cfg.Logger,
		maxBody: cfg.MaxBodyBytes,
		token:   []byte(cfg.AuthToken),
		version: cfg.Version,
		started: time.Now(),
	}
	if s.log == nil {
		s.log = slog.New(slog.DiscardHandler)
	}
	if s.maxBody <= 0 {
		s.maxBody = DefaultMaxBodyBytes
	}
	s.ready.Store(true)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /readyz", s.handleReady)
	mux.HandleFunc("GET /metrics", s.handleMetrics)

	mux.HandleFunc("GET /v1/collections", s.handleListCollections)
	mux.HandleFunc("POST /v1/collections", s.handleCreateCollection)
	mux.HandleFunc("GET /v1/collections/{name}", s.handleGetCollection)
	mux.HandleFunc("DELETE /v1/collections/{name}", s.handleDropCollection)

	mux.HandleFunc("POST /v1/collections/{name}/vectors", s.handleAddVectors)
	mux.HandleFunc("GET /v1/collections/{name}/vectors/{id}", s.handleGetVector)
	mux.HandleFunc("DELETE /v1/collections/{name}/vectors/{id}", s.handleDeleteVector)
	mux.HandleFunc("POST /v1/collections/{name}/search", s.handleSearch)

	mux.HandleFunc("POST /v1/collections/{name}/snapshot", s.handleSnapshot)
	mux.HandleFunc("POST /v1/collections/{name}/compact", s.handleCompact)

	// Outermost first: a panic in the auth check should still become a 500 rather
	// than a dropped connection, and a request refused by auth should still be
	// counted and logged. The shaper is innermost because the only responses it
	// has to rewrite are the ones the mux itself produces.
	s.handler = s.recoverer(s.observer(s.authenticator(s.shaper(mux))))
	return s, nil
}

// shaper rewrites the two responses net/http generates on its own.
//
// ServeMux answers an unrouted path with a plain-text 404 and a known path with
// the wrong method with a plain-text 405. Both are correct and both break the
// promise this package makes everywhere else — that a failure is a JSON object
// with a code — which would leave a client parsing two formats and discovering
// the second one in production.
//
// It recognises "the mux wrote this" by the content type, because every response
// this package produces sets application/json before its status. The Allow
// header the mux attaches to a 405 is preserved: it is the useful part.
func (s *Server) shaper(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(&shaper{ResponseWriter: w, server: s, req: r}, r)
	})
}

type shaper struct {
	http.ResponseWriter
	server *Server
	req    *http.Request
	taken  bool
}

func (sh *shaper) WriteHeader(status int) {
	ct := sh.Header().Get("Content-Type")
	if !strings.HasPrefix(ct, "application/json") {
		switch status {
		case http.StatusNotFound:
			sh.take(status, codeNotFound, "no such route: "+sh.req.Method+" "+sh.req.URL.Path)
			return
		case http.StatusMethodNotAllowed:
			sh.take(status, codeInvalidRequest, sh.req.Method+" is not allowed on "+sh.req.URL.Path)
			return
		}
	}
	sh.ResponseWriter.WriteHeader(status)
}

func (sh *shaper) take(status int, code, message string) {
	sh.taken = true
	body, err := json.Marshal(errorResponse{Error: errorDetail{Code: code, Message: message}})
	if err != nil {
		// Two constant strings and a path cannot fail to marshal, but leaving the
		// status unwritten if it somehow did would hang the connection.
		body = []byte(`{"error":{"code":"internal","message":"internal error"}}`)
	}
	sh.Header().Set("Content-Type", "application/json; charset=utf-8")
	sh.Header().Set("X-Content-Type-Options", "nosniff")
	sh.ResponseWriter.WriteHeader(status)
	sh.ResponseWriter.Write(append(body, '\n'))
}

// Write swallows the plain-text body the mux was about to send once take has
// replaced it. Reporting the full length keeps net/http from treating the
// difference as a short write.
func (sh *shaper) Write(b []byte) (int, error) {
	if sh.taken {
		return len(b), nil
	}
	return sh.ResponseWriter.Write(b)
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.handler.ServeHTTP(w, r)
}

// SetReady controls what /readyz answers. See Server.ready.
func (s *Server) SetReady(ready bool) { s.ready.Store(ready) }

// recoverer turns a panic into a 500 instead of a dropped connection.
//
// A handler that panics has already failed; what this stops is the failure
// taking the client's connection with it and leaving no record anywhere. The
// stack goes to the log and never to the response — it names internal paths and
// helps nobody on the other end.
func (s *Server) recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			v := recover()
			if v == nil {
				return
			}
			// The one panic that must propagate: net/http uses it to abandon a
			// connection on purpose, and recovering it breaks that.
			if v == http.ErrAbortHandler {
				panic(v)
			}
			s.log.Error("panic serving request",
				"method", r.Method, "path", r.URL.Path, "panic", fmt.Sprint(v))
			s.write(w, r, http.StatusInternalServerError, errorResponse{
				Error: errorDetail{Code: codeInternal, Message: "internal error"},
			})
		}()
		next.ServeHTTP(w, r)
	})
}

// observer logs and counts every request.
func (s *Server) observer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &recorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)

		took := time.Since(start)
		s.metrics.observe(rec.status, took)
		s.log.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rec.status,
			"bytes", rec.written,
			"duration", took,
		)
	})
}

// authenticator enforces the bearer token, if there is one.
//
// The comparison is constant-time. The timing signal from a byte-by-byte string
// compare is small and awkward to exploit over a network, and it is also free to
// remove, which settles it.
func (s *Server) authenticator(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(s.token) == 0 || isUnauthenticatedPath(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		token, ok := bearer(r)
		if !ok || subtle.ConstantTimeCompare([]byte(token), s.token) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="govecdb"`)
			s.fail(w, r, errUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// isUnauthenticatedPath reports the two probes that must answer without
// credentials.
//
// Health checks come from a load balancer that has no token and should not be
// given one: a probe that can fail for an authentication reason reports the
// wrong thing about the process. /metrics is deliberately *not* on this list —
// it names every collection and its size.
func isUnauthenticatedPath(p string) bool {
	return p == "/healthz" || p == "/readyz"
}

func bearer(r *http.Request) (string, bool) {
	h := r.Header.Get("Authorization")
	const prefix = "Bearer "
	if len(h) <= len(prefix) || !strings.EqualFold(h[:len(prefix)], prefix) {
		return "", false
	}
	return h[len(prefix):], true
}

// recorder captures the status and size for the log and the counters.
type recorder struct {
	http.ResponseWriter
	status  int
	written int64
}

func (r *recorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

func (r *recorder) Write(b []byte) (int, error) {
	n, err := r.ResponseWriter.Write(b)
	r.written += int64(n)
	return n, err
}

// write sends a JSON response.
//
// The body is marshalled before the status is written, so an encoding failure
// becomes a 500 rather than a 200 with half a document in it. That costs one
// buffer per response and buys the invariant that a status line is never a lie.
func (s *Server) write(w http.ResponseWriter, r *http.Request, status int, body any) {
	b, err := json.Marshal(body)
	if err != nil {
		s.log.Error("encoding response", "path", r.URL.Path, "error", err)
		http.Error(w, `{"error":{"code":"internal","message":"internal error"}}`,
			http.StatusInternalServerError)
		return
	}
	b = append(b, '\n')

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	// Nothing here is cacheable and a stale search result is worse than a slow
	// one, so intermediaries are told so rather than left to guess.
	w.Header().Set("Cache-Control", "no-store")
	// A JSON body that reaches a browser must not be sniffed into something the
	// browser will execute. Free, and the class of bug it closes is not.
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	if r.Method != http.MethodHead {
		w.Write(b)
	}
}

// fail maps an error onto a response and logs the ones the client is not told
// about in full.
func (s *Server) fail(w http.ResponseWriter, r *http.Request, err error) {
	status, code, message := classify(err)
	if status >= 500 {
		// The client gets "internal error"; the operator gets the cause, next to
		// the request that produced it.
		s.log.Error("request failed", "method", r.Method, "path", r.URL.Path, "error", err)
	}
	if status == http.StatusServiceUnavailable && code == codeTooManyOpen {
		w.Header().Set("Retry-After", "1")
	}
	s.write(w, r, status, errorResponse{Error: errorDetail{Code: code, Message: message}})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.write(w, r, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if !s.ready.Load() {
		s.write(w, r, http.StatusServiceUnavailable, map[string]string{"status": "shutting down"})
		return
	}
	s.write(w, r, http.StatusOK, map[string]string{"status": "ready"})
}
