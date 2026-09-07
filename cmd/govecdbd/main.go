// Command govecdbd serves a directory of GoVecDB collections over HTTP.
//
//	govecdbd -dir /var/lib/govecdb -addr 127.0.0.1:8080
//
// The API and its semantics are documented in docs/SERVICE.md and in the
// httpapi package. This program is flags, a listener, and a shutdown sequence;
// everything it serves lives in httpapi and service, which is what makes both
// testable without a socket and this file short enough to read in one sitting.
//
// # Two defaults that are policy rather than taste
//
// It binds 127.0.0.1 unless told otherwise. A database that becomes reachable
// from the network because somebody omitted a flag is the wrong way for that to
// happen; making it public is a decision, so it takes an argument.
//
// The bearer token is read from GOVECDB_AUTH_TOKEN and there is deliberately no
// flag for it. A flag lands in `ps`, in shell history, and in whatever collects
// the container's command line — all places a credential outlives the process
// that used it.
package main

import (
	"context"
	"crypto/tls"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	"github.com/khambampati-subhash/govecdb/httpapi"
	"github.com/khambampati-subhash/govecdb/service"
)

// version is overridable at build time with
// -ldflags "-X main.version=1.2.3". Left empty it is taken from the module's
// build information, which is what `go install ...@v1.1.0` fills in.
var version = ""

func main() {
	// The signal context is the whole shutdown trigger. stop() is called as soon
	// as the first signal arrives so a second one kills the process outright:
	// an operator pressing Ctrl-C twice means it, and a graceful shutdown that
	// cannot be interrupted is a hang with better manners.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx, os.Args[1:], os.Stderr, stop, nil); err != nil {
		fmt.Fprintf(os.Stderr, "govecdbd: %v\n", err)
		os.Exit(1)
	}
}

type config struct {
	dir      string
	addr     string
	maxOpen  int
	idle     time.Duration
	maxBody  int64
	timeout  time.Duration
	drain    time.Duration
	shutdown time.Duration
	tlsCert  string
	tlsKey   string
	logLevel string
	logJSON  bool
	version  bool
}

func parseFlags(args []string, stderr io.Writer) (config, error) {
	var c config

	fs := flag.NewFlagSet("govecdbd", flag.ContinueOnError)
	fs.SetOutput(stderr)

	fs.StringVar(&c.dir, "dir", "", "directory holding the collections (required)")
	fs.StringVar(&c.addr, "addr", "127.0.0.1:8080", "address to listen on")
	fs.IntVar(&c.maxOpen, "max-open", 0, "maximum collections loaded at once (0 = no limit)")
	fs.DurationVar(&c.idle, "idle-timeout", 0, "close a collection nothing has used for this long (0 = never)")
	fs.Int64Var(&c.maxBody, "max-body", httpapi.DefaultMaxBodyBytes, "maximum request body in bytes")
	fs.DurationVar(&c.timeout, "timeout", 2*time.Minute, "per-request read and write timeout")
	fs.DurationVar(&c.drain, "drain", 0, "keep serving for this long after /readyz starts failing")
	fs.DurationVar(&c.shutdown, "shutdown-timeout", 30*time.Second, "how long to wait for requests in flight")
	fs.StringVar(&c.tlsCert, "tls-cert", "", "PEM certificate; enables TLS with -tls-key")
	fs.StringVar(&c.tlsKey, "tls-key", "", "PEM private key")
	fs.StringVar(&c.logLevel, "log-level", "info", "debug, info, warn or error")
	fs.BoolVar(&c.logJSON, "log-json", false, "emit structured JSON logs")
	fs.BoolVar(&c.version, "version", false, "print the version and exit")

	fs.Usage = func() {
		fmt.Fprintf(stderr, "govecdbd serves a directory of GoVecDB collections over HTTP.\n\n"+
			"Usage:\n  govecdbd -dir <directory> [flags]\n\nFlags:\n")
		fs.PrintDefaults()
		fmt.Fprintf(stderr, "\nThe bearer token is read from GOVECDB_AUTH_TOKEN. There is no flag for it:\n"+
			"a flag lands in ps output and shell history.\n")
	}

	if err := fs.Parse(args); err != nil {
		return c, err
	}
	if c.version {
		return c, nil
	}
	if c.dir == "" {
		fs.Usage()
		return c, errors.New("-dir is required")
	}
	if (c.tlsCert == "") != (c.tlsKey == "") {
		return c, errors.New("-tls-cert and -tls-key must be given together")
	}
	if c.maxBody <= 0 {
		return c, errors.New("-max-body must be positive")
	}
	return c, nil
}

// run is main with its dependencies passed in, so the whole program can be
// tested over a real socket: ctx replaces the signal, and ready reports the
// address actually bound — which matters because a test asks for port 0.
func run(ctx context.Context, args []string, stderr io.Writer, stop func(), ready func(net.Addr)) error {
	cfg, err := parseFlags(args, stderr)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if cfg.version {
		fmt.Fprintln(stderr, buildVersion())
		return nil
	}

	log, err := newLogger(stderr, cfg)
	if err != nil {
		return err
	}

	mgr, err := service.NewManager(cfg.dir, service.Options{
		MaxOpen:     cfg.maxOpen,
		IdleTimeout: cfg.idle,
	})
	if err != nil {
		return err
	}
	// Closed after the HTTP server has stopped, never before: a handler holding a
	// borrowed collection must finish against a live database rather than
	// discover a closed one.
	defer mgr.Close()

	token := os.Getenv("GOVECDB_AUTH_TOKEN")
	api, err := httpapi.New(httpapi.Config{
		Manager:      mgr,
		Logger:       log,
		MaxBodyBytes: cfg.maxBody,
		AuthToken:    token,
		Version:      buildVersion(),
	})
	if err != nil {
		return err
	}

	ln, err := net.Listen("tcp", cfg.addr)
	if err != nil {
		return fmt.Errorf("listen on %s: %w", cfg.addr, err)
	}

	srv := &http.Server{
		Handler: api,

		// A header that never finishes arriving is the cheapest way to hold a
		// connection open forever, and it is the one timeout with no legitimate
		// reason to be long.
		ReadHeaderTimeout: 10 * time.Second,

		// These two cover a whole request, so they bound the largest batch and
		// the longest search. Compaction on a large collection can exceed the
		// default — it rebuilds the index — which is why it is a flag.
		ReadTimeout:  cfg.timeout,
		WriteTimeout: cfg.timeout,

		IdleTimeout:    2 * time.Minute,
		MaxHeaderBytes: 64 << 10,
		ErrorLog:       slog.NewLogLogger(log.Handler(), slog.LevelWarn),
	}

	log.Info("listening",
		"addr", ln.Addr().String(),
		"dir", mgr.Root(),
		"tls", cfg.tlsCert != "",
		"auth", token != "",
		"version", buildVersion(),
	)
	if token == "" && !isLoopback(ln.Addr()) {
		// Not refused, because binding 0.0.0.0 inside a container is ordinary and
		// refusing it would make the common deployment the awkward one. Said
		// loudly, because an unauthenticated database on a routable address is
		// almost never what somebody meant.
		log.Warn("serving without authentication on a non-loopback address",
			"addr", ln.Addr().String(),
			"fix", "set GOVECDB_AUTH_TOKEN, or bind 127.0.0.1")
	}
	if ready != nil {
		ready(ln.Addr())
	}

	serveErr := make(chan error, 1)
	go func() {
		var err error
		if cfg.tlsCert != "" {
			srv.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12}
			err = srv.ServeTLS(ln, cfg.tlsCert, cfg.tlsKey)
		} else {
			err = srv.Serve(ln)
		}
		// The error Shutdown causes is not a failure; it is how Serve reports
		// that it was asked to stop.
		if errors.Is(err, http.ErrServerClosed) {
			err = nil
		}
		serveErr <- err
	}()

	select {
	case err := <-serveErr:
		return err
	case <-ctx.Done():
	}
	if stop != nil {
		// From here a second signal terminates the process rather than being
		// swallowed by a shutdown that is already under way.
		stop()
	}

	// Readiness fails first and the listener stays open for -drain, so a load
	// balancer notices and stops sending new work before connections are cut.
	// Liveness keeps answering throughout: the process is fine, it is leaving.
	api.SetReady(false)
	if cfg.drain > 0 {
		log.Info("draining", "for", cfg.drain)
		select {
		case <-time.After(cfg.drain):
		case <-serveErr:
		}
	}

	log.Info("shutting down", "grace", cfg.shutdown)
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.shutdown)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		// Requests still in flight when the grace period expired. Reported rather
		// than swallowed: it is the difference between a clean stop and one that
		// cut somebody off.
		log.Error("shutdown did not finish in time", "error", err)
		return err
	}
	return <-serveErr
}

func newLogger(w io.Writer, cfg config) (*slog.Logger, error) {
	var level slog.Level
	if err := level.UnmarshalText([]byte(cfg.logLevel)); err != nil {
		return nil, fmt.Errorf("-log-level %q: want debug, info, warn or error", cfg.logLevel)
	}
	opts := &slog.HandlerOptions{Level: level}
	if cfg.logJSON {
		return slog.New(slog.NewJSONHandler(w, opts)), nil
	}
	return slog.New(slog.NewTextHandler(w, opts)), nil
}

// isLoopback reports whether an address is only reachable from this machine.
//
// An unresolved or wildcard address is treated as *not* loopback, which is the
// direction that errs toward warning: a false warning costs a log line, and a
// missing one costs an open database.
func isLoopback(addr net.Addr) bool {
	host, _, err := net.SplitHostPort(addr.String())
	if err != nil {
		return false
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// buildVersion prefers the ldflags value, then the module version the toolchain
// recorded, so `go install ...@v1.1.0` reports something true without anyone
// remembering to pass a flag.
func buildVersion() string {
	if version != "" {
		return version
	}
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	if v := strings.TrimSpace(info.Main.Version); v != "" && v != "(devel)" {
		return v
	}
	// A build from a working tree: the revision is more useful than "(devel)".
	for _, s := range info.Settings {
		if s.Key == "vcs.revision" && len(s.Value) >= 12 {
			return s.Value[:12]
		}
	}
	return "devel"
}
