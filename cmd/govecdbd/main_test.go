package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

// daemon runs the whole program over a real socket, which is the point: every
// other test in this module stops at an http.Handler, and the things that only
// break for real — binding, shutdown, the listener actually closing — live here.
type daemon struct {
	t    *testing.T
	base string
	stop func()
	done chan error
}

func start(t *testing.T, args ...string) *daemon {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())
	addr := make(chan net.Addr, 1)
	done := make(chan error, 1)

	var stderr syncBuffer
	go func() {
		done <- run(ctx, args, &stderr, nil, func(a net.Addr) { addr <- a })
	}()

	var bound net.Addr
	select {
	case bound = <-addr:
	case err := <-done:
		cancel()
		t.Fatalf("the daemon exited before it was listening: %v\n%s", err, stderr.String())
	case <-time.After(10 * time.Second):
		cancel()
		t.Fatal("the daemon never reported a listening address")
	}

	d := &daemon{
		t:    t,
		base: "http://" + bound.String(),
		done: done,
	}
	// Once, because a test that stops the daemon explicitly still has the
	// cleanup registered below — and the second wait would block on a channel
	// nothing will send to again.
	var once sync.Once
	d.stop = func() {
		once.Do(func() {
			cancel()
			select {
			case err := <-done:
				if err != nil {
					t.Errorf("run: %v\n%s", err, stderr.String())
				}
			case <-time.After(30 * time.Second):
				t.Error("the daemon did not shut down")
			}
		})
	}
	t.Cleanup(d.stop)
	return d
}

func (d *daemon) do(method, path, body, token string) (int, map[string]any) {
	d.t.Helper()

	var r io.Reader
	if body != "" {
		r = strings.NewReader(body)
	}
	req, err := http.NewRequest(method, d.base+path, r)
	if err != nil {
		d.t.Fatal(err)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		d.t.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		d.t.Fatal(err)
	}
	var out map[string]any
	if len(raw) > 0 && strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
		if err := json.Unmarshal(raw, &out); err != nil {
			d.t.Fatalf("decode %q: %v", raw, err)
		}
	}
	return resp.StatusCode, out
}

func (d *daemon) mustDo(method, path, body, token string, want int) map[string]any {
	d.t.Helper()
	status, out := d.do(method, path, body, token)
	if status != want {
		d.t.Fatalf("%s %s = %d, want %d: %v", method, path, status, want, out)
	}
	return out
}

// syncBuffer collects the daemon's log without racing the test goroutine that
// prints it on failure.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func TestEndToEnd(t *testing.T) {
	dir := t.TempDir()
	d := start(t, "-dir", dir, "-addr", "127.0.0.1:0")

	d.mustDo("GET", "/healthz", "", "", http.StatusOK)
	d.mustDo("GET", "/readyz", "", "", http.StatusOK)

	d.mustDo("POST", "/v1/collections",
		`{"name":"docs","dimension":4,"sync_policy":"never"}`, "", http.StatusCreated)

	d.mustDo("POST", "/v1/collections/docs/vectors", `{"vectors":[
		{"id":"a","values":[1,0,0,0],"metadata":{"page":1}},
		{"id":"b","values":[0,1,0,0],"metadata":{"page":2}},
		{"id":"c","values":[0,0,1,0],"metadata":{"page":3}}
	]}`, "", http.StatusOK)

	out := d.mustDo("POST", "/v1/collections/docs/search",
		`{"query":[1,0,0,0],"k":2}`, "", http.StatusOK)
	matches, _ := out["matches"].([]any)
	if len(matches) != 2 || matches[0].(map[string]any)["id"] != "a" {
		t.Fatalf("search = %v", out)
	}

	filtered := d.mustDo("POST", "/v1/collections/docs/search",
		`{"query":[1,0,0,0],"k":3,"filter":{"op":"gte","key":"page","value":2}}`,
		"", http.StatusOK)
	if got, _ := filtered["matches"].([]any); len(got) != 2 {
		t.Fatalf("filtered search returned %d matches, want 2", len(got))
	}

	d.mustDo("POST", "/v1/collections/docs/snapshot", "", "", http.StatusOK)
}

// The claim a service makes that a library does not have to: stop the process,
// start it again, and the data is still there — through a snapshot and a log
// replay that nothing in the test arranges.
func TestDataSurvivesARestart(t *testing.T) {
	dir := t.TempDir()

	first := start(t, "-dir", dir, "-addr", "127.0.0.1:0")
	first.mustDo("POST", "/v1/collections",
		`{"name":"docs","dimension":4,"sync_policy":"never"}`, "", http.StatusCreated)
	first.mustDo("POST", "/v1/collections/docs/vectors",
		`{"vectors":[{"id":"a","values":[1,0,0,0],"metadata":{"kept":true}}]}`, "", http.StatusOK)
	first.stop()

	second := start(t, "-dir", dir, "-addr", "127.0.0.1:0")

	list := second.mustDo("GET", "/v1/collections", "", "", http.StatusOK)
	cols, _ := list["collections"].([]any)
	if len(cols) != 1 {
		t.Fatalf("after a restart the server lists %d collections, want 1", len(cols))
	}

	got := second.mustDo("GET", "/v1/collections/docs/vectors/a", "", "", http.StatusOK)
	md, _ := got["metadata"].(map[string]any)
	if md["kept"] != true {
		t.Fatalf("vector came back as %v, want its metadata intact", got)
	}
}

func TestAuthTokenComesFromTheEnvironment(t *testing.T) {
	t.Setenv("GOVECDB_AUTH_TOKEN", "s3cret")
	d := start(t, "-dir", t.TempDir(), "-addr", "127.0.0.1:0")

	if status, _ := d.do("GET", "/v1/collections", "", ""); status != http.StatusUnauthorized {
		t.Errorf("unauthenticated request = %d, want 401", status)
	}
	if status, _ := d.do("GET", "/v1/collections", "", "wrong"); status != http.StatusUnauthorized {
		t.Errorf("wrong token = %d, want 401", status)
	}
	d.mustDo("GET", "/v1/collections", "", "s3cret", http.StatusOK)

	// The probe a load balancer makes, which has no token to offer.
	d.mustDo("GET", "/healthz", "", "", http.StatusOK)
}

// Shutdown has to release the port, or a restart — a rolling deploy, a systemd
// unit — fails on an address that is still in use.
func TestShutdownReleasesTheListener(t *testing.T) {
	dir := t.TempDir()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()

	first := start(t, "-dir", dir, "-addr", addr)
	first.mustDo("GET", "/healthz", "", "", http.StatusOK)
	first.stop()

	second := start(t, "-dir", dir, "-addr", addr)
	second.mustDo("GET", "/healthz", "", "", http.StatusOK)
}

func TestFlagValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{"no dir", nil, "-dir is required"},
		{"a certificate with no key", []string{"-dir", "x", "-tls-cert", "c.pem"}, "must be given together"},
		{"a key with no certificate", []string{"-dir", "x", "-tls-key", "k.pem"}, "must be given together"},
		{"a body limit of zero", []string{"-dir", "x", "-max-body", "0"}, "-max-body must be positive"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseFlags(tc.args, io.Discard)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("parseFlags = %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestBadLogLevelIsReportedRatherThanIgnored(t *testing.T) {
	var stderr bytes.Buffer
	err := run(context.Background(), []string{"-dir", t.TempDir(), "-log-level", "chatty"}, &stderr, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "log-level") {
		t.Fatalf("run = %v, want a complaint about -log-level", err)
	}
}

func TestVersionExits(t *testing.T) {
	var stderr bytes.Buffer
	if err := run(context.Background(), []string{"-version"}, &stderr, nil, nil); err != nil {
		t.Fatalf("run -version: %v", err)
	}
	if strings.TrimSpace(stderr.String()) == "" {
		t.Fatal("-version printed nothing")
	}
}

func TestHelpIsNotAnError(t *testing.T) {
	var stderr bytes.Buffer
	if err := run(context.Background(), []string{"-h"}, &stderr, nil, nil); err != nil {
		t.Fatalf("run -h = %v, want nil", err)
	}
	if !strings.Contains(stderr.String(), "GOVECDB_AUTH_TOKEN") {
		t.Errorf("the usage text does not say where the token comes from:\n%s", stderr.String())
	}
}

// The default is loopback, so a database does not become reachable from the
// network because somebody omitted a flag.
func TestTheDefaultAddressIsLoopback(t *testing.T) {
	cfg, err := parseFlags([]string{"-dir", "x"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	host, _, err := net.SplitHostPort(cfg.addr)
	if err != nil {
		t.Fatal(err)
	}
	if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() {
		t.Fatalf("default -addr is %q, which is not loopback", cfg.addr)
	}
}

func TestIsLoopback(t *testing.T) {
	for _, tc := range []struct {
		addr string
		want bool
	}{
		{"127.0.0.1:8080", true},
		{"[::1]:8080", true},
		{"0.0.0.0:8080", false},
		{"10.0.0.5:8080", false},
		{"[::]:8080", false},
	} {
		got := isLoopback(fakeAddr(tc.addr))
		if got != tc.want {
			t.Errorf("isLoopback(%s) = %v, want %v", tc.addr, got, tc.want)
		}
	}
}

type fakeAddr string

func (fakeAddr) Network() string    { return "tcp" }
func (a fakeAddr) String() string   { return string(a) }
func (a fakeAddr) GoString() string { return fmt.Sprintf("fakeAddr(%q)", string(a)) }
