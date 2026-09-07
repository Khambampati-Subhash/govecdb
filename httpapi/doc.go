// Package httpapi serves a [service.Manager] over HTTP and JSON.
//
// It is an http.Handler, not a program: the daemon in cmd/govecdbd is thirty
// lines of flags around this, and anything already running a Go server can mount
// it on a path of its own instead.
//
// # net/http and nothing else
//
// The module has no third-party dependencies and that is a design goal rather
// than an accident, so this package is written against the standard library
// alone — net/http for the server, encoding/json for the wire, log/slog for the
// log. That is a real constraint and it shaped what is here: there is no
// framework, no code generation, and no middleware library, because each of them
// would cost the guarantee that `go.mod` has no require block.
//
// The decision it forces, made before any of this was written: **gRPC and Raft
// are not standard library, so when they arrive they arrive as a separate
// module** importing this one. Nothing here has to move for that to happen,
// which is exactly why the choice was made first. See docs/SERVICE.md.
//
// # The routes
//
//	GET    /healthz                              liveness, never authenticated
//	GET    /readyz                               readiness, never authenticated
//	GET    /metrics                              Prometheus text
//
//	GET    /v1/collections                       list
//	POST   /v1/collections                       create
//	GET    /v1/collections/{name}                spec and statistics
//	DELETE /v1/collections/{name}                drop
//
//	POST   /v1/collections/{name}/vectors        add or replace
//	GET    /v1/collections/{name}/vectors/{id}   fetch one
//	DELETE /v1/collections/{name}/vectors/{id}   delete one
//	POST   /v1/collections/{name}/search         nearest neighbours
//
//	POST   /v1/collections/{name}/snapshot       snapshot now
//	POST   /v1/collections/{name}/compact        reclaim tombstoned slots
//
// # Decoding is the security boundary, again
//
// The root package says validation is where disk bytes become live objects.
// Here they are network bytes, which is the same problem with a shorter fuse, so
// this package refuses before it allocates:
//
//   - Bodies are capped by http.MaxBytesReader, and the cap is a configuration
//     value rather than a constant because a batch of 10,000 embeddings is
//     legitimately large.
//   - Unknown fields are rejected. A silently ignored "dimensions" is a
//     collection built at the wrong width, discovered weeks later.
//   - Filter trees are depth-limited, because a filter is recursion driven by a
//     request body.
//   - Metadata values are the same closed set the database accepts — string,
//     bool, int64, float64 — and nothing else decodes at all.
//
// What it does not do is invent a second validator. Dimension, K, Ef, id length
// and metadata size are all bounded by the root package, which is where those
// bounds are defined and tested; this layer's job is to hand them well-formed Go
// values and translate the error that comes back.
package httpapi
