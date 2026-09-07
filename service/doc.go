// Package service turns one embeddable database into many.
//
// A [Manager] owns a directory of collections: independent databases, each with
// its own dimension, metric and index, created and dropped at runtime and loaded
// only when something asks for one. It is the layer a server needs and a library
// does not, which is exactly why it sits above [govecdb.DB] rather than inside
// it — nothing here changes how a single database behaves.
//
// # Why this exists before the HTTP API
//
// A server with exactly one index is not a useful server: the resource model of
// the API is the collection lifecycle, so the lifecycle has to be settled first.
// It is settled here, in a package that can be tested without a socket.
//
// # The three things that needed thought
//
// Everything else is bookkeeping.
//
// Lifecycle. A collection's dimension, metric and M are structural — reopening a
// database under different ones is refused — so they cannot live in a command
// line flag that somebody edits between restarts. Each collection writes them to
// a spec file when it is created, and every later open reads them back. The
// effective values are recorded, never the zero that meant "default", so a change
// to this library's defaults cannot silently reshape a collection that already
// exists.
//
// Loading. Opening a collection replays a log and may load a snapshot, which
// takes as long as it takes. Doing that while holding the manager's lock would
// stall every other collection behind it, so an opening collection is registered
// as a placeholder and the lock is released for the duration; concurrent callers
// wait on that one open rather than starting a second.
//
// Eviction. An idle collection should stop costing memory, and a collection
// being used must not be closed underneath a search in flight. Both fall out of
// reference counting: [Manager.Use] borrows a database for the length of a
// callback, and only a collection with no borrowers is a candidate for eviction
// or for [Manager.Drop].
//
// # Names become directories
//
// This is the one place where a caller's string reaches the filesystem — vector
// ids never do — so [ValidateName] is a security boundary and not a style
// preference. See its documentation for what the rules buy.
package service
