// Package dirlock is an advisory, kernel-held lock on a directory: what keeps a
// second process off a database or a collection root that one is already using.
//
// # Why flock, when a lock file was rejected
//
// The objection to a lock file is that a crash leaves it behind, and the restart
// that should have succeeded is refused until somebody deletes a file by hand on
// a guess that nothing is running. flock does not have that failure: the kernel
// drops the lock when the last descriptor holding it is closed, and a process
// that dies — SIGKILL, OOM, power loss — closes everything. So there is no stale
// state to clean up, and nothing is written at all: the lock is taken on the
// directory's own descriptor rather than on a file inside it.
//
// It is advisory, so it stops another govecdb and not a stray `rm`. It is per
// open file description, so a second Lock in the *same* process conflicts too —
// callers keep their own in-process registry in front of it so that case gets
// an error that does not blame another process. And on a network filesystem it
// is only as good as that filesystem's flock, which is one more reason the data
// directory belongs on local disk.
package dirlock

import (
	"errors"
	"os"
)

// ErrLocked means another holder has the directory: exclusively, or shared when
// an exclusive lock was asked for.
var ErrLocked = errors.New("dirlock: directory is locked by another process")

// Handle is a held lock. The zero value and nil both release as a no-op.
type Handle struct{ f *os.File }
