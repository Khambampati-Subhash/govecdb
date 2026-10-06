//go:build unix

package dirlock

import (
	"errors"
	"fmt"
	"os"
	"syscall"
)

// Lock takes the lock on dir, shared or exclusive, without blocking. A lock held
// elsewhere is ErrLocked.
func Lock(dir string, shared bool) (*Handle, error) {
	f, err := os.Open(dir)
	if err != nil {
		return nil, fmt.Errorf("dirlock: open %q: %w", dir, err)
	}
	how := syscall.LOCK_EX
	if shared {
		how = syscall.LOCK_SH
	}
	// Non-blocking: a second process waiting silently behind the first looks
	// exactly like a hang, and the caller is better placed to decide whether to
	// retry than this function is.
	for {
		err = syscall.Flock(int(f.Fd()), how|syscall.LOCK_NB)
		if !errors.Is(err, syscall.EINTR) {
			break
		}
	}
	if err != nil {
		f.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) {
			return nil, fmt.Errorf("%w: %s", ErrLocked, dir)
		}
		return nil, fmt.Errorf("dirlock: lock %q: %w", dir, err)
	}
	return &Handle{f: f}, nil
}

// Release drops the lock. Closing the descriptor is the release; there is
// nothing on disk to remove. Safe on a nil Handle.
func (h *Handle) Release() error {
	if h == nil || h.f == nil {
		return nil
	}
	return h.f.Close()
}
