//go:build !unix

package dirlock

// Lock is a no-op where flock is not available. Callers still enforce one
// writer per directory inside their own process; across processes it is the
// gap it was before this package, on a platform CI does not cover.
func Lock(string, bool) (*Handle, error) { return &Handle{}, nil }

// Release is a no-op to match.
func (*Handle) Release() error { return nil }
