package service

import "fmt"

// MaxNameBytes caps a collection name.
//
// 64 is well under every filesystem's per-component limit (255 on ext4 and APFS)
// with room for the suffixes a future format might add, and it is short enough
// that a name is readable in a log line and in a metric label.
const MaxNameBytes = 64

// ValidateName reports whether name may be used for a collection.
//
// # This is the security boundary of this package
//
// A collection name becomes a directory name, which vector ids never do. That
// single fact is what the rules below are for, and they are strict on purpose:
// the set of allowed names is small enough to reason about in one sitting, which
// is the only kind of allowlist worth having in front of a filesystem.
//
// What each rule stops:
//
//   - Only ASCII letters, digits, '-' and '_'. No separators, so a name cannot
//     name a directory other than its own; no '.', so "." and ".." are refused by
//     construction rather than by a special case somebody could later delete; no
//     NUL or control characters, which some kernels truncate at rather than
//     reject; and no spaces, which turn a shell command in a runbook into two
//     arguments.
//   - ASCII only. macOS normalizes filenames to NFD and Linux does not, so a name
//     written on one and read on the other would not compare equal to itself —
//     a database that appears to lose a collection when the volume is moved.
//   - Must start with a letter or digit. A leading '-' is read as a flag by every
//     command line tool an operator will point at the directory.
//   - Non-empty and at most MaxNameBytes.
//
// The payoff reaches further than the filesystem: names this narrow need no
// escaping in a URL path, in a Prometheus label, or in a log message, so nothing
// downstream has to re-derive the rules or get them subtly wrong.
func ValidateName(name string) error {
	if name == "" {
		return fmt.Errorf("%w: name is empty", ErrInvalidName)
	}
	if len(name) > MaxNameBytes {
		return fmt.Errorf("%w: %d bytes, max %d", ErrInvalidName, len(name), MaxNameBytes)
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case (c == '-' || c == '_') && i > 0:
		default:
			return fmt.Errorf("%w: %q at byte %d, want a letter, digit, '-' or '_' "+
				"(and a letter or digit first)", ErrInvalidName, string(rune(c)), i)
		}
	}
	return nil
}
