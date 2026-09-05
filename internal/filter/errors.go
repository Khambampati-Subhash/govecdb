package filter

import "errors"

var (
	// ErrOperand means a comparison value is not one of the four metadata types
	// and cannot be converted into one without losing information.
	ErrOperand = errors.New("filter: unsupported comparison operand")

	// ErrNilFilter means a nil filter was passed to And, Or or Not. It is caught
	// by Validate rather than tolerated by Match: a nil child is a construction
	// bug, and silently treating it as "matches nothing" would turn one typo into
	// a search that returns nothing and explains nothing.
	ErrNilFilter = errors.New("filter: nil filter")
)
