package httpapi

import (
	"errors"
	"net/http"

	"github.com/khambampati-subhash/govecdb"
	"github.com/khambampati-subhash/govecdb/service"
)

// errorResponse is the body of every failure. One shape, always, so a client
// parses errors the same way whatever produced them.
type errorResponse struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	// Code is a stable machine-readable token. It is what a client should branch
	// on; the message is for a human reading a log.
	Code string `json:"code"`

	// Message describes what was refused. For a 5xx it is deliberately vague —
	// see classify.
	Message string `json:"message"`
}

// Codes. Stable API surface: adding one is additive, changing one is not.
const (
	codeInvalidRequest  = "invalid_request"
	codeInvalidVector   = "invalid_vector"
	codeInvalidMetadata = "invalid_metadata"
	codeInvalidFilter   = "invalid_filter"
	codeInvalidSpec     = "invalid_spec"
	codeInvalidName     = "invalid_name"
	codeNotFound        = "not_found"
	codeAlreadyExists   = "already_exists"
	codePayloadTooLarge = "payload_too_large"
	codeUnsupportedType = "unsupported_media_type"
	codeUnauthorized    = "unauthorized"
	codeReadOnly        = "read_only"
	codeUnavailable     = "unavailable"
	codeTooManyOpen     = "too_many_open"
	codeInternal        = "internal"
)

// Transport-level refusals, which have no equivalent below this package: they
// are about the envelope rather than about what it contains.
var (
	errPayloadTooLarge      = errors.New("httpapi: request body exceeds the configured limit")
	errUnsupportedMediaType = errors.New("httpapi: unsupported media type, want application/json")
	errUnauthorized         = errors.New("httpapi: missing or invalid credentials")
)

// classify maps an error onto a status and a code.
//
// # Why 5xx messages are vague
//
// A 4xx says what the caller did wrong, in detail, because that is the only way
// they can fix it. A 5xx is the server's problem, and its error text is full of
// filesystem paths and internal state — none of which helps the client and all of
// which is worth not handing out. The detail goes to the log, where the operator
// is, along with the same request line.
//
// # The order matters
//
// service.ErrInvalidSpec wraps the database's ErrInvalidConfig when a collection
// is refused at open, so the more specific sentinel is tested first. Every pair
// like that is checked most-specific-first for the same reason.
func classify(err error) (status int, code, message string) {
	switch {
	case errors.Is(err, errPayloadTooLarge):
		return http.StatusRequestEntityTooLarge, codePayloadTooLarge, err.Error()
	case errors.Is(err, errUnsupportedMediaType):
		return http.StatusUnsupportedMediaType, codeUnsupportedType, err.Error()
	case errors.Is(err, errUnauthorized):
		return http.StatusUnauthorized, codeUnauthorized, err.Error()

	case errors.Is(err, service.ErrInvalidName):
		return http.StatusBadRequest, codeInvalidName, err.Error()
	case errors.Is(err, service.ErrExists):
		return http.StatusConflict, codeAlreadyExists, err.Error()
	case errors.Is(err, service.ErrInvalidSpec):
		return http.StatusBadRequest, codeInvalidSpec, err.Error()
	case errors.Is(err, service.ErrNotFound), errors.Is(err, govecdb.ErrNotFound):
		return http.StatusNotFound, codeNotFound, err.Error()

	case errors.Is(err, govecdb.ErrInvalidVector):
		return http.StatusBadRequest, codeInvalidVector, err.Error()
	// Before metadata, and this pairing is the reason the order is documented at
	// all: one function decodes both a metadata value and a filter operand, so a
	// filter carrying a bad operand wraps ErrInvalidMetadata inside
	// ErrInvalidFilter. To whoever wrote the query it is a filter that was
	// refused, and the outer sentinel is the one that says so.
	case errors.Is(err, govecdb.ErrInvalidFilter):
		return http.StatusBadRequest, codeInvalidFilter, err.Error()
	case errors.Is(err, govecdb.ErrInvalidMetadata):
		return http.StatusBadRequest, codeInvalidMetadata, err.Error()
	case errors.Is(err, govecdb.ErrInvalidRequest):
		return http.StatusBadRequest, codeInvalidRequest, err.Error()
	case errors.Is(err, govecdb.ErrInvalidConfig):
		return http.StatusBadRequest, codeInvalidSpec, err.Error()

	// A durability failure is fail-closed and permanent until the process is
	// restarted against a working disk, so it is 503 rather than 500: the
	// distinction a client cares about is "retrying elsewhere might work".
	case errors.Is(err, govecdb.ErrReadOnly):
		return http.StatusServiceUnavailable, codeReadOnly, "the database is read-only after a durability failure"

	// Capacity, not failure. The client should back off and retry, which is
	// exactly what 503 with a Retry-After means.
	case errors.Is(err, service.ErrTooManyOpen):
		return http.StatusServiceUnavailable, codeTooManyOpen, "too many collections are open; retry shortly"

	case errors.Is(err, service.ErrClosed), errors.Is(err, govecdb.ErrClosed):
		return http.StatusServiceUnavailable, codeUnavailable, "the server is shutting down"

	default:
		return http.StatusInternalServerError, codeInternal, "internal error"
	}
}
