package errors

import (
	"errors"
	"fmt"
	"runtime"
	"time"
)

// ErrorCode represents a specific error type
type ErrorCode string

const (
	// Core errors
	ErrCodeInvalidInput      ErrorCode = "INVALID_INPUT"
	ErrCodeNotFound          ErrorCode = "NOT_FOUND"
	ErrCodeAlreadyExists     ErrorCode = "ALREADY_EXISTS"
	ErrCodeDimensionMismatch ErrorCode = "DIMENSION_MISMATCH"

	// Storage errors
	ErrCodeStorageFull ErrorCode = "STORAGE_FULL"
	ErrCodeCorruption  ErrorCode = "DATA_CORRUPTION"
	ErrCodeIOError     ErrorCode = "IO_ERROR"

	// Network errors
	ErrCodeNetworkError     ErrorCode = "NETWORK_ERROR"
	ErrCodeTimeout          ErrorCode = "TIMEOUT"
	ErrCodeConnectionFailed ErrorCode = "CONNECTION_FAILED"

	// Cluster errors
	ErrCodeNodeUnavailable ErrorCode = "NODE_UNAVAILABLE"
	ErrCodeConsensusError  ErrorCode = "CONSENSUS_ERROR"
	ErrCodeSplitBrain      ErrorCode = "SPLIT_BRAIN"

	// Recovery errors
	ErrCodeRecoveryFailed  ErrorCode = "RECOVERY_FAILED"
	ErrCodeChecksumError   ErrorCode = "CHECKSUM_ERROR"
	ErrCodeVersionMismatch ErrorCode = "VERSION_MISMATCH"

	// System errors
	ErrCodeResourceExhausted ErrorCode = "RESOURCE_EXHAUSTED"
	ErrCodeInternalError     ErrorCode = "INTERNAL_ERROR"
	ErrCodeNotImplemented    ErrorCode = "NOT_IMPLEMENTED"
)

// ErrorSeverity represents the severity of an error
type ErrorSeverity int

const (
	SeverityLow ErrorSeverity = iota
	SeverityMedium
	SeverityHigh
	SeverityCritical
)

// String returns the string representation of error severity
func (s ErrorSeverity) String() string {
	switch s {
	case SeverityLow:
		return "LOW"
	case SeverityMedium:
		return "MEDIUM"
	case SeverityHigh:
		return "HIGH"
	case SeverityCritical:
		return "CRITICAL"
	default:
		return "UNKNOWN"
	}
}

// VecDBError represents a structured error with additional context
type VecDBError struct {
	Code      ErrorCode     `json:"code"`
	Message   string        `json:"message"`
	Severity  ErrorSeverity `json:"severity"`
	Component string        `json:"component"`
	Operation string        `json:"operation"`
	Cause     error         `json:"cause,omitempty"`
	Timestamp time.Time     `json:"timestamp"`

	// Additional context
	Details    map[string]interface{} `json:"details,omitempty"`
	StackTrace string                 `json:"stack_trace,omitempty"`
	NodeID     string                 `json:"node_id,omitempty"`
}

// Error implements the error interface
func (e *VecDBError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("[%s] %s: %v", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

// Unwrap returns the underlying cause
func (e *VecDBError) Unwrap() error {
	return e.Cause
}

// Is checks if the error matches the target
func (e *VecDBError) Is(target error) bool {
	if t, ok := target.(*VecDBError); ok {
		return e.Code == t.Code
	}
	return errors.Is(e.Cause, target)
}

// WithDetail adds additional detail to the error
func (e *VecDBError) WithDetail(key string, value interface{}) *VecDBError {
	if e.Details == nil {
		e.Details = make(map[string]interface{})
	}
	e.Details[key] = value
	return e
}

// WithStackTrace adds stack trace information
func (e *VecDBError) WithStackTrace() *VecDBError {
	if e.StackTrace == "" {
		e.StackTrace = captureStackTrace()
	}
	return e
}

// WithNodeID adds node identification
func (e *VecDBError) WithNodeID(nodeID string) *VecDBError {
	e.NodeID = nodeID
	return e
}

// NewError creates a new VecDBError
func NewError(code ErrorCode, message string) *VecDBError {
	return &VecDBError{
		Code:      code,
		Message:   message,
		Severity:  SeverityMedium,
		Timestamp: time.Now(),
	}
}

// NewErrorWithCause creates a new VecDBError with a cause
func NewErrorWithCause(code ErrorCode, message string, cause error) *VecDBError {
	return &VecDBError{
		Code:      code,
		Message:   message,
		Cause:     cause,
		Severity:  SeverityMedium,
		Timestamp: time.Now(),
	}
}

// WithSeverity sets the error severity
func WithSeverity(err *VecDBError, severity ErrorSeverity) *VecDBError {
	err.Severity = severity
	return err
}

// WithComponent sets the component that generated the error
func WithComponent(err *VecDBError, component string) *VecDBError {
	err.Component = component
	return err
}

// WithOperation sets the operation that failed
func WithOperation(err *VecDBError, operation string) *VecDBError {
	err.Operation = operation
	return err
}

// Predefined error constructors for common scenarios

// InvalidInput creates an invalid input error
func InvalidInput(message string) *VecDBError {
	return WithSeverity(NewError(ErrCodeInvalidInput, message), SeverityLow)
}

// NotFound creates a not found error
func NotFound(resource string) *VecDBError {
	return WithSeverity(NewError(ErrCodeNotFound, fmt.Sprintf("%s not found", resource)), SeverityLow)
}

// AlreadyExists creates an already exists error
func AlreadyExists(resource string) *VecDBError {
	return WithSeverity(NewError(ErrCodeAlreadyExists, fmt.Sprintf("%s already exists", resource)), SeverityLow)
}

// DimensionMismatch creates a dimension mismatch error
func DimensionMismatch(expected, actual int) *VecDBError {
	return WithSeverity(
		NewError(ErrCodeDimensionMismatch,
			fmt.Sprintf("dimension mismatch: expected %d, got %d", expected, actual)),
		SeverityMedium)
}

// StorageFull creates a storage full error
func StorageFull() *VecDBError {
	return WithSeverity(NewError(ErrCodeStorageFull, "storage capacity exceeded"), SeverityHigh)
}

// DataCorruption creates a data corruption error
func DataCorruption(details string) *VecDBError {
	return WithSeverity(NewError(ErrCodeCorruption, fmt.Sprintf("data corruption detected: %s", details)), SeverityCritical)
}

// IOError creates an I/O error
func IOError(operation string, cause error) *VecDBError {
	return WithSeverity(
		WithOperation(NewErrorWithCause(ErrCodeIOError, fmt.Sprintf("I/O error during %s", operation), cause), operation),
		SeverityHigh)
}

// NetworkError creates a network error
func NetworkError(operation string, cause error) *VecDBError {
	return WithSeverity(
		WithOperation(NewErrorWithCause(ErrCodeNetworkError, fmt.Sprintf("network error during %s", operation), cause), operation),
		SeverityMedium)
}

// Timeout creates a timeout error
func Timeout(operation string, duration time.Duration) *VecDBError {
	return WithSeverity(
		WithOperation(NewError(ErrCodeTimeout, fmt.Sprintf("operation %s timed out after %v", operation, duration)), operation),
		SeverityMedium).WithDetail("timeout_duration", duration)
}

// NodeUnavailable creates a node unavailable error
func NodeUnavailable(nodeID string) *VecDBError {
	return WithSeverity(
		NewError(ErrCodeNodeUnavailable, fmt.Sprintf("node %s is unavailable", nodeID)),
		SeverityHigh).WithNodeID(nodeID)
}

// ConsensusError creates a consensus error
func ConsensusError(message string) *VecDBError {
	return WithSeverity(NewError(ErrCodeConsensusError, message), SeverityCritical)
}

// RecoveryFailed creates a recovery failed error
func RecoveryFailed(phase string, cause error) *VecDBError {
	return WithSeverity(
		WithOperation(NewErrorWithCause(ErrCodeRecoveryFailed, fmt.Sprintf("recovery failed during %s phase", phase), cause), "recovery"),
		SeverityCritical).WithDetail("recovery_phase", phase)
}

// ChecksumError creates a checksum error
func ChecksumError(resource string, expected, actual string) *VecDBError {
	return WithSeverity(
		NewError(ErrCodeChecksumError, fmt.Sprintf("checksum mismatch for %s", resource)),
		SeverityCritical).
		WithDetail("expected_checksum", expected).
		WithDetail("actual_checksum", actual)
}

// InternalError creates an internal error
func InternalError(message string, cause error) *VecDBError {
	return WithSeverity(
		NewErrorWithCause(ErrCodeInternalError, message, cause),
		SeverityCritical).WithStackTrace()
}

// ResourceExhausted creates a resource exhausted error
func ResourceExhausted(resource string, limit interface{}) *VecDBError {
	return WithSeverity(
		NewError(ErrCodeResourceExhausted, fmt.Sprintf("%s resource exhausted", resource)),
		SeverityHigh).WithDetail("resource_limit", limit)
}

// Helper functions

// IsRetryable checks if an error is retryable
func IsRetryable(err error) bool {
	var vecdbErr *VecDBError
	if errors.As(err, &vecdbErr) {
		switch vecdbErr.Code {
		case ErrCodeTimeout, ErrCodeNetworkError, ErrCodeNodeUnavailable, ErrCodeResourceExhausted:
			return true
		default:
			return false
		}
	}
	return false
}

// IsCritical checks if an error is critical
func IsCritical(err error) bool {
	var vecdbErr *VecDBError
	if errors.As(err, &vecdbErr) {
		return vecdbErr.Severity == SeverityCritical
	}
	return false
}

// GetErrorCode extracts the error code from an error
func GetErrorCode(err error) ErrorCode {
	var vecdbErr *VecDBError
	if errors.As(err, &vecdbErr) {
		return vecdbErr.Code
	}
	return ErrCodeInternalError
}

// GetErrorSeverity extracts the error severity from an error
func GetErrorSeverity(err error) ErrorSeverity {
	var vecdbErr *VecDBError
	if errors.As(err, &vecdbErr) {
		return vecdbErr.Severity
	}
	return SeverityMedium
}

// Wrap wraps an existing error with VecDB error context
func Wrap(err error, code ErrorCode, message string) *VecDBError {
	if err == nil {
		return nil
	}

	// If it's already a VecDBError, preserve the original
	var vecdbErr *VecDBError
	if errors.As(err, &vecdbErr) {
		return vecdbErr
	}

	return NewErrorWithCause(code, message, err)
}

// WrapWithComponent wraps an error with component context
func WrapWithComponent(err error, component string, code ErrorCode, message string) *VecDBError {
	if err == nil {
		return nil
	}
	return WithComponent(Wrap(err, code, message), component)
}

// captureStackTrace captures the current stack trace
func captureStackTrace() string {
	const depth = 32
	var pcs [depth]uintptr
	n := runtime.Callers(3, pcs[:])

	frames := runtime.CallersFrames(pcs[:n])
	var trace []string

	for {
		frame, more := frames.Next()
		trace = append(trace, fmt.Sprintf("%s:%d %s", frame.File, frame.Line, frame.Function))
		if !more {
			break
		}
	}

	result := ""
	for _, line := range trace {
		result += line + "\n"
	}
	return result
}

// Error aggregation for batch operations

// ErrorList represents a collection of errors
type ErrorList struct {
	Errors []error `json:"errors"`
}

// Error implements the error interface
func (el *ErrorList) Error() string {
	if len(el.Errors) == 0 {
		return "no errors"
	}
	if len(el.Errors) == 1 {
		return el.Errors[0].Error()
	}
	return fmt.Sprintf("multiple errors occurred (%d total)", len(el.Errors))
}

// Add adds an error to the list
func (el *ErrorList) Add(err error) {
	if err != nil {
		el.Errors = append(el.Errors, err)
	}
}

// HasErrors returns true if there are any errors
func (el *ErrorList) HasErrors() bool {
	return len(el.Errors) > 0
}

// First returns the first error or nil
func (el *ErrorList) First() error {
	if len(el.Errors) == 0 {
		return nil
	}
	return el.Errors[0]
}

// Count returns the number of errors
func (el *ErrorList) Count() int {
	return len(el.Errors)
}

// ToError returns the ErrorList as an error if it contains errors, otherwise nil
func (el *ErrorList) ToError() error {
	if el.HasErrors() {
		return el
	}
	return nil
}

// NewErrorList creates a new error list
func NewErrorList() *ErrorList {
	return &ErrorList{
		Errors: make([]error, 0),
	}
}
