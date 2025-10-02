package logging

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"sync"
	"time"
)

// LogLevel represents the logging level
type LogLevel int

const (
	LevelDebug LogLevel = iota
	LevelInfo
	LevelWarn
	LevelError
	LevelFatal
)

// String returns the string representation of the log level
func (l LogLevel) String() string {
	switch l {
	case LevelDebug:
		return "DEBUG"
	case LevelInfo:
		return "INFO"
	case LevelWarn:
		return "WARN"
	case LevelError:
		return "ERROR"
	case LevelFatal:
		return "FATAL"
	default:
		return "UNKNOWN"
	}
}

// Logger defines the interface for structured logging
type Logger interface {
	// Basic logging methods
	Debug(msg string, args ...interface{})
	Info(msg string, args ...interface{})
	Warn(msg string, args ...interface{})
	Error(msg string, args ...interface{})
	Fatal(msg string, args ...interface{})

	// Context-aware logging
	DebugContext(ctx context.Context, msg string, args ...interface{})
	InfoContext(ctx context.Context, msg string, args ...interface{})
	WarnContext(ctx context.Context, msg string, args ...interface{})
	ErrorContext(ctx context.Context, msg string, args ...interface{})

	// Structured logging
	With(args ...interface{}) Logger
	WithComponent(component string) Logger
	WithOperation(operation string) Logger
	WithError(err error) Logger

	// Configuration
	SetLevel(level LogLevel)
	IsEnabled(level LogLevel) bool

	// Resource management
	Flush() error
	Close() error
}

// LoggerConfig represents logger configuration
type LoggerConfig struct {
	Level         LogLevel      `json:"level"`
	Format        string        `json:"format"` // "json" or "text"
	Output        io.Writer     `json:"-"`      // Output destination
	TimeFormat    string        `json:"time_format"`
	BufferSize    int           `json:"buffer_size"`
	FlushInterval time.Duration `json:"flush_interval"`

	// Component identification
	Component string `json:"component"`
	Version   string `json:"version"`
	NodeID    string `json:"node_id"`

	// Additional fields
	ExtraFields map[string]interface{} `json:"extra_fields"`
}

// DefaultLoggerConfig returns a default logger configuration
func DefaultLoggerConfig() *LoggerConfig {
	return &LoggerConfig{
		Level:         LevelInfo,
		Format:        "json",
		Output:        os.Stdout,
		TimeFormat:    time.RFC3339,
		BufferSize:    4096,
		FlushInterval: 5 * time.Second,
		Component:     "govecdb",
		ExtraFields:   make(map[string]interface{}),
	}
}

// StructuredLogger implements the Logger interface using slog
type StructuredLogger struct {
	config *LoggerConfig
	logger *slog.Logger
	
	// State
	level  LogLevel
	mu     sync.RWMutex
	closed bool

	// Background flushing
	flushTicker *time.Ticker
	stopChan    chan struct{}
	wg          sync.WaitGroup
}

// NewStructuredLogger creates a new structured logger
func NewStructuredLogger(config *LoggerConfig) (*StructuredLogger, error) {
	if config == nil {
		config = DefaultLoggerConfig()
	}

	// Create slog options
	opts := &slog.HandlerOptions{
		Level:     slog.Level(config.Level),
		AddSource: config.Level == LevelDebug,
	}

	// Create appropriate handler
	var handler slog.Handler
	if config.Format == "json" {
		handler = slog.NewJSONHandler(config.Output, opts)
	} else {
		handler = slog.NewTextHandler(config.Output, opts)
	}

	// Create logger with default fields
	logger := slog.New(handler).With(
		slog.String("component", config.Component),
		slog.String("version", config.Version),
		slog.String("node_id", config.NodeID),
	)

	// Add extra fields
	for key, value := range config.ExtraFields {
		logger = logger.With(slog.Any(key, value))
	}

	sl := &StructuredLogger{
		config:   config,
		logger:   logger,
		level:    config.Level,
		stopChan: make(chan struct{}),
	}

	// Start background flushing if configured
	if config.FlushInterval > 0 {
		sl.startBackgroundFlush()
	}

	return sl, nil
}

// Debug logs a debug message
func (sl *StructuredLogger) Debug(msg string, args ...interface{}) {
	if sl.IsEnabled(LevelDebug) {
		sl.logger.Debug(msg, sl.argsToSlogAny(args...)...)
	}
}

// Info logs an info message
func (sl *StructuredLogger) Info(msg string, args ...interface{}) {
	if sl.IsEnabled(LevelInfo) {
		sl.logger.Info(msg, sl.argsToSlogAny(args...)...)
	}
}

// Warn logs a warning message
func (sl *StructuredLogger) Warn(msg string, args ...interface{}) {
	if sl.IsEnabled(LevelWarn) {
		sl.logger.Warn(msg, sl.argsToSlogAny(args...)...)
	}
}

// Error logs an error message
func (sl *StructuredLogger) Error(msg string, args ...interface{}) {
	if sl.IsEnabled(LevelError) {
		sl.logger.Error(msg, sl.argsToSlogAny(args...)...)
	}
}

// Fatal logs a fatal message and exits
func (sl *StructuredLogger) Fatal(msg string, args ...interface{}) {
	sl.logger.Error(msg, sl.argsToSlogAny(args...)...)
	sl.Flush()
	os.Exit(1)
}

// DebugContext logs a debug message with context
func (sl *StructuredLogger) DebugContext(ctx context.Context, msg string, args ...interface{}) {
	if sl.IsEnabled(LevelDebug) {
		sl.logger.DebugContext(ctx, msg, sl.argsToSlogAny(args...)...)
	}
}

// InfoContext logs an info message with context
func (sl *StructuredLogger) InfoContext(ctx context.Context, msg string, args ...interface{}) {
	if sl.IsEnabled(LevelInfo) {
		sl.logger.InfoContext(ctx, msg, sl.argsToSlogAny(args...)...)
	}
}

// WarnContext logs a warning message with context
func (sl *StructuredLogger) WarnContext(ctx context.Context, msg string, args ...interface{}) {
	if sl.IsEnabled(LevelWarn) {
		sl.logger.WarnContext(ctx, msg, sl.argsToSlogAny(args...)...)
	}
}

// ErrorContext logs an error message with context
func (sl *StructuredLogger) ErrorContext(ctx context.Context, msg string, args ...interface{}) {
	if sl.IsEnabled(LevelError) {
		sl.logger.ErrorContext(ctx, msg, sl.argsToSlogAny(args...)...)
	}
}

// With returns a new logger with additional fields
func (sl *StructuredLogger) With(args ...interface{}) Logger {
	newLogger := &StructuredLogger{
		config:   sl.config,
		logger:   sl.logger.With(sl.argsToSlogAny(args...)...),
		level:    sl.level,
		stopChan: sl.stopChan,
	}
	return newLogger
}

// WithComponent returns a new logger with a component field
func (sl *StructuredLogger) WithComponent(component string) Logger {
	return sl.With("component", component)
}

// WithOperation returns a new logger with an operation field
func (sl *StructuredLogger) WithOperation(operation string) Logger {
	return sl.With("operation", operation)
}

// WithError returns a new logger with an error field
func (sl *StructuredLogger) WithError(err error) Logger {
	if err == nil {
		return sl
	}
	return sl.With("error", err.Error())
}

// SetLevel sets the logging level
func (sl *StructuredLogger) SetLevel(level LogLevel) {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	sl.level = level
}

// IsEnabled checks if a log level is enabled
func (sl *StructuredLogger) IsEnabled(level LogLevel) bool {
	sl.mu.RLock()
	defer sl.mu.RUnlock()
	return level >= sl.level && !sl.closed
}

// Flush flushes any buffered log entries
func (sl *StructuredLogger) Flush() error {
	// slog handles flushing internally, but we can add custom logic here
	return nil
}

// Close closes the logger and stops background operations
func (sl *StructuredLogger) Close() error {
	sl.mu.Lock()
	defer sl.mu.Unlock()

	if sl.closed {
		return nil
	}

	sl.closed = true

	// Stop background operations
	if sl.flushTicker != nil {
		sl.flushTicker.Stop()
		close(sl.stopChan)
		sl.wg.Wait()
	}

	return sl.Flush()
}

// Private methods

// argsToSlogAny converts variadic args to slog any values
func (sl *StructuredLogger) argsToSlogAny(args ...interface{}) []any {
	result := make([]any, 0, len(args))

	for i := 0; i < len(args)-1; i += 2 {
		key, ok := args[i].(string)
		if !ok {
			key = fmt.Sprintf("arg%d", i)
		}
		result = append(result, key, args[i+1])
	}

	// Handle odd number of args
	if len(args)%2 == 1 {
		result = append(result, "extra", args[len(args)-1])
	}

	return result
}

// startBackgroundFlush starts the background flush routine
func (sl *StructuredLogger) startBackgroundFlush() {
	sl.flushTicker = time.NewTicker(sl.config.FlushInterval)
	sl.wg.Add(1)

	go func() {
		defer sl.wg.Done()
		for {
			select {
			case <-sl.flushTicker.C:
				sl.Flush()
			case <-sl.stopChan:
				return
			}
		}
	}()
}

// Global logger instance
var globalLogger Logger

// SetGlobalLogger sets the global logger instance
func SetGlobalLogger(logger Logger) {
	globalLogger = logger
}

// GetGlobalLogger returns the global logger instance
func GetGlobalLogger() Logger {
	if globalLogger == nil {
		// Create default logger if none set
		config := DefaultLoggerConfig()
		logger, _ := NewStructuredLogger(config)
		globalLogger = logger
	}
	return globalLogger
}

// Convenience functions for global logger
func Debug(msg string, args ...interface{}) {
	GetGlobalLogger().Debug(msg, args...)
}

func Info(msg string, args ...interface{}) {
	GetGlobalLogger().Info(msg, args...)
}

func Warn(msg string, args ...interface{}) {
	GetGlobalLogger().Warn(msg, args...)
}

func Error(msg string, args ...interface{}) {
	GetGlobalLogger().Error(msg, args...)
}

func Fatal(msg string, args ...interface{}) {
	GetGlobalLogger().Fatal(msg, args...)
}

func WithComponent(component string) Logger {
	return GetGlobalLogger().WithComponent(component)
}

func WithOperation(operation string) Logger {
	return GetGlobalLogger().WithOperation(operation)
}

func WithError(err error) Logger {
	return GetGlobalLogger().WithError(err)
}
