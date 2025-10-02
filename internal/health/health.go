package health

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// HealthStatus represents the health status of a component
type HealthStatus int

const (
	StatusHealthy HealthStatus = iota
	StatusDegraded
	StatusUnhealthy
	StatusUnknown
)

// String returns the string representation of health status
func (hs HealthStatus) String() string {
	switch hs {
	case StatusHealthy:
		return "healthy"
	case StatusDegraded:
		return "degraded"
	case StatusUnhealthy:
		return "unhealthy"
	case StatusUnknown:
		return "unknown"
	default:
		return "invalid"
	}
}

// HealthCheck represents a single health check
type HealthCheck struct {
	Name        string                 `json:"name"`
	Status      HealthStatus           `json:"status"`
	Message     string                 `json:"message,omitempty"`
	Details     map[string]interface{} `json:"details,omitempty"`
	LastChecked time.Time              `json:"last_checked"`
	Duration    time.Duration          `json:"duration"`
	Error       string                 `json:"error,omitempty"`
}

// CheckFunc represents a health check function
type CheckFunc func(ctx context.Context) *HealthCheck

// Checker interface for health checks
type Checker interface {
	Name() string
	Check(ctx context.Context) *HealthCheck
}

// SimpleChecker implements Checker interface with a function
type SimpleChecker struct {
	name     string
	checkFn  CheckFunc
	timeout  time.Duration
	interval time.Duration
}

// NewSimpleChecker creates a new simple checker
func NewSimpleChecker(name string, checkFn CheckFunc) *SimpleChecker {
	return &SimpleChecker{
		name:     name,
		checkFn:  checkFn,
		timeout:  30 * time.Second,
		interval: 30 * time.Second,
	}
}

// WithTimeout sets the check timeout
func (sc *SimpleChecker) WithTimeout(timeout time.Duration) *SimpleChecker {
	sc.timeout = timeout
	return sc
}

// WithInterval sets the check interval
func (sc *SimpleChecker) WithInterval(interval time.Duration) *SimpleChecker {
	sc.interval = interval
	return sc
}

// Name returns the checker name
func (sc *SimpleChecker) Name() string {
	return sc.name
}

// Check executes the health check
func (sc *SimpleChecker) Check(ctx context.Context) *HealthCheck {
	start := time.Now()
	
	// Create timeout context
	checkCtx, cancel := context.WithTimeout(ctx, sc.timeout)
	defer cancel()
	
	// Run the check
	result := sc.checkFn(checkCtx)
	if result == nil {
		result = &HealthCheck{
			Name:   sc.name,
			Status: StatusUnknown,
		}
	}
	
	// Fill in timing information
	result.Name = sc.name
	result.LastChecked = start
	result.Duration = time.Since(start)
	
	// Handle timeout
	if checkCtx.Err() == context.DeadlineExceeded {
		result.Status = StatusUnhealthy
		result.Error = "health check timed out"
	}
	
	return result
}

// HealthChecker manages multiple health checks
type HealthChecker struct {
	checkers    map[string]Checker
	lastResults map[string]*HealthCheck
	config      *HealthCheckerConfig
	mu          sync.RWMutex
	
	// Background checking
	stopChan chan struct{}
	wg       sync.WaitGroup
	running  bool
}

// HealthCheckerConfig represents health checker configuration
type HealthCheckerConfig struct {
	CheckInterval    time.Duration `json:"check_interval"`
	DefaultTimeout   time.Duration `json:"default_timeout"`
	EnableBackground bool          `json:"enable_background"`
	MaxConcurrent    int           `json:"max_concurrent"`
}

// DefaultHealthCheckerConfig returns default configuration
func DefaultHealthCheckerConfig() *HealthCheckerConfig {
	return &HealthCheckerConfig{
		CheckInterval:    30 * time.Second,
		DefaultTimeout:   10 * time.Second,
		EnableBackground: true,
		MaxConcurrent:    10,
	}
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(config *HealthCheckerConfig) *HealthChecker {
	if config == nil {
		config = DefaultHealthCheckerConfig()
	}
	
	return &HealthChecker{
		checkers:    make(map[string]Checker),
		lastResults: make(map[string]*HealthCheck),
		config:      config,
		stopChan:    make(chan struct{}),
	}
}

// RegisterChecker registers a health checker
func (hc *HealthChecker) RegisterChecker(checker Checker) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	
	hc.checkers[checker.Name()] = checker
}

// RegisterFunc registers a health check function
func (hc *HealthChecker) RegisterFunc(name string, checkFn CheckFunc) {
	checker := NewSimpleChecker(name, checkFn).
		WithTimeout(hc.config.DefaultTimeout).
		WithInterval(hc.config.CheckInterval)
	hc.RegisterChecker(checker)
}

// UnregisterChecker removes a health checker
func (hc *HealthChecker) UnregisterChecker(name string) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	
	delete(hc.checkers, name)
	delete(hc.lastResults, name)
}

// CheckAll runs all health checks
func (hc *HealthChecker) CheckAll(ctx context.Context) map[string]*HealthCheck {
	hc.mu.RLock()
	checkers := make([]Checker, 0, len(hc.checkers))
	for _, checker := range hc.checkers {
		checkers = append(checkers, checker)
	}
	hc.mu.RUnlock()
	
	results := make(map[string]*HealthCheck)
	resultChan := make(chan *HealthCheck, len(checkers))
	
	// Limit concurrent checks
	semaphore := make(chan struct{}, hc.config.MaxConcurrent)
	
	// Start checks
	for _, checker := range checkers {
		go func(c Checker) {
			semaphore <- struct{}{} // Acquire
			defer func() { <-semaphore }() // Release
			
			result := c.Check(ctx)
			resultChan <- result
		}(checker)
	}
	
	// Collect results
	for i := 0; i < len(checkers); i++ {
		result := <-resultChan
		results[result.Name] = result
	}
	
	// Update cache
	hc.mu.Lock()
	for name, result := range results {
		hc.lastResults[name] = result
	}
	hc.mu.Unlock()
	
	return results
}

// CheckOne runs a specific health check
func (hc *HealthChecker) CheckOne(ctx context.Context, name string) (*HealthCheck, error) {
	hc.mu.RLock()
	checker, exists := hc.checkers[name]
	hc.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("health checker not found: %s", name)
	}
	
	result := checker.Check(ctx)
	
	// Update cache
	hc.mu.Lock()
	hc.lastResults[name] = result
	hc.mu.Unlock()
	
	return result, nil
}

// GetLastResults returns the last health check results
func (hc *HealthChecker) GetLastResults() map[string]*HealthCheck {
	hc.mu.RLock()
	defer hc.mu.RUnlock()
	
	results := make(map[string]*HealthCheck, len(hc.lastResults))
	for name, result := range hc.lastResults {
		// Create a copy to avoid race conditions
		resultCopy := *result
		results[name] = &resultCopy
	}
	
	return results
}

// GetOverallStatus returns the overall system health status
func (hc *HealthChecker) GetOverallStatus() HealthStatus {
	hc.mu.RLock()
	defer hc.mu.RUnlock()
	
	if len(hc.lastResults) == 0 {
		return StatusUnknown
	}
	
	hasUnhealthy := false
	hasDegraded := false
	
	for _, result := range hc.lastResults {
		switch result.Status {
		case StatusUnhealthy:
			hasUnhealthy = true
		case StatusDegraded:
			hasDegraded = true
		}
	}
	
	if hasUnhealthy {
		return StatusUnhealthy
	}
	if hasDegraded {
		return StatusDegraded
	}
	
	return StatusHealthy
}

// Start begins background health checking
func (hc *HealthChecker) Start(ctx context.Context) error {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	
	if hc.running {
		return fmt.Errorf("health checker already running")
	}
	
	if !hc.config.EnableBackground {
		return nil
	}
	
	hc.running = true
	hc.wg.Add(1)
	
	go func() {
		defer hc.wg.Done()
		ticker := time.NewTicker(hc.config.CheckInterval)
		defer ticker.Stop()
		
		for {
			select {
			case <-ticker.C:
				hc.CheckAll(ctx)
			case <-hc.stopChan:
				return
			case <-ctx.Done():
				return
			}
		}
	}()
	
	return nil
}

// Stop stops background health checking
func (hc *HealthChecker) Stop() error {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	
	if !hc.running {
		return nil
	}
	
	hc.running = false
	close(hc.stopChan)
	hc.wg.Wait()
	
	return nil
}

// IsRunning returns true if background checking is running
func (hc *HealthChecker) IsRunning() bool {
	hc.mu.RLock()
	defer hc.mu.RUnlock()
	return hc.running
}

// HealthSummary represents a summary of system health
type HealthSummary struct {
	Status      HealthStatus             `json:"status"`
	Timestamp   time.Time                `json:"timestamp"`
	Checks      map[string]*HealthCheck  `json:"checks"`
	Summary     map[string]int           `json:"summary"`
	Details     map[string]interface{}   `json:"details,omitempty"`
}

// GetHealthSummary returns a complete health summary
func (hc *HealthChecker) GetHealthSummary() *HealthSummary {
	results := hc.GetLastResults()
	summary := make(map[string]int)
	
	for _, result := range results {
		summary[result.Status.String()]++
	}
	
	return &HealthSummary{
		Status:    hc.GetOverallStatus(),
		Timestamp: time.Now(),
		Checks:    results,
		Summary:   summary,
	}
}

// Common health check functions

// DatabaseHealthCheck creates a database connectivity health check
func DatabaseHealthCheck(name string, pingFn func(ctx context.Context) error) CheckFunc {
	return func(ctx context.Context) *HealthCheck {
		err := pingFn(ctx)
		if err != nil {
			return &HealthCheck{
				Status:  StatusUnhealthy,
				Message: "Database connection failed",
				Error:   err.Error(),
			}
		}
		
		return &HealthCheck{
			Status:  StatusHealthy,
			Message: "Database connection successful",
		}
	}
}

// DiskSpaceHealthCheck creates a disk space health check
func DiskSpaceHealthCheck(path string, warningThreshold, criticalThreshold float64) CheckFunc {
	return func(ctx context.Context) *HealthCheck {
		// This is a simplified implementation
		// In a real implementation, you would check actual disk usage
		usage := 0.1 // Placeholder: 10% usage
		
		details := map[string]interface{}{
			"path":               path,
			"usage_percent":      usage * 100,
			"warning_threshold":  warningThreshold * 100,
			"critical_threshold": criticalThreshold * 100,
		}
		
		if usage >= criticalThreshold {
			return &HealthCheck{
				Status:  StatusUnhealthy,
				Message: fmt.Sprintf("Disk usage critical: %.1f%%", usage*100),
				Details: details,
			}
		}
		
		if usage >= warningThreshold {
			return &HealthCheck{
				Status:  StatusDegraded,
				Message: fmt.Sprintf("Disk usage warning: %.1f%%", usage*100),
				Details: details,
			}
		}
		
		return &HealthCheck{
			Status:  StatusHealthy,
			Message: fmt.Sprintf("Disk usage normal: %.1f%%", usage*100),
			Details: details,
		}
	}
}

// MemoryHealthCheck creates a memory usage health check
func MemoryHealthCheck(warningThreshold, criticalThreshold float64) CheckFunc {
	return func(ctx context.Context) *HealthCheck {
		// This is a simplified implementation
		// In a real implementation, you would check actual memory usage
		usage := 0.15 // Placeholder: 15% usage
		
		details := map[string]interface{}{
			"usage_percent":      usage * 100,
			"warning_threshold":  warningThreshold * 100,
			"critical_threshold": criticalThreshold * 100,
		}
		
		if usage >= criticalThreshold {
			return &HealthCheck{
				Status:  StatusUnhealthy,
				Message: fmt.Sprintf("Memory usage critical: %.1f%%", usage*100),
				Details: details,
			}
		}
		
		if usage >= warningThreshold {
			return &HealthCheck{
				Status:  StatusDegraded,
				Message: fmt.Sprintf("Memory usage warning: %.1f%%", usage*100),
				Details: details,
			}
		}
		
		return &HealthCheck{
			Status:  StatusHealthy,
			Message: fmt.Sprintf("Memory usage normal: %.1f%%", usage*100),
			Details: details,
		}
	}
}

// Global health checker
var globalHealthChecker = NewHealthChecker(DefaultHealthCheckerConfig())

// GetGlobalHealthChecker returns the global health checker
func GetGlobalHealthChecker() *HealthChecker {
	return globalHealthChecker
}

// RegisterGlobalCheck registers a health check with the global checker
func RegisterGlobalCheck(name string, checkFn CheckFunc) {
	globalHealthChecker.RegisterFunc(name, checkFn)
}

// CheckGlobalHealth checks all global health checks
func CheckGlobalHealth(ctx context.Context) map[string]*HealthCheck {
	return globalHealthChecker.CheckAll(ctx)
}

// GetGlobalHealthSummary returns the global health summary
func GetGlobalHealthSummary() *HealthSummary {
	return globalHealthChecker.GetHealthSummary()
}