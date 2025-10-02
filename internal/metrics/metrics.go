package metrics

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// MetricType represents the type of metric
type MetricType int

const (
	MetricTypeCounter MetricType = iota
	MetricTypeGauge
	MetricTypeHistogram
	MetricTypeTiming
)

// String returns the string representation of metric type
func (mt MetricType) String() string {
	switch mt {
	case MetricTypeCounter:
		return "counter"
	case MetricTypeGauge:
		return "gauge"
	case MetricTypeHistogram:
		return "histogram"
	case MetricTypeTiming:
		return "timing"
	default:
		return "unknown"
	}
}

// Metric represents a single metric
type Metric interface {
	Name() string
	Type() MetricType
	Tags() map[string]string
	Value() interface{}
	Reset()
}

// Counter represents a monotonically increasing counter
type Counter struct {
	name  string
	tags  map[string]string
	value int64
}

// NewCounter creates a new counter
func NewCounter(name string, tags map[string]string) *Counter {
	return &Counter{
		name: name,
		tags: copyTags(tags),
	}
}

// Name returns the metric name
func (c *Counter) Name() string {
	return c.name
}

// Type returns the metric type
func (c *Counter) Type() MetricType {
	return MetricTypeCounter
}

// Tags returns the metric tags
func (c *Counter) Tags() map[string]string {
	return c.tags
}

// Value returns the current value
func (c *Counter) Value() interface{} {
	return atomic.LoadInt64(&c.value)
}

// Inc increments the counter by 1
func (c *Counter) Inc() {
	atomic.AddInt64(&c.value, 1)
}

// Add adds the given value to the counter
func (c *Counter) Add(delta int64) {
	atomic.AddInt64(&c.value, delta)
}

// Reset resets the counter to 0
func (c *Counter) Reset() {
	atomic.StoreInt64(&c.value, 0)
}

// Gauge represents a value that can go up and down
type Gauge struct {
	name  string
	tags  map[string]string
	value int64
}

// NewGauge creates a new gauge
func NewGauge(name string, tags map[string]string) *Gauge {
	return &Gauge{
		name: name,
		tags: copyTags(tags),
	}
}

// Name returns the metric name
func (g *Gauge) Name() string {
	return g.name
}

// Type returns the metric type
func (g *Gauge) Type() MetricType {
	return MetricTypeGauge
}

// Tags returns the metric tags
func (g *Gauge) Tags() map[string]string {
	return g.tags
}

// Value returns the current value
func (g *Gauge) Value() interface{} {
	return atomic.LoadInt64(&g.value)
}

// Set sets the gauge to a specific value
func (g *Gauge) Set(value int64) {
	atomic.StoreInt64(&g.value, value)
}

// Inc increments the gauge by 1
func (g *Gauge) Inc() {
	atomic.AddInt64(&g.value, 1)
}

// Dec decrements the gauge by 1
func (g *Gauge) Dec() {
	atomic.AddInt64(&g.value, -1)
}

// Add adds the given value to the gauge
func (g *Gauge) Add(delta int64) {
	atomic.AddInt64(&g.value, delta)
}

// Reset resets the gauge to 0
func (g *Gauge) Reset() {
	atomic.StoreInt64(&g.value, 0)
}

// Histogram represents a distribution of values
type Histogram struct {
	name    string
	tags    map[string]string
	buckets []float64
	counts  []int64
	sum     int64
	count   int64
	mu      sync.RWMutex
}

// NewHistogram creates a new histogram with default buckets
func NewHistogram(name string, tags map[string]string) *Histogram {
	return NewHistogramWithBuckets(name, tags, DefaultBuckets())
}

// NewHistogramWithBuckets creates a new histogram with custom buckets
func NewHistogramWithBuckets(name string, tags map[string]string, buckets []float64) *Histogram {
	return &Histogram{
		name:    name,
		tags:    copyTags(tags),
		buckets: append([]float64{}, buckets...),
		counts:  make([]int64, len(buckets)+1), // +1 for +Inf bucket
	}
}

// DefaultBuckets returns default histogram buckets
func DefaultBuckets() []float64 {
	return []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10}
}

// Name returns the metric name
func (h *Histogram) Name() string {
	return h.name
}

// Type returns the metric type
func (h *Histogram) Type() MetricType {
	return MetricTypeHistogram
}

// Tags returns the metric tags
func (h *Histogram) Tags() map[string]string {
	return h.tags
}

// Value returns the histogram summary
func (h *Histogram) Value() interface{} {
	h.mu.RLock()
	defer h.mu.RUnlock()
	
	return map[string]interface{}{
		"count":   atomic.LoadInt64(&h.count),
		"sum":     atomic.LoadInt64(&h.sum),
		"buckets": h.getBucketCounts(),
	}
}

// Observe adds an observation to the histogram
func (h *Histogram) Observe(value float64) {
	atomic.AddInt64(&h.count, 1)
	atomic.AddInt64(&h.sum, int64(value*1000)) // Store as milliseconds
	
	h.mu.RLock()
	defer h.mu.RUnlock()
	
	for i, bucket := range h.buckets {
		if value <= bucket {
			atomic.AddInt64(&h.counts[i], 1)
			return
		}
	}
	// Value is greater than all buckets, add to +Inf bucket
	atomic.AddInt64(&h.counts[len(h.buckets)], 1)
}

// Reset resets the histogram
func (h *Histogram) Reset() {
	atomic.StoreInt64(&h.count, 0)
	atomic.StoreInt64(&h.sum, 0)
	
	h.mu.Lock()
	defer h.mu.Unlock()
	
	for i := range h.counts {
		atomic.StoreInt64(&h.counts[i], 0)
	}
}

// getBucketCounts returns bucket counts (caller must hold read lock)
func (h *Histogram) getBucketCounts() map[string]int64 {
	result := make(map[string]int64)
	for i, bucket := range h.buckets {
		result[fmt.Sprintf("%.3f", bucket)] = atomic.LoadInt64(&h.counts[i])
	}
	result["+Inf"] = atomic.LoadInt64(&h.counts[len(h.buckets)])
	return result
}

// Timer helps measure execution time
type Timer struct {
	histogram *Histogram
	startTime time.Time
}

// NewTimer creates a new timer
func NewTimer(name string, tags map[string]string) *Timer {
	return &Timer{
		histogram: NewHistogram(name, tags),
		startTime: time.Now(),
	}
}

// Start starts/restarts the timer
func (t *Timer) Start() {
	t.startTime = time.Now()
}

// Stop stops the timer and records the duration
func (t *Timer) Stop() time.Duration {
	duration := time.Since(t.startTime)
	t.histogram.Observe(duration.Seconds())
	return duration
}

// Name returns the metric name
func (t *Timer) Name() string {
	return t.histogram.Name()
}

// Type returns the metric type
func (t *Timer) Type() MetricType {
	return MetricTypeTiming
}

// Tags returns the metric tags
func (t *Timer) Tags() map[string]string {
	return t.histogram.Tags()
}

// Value returns the timer histogram value
func (t *Timer) Value() interface{} {
	return t.histogram.Value()
}

// Reset resets the timer
func (t *Timer) Reset() {
	t.histogram.Reset()
	t.startTime = time.Now()
}

// MetricsRegistry manages all metrics
type MetricsRegistry struct {
	metrics map[string]Metric
	mu      sync.RWMutex
}

// NewMetricsRegistry creates a new metrics registry
func NewMetricsRegistry() *MetricsRegistry {
	return &MetricsRegistry{
		metrics: make(map[string]Metric),
	}
}

// Register registers a metric
func (r *MetricsRegistry) Register(metric Metric) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	key := r.metricKey(metric.Name(), metric.Tags())
	if _, exists := r.metrics[key]; exists {
		return fmt.Errorf("metric already exists: %s", key)
	}
	
	r.metrics[key] = metric
	return nil
}

// GetOrCreateCounter gets or creates a counter
func (r *MetricsRegistry) GetOrCreateCounter(name string, tags map[string]string) *Counter {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	key := r.metricKey(name, tags)
	if metric, exists := r.metrics[key]; exists {
		if counter, ok := metric.(*Counter); ok {
			return counter
		}
	}
	
	counter := NewCounter(name, tags)
	r.metrics[key] = counter
	return counter
}

// GetOrCreateGauge gets or creates a gauge
func (r *MetricsRegistry) GetOrCreateGauge(name string, tags map[string]string) *Gauge {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	key := r.metricKey(name, tags)
	if metric, exists := r.metrics[key]; exists {
		if gauge, ok := metric.(*Gauge); ok {
			return gauge
		}
	}
	
	gauge := NewGauge(name, tags)
	r.metrics[key] = gauge
	return gauge
}

// GetOrCreateHistogram gets or creates a histogram
func (r *MetricsRegistry) GetOrCreateHistogram(name string, tags map[string]string) *Histogram {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	key := r.metricKey(name, tags)
	if metric, exists := r.metrics[key]; exists {
		if histogram, ok := metric.(*Histogram); ok {
			return histogram
		}
	}
	
	histogram := NewHistogram(name, tags)
	r.metrics[key] = histogram
	return histogram
}

// GetOrCreateTimer gets or creates a timer
func (r *MetricsRegistry) GetOrCreateTimer(name string, tags map[string]string) *Timer {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	key := r.metricKey(name+"_timer", tags)
	if metric, exists := r.metrics[key]; exists {
		if timer, ok := metric.(*Timer); ok {
			return timer
		}
	}
	
	timer := NewTimer(name+"_timer", tags)
	r.metrics[key] = timer
	return timer
}

// GetAllMetrics returns all registered metrics
func (r *MetricsRegistry) GetAllMetrics() []Metric {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	metrics := make([]Metric, 0, len(r.metrics))
	for _, metric := range r.metrics {
		metrics = append(metrics, metric)
	}
	return metrics
}

// Reset resets all metrics
func (r *MetricsRegistry) Reset() {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	for _, metric := range r.metrics {
		metric.Reset()
	}
}

// metricKey generates a unique key for a metric
func (r *MetricsRegistry) metricKey(name string, tags map[string]string) string {
	key := name
	if len(tags) > 0 {
		key += "{"
		first := true
		for k, v := range tags {
			if !first {
				key += ","
			}
			key += fmt.Sprintf("%s=%s", k, v)
			first = false
		}
		key += "}"
	}
	return key
}

// MetricsReporter exports metrics to various backends
type MetricsReporter interface {
	Report(ctx context.Context, metrics []Metric) error
}

// ConsoleReporter logs metrics to console
type ConsoleReporter struct{}

// NewConsoleReporter creates a new console reporter
func NewConsoleReporter() *ConsoleReporter {
	return &ConsoleReporter{}
}

// Report reports metrics to console
func (cr *ConsoleReporter) Report(ctx context.Context, metrics []Metric) error {
	for _, metric := range metrics {
		fmt.Printf("%s{%v} %s = %v\n", 
			metric.Name(), 
			metric.Tags(), 
			metric.Type().String(), 
			metric.Value())
	}
	return nil
}

// Global registry
var globalRegistry = NewMetricsRegistry()

// GetGlobalRegistry returns the global metrics registry
func GetGlobalRegistry() *MetricsRegistry {
	return globalRegistry
}

// Helper functions for common metrics

// IncrementCounter increments a counter in the global registry
func IncrementCounter(name string, tags map[string]string) {
	globalRegistry.GetOrCreateCounter(name, tags).Inc()
}

// SetGauge sets a gauge value in the global registry
func SetGauge(name string, value int64, tags map[string]string) {
	globalRegistry.GetOrCreateGauge(name, tags).Set(value)
}

// ObserveHistogram adds an observation to a histogram in the global registry
func ObserveHistogram(name string, value float64, tags map[string]string) {
	globalRegistry.GetOrCreateHistogram(name, tags).Observe(value)
}

// TimeOperation times an operation and records it in a histogram
func TimeOperation(name string, tags map[string]string, operation func()) {
	start := time.Now()
	operation()
	duration := time.Since(start)
	ObserveHistogram(name, duration.Seconds(), tags)
}

// copyTags creates a copy of the tags map
func copyTags(tags map[string]string) map[string]string {
	if tags == nil {
		return make(map[string]string)
	}
	
	result := make(map[string]string, len(tags))
	for k, v := range tags {
		result[k] = v
	}
	return result
}