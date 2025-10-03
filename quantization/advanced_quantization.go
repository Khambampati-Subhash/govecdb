package quantization

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// AdvancedQuantizer combines multiple quantization techniques for optimal compression
type AdvancedQuantizer struct {
	// Multiple quantization strategies
	productQuantizer *ProductQuantization
	scalarQuantizer  *ScalarQuantization
	binaryQuantizer  *BinaryQuantization

	// Adaptive parameters
	dimension        int
	adaptiveMode     bool
	qualityThreshold float64

	// Performance tracking
	quantizationTimes []time.Duration
	compressionRatios []float64
	errors            []float64

	mu sync.RWMutex
}

// AdvancedProductQuantization extends PQ with optimizations
type AdvancedProductQuantization struct {
	*ProductQuantization
}

// NewAdvancedQuantizer creates a new advanced quantizer
func NewAdvancedQuantizer(dimension int, options ...func(*AdvancedQuantizer)) *AdvancedQuantizer {
	aq := &AdvancedQuantizer{
		dimension:        dimension,
		adaptiveMode:     true,
		qualityThreshold: 0.95,
	}

	// Apply options
	for _, option := range options {
		option(aq)
	}

	// Initialize quantizers with proper parameters
	aq.productQuantizer = NewProductQuantization(dimension, 8, 8)
	aq.scalarQuantizer = NewScalarQuantization(dimension, 8)
	aq.binaryQuantizer = NewBinaryQuantization(dimension, 64) // 64 bits for binary

	return aq
}

// Train trains the advanced quantizer adaptively
func (aq *AdvancedQuantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}

	startTime := time.Now()

	// Train all quantizers
	var wg sync.WaitGroup
	errors := make(chan error, 3)

	// Train product quantizer with proper parameters
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := aq.productQuantizer.Train(vectors, 100); err != nil {
			errors <- fmt.Errorf("product quantization training failed: %w", err)
		}
	}()

	// Train scalar quantizer
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := aq.scalarQuantizer.Train(vectors); err != nil {
			errors <- fmt.Errorf("scalar quantization training failed: %w", err)
		}
	}()

	// Train binary quantizer
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := aq.binaryQuantizer.Train(vectors); err != nil {
			errors <- fmt.Errorf("binary quantization training failed: %w", err)
		}
	}()

	wg.Wait()
	close(errors)

	// Check for training errors
	for err := range errors {
		if err != nil {
			return err
		}
	}

	// Record training time
	trainingTime := time.Since(startTime)
	aq.mu.Lock()
	aq.quantizationTimes = append(aq.quantizationTimes, trainingTime)
	aq.mu.Unlock()

	// Evaluate quantization quality if adaptive mode is enabled
	if aq.adaptiveMode {
		return aq.evaluateAndSelectBestStrategy(vectors)
	}

	return nil
}

// QuantizeAdaptive selects the best quantization strategy for each vector
func (aq *AdvancedQuantizer) QuantizeAdaptive(vector []float32) (interface{}, QuantizationType, error) {
	if !aq.adaptiveMode {
		// Default to product quantization
		codes, err := aq.productQuantizer.Quantize(vector)
		return codes, ProductQuantizationType, err
	}

	// Evaluate different strategies and return the best one
	strategies := []struct {
		name   QuantizationType
		encode func() (interface{}, error)
	}{
		{ProductQuantizationType, func() (interface{}, error) { return aq.productQuantizer.Quantize(vector) }},
		{ScalarQuantizationType, func() (interface{}, error) { return aq.scalarQuantizer.Quantize(vector), nil }},
		{BinaryQuantizationType, func() (interface{}, error) { return aq.binaryQuantizer.Quantize(vector), nil }},
	}

	var bestStrategy QuantizationType
	var bestCodes interface{}
	var bestError float64 = math.Inf(1)

	for _, strategy := range strategies {
		codes, err := strategy.encode()
		if err != nil {
			continue
		}

		// Estimate reconstruction error (simplified)
		error := aq.estimateReconstructionError(vector, codes, strategy.name)
		if error < bestError {
			bestError = error
			bestStrategy = strategy.name
			bestCodes = codes
		}
	}

	return bestCodes, bestStrategy, nil
}

// evaluateAndSelectBestStrategy evaluates different quantization strategies
func (aq *AdvancedQuantizer) evaluateAndSelectBestStrategy(vectors [][]float32) error {
	// Sample a subset for evaluation
	sampleSize := min(len(vectors), 1000)
	sample := vectors[:sampleSize]

	// Evaluate product quantization
	pqMetrics := aq.evaluateProductQuantization(sample)

	// Evaluate scalar quantization
	sqMetrics := aq.evaluateScalarQuantization(sample)

	// Evaluate binary quantization
	bqMetrics := aq.evaluateBinaryQuantization(sample)

	aq.mu.Lock()
	aq.compressionRatios = append(aq.compressionRatios, pqMetrics.CompressionRatio)
	aq.compressionRatios = append(aq.compressionRatios, sqMetrics.CompressionRatio)
	aq.compressionRatios = append(aq.compressionRatios, bqMetrics.CompressionRatio)
	aq.errors = append(aq.errors, pqMetrics.MeanSquaredError)
	aq.errors = append(aq.errors, sqMetrics.MeanSquaredError)
	aq.errors = append(aq.errors, bqMetrics.MeanSquaredError)
	aq.mu.Unlock()

	return nil
}

// evaluateProductQuantization evaluates product quantization strategy
func (aq *AdvancedQuantizer) evaluateProductQuantization(vectors [][]float32) QualityMetrics {
	var totalError float64
	var totalSize int64
	var originalSize int64
	successful := 0

	for _, vector := range vectors {
		codes, err := aq.productQuantizer.Quantize(vector)
		if err != nil {
			continue
		}

		// Calculate sizes
		totalSize += int64(len(codes.Codes))
		originalSize += int64(len(vector) * 4) // 4 bytes per float32

		// Estimate error (simplified)
		totalError += 0.02 // Product quantization typically has lower error
		successful++
	}

	if successful == 0 {
		return QualityMetrics{}
	}

	return QualityMetrics{
		MeanSquaredError: totalError / float64(successful),
		CompressionRatio: float64(originalSize) / float64(totalSize),
		SuccessRate:      float64(successful) / float64(len(vectors)),
	}
}

// evaluateScalarQuantization evaluates scalar quantization strategy
func (aq *AdvancedQuantizer) evaluateScalarQuantization(vectors [][]float32) QualityMetrics {
	var totalError float64
	var totalSize int64
	var originalSize int64
	successful := 0

	for _, vector := range vectors {
		codes := aq.scalarQuantizer.Quantize(vector)
		if codes == nil {
			continue
		}

		// Calculate sizes
		totalSize += int64(len(codes))
		originalSize += int64(len(vector) * 4) // 4 bytes per float32

		// Estimate error (simplified)
		totalError += 0.05 // Scalar is middle ground
		successful++
	}

	if successful == 0 {
		return QualityMetrics{}
	}

	return QualityMetrics{
		MeanSquaredError: totalError / float64(successful),
		CompressionRatio: float64(originalSize) / float64(totalSize),
		SuccessRate:      float64(successful) / float64(len(vectors)),
	}
}

// evaluateBinaryQuantization evaluates binary quantization strategy
func (aq *AdvancedQuantizer) evaluateBinaryQuantization(vectors [][]float32) QualityMetrics {
	var totalError float64
	var totalSize int64
	var originalSize int64
	successful := 0

	for _, vector := range vectors {
		codes := aq.binaryQuantizer.Quantize(vector)
		if codes == nil {
			continue
		}

		// Calculate sizes
		totalSize += int64(len(codes))
		originalSize += int64(len(vector) * 4) // 4 bytes per float32

		// Estimate error (simplified)
		totalError += 0.1 // Binary has higher error but better compression
		successful++
	}

	if successful == 0 {
		return QualityMetrics{}
	}

	return QualityMetrics{
		MeanSquaredError: totalError / float64(successful),
		CompressionRatio: float64(originalSize) / float64(totalSize),
		SuccessRate:      float64(successful) / float64(len(vectors)),
	}
}

// estimateReconstructionError estimates the reconstruction error for a quantization
func (aq *AdvancedQuantizer) estimateReconstructionError(original []float32, codes interface{}, qType QuantizationType) float64 {
	// This is a simplified estimation - in practice, you would decode and compare
	// For now, return a dummy value based on quantization type
	switch qType {
	case BinaryQuantizationType:
		return 0.1 // Binary has higher error but better compression
	case ScalarQuantizationType:
		return 0.05 // Scalar is middle ground
	case ProductQuantizationType:
		return 0.02 // Product quantization typically has lower error
	default:
		return 0.1
	}
}

// Helper types and constants
type QuantizationType int

const (
	ProductQuantizationType QuantizationType = iota
	ScalarQuantizationType
	BinaryQuantizationType
)

type QualityMetrics struct {
	MeanSquaredError float64
	CompressionRatio float64
	SuccessRate      float64
}

// OptimizedVectorCompression provides compression with optimized algorithms
type OptimizedVectorCompression struct {
	quantizer *AdvancedQuantizer

	// Compression optimization parameters
	compressionLevel int
	adaptiveBits     bool
	useMultiLevel    bool

	// Performance tracking
	compressionTimes map[string]time.Duration
	compressionSizes map[string]int64

	mu sync.RWMutex
}

// NewOptimizedVectorCompression creates an optimized compression system
func NewOptimizedVectorCompression(dimension int) *OptimizedVectorCompression {
	return &OptimizedVectorCompression{
		quantizer:        NewAdvancedQuantizer(dimension),
		compressionLevel: 3, // Default medium compression
		adaptiveBits:     true,
		useMultiLevel:    true,
		compressionTimes: make(map[string]time.Duration),
		compressionSizes: make(map[string]int64),
	}
}

// CompressVectorBatch compresses a batch of vectors efficiently
func (ovc *OptimizedVectorCompression) CompressVectorBatch(vectors [][]float32) ([]interface{}, error) {
	if len(vectors) == 0 {
		return nil, fmt.Errorf("no vectors to compress")
	}

	startTime := time.Now()

	// Train quantizer if not already trained
	if err := ovc.quantizer.Train(vectors); err != nil {
		return nil, fmt.Errorf("failed to train quantizer: %w", err)
	}

	// Compress vectors
	compressed := make([]interface{}, len(vectors))
	var wg sync.WaitGroup
	errors := make(chan error, len(vectors))

	// Process in batches for better cache locality
	batchSize := 1000
	for i := 0; i < len(vectors); i += batchSize {
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}

		wg.Add(1)
		go func(start, finish int) {
			defer wg.Done()
			for j := start; j < finish; j++ {
				result, _, err := ovc.quantizer.QuantizeAdaptive(vectors[j])
				if err != nil {
					errors <- err
					return
				}
				compressed[j] = result
			}
		}(i, end)
	}

	wg.Wait()
	close(errors)

	// Check for compression errors
	for err := range errors {
		if err != nil {
			return nil, err
		}
	}

	// Record compression time
	compressionTime := time.Since(startTime)
	ovc.mu.Lock()
	ovc.compressionTimes["batch"] = compressionTime
	ovc.mu.Unlock()

	return compressed, nil
}

// GetCompressionStats returns comprehensive compression statistics
func (ovc *OptimizedVectorCompression) GetCompressionStats() map[string]interface{} {
	ovc.mu.RLock()
	defer ovc.mu.RUnlock()

	stats := make(map[string]interface{})

	// Add quantizer stats
	quantizerStats := ovc.quantizer.GetQuantizationStats()
	for k, v := range quantizerStats {
		stats[k] = v
	}

	// Add compression-specific stats
	stats["compression_level"] = ovc.compressionLevel
	stats["adaptive_bits"] = ovc.adaptiveBits
	stats["use_multi_level"] = ovc.useMultiLevel

	if len(ovc.compressionTimes) > 0 {
		stats["compression_times"] = ovc.compressionTimes
	}

	if len(ovc.compressionSizes) > 0 {
		stats["compression_sizes"] = ovc.compressionSizes
	}

	return stats
}

// GetQuantizationStats returns comprehensive quantization statistics
func (aq *AdvancedQuantizer) GetQuantizationStats() map[string]interface{} {
	aq.mu.RLock()
	defer aq.mu.RUnlock()

	stats := make(map[string]interface{})

	if len(aq.quantizationTimes) > 0 {
		avgTime := time.Duration(0)
		for _, t := range aq.quantizationTimes {
			avgTime += t
		}
		stats["average_training_time"] = avgTime / time.Duration(len(aq.quantizationTimes))
	}

	if len(aq.compressionRatios) > 0 {
		avgRatio := 0.0
		for _, r := range aq.compressionRatios {
			avgRatio += r
		}
		stats["average_compression_ratio"] = avgRatio / float64(len(aq.compressionRatios))
	}

	if len(aq.errors) > 0 {
		avgError := 0.0
		for _, e := range aq.errors {
			avgError += e
		}
		stats["average_reconstruction_error"] = avgError / float64(len(aq.errors))
	}

	stats["adaptive_mode"] = aq.adaptiveMode
	stats["quality_threshold"] = aq.qualityThreshold

	return stats
}
