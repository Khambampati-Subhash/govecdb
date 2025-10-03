package quantization

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// ProductQuantization implements Product Quantization for vector compression
type ProductQuantization struct {
	// Configuration
	subspaces   int // Number of subspaces (M)
	bitsPerCode int // Bits per code (typically 8)
	dimension   int // Original vector dimension
	subspaceDim int // Dimension per subspace (D/M)

	// Codebooks for each subspace
	codebooks [][]Centroid // [subspace][code]

	// Quantization parameters
	trained  bool
	numCodes int // 2^bitsPerCode

	// Memory management
	vectorPool sync.Pool
	codePool   sync.Pool

	// Statistics
	compressionRatio float64
}

// Centroid represents a cluster centroid
type Centroid struct {
	Vector []float32
	Count  int64
}

// QuantizedVector represents a quantized vector
type QuantizedVector struct {
	Codes    []uint8   // Quantization codes
	Norm     float32   // L2 norm (for normalized similarity)
	Original []float32 // Optional: keep original for exact computation
}

// ScalarQuantization implements scalar quantization
type ScalarQuantization struct {
	min              []float32 // Min value per dimension
	max              []float32 // Max value per dimension
	scale            []float32 // Scale factor per dimension
	dimension        int
	bitsPerComponent int
	trained          bool
}

// BinaryQuantization implements binary quantization (LSH-style)
type BinaryQuantization struct {
	randomVectors [][]float32 // Random projection vectors
	dimension     int
	outputBits    int
	trained       bool
}

// NewProductQuantization creates a new Product Quantization instance
func NewProductQuantization(dimension, subspaces, bitsPerCode int) *ProductQuantization {
	if dimension%subspaces != 0 {
		panic("Dimension must be divisible by number of subspaces")
	}

	numCodes := 1 << bitsPerCode
	subspaceDim := dimension / subspaces

	pq := &ProductQuantization{
		subspaces:   subspaces,
		bitsPerCode: bitsPerCode,
		dimension:   dimension,
		subspaceDim: subspaceDim,
		numCodes:    numCodes,
		codebooks:   make([][]Centroid, subspaces),
		vectorPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, dimension)
			},
		},
		codePool: sync.Pool{
			New: func() interface{} {
				return make([]uint8, subspaces)
			},
		},
	}

	// Initialize codebooks
	for i := 0; i < subspaces; i++ {
		pq.codebooks[i] = make([]Centroid, numCodes)
		for j := 0; j < numCodes; j++ {
			pq.codebooks[i][j].Vector = make([]float32, subspaceDim)
		}
	}

	return pq
}

// Train trains the Product Quantization codebooks using k-means
func (pq *ProductQuantization) Train(vectors [][]float32, iterations int) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}

	if len(vectors[0]) != pq.dimension {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", pq.dimension, len(vectors[0]))
	}

	// Train each subspace independently
	for subspace := 0; subspace < pq.subspaces; subspace++ {
		startDim := subspace * pq.subspaceDim
		endDim := startDim + pq.subspaceDim

		// Extract subspace vectors
		subspaceVectors := make([][]float32, len(vectors))
		for i, vec := range vectors {
			subspaceVectors[i] = vec[startDim:endDim]
		}

		// Train codebook for this subspace using k-means
		err := pq.trainSubspaceCodebook(subspace, subspaceVectors, iterations)
		if err != nil {
			return fmt.Errorf("failed to train subspace %d: %w", subspace, err)
		}
	}

	pq.trained = true
	pq.calculateCompressionRatio()

	return nil
}

// trainSubspaceCodebook trains a single subspace codebook using k-means
func (pq *ProductQuantization) trainSubspaceCodebook(subspace int, vectors [][]float32, iterations int) error {
	codebooks := pq.codebooks[subspace]
	k := len(codebooks)

	// Initialize centroids randomly
	for i := 0; i < k; i++ {
		// Use random vector as initial centroid
		randIdx := i % len(vectors)
		copy(codebooks[i].Vector, vectors[randIdx])
		codebooks[i].Count = 0
	}

	// K-means iterations
	for iter := 0; iter < iterations; iter++ {
		// Reset centroids
		for i := 0; i < k; i++ {
			for j := 0; j < len(codebooks[i].Vector); j++ {
				codebooks[i].Vector[j] = 0
			}
			codebooks[i].Count = 0
		}

		// Assign vectors to centroids
		for _, vec := range vectors {
			closest := pq.findClosestCentroid(subspace, vec)

			// Update centroid
			for j := 0; j < len(vec); j++ {
				codebooks[closest].Vector[j] += vec[j]
			}
			codebooks[closest].Count++
		}

		// Normalize centroids
		for i := 0; i < k; i++ {
			if codebooks[i].Count > 0 {
				count := float32(codebooks[i].Count)
				for j := 0; j < len(codebooks[i].Vector); j++ {
					codebooks[i].Vector[j] /= count
				}
			}
		}
	}

	return nil
}

// findClosestCentroid finds the closest centroid to a vector
func (pq *ProductQuantization) findClosestCentroid(subspace int, vector []float32) int {
	codebooks := pq.codebooks[subspace]
	bestIdx := 0
	bestDist := pq.calculateDistance(vector, codebooks[0].Vector)

	for i := 1; i < len(codebooks); i++ {
		dist := pq.calculateDistance(vector, codebooks[i].Vector)
		if dist < bestDist {
			bestDist = dist
			bestIdx = i
		}
	}

	return bestIdx
}

// calculateDistance calculates L2 distance between vectors
func (pq *ProductQuantization) calculateDistance(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// Quantize quantizes a vector using the trained codebooks
func (pq *ProductQuantization) Quantize(vector []float32) (*QuantizedVector, error) {
	if !pq.trained {
		return nil, fmt.Errorf("quantizer not trained")
	}

	if len(vector) != pq.dimension {
		return nil, fmt.Errorf("vector dimension mismatch")
	}

	codes := pq.codePool.Get().([]uint8)
	defer pq.codePool.Put(codes)

	codes = codes[:pq.subspaces] // Ensure correct length

	// Calculate norm for similarity computation
	var norm float32
	for _, v := range vector {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))

	// Quantize each subspace
	for subspace := 0; subspace < pq.subspaces; subspace++ {
		startDim := subspace * pq.subspaceDim
		endDim := startDim + pq.subspaceDim

		subvector := vector[startDim:endDim]
		codes[subspace] = uint8(pq.findClosestCentroid(subspace, subvector))
	}

	// Create result
	result := &QuantizedVector{
		Codes: make([]uint8, pq.subspaces),
		Norm:  norm,
	}
	copy(result.Codes, codes)

	return result, nil
}

// Reconstruct reconstructs an approximate vector from quantization codes
func (pq *ProductQuantization) Reconstruct(qv *QuantizedVector) []float32 {
	if !pq.trained {
		return nil
	}

	vector := pq.vectorPool.Get().([]float32)
	vector = vector[:pq.dimension] // Ensure correct length

	// Reconstruct each subspace
	for subspace := 0; subspace < pq.subspaces; subspace++ {
		startDim := subspace * pq.subspaceDim
		code := qv.Codes[subspace]

		centroid := pq.codebooks[subspace][code].Vector
		copy(vector[startDim:startDim+pq.subspaceDim], centroid)
	}

	return vector
}

// CalculateAsymmetricDistance calculates asymmetric distance for search
// This is the key to PQ's search efficiency
func (pq *ProductQuantization) CalculateAsymmetricDistance(query []float32, qv *QuantizedVector) float32 {
	if !pq.trained {
		return math32.Inf(1)
	}

	var distance float32

	// Calculate distance per subspace
	for subspace := 0; subspace < pq.subspaces; subspace++ {
		startDim := subspace * pq.subspaceDim
		endDim := startDim + pq.subspaceDim

		querySubvector := query[startDim:endDim]
		code := qv.Codes[subspace]
		centroid := pq.codebooks[subspace][code].Vector

		// Add squared distance for this subspace
		for i := 0; i < pq.subspaceDim; i++ {
			diff := querySubvector[i] - centroid[i]
			distance += diff * diff
		}
	}

	return distance
}

// calculateCompressionRatio calculates the compression ratio
func (pq *ProductQuantization) calculateCompressionRatio() {
	originalBytes := pq.dimension * 4 // 4 bytes per float32
	compressedBytes := pq.subspaces * (pq.bitsPerCode / 8)
	pq.compressionRatio = float64(originalBytes) / float64(compressedBytes)
}

// GetCompressionRatio returns the compression ratio
func (pq *ProductQuantization) GetCompressionRatio() float64 {
	return pq.compressionRatio
}

// NewScalarQuantization creates a new scalar quantization instance
func NewScalarQuantization(dimension, bitsPerComponent int) *ScalarQuantization {
	return &ScalarQuantization{
		min:              make([]float32, dimension),
		max:              make([]float32, dimension),
		scale:            make([]float32, dimension),
		dimension:        dimension,
		bitsPerComponent: bitsPerComponent,
	}
}

// Train trains the scalar quantization parameters
func (sq *ScalarQuantization) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no training vectors provided")
	}

	// Initialize min/max
	for i := 0; i < sq.dimension; i++ {
		sq.min[i] = math32.Inf(1)
		sq.max[i] = math32.Inf(-1)
	}

	// Find min/max for each dimension
	for _, vec := range vectors {
		for i, v := range vec {
			if v < sq.min[i] {
				sq.min[i] = v
			}
			if v > sq.max[i] {
				sq.max[i] = v
			}
		}
	}

	// Calculate scale factors
	maxValInt := (1 << uint(sq.bitsPerComponent)) - 1
	maxVal := float32(maxValInt)
	for i := 0; i < sq.dimension; i++ {
		range_val := sq.max[i] - sq.min[i]
		if range_val > 0 {
			sq.scale[i] = maxVal / range_val
		} else {
			sq.scale[i] = 1.0
		}
	}

	sq.trained = true
	return nil
}

// Quantize quantizes a vector using scalar quantization
func (sq *ScalarQuantization) Quantize(vector []float32) []uint8 {
	if !sq.trained {
		return nil
	}

	quantized := make([]uint8, sq.dimension)

	for i, v := range vector {
		// Clamp to min/max
		if v < sq.min[i] {
			v = sq.min[i]
		}
		if v > sq.max[i] {
			v = sq.max[i]
		}

		// Scale and quantize
		scaled := (v - sq.min[i]) * sq.scale[i]
		quantized[i] = uint8(scaled + 0.5) // Round to nearest
	}

	return quantized
}

// Reconstruct reconstructs a vector from scalar quantized values
func (sq *ScalarQuantization) Reconstruct(quantized []uint8) []float32 {
	if !sq.trained {
		return nil
	}

	vector := make([]float32, sq.dimension)

	for i, q := range quantized {
		// Dequantize
		vector[i] = sq.min[i] + float32(q)/sq.scale[i]
	}

	return vector
}

// NewBinaryQuantization creates a new binary quantization instance
func NewBinaryQuantization(dimension, outputBits int) *BinaryQuantization {
	return &BinaryQuantization{
		dimension:  dimension,
		outputBits: outputBits,
	}
}

// Train trains the binary quantization using random projections
func (bq *BinaryQuantization) Train(vectors [][]float32) error {
	// Generate random projection vectors
	bq.randomVectors = make([][]float32, bq.outputBits)

	for i := 0; i < bq.outputBits; i++ {
		bq.randomVectors[i] = make([]float32, bq.dimension)

		// Generate random Gaussian vector
		for j := 0; j < bq.dimension; j++ {
			// Simplified random generation (use proper Gaussian in production)
			bq.randomVectors[i][j] = float32(math.Sin(float64(i*bq.dimension + j))) // Deterministic for testing
		}

		// Normalize
		var norm float32
		for _, v := range bq.randomVectors[i] {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))

		for j := 0; j < bq.dimension; j++ {
			bq.randomVectors[i][j] /= norm
		}
	}

	bq.trained = true
	return nil
}

// Quantize quantizes a vector to binary representation
func (bq *BinaryQuantization) Quantize(vector []float32) []uint8 {
	if !bq.trained {
		return nil
	}

	// Calculate number of bytes needed
	numBytes := (bq.outputBits + 7) / 8
	result := make([]uint8, numBytes)

	// Project and threshold
	for i := 0; i < bq.outputBits; i++ {
		// Dot product with random vector
		var dot float32
		for j := 0; j < bq.dimension; j++ {
			dot += vector[j] * bq.randomVectors[i][j]
		}

		// Set bit if positive
		if dot > 0 {
			byteIdx := i / 8
			bitIdx := uint(i % 8)
			result[byteIdx] |= 1 << bitIdx
		}
	}

	return result
}

// HammingDistance calculates Hamming distance between binary vectors
func (bq *BinaryQuantization) HammingDistance(a, b []uint8) int {
	distance := 0

	for i := 0; i < len(a) && i < len(b); i++ {
		xor := a[i] ^ b[i]

		// Count set bits (population count)
		for xor != 0 {
			distance++
			xor &= xor - 1 // Clear lowest set bit
		}
	}

	return distance
}

// VectorQuantizer interface for different quantization methods
type VectorQuantizer interface {
	Train(vectors [][]float32) error
	Quantize(vector []float32) interface{}
	GetCompressionRatio() float64
}

// Multi-level quantization for progressive precision
type MultiLevelQuantization struct {
	levels  []VectorQuantizer
	weights []float64
}

// NewMultiLevelQuantization creates a multi-level quantizer
func NewMultiLevelQuantization() *MultiLevelQuantization {
	return &MultiLevelQuantization{
		levels:  make([]VectorQuantizer, 0),
		weights: make([]float64, 0),
	}
}

// AddLevel adds a quantization level
func (mlq *MultiLevelQuantization) AddLevel(quantizer VectorQuantizer, weight float64) {
	mlq.levels = append(mlq.levels, quantizer)
	mlq.weights = append(mlq.weights, weight)
}

// Train trains all quantization levels
func (mlq *MultiLevelQuantization) Train(vectors [][]float32) error {
	for _, level := range mlq.levels {
		if err := level.Train(vectors); err != nil {
			return err
		}
	}
	return nil
}

// QuantizationStats provides statistics about quantization performance
type QuantizationStats struct {
	CompressionRatio  float64
	QuantizationError float64
	SearchSpeedup     float64
	MemoryReduction   float64
	TrainingTime      time.Duration
	QuantizationTime  time.Duration
}

// Constants and helper functions

var math32 = struct {
	Inf func(int) float32
}{
	Inf: func(sign int) float32 {
		if sign >= 0 {
			return float32(math.Inf(1))
		}
		return float32(math.Inf(-1))
	},
}
