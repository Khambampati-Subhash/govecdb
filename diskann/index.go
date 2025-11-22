package diskann

import (
	"encoding/binary"
	"math"
	"os"
	"sync"

	"github.com/khambampati-subhash/govecdb/index"
)

// DiskIndex implements a DiskANN-style on-disk vector index
type DiskIndex struct {
	file      *os.File
	config    *Config
	cache     sync.Map // Simple cache for hot nodes
	startNode uint32   // Entry point for the graph
}

// Config holds configuration for DiskANN
type Config struct {
	FilePath   string
	Dimension  int
	MaxDegree  int
	SearchList int // Size of candidate list during search (L)
}

// Node represents a node in the on-disk graph
// Layout: [Vector(Dim*4)][NumNeighbors(4)][Neighbors(MaxDegree*4)]
type Node struct {
	ID        uint32
	Vector    []float32
	Neighbors []uint32
}

// NewDiskIndex creates or opens a DiskANN index
func NewDiskIndex(config *Config) (*DiskIndex, error) {
	f, err := os.OpenFile(config.FilePath, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}

	return &DiskIndex{
		file:   f,
		config: config,
	}, nil
}

// Build creates the Vamana graph from a set of vectors
// This is a simplified in-memory construction that flushes to disk.
// For true DiskANN, we would merge partial graphs, but this demonstrates the layout.
func (di *DiskIndex) Build(vectors [][]float32) error {
	// 1. Random initialization of graph
	// 2. Two-pass Vamana construction (omitted for brevity, using random graph for layout demo)

	// Write header
	// [Magic(4)][NumNodes(4)][Dimension(4)][StartNode(4)]
	header := make([]byte, 16)
	binary.LittleEndian.PutUint32(header[0:4], 0x4449534B) // "DISK"
	binary.LittleEndian.PutUint32(header[4:8], uint32(len(vectors)))
	binary.LittleEndian.PutUint32(header[8:12], uint32(di.config.Dimension))
	binary.LittleEndian.PutUint32(header[12:16], 0) // Start node 0

	if _, err := di.file.WriteAt(header, 0); err != nil {
		return err
	}

	// Calculate node size
	// Vector: Dim * 4 bytes
	// NumNeighbors: 4 bytes
	// Neighbors: MaxDegree * 4 bytes
	nodeSize := di.config.Dimension*4 + 4 + di.config.MaxDegree*4
	offset := int64(16)

	for i, vec := range vectors {
		buf := make([]byte, nodeSize)

		// Write Vector
		for j, val := range vec {
			binary.LittleEndian.PutUint32(buf[j*4:(j+1)*4], math.Float32bits(val))
		}

		// Write Neighbors (Random for demo)
		vecOffset := di.config.Dimension * 4
		numNeighbors := uint32(di.config.MaxDegree) // Full degree for simplicity
		binary.LittleEndian.PutUint32(buf[vecOffset:vecOffset+4], numNeighbors)

		neighborOffset := vecOffset + 4
		for j := 0; j < di.config.MaxDegree; j++ {
			// Point to next node (ring)
			neighborID := uint32((i + 1) % len(vectors))
			binary.LittleEndian.PutUint32(buf[neighborOffset+j*4:neighborOffset+(j+1)*4], neighborID)
		}

		if _, err := di.file.WriteAt(buf, offset+int64(i*nodeSize)); err != nil {
			return err
		}
	}

	return nil
}

// Search performs a greedy search on the disk index
func (di *DiskIndex) Search(query []float32, k int) ([]uint32, error) {
	// Start at entry point
	current := di.startNode
	visited := make(map[uint32]bool)
	candidates := NewMinHeap()

	// Initial candidate
	dist := di.dist(query, current)
	candidates.Push(current, dist)
	visited[current] = true

	// Greedy search (simplified Vamana)
	// In real Vamana, we maintain a list of L closest nodes

	// For this demo, we just walk 10 steps to show disk access
	path := []uint32{current}

	for i := 0; i < 10; i++ {
		node, err := di.readNode(current)
		if err != nil {
			return nil, err
		}

		bestNeighbor := current
		bestDist := dist

		for _, neighbor := range node.Neighbors {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true

			d := di.dist(query, neighbor)
			if d < bestDist {
				bestDist = d
				bestNeighbor = neighbor
			}
		}

		if bestNeighbor == current {
			break // Local minimum
		}
		current = bestNeighbor
		path = append(path, current)
	}

	return path, nil
}

// readNode reads a node from disk or cache
func (di *DiskIndex) readNode(id uint32) (*Node, error) {
	// Check cache
	if val, ok := di.cache.Load(id); ok {
		return val.(*Node), nil
	}

	// Read from disk
	nodeSize := di.config.Dimension*4 + 4 + di.config.MaxDegree*4
	offset := int64(16) + int64(id)*int64(nodeSize)

	buf := make([]byte, nodeSize)
	if _, err := di.file.ReadAt(buf, offset); err != nil {
		return nil, err
	}

	node := &Node{
		ID:        id,
		Vector:    make([]float32, di.config.Dimension),
		Neighbors: make([]uint32, di.config.MaxDegree),
	}

	// Parse Vector
	for j := 0; j < di.config.Dimension; j++ {
		bits := binary.LittleEndian.Uint32(buf[j*4 : (j+1)*4])
		node.Vector[j] = math.Float32frombits(bits)
	}

	// Parse Neighbors
	vecOffset := di.config.Dimension * 4
	// numNeighbors := binary.LittleEndian.Uint32(buf[vecOffset : vecOffset+4])
	neighborOffset := vecOffset + 4

	for j := 0; j < di.config.MaxDegree; j++ {
		node.Neighbors[j] = binary.LittleEndian.Uint32(buf[neighborOffset+j*4 : neighborOffset+(j+1)*4])
	}

	// Update cache
	di.cache.Store(id, node)

	return node, nil
}

// dist computes distance between query and node (reads node vector from disk/cache)
func (di *DiskIndex) dist(query []float32, id uint32) float32 {
	node, err := di.readNode(id)
	if err != nil {
		return math.MaxFloat32
	}
	return index.EuclideanAVX2(query, node.Vector)
}

// Close closes the index file
func (di *DiskIndex) Close() error {
	return di.file.Close()
}
