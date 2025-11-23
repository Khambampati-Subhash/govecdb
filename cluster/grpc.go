package cluster

import (
	"context"

	pb "github.com/khambampati-subhash/govecdb/proto"
)

// GRPCServer implements the VectorServiceServer interface
type GRPCServer struct {
	pb.UnimplementedVectorServiceServer
	node *Node
}

// NewGRPCServer creates a new gRPC server
func NewGRPCServer(node *Node) *GRPCServer {
	return &GRPCServer{node: node}
}

// Put handles insert requests
func (s *GRPCServer) Put(ctx context.Context, req *pb.PutRequest) (*pb.PutResponse, error) {
	if req.Vector == nil {
		return &pb.PutResponse{Success: false, Error: "vector is nil"}, nil
	}

	// If this is a replication request, write directly to store
	if req.IsReplication {
		err := s.node.Store.Insert(req.Vector.Id, req.Vector.Data)
		if err != nil {
			return &pb.PutResponse{Success: false, Error: err.Error()}, nil
		}
		return &pb.PutResponse{Success: true}, nil
	}

	// Otherwise, use Node.Put which handles routing and replication coordination
	err := s.node.Put(req.Vector.Id, req.Vector.Data)
	if err != nil {
		return &pb.PutResponse{Success: false, Error: err.Error()}, nil
	}

	return &pb.PutResponse{Success: true}, nil
}

// BatchPut handles batch insert requests
func (s *GRPCServer) BatchPut(ctx context.Context, req *pb.BatchPutRequest) (*pb.BatchPutResponse, error) {
	if len(req.Vectors) == 0 {
		return &pb.BatchPutResponse{Success: true, Count: 0}, nil
	}

	// Call Node.BatchPut
	err := s.node.BatchPut(req.Vectors)
	if err != nil {
		return &pb.BatchPutResponse{Success: false, Error: err.Error()}, nil
	}

	return &pb.BatchPutResponse{Success: true, Count: int32(len(req.Vectors))}, nil
}

// Get handles retrieval requests
func (s *GRPCServer) Get(ctx context.Context, req *pb.GetRequest) (*pb.GetResponse, error) {
	// Directly get from local store
	vec, err := s.node.Store.Index.Get(req.Id)
	if err != nil {
		return &pb.GetResponse{Error: err.Error()}, nil
	}

	return &pb.GetResponse{
		Vector: &pb.Vector{
			Id:   vec.ID,
			Data: vec.Data,
		},
	}, nil
}

// Search handles search requests
func (s *GRPCServer) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	results, err := s.node.Store.Index.Search(req.Vector, int(req.K))
	if err != nil {
		return &pb.SearchResponse{Error: err.Error()}, nil
	}

	pbResults := make([]*pb.SearchResult, len(results))
	for i, r := range results {
		pbResults[i] = &pb.SearchResult{
			Id:    r.ID,
			Score: r.Score,
		}
	}

	return &pb.SearchResponse{Results: pbResults}, nil
}
