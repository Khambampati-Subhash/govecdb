package client

import (
	"context"
	"fmt"

	pb "github.com/khambampati-subhash/govecdb/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client is the Go client for the vector database
type Client struct {
	conn   *grpc.ClientConn
	client pb.VectorServiceClient
}

// NewClient creates a new client connected to the specified address
func NewClient(address string) (*Client, error) {
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect: %w", err)
	}

	return &Client{
		conn:   conn,
		client: pb.NewVectorServiceClient(conn),
	}, nil
}

// Close closes the client connection
func (c *Client) Close() error {
	return c.conn.Close()
}

// Put inserts a vector into the database
func (c *Client) Put(ctx context.Context, id string, vector []float32) error {
	resp, err := c.client.Put(ctx, &pb.PutRequest{
		Vector: &pb.Vector{
			Id:   id,
			Data: vector,
		},
	})
	if err != nil {
		return err
	}
	if !resp.Success {
		return fmt.Errorf("put failed: %s", resp.Error)
	}
	return nil
}

// Get retrieves a vector from the database
func (c *Client) Get(ctx context.Context, id string) ([]float32, error) {
	resp, err := c.client.Get(ctx, &pb.GetRequest{Id: id})
	if err != nil {
		return nil, err
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("get failed: %s", resp.Error)
	}
	if resp.Vector == nil {
		return nil, fmt.Errorf("vector not found")
	}
	return resp.Vector.Data, nil
}

// Search finds the nearest neighbors for a given vector
func (c *Client) Search(ctx context.Context, vector []float32, k int) ([]SearchResult, error) {
	resp, err := c.client.Search(ctx, &pb.SearchRequest{
		Vector: vector,
		K:      int32(k),
	})
	if err != nil {
		return nil, err
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("search failed: %s", resp.Error)
	}

	results := make([]SearchResult, len(resp.Results))
	for i, r := range resp.Results {
		results[i] = SearchResult{
			ID:    r.Id,
			Score: r.Score,
		}
	}
	return results, nil
}

// SearchResult represents a search result
type SearchResult struct {
	ID    string
	Score float32
}
