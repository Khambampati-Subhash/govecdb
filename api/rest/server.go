package rest

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/khambampati-subhash/govecdb/cluster"
)

// Server represents the REST API server
type Server struct {
	node *cluster.Node
	host string
	port int
}

// NewServer creates a new REST server
func NewServer(node *cluster.Node, host string, port int) *Server {
	return &Server{
		node: node,
		host: host,
		port: port,
	}
}

// Start starts the HTTP server
func (s *Server) Start() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/vectors", s.handleVectors)
	mux.HandleFunc("/search", s.handleSearch)
	mux.HandleFunc("/health", s.handleHealth)

	addr := fmt.Sprintf("%s:%d", s.host, s.port)
	log.Printf("REST API listening on %s", addr)

	server := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	return server.ListenAndServe()
}

func (s *Server) handleVectors(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		s.handlePut(w, r)
	case http.MethodGet:
		s.handleGet(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

type PutRequest struct {
	ID     string    `json:"id"`
	Vector []float32 `json:"vector"`
}

func (s *Server) handlePut(w http.ResponseWriter, r *http.Request) {
	var req PutRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.ID == "" || len(req.Vector) == 0 {
		http.Error(w, "Invalid request: missing id or vector", http.StatusBadRequest)
		return
	}

	if err := s.node.Put(req.ID, req.Vector); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"status": "created", "id": req.ID})
}

func (s *Server) handleGet(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if id == "" {
		http.Error(w, "Missing id parameter", http.StatusBadRequest)
		return
	}

	vec, err := s.node.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":     id,
		"vector": vec,
	})
}

type SearchRequest struct {
	Vector []float32 `json:"vector"`
	K      int       `json:"k"`
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Vector) == 0 || req.K <= 0 {
		http.Error(w, "Invalid request: missing vector or invalid k", http.StatusBadRequest)
		return
	}

	// TODO: Implement distributed search or local search
	// For now, just local search on the node's store (which might be partial)
	// In a real distributed system, we need scatter-gather.
	// But Node.Store.Index.Search is local.

	// Let's assume for now we search locally on this node.
	// Ideally, we should forward to all shards and aggregate.
	// But sharding is by key, so vectors are distributed.
	// A search query needs to go to ALL shards.

	// For this MVP, let's just expose local search and note the limitation.
	results, err := s.node.Store.Index.Search(req.Vector, req.K)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(results)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}
