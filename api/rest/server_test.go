package rest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/khambampati-subhash/govecdb/cluster"
	"github.com/khambampati-subhash/govecdb/index"
	"github.com/khambampati-subhash/govecdb/store"
)

func TestServer_Endpoints(t *testing.T) {
	// Setup Node
	walPath := "rest_test_wal.log"
	defer os.Remove(walPath)
	idx, _ := index.NewHNSWIndex(index.DefaultConfig(4))
	s, _ := store.NewStore(walPath, idx)
	defer s.Close()
	node := cluster.NewNode("node1", "localhost:9001", "us-east", "1a", s)

	// Setup Server
	server := NewServer(node, "localhost", 9090)

	// Test Put
	vec := []float32{0.1, 0.2, 0.3, 0.4}
	reqBody, _ := json.Marshal(PutRequest{ID: "vec1", Vector: vec})
	req := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewReader(reqBody))
	w := httptest.NewRecorder()

	server.handleVectors(w, req)

	if w.Code != http.StatusCreated {
		t.Errorf("Expected status 201, got %d", w.Code)
	}

	// Test Get
	req = httptest.NewRequest(http.MethodGet, "/vectors?id=vec1", nil)
	w = httptest.NewRecorder()

	server.handleVectors(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["id"] != "vec1" {
		t.Errorf("Expected id vec1, got %v", resp["id"])
	}

	// Test Search
	searchReq, _ := json.Marshal(SearchRequest{Vector: vec, K: 1})
	req = httptest.NewRequest(http.MethodPost, "/search", bytes.NewReader(searchReq))
	w = httptest.NewRecorder()

	server.handleSearch(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}
