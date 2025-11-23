package main

import (
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/khambampati-subhash/govecdb/api/rest"
	"github.com/khambampati-subhash/govecdb/cluster"
	"github.com/khambampati-subhash/govecdb/index"
	"github.com/khambampati-subhash/govecdb/store"
)

var (
	port      = flag.Int("port", 8080, "HTTP port")
	raftPort  = flag.Int("raft_port", 9090, "Raft port")
	nodeID    = flag.String("id", "node1", "Node ID")
	region    = flag.String("region", "us-east", "Region")
	zone      = flag.String("zone", "1a", "Zone")
	walPath   = flag.String("wal", "wal.log", "Path to WAL file")
	peers     = flag.String("peers", "", "Comma-separated list of peers (id=addr=region=zone)")
	bootstrap = flag.Bool("bootstrap", false, "Bootstrap the cluster")
	host      = flag.String("host", "localhost", "Bind host address")
	dim       = flag.Int("dim", 128, "Vector dimension")
)

// Get handles get requests
type VectorRequest struct {
	ID     string    `json:"id"`
	Vector []float32 `json:"vector"`
}

func main() {
	flag.Parse()

	// 1. Initialize Store
	cfg := index.DefaultConfig(*dim)
	idx, err := index.NewHNSWIndex(cfg)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}

	s, err := store.NewStore(*walPath, idx)
	if err != nil {
		log.Fatalf("Failed to create store: %v", err)
	}
	defer s.Close()

	// 2. Initialize Node
	addr := fmt.Sprintf("%s:%d", *host, *port)
	node := cluster.NewNode(*nodeID, addr, *region, *zone, s)

	// Setup Raft
	if err := node.SetupRaft(*raftPort, *bootstrap); err != nil {
		log.Fatalf("Failed to setup Raft: %v", err)
	}

	// Start gRPC server
	if err := node.Start(); err != nil {
		log.Fatalf("Failed to start node: %v", err)
	}
	defer node.Stop()

	// Join peers
	if *peers != "" {
		peerList := strings.Split(*peers, ",")
		for _, p := range peerList {
			parts := strings.Split(p, "=")
			if len(parts) == 4 {
				node.Join(parts[0], parts[1], parts[2], parts[3])
				fmt.Printf("Joined peer %s at %s (%s/%s)\n", parts[0], parts[1], parts[2], parts[3])
			}
		}
	}

	// 3. Start REST API
	httpPort := *port + 1
	restServer := rest.NewServer(node, *host, httpPort)
	log.Fatal(restServer.Start())
}
