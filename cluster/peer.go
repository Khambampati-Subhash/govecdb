package cluster

// PeerInfo holds metadata about a peer node
type PeerInfo struct {
	ID      string `json:"id"`
	Address string `json:"address"`
	Region  string `json:"region"`
	Zone    string `json:"zone"`
}

// DefaultRegion is the default region if none is specified
const DefaultRegion = "default"
