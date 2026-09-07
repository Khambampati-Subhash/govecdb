package httpapi

import (
	"fmt"
	"net/http"

	"github.com/khambampati-subhash/govecdb"
)

// vectorJSON is one record on the wire.
type vectorJSON struct {
	ID       string         `json:"id"`
	Values   []float32      `json:"values"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// addRequest is the body of POST .../vectors.
//
// One shape for one and for many. A single-vector spelling alongside the batch
// would be two code paths through the only endpoint that writes, for the sake of
// two characters in a curl command.
type addRequest struct {
	Vectors []vectorJSON `json:"vectors"`
}

func (s *Server) handleAddVectors(w http.ResponseWriter, r *http.Request) {
	var req addRequest
	if err := decode(w, r, s.maxBody, &req); err != nil {
		s.fail(w, r, err)
		return
	}
	if len(req.Vectors) == 0 {
		s.fail(w, r, fmt.Errorf("%w: no vectors", govecdb.ErrInvalidRequest))
		return
	}

	vs := make([]govecdb.Vector, len(req.Vectors))
	for i, v := range req.Vectors {
		md, err := metadata(v.Metadata)
		if err != nil {
			s.fail(w, r, fmt.Errorf("vector %d: %w", i, err))
			return
		}
		vs[i] = govecdb.Vector{ID: v.ID, Values: v.Values, Metadata: md}
	}

	// AddBatch validates every vector before writing any, so a batch with one bad
	// record leaves the collection untouched. That guarantee is about validation
	// and not about durability: a batch that fails partway through *writing* has
	// durably applied its prefix, which is why the response reports what the call
	// asked for rather than claiming a transaction.
	err := s.mgr.Use(r.PathValue("name"), func(db *govecdb.DB) error {
		return db.AddBatch(vs)
	})
	if err != nil {
		s.fail(w, r, err)
		return
	}
	s.write(w, r, http.StatusOK, map[string]any{"added": len(vs)})
}

// handleGetVector returns one vector.
//
// The values come back in the form the index holds them, which for the cosine
// metric is the unit vector rather than what was sent. That metric is a
// statement that magnitude carries no meaning, and keeping a second copy of every
// embedding to hand back a number nothing uses would double the memory of the
// largest thing in the process.
func (s *Server) handleGetVector(w http.ResponseWriter, r *http.Request) {
	var v govecdb.Vector
	err := s.mgr.Use(r.PathValue("name"), func(db *govecdb.DB) error {
		var err error
		v, err = db.Get(r.PathValue("id"))
		return err
	})
	if err != nil {
		s.fail(w, r, err)
		return
	}
	s.write(w, r, http.StatusOK, vectorJSON{ID: v.ID, Values: v.Values, Metadata: v.Metadata})
}

// handleDeleteVector removes a vector.
//
// Deleting an id that is not there succeeds, matching the database: replay
// applies records more than once across a snapshot boundary, so an operation
// that failed the second time would make recovery order-sensitive. The response
// says nothing about whether anything was there, because the database does not
// either — and a count that was sometimes a lie is worse than no count.
func (s *Server) handleDeleteVector(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	err := s.mgr.Use(r.PathValue("name"), func(db *govecdb.DB) error {
		return db.Delete(id)
	})
	if err != nil {
		s.fail(w, r, err)
		return
	}
	s.write(w, r, http.StatusOK, map[string]any{"deleted": id})
}
