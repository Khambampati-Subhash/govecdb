package httpapi

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/khambampati-subhash/govecdb"
)

// vectorJSON is one record on its way in.
type vectorJSON struct {
	ID       string         `json:"id"`
	Values   float32s       `json:"values"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Records on their way out are written by appendVector (encode.go), because
// metadata is encoded differently in each direction — see appendMetadata.

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
	// An empty batch is a no-op, as it is in the library. Refusing it here made
	// the two deployments disagree about the same call, and a client batching
	// whatever a document produced had to special-case the document that
	// produced nothing. The collection is still resolved, so a typo in its name
	// is a 404 rather than a silent success.
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
	s.writeEncoded(w, r, http.StatusOK, func(b []byte) ([]byte, error) { return appendVector(b, v) })
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

// getBatchRequest is the body of POST .../vectors/get.
type getBatchRequest struct {
	IDs []string `json:"ids"`
}

// handleGetVectors reads many vectors in one request.
//
// A POST, because the ids are a body and not a query string: a thousand ids do
// not fit in a URL any proxy will pass. Absent ids are listed in "missing"
// rather than failing the call — in a batch, one stale id is not a reason to
// withhold the rest. The batch is read under one lock, so it is a consistent
// picture of the collection.
func (s *Server) handleGetVectors(w http.ResponseWriter, r *http.Request) {
	var req getBatchRequest
	if err := decode(w, r, s.maxBody, &req); err != nil {
		s.fail(w, r, err)
		return
	}
	var vs []govecdb.Vector
	err := s.mgr.Use(r.PathValue("name"), func(db *govecdb.DB) error {
		var err error
		vs, err = db.GetBatch(req.IDs)
		return err
	})
	if err != nil {
		s.fail(w, r, err)
		return
	}

	// GetBatch keeps request order and drops the absent, so one forward walk
	// recovers which ones those were.
	missing := []string{}
	j := 0
	for _, id := range req.IDs {
		if j < len(vs) && vs[j].ID == id {
			j++
			continue
		}
		missing = append(missing, id)
	}
	s.writeEncoded(w, r, http.StatusOK, func(b []byte) ([]byte, error) { return encodeBatch(b, vs, missing) })
}

// Page sizes for GET .../vectors. The cap is about the response, not the
// database: a page of a thousand 1,536-dimension vectors is already ~15 MB of
// JSON, and nothing else bounds how much one response body can make this
// process buffer.
const (
	defaultPageLimit = 100
	maxPageLimit     = 1000
)

// handleListVectors pages through a collection in id order.
//
// The cursor is the last id of the previous page, passed back as ?after=, and
// "next" is omitted on the last page. Pages are weakly consistent: each is one
// consistent moment, but a vector written between two requests may or may not
// appear, which is the most a cursor held by a client between requests can
// honestly promise.
func (s *Server) handleListVectors(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit := defaultPageLimit
	if v := q.Get("limit"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 || n > maxPageLimit {
			s.fail(w, r, fmt.Errorf("%w: limit %q, want 1..%d", govecdb.ErrInvalidRequest, v, maxPageLimit))
			return
		}
		limit = n
	}
	after := q.Get("after")

	var vs []govecdb.Vector
	err := s.mgr.Use(r.PathValue("name"), func(db *govecdb.DB) error {
		var err error
		vs, err = db.Scan(after, limit)
		return err
	})
	if err != nil {
		s.fail(w, r, err)
		return
	}
	more := len(vs) == limit
	next := ""
	if more {
		next = vs[len(vs)-1].ID
	}
	s.writeEncoded(w, r, http.StatusOK, func(b []byte) ([]byte, error) { return encodePage(b, vs, next, more) })
}
