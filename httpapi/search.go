package httpapi

import (
	"net/http"

	"github.com/khambampati-subhash/govecdb"
)

// searchRequest is the body of POST .../search.
type searchRequest struct {
	// Query is the vector to search near.
	Query []float32 `json:"query"`

	// K is how many results to return.
	K int `json:"k"`

	// Ef is the search width. Omit it — the width is then chosen from the corpus
	// size and the target recall, which is the right default because recall at a
	// fixed width *falls* as a collection grows: a constant that works at ten
	// thousand vectors quietly stops working at a million.
	Ef int `json:"ef,omitempty"`

	// TargetRecall is what an omitted Ef aims for, treated as a floor.
	TargetRecall float64 `json:"target_recall,omitempty"`

	// Filter restricts results by metadata. It is applied *during* the traversal
	// rather than to the results, so a filtered search returns K matches rather
	// than however many of the nearest K happened to match.
	Filter *filterJSON `json:"filter,omitempty"`
}

type matchJSON struct {
	ID       string           `json:"id"`
	Distance float32          `json:"distance"`
	Metadata govecdb.Metadata `json:"metadata,omitempty"`
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req searchRequest
	if err := decode(w, r, s.maxBody, &req); err != nil {
		s.fail(w, r, err)
		return
	}

	search := govecdb.SearchRequest{
		Query:        req.Query,
		K:            req.K,
		Ef:           req.Ef,
		TargetRecall: req.TargetRecall,
	}
	if req.Filter != nil {
		f, err := req.Filter.build(0)
		if err != nil {
			s.fail(w, r, err)
			return
		}
		search.Filter = f
	}

	var matches []govecdb.Match
	err := s.mgr.Use(r.PathValue("name"), func(db *govecdb.DB) error {
		var err error
		matches, err = db.Search(search)
		return err
	})
	if err != nil {
		s.fail(w, r, err)
		return
	}

	out := make([]matchJSON, len(matches))
	for i, m := range matches {
		out[i] = matchJSON{ID: m.ID, Distance: m.Distance, Metadata: m.Metadata}
	}
	s.write(w, r, http.StatusOK, map[string]any{"matches": out})
}
