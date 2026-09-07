package httpapi

import (
	"fmt"
	"net/http"
	"time"

	"github.com/khambampati-subhash/govecdb"
	"github.com/khambampati-subhash/govecdb/service"
)

// createRequest is the body of POST /v1/collections.
//
// Durations are strings — "5m", "50ms" — matching the spec file on disk. An
// operator comparing a request against the file it produced should not have to
// convert nanoseconds in their head.
type createRequest struct {
	Name             string  `json:"name"`
	Dimension        int     `json:"dimension"`
	Metric           string  `json:"metric,omitempty"`
	M                int     `json:"m,omitempty"`
	EfConstruction   int     `json:"ef_construction,omitempty"`
	Seed             int64   `json:"seed,omitempty"`
	SyncPolicy       string  `json:"sync_policy,omitempty"`
	SyncInterval     string  `json:"sync_interval,omitempty"`
	SnapshotInterval string  `json:"snapshot_interval,omitempty"`
	SnapshotsKept    int     `json:"snapshots_kept,omitempty"`
	TargetRecall     float64 `json:"target_recall,omitempty"`
}

func (req createRequest) spec() (service.Spec, error) {
	metric, err := service.ParseMetric(req.Metric)
	if err != nil {
		return service.Spec{}, err
	}
	policy, err := service.ParseSyncPolicy(req.SyncPolicy)
	if err != nil {
		return service.Spec{}, err
	}
	syncEvery, err := duration(req.SyncInterval, "sync_interval")
	if err != nil {
		return service.Spec{}, err
	}
	snapshotEvery, err := duration(req.SnapshotInterval, "snapshot_interval")
	if err != nil {
		return service.Spec{}, err
	}
	return service.Spec{
		Dimension:        req.Dimension,
		Metric:           metric,
		M:                req.M,
		EfConstruction:   req.EfConstruction,
		Seed:             req.Seed,
		SyncPolicy:       policy,
		SyncInterval:     syncEvery,
		SnapshotInterval: snapshotEvery,
		SnapshotsKept:    req.SnapshotsKept,
		TargetRecall:     req.TargetRecall,
	}, nil
}

func duration(s, field string) (time.Duration, error) {
	if s == "" {
		return 0, nil
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return 0, fmt.Errorf("%w: %s: %s", govecdb.ErrInvalidRequest, field, err)
	}
	return d, nil
}

// collectionResponse describes a collection. Stats is omitted rather than zeroed
// when the collection is not loaded, because a zero and an unknown are different
// answers and "0 live vectors" is the wrong one.
type collectionResponse struct {
	Name             string         `json:"name"`
	Dimension        int            `json:"dimension"`
	Metric           string         `json:"metric"`
	M                int            `json:"m"`
	EfConstruction   int            `json:"ef_construction"`
	Seed             int64          `json:"seed"`
	SyncPolicy       string         `json:"sync_policy"`
	SyncInterval     string         `json:"sync_interval"`
	SnapshotInterval string         `json:"snapshot_interval,omitempty"`
	SnapshotsKept    int            `json:"snapshots_kept"`
	TargetRecall     float64        `json:"target_recall"`
	Loaded           bool           `json:"loaded"`
	Stats            *statsResponse `json:"stats,omitempty"`
}

type statsResponse struct {
	Live             int     `json:"live"`
	Deleted          int     `json:"deleted"`
	Slots            int     `json:"slots"`
	WithMetadata     int     `json:"with_metadata"`
	DeadRatio        float64 `json:"dead_ratio"`
	LastSequence     uint64  `json:"last_sequence"`
	SnapshotSequence uint64  `json:"snapshot_sequence"`
}

func describe(info service.Info) collectionResponse {
	out := collectionResponse{
		Name:           info.Name,
		Dimension:      info.Spec.Dimension,
		Metric:         info.Spec.Metric.String(),
		M:              info.Spec.M,
		EfConstruction: info.Spec.EfConstruction,
		Seed:           info.Spec.Seed,
		SyncPolicy:     info.Spec.SyncPolicy.String(),
		SyncInterval:   info.Spec.SyncInterval.String(),
		SnapshotsKept:  info.Spec.SnapshotsKept,
		TargetRecall:   info.Spec.TargetRecall,
		Loaded:         info.Loaded,
	}
	if info.Spec.SnapshotInterval > 0 {
		out.SnapshotInterval = info.Spec.SnapshotInterval.String()
	}
	if info.Loaded {
		s := info.Stats
		out.Stats = &statsResponse{
			Live:             s.Live,
			Deleted:          s.Deleted,
			Slots:            s.Slots,
			WithMetadata:     s.WithMetadata,
			DeadRatio:        s.DeadRatio(),
			LastSequence:     s.LastSeq,
			SnapshotSequence: s.SnapshotSeq,
		}
	}
	return out
}

func (s *Server) handleCreateCollection(w http.ResponseWriter, r *http.Request) {
	var req createRequest
	if err := decode(w, r, s.maxBody, &req); err != nil {
		s.fail(w, r, err)
		return
	}
	spec, err := req.spec()
	if err != nil {
		s.fail(w, r, err)
		return
	}
	if err := s.mgr.Create(req.Name, spec); err != nil {
		s.fail(w, r, err)
		return
	}
	info, err := s.mgr.Get(req.Name)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	w.Header().Set("Location", "/v1/collections/"+req.Name)
	s.write(w, r, http.StatusCreated, describe(info))
}

func (s *Server) handleListCollections(w http.ResponseWriter, r *http.Request) {
	infos, err := s.mgr.List()
	if err != nil {
		s.fail(w, r, err)
		return
	}
	out := make([]collectionResponse, len(infos))
	for i, info := range infos {
		out[i] = describe(info)
	}
	// Wrapped in an object rather than returned as a bare array: a top-level
	// array has nowhere to put a cursor when this eventually needs paging, and
	// changing the shape later would break every client at once.
	s.write(w, r, http.StatusOK, map[string]any{"collections": out})
}

func (s *Server) handleGetCollection(w http.ResponseWriter, r *http.Request) {
	info, err := s.mgr.Get(r.PathValue("name"))
	if err != nil {
		s.fail(w, r, err)
		return
	}
	s.write(w, r, http.StatusOK, describe(info))
}

// handleDropCollection deletes a collection and its data.
//
// 200 with a body rather than 204, so the response says what was dropped. A
// destructive call that answers with nothing at all is the one place an empty
// body is least reassuring.
func (s *Server) handleDropCollection(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if err := s.mgr.Drop(name); err != nil {
		s.fail(w, r, err)
		return
	}
	s.write(w, r, http.StatusOK, map[string]any{"dropped": name})
}

func (s *Server) handleSnapshot(w http.ResponseWriter, r *http.Request) {
	var stats govecdb.Stats
	err := s.mgr.Use(r.PathValue("name"), func(db *govecdb.DB) error {
		if err := db.Snapshot(); err != nil {
			return err
		}
		stats = db.Stats()
		return nil
	})
	if err != nil {
		s.fail(w, r, err)
		return
	}
	s.write(w, r, http.StatusOK, map[string]any{"snapshot_sequence": stats.SnapshotSeq})
}

// handleCompact rebuilds the index over its live vectors.
//
// It stops the world for that collection — no search runs while it does — which
// is why it is an explicit call and never a background one. Stats.dead_ratio is
// the signal; around 0.5 is where compacting pays, because the pause tracks
// survivors rather than garbage.
func (s *Server) handleCompact(w http.ResponseWriter, r *http.Request) {
	var reclaimed int
	err := s.mgr.Use(r.PathValue("name"), func(db *govecdb.DB) error {
		var err error
		reclaimed, err = db.Compact()
		return err
	})
	if err != nil {
		s.fail(w, r, err)
		return
	}
	s.write(w, r, http.StatusOK, map[string]any{"reclaimed": reclaimed})
}
