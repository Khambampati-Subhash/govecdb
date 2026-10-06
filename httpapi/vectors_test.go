package httpapi

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/khambampati-subhash/govecdb"
	"github.com/khambampati-subhash/govecdb/service"
)

func (a *api) addN(name string, n int) {
	a.t.Helper()
	vs := make([]map[string]any, n)
	for i := range vs {
		vs[i] = map[string]any{"id": fmt.Sprintf("v%02d", i), "values": []float32{1, float32(i), 0, 0}}
	}
	a.expect(a.do("POST", "/v1/collections/"+name+"/vectors", map[string]any{"vectors": vs}), http.StatusOK)
}

func TestEmptyBatchIsANoOp(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	body := a.expect(a.do("POST", "/v1/collections/docs/vectors", `{"vectors":[]}`), http.StatusOK)
	if body["added"] != float64(0) {
		t.Fatalf("added = %v, want 0", body["added"])
	}
	// Still resolved: a no-op against a collection that does not exist is a
	// typo the client should hear about.
	a.expectError(a.do("POST", "/v1/collections/nope/vectors", `{"vectors":[]}`),
		http.StatusNotFound, codeNotFound)
}

func TestNotFoundSaysWhatIsMissing(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)

	for _, tc := range []struct{ path, resource string }{
		{"/v1/collections/nope/vectors/a", resourceCollection},
		{"/v1/collections/docs/vectors/a", resourceVector},
		{"/v1/collections/nope", resourceCollection},
	} {
		rec := a.do("GET", tc.path, nil)
		a.expectError(rec, http.StatusNotFound, codeNotFound)
		detail := a.expect(rec, http.StatusNotFound)["error"].(map[string]any)
		if detail["resource"] != tc.resource {
			t.Errorf("GET %s: resource = %v, want %q", tc.path, detail["resource"], tc.resource)
		}
	}
}

// TestIntegralFloatsKeepTheirType is the round trip that used to change types:
// a float64(1) went out as `1` and came back in as an int64.
func TestIntegralFloatsKeepTheirType(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)
	a.expect(a.do("POST", "/v1/collections/docs/vectors", `{"vectors":[{"id":"a","values":[1,0,0,0],
		"metadata":{"score":1.0,"page":1,"neg":-2.0,"half":0.5,"huge":1e300,"tag":"x"}}]}`), http.StatusOK)

	raw := a.do("GET", "/v1/collections/docs/vectors/a", nil).Body.String()
	for _, want := range []string{`"score":1.0`, `"neg":-2.0`, `"page":1,`, `"half":0.5`, `"huge":1e+300`} {
		if !strings.Contains(raw, want) {
			t.Errorf("response %s does not contain %s", raw, want)
		}
	}

	// Send back exactly what came out, and the stored types must not move.
	body := `{"vectors":[` + strings.TrimSpace(raw) + `]}`
	a.expect(a.do("POST", "/v1/collections/docs/vectors", body), http.StatusOK)

	err := a.server.mgr.Use("docs", func(db *govecdb.DB) error {
		v, err := db.Get("a")
		if err != nil {
			return err
		}
		if _, ok := v.Metadata["score"].(float64); !ok {
			t.Errorf("score came back as %T", v.Metadata["score"])
		}
		if _, ok := v.Metadata["page"].(int64); !ok {
			t.Errorf("page came back as %T", v.Metadata["page"])
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestGetVectorsInOneRequest(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)
	a.addN("docs", 5)

	body := a.expect(a.do("POST", "/v1/collections/docs/vectors/get",
		`{"ids":["v03","gone","v01","also-gone"]}`), http.StatusOK)
	vs := body["vectors"].([]any)
	if len(vs) != 2 || vs[0].(map[string]any)["id"] != "v03" || vs[1].(map[string]any)["id"] != "v01" {
		t.Fatalf("vectors = %v", vs)
	}
	if m := fmt.Sprint(body["missing"]); m != "[gone also-gone]" {
		t.Fatalf("missing = %s", m)
	}

	// None missing is an empty list, not an absent key.
	body = a.expect(a.do("POST", "/v1/collections/docs/vectors/get", `{"ids":["v00"]}`), http.StatusOK)
	if m, ok := body["missing"].([]any); !ok || len(m) != 0 {
		t.Fatalf("missing = %#v", body["missing"])
	}
}

func TestListVectorsPages(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)
	a.addN("docs", 25)

	var seen []string
	path := "/v1/collections/docs/vectors?limit=10"
	for pages := 0; ; pages++ {
		if pages > 5 {
			t.Fatal("paging never ended")
		}
		body := a.expect(a.do("GET", path, nil), http.StatusOK)
		for _, v := range body["vectors"].([]any) {
			seen = append(seen, v.(map[string]any)["id"].(string))
		}
		next, ok := body["next"].(string)
		if !ok {
			break
		}
		path = "/v1/collections/docs/vectors?limit=10&after=" + next
	}
	if len(seen) != 25 || seen[0] != "v00" || seen[24] != "v24" {
		t.Fatalf("paged through %d: %v", len(seen), seen)
	}

	for _, bad := range []string{"0", "-1", "1001", "ten"} {
		a.expectError(a.do("GET", "/v1/collections/docs/vectors?limit="+bad, nil),
			http.StatusBadRequest, codeInvalidRequest)
	}
}

func TestSyncEndpoint(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)
	a.addN("docs", 3)

	body := a.expect(a.do("POST", "/v1/collections/docs/sync", nil), http.StatusOK)
	if body["last_sequence"] != float64(3) {
		t.Fatalf("last_sequence = %v, want 3 (one record per vector)", body["last_sequence"])
	}
	a.expectError(a.do("POST", "/v1/collections/nope/sync", nil), http.StatusNotFound, codeNotFound)
}

// TestGetCollectionCanLoad: describing never loads by default, and ?load=true
// is the way to get stats for a cold collection without a throwaway search.
func TestGetCollectionCanLoad(t *testing.T) {
	root := t.TempDir()
	first, err := service.NewManager(root, service.Options{})
	if err != nil {
		t.Fatal(err)
	}
	a := newAPI(t, Config{Manager: first})
	a.createCollection("docs", 4)
	a.addN("docs", 7)
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}

	cold, err := service.NewManager(root, service.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { cold.Close() })
	a = newAPI(t, Config{Manager: cold})

	body := a.expect(a.do("GET", "/v1/collections/docs", nil), http.StatusOK)
	if body["loaded"] != false || body["stats"] != nil {
		t.Fatalf("a plain GET loaded the collection: %v", body)
	}
	body = a.expect(a.do("GET", "/v1/collections/docs?load=true", nil), http.StatusOK)
	stats, _ := body["stats"].(map[string]any)
	if body["loaded"] != true || stats["live"] != float64(7) {
		t.Fatalf("?load=true = %v", body)
	}
	a.expectError(a.do("GET", "/v1/collections/docs?load=maybe", nil), http.StatusBadRequest, codeInvalidRequest)
}
