package httpapi

import (
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"testing"

	"github.com/khambampati-subhash/govecdb/service"
)

func TestAnUnavailableCollectionIs503(t *testing.T) {
	status, code, msg := classify(fmt.Errorf("%w: %q: open /var/lib/x: no space left", service.ErrUnavailable, "docs"))
	if status != http.StatusServiceUnavailable || code != codeUnavailable {
		t.Fatalf("classify = %d %s, want 503 %s", status, code, codeUnavailable)
	}
	if strings.Contains(msg, "/var/lib") {
		t.Fatalf("message %q leaks the path", msg)
	}
}

// A batch get is a response built in memory, so it has the page's bound.
func TestGetBatchIsCappedAtThePageLimit(t *testing.T) {
	a := newAPI(t, Config{})
	a.createCollection("docs", 4)
	ids := func(n int) []string {
		out := make([]string, n)
		for i := range out {
			out[i] = "x" // duplicates count: each is encoded
		}
		return out
	}
	a.expect(a.do("POST", "/v1/collections/docs/vectors/get", map[string]any{"ids": ids(maxPageLimit)}),
		http.StatusOK)
	a.expectError(a.do("POST", "/v1/collections/docs/vectors/get", map[string]any{"ids": ids(maxPageLimit + 1)}),
		http.StatusBadRequest, codeInvalidRequest)
}

func TestMetricsIncludeTheRuntimeAndAreOrdered(t *testing.T) {
	a := newAPI(t, Config{})
	body := a.do("GET", "/metrics", nil).Body.String()
	for _, name := range []string{"go_goroutines", "go_memory_heap_objects_bytes", "go_memory_total_bytes",
		"go_gc_heap_goal_bytes", "go_gc_cycles_total"} {
		if !regexp.MustCompile(`(?m)^` + name + ` \d+$`).MatchString(body) {
			t.Errorf("metrics missing %s", name)
		}
	}
	classes := regexp.MustCompile(`govecdb_http_requests_total\{class="(\w+)"\}`).FindAllStringSubmatch(body, -1)
	var got []string
	for _, c := range classes {
		got = append(got, c[1])
	}
	if want := "other 1xx 2xx 3xx 4xx 5xx"; strings.Join(got, " ") != want {
		t.Fatalf("status classes in order %v, want %s", got, want)
	}
}
