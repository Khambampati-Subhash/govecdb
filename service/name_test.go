package service

import (
	"errors"
	"strings"
	"testing"
)

func TestValidateNameAccepts(t *testing.T) {
	for _, name := range []string{
		"a",
		"docs",
		"docs-v2",
		"docs_v2",
		"D0cs",
		"0",
		strings.Repeat("x", MaxNameBytes),
	} {
		if err := ValidateName(name); err != nil {
			t.Errorf("ValidateName(%q) = %v, want nil", name, err)
		}
	}
}

// The rejected set is the security boundary, so it is spelled out rather than
// summarized: every entry here is a way a name could have escaped its own
// directory, or compared unequal to itself on another filesystem.
func TestValidateNameRejects(t *testing.T) {
	for _, tc := range []struct{ name, why string }{
		{"", "empty"},
		{strings.Repeat("x", MaxNameBytes+1), "one byte over the cap"},
		{".", "the current directory"},
		{"..", "the parent directory"},
		{"../etc", "traversal"},
		{"a/b", "a separator"},
		{`a\b`, "a Windows separator"},
		{"a.b", "a dot, which is what makes .. impossible to spell"},
		{"-lead", "a leading dash, which every CLI reads as a flag"},
		{"_lead", "a leading underscore"},
		{"a b", "a space"},
		{"a\x00b", "a NUL some kernels truncate at"},
		{"a\nb", "a newline that would forge a log line"},
		{"café", "non-ASCII, which macOS stores in NFD and Linux does not"},
		{"a:b", "a drive separator"},
		{"a*b", "a glob"},
	} {
		err := ValidateName(tc.name)
		if err == nil {
			t.Errorf("ValidateName(%q) = nil, want an error (%s)", tc.name, tc.why)
			continue
		}
		if !errors.Is(err, ErrInvalidName) {
			t.Errorf("ValidateName(%q) = %v, want ErrInvalidName", tc.name, err)
		}
	}
}
