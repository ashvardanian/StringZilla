// Logs the SIMD capabilities the dispatched build detected, before the other tests run.
//
// File: golang/capabilities_test.go
// Author: Ash Vardanian

package sz_test

import (
	"testing"

	sz "github.com/ashvardanian/stringzilla/golang"
)

// TestCapabilities logs the detected CPU features for debugging.
// This test should run first to help diagnose SIMD backend issues.
func TestCapabilities(t *testing.T) {
	caps := sz.Capabilities()
	t.Logf("StringZilla detected capabilities: %s", caps)
	if caps == "" {
		t.Error("No capabilities detected - this may indicate a problem with dynamic dispatch")
	}
}
