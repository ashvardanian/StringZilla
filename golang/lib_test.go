package sz_test

import (
	"strings"
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

// TestContains verifies that the Contains function behaves as expected.
func TestContains(t *testing.T) {
	tests := []struct {
		s, substr string
		want      bool
	}{
		{"test", "s", true},
		{"test", "test", true},
		{"test", "zest", false},
		{"test", "z", false},
	}

	for _, tt := range tests {
		if got := sz.Contains(tt.s, tt.substr); got != tt.want {
			t.Errorf("Contains(%q, %q) = %v, want %v", tt.s, tt.substr, got, tt.want)
		}
	}
}

// TestIndex compares our binding's Index against the standard strings.Index.
func TestIndex(t *testing.T) {
	// We'll use both a long string and some simple cases.
	longStr := strings.Repeat("0123456789", 100000) + "something"
	tests := []struct {
		s, substr string
	}{
		{longStr, "some"},
		{"test", ""},
		{"test", "t"},
		{"test", "s"},
		{"test", "z"},
	}

	for _, tt := range tests {
		std := strings.Index(tt.s, tt.substr)
		got := int(sz.Index(tt.s, tt.substr))
		if got != std {
			t.Errorf("Index(%q, %q) = %d, want %d", tt.s, tt.substr, got, std)
		}
	}
}

// TestLastIndex compares our binding's LastIndex against the standard strings.LastIndex.
func TestLastIndex(t *testing.T) {
	tests := []struct {
		s, substr string
	}{
		{"test", "t"},
		{"test", "s"},
		{"test", ""},
		{"test", "z"},
	}

	for _, tt := range tests {
		std := strings.LastIndex(tt.s, tt.substr)
		got := int(sz.LastIndex(tt.s, tt.substr))
		if got != std {
			t.Errorf("LastIndex(%q, %q) = %d, want %d", tt.s, tt.substr, got, std)
		}
	}
}

// TestIndexByte compares our binding's IndexByte against the standard strings.IndexByte.
func TestIndexByte(t *testing.T) {
	tests := []struct {
		s string
		c byte
	}{
		{"test", 't'},
		{"test", 's'},
		{"test", 'z'},
	}

	for _, tt := range tests {
		std := strings.IndexByte(tt.s, tt.c)
		got := int(sz.IndexByte(tt.s, tt.c))
		if got != std {
			t.Errorf("IndexByte(%q, %q) = %d, want %d", tt.s, string(tt.c), got, std)
		}
	}
}

// TestLastIndexByte compares our binding's LastIndexByte against the standard strings.LastIndexByte.
func TestLastIndexByte(t *testing.T) {
	tests := []struct {
		s string
		c byte
	}{
		{"test", 't'},
		{"test", 's'},
		{"test", 'z'},
	}

	for _, tt := range tests {
		std := strings.LastIndexByte(tt.s, tt.c)
		got := int(sz.LastIndexByte(tt.s, tt.c))
		if got != std {
			t.Errorf("LastIndexByte(%q, %q) = %d, want %d", tt.s, string(tt.c), got, std)
		}
	}
}

// TestIndexAny compares our binding's IndexAny against the standard strings.IndexAny.
func TestIndexAny(t *testing.T) {
	tests := []struct {
		s, charset string
	}{
		{"test", "st"},
		{"west east", "ta"},
		{"test", "z"},
	}

	for _, tt := range tests {
		std := strings.IndexAny(tt.s, tt.charset)
		got := int(sz.IndexAny(tt.s, tt.charset))
		if got != std {
			t.Errorf("IndexAny(%q, %q) = %d, want %d", tt.s, tt.charset, got, std)
		}
	}
}

// TestLastIndexAny compares our binding's LastIndexAny against the standard strings.LastIndexAny.
func TestLastIndexAny(t *testing.T) {
	tests := []struct {
		s, charset string
	}{
		{"test", "st"},
		{"west east", "ta"},
		{"test", "z"},
	}

	for _, tt := range tests {
		std := strings.LastIndexAny(tt.s, tt.charset)
		got := int(sz.LastIndexAny(tt.s, tt.charset))
		if got != std {
			t.Errorf("LastIndexAny(%q, %q) = %d, want %d", tt.s, tt.charset, got, std)
		}
	}
}

// TestCount verifies the Count function for overlapping and non-overlapping cases.
func TestCount(t *testing.T) {
	tests := []struct {
		s, substr string
		overlap   bool
		want      int
	}{
		{"aaaaa", "a", false, 5},
		{"aaaaa", "aa", false, 2},
		{"aaaaa", "aa", true, 4},
		{"", "", false, 1},    // empty substring counts as len("") + 1
		{"", "", true, 1},     // overlap flag doesn't affect empty-substring semantics
		{"abc", "", false, 4}, // empty substring counts as len("abc") + 1
		{"", "a", false, 0},   // non-empty needle in empty haystack
	}

	for _, tt := range tests {
		got := int(sz.Count(tt.s, tt.substr, tt.overlap))
		if got != tt.want {
			t.Errorf("Count(%q, %q, %v) = %d, want %d", tt.s, tt.substr, tt.overlap, got, tt.want)
		}
	}
}

// TestHashing verifies hashing and streaming API properties.
func TestHashing(t *testing.T) {
	// Deterministic and seed-sensitive
	a := sz.Hash("Hello, world!", 42)
	b := sz.Hash("Hello, world!", 42)
	c := sz.Hash("Hello, world!", 43)
	if a != b {
		t.Fatalf("Hash not deterministic: %d != %d", a, b)
	}
	if a == c {
		t.Fatalf("Different seeds should yield different hashes: %d == %d", a, c)
	}

	// Streaming equals one-shot
	h := sz.NewHasher(42)
	h.Write([]byte("Hello, "))
	h.Write([]byte("world!"))
	if a != h.Digest() {
		t.Fatalf("Streaming digest mismatch: %d != %d", a, h.Digest())
	}

	// Test hash.Hash64 interface compliance
	if h.Size() != 8 {
		t.Fatalf("Size() should return 8")
	}
	if h.Sum64() != a {
		t.Fatalf("Sum64() mismatch: %d != %d", h.Sum64(), a)
	}

	// Test Sum() method
	sum := h.Sum(nil)
	if len(sum) != 8 {
		t.Fatalf("Sum(nil) should return 8 bytes")
	}

	// Test Reset()
	h.Reset()
	h.Write([]byte("test"))
	if h.Sum64() == a {
		t.Fatalf("After reset, hash should be different")
	}

	// Bytesum should be monotonic with appended byte (sanity check)
	if sz.Bytesum("A") >= sz.Bytesum("AB") {
		t.Fatalf("Bytesum not increasing with appended byte")
	}
}

func TestUtf8CaseFold(t *testing.T) {
	folded, err := sz.Utf8CaseFold("Straße", true)
	if err != nil {
		t.Fatalf("Utf8CaseFold returned error: %v", err)
	}
	if folded != "strasse" {
		t.Fatalf("Utf8CaseFold(\"Straße\") = %q, want %q", folded, "strasse")
	}
}

func TestUtf8CaseInsensitiveFind(t *testing.T) {
	haystack := "Die Temperaturschwankungen im kosmischen Mikrowellenhintergrund sind ein Maß von etwa 20 µK.\n" +
		"Typografisch sieht man auch: ein Maß von etwa 20 μK."
	needle := "EIN MASS VON ETWA 20 μK"

	firstIndex, firstLength, err := sz.Utf8CaseInsensitiveFind(haystack, needle, true)
	if err != nil {
		t.Fatalf("Utf8CaseInsensitiveFind returned error: %v", err)
	}
	if firstIndex < 0 || firstLength <= 0 {
		t.Fatalf("Utf8CaseInsensitiveFind failed: index=%d length=%d", firstIndex, firstLength)
	}
	firstMatch := haystack[firstIndex : firstIndex+firstLength]
	if firstMatch != "ein Maß von etwa 20 µK" {
		t.Fatalf("first match = %q, want %q", firstMatch, "ein Maß von etwa 20 µK")
	}

	compiledNeedle, err := sz.NewUtf8CaseInsensitiveNeedle(needle, true)
	if err != nil {
		t.Fatalf("NewUtf8CaseInsensitiveNeedle returned error: %v", err)
	}

	remainingHaystack := haystack[firstIndex+firstLength:]
	secondIndex, secondLength, err := compiledNeedle.FindIn(remainingHaystack, true)
	if err != nil {
		t.Fatalf("Utf8CaseInsensitiveNeedle.FindIn returned error: %v", err)
	}
	if secondIndex < 0 || secondLength <= 0 {
		t.Fatalf("Utf8CaseInsensitiveNeedle.FindIn failed: index=%d length=%d", secondIndex, secondLength)
	}
	secondMatch := remainingHaystack[secondIndex : secondIndex+secondLength]
	if secondMatch != "ein Maß von etwa 20 μK" {
		t.Fatalf("second match = %q, want %q", secondMatch, "ein Maß von etwa 20 μK")
	}
}
