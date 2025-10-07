package sz_test

import (
	"fmt"
	"strings"
	"testing"

	sz "github.com/ashvardanian/stringzilla/golang"
)

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
	h.Write([]byte("Hello, ")).Write([]byte("world!"))
	if a != h.Digest() {
		t.Fatalf("Streaming digest mismatch: %d != %d", a, h.Digest())
	}

	// Bytesum should be monotonic with appended byte (sanity check)
	if sz.Bytesum("A") >= sz.Bytesum("AB") {
		t.Fatalf("Bytesum not increasing with appended byte")
	}
}

// TestSha256 verifies SHA-256 hashing with NIST test vectors.
func TestSha256(t *testing.T) {
	// NIST test vectors
	empty := sz.Sha256([]byte(""))
	if fmt.Sprintf("%x", empty) != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" {
		t.Fatalf("SHA-256 empty string mismatch")
	}

	abc := sz.Sha256([]byte("abc"))
	if fmt.Sprintf("%x", abc) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" {
		t.Fatalf("SHA-256 'abc' mismatch")
	}
}

// TestSha256Streaming verifies streaming SHA-256 hasher.
func TestSha256Streaming(t *testing.T) {
	hasher := sz.NewSha256()
	hasher.Write([]byte("Hello, ")).Write([]byte("world!"))
	progressive := hasher.Digest()

	oneshot := sz.Sha256([]byte("Hello, world!"))
	if progressive != oneshot {
		t.Fatalf("Streaming SHA-256 mismatch: %x != %x", progressive, oneshot)
	}

	// Hexdigest and reset
	if len(hasher.Hexdigest()) != 64 {
		t.Fatalf("Hexdigest wrong length")
	}
	hasher.Reset()
	hasher.Write([]byte("test"))
	if len(hasher.Digest()) != 32 {
		t.Fatalf("Digest wrong length after reset")
	}
}
