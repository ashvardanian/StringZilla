// Byte sums and seeded 64-bit hashes, one-shot against streaming.
//
// File: golang/hash_test.go
// Author: Ash Vardanian

package sz_test

import (
	"testing"

	sz "github.com/ashvardanian/stringzilla/golang"
)

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

	// Test [Hasher.Sum]
	sum := h.Sum(nil)
	if len(sum) != 8 {
		t.Fatalf("Sum(nil) should return 8 bytes")
	}

	// Test [Hasher.Reset]
	h.Reset()
	h.Write([]byte("test"))
	if h.Sum64() == a {
		t.Fatalf("After reset, hash should be different")
	}

	// A sanity check: Bytesum should be monotonic with an appended byte
	if sz.Bytesum("A") >= sz.Bytesum("AB") {
		t.Fatalf("Bytesum not increasing with appended byte")
	}
}
