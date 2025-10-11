//go:build !no_sha256
// +build !no_sha256

package sz_test

import (
	"fmt"
	"testing"

	sz "github.com/ashvardanian/stringzilla/golang"
)

// TestSha256 verifies SHA-256 hashing with NIST test vectors.
// This test is skipped when building with: go test -tags no_sha256
func TestSha256(t *testing.T) {
	// NIST test vectors
	empty := sz.HashSha256([]byte(""))
	if fmt.Sprintf("%x", empty) != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" {
		t.Fatalf("SHA-256 empty string mismatch")
	}

	abc := sz.HashSha256([]byte("abc"))
	if fmt.Sprintf("%x", abc) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" {
		t.Fatalf("SHA-256 'abc' mismatch")
	}
}

// TestSha256Streaming verifies streaming SHA-256 hasher.
// This test is skipped when building with: go test -tags no_sha256
func TestSha256Streaming(t *testing.T) {
	hasher := sz.NewSha256()
	hasher.Write([]byte("Hello, "))
	hasher.Write([]byte("world!"))
	progressive := hasher.Digest()

	oneshot := sz.HashSha256([]byte("Hello, world!"))
	if progressive != oneshot {
		t.Fatalf("Streaming SHA-256 mismatch: %x != %x", progressive, oneshot)
	}

	// Test hash.Hash interface compliance
	if hasher.Size() != 32 {
		t.Fatalf("Size() should return 32")
	}
	if hasher.BlockSize() != 64 {
		t.Fatalf("BlockSize() should return 64")
	}

	// Test Sum() method
	sum := hasher.Sum(nil)
	if len(sum) != 32 {
		t.Fatalf("Sum(nil) should return 32 bytes")
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
