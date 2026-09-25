// UTF-8 counting, normalization and case-insensitive search, with their benchmarks.
//
// File: golang/utf8_test.go
// Author: Ash Vardanian

package sz_test

import (
	"strings"
	"testing"

	sz "github.com/ashvardanian/stringzilla/golang"
)

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

// TestUtf8Count verifies SIMD codepoint counting across byte widths.
func TestUtf8Count(t *testing.T) {
	tests := []struct {
		s    string
		want int
	}{
		{"", 0},
		{"hello", 5}, // 1-byte ASCII
		{"héllo", 5}, // é is 2 bytes, 1 codepoint
		{"日本語", 3},   // 3-byte CJK
		{"a👍b", 3},   // 👍 is 4 bytes, 1 codepoint
		{"éé", 2},    // two précomposed é
	}
	for _, tt := range tests {
		if got := sz.Utf8Count(tt.s); got != tt.want {
			t.Errorf("Utf8Count(%q) = %d, want %d", tt.s, got, tt.want)
		}
	}
}

// TestUtf8Normalize verifies NFC composition, NFKC folding, and idempotence.
func TestUtf8Normalize(t *testing.T) {
	// NFC composes base + combining mark: "e" + U+0301 → "é", which is U+00E9.
	if got := sz.Utf8Normalize("é", sz.NFC); got != "é" {
		t.Errorf("Utf8Normalize(NFC, e+combining acute) = %q, want %q", got, "é")
	}
	// NFKC folds the compatibility ligature U+FB01 (ﬁ) to "fi".
	if got := sz.Utf8Normalize("ﬁ", sz.NFKC); got != "fi" {
		t.Errorf("Utf8Normalize(NFKC, ligature fi) = %q, want %q", got, "fi")
	}
	// Idempotence: normalizing an already-NFC string is a no-op.
	nfc := sz.Utf8Normalize("é", sz.NFC)
	if got := sz.Utf8Normalize(nfc, sz.NFC); got != nfc {
		t.Errorf("Utf8Normalize not idempotent: %q -> %q", nfc, got)
	}
	// Empty input.
	if got := sz.Utf8Normalize("", sz.NFC); got != "" {
		t.Errorf("Utf8Normalize(\"\") = %q, want \"\"", got)
	}
}

func BenchmarkUtf8Count(b *testing.B) {
	s := strings.Repeat("Hello, 世界! café 👍 ", 64)
	b.SetBytes(int64(len(s)))
	for i := 0; i < b.N; i++ {
		_ = sz.Utf8Count(s)
	}
}

func BenchmarkUtf8Normalize(b *testing.B) {
	s := strings.Repeat("café é ﬁ ", 64)
	b.SetBytes(int64(len(s)))
	for i := 0; i < b.N; i++ {
		_ = sz.Utf8Normalize(s, sz.NFC)
	}
}
