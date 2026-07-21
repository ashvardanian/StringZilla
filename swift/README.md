# StringZilla for Swift

StringZilla is a __Foundation-free__ Swift package that runs everywhere a Swift toolchain does, __not just on Apple platforms__.
Its SIMD-accelerated string kernels are built directly on the C core, so the same search, comparison, hashing, and Unicode-aware methods work identically on __Linux servers and embedded targets__ as they do on macOS, iOS, tvOS, watchOS, and visionOS.
There is no dependency on Foundation, `Darwin`, or any Apple-only runtime, which keeps binaries small and portable.

The package adds these methods directly to `String`, `String.UTF8View`, and `Substring.UTF8View`, all operating on the underlying UTF-8 bytes without intermediate copies.
All operations are exposed through the `StringZillaViewable` protocol, which every supported string type conforms to.
Search methods return native Swift `String.Index` values or a `Range<Index>`, so results splice cleanly back into your strings.

## Installation

Add StringZilla as a Swift Package Manager dependency in your `Package.swift`.

```swift
let package = Package(
    name: "MyApp",
    dependencies: [
        .package(url: "https://github.com/ashvardanian/StringZilla.git", from: "4.0.0")
    ],
    targets: [
        .target(
            name: "MyApp",
            dependencies: [
                .product(name: "StringZilla", package: "StringZilla")
            ]
        )
    ]
)
```

SwiftPM is the only build path on every platform, and it is the same `swift build` on Linux and embedded targets as on Apple hardware.
Linux needs no special configuration: the package compiles the C kernels directly and links against `Glibc` instead of `Darwin` automatically.

Then import the module where you need it.

```swift
import StringZilla
```

The product name is `StringZilla`; importing it pulls in both the Swift extension and the underlying C kernels.

## Searching and Counting

Substring search returns the `Index?` of the first or last match, or `nil` when the needle is absent.

```swift
let haystack = "Hello, world! Hello, Swift!"
let first = haystack.findFirst(substring: "Hello")   // Index of position 0
let last = haystack.findLast(substring: "Hello")     // Index of the second "Hello"
assert(first == haystack.startIndex)
assert(haystack.findFirst(substring: "Rust") == nil)
```

Byte-set search locates the first or last byte that belongs to, or is excluded from, a set of characters.
The character set is itself any string-like value, treated as a bag of bytes.

```swift
let text = "  trim me  "
let firstNonSpace = text.findFirst(characterNotFrom: " ")  // first non-blank byte
let lastNonSpace = text.findLast(characterNotFrom: " ")    // last non-blank byte
let firstVowel = text.findFirst(characterFrom: "aeiou")    // first vowel
let lastVowel = text.findLast(characterFrom: "aeiou")      // last vowel
```

All six finders are generic over the needle type, so you can search a `String` for a `String.UTF8View` needle and vice versa.

## Comparison and Equality

Comparisons are SIMD-accelerated and return a `StringZillaOrdering` of `.ascending`, `.equal`, or `.descending`, or a `Bool`.

```swift
assert("apple".compare("banana") == .ascending)  // byte-order lexicographic
assert("abc".equals("abc"))                       // byte-level equality
```

`utf8UncasedOrder(_:)` performs the same ordering but with full Unicode case folding, so `"STRASSE"` and `"straße"` compare as equal.

```swift
assert("STRASSE".utf8UncasedOrder("straße") == .equal)
```

## Unicode Case Folding and Normalization

`utf8UncasedFind(substring:)` performs a case-insensitive search using full Unicode folding and returns a byte-accurate `Range<Index>?`.
The matched length can differ from the needle length, since folding can change byte counts.

```swift
if let range = "Grüße".utf8UncasedFind(substring: "GRÜSSE") {
    print("matched", "Grüße"[range])
}
```

For repeated case-insensitive searches with the same needle, build a `Utf8UncasedNeedle` once and reuse it.
The needle caches its precomputed metadata, but is not safe for concurrent use.

```swift
let needle = Utf8UncasedNeedle("hello")
let r = needle.findFirst(in: "Say HELLO to the world")
assert(r != nil)
```

`utf8UncasedFoldedBytes()` returns the fully case-folded UTF-8 bytes, which may be longer than the input since `"ß"` folds to `"ss"`.

```swift
let folded = "Straße".utf8UncasedFoldedBytes()  // [UInt8] of "strasse"
assert(folded == Array("strasse".utf8))
```

Normalization is driven by `StringZillaNormalizationForm`, one of `.nfc`, `.nfd`, `.nfkc`, or `.nfkd`.
`utf8Normalized(_:)` returns the normalized UTF-8 bytes and defaults to `.nfc`, `utf8NormalizationViolation(_:)` returns the `Index?` of the first non-conforming byte, and `isUtf8Normalized(_:)` is a convenience `Bool`.

```swift
let nfc = "e\u{0301}".utf8Normalized(.nfc)  // composed "é" bytes
assert("café".isUtf8Normalized(.nfc))       // already composed
let fi = "\u{FB01}".utf8Normalized(.nfkc)   // ligature "ﬁ" → "fi" bytes
assert(fi == Array("fi".utf8))
let bad = "e\u{0301}".utf8NormalizationViolation(.nfc) // Index of the violation
assert(bad != nil)
```

## Splitting and Segmentation

These methods return byte-accurate `[Range<Index>]` arrays you can subscript back into the source string.

`utf8Words()` splits into UAX-29 words that tile the input, so every byte belongs to exactly one word.

```swift
for range in "Hello, 世界!".utf8Words() { // tiles the Latin run and the CJK run
    print("Hello, 世界!"[range])
}
```

`utf8Lines(skipEmpty:)` splits on the Unicode line-break characters and CRLF, and `utf8Tokens(skipEmpty:)` splits on the 25 Unicode `White_Space` characters.
Both keep empty segments by default under the cross-language KEEP policy; pass `skipEmpty: true` to drop them.
With N delimiters you get N+1 segments, so `"a\n\nb\n".utf8Lines()` yields four ranges: `"a"`, `""`, `"b"`, and `""`.

```swift
let lines = "a\n\nb\n".utf8Lines()                      // 4 ranges, including empties
assert(lines.count == 4)
let words = "  hi  there  ".utf8Tokens(skipEmpty: true) // ["hi", "there"]
assert(words.count == 2)
```

## Hashing and Checksums

`hash(seed:)` computes a fast 64-bit `UInt64` hash of the content, with an optional seed.

```swift
let h = "the quick brown fox".hash()
let seeded = "the quick brown fox".hash(seed: 42)
assert(h != seeded)  // a different seed yields a different hash
```

For data arriving in chunks, `StringZillaHasher` hashes incrementally.
`update(_:)` is chainable, `finalize()` and its alias `digest()` return the `UInt64` without consuming the state, and `reset(seed:)` restarts it.

```swift
let hasher = StringZillaHasher(seed: 0)
hasher.update("the quick ").update("brown fox")
let digest = hasher.finalize()
assert(digest == "the quick brown fox".hash())
```

`sha256()` returns the SHA-256 digest of the content as a 32-byte `[UInt8]`.

```swift
let sum = "hello".sha256()  // [UInt8] of length 32
assert(sum.count == 32)
```

The streaming `StringZillaSha256` mirrors the incremental hasher.
`update(_:)` accepts either a string view or a `[UInt8]`, `finalize()` and its alias `digest()` return the 32-byte digest, `hexdigest()` returns the 64-character lowercase hex string, and `reset()` restarts it.

```swift
let sha = StringZillaSha256()
sha.update("hello, ").update("world")
let hex = sha.hexdigest()  // 64-char hex string
assert(hex.count == 64)
```
