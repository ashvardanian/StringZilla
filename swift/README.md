# StringZilla for Swift

StringZilla is a __Foundation-free__ Swift package that runs everywhere a Swift toolchain does, __not just on Apple platforms__.
Its SIMD-accelerated string kernels are built directly on the C core, so the same search, comparison, hashing, and Unicode-aware methods work identically on __Linux servers and embedded targets__ as they do on macOS, iOS, tvOS, watchOS, and visionOS.
There is no dependency on Foundation, `Darwin`, or any Apple-only runtime, which keeps binaries small and portable.

The package adds these methods directly to `String`, `String.UTF8View`, and `Substring.UTF8View`, all operating on the underlying UTF-8 bytes without intermediate copies.
The one exception is a lazily bridged `NSString`, whose UTF-8 bytes are not contiguous, so each call copies them first.
All operations are exposed through the `StringZillaViewable` protocol, which every supported string type conforms to.
Search methods return native Swift `String.Index` values or a `Range<Index>`, so results splice cleanly back into your strings.

## Installation

Add StringZilla as a Swift Package Manager dependency in your `Package.swift`.

```swift
let package = Package(
    name: "MyApp",
    dependencies: [
        .package(url: "https://github.com/ashvardanian/StringZilla.git", from: "5.0.0")
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

SwiftPM downloads the C kernels prebuilt, as the release's `StringZillaC` XCFramework on Apple platforms and as its artifact bundle on Linux, Android, Windows and WebAssembly, so no C toolchain is needed.
A checkout builds its own with CMake and points `STRINGZILLA_SWIFT_ARTIFACT` at it, relative to the package root:

```sh
cmake --preset swift
cmake --build --preset swift
STRINGZILLA_SWIFT_ARTIFACT=build_swift/StringZillaC.xcframework swift test # StringZillaC.artifactbundle off Apple platforms
```

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
The needle is prepared once and never changes, so one needle can search from many threads at once.
An empty needle matches at the start of any haystack with an empty range.

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

## Devices and Capabilities

Every call runs on the CPU and dispatches to the best kernel of `Device.cpu.capabilitiesEnabled`, a `Capabilities` `OptionSet`:

```swift
import StringZilla

// `enabled` is what dispatch uses: detected on this CPU AND compiled into the binary.
let cpu = Device.cpu
print(try cpu.capabilitiesEnabled)                 // like "serial,neon,neonaes,neonsha"
print(try cpu.capabilitiesEnabled.contains(.neon)) // `Capabilities` is an `OptionSet`

// The two raw axes, when you specifically mean one of them:
let onThisCpu = try cpu.capabilitiesDetected
let inThisBinary = cpu.capabilitiesCompiled

// Narrow dispatch before starting threads, and prepare each thread that runs kernels:
try cpu.capabilitiesEnable(cpu.capabilitiesEnabled.subtracting(.neon))
try cpu.configureThread(cpu.capabilitiesEnabled)
```

- `capabilitiesDetected` is what the device can execute, from CPUID or HWCAP on the CPU.
- `capabilitiesCompiled` is what this binary contains for devices of its kind, from the ISA probes at build time.
- `capabilitiesEnabled` is what dispatch uses, both axes at once unless narrowed, and on the CPU always contains `.serial`.
- `capabilitiesEnable(_:)` makes its argument the CPU's enabled set, clamped to both axes, and returns what took effect.
- `configureThread(_:)` prepares the calling thread for the kernels of its argument, once per thread that runs them.

Reach for `capabilitiesEnabled` unless you specifically mean one of the raw axes.
`capabilitiesDetected` describes the machine and says nothing about whether a kernel was compiled in, so a build whose ISA probes failed still reports your CPU's full feature set while containing no SIMD kernels at all.

A `Device` is the host CPU or one GPU of a runtime, named by that runtime's own ordinal:

```swift
let gpus = try Device.count(.metal)            // throws without a Metal device
let gpu = try Device(kind: .metal, ordinal: 0) // throws past the last one
print(try gpu.capabilitiesEnabled)             // like "metal"
```

`capabilitiesEnable(_:)` and `configureThread(_:)` throw a `StringZilla.Error` on a GPU, which keeps no enabled set or thread state of its own: this package only reports GPU capabilities.
`.cpus`, `.devices` and `.any` group the CPU capabilities, the GPU ones, and all of them.
