//
//  swift/Test.swift
//  Tests for the StringZilla Swift binding.
//
//  - Author: Ash Vardanian
//  - Date: January 21, 2024
//

import StringZilla
import Testing

/// Mixes ASCII with one astral codepoint, so runes and UTF-8 bytes disagree.
let greeting = "Hello, world! Welcome to StringZilla. 👋"

@Test func countRunes() {
    #expect(greeting.utf8.count == 42)
    #expect(greeting.countRunes() == greeting.unicodeScalars.count)
    #expect(greeting.countRunes() == 39)
    #expect("".countRunes() == 0)
    #expect("abc".countRunes() == 3)
}

@Test func indexOfRune() throws {
    let mixed = "aé中𝄞b"  // 1, 2, 3, 4 and 1 bytes
    #expect((mixed.countRunes(), mixed.utf8.count) == (5, 11))
    #expect(mixed.index(ofRune: 0) == mixed.startIndex)
    #expect(mixed[try #require(mixed.index(ofRune: 2))...] == "中𝄞b")
    #expect(mixed[try #require(mixed.index(ofRune: 3))...] == "𝄞b")
    #expect(mixed[try #require(mixed.index(ofRune: 4))...] == "b")
    #expect(mixed.index(ofRune: 5) == nil)
    #expect(mixed.index(ofRune: 99) == nil)
    #expect(mixed.index(ofRune: -1) == nil)
}

@Test func indexOfRuneMatchesUnicodeScalars() {
    for (offset, position) in greeting.unicodeScalars.indices.enumerated() {
        #expect(greeting.index(ofRune: offset) == position.samePosition(in: greeting))
    }
}

/// One search over `greeting`, and the suffix it must land on.
struct Search: Sendable, CustomTestStringConvertible {
    let testDescription: String
    let suffix: String
    let find: @Sendable (String) -> String.Index?

    init(_ name: String, _ suffix: String, _ find: @escaping @Sendable (String) -> String.Index?) {
        self.testDescription = name
        self.suffix = suffix
        self.find = find
    }
}

@Test(arguments: [
    Search("first substring", "world! Welcome to StringZilla. 👋") { $0.findFirst(substring: "world") },
    Search("last substring", "o StringZilla. 👋") { $0.findLast(substring: "o") },
    Search("first vowel", "ello, world! Welcome to StringZilla. 👋") { $0.findFirst(characterFrom: "aeiou") },
    Search("last vowel", "a. 👋") { $0.findLast(characterFrom: "aeiou") },
    Search("first non-vowel", greeting) { $0.findFirst(characterNotFrom: "aeiou") },
    Search("last non-vowel", "👋") { $0.findLast(characterNotFrom: "aeiou") },
])
func search(_ search: Search) throws {
    let index = try #require(search.find(greeting))
    #expect(greeting[index...] == search.suffix)
}

@Test func searchWithoutMatch() {
    #expect("aeiou".findLast(characterNotFrom: "aeiou") == nil)
}

@Test func utf8UncasedFoldedBytes() {
    #expect("Straße".utf8UncasedFoldedBytes() == Array("strasse".utf8))
}

@Test func utf8UncasedFind() throws {
    let haystack =
        "Die Temperaturschwankungen im kosmischen Mikrowellenhintergrund sind ein Maß von etwa 20 µK.\n"
        + "Typografisch sieht man auch: ein Maß von etwa 20 μK."
    let needle = "EIN MASS VON ETWA 20 μK"
    let first = try #require(haystack.utf8UncasedFind(substring: needle))
    #expect(haystack[first] == "ein Maß von etwa 20 µK")
    let rest = String(haystack[first.upperBound...])
    let second = try #require(Utf8UncasedNeedle(needle).findFirst(in: rest))
    #expect(rest[second] == "ein Maß von etwa 20 μK")
}

@Test func hash() {
    let text = "Hello, world!"
    #expect(text.hash() == text.hash(seed: 0))
    #expect(text.hash() != 0)
    #expect(text.hash(seed: 0) != text.hash(seed: 123))
    #expect("StringZilla".hash(seed: 42) == "StringZilla".hash(seed: 42))
    #expect("StringZilla".hash() != "StringZillb".hash())
    #expect("Hello 世界".hash() != "Hello 👋".hash())
    #expect("".hash() == "".hash(seed: 0))
}

/// Hashes `chunks` in order with one streaming hasher.
func streamed(_ chunks: String..., seed: UInt64 = 0) -> UInt64 {
    let hasher = StringZillaHasher(seed: seed)
    for chunk in chunks { hasher.update(chunk) }
    return hasher.finalize()
}

@Test func streamingHash() {
    #expect(streamed("Hello, world!") != 0)
    #expect(streamed("Hello, ", "world!") == streamed("Hello", ", world!"))
    #expect(streamed("Hello, world!") == streamed("Hello", ", world!"))
    #expect(streamed("", "test", "") == streamed("test"))
    #expect(streamed("test", seed: 0) != streamed("test", seed: 123))
    #expect(StringZillaHasher().update("Hello").update(", ").update("world!").finalize() == streamed("Hello, world!"))
}

@Test func streamingHashReset() {
    let hasher = StringZillaHasher(seed: 42)
    hasher.update("first")
    let seeded = hasher.finalize()
    hasher.reset(seed: 42)
    hasher.update("first")
    #expect(hasher.finalize() == seeded)
    hasher.reset(seed: 123)
    hasher.update("first")
    #expect(hasher.finalize() != seeded)
}

/// NIST test vectors.
let sha256Vectors = [
    ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
]

@Test(arguments: sha256Vectors)
func sha256(_ message: String, _ hex: String) {
    let hasher = StringZillaSha256().update(message)
    #expect(hasher.hexdigest() == hex)
    #expect(hasher.digest() == message.sha256())
}

@Test func sha256Streaming() {
    let hasher = StringZillaSha256().update("Hello, ").update("world!")
    #expect(hasher.digest() == "Hello, world!".sha256())
    hasher.reset()
    hasher.update("test")
    #expect(hasher.digest() == "test".sha256())
}

@Test func utf8WordsTileInput() {
    let text = "Hello, world! 👋"
    let words = text.utf8Words().map { String(text[$0]) }
    #expect(words.joined() == text)
    #expect(words.contains("Hello"))
    #expect(words.contains("world"))
}

/// One split of `text`, and the segments it must produce.
struct Split: Sendable, CustomTestStringConvertible {
    let testDescription: String
    let text: String
    let segments: [String]
    let split: @Sendable (String) -> [Range<String.Index>]

    init(
        _ name: String, _ text: String, _ segments: [String],
        _ split: @escaping @Sendable (String) -> [Range<String.Index>]
    ) {
        self.testDescription = name
        self.text = text
        self.segments = segments
        self.split = split
    }
}

@Test(arguments: [
    Split("lines keep empty gaps", "a\n\nb\n", ["a", "", "b", ""]) { $0.utf8Lines() },
    Split("lines skip empty gaps", "a\n\nb\n", ["a", "b"]) { $0.utf8Lines(skipEmpty: true) },
    Split("CR LF is one delimiter", "a\r\nb", ["a", "b"]) { $0.utf8Lines() },
    Split("lines without a trailing newline", "a\nb", ["a", "b"]) { $0.utf8Lines() },
    Split("an empty text is one empty line", "", [""]) { $0.utf8Lines() },
    Split("an empty text has no non-empty lines", "", []) { $0.utf8Lines(skipEmpty: true) },
    Split("tokens keep empty gaps", "  hi  ", ["", "", "hi", "", ""]) { $0.utf8Tokens() },
    Split("tokens skip empty gaps", "  hi  ", ["hi"]) { $0.utf8Tokens(skipEmpty: true) },
    Split("spaces, tabs and newlines all separate", "the quick\tbrown\nfox", ["the", "quick", "brown", "fox"]) {
        $0.utf8Tokens(skipEmpty: true)
    },
    Split("U+3000 separates as one 3-byte codepoint", "a\u{3000}b", ["a", "b"]) { $0.utf8Tokens(skipEmpty: true) },
])
func split(_ row: Split) {
    #expect(row.split(row.text).map { String(row.text[$0]) } == row.segments)
}

@Test func byteOrder() {
    #expect("apple".compare("banana") == .ascending)
    #expect("banana".compare("apple") == .descending)
    #expect("apple".compare("apple") == .equal)
    #expect("app".compare("apple") == .ascending)  // a prefix orders before its extensions
}

@Test func equals() {
    #expect("StringZilla".equals("StringZilla"))
    #expect(!"StringZilla".equals("StringZillb"))
    #expect(!"String".equals("StringZilla"))  // a shared prefix is not enough
}

@Test func utf8UncasedOrder() {
    #expect("HELLO".utf8UncasedOrder("hello") == .equal)
    #expect("Straße".utf8UncasedOrder("STRASSE") == .equal)  // ß folds to "ss"
    #expect("apple".utf8UncasedOrder("BANANA") == .ascending)
}

/// Text, form, and the bytes it normalizes to.
let normalizations: [(String, StringZillaNormalizationForm, String)] = [
    ("e\u{0301}", .nfc, "\u{00E9}"),
    ("\u{00E9}", .nfd, "e\u{0301}"),
    ("", .nfc, ""),
    ("", .nfd, ""),
]

@Test(arguments: normalizations)
func normalize(_ text: String, _ form: StringZillaNormalizationForm, _ normalized: String) {
    // Bytes, since `String` equality already treats canonical equivalents as equal.
    #expect(text.utf8Normalized(form) == Array(normalized.utf8))
}

/// Text, form, and whether the text is already in that form.
let normalizationChecks: [(String, StringZillaNormalizationForm, Bool)] = [
    ("\u{00E9}", .nfc, true),
    ("e\u{0301}", .nfc, false),
    ("e\u{0301}", .nfd, true),
    ("\u{00E9}", .nfd, false),
    ("hello", .nfc, true),
    ("hello", .nfd, true),
    ("", .nfc, true),
]

@Test(arguments: normalizationChecks)
func detectNormalization(_ text: String, _ form: StringZillaNormalizationForm, _ isNormalized: Bool) {
    #expect(text.isUtf8Normalized(form) == isNormalized)
    #expect((text.utf8NormalizationViolation(form) == nil) == isNormalized)
}
