//
//  Test.swift
//
//
//  Created by Ash Vardanian on 18/1/24.
//

import XCTest

@testable import StringZilla

class StringZillaTests: XCTestCase {
    var testString: String!

    override func setUp() {
        super.setUp()
        testString = "Hello, world! Welcome to StringZilla. 👋"
        XCTAssertEqual(testString.count, 39)
        XCTAssertEqual(testString.utf8.count, 42)
    }

    func testFindFirstSubstring() {
        let index = testString.findFirst(substring: "world")!
        XCTAssertEqual(testString[index...], "world! Welcome to StringZilla. 👋")
    }

    func testFindLastSubstring() {
        let index = testString.findLast(substring: "o")!
        XCTAssertEqual(testString[index...], "o StringZilla. 👋")
    }

    func testFindFirstCharacterFromSet() {
        let index = testString.findFirst(characterFrom: "aeiou")!
        XCTAssertEqual(testString[index...], "ello, world! Welcome to StringZilla. 👋")
    }

    func testFindLastCharacterFromSet() {
        let index = testString.findLast(characterFrom: "aeiou")!
        XCTAssertEqual(testString[index...], "a. 👋")
    }

    func testFindFirstCharacterNotFromSet() {
        let index = testString.findFirst(characterNotFrom: "aeiou")!
        XCTAssertEqual(testString[index...], "Hello, world! Welcome to StringZilla. 👋")
    }

    func testFindLastCharacterNotFromSet() {
        let index = testString.findLast(characterNotFrom: "aeiou")!
        XCTAssertEqual(testString.distance(from: testString.startIndex, to: index), 38)
        XCTAssertEqual(testString[index...], "👋")
    }

    func testFindLastCharacterNotFromSetNoMatch() {
        let index = "aeiou".findLast(characterNotFrom: "aeiou")
        XCTAssertNil(index)
    }

    func testUtf8CaseFoldedBytes() {
        let folded = "Straße".utf8CaseFoldedBytes()
        XCTAssertEqual(String(decoding: folded, as: UTF8.self), "strasse")
    }

    func testUtf8CaseInsensitiveFind() {
        let haystack =
            "Die Temperaturschwankungen im kosmischen Mikrowellenhintergrund sind ein Maß von etwa 20 µK.\n"
            + "Typografisch sieht man auch: ein Maß von etwa 20 μK."
        let needle = "EIN MASS VON ETWA 20 μK"

        let firstMatchRange = haystack.utf8CaseInsensitiveFind(substring: needle)
        XCTAssertNotNil(firstMatchRange)
        XCTAssertEqual(String(haystack[firstMatchRange!]), "ein Maß von etwa 20 µK")

        let compiledNeedle = Utf8CaseInsensitiveNeedle(needle)
        let remainingHaystack = String(haystack[firstMatchRange!.upperBound...])
        let secondMatchRange = compiledNeedle.findFirst(in: remainingHaystack)
        XCTAssertNotNil(secondMatchRange)
        XCTAssertEqual(String(remainingHaystack[secondMatchRange!]), "ein Maß von etwa 20 μK")
    }

    // MARK: - Hash Function Tests

    func testHashBasic() {
        let text = "Hello, world!"
        let hash = text.hash()

        XCTAssertEqual(text.hash(), hash)
        XCTAssertEqual(text.hash(seed: 0), hash)
        XCTAssertNotEqual(hash, 0)
    }

    func testHashWithSeed() {
        let text = "Hello, world!"
        let hashWithSeedZero = text.hash(seed: 0)
        let hashWithSeed123 = text.hash(seed: 123)

        XCTAssertNotEqual(hashWithSeedZero, hashWithSeed123)
    }

    func testHashConsistency() {
        let identicalText1 = "StringZilla"
        let identicalText2 = "StringZilla"

        XCTAssertEqual(identicalText1.hash(), identicalText2.hash())
        XCTAssertEqual(identicalText1.hash(seed: 42), identicalText2.hash(seed: 42))
    }

    func testHashDistribution() {
        let originalText = "StringZilla"
        let modifiedText = "StringZillb"

        XCTAssertNotEqual(originalText.hash(), modifiedText.hash())
    }

    func testHashEmptyString() {
        let emptyString = ""
        let emptyStringHash = emptyString.hash()

        XCTAssertEqual(emptyString.hash(), emptyStringHash)
        XCTAssertEqual(emptyString.hash(seed: 0), emptyStringHash)
    }

    func testHashUnicodeStrings() {
        let chineseText = "Hello 世界"
        let emojiText = "Hello 👋"

        XCTAssertEqual(chineseText.hash(), chineseText.hash())
        XCTAssertEqual(emojiText.hash(), emojiText.hash())
        XCTAssertNotEqual(chineseText.hash(), emojiText.hash())
    }

    // MARK: - Progressive Hasher Tests

    func testProgressiveHasherBasic() {
        let text = "Hello, world!"
        let hasher = StringZillaHasher()

        hasher.update(text)
        let progressiveHash = hasher.finalize()

        // Compare with single-shot hash to verify consistency
        let singleShotHash = text.hash()

        let secondHasher = StringZillaHasher()
        secondHasher.update(text)
        let secondProgressiveHash = secondHasher.finalize()

        XCTAssertEqual(progressiveHash, secondProgressiveHash)
        XCTAssertNotEqual(progressiveHash, 0)
        XCTAssertNotEqual(singleShotHash, 0)
        XCTAssertEqual(singleShotHash, text.hash())
    }

    func testProgressiveHasherMultipleUpdates() {
        let hasher = StringZillaHasher()
        hasher.update("Hello, ")
        hasher.update("world!")
        let firstChunkingHash = hasher.finalize()

        let sameChunkingHasher = StringZillaHasher()
        sameChunkingHasher.update("Hello, ")
        sameChunkingHasher.update("world!")
        let sameChunkingHash = sameChunkingHasher.finalize()

        let differentChunkingHasher = StringZillaHasher()
        differentChunkingHasher.update("Hello")
        differentChunkingHasher.update(", world!")
        let differentChunkingHash = differentChunkingHasher.finalize()

        XCTAssertEqual(firstChunkingHash, sameChunkingHash)
        XCTAssertEqual(firstChunkingHash, differentChunkingHash)
    }

    func testProgressiveHasherWithSeed() {
        let hasher1 = StringZillaHasher(seed: 0)
        let hasher2 = StringZillaHasher(seed: 123)

        hasher1.update("test")
        hasher2.update("test")

        let hash1 = hasher1.finalize()
        let hash2 = hasher2.finalize()

        // Different seeds should produce different hash values
        XCTAssertNotEqual(hash1, hash2)
    }

    func testProgressiveHasherReset() {
        let hasher = StringZillaHasher(seed: 42)
        hasher.update("first")
        let hashBeforeReset = hasher.finalize()

        hasher.reset(seed: 42)
        hasher.update("first")
        let hashAfterReset = hasher.finalize()

        XCTAssertEqual(hashBeforeReset, hashAfterReset)
    }

    func testProgressiveHasherResetWithNewSeed() {
        let hasher = StringZillaHasher(seed: 0)
        hasher.update("test")
        let hashWithSeedZero = hasher.finalize()

        hasher.reset(seed: 123)
        hasher.update("test")
        let hashWithSeed123 = hasher.finalize()

        XCTAssertNotEqual(hashWithSeedZero, hashWithSeed123)
    }

    func testProgressiveHasherMethodChaining() {
        let hasher = StringZillaHasher()
        let chainedMethodHash = hasher.update("Hello")
            .update(", ")
            .update("world!")
            .finalize()

        XCTAssertNotEqual(chainedMethodHash, 0)
    }

    func testProgressiveHasherEmptyUpdates() {
        let hasherWithEmptyUpdates = StringZillaHasher()
        hasherWithEmptyUpdates.update("")
        hasherWithEmptyUpdates.update("test")
        hasherWithEmptyUpdates.update("")
        let hashWithEmptyUpdates = hasherWithEmptyUpdates.finalize()

        let hasherWithoutEmptyUpdates = StringZillaHasher()
        hasherWithoutEmptyUpdates.update("test")
        let hashWithoutEmptyUpdates = hasherWithoutEmptyUpdates.finalize()

        XCTAssertEqual(hashWithEmptyUpdates, hashWithoutEmptyUpdates)
    }

    // MARK: - SHA-256 Tests

    func testSha256TestVectors() {
        // Helper to convert bytes to hex
        func toHex(_ bytes: [UInt8]) -> String {
            let hexDigits = "0123456789abcdef"
            var result = ""
            for byte in bytes {
                result.append(hexDigits[hexDigits.index(hexDigits.startIndex, offsetBy: Int(byte >> 4))])
                result.append(hexDigits[hexDigits.index(hexDigits.startIndex, offsetBy: Int(byte & 0x0F))])
            }
            return result
        }

        // NIST test vectors
        XCTAssertEqual(toHex("".sha256()),
                      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        XCTAssertEqual(toHex("abc".sha256()),
                      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    func testSha256Streaming() {
        let hasher = StringZillaSha256()
        hasher.update("Hello, ").update("world!")
        let progressive = hasher.digest()
        let oneshot = "Hello, world!".sha256()
        XCTAssertEqual(progressive, oneshot)

        // Hexdigest and reset
        XCTAssertEqual(hasher.hexdigest().count, 64)
        hasher.reset()
        hasher.update("test")
        XCTAssertEqual(hasher.digest().count, 32)
    }
}
