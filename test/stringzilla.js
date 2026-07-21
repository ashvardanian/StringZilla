import test from "node:test";
import assert from "node:assert";

// Import our zero-copy buffer-only StringZilla
import stringzilla from "../javascript/stringzilla.js";

test("Buffer Find - Positive Case", () => {
    const haystack = Buffer.from("hello world, hello john");
    const needle = Buffer.from("hello");

    const result = stringzilla.find(haystack, needle);
    assert.strictEqual(result, 0n);
});

test("Buffer Find - Negative Case (Not Found)", () => {
    const haystack = Buffer.from("hello world");
    const needle = Buffer.from("xyz");

    const result = stringzilla.find(haystack, needle);
    assert.strictEqual(result, -1n);
});

test("Buffer Find - Empty Needle", () => {
    const haystack = Buffer.from("hello world");
    const needle = Buffer.alloc(0);

    const result = stringzilla.find(haystack, needle);
    assert.strictEqual(result, 0n);
});

test("Buffer Count - Single Occurrence", () => {
    const haystack = Buffer.from("hello world");
    const needle = Buffer.from("world");

    const result = stringzilla.count(haystack, needle);
    assert.strictEqual(result, 1n);
});

test("Buffer Count - Multiple Occurrences", () => {
    const haystack = Buffer.from("hello world, hello John");
    const needle = Buffer.from("hello");

    const result = stringzilla.count(haystack, needle, false);
    assert.strictEqual(result, 2n);
});

test("Buffer Count - Overlapping Test", () => {
    const haystack = Buffer.from("abababab");
    const needle = Buffer.from("aba");

    // Non-overlapping count
    const resultNonOverlap = stringzilla.count(haystack, needle, false);
    assert.strictEqual(resultNonOverlap, 2n);

    // Overlapping count
    const resultOverlap = stringzilla.count(haystack, needle, true);
    assert.strictEqual(resultOverlap, 3n);
});

test("Buffer Hash - Basic Hashing", () => {
    const buffer = Buffer.from("hello world");

    const hash = stringzilla.hash(buffer);
    assert.strictEqual(typeof hash, "bigint");
    assert(hash > 0n);
});

test("Buffer Hash - Same Input Same Output", () => {
    const buffer1 = Buffer.from("hello world");
    const buffer2 = Buffer.from("hello world");

    const hash1 = stringzilla.hash(buffer1);
    const hash2 = stringzilla.hash(buffer2);

    assert.strictEqual(hash1, hash2);
});

test("Buffer Hash - Different Seeds Different Output", () => {
    const buffer = Buffer.from("hello world");

    const hash1 = stringzilla.hash(buffer, 0n);
    const hash2 = stringzilla.hash(buffer, 123n);

    assert.notStrictEqual(hash1, hash2);
});

test("Hasher Class - Single Buffer", () => {
    const buffer = Buffer.from("hello world");

    const hasher = new stringzilla.Hasher();
    hasher.update(buffer);
    const hashStreaming = hasher.digest();

    const hashSingle = stringzilla.hash(buffer);
    assert.strictEqual(hashSingle, hashStreaming);
});

test("Hasher Class - Multiple Buffers", () => {
    const bufferPrefix = Buffer.from("hello ");
    const bufferSuffix = Buffer.from("world");
    const bufferCombined = Buffer.from("hello world");

    const hasher = new stringzilla.Hasher();
    hasher.update(bufferPrefix).update(bufferSuffix);
    const hashStreaming = hasher.digest();

    const hashCombined = stringzilla.hash(bufferCombined);

    // Progressive hashing should match single-shot hashing
    assert.strictEqual(hashCombined, hashStreaming);
});

test("Hasher Class - Reset Functionality", () => {
    const buffer = Buffer.from("hello world");

    const hasher = new stringzilla.Hasher();
    hasher.update(buffer);
    const hash1 = hasher.digest();

    hasher.reset();
    hasher.update(buffer);
    const hash2 = hasher.digest();

    assert.strictEqual(hash1, hash2);
});

test("Find Last - Basic Test", () => {
    const haystack = Buffer.from("hello world, hello john");
    const needle = Buffer.from("hello");

    const result = stringzilla.findLast(haystack, needle);
    assert.strictEqual(result, 13n);
});

test("Find Byte - Basic Test", () => {
    const haystack = Buffer.from("hello world");
    const byte = "o".charCodeAt(0);

    const result = stringzilla.findByte(haystack, byte);
    assert.strictEqual(result, 4n);
});

test("Find Last Byte - Basic Test", () => {
    const haystack = Buffer.from("hello world");
    const byte = "o".charCodeAt(0);

    const result = stringzilla.findLastByte(haystack, byte);
    assert.strictEqual(result, 7n);
});

test("Find Byte From - Basic Test", () => {
    const haystack = Buffer.from("hello world");
    const charset = Buffer.from("aeiou");

    const result = stringzilla.findByteFrom(haystack, charset);
    assert.strictEqual(result, 1n); // First vowel 'e'
});

test("Find Last Byte From - Basic Test", () => {
    const haystack = Buffer.from("hello world");
    const charset = Buffer.from("aeiou");

    const result = stringzilla.findLastByteFrom(haystack, charset);
    assert.strictEqual(result, 7n); // Last vowel 'o'
});

test("Equal - Basic Test", () => {
    const buffer1 = Buffer.from("hello");
    const buffer2 = Buffer.from("hello");
    const buffer3 = Buffer.from("world");

    assert.strictEqual(stringzilla.equal(buffer1, buffer2), true);
    assert.strictEqual(stringzilla.equal(buffer1, buffer3), false);
});

test("Compare - Basic Test", () => {
    const buffer1 = Buffer.from("abc");
    const buffer2 = Buffer.from("abc");
    const buffer3 = Buffer.from("def");
    const buffer4 = Buffer.from("ab");

    assert.strictEqual(stringzilla.compare(buffer1, buffer2), 0);
    assert(stringzilla.compare(buffer1, buffer3) < 0);
    assert(stringzilla.compare(buffer3, buffer1) > 0);
    assert(stringzilla.compare(buffer1, buffer4) > 0);
});

test("Byte Sum - Basic Test", () => {
    const buffer = Buffer.from([1, 2, 3, 4, 5]);
    const expectedSum = 1 + 2 + 3 + 4 + 5;

    const result = stringzilla.byteSum(buffer);
    assert.strictEqual(result, BigInt(expectedSum));
});

test("Zero-Copy Performance Test", () => {
    // Test with larger buffers to demonstrate zero-copy benefits
    const largeHaystack = Buffer.alloc(10000, "a");
    const needle = Buffer.from("aaa");

    // This should be fast due to zero-copy buffer access
    const result = stringzilla.find(largeHaystack, needle);
    assert.strictEqual(result, 0n);

    // Hash performance test
    const hash = stringzilla.hash(largeHaystack);
    assert.strictEqual(typeof hash, "bigint");

    // Byte sum performance test
    const byteSum = stringzilla.byteSum(largeHaystack);
    assert.strictEqual(typeof byteSum, "bigint");

    // Find byte performance test
    const byteResult = stringzilla.findByte(largeHaystack, 97); // 'a'
    assert.strictEqual(byteResult, 0n);
});

test("Edge Cases - Empty Buffers", () => {
    const haystack = Buffer.from("hello world");
    const empty = Buffer.alloc(0);

    // Finding empty in non-empty should return 0
    assert.strictEqual(stringzilla.find(haystack, empty), 0n);
    assert.strictEqual(
        stringzilla.findLast(haystack, empty),
        BigInt(haystack.length)
    );

    // Finding non-empty in empty should return -1
    assert.strictEqual(stringzilla.find(empty, haystack), -1n);
    assert.strictEqual(stringzilla.findLast(empty, haystack), -1n);

    // Empty in empty
    assert.strictEqual(stringzilla.find(empty, empty), 0n);
    assert.strictEqual(stringzilla.count(empty, empty), 0n);
});

test("Find Byte - Boundary Values", () => {
    const buffer = Buffer.from([0, 127, 128, 255]);

    // Test boundary byte values
    assert.strictEqual(stringzilla.findByte(buffer, 0), 0n);
    assert.strictEqual(stringzilla.findByte(buffer, 127), 1n);
    assert.strictEqual(stringzilla.findByte(buffer, 128), 2n);
    assert.strictEqual(stringzilla.findByte(buffer, 255), 3n);

    // Test not found
    assert.strictEqual(stringzilla.findByte(buffer, 1), -1n);
});

test("UTF-8 Multi-byte Character Handling", () => {
    const haystack = Buffer.from("Hello 世界 World");
    const needle = Buffer.from("世界");

    // Should work at byte level, not character level
    const result = stringzilla.find(haystack, needle);
    assert(result > 0n);

    // Test with emoji
    const emojiBuffer = Buffer.from("Hello 👋 World");
    const emoji = Buffer.from("👋");
    assert(stringzilla.find(emojiBuffer, emoji) > 0n);
});

test("UTF-8 Case Fold - Sharp S and Ligature", () => {
    const folded1 = stringzilla.utf8UncasedFold(Buffer.from("Straße"));
    assert.strictEqual(folded1.toString("utf8"), "strasse");

    const folded2 = stringzilla.utf8UncasedFold(Buffer.from("ofﬁce")); // contains U+FB01
    assert.strictEqual(folded2.toString("utf8"), "office");
});

test("UTF-8 Uncased Find - Full Case Folding", () => {
    const haystack = Buffer.from(
        "Die Temperaturschwankungen im kosmischen Mikrowellenhintergrund sind ein Maß von etwa 20 µK.\n" +
        "Typografisch sieht man auch: ein Maß von etwa 20 μK."
    );
    const needle = Buffer.from("EIN MASS VON ETWA 20 μK");

    const firstResult = stringzilla.utf8UncasedFind(haystack, needle);
    assert.notStrictEqual(firstResult.index, -1n);
    assert(firstResult.length > 0n);
    const firstStart = Number(firstResult.index);
    const firstEnd = Number(firstResult.index + firstResult.length);
    assert.strictEqual(
        haystack.subarray(firstStart, firstEnd).toString("utf8"),
        "ein Maß von etwa 20 µK"
    );

    // Find the second match after the first one
    const remainingHaystack = haystack.subarray(firstEnd);
    const secondResult = stringzilla.utf8UncasedFind(
        remainingHaystack,
        needle
    );
    assert.notStrictEqual(secondResult.index, -1n);
    const secondStart = firstEnd + Number(secondResult.index);
    const secondEnd = secondStart + Number(secondResult.length);
    assert.strictEqual(
        haystack.subarray(secondStart, secondEnd).toString("utf8"),
        "ein Maß von etwa 20 μK"
    );
});

test("UTF-8 Uncased Needle - Reuse Metadata", () => {
    const haystack = Buffer.from(
        "Die Temperaturschwankungen im kosmischen Mikrowellenhintergrund sind ein Maß von etwa 20 µK.\n" +
        "Typografisch sieht man auch: ein Maß von etwa 20 μK."
    );
    const needleBytes = Buffer.from("EIN MASS VON ETWA 20 μK");
    const compiledNeedle = new stringzilla.Utf8UncasedNeedle(
        needleBytes
    );

    const firstResult = compiledNeedle.findIn(haystack);
    assert.notStrictEqual(firstResult.index, -1n);
    const firstStart = Number(firstResult.index);
    const firstEnd = Number(firstResult.index + firstResult.length);
    assert.strictEqual(
        haystack.subarray(firstStart, firstEnd).toString("utf8"),
        "ein Maß von etwa 20 µK"
    );

    const secondResult = compiledNeedle.findIn(haystack.subarray(firstEnd));
    assert.notStrictEqual(secondResult.index, -1n);
});

test("Pattern at Buffer Boundaries", () => {
    const haystack = Buffer.from("abcdefghijk");

    // Pattern at start
    assert.strictEqual(stringzilla.find(haystack, Buffer.from("abc")), 0n);

    // Pattern at end
    assert.strictEqual(stringzilla.find(haystack, Buffer.from("ijk")), 8n);
    assert.strictEqual(stringzilla.findLast(haystack, Buffer.from("ijk")), 8n);

    // Pattern spans entire buffer
    assert.strictEqual(stringzilla.find(haystack, haystack), 0n);
});

test("Repeated Patterns", () => {
    const haystack = Buffer.from("aaaaaaaaaa");
    const needle = Buffer.from("aa");

    // Test first and last occurrence
    assert.strictEqual(stringzilla.find(haystack, needle), 0n);
    assert.strictEqual(stringzilla.findLast(haystack, needle), 8n);

    // Count with and without overlap
    assert.strictEqual(stringzilla.count(haystack, needle, false), 5n);
    assert.strictEqual(stringzilla.count(haystack, needle, true), 9n);
});

test("Find Byte From - Edge Cases", () => {
    const haystack = Buffer.from("1234567890");

    // Empty charset
    const emptyCharset = Buffer.alloc(0);
    assert.strictEqual(stringzilla.findByteFrom(haystack, emptyCharset), -1n);

    // Charset with all possible bytes
    const allBytes = Buffer.alloc(256);
    for (let i = 0; i < 256; i++) allBytes[i] = i;
    assert.strictEqual(stringzilla.findByteFrom(haystack, allBytes), 0n);

    // Charset with duplicates
    const duplicates = Buffer.from("1111");
    assert.strictEqual(stringzilla.findByteFrom(haystack, duplicates), 0n);
});

test("Binary Data Handling", () => {
    // Test with null bytes and binary data
    const binaryData = Buffer.from([0x00, 0x01, 0x02, 0x00, 0x03, 0x00]);
    const nullByte = Buffer.from([0x00]);

    assert.strictEqual(stringzilla.find(binaryData, nullByte), 0n);
    assert.strictEqual(stringzilla.findLast(binaryData, nullByte), 5n);
    assert.strictEqual(stringzilla.count(binaryData, nullByte), 3n);

    // Test hash consistency with binary data
    const hash1 = stringzilla.hash(binaryData);
    const hash2 = stringzilla.hash(binaryData);
    assert.strictEqual(hash1, hash2);
});

test("Large Buffer Operations", () => {
    const size = 100000; // 100KB (smaller than 1MB for faster tests)
    const largeBuffer = Buffer.alloc(size);

    // Fill with pattern
    for (let i = 0; i < size; i++) {
        largeBuffer[i] = i % 256;
    }

    // Test operations on large buffer
    const pattern = Buffer.from([0, 1, 2, 3]);
    assert(stringzilla.count(largeBuffer, pattern) > 0n);

    // Test hash performance
    const start = Date.now();
    const hash = stringzilla.hash(largeBuffer);
    const duration = Date.now() - start;
    assert(duration < 100); // Should be fast
    assert(typeof hash === "bigint");
});

test("Hasher - Incremental vs Single Shot", () => {
    const data = Buffer.from("a".repeat(1000));

    // Single shot
    const hashSingle = stringzilla.hash(data);

    // Progressive hashing with different chunk sizes should be consistent
    const hasher1 = new stringzilla.Hasher();
    hasher1.update(data.subarray(0, 100));
    hasher1.update(data.subarray(100, 500));
    hasher1.update(data.subarray(500));
    const hashProgressive1 = hasher1.digest();

    const hasher2 = new stringzilla.Hasher();
    hasher2.update(data.subarray(0, 300));
    hasher2.update(data.subarray(300));
    const hashProgressive2 = hasher2.digest();

    // Progressive hashing with same data should be consistent
    assert.strictEqual(hashProgressive1, hashProgressive2);

    // Test that single-shot and progressive produce valid hashes
    assert.strictEqual(typeof hashSingle, "bigint");
    assert.strictEqual(typeof hashProgressive1, "bigint");
    assert(hashSingle > 0n);
    assert(hashProgressive1 > 0n);
});

test("Compare - Special Cases", () => {
    // Different lengths
    assert(stringzilla.compare(Buffer.from("a"), Buffer.from("aa")) < 0);
    assert(stringzilla.compare(Buffer.from("aa"), Buffer.from("a")) > 0);

    // Empty buffers
    assert.strictEqual(
        stringzilla.compare(Buffer.alloc(0), Buffer.alloc(0)),
        0
    );
    assert(stringzilla.compare(Buffer.alloc(0), Buffer.from("a")) < 0);
    assert(stringzilla.compare(Buffer.from("a"), Buffer.alloc(0)) > 0);

    // Binary data comparison
    const binary1 = Buffer.from([0x00, 0x01, 0x02]);
    const binary2 = Buffer.from([0x00, 0x01, 0x03]);
    assert(stringzilla.compare(binary1, binary2) < 0);
});

test("SHA-256 - Test Vectors", () => {
    // NIST test vectors
    assert.strictEqual(
        stringzilla.sha256(Buffer.from("")).toString("hex"),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert.strictEqual(
        stringzilla.sha256(Buffer.from("abc")).toString("hex"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
});

test("Sha256 Class - Streaming", () => {
    const hasher = new stringzilla.Sha256();
    hasher.update(Buffer.from("Hello, ")).update(Buffer.from("world!"));
    const progressive = hasher.digest();
    const oneshot = stringzilla.sha256(Buffer.from("Hello, world!"));
    assert.strictEqual(progressive.toString("hex"), oneshot.toString("hex"));

    // Hexdigest and reset
    assert.strictEqual(hasher.hexdigest(), progressive.toString("hex"));
    hasher.reset();
    hasher.update(Buffer.from("test"));
    assert.strictEqual(hasher.digest().length, 32);
});

test("Utf8 Normalization - NFC and NFD Round-Trips", () => {
    // "café" with a decomposed e + combining acute composes into NFC
    const decomposed = Buffer.from("cafe\u0301"); // NFD: e + combining acute
    const composed = Buffer.from("caf\u00e9"); // NFC: precomposed é
    assert.strictEqual(stringzilla.utf8Norm(decomposed, stringzilla.Utf8NormalForm.NFC).toString(), composed.toString());
    assert.strictEqual(stringzilla.utf8Norm(composed, stringzilla.Utf8NormalForm.NFD).toString(), decomposed.toString());

    // Match the engine against JavaScript's built-in normalization
    const sample = Buffer.from("Ångström: ﬁt ½ ℕ 가각");
    for (const [name, form] of Object.entries(stringzilla.Utf8NormalForm)) {
        assert.strictEqual(
            stringzilla.utf8Norm(sample, form).toString(),
            sample.toString().normalize(name),
            `mismatch in ${name}`
        );
    }

    // Violation detection: composed text is clean NFC, decomposed text is flagged
    assert.strictEqual(stringzilla.utf8FindDenormalized(composed, stringzilla.Utf8NormalForm.NFC), -1n);
    assert.strictEqual(stringzilla.utf8FindDenormalized(decomposed, stringzilla.Utf8NormalForm.NFC), 3n);

    // Empty input stays empty
    assert.strictEqual(stringzilla.utf8Norm(Buffer.alloc(0), stringzilla.Utf8NormalForm.NFC).length, 0);
});

test("Utf8 Segmentation - Words, Graphemes, Sentences, Linebreaks", () => {
    const text = Buffer.from("Hello world! Пример текста.");
    const words = [...new stringzilla.Utf8Wordbreaks(text)].map((b) => b.toString());
    assert.deepStrictEqual(words, ["Hello", " ", "world", "!", " ", "Пример", " ", "текста", "."]);

    // A ZWJ family emoji is a single grapheme cluster
    const graphemes = [...new stringzilla.Utf8Graphemes(Buffer.from("a👩‍👩‍👧‍👦b"))].map((b) => b.toString());
    assert.deepStrictEqual(graphemes, ["a", "👩‍👩‍👧‍👦", "b"]);

    const sentences = [...new stringzilla.Utf8Sentences(text)].map((b) => b.toString());
    assert.deepStrictEqual(sentences, ["Hello world! ", "Пример текста."]);

    const lines = [...new stringzilla.Utf8Linebreaks(Buffer.from("one two\nthree"))].map((b) => b.toString());
    assert.strictEqual(lines.join(""), "one two\nthree");
    assert(lines.length > 1);

    // Segments are zero-copy views into the source buffer
    const buf = Buffer.from("live view");
    const first = new stringzilla.Utf8Wordbreaks(buf).next().value;
    buf[0] = "L".charCodeAt(0);
    assert.strictEqual(first.toString(), "Live");

    // Empty input yields nothing
    assert.deepStrictEqual([...new stringzilla.Utf8Wordbreaks(Buffer.alloc(0))], []);
});

test("Utf8 Segmentation - Batch Refill Beyond 64 Segments", () => {
    // Over 64 words forces the native iterator to refill its inline batch buffer
    const count = 300;
    const text = Buffer.from(Array.from({ length: count }, (_, i) => `w${i}`).join(" "));
    const words = [...new stringzilla.Utf8Wordbreaks(text)].map((b) => b.toString());
    assert.strictEqual(words.length, 2 * count - 1); // words interleaved with single spaces
    assert.strictEqual(words[0], "w0");
    assert.strictEqual(words[words.length - 1], `w${count - 1}`);
    assert.strictEqual(words.join(""), text.toString());
});
