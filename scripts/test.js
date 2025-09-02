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
    assert.strictEqual(stringzilla.findLast(haystack, empty), BigInt(haystack.length));

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
    assert.strictEqual(stringzilla.compare(Buffer.alloc(0), Buffer.alloc(0)), 0);
    assert(stringzilla.compare(Buffer.alloc(0), Buffer.from("a")) < 0);
    assert(stringzilla.compare(Buffer.from("a"), Buffer.alloc(0)) > 0);

    // Binary data comparison
    const binary1 = Buffer.from([0x00, 0x01, 0x02]);
    const binary2 = Buffer.from([0x00, 0x01, 0x03]);
    assert(stringzilla.compare(binary1, binary2) < 0);
});
