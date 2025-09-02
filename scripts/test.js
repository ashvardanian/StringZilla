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
