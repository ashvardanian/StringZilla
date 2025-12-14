import bindings from "bindings";

const compiled = bindings("stringzilla");

export default {
    /**
     *  Searches for a short buffer in a long one (zero-copy).
     *
     *  @param {Buffer} haystack - Buffer to search in
     *  @param {Buffer} needle - Buffer to search for
     *  @returns {bigint} Index of needle in haystack, or -1n if not found
     */
    find: compiled.indexOf,

    /**
     *  Searches for the last occurrence of a short buffer in a long one (zero-copy).
     *
     *  @param {Buffer} haystack - Buffer to search in
     *  @param {Buffer} needle - Buffer to search for
     *  @returns {bigint} Index of last needle in haystack, or -1n if not found
     */
    findLast: compiled.lastIndexOf,

    /**
     *  Finds the first occurrence of a specific byte value (zero-copy).
     *
     *  @param {Buffer} haystack - Buffer to search in
     *  @param {number} byte - Byte value to search for (0-255)
     *  @returns {bigint} Index of byte in haystack, or -1n if not found
     */
    findByte: compiled.findByte,

    /**
     *  Finds the last occurrence of a specific byte value (zero-copy).
     *
     *  @param {Buffer} haystack - Buffer to search in
     *  @param {number} byte - Byte value to search for (0-255)
     *  @returns {bigint} Index of last byte in haystack, or -1n if not found
     */
    findLastByte: compiled.findLastByte,

    /**
     *  Finds the first occurrence of any byte from a set (zero-copy).
     *
     *  @param {Buffer} haystack - Buffer to search in
     *  @param {Buffer} charset - Buffer containing allowed byte values
     *  @returns {bigint} Index of first matching byte in haystack, or -1n if not found
     */
    findByteFrom: compiled.findByteFrom,

    /**
     *  Finds the last occurrence of any byte from a set (zero-copy).
     *
     *  @param {Buffer} haystack - Buffer to search in
     *  @param {Buffer} charset - Buffer containing allowed byte values
     *  @returns {bigint} Index of last matching byte in haystack, or -1n if not found
     */
    findLastByteFrom: compiled.findLastByteFrom,

    /**
     *  Counts occurrences of a buffer in a larger buffer (zero-copy).
     *
     *  @param {Buffer} haystack - Buffer to search in
     *  @param {Buffer} needle - Buffer to search for
     *  @param {boolean} overlap - Whether to count overlapping matches
     *  @returns {bigint} Number of matches found
     */
    count: compiled.count,

    /**
     *  Computes hash of a buffer using StringZilla's fast hash algorithm (zero-copy).
     *
     *  @param {Buffer} buffer - Buffer to hash
     *  @param {bigint|number} seed - Optional seed for hash (default: 0)
     *  @returns {bigint} 64-bit hash value
     */
    hash: compiled.hash,

    /**
     *  Stateful hasher class for streaming hash computation.
     *  Use this for hashing data that arrives in chunks.
     */
    Hasher: compiled.Hasher,

    /**
     *  Computes SHA-256 cryptographic hash of a buffer (zero-copy).
     *
     *  @param {Buffer} buffer - Buffer to hash
     *  @returns {Buffer} 32-byte SHA-256 digest
     */
    sha256: compiled.sha256,

    /**
     *  Stateful SHA-256 hasher class for streaming hash computation.
     *  Use this for hashing data that arrives in chunks.
     */
    Sha256: compiled.Sha256,

    /**
     *  Compares two buffers for equality (zero-copy).
     *
     *  @param {Buffer} first - First buffer to compare
     *  @param {Buffer} second - Second buffer to compare
     *  @returns {boolean} True if buffers are equal, false otherwise
     */
    equal: compiled.equal,

    /**
     *  Compares two buffers lexicographically (zero-copy).
     *
     *  @param {Buffer} first - First buffer to compare
     *  @param {Buffer} second - Second buffer to compare
     *  @returns {number} -1 if first < second, 0 if equal, 1 if first > second
     */
    compare: compiled.compare,

    /**
     *  Computes the sum of all byte values in a buffer (zero-copy).
     *
     *  @param {Buffer} buffer - Buffer to sum
     *  @returns {bigint} Sum of all byte values
     */
    byteSum: compiled.byteSum,

    /**
     *  Returns a comma-separated string of backend capabilities, e.g. "serial,haswell".
     *  Use this to inspect which SIMD/GPU backends are active.
     *  @returns {string}
     */
    capabilities: compiled.capabilities,

    /**
     *  Applies full Unicode case folding to a UTF-8 buffer.
     *
     *  @param {Buffer} buffer - UTF-8 encoded input
     *  @param {boolean} validate - If true, validates UTF-8 and throws on invalid input
     *  @returns {Buffer} Case-folded UTF-8 bytes (may be longer than input due to expansions)
     */
    utf8CaseFold: compiled.utf8CaseFold,

    /**
     *  Finds the first case-insensitive occurrence of `needle` in `haystack` using full Unicode case folding.
     *
     *  @param {Buffer} haystack - UTF-8 encoded haystack
     *  @param {Buffer} needle - UTF-8 encoded needle
     *  @param {boolean} validate - If true, validates UTF-8 and throws on invalid input
     *  @returns {{index: bigint, length: bigint}} Object with byte index and matched byte length; `index` is -1n if not found
     */
    utf8CaseInsensitiveFind: compiled.utf8CaseInsensitiveFind,

    /**
     *  Precompiled case-insensitive UTF-8 needle for repeated searches.
     *
     *  Construct with `new`, then call `findIn(haystack, validate?)`.
     */
    Utf8CaseInsensitiveNeedle: compiled.Utf8CaseInsensitiveNeedle,
};
