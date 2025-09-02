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
};
