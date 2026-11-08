/**
 *  @file javascript/stringzilla.d.ts
 *  @author Ash Vardanian
 *  @date November 7, 2026
 *  @brief Type declarations for the default export of `javascript/stringzilla.js`.
 */
/// <reference types="node" />

/** A capability name, as `Capability` keys it: one capability, or the `cpus`, `gpus` and `any`
 *  groups. */
export type CapabilityName =
    | "serial"
    | "westmere"
    | "goldmont"
    | "haswell"
    | "skylake"
    | "icelake"
    | "neon"
    | "neonaes"
    | "neonsha"
    | "sve"
    | "sve2"
    | "sve2aes"
    | "rvv"
    | "rvvcrypto"
    | "v128"
    | "v128relaxed"
    | "loongsonasx"
    | "powervsx"
    | "cuda"
    | "hopper"
    | "blackwell"
    | "rocm"
    | "metal"
    | "cpus"
    | "gpus"
    | "any";

/** The runtime a device belongs to. */
export type DeviceKind = "cpu" | "cuda" | "rocm" | "metal";

/** One of the `Utf8NormalForm` constants: NFD, NFC, NFKD or NFKC. */
export type Utf8NormalFormValue = 0 | 1 | 2 | 3;

/** Where an uncased search matched: the byte `index`, -1n if not found, and the matched byte
 *  `length`. */
export interface FindResult {
    index: bigint;
    length: bigint;
}

/** Stateful hasher for streaming hash computation, for data that arrives in chunks. */
export interface Hasher {
    /**
     *  Hashes one more chunk.
     *
     *  @param buffer - Chunk to hash
     *  @returns This hasher, for chaining
     */
    update(buffer: Buffer): this;

    /** The 64-bit hash of every chunk so far. */
    digest(): bigint;

    /** Starts over with the seed the hasher was constructed with, returning this hasher. */
    reset(): this;
}

/** Constructs a `Hasher`. */
export interface HasherConstructor {
    /** @param seed - Optional seed, a non-negative integer that fits in 64 bits, defaulting to 0 */
    new (seed?: number | bigint): Hasher;
}

/** Stateful SHA-256 hasher for streaming hash computation, for data that arrives in chunks. */
export interface Sha256 {
    /**
     *  Hashes one more chunk.
     *
     *  @param buffer - Chunk to hash
     *  @returns This hasher, for chaining
     */
    update(buffer: Buffer): this;

    /** The 32-byte SHA-256 digest of every chunk so far. */
    digest(): Buffer;

    /** The SHA-256 digest of every chunk so far, as 64 lowercase hexadecimal digits. */
    hexdigest(): string;

    /** Starts over with no chunks hashed, returning this hasher. */
    reset(): this;
}

/** Constructs a `Sha256`. */
export interface Sha256Constructor {
    new (): Sha256;
}

/** Uncased UTF-8 needle prepared once, over its own copy of the bytes, for repeated searches. */
export interface Utf8UncasedNeedle {
    /**
     *  Finds the first occurrence of the needle in `haystack` under full Unicode case folding.
     *
     *  @param haystack - UTF-8 encoded haystack
     *  @param validate - If true, validates UTF-8 and throws on invalid input
     *  @returns The byte `index` and matched byte `length`; `index` is -1n if not found
     */
    findIn(haystack: Buffer, validate?: boolean): FindResult;
}

/** Constructs a `Utf8UncasedNeedle`. */
export interface Utf8UncasedNeedleConstructor {
    /**
     *  @param needle - UTF-8 encoded needle
     *  @param validate - If true, validates UTF-8 and throws on invalid input
     */
    new (needle: Buffer, validate?: boolean): Utf8UncasedNeedle;
}

/** Iterator over segments of a UTF-8 buffer, as zero-copy `subarray` views of the buffer it keeps
 *  alive. */
export interface Utf8Segments extends Iterable<Buffer> {
    /** The next segment, or `done` once the buffer is exhausted. */
    next(): IteratorResult<Buffer, undefined>;

    /** This iterator, so it can be spread or used in `for...of` directly. */
    [Symbol.iterator](): this;
}

/** Constructs a segment iterator over a UTF-8 buffer. */
export interface Utf8SegmentsConstructor {
    /**
     *  @param buffer - UTF-8 encoded input
     *  @param validate - If true, validates UTF-8 and throws on invalid input
     */
    new (buffer: Buffer, validate?: boolean): Utf8Segments;
}

/** One device StringZilla knows: the host CPU, or a GPU by its runtime's own ordinal, the one
 *  `cudaSetDevice` or `hipSetDevice` takes, or the position in Metal's device list.
 *
 *  Every call of this binding runs on the CPU and dispatches over its
 *  {@link Device.capabilitiesEnabled}, so a GPU device only reports its masks here. */
export interface Device {
    /** The runtime the device belongs to. */
    readonly kind: DeviceKind;

    /** The runtime's own index of the device. */
    readonly ordinal: number;

    /** The capabilities this device runs, compiled in or not, as a bitmask. */
    capabilitiesDetected(): bigint;

    /** The capabilities compiled into this build for devices of this kind, as a bitmask. */
    capabilitiesCompiled(): bigint;

    /** The capabilities this device's calls run: detected and compiled at once, as a bitmask.
     *  On the CPU it always includes `Capability.serial`. */
    capabilitiesEnabled(): bigint;
}

/** Constructs a `Device` and counts the devices of a kind. */
export interface DeviceConstructor {
    /**
     *  @param kind - The runtime the device belongs to
     *  @param ordinal - The runtime's own index of the device, below
     *      {@link DeviceConstructor.count}
     *  @throws Without a device of `kind`, or past the last one
     */
    new (kind: DeviceKind, ordinal?: number): Device;

    /** The host CPU, which every build has. */
    cpu(): Device;

    /**
     *  How many devices of `kind` the process sees: one CPU, or the GPUs its runtime counts.
     *  @throws Without a GPU of `kind`
     */
    count(kind: DeviceKind): number;
}

/** The StringZilla bindings, the default export of `stringzilla`. */
export interface StringZilla {
    /**
     *  Searches for a short buffer in a long one, zero-copy.
     *
     *  @param haystack - Buffer to search in
     *  @param needle - Buffer to search for
     *  @returns Index of needle in haystack, or -1n if not found
     */
    find(haystack: Buffer, needle: Buffer): bigint;

    /**
     *  Searches for the last occurrence of a short buffer in a long one, zero-copy.
     *
     *  @param haystack - Buffer to search in
     *  @param needle - Buffer to search for
     *  @returns Index of last needle in haystack, or -1n if not found
     */
    findLast(haystack: Buffer, needle: Buffer): bigint;

    /**
     *  Finds the first occurrence of a specific byte value, zero-copy.
     *
     *  @param haystack - Buffer to search in
     *  @param byte - Byte value to search for (0-255)
     *  @returns Index of byte in haystack, or -1n if not found
     */
    findByte(haystack: Buffer, byte: number): bigint;

    /**
     *  Finds the last occurrence of a specific byte value, zero-copy.
     *
     *  @param haystack - Buffer to search in
     *  @param byte - Byte value to search for (0-255)
     *  @returns Index of last byte in haystack, or -1n if not found
     */
    findLastByte(haystack: Buffer, byte: number): bigint;

    /**
     *  Finds the first occurrence of any byte from a set, zero-copy.
     *
     *  @param haystack - Buffer to search in
     *  @param charset - Buffer containing allowed byte values
     *  @returns Index of first matching byte in haystack, or -1n if not found
     */
    findByteFrom(haystack: Buffer, charset: Buffer): bigint;

    /**
     *  Finds the last occurrence of any byte from a set, zero-copy.
     *
     *  @param haystack - Buffer to search in
     *  @param charset - Buffer containing allowed byte values
     *  @returns Index of last matching byte in haystack, or -1n if not found
     */
    findLastByteFrom(haystack: Buffer, charset: Buffer): bigint;

    /**
     *  Counts occurrences of a buffer in a larger buffer, zero-copy.
     *
     *  @param haystack - Buffer to search in
     *  @param needle - Buffer to search for
     *  @param overlap - Whether to count overlapping matches
     *  @returns Number of matches found; an empty needle matches `haystack.length + 1` times
     */
    count(haystack: Buffer, needle: Buffer, overlap?: boolean): bigint;

    /**
     *  Computes hash of a buffer using StringZilla's fast hash algorithm, zero-copy.
     *
     *  @param buffer - Buffer to hash
     *  @param seed - Optional seed for hash, defaulting to 0
     *  @returns 64-bit hash value
     */
    hash(buffer: Buffer, seed?: number | bigint): bigint;

    /** Stateful hasher class for streaming hash computation.
     *  Use this for hashing data that arrives in chunks. */
    Hasher: HasherConstructor;

    /**
     *  Computes SHA-256 cryptographic hash of a buffer, zero-copy.
     *
     *  @param buffer - Buffer to hash
     *  @returns 32-byte SHA-256 digest
     */
    sha256(buffer: Buffer): Buffer;

    /** Stateful SHA-256 hasher class for streaming hash computation.
     *  Use this for hashing data that arrives in chunks. */
    Sha256: Sha256Constructor;

    /**
     *  Compares two buffers for equality, zero-copy.
     *
     *  @param first - First buffer to compare
     *  @param second - Second buffer to compare
     *  @returns True if buffers are equal, false otherwise
     */
    equal(first: Buffer, second: Buffer): boolean;

    /**
     *  Compares two buffers lexicographically, zero-copy.
     *
     *  @param first - First buffer to compare
     *  @param second - Second buffer to compare
     *  @returns -1 if first < second, 0 if equal, 1 if first > second
     */
    compare(first: Buffer, second: Buffer): number;

    /**
     *  Computes the sum of all byte values in a buffer, zero-copy.
     *
     *  @param buffer - Buffer to sum
     *  @returns Sum of all byte values
     */
    byteSum(buffer: Buffer): bigint;

    /** The CPU or a GPU, whose capabilities it reports; every call dispatches over the CPU's. */
    Device: DeviceConstructor;

    /** Lowercase capability names, like `haswell`, `neon` or `cuda`, mapped to their `BigInt` bits,
     *  and the `cpus`, `gpus` and `any` groups to theirs. */
    readonly Capability: Readonly<Record<CapabilityName, bigint>>;

    /**
     *  Applies full Unicode case folding to a UTF-8 buffer.
     *
     *  @param buffer - UTF-8 encoded input
     *  @param validate - If true, validates UTF-8 and throws on invalid input
     *  @returns Case-folded UTF-8 bytes, possibly longer than the input due to expansions
     */
    utf8UncasedFold(buffer: Buffer, validate?: boolean): Buffer;

    /**
     *  Finds the first occurrence of `needle` in `haystack` under full Unicode case folding.
     *
     *  @param haystack - UTF-8 encoded haystack
     *  @param needle - UTF-8 encoded needle
     *  @param validate - If true, validates UTF-8 and throws on invalid input
     *  @returns Object with the byte `index` and matched byte `length`; `index` is -1n if not found
     */
    utf8UncasedFind(haystack: Buffer, needle: Buffer, validate?: boolean): FindResult;

    /** Uncased UTF-8 needle prepared once, over its own copy of the bytes, for repeated searches.
     *
     *  Construct with `new`, then call `findIn(haystack, validate?)`. */
    Utf8UncasedNeedle: Utf8UncasedNeedleConstructor;

    /**
     *  Counts the Unicode codepoints in a UTF-8 buffer.
     *
     *  JavaScript strings are UTF-16, so `String.prototype.length` counts code units and
     *  disagrees with this for anything outside the Basic Multilingual Plane.
     *
     *  @param buffer - UTF-8 encoded input
     *  @returns Number of codepoints
     */
    utf8Count(buffer: Buffer): bigint;

    /**
     *  Resolves a codepoint index to a byte offset in a UTF-8 buffer.
     *
     *  Every offset the other exports return is a byte offset, so this is the bridge between
     *  those and codepoint-indexed positions.
     *
     *  @param buffer - UTF-8 encoded input
     *  @param index - Zero-based codepoint index
     *  @returns Byte offset of that codepoint, or -1n if the buffer holds fewer
     */
    utf8Seek(buffer: Buffer, index: number | bigint): bigint;

    /** Unicode normalization form constants for `utf8Norm` and `utf8FindDenormalized`. */
    readonly Utf8NormalForm: { readonly NFD: 0; readonly NFC: 1; readonly NFKD: 2; readonly NFKC: 3 };

    /**
     *  Normalizes a UTF-8 buffer into the requested Unicode normal form.
     *
     *  @param buffer - UTF-8 encoded input
     *  @param form - One of the `Utf8NormalForm` constants
     *  @param validate - If true, validates UTF-8 and throws on invalid input
     *  @returns Normalized UTF-8 bytes
     */
    utf8Norm(buffer: Buffer, form: Utf8NormalFormValue, validate?: boolean): Buffer;

    /**
     *  Finds the first byte violating the requested Unicode normal form.
     *
     *  @param buffer - UTF-8 encoded input
     *  @param form - One of the `Utf8NormalForm` constants
     *  @returns Byte index of the first violation, or -1n if already normalized
     */
    utf8FindDenormalized(buffer: Buffer, form: Utf8NormalFormValue): bigint;

    /** Iterator over TR29 word segments as zero-copy `subarray` views of the buffer it keeps alive:
     *  `for (const word of new sz.Utf8Wordbreaks(buffer, validate?)) ...` */
    Utf8Wordbreaks: Utf8SegmentsConstructor;

    /** Iterator over TR29 grapheme clusters, including multi-codepoint ZWJ emoji. */
    Utf8Graphemes: Utf8SegmentsConstructor;

    /** Iterator over TR29 sentence segments. */
    Utf8Sentences: Utf8SegmentsConstructor;

    /** Iterator over UAX14 line-break opportunities. */
    Utf8Linebreaks: Utf8SegmentsConstructor;
}

declare const stringzilla: StringZilla;
export default stringzilla;
