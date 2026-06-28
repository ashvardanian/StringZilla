/**
 *  @brief Zero-copy Java bindings for the StringZilla core C library, via the Foreign Function &
 *         Memory API (FFM, JDK 22+). No JNI.
 *  @file java/src/main/java/com/stringzilla/StringZilla.java
 *  @author Ash Vardanian
 *
 *  Exposes the core single-string operations over UTF-8 byte buffers:
 *
 *  - Search: `indexOf`, `lastIndexOf`, and byte-set search
 *  - Comparison: `equal` and `compare`
 *  - Hashing: `hash`, `byteSum`, incremental `Hasher`, and `Sha256`
 *  - UTF-8: codepoint count/seek/decode, segmentation, case folding, and normalization
 *  - Collections: `argSort` and `intersect`
 *
 *  All offsets are UTF-8 byte offsets. Inputs may be on-heap `byte[]` or off-heap `MemorySegment`
 *  (e.g. a Lucene `BytesRef` or Spark `UTF8String`); see the README for zero-copy guidance.
 */
package com.stringzilla;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Core StringZilla operations: search, compare, hash. Pure FFM, JDK 22+. */
public final class StringZilla {

    private StringZilla() {}

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP = NativeLoader.load();

    private static final MethodHandle SZ_FIND =
            down("sz_find", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));
    private static final MethodHandle SZ_RFIND =
            down("sz_rfind", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));
    private static final MethodHandle SZ_FIND_BYTESET =
            down("sz_find_byteset", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle SZ_RFIND_BYTESET =
            down("sz_rfind_byteset", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));

    // Value-returning ops: `critical(true)` so on-heap byte[] is read with no copy and no
    // thread-state transition.
    private static final MethodHandle SZ_HASH =
            downCritical("sz_hash", FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_LONG));
    private static final MethodHandle SZ_BYTESUM =
            downCritical("sz_bytesum", FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG));
    private static final MethodHandle SZ_EQUAL =
            downCritical("sz_equal", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG));
    private static final MethodHandle SZ_ORDER =
            downCritical("sz_order", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));

    private static final MethodHandle SZ_HASH_STATE_INIT =
            down("sz_hash_state_init", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG));
    private static final MethodHandle SZ_HASH_STATE_UPDATE =
            downCritical("sz_hash_state_update", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG));
    private static final MethodHandle SZ_HASH_STATE_DIGEST =
            down("sz_hash_state_digest", FunctionDescriptor.of(JAVA_LONG, ADDRESS));

    private static final MethodHandle SZ_CAPABILITIES = down("sz_capabilities", FunctionDescriptor.of(JAVA_INT));
    private static final MethodHandle SZ_CAPABILITIES_TO_STRING =
            down("sz_capabilities_to_string", FunctionDescriptor.of(ADDRESS, JAVA_INT));
    private static final MethodHandle SZ_VERSION_MAJOR = down("sz_version_major", FunctionDescriptor.of(JAVA_INT));
    private static final MethodHandle SZ_VERSION_MINOR = down("sz_version_minor", FunctionDescriptor.of(JAVA_INT));
    private static final MethodHandle SZ_VERSION_PATCH = down("sz_version_patch", FunctionDescriptor.of(JAVA_INT));

    // region UTF-8 Codepoints
    private static final MethodHandle SZ_UTF8_COUNT =
            downCritical("sz_utf8_count", FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG));
    private static final MethodHandle SZ_UTF8_SEEK =
            down("sz_utf8_seek", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG));
    private static final MethodHandle SZ_UTF8_DECODE = downCritical(
            "sz_utf8_decode", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));

    // endregion

    // region UTF-8 Segmentation
    private static final MethodHandle SZ_UTF8_GRAPHEMES = downSeg("sz_utf8_graphemes");
    private static final MethodHandle SZ_UTF8_WORDS = downSeg("sz_utf8_words");
    private static final MethodHandle SZ_UTF8_SENTENCES = downSeg("sz_utf8_sentences");
    private static final MethodHandle SZ_UTF8_LINEBREAKS = downSeg("sz_utf8_linebreaks");
    private static final MethodHandle SZ_UTF8_NEWLINES = downSeg("sz_utf8_newlines");
    private static final MethodHandle SZ_UTF8_WHITESPACES = downSeg("sz_utf8_whitespaces");
    private static final MethodHandle SZ_UTF8_DELIMITERS = downSeg("sz_utf8_delimiters");

    // endregion

    // region UTF-8 Case Folding and Uncased
    private static final MethodHandle SZ_UTF8_UNCASED_FOLD =
            downCritical("sz_utf8_uncased_fold", FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle SZ_UTF8_UNCASED_SEARCH = down(
            "sz_utf8_uncased_search",
            FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS));
    private static final MethodHandle SZ_UTF8_UNCASED_ORDER = downCritical(
            "sz_utf8_uncased_order", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));

    // endregion

    // region Hash Extras, SHA-256, Random and Lookup
    private static final MethodHandle SZ_HASH_MULTISEED = downCritical(
            "sz_hash_multiseed", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle SZ_SHA256_INIT = down("sz_sha256_state_init", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle SZ_SHA256_UPDATE =
            downCritical("sz_sha256_state_update", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG));
    private static final MethodHandle SZ_SHA256_DIGEST =
            downCritical("sz_sha256_state_digest", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
    private static final MethodHandle SZ_FILL_RANDOM =
            downCritical("sz_fill_random", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_LONG));
    private static final MethodHandle SZ_LOOKUP =
            downCritical("sz_lookup", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, ADDRESS, ADDRESS));

    private static MethodHandle downSeg(String name) {
        return downCritical(
                name, FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
    }

    // endregion

    // region Normalization and Collections
    private static final MethodHandle SZ_UTF8_NORM =
            downCritical("sz_utf8_norm", FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS));
    private static final MethodHandle SZ_UTF8_FIND_DENORMALIZED =
            downCritical("sz_utf8_find_denormalized", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT));
    private static final MethodHandle SZ_SEQUENCE_ARGSORT = down(
            "sz_sequence_argsort", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT));
    private static final MethodHandle SZ_SEQUENCE_ARGSORT_UNCASED = down(
            "sz_sequence_argsort_uncased",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT));
    private static final MethodHandle SZ_SEQUENCE_INTERSECT = down(
            "sz_sequence_intersect",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS));

    // endregion

    static {
        try {
            down("sz_dispatch_table_init", FunctionDescriptor.ofVoid()).invokeExact();
        } catch (Throwable t) {
            throw new ExceptionInInitializerError(t);
        }
    }

    private static MethodHandle down(String name, FunctionDescriptor desc, Linker.Option... opts) {
        MemorySegment addr = LOOKUP.find(name).orElseThrow(() -> new UnsatisfiedLinkError("missing symbol: " + name));
        return LINKER.downcallHandle(addr, desc, opts);
    }

    private static MethodHandle downCritical(String name, FunctionDescriptor desc) {
        return down(name, desc, Linker.Option.critical(true));
    }

    // region Search (MemorySegment)

    /** First byte offset of {@code needle} in {@code haystack}, or -1. */
    public static long indexOf(MemorySegment haystack, MemorySegment needle) {
        long n = needle.byteSize();
        if (n == 0) return 0;
        if (n > haystack.byteSize()) return -1;
        try {
            MemorySegment r = (MemorySegment) SZ_FIND.invokeExact(haystack, haystack.byteSize(), needle, n);
            return r.address() == 0 ? -1 : r.address() - haystack.address();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static long lastIndexOf(MemorySegment haystack, MemorySegment needle) {
        long n = needle.byteSize();
        if (n == 0) return haystack.byteSize();
        if (n > haystack.byteSize()) return -1;
        try {
            MemorySegment r = (MemorySegment) SZ_RFIND.invokeExact(haystack, haystack.byteSize(), needle, n);
            return r.address() == 0 ? -1 : r.address() - haystack.address();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // endregion

    // region Search (byte[])

    /** First byte offset of {@code needle} in {@code haystack}, or -1. Like {@link String#indexOf(String)},
     *  but SIMD-accelerated over UTF-8 bytes. Keywords: find, search, substring. */
    public static long indexOf(byte[] haystack, byte[] needle) {
        return indexOf(haystack, 0, haystack.length, needle);
    }

    public static long indexOf(byte[] haystack, int off, int len, byte[] needle) {
        if (needle.length == 0) return 0;
        if (needle.length > len) return -1;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment h = a.allocate(len);
            MemorySegment.copy(haystack, off, h, JAVA_BYTE, 0, len);
            MemorySegment n = a.allocate(needle.length);
            MemorySegment.copy(needle, 0, n, JAVA_BYTE, 0, needle.length);
            return indexOf(h, n);
        }
    }

    public static long lastIndexOf(byte[] haystack, byte[] needle) {
        if (needle.length == 0) return haystack.length;
        if (needle.length > haystack.length) return -1;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment h = a.allocate(haystack.length);
            MemorySegment.copy(haystack, 0, h, JAVA_BYTE, 0, haystack.length);
            MemorySegment n = a.allocate(needle.length);
            MemorySegment.copy(needle, 0, n, JAVA_BYTE, 0, needle.length);
            return lastIndexOf(h, n);
        }
    }

    /** First offset of any byte in {@code set}, or -1. */
    public static long indexOfAny(byte[] haystack, Byteset set) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment h = a.allocate(haystack.length == 0 ? 1 : haystack.length);
            MemorySegment.copy(haystack, 0, h, JAVA_BYTE, 0, haystack.length);
            MemorySegment s = set.toSegment(a);
            MemorySegment r = (MemorySegment) SZ_FIND_BYTESET.invokeExact(h, (long) haystack.length, s);
            return r.address() == 0 ? -1 : r.address() - h.address();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static long lastIndexOfAny(byte[] haystack, Byteset set) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment h = a.allocate(haystack.length == 0 ? 1 : haystack.length);
            MemorySegment.copy(haystack, 0, h, JAVA_BYTE, 0, haystack.length);
            MemorySegment s = set.toSegment(a);
            MemorySegment r = (MemorySegment) SZ_RFIND_BYTESET.invokeExact(h, (long) haystack.length, s);
            return r.address() == 0 ? -1 : r.address() - h.address();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // endregion

    // region Comparison

    /** Byte-wise equality. Like {@link java.util.Arrays#equals(byte[], byte[])}, SIMD-accelerated. */
    public static boolean equal(byte[] a, byte[] b) {
        if (a.length != b.length) return false;
        if (a.length == 0) return true;
        try {
            int eq = (int) SZ_EQUAL.invokeExact(MemorySegment.ofArray(a), MemorySegment.ofArray(b), (long) a.length);
            return eq != 0;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Lexicographic byte comparison: -1, 0, or 1. Like {@link java.util.Arrays#compare(byte[], byte[])}. */
    public static int compare(byte[] a, byte[] b) {
        try {
            return (int) SZ_ORDER.invokeExact(
                    MemorySegment.ofArray(a), (long) a.length, MemorySegment.ofArray(b), (long) b.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // endregion

    // region Hashing

    /** Fast 64-bit AES-based hash (SMHasher-quality), stable across runs and identical across all
     *  StringZilla bindings — unlike {@link String#hashCode()}. For cryptographic hashing use {@link Sha256}. */
    public static long hash(byte[] data) {
        return hash(data, 0L);
    }

    public static long hash(byte[] data, long seed) {
        try {
            return (long) SZ_HASH.invokeExact(MemorySegment.ofArray(data), (long) data.length, seed);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static long hash(MemorySegment data, long seed) {
        try {
            return (long) SZ_HASH.invokeExact(data, data.byteSize(), seed);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static long byteSum(byte[] data) {
        try {
            return (long) SZ_BYTESUM.invokeExact(MemorySegment.ofArray(data), (long) data.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // endregion

    // region UTF-8 Codepoints

    /** Counts Unicode scalar values (codepoints). Like {@code String.codePointCount} /
     *  {@link Character#codePointCount}, SIMD-accelerated over UTF-8 bytes. */
    public static long countRunes(byte[] text) {
        try {
            return (long) SZ_UTF8_COUNT.invokeExact(MemorySegment.ofArray(text), (long) text.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Byte offset of the {@code index}-th codepoint (0-based), or -1 if fewer exist. */
    public static long seekRune(byte[] text, long index) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment t = a.allocate(Math.max(text.length, 1));
            MemorySegment.copy(text, 0, t, JAVA_BYTE, 0, text.length);
            MemorySegment r = (MemorySegment) SZ_UTF8_SEEK.invokeExact(t, (long) text.length, index);
            return r.address() == 0 ? -1 : r.address() - t.address();
        } catch (Throwable e) {
            throw rethrow(e);
        }
    }

    /** Decodes codepoints into {@code destination} (one Int32 scalar each; ill-formed → U+FFFD) and
     *  returns the count written. Like {@code String.codePoints()}, SIMD-accelerated. */
    public static int decode(byte[] text, int[] destination) {
        long[] unpacked = new long[1];
        try {
            MemorySegment cursorIgnored = (MemorySegment) SZ_UTF8_DECODE.invokeExact(
                    MemorySegment.ofArray(text),
                    (long) text.length,
                    MemorySegment.ofArray(destination),
                    (long) destination.length,
                    MemorySegment.ofArray(unpacked));
            return (int) unpacked[0];
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Allocating convenience: returns all codepoints as a fresh Int32 array. */
    public static int[] decodeAll(byte[] text) {
        int[] out = new int[(int) countRunes(text)];
        decode(text, out);
        return out;
    }

    // endregion

    // region UTF-8 Segmentation

    public enum SegmentKind {
        GRAPHEMES,
        WORDS,
        SENTENCES,
        LINE_BREAKS,
        NEWLINES,
        WHITESPACES,
        DELIMITERS
    }

    private static MethodHandle segHandle(SegmentKind k) {
        return switch (k) {
            case GRAPHEMES -> SZ_UTF8_GRAPHEMES;
            case WORDS -> SZ_UTF8_WORDS;
            case SENTENCES -> SZ_UTF8_SENTENCES;
            case LINE_BREAKS -> SZ_UTF8_LINEBREAKS;
            case NEWLINES -> SZ_UTF8_NEWLINES;
            case WHITESPACES -> SZ_UTF8_WHITESPACES;
            case DELIMITERS -> SZ_UTF8_DELIMITERS;
        };
    }

    /** Writes the byte (start, length) ranges of the next segments into the caller's {@code starts}
     *  and {@code lengths} arrays and returns the count written. Allocation-free. Resumable: advance
     *  the input by {@code bytesConsumed[0]} and call again until it returns 0. Starts are relative to
     *  {@code textOffset}. Like {@link java.text.BreakIterator} / ICU4J, but deterministic, SIMD-accelerated,
     *  over UTF-8 bytes (no locale, no transcode). */
    public static int segment(
            byte[] text,
            int textOffset,
            int textLength,
            SegmentKind kind,
            long[] starts,
            long[] lengths,
            long[] bytesConsumed) {
        int cap = Math.min(starts.length, lengths.length);
        MethodHandle h = segHandle(kind);
        MemorySegment slice = MemorySegment.ofArray(text).asSlice(textOffset, textLength);
        try {
            long count = (long) h.invokeExact(
                    slice,
                    (long) textLength,
                    MemorySegment.ofArray(starts),
                    MemorySegment.ofArray(lengths),
                    (long) cap,
                    MemorySegment.ofArray(bytesConsumed));
            return (int) count;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // endregion

    // region UTF-8 Case Folding and Uncased Matching

    /** Folds into the caller's {@code destination} (must hold ≥ text.length*3 bytes) and returns the
     *  bytes written. Allocation-free. */
    public static int caseFold(byte[] text, byte[] destination) {
        if (destination.length < text.length * 3)
            throw new IllegalArgumentException("destination must hold at least text.length*3 bytes");
        try {
            return (int) (long) SZ_UTF8_UNCASED_FOLD.invokeExact(
                    MemorySegment.ofArray(text), (long) text.length, MemorySegment.ofArray(destination));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Full Unicode case folding (UAX #21, one-to-many, e.g. ß → "ss"). Fills a gap: the JDK ships
     *  no case folding (only locale {@code toLowerCase}/{@code toUpperCase}). */
    public static byte[] caseFold(byte[] text) {
        if (text.length == 0) return new byte[0];
        byte[] dst = new byte[text.length * 3]; // worst-case 3x expansion
        try {
            long n = (long) SZ_UTF8_UNCASED_FOLD.invokeExact(
                    MemorySegment.ofArray(text), (long) text.length, MemorySegment.ofArray(dst));
            return java.util.Arrays.copyOf(dst, (int) n);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Result of a case-insensitive search: byte {@code offset} (-1 if not found) and the matched
     *  byte length (may differ from the needle due to folding). */
    public record Match(long offset, long matchedLength) {}

    /** Case-insensitive (full-fold) search. Fills a gap: no JDK caseless UTF-8 substring search.
     *  For repeated searches with the same needle, prefer {@link UncasedNeedle}. */
    public static Match uncasedIndexOf(byte[] haystack, byte[] needle) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment h = a.allocate(Math.max(haystack.length, 1));
            MemorySegment.copy(haystack, 0, h, JAVA_BYTE, 0, haystack.length);
            MemorySegment n = a.allocate(Math.max(needle.length, 1));
            MemorySegment.copy(needle, 0, n, JAVA_BYTE, 0, needle.length);
            MemorySegment meta = a.allocate(64); // zeroed; the native dereferences it (NULL is unsafe)
            MemorySegment ml = a.allocate(JAVA_LONG.byteSize());
            MemorySegment r = (MemorySegment)
                    SZ_UTF8_UNCASED_SEARCH.invokeExact(h, (long) haystack.length, n, (long) needle.length, meta, ml);
            long off = r.address() == 0 ? -1 : r.address() - h.address();
            return new Match(off, off < 0 ? 0 : ml.get(JAVA_LONG, 0));
        } catch (Throwable e) {
            throw rethrow(e);
        }
    }

    /** Case-insensitive (full-fold) lexicographic comparison: -1, 0, or 1. Fills a gap: no JDK
     *  Unicode case-folded ordinal comparison. */
    public static int uncasedCompare(byte[] a, byte[] b) {
        try {
            return (int) SZ_UTF8_UNCASED_ORDER.invokeExact(
                    MemorySegment.ofArray(a), (long) a.length, MemorySegment.ofArray(b), (long) b.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // endregion

    // region Hash Extras, Random and Lookup

    /** Hashes {@code data} under each seed, writing results into {@code hashes} (must hold ≥ seeds.length). */
    public static void hashMultiseed(byte[] data, long[] seeds, long[] hashes) {
        if (hashes.length < seeds.length)
            throw new IllegalArgumentException("hashes must hold at least seeds.length entries");
        try {
            SZ_HASH_MULTISEED.invokeExact(
                    MemorySegment.ofArray(data),
                    (long) data.length,
                    MemorySegment.ofArray(seeds),
                    (long) seeds.length,
                    MemorySegment.ofArray(hashes));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Allocating convenience: returns one hash per seed. */
    public static long[] hashMultiseed(byte[] data, long[] seeds) {
        long[] out = new long[seeds.length];
        hashMultiseed(data, seeds, out);
        return out;
    }

    /** Fills {@code buffer} with deterministic pseudo-random bytes derived from {@code nonce}. */
    public static void fillRandom(byte[] buffer, long nonce) {
        try {
            SZ_FILL_RANDOM.invokeExact(MemorySegment.ofArray(buffer), (long) buffer.length, nonce);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Byte-wise table transform: {@code destination[i] = lut[source[i]]}. {@code lut} must be 256 bytes. */
    public static void lookup(byte[] destination, byte[] source, byte[] lut) {
        if (lut.length != 256) throw new IllegalArgumentException("LUT must be 256 bytes");
        if (destination.length < source.length) throw new IllegalArgumentException("destination too small");
        try {
            SZ_LOOKUP.invokeExact(
                    MemorySegment.ofArray(destination),
                    (long) source.length,
                    MemorySegment.ofArray(source),
                    MemorySegment.ofArray(lut));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // endregion

    // region UTF-8 Normalization

    public enum NormalForm {
        NFD,
        NFC,
        NFKD,
        NFKC
    }

    private static int formCode(NormalForm f) {
        return switch (f) {
            case NFD -> 0;
            case NFC -> 1;
            case NFKD -> 2;
            case NFKC -> 3;
        };
    }

    /** Normalizes into the caller's {@code destination} (must hold ≥ text.length*18 bytes) and returns
     *  the bytes written. Allocation-free. */
    public static int normalize(byte[] text, NormalForm form, byte[] destination) {
        if (destination.length < text.length * 18)
            throw new IllegalArgumentException("destination must hold at least text.length*18 bytes");
        try {
            return (int) (long) SZ_UTF8_NORM.invokeExact(
                    MemorySegment.ofArray(text),
                    (long) text.length,
                    formCode(form),
                    MemorySegment.ofArray(destination));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Normalizes {@code text} to the given form. Equivalent to {@link java.text.Normalizer#normalize},
     *  over UTF-8 bytes with no UTF-16 transcode. */
    public static byte[] normalize(byte[] text, NormalForm form) {
        if (text.length == 0) return new byte[0];
        byte[] dst = new byte[text.length * 18]; // worst-case per-codepoint expansion
        try {
            long n = (long) SZ_UTF8_NORM.invokeExact(
                    MemorySegment.ofArray(text), (long) text.length, formCode(form), MemorySegment.ofArray(dst));
            return java.util.Arrays.copyOf(dst, (int) n);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** True if {@code text} is already in the given form. Like {@link java.text.Normalizer#isNormalized}. */
    public static boolean isNormalized(byte[] text, NormalForm form) {
        try {
            MemorySegment r = (MemorySegment) SZ_UTF8_FIND_DENORMALIZED.invokeExact(
                    MemorySegment.ofArray(text), (long) text.length, formCode(form));
            return r.address() == 0;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // endregion

    // region Collections

    /** Returns the permutation that sorts {@code items} lexicographically by bytes (stable).
     *  {@code top} &gt; 0 partial-sorts the smallest N. Like {@code Collections.sort} with an ordinal
     *  comparator, but returns indices over UTF-8 bytes. */
    /** Writes the sorting permutation of {@code items} into {@code order} (must hold ≥ items.size()
     *  entries) and returns the count of sorted indices ({@code top} if &gt; 0, else items.size()).
     *  Like {@code Collections.sort} with an ordinal comparator, but returns indices over UTF-8 bytes. */
    public static int argSort(java.util.List<byte[]> items, long[] order, boolean reverse, long top, boolean uncased) {
        int n = items.size();
        if (n == 0) return 0;
        if (order.length < n) throw new IllegalArgumentException("order must hold at least " + n + " entries");
        try (Arena a = Arena.ofConfined()) {
            SeqTable t = new SeqTable(items, a);
            MemorySegment orderSeg = a.allocate(JAVA_LONG.byteSize() * n);
            MethodHandle mh = uncased ? SZ_SEQUENCE_ARGSORT_UNCASED : SZ_SEQUENCE_ARGSORT;
            int status = (int) mh.invokeExact(t.sequence, MemorySegment.NULL, orderSeg, top, reverse ? 1 : 0);
            if (status != 0) throw new IllegalStateException("sz_sequence_argsort failed with status " + status);
            int count = top > 0 ? (int) Math.min(top, n) : n;
            MemorySegment.copy(orderSeg, JAVA_LONG, 0, order, 0, count);
            return count;
        } catch (Throwable e) {
            throw rethrow(e);
        }
    }

    /** Allocating convenience: returns a fresh permutation array. */
    public static long[] argSort(java.util.List<byte[]> items, boolean reverse, long top, boolean uncased) {
        int n = items.size();
        if (n == 0) return new long[0];
        long[] order = new long[n];
        int count = argSort(items, order, reverse, top, uncased);
        return count == n ? order : java.util.Arrays.copyOf(order, count);
    }

    public static long[] argSort(java.util.List<byte[]> items) {
        return argSort(items, false, 0, false);
    }

    /** Inner-join two de-duplicated sequences. Returns {@code {firstPositions, secondPositions}} —
     *  the matching index pairs into {@code a} and {@code b}. */
    /** Inner-joins two de-duplicated sequences, writing matching index pairs into {@code firstPositions}
     *  and {@code secondPositions} (each must hold ≥ min(a.size(), b.size()) entries) and returns the
     *  match count. Allocation-free for the result. */
    public static int intersect(
            java.util.List<byte[]> a,
            java.util.List<byte[]> b,
            long[] firstPositions,
            long[] secondPositions,
            long seed) {
        int na = a.size(), nb = b.size();
        if (na == 0 || nb == 0) return 0;
        int minN = Math.min(na, nb);
        if (firstPositions.length < minN || secondPositions.length < minN)
            throw new IllegalArgumentException("position arrays must hold at least " + minN + " entries");
        try (Arena ar = Arena.ofConfined()) {
            SeqTable ta = new SeqTable(a, ar);
            SeqTable tb = new SeqTable(b, ar);
            MemorySegment size = ar.allocate(JAVA_LONG.byteSize());
            MemorySegment fp = ar.allocate(JAVA_LONG.byteSize() * minN);
            MemorySegment sp = ar.allocate(JAVA_LONG.byteSize() * minN);
            int status = (int)
                    SZ_SEQUENCE_INTERSECT.invokeExact(ta.sequence, tb.sequence, MemorySegment.NULL, seed, size, fp, sp);
            if (status != 0) throw new IllegalStateException("sz_sequence_intersect failed with status " + status);
            int count = (int) size.get(JAVA_LONG, 0);
            MemorySegment.copy(fp, JAVA_LONG, 0, firstPositions, 0, count);
            MemorySegment.copy(sp, JAVA_LONG, 0, secondPositions, 0, count);
            return count;
        } catch (Throwable e) {
            throw rethrow(e);
        }
    }

    public static int intersect(
            java.util.List<byte[]> a, java.util.List<byte[]> b, long[] firstPositions, long[] secondPositions) {
        return intersect(a, b, firstPositions, secondPositions, 0L);
    }

    // endregion

    // region Library Metadata

    /** Active SIMD backend(s), e.g. "serial,haswell,skylake,icelake". */
    public static String backend() {
        try {
            int caps = (int) SZ_CAPABILITIES.invokeExact();
            MemorySegment s = (MemorySegment) SZ_CAPABILITIES_TO_STRING.invokeExact(caps);
            return s.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static String version() {
        try {
            int major = (int) SZ_VERSION_MAJOR.invokeExact();
            int minor = (int) SZ_VERSION_MINOR.invokeExact();
            int patch = (int) SZ_VERSION_PATCH.invokeExact();
            return major + "." + minor + "." + patch;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException re) return re;
        if (t instanceof Error e) throw e;
        return new RuntimeException(t);
    }

    // endregion

    /** A 256-bit set of byte values (mirrors C sz_byteset_t). */
    public static final class Byteset {
        private final long[] w = new long[4];

        public Byteset add(byte value) {
            int c = value & 0xFF;
            w[c >> 6] |= 1L << (c & 63);
            return this;
        }

        public Byteset addAll(byte[] values) {
            for (byte v : values) add(v);
            return this;
        }

        public boolean contains(byte value) {
            int c = value & 0xFF;
            return (w[c >> 6] & (1L << (c & 63))) != 0;
        }

        MemorySegment toSegment(Arena a) {
            MemorySegment s = a.allocate(32);
            for (int i = 0; i < 4; i++) s.setAtIndex(JAVA_LONG, i, w[i]);
            return s;
        }

        public static Byteset of(byte[] values) {
            return new Byteset().addAll(values);
        }
    }

    /** Incremental (streaming) hash. Close to free the native state. */
    public static final class Hasher implements AutoCloseable {
        private final Arena arena = Arena.ofShared();
        private final MemorySegment state;

        public Hasher() {
            this(0L);
        }

        public Hasher(long seed) {
            // sz_hash_state_t is 216 bytes; over-allocate and 64-byte align for SIMD safety.
            state = arena.allocate(256, 64);
            try {
                SZ_HASH_STATE_INIT.invokeExact(state, seed);
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        public Hasher update(byte[] data) {
            try {
                SZ_HASH_STATE_UPDATE.invokeExact(state, MemorySegment.ofArray(data), (long) data.length);
                return this;
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        /** Finalize without mutating state (may be called repeatedly). */
        public long digest() {
            try {
                return (long) SZ_HASH_STATE_DIGEST.invokeExact(state);
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        @Override
        public void close() {
            arena.close();
        }
    }

    /** A reusable case-folding search needle. Close to free native memory. */
    public static final class UncasedNeedle implements AutoCloseable {
        private final Arena arena = Arena.ofShared();
        private final MemorySegment needle;
        private final long needleLen;
        private final MemorySegment meta; // zeroed; populated on use

        public UncasedNeedle(byte[] needle) {
            this.needleLen = needle.length;
            this.needle = arena.allocate(Math.max(needle.length, 1));
            MemorySegment.copy(needle, 0, this.needle, JAVA_BYTE, 0, needle.length);
            this.meta = arena.allocate(64);
        }

        /** First match of this needle in {@code haystack} (caseless), or offset -1. */
        public Match indexIn(byte[] haystack) {
            try (Arena a = Arena.ofConfined()) {
                MemorySegment h = a.allocate(Math.max(haystack.length, 1));
                MemorySegment.copy(haystack, 0, h, JAVA_BYTE, 0, haystack.length);
                MemorySegment ml = a.allocate(JAVA_LONG.byteSize());
                MemorySegment r = (MemorySegment)
                        SZ_UTF8_UNCASED_SEARCH.invokeExact(h, (long) haystack.length, needle, needleLen, meta, ml);
                long off = r.address() == 0 ? -1 : r.address() - h.address();
                return new Match(off, off < 0 ? 0 : ml.get(JAVA_LONG, 0));
            } catch (Throwable e) {
                throw rethrow(e);
            }
        }

        @Override
        public void close() {
            arena.close();
        }
    }

    /** Incremental SHA-256. Close to free the native state. */
    public static final class Sha256 implements AutoCloseable {
        private final Arena arena = Arena.ofShared();
        private final MemorySegment state;

        public Sha256() {
            state = arena.allocate(128, 64); // sz_sha256_state_t is 112 bytes
            try {
                SZ_SHA256_INIT.invokeExact(state);
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        public Sha256 update(byte[] data) {
            try {
                SZ_SHA256_UPDATE.invokeExact(state, MemorySegment.ofArray(data), (long) data.length);
                return this;
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        /** Writes the 32-byte digest into {@code destination} (must be ≥ 32 bytes; non-mutating). Allocation-free. */
        public void digest(byte[] destination) {
            if (destination.length < 32) throw new IllegalArgumentException("destination must hold at least 32 bytes");
            try {
                SZ_SHA256_DIGEST.invokeExact(state, MemorySegment.ofArray(destination));
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        /** Finalize into a fresh 32-byte digest (non-mutating; may be called repeatedly). */
        public byte[] digest() {
            byte[] out = new byte[32];
            digest(out);
            return out;
        }

        /** One-shot SHA-256. Equivalent to {@code MessageDigest.getInstance("SHA-256")}, SIMD-accelerated. */
        public static byte[] hashData(byte[] data) {
            try (Sha256 h = new Sha256()) {
                return h.update(data).digest();
            }
        }

        @Override
        public void close() {
            arena.close();
        }
    }

    /** Off-heap string table backing a sz_sequence_t, with FFM upcall callbacks */
    private static final class SeqTable {
        final MemorySegment data; // concatenated bytes (off-heap)
        final long[] starts; // per-item byte offset into data
        final long[] lengths; // per-item byte length
        final MemorySegment sequence; // the sz_sequence_t struct { handle, count, get_start, get_length }

        SeqTable(java.util.List<byte[]> items, Arena arena) {
            int n = items.size();
            long total = 0;
            for (byte[] it : items) total += it.length;
            data = arena.allocate(Math.max(total, 1));
            starts = new long[n];
            lengths = new long[n];
            long off = 0;
            for (int i = 0; i < n; i++) {
                byte[] it = items.get(i);
                starts[i] = off;
                lengths[i] = it.length;
                if (it.length > 0) MemorySegment.copy(it, 0, data, JAVA_BYTE, off, it.length);
                off += it.length;
            }
            MethodHandle startMH, lengthMH;
            try {
                startMH = MethodHandles.lookup()
                        .bind(
                                this,
                                "start",
                                MethodType.methodType(MemorySegment.class, MemorySegment.class, long.class));
                lengthMH = MethodHandles.lookup()
                        .bind(this, "length", MethodType.methodType(long.class, MemorySegment.class, long.class));
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
            MemorySegment startStub =
                    LINKER.upcallStub(startMH, FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG), arena);
            MemorySegment lengthStub =
                    LINKER.upcallStub(lengthMH, FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG), arena);
            sequence = arena.allocate(32);
            sequence.set(ADDRESS, 0, MemorySegment.NULL); // handle (unused; data is captured in the upcall)
            sequence.set(JAVA_LONG, 8, n); // count
            sequence.set(ADDRESS, 16, startStub); // get_start
            sequence.set(ADDRESS, 24, lengthStub); // get_length
        }

        // Invoked by the native sort/intersect. The handle arg is unused (data is captured here).
        MemorySegment start(MemorySegment handle, long idx) {
            return data.asSlice(starts[(int) idx], lengths[(int) idx]);
        }

        long length(MemorySegment handle, long idx) {
            return lengths[(int) idx];
        }
    }

    /** Native library extraction + lookup */
    private static final class NativeLoader {
        static SymbolLookup load() {
            String os = osName();
            String resource = "/native/" + os + "-" + archName() + "/" + libFileName(os);
            try (InputStream in = StringZilla.class.getResourceAsStream(resource)) {
                if (in == null) throw new UnsatisfiedLinkError("bundled native library not found: " + resource);
                Path tmp = Files.createTempFile("stringzilla", suffix(os));
                tmp.toFile().deleteOnExit();
                Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
                return SymbolLookup.libraryLookup(tmp, Arena.global());
            } catch (Exception e) {
                throw new UnsatisfiedLinkError("failed to load native StringZilla: " + e.getMessage());
            }
        }

        private static String osName() {
            String n = System.getProperty("os.name", "").toLowerCase();
            if (n.contains("win")) return "windows";
            if (n.contains("mac") || n.contains("darwin")) return "darwin";
            return "linux";
        }

        private static String archName() {
            String a = System.getProperty("os.arch", "").toLowerCase();
            if (a.equals("amd64") || a.equals("x86_64")) return "x86_64";
            if (a.equals("aarch64") || a.equals("arm64")) return "aarch64";
            return a;
        }

        private static String libFileName(String os) {
            return switch (os) {
                case "windows" -> "stringzilla.dll";
                case "darwin" -> "libstringzilla.dylib";
                default -> "libstringzilla.so";
            };
        }

        private static String suffix(String os) {
            return switch (os) {
                case "windows" -> ".dll";
                case "darwin" -> ".dylib";
                default -> ".so";
            };
        }
    }
}
