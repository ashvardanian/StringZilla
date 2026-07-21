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
 *  - Iteration: lazy `runes`, `words`/`graphemes`/`sentences`/`lineBreaks` over zero-copy views
 *  - Splitting: `split`/`rsplit`/`splitAny`, separator-token splits, `matches`, and `partition`
 *  - Collections: `argSort` (of a list or of one buffer's segments) and `intersect`
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
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

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
    private static final MethodHandle SZ_UTF8_WORDBREAKS = downSeg("sz_utf8_wordbreaks");
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
     *  {@link Character#codePointCount}, SIMD-accelerated over UTF-8 bytes. For example, "你好世界" counts
     *  4 and "Hello🌍" counts 6 — the astral emoji is a single scalar. */
    public static long countRunes(byte[] text) {
        return countRunes(MemorySegment.ofArray(text));
    }

    public static long countRunes(MemorySegment text) {
        try {
            return (long) SZ_UTF8_COUNT.invokeExact(text, text.byteSize());
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
        return decode(MemorySegment.ofArray(text), destination);
    }

    public static int decode(MemorySegment text, int[] destination) {
        long[] unpacked = new long[1];
        try {
            MemorySegment cursorIgnored = (MemorySegment) SZ_UTF8_DECODE.invokeExact(
                    text,
                    text.byteSize(),
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
        return decodeAll(MemorySegment.ofArray(text));
    }

    public static int[] decodeAll(MemorySegment text) {
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
            case WORDS -> SZ_UTF8_WORDBREAKS;
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
     *  over UTF-8 bytes (no locale, no transcode). Segmentation tiles the text — every byte lands in exactly
     *  one segment. By {@code GRAPHEMES}, the flag 🇺🇸 is one cluster spanning two codepoints; by
     *  {@code WORDS}, "Hello, 世界" splits the Latin run from the CJK run. */
    public static int segment(
            byte[] text,
            int textOffset,
            int textLength,
            SegmentKind kind,
            long[] starts,
            long[] lengths,
            long[] bytesConsumed) {
        return segment(MemorySegment.ofArray(text), textOffset, textLength, kind, starts, lengths, bytesConsumed);
    }

    public static int segment(
            MemorySegment text,
            int textOffset,
            int textLength,
            SegmentKind kind,
            long[] starts,
            long[] lengths,
            long[] bytesConsumed) {
        int cap = Math.min(starts.length, lengths.length);
        MethodHandle h = segHandle(kind);
        MemorySegment slice = text.asSlice(textOffset, textLength);
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
     *  over UTF-8 bytes with no UTF-16 transcode. For example, "e"+U+0301 composes into a precomposed "é"
     *  under {@code NFC} (and back under {@code NFD}); the ligature "ﬁ" expands into "fi" under {@code NFKC}. */
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

    /** Sorts the segments of a single buffer — given per-segment {@code starts} and {@code lengths} (as
     *  produced by the split iterators) — writing their sorting permutation into {@code order} (must hold
     *  ≥ starts.length entries) and returning the count. Sort the lines of a file without building a
     *  {@code byte[][]}: collect ranges from a {@link Splitter}, then sort them here. */
    public static int argSort(
            byte[] text, long[] starts, long[] lengths, long[] order, boolean reverse, long top, boolean uncased) {
        int n = starts.length;
        if (n == 0) return 0;
        if (lengths.length < n) throw new IllegalArgumentException("lengths must hold at least " + n + " entries");
        if (order.length < n) throw new IllegalArgumentException("order must hold at least " + n + " entries");
        try (Arena a = Arena.ofConfined()) {
            SeqTable t = new SeqTable(text, starts, lengths, n, a);
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

    // region Iteration

    /** Decodes the Unicode scalar values (codepoints) as an {@code IntStream}. Like {@code String.codePoints()}.
     *  Eager (decodes the whole input up front); for incremental decoding use {@link #decode(byte[], int[])}. */
    public static IntStream runes(byte[] text) {
        return IntStream.of(decodeAll(text));
    }

    public static IntStream runes(MemorySegment text) {
        return IntStream.of(decodeAll(text));
    }

    /** Lazily yields the word segments (UAX #29) as zero-copy {@link MemorySegment} views, or a
     *  zero-allocation {@link Segmenter} cursor via {@link Segments#cursor()}. Accepts an on-heap
     *  {@code byte[]} or any {@link MemorySegment} (e.g. a Spark {@code UTF8String} or an mmap'd file). */
    public static Segments words(byte[] text) {
        return new Segments(MemorySegment.ofArray(text), SegmentKind.WORDS);
    }

    public static Segments words(MemorySegment text) {
        return new Segments(text, SegmentKind.WORDS);
    }

    public static Segments graphemes(byte[] text) {
        return new Segments(MemorySegment.ofArray(text), SegmentKind.GRAPHEMES);
    }

    public static Segments graphemes(MemorySegment text) {
        return new Segments(text, SegmentKind.GRAPHEMES);
    }

    public static Segments sentences(byte[] text) {
        return new Segments(MemorySegment.ofArray(text), SegmentKind.SENTENCES);
    }

    public static Segments sentences(MemorySegment text) {
        return new Segments(text, SegmentKind.SENTENCES);
    }

    public static Segments lineBreaks(byte[] text) {
        return new Segments(MemorySegment.ofArray(text), SegmentKind.LINE_BREAKS);
    }

    public static Segments lineBreaks(MemorySegment text) {
        return new Segments(text, SegmentKind.LINE_BREAKS);
    }

    // endregion

    // region Splitting

    /** Lazily splits {@code text} on each occurrence of {@code separator}, yielding the segments between
     *  separators as zero-copy views. Empty segments are kept by default (like {@code String.split} with a
     *  negative limit); chain {@code .skipEmpty()}, {@code .keepSeparator()}, or {@code .withMaxSplit(n)}.
     *  Off-heap {@link MemorySegment} inputs are scanned in place; on-heap inputs are copied off-heap once
     *  per {@code cursor()} (pointer-returning finds need a stable address). */
    public static Splits split(byte[] text, byte[] separator) {
        return Splits.substring(MemorySegment.ofArray(text), MemorySegment.ofArray(separator), false);
    }

    public static Splits split(MemorySegment text, MemorySegment separator) {
        return Splits.substring(text, separator, false);
    }

    /** As {@link #split} but scanning from the end; yields the last segment first. */
    public static Splits rsplit(byte[] text, byte[] separator) {
        return Splits.substring(MemorySegment.ofArray(text), MemorySegment.ofArray(separator), true);
    }

    public static Splits rsplit(MemorySegment text, MemorySegment separator) {
        return Splits.substring(text, separator, true);
    }

    /** Lazily splits on any byte in {@code separators} (like splitting on a character set). */
    public static Splits splitAny(byte[] text, Byteset separators) {
        return Splits.byteset(MemorySegment.ofArray(text), separators, false);
    }

    public static Splits splitAny(MemorySegment text, Byteset separators) {
        return Splits.byteset(text, separators, false);
    }

    /** As {@link #splitAny} but scanning from the end. */
    public static Splits rsplitAny(byte[] text, Byteset separators) {
        return Splits.byteset(MemorySegment.ofArray(text), separators, true);
    }

    public static Splits rsplitAny(MemorySegment text, Byteset separators) {
        return Splits.byteset(text, separators, true);
    }

    /** Lazily splits on Unicode newline runs, yielding the lines between them. Chain
     *  {@code .withSeparators()} for a lossless interleave or {@code .skipEmpty()} to drop blanks. */
    public static TokenSplits splitNewlines(byte[] text) {
        return new TokenSplits(MemorySegment.ofArray(text), SegmentKind.NEWLINES, SplitParts.BETWEEN, false);
    }

    public static TokenSplits splitNewlines(MemorySegment text) {
        return new TokenSplits(text, SegmentKind.NEWLINES, SplitParts.BETWEEN, false);
    }

    public static TokenSplits splitWhitespaces(byte[] text) {
        return new TokenSplits(MemorySegment.ofArray(text), SegmentKind.WHITESPACES, SplitParts.BETWEEN, false);
    }

    public static TokenSplits splitWhitespaces(MemorySegment text) {
        return new TokenSplits(text, SegmentKind.WHITESPACES, SplitParts.BETWEEN, false);
    }

    public static TokenSplits splitDelimiters(byte[] text) {
        return new TokenSplits(MemorySegment.ofArray(text), SegmentKind.DELIMITERS, SplitParts.BETWEEN, false);
    }

    public static TokenSplits splitDelimiters(MemorySegment text) {
        return new TokenSplits(text, SegmentKind.DELIMITERS, SplitParts.BETWEEN, false);
    }

    /** Lazily enumerates the byte offsets of every match of {@code needle} in {@code haystack}; chain
     *  {@code .overlapping()} for overlapping matches. Counting the result is the occurrence count. */
    public static Matches matches(byte[] haystack, byte[] needle) {
        return new Matches(MemorySegment.ofArray(haystack), MemorySegment.ofArray(needle), false);
    }

    public static Matches matches(MemorySegment haystack, MemorySegment needle) {
        return new Matches(haystack, needle, false);
    }

    /** Lazily enumerates case-insensitive (full-fold) matches; chain {@code .overlapping()}. Each
     *  {@link Match} carries the offset and matched length (folding may make the latter differ). */
    public static UncasedMatches uncasedMatches(byte[] haystack, byte[] needle) {
        return new UncasedMatches(MemorySegment.ofArray(haystack), MemorySegment.ofArray(needle), false);
    }

    public static UncasedMatches uncasedMatches(MemorySegment haystack, MemorySegment needle) {
        return new UncasedMatches(haystack, needle, false);
    }

    /** Splits at the first occurrence of {@code separator} into (before, separator, after) views; if
     *  absent, returns ({@code text}, empty, empty). Mirrors Python's {@code str.partition}. */
    public static Partition partition(byte[] text, byte[] separator) {
        return partition(MemorySegment.ofArray(text), MemorySegment.ofArray(separator));
    }

    public static Partition partition(MemorySegment text, MemorySegment separator) {
        long at;
        try (Arena arena = Arena.ofConfined()) {
            at = indexOf(nativeView(text, arena), nativeView(separator, arena));
        }
        long separatorLength = separator.byteSize();
        if (at < 0) return new Partition(text, text.asSlice(text.byteSize(), 0), text.asSlice(text.byteSize(), 0));
        return new Partition(
                text.asSlice(0, at), text.asSlice(at, separatorLength), text.asSlice(at + separatorLength));
    }

    /** As {@link #partition} but at the last occurrence; if absent, returns (empty, empty, {@code text}).
     *  Mirrors Python's {@code str.rpartition}. */
    public static Partition rpartition(byte[] text, byte[] separator) {
        return rpartition(MemorySegment.ofArray(text), MemorySegment.ofArray(separator));
    }

    public static Partition rpartition(MemorySegment text, MemorySegment separator) {
        long at;
        try (Arena arena = Arena.ofConfined()) {
            at = lastIndexOf(nativeView(text, arena), nativeView(separator, arena));
        }
        long separatorLength = separator.byteSize();
        if (at < 0) return new Partition(text.asSlice(0, 0), text.asSlice(0, 0), text);
        return new Partition(
                text.asSlice(0, at), text.asSlice(at, separatorLength), text.asSlice(at + separatorLength));
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

    private static int utf8LeadWidth(byte lead) {
        int b = lead & 0xFF;
        if ((b & 0x80) == 0) return 1;
        if ((b & 0xE0) == 0xC0) return 2;
        if ((b & 0xF0) == 0xE0) return 3;
        if ((b & 0xF8) == 0xF0) return 4;
        return 1;
    }

    /** Returns a native-addressable view of {@code segment}: the segment itself when it is already native
     *  (off-heap — e.g. a Spark Tungsten row or an mmap'd file), or a one-time {@code arena} copy when it is
     *  heap-backed (e.g. a {@code byte[]} or a Lucene {@code BytesRef}). Pointer-returning finds (unlike the
     *  offset-writing segmenters) need a stable address, which heap segments do not expose. */
    private static MemorySegment nativeView(MemorySegment segment, Arena arena) {
        if (segment.isNative()) return segment;
        long length = segment.byteSize();
        MemorySegment copy = arena.allocate(Math.max(length, 1));
        if (length > 0) MemorySegment.copy(segment, 0, copy, 0, length);
        return length == copy.byteSize() ? copy : copy.asSlice(0, length); // keep byteSize() the logical length
    }

    private static long findByteset(MemorySegment haystack, MemorySegment bytesetSeg) {
        if (haystack.byteSize() == 0) return -1;
        try {
            MemorySegment r = (MemorySegment) SZ_FIND_BYTESET.invokeExact(haystack, haystack.byteSize(), bytesetSeg);
            return r.address() == 0 ? -1 : r.address() - haystack.address();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static long rfindByteset(MemorySegment haystack, MemorySegment bytesetSeg) {
        if (haystack.byteSize() == 0) return -1;
        try {
            MemorySegment r = (MemorySegment) SZ_RFIND_BYTESET.invokeExact(haystack, haystack.byteSize(), bytesetSeg);
            return r.address() == 0 ? -1 : r.address() - haystack.address();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Which spans a separator-token split yields: the gaps, the separator runs, or both (lossless). */
    public enum SplitParts {
        BETWEEN,
        SEPARATORS,
        BOTH
    }

    /** A single split point: the spans before, of, and after a separator. See {@link #partition}. */
    public record Partition(MemorySegment before, MemorySegment separator, MemorySegment after) {}

    /** Zero-allocation forward cursor over tiling UTF-8 segments (words, graphemes, …). Reuses one 64-entry
     *  batch buffer; call {@link #next()} then read {@link #startOffset()}/{@link #length()}. */
    public static final class Segmenter {
        private final MemorySegment text;
        private final SegmentKind kind;
        private final long[] starts = new long[64];
        private final long[] lengths = new long[64];
        private final long[] consumed = new long[1];
        private int count, index, scanCursor;
        private long chunkBase, curStart, curLength;

        Segmenter(MemorySegment text, SegmentKind kind) {
            this.text = text;
            this.kind = kind;
        }

        public boolean next() {
            while (true) {
                if (index < count) {
                    curStart = chunkBase + starts[index];
                    curLength = lengths[index];
                    index++;
                    return true;
                }
                long remaining = text.byteSize() - scanCursor;
                if (remaining <= 0) return false;
                chunkBase = scanCursor;
                consumed[0] = 0;
                count = segment(text, scanCursor, (int) remaining, kind, starts, lengths, consumed);
                if (count == 0 || consumed[0] == 0) return false;
                index = 0;
                scanCursor += (int) consumed[0];
            }
        }

        public long startOffset() {
            return curStart;
        }

        public long length() {
            return curLength;
        }

        public MemorySegment current() {
            return text.asSlice(curStart, curLength);
        }
    }

    /** Iterable / streamable view over tiling segments, yielding zero-copy {@link MemorySegment} slices.
     *  Use {@link #cursor()} for the zero-allocation path. */
    public static final class Segments implements Iterable<MemorySegment> {
        private final MemorySegment text;
        private final SegmentKind kind;

        Segments(MemorySegment text, SegmentKind kind) {
            this.text = text;
            this.kind = kind;
        }

        public Segmenter cursor() {
            return new Segmenter(text, kind);
        }

        @Override
        public Iterator<MemorySegment> iterator() {
            Segmenter cursor = new Segmenter(text, kind);
            return new Iterator<>() {
                private boolean has, pulled;

                @Override
                public boolean hasNext() {
                    if (!pulled) {
                        has = cursor.next();
                        pulled = true;
                    }
                    return has;
                }

                @Override
                public MemorySegment next() {
                    if (!hasNext()) throw new NoSuchElementException();
                    pulled = false;
                    return cursor.current();
                }
            };
        }

        public Stream<MemorySegment> stream() {
            return StreamSupport.stream(spliterator(), false);
        }
    }

    /** Zero-allocation cursor over substring / byte-set splits, with forward and reverse scanning and the
     *  {@code keepSeparator} / {@code skipEmpty} / {@code maxSplit} policies. */
    public static final class Splitter {
        private final MemorySegment data; // native-addressable view; finds run on it and yielded views slice it
        private final MemorySegment needle; // null for a byte-set split
        private final MemorySegment bytesetSeg; // null for a substring split
        private final boolean reverse, keepSeparator, skipEmpty;
        private final long maxSplit; // < 0 == unlimited
        private long forwardCursor, reverseEnd, splitsDone;
        private boolean finished;
        private long curStart, curLength;

        Splitter(
                MemorySegment data,
                MemorySegment needle,
                MemorySegment bytesetSeg,
                boolean reverse,
                boolean keepSeparator,
                boolean skipEmpty,
                long maxSplit) {
            this.data = data;
            this.needle = needle;
            this.bytesetSeg = bytesetSeg;
            this.reverse = reverse;
            this.keepSeparator = keepSeparator;
            this.skipEmpty = skipEmpty;
            this.maxSplit = maxSplit;
            this.forwardCursor = 0;
            this.reverseEnd = data.byteSize();
        }

        public boolean next() {
            while (true) {
                if (finished) return false;
                if (!(reverse ? stepReverse() : stepForward())) {
                    finished = true;
                    return false;
                }
                if (skipEmpty && curLength == 0) continue;
                return true;
            }
        }

        private long separatorLength() {
            return needle != null ? needle.byteSize() : 1;
        }

        private long findForward(long from) {
            MemorySegment rest = data.asSlice(from, data.byteSize() - from);
            long rel = needle != null
                    ? (needle.byteSize() == 0 ? -1 : indexOf(rest, needle))
                    : findByteset(rest, bytesetSeg);
            return rel < 0 ? -1 : from + rel;
        }

        private long findReverse(long end) {
            MemorySegment prefix = data.asSlice(0, end);
            if (needle != null) return needle.byteSize() == 0 ? -1 : lastIndexOf(prefix, needle);
            return rfindByteset(prefix, bytesetSeg);
        }

        private boolean stepForward() {
            long textLength = data.byteSize();
            if (forwardCursor > textLength) return false;
            long sepAt = (maxSplit >= 0 && splitsDone >= maxSplit) ? -1 : findForward(forwardCursor);
            if (sepAt < 0) {
                curStart = forwardCursor;
                curLength = textLength - forwardCursor;
                forwardCursor = textLength + 1;
                return true;
            }
            long segmentEnd = keepSeparator ? sepAt + separatorLength() : sepAt;
            curStart = forwardCursor;
            curLength = segmentEnd - forwardCursor;
            forwardCursor = sepAt + separatorLength();
            splitsDone++;
            return true;
        }

        private boolean stepReverse() {
            if (reverseEnd < 0) return false;
            long sepAt = (maxSplit >= 0 && splitsDone >= maxSplit) ? -1 : findReverse(reverseEnd);
            if (sepAt < 0) {
                curStart = 0;
                curLength = reverseEnd;
                reverseEnd = -1;
                return true;
            }
            long segmentStart = keepSeparator ? sepAt : sepAt + separatorLength();
            curStart = segmentStart;
            curLength = reverseEnd - segmentStart;
            reverseEnd = sepAt;
            splitsDone++;
            return true;
        }

        public long startOffset() {
            return curStart;
        }

        public long length() {
            return curLength;
        }

        public MemorySegment current() {
            return data.asSlice(curStart, curLength);
        }
    }

    /** Iterable / streamable view over substring or byte-set splits. Fluent policies return a new view;
     *  {@link #cursor()} is the zero-allocation path. */
    public static final class Splits implements Iterable<MemorySegment> {
        private final MemorySegment text;
        private final MemorySegment separator; // null for a byte-set split
        private final Byteset byteset; // null for a substring split
        private final boolean reverse, keepSeparator, skipEmpty;
        private final long maxSplit;

        private Splits(
                MemorySegment text,
                MemorySegment separator,
                Byteset byteset,
                boolean reverse,
                boolean keepSeparator,
                boolean skipEmpty,
                long maxSplit) {
            this.text = text;
            this.separator = separator;
            this.byteset = byteset;
            this.reverse = reverse;
            this.keepSeparator = keepSeparator;
            this.skipEmpty = skipEmpty;
            this.maxSplit = maxSplit;
        }

        static Splits substring(MemorySegment text, MemorySegment separator, boolean reverse) {
            return new Splits(text, separator, null, reverse, false, false, -1);
        }

        static Splits byteset(MemorySegment text, Byteset byteset, boolean reverse) {
            return new Splits(text, null, byteset, reverse, false, false, -1);
        }

        public Splits skipEmpty() {
            return new Splits(text, separator, byteset, reverse, keepSeparator, true, maxSplit);
        }

        /** Keep the separator attached to each segment: trailing when splitting forward, leading in reverse. */
        public Splits keepSeparator() {
            return new Splits(text, separator, byteset, reverse, true, skipEmpty, maxSplit);
        }

        public Splits withMaxSplit(long limit) {
            return new Splits(text, separator, byteset, reverse, keepSeparator, skipEmpty, limit);
        }

        public Splitter cursor() {
            Arena arena = Arena.ofAuto();
            MemorySegment data = nativeView(text, arena);
            MemorySegment needleSeg = separator != null ? nativeView(separator, arena) : null;
            MemorySegment bytesetSeg = byteset != null ? byteset.toSegment(arena) : null;
            return new Splitter(data, needleSeg, bytesetSeg, reverse, keepSeparator, skipEmpty, maxSplit);
        }

        @Override
        public Iterator<MemorySegment> iterator() {
            Splitter cursor = cursor();
            return new Iterator<>() {
                private boolean has, pulled;

                @Override
                public boolean hasNext() {
                    if (!pulled) {
                        has = cursor.next();
                        pulled = true;
                    }
                    return has;
                }

                @Override
                public MemorySegment next() {
                    if (!hasNext()) throw new NoSuchElementException();
                    pulled = false;
                    return cursor.current();
                }
            };
        }

        public Stream<MemorySegment> stream() {
            return StreamSupport.stream(spliterator(), false);
        }
    }

    /** Zero-allocation cursor over separator-token splits (newlines, whitespace, delimiters), batched
     *  64 separators per native call, applying the {@link SplitParts} policy. */
    public static final class TokenSplitter {
        private final MemorySegment text;
        private final SegmentKind kind;
        private final SplitParts parts;
        private final boolean skipEmpty;
        private final long[] starts = new long[64];
        private final long[] lengths = new long[64];
        private final long[] consumed = new long[1];
        private int count, index, scanCursor;
        private long chunkBase, prevEnd, curStart, curLength;
        private boolean atSeparator, exhausted, emittedTail;

        TokenSplitter(MemorySegment text, SegmentKind kind, SplitParts parts, boolean skipEmpty) {
            this.text = text;
            this.kind = kind;
            this.parts = parts;
            this.skipEmpty = skipEmpty;
        }

        public boolean next() {
            while (true) {
                if (!exhausted && index >= count) {
                    long remaining = text.byteSize() - scanCursor;
                    if (remaining <= 0) {
                        exhausted = true;
                    } else {
                        chunkBase = scanCursor;
                        consumed[0] = 0;
                        count = segment(text, scanCursor, (int) remaining, kind, starts, lengths, consumed);
                        if (count == 0 || consumed[0] == 0) {
                            exhausted = true;
                        } else {
                            index = 0;
                            scanCursor += (int) consumed[0];
                        }
                    }
                }
                if (exhausted) {
                    if (emittedTail) return false;
                    emittedTail = true;
                    if (parts != SplitParts.SEPARATORS) {
                        long length = text.byteSize() - prevEnd;
                        if (!(skipEmpty && length == 0)) {
                            curStart = prevEnd;
                            curLength = length;
                            return true;
                        }
                    }
                    return false;
                }
                long runStart = chunkBase + starts[index];
                long runEnd = runStart + lengths[index];
                if (!atSeparator) {
                    atSeparator = true;
                    if (parts != SplitParts.SEPARATORS) {
                        long length = runStart - prevEnd;
                        if (!(skipEmpty && length == 0)) {
                            curStart = prevEnd;
                            curLength = length;
                            return true;
                        }
                    }
                }
                atSeparator = false;
                prevEnd = runEnd;
                index++;
                if (parts != SplitParts.BETWEEN) {
                    long length = runEnd - runStart;
                    if (!(skipEmpty && length == 0)) {
                        curStart = runStart;
                        curLength = length;
                        return true;
                    }
                }
            }
        }

        public long startOffset() {
            return curStart;
        }

        public long length() {
            return curLength;
        }

        public MemorySegment current() {
            return text.asSlice(curStart, curLength);
        }
    }

    /** Iterable / streamable view over separator-token splits, yielding zero-copy slices. */
    public static final class TokenSplits implements Iterable<MemorySegment> {
        private final MemorySegment text;
        private final SegmentKind kind;
        private final SplitParts parts;
        private final boolean skipEmpty;

        TokenSplits(MemorySegment text, SegmentKind kind, SplitParts parts, boolean skipEmpty) {
            this.text = text;
            this.kind = kind;
            this.parts = parts;
            this.skipEmpty = skipEmpty;
        }

        /** Also yield the separator runs, interleaved with the gaps (lossless reconstruction). */
        public TokenSplits withSeparators() {
            return new TokenSplits(text, kind, SplitParts.BOTH, skipEmpty);
        }

        /** Yield only the separator runs. */
        public TokenSplits onlySeparators() {
            return new TokenSplits(text, kind, SplitParts.SEPARATORS, skipEmpty);
        }

        public TokenSplits skipEmpty() {
            return new TokenSplits(text, kind, parts, true);
        }

        public TokenSplitter cursor() {
            return new TokenSplitter(text, kind, parts, skipEmpty);
        }

        @Override
        public Iterator<MemorySegment> iterator() {
            TokenSplitter cursor = new TokenSplitter(text, kind, parts, skipEmpty);
            return new Iterator<>() {
                private boolean has, pulled;

                @Override
                public boolean hasNext() {
                    if (!pulled) {
                        has = cursor.next();
                        pulled = true;
                    }
                    return has;
                }

                @Override
                public MemorySegment next() {
                    if (!hasNext()) throw new NoSuchElementException();
                    pulled = false;
                    return cursor.current();
                }
            };
        }

        public Stream<MemorySegment> stream() {
            return StreamSupport.stream(spliterator(), false);
        }
    }

    /** Zero-allocation cursor over substring matches, yielding each match's byte offset. */
    public static final class Matcher {
        private final MemorySegment data;
        private final MemorySegment needle;
        private final boolean overlapping;
        private long cursor, curOffset;

        Matcher(MemorySegment data, MemorySegment needle, boolean overlapping) {
            this.data = data;
            this.needle = needle;
            this.overlapping = overlapping;
        }

        public boolean next() {
            long textLength = data.byteSize();
            if (needle.byteSize() == 0 || cursor > textLength) return false;
            long rel = indexOf(data.asSlice(cursor, textLength - cursor), needle);
            if (rel < 0) {
                cursor = textLength + 1;
                return false;
            }
            curOffset = cursor + rel;
            cursor = curOffset + (overlapping ? 1 : needle.byteSize());
            return true;
        }

        public long offset() {
            return curOffset;
        }
    }

    /** Streamable view over substring matches; {@link #overlapping()} reports overlapping matches and
     *  {@link #cursor()} is the zero-allocation path. */
    public static final class Matches {
        private final MemorySegment haystack;
        private final MemorySegment needle;
        private final boolean overlapping;

        Matches(MemorySegment haystack, MemorySegment needle, boolean overlapping) {
            this.haystack = haystack;
            this.needle = needle;
            this.overlapping = overlapping;
        }

        public Matches overlapping() {
            return new Matches(haystack, needle, true);
        }

        public Matcher cursor() {
            Arena arena = Arena.ofAuto();
            return new Matcher(nativeView(haystack, arena), nativeView(needle, arena), overlapping);
        }

        public long[] toArray() {
            Matcher cursor = cursor();
            long[] buffer = new long[16];
            int count = 0;
            while (cursor.next()) {
                if (count == buffer.length) buffer = java.util.Arrays.copyOf(buffer, count * 2);
                buffer[count++] = cursor.offset();
            }
            return java.util.Arrays.copyOf(buffer, count);
        }

        public LongStream stream() {
            return LongStream.of(toArray());
        }
    }

    /** Zero-allocation cursor over case-insensitive matches, reusing one cached needle-metadata buffer. */
    public static final class UncasedMatcher {
        private final MemorySegment data;
        private final MemorySegment needle;
        private final MemorySegment meta; // populated on first search, reused thereafter
        private final MemorySegment matchedLengthOut;
        private final boolean overlapping;
        private long cursor;
        private Match current;

        UncasedMatcher(
                MemorySegment data,
                MemorySegment needle,
                MemorySegment meta,
                MemorySegment matchedLengthOut,
                boolean overlapping) {
            this.data = data;
            this.needle = needle;
            this.meta = meta;
            this.matchedLengthOut = matchedLengthOut;
            this.overlapping = overlapping;
        }

        public boolean next() {
            long textLength = data.byteSize();
            if (cursor > textLength) return false;
            try {
                MemorySegment found = (MemorySegment) SZ_UTF8_UNCASED_SEARCH.invokeExact(
                        data.asSlice(cursor, textLength - cursor),
                        textLength - cursor,
                        needle,
                        needle.byteSize(),
                        meta,
                        matchedLengthOut);
                if (found.address() == 0) {
                    cursor = textLength + 1;
                    return false;
                }
                long offset = found.address() - data.address();
                long matchedLength = matchedLengthOut.get(JAVA_LONG, 0);
                current = new Match(offset, matchedLength);
                // Overlapping advances one codepoint (not one byte): caseless folding is Unicode-aware, so
                // the next search must start on a UTF-8 boundary rather than mid-codepoint.
                cursor = offset
                        + (overlapping ? utf8LeadWidth(data.get(JAVA_BYTE, offset)) : Math.max(matchedLength, 1));
                return true;
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        public Match current() {
            return current;
        }
    }

    /** Streamable view over case-insensitive matches; {@link #overlapping()} and {@link #cursor()} mirror
     *  {@link Matches}. */
    public static final class UncasedMatches {
        private final MemorySegment haystack;
        private final MemorySegment needle;
        private final boolean overlapping;

        UncasedMatches(MemorySegment haystack, MemorySegment needle, boolean overlapping) {
            this.haystack = haystack;
            this.needle = needle;
            this.overlapping = overlapping;
        }

        public UncasedMatches overlapping() {
            return new UncasedMatches(haystack, needle, true);
        }

        public UncasedMatcher cursor() {
            Arena arena = Arena.ofAuto();
            MemorySegment data = nativeView(haystack, arena);
            MemorySegment needleSeg = nativeView(needle, arena);
            MemorySegment meta = arena.allocate(64); // zeroed; the native dereferences it (NULL is unsafe)
            MemorySegment matchedLengthOut = arena.allocate(JAVA_LONG.byteSize());
            return new UncasedMatcher(data, needleSeg, meta, matchedLengthOut, overlapping);
        }

        public java.util.List<Match> toList() {
            UncasedMatcher cursor = cursor();
            java.util.List<Match> out = new java.util.ArrayList<>();
            while (cursor.next()) out.add(cursor.current());
            return out;
        }

        public Stream<Match> stream() {
            return toList().stream();
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
            sequence = buildSequence(n, arena);
        }

        // Sorts the (start, length) ranges of a single buffer; reuses the caller's offset arrays.
        SeqTable(byte[] text, long[] segmentStarts, long[] segmentLengths, int n, Arena arena) {
            data = arena.allocate(Math.max(text.length, 1));
            if (text.length > 0) MemorySegment.copy(text, 0, data, JAVA_BYTE, 0, text.length);
            starts = segmentStarts;
            lengths = segmentLengths;
            sequence = buildSequence(n, arena);
        }

        private MemorySegment buildSequence(int n, Arena arena) {
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
            MemorySegment seq = arena.allocate(32);
            seq.set(ADDRESS, 0, MemorySegment.NULL); // handle (unused; data is captured in the upcall)
            seq.set(JAVA_LONG, 8, n); // count
            seq.set(ADDRESS, 16, startStub); // get_start
            seq.set(ADDRESS, 24, lengthStub); // get_length
            return seq;
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
