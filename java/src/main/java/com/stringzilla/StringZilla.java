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
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/** Core StringZilla operations: search, compare, hash. Pure FFM, JDK 22+. */
public final class StringZilla {

    private StringZilla() {}

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP = NativeLoader.load();

    // `critical(true)` reads on-heap arrays and fills one-element out-arrays in place; upcalling handles can't use it.
    private static final MethodHandle STRINGZILLA_FIND = downCritical(
            "sz_find_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_RFIND = downCritical(
            "sz_rfind_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_FIND_BYTESET = downCritical(
            "sz_find_byteset_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_RFIND_BYTESET = downCritical(
            "sz_rfind_byteset_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));

    private static final MethodHandle STRINGZILLA_HASH = downCritical(
            "sz_hash_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_BYTESUM = downCritical(
            "sz_bytesum_best", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_EQUAL = downCritical(
            "sz_equal_best", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_ORDER = downCritical(
            "sz_order_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));

    private static final MethodHandle STRINGZILLA_HASH_STATE_INIT =
            down("sz_hash_state_init_best", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_HASH_STATE_UPDATE = downCritical(
            "sz_hash_state_update_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_HASH_STATE_DIGEST = downCritical(
            "sz_hash_state_digest_best", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));

    // Device queries: a status and one out-word, some after a device ordinal. GPU runtimes can take long to
    // initialize, so none of these is `critical(true)`.
    private static final FunctionDescriptor QUERY = FunctionDescriptor.of(JAVA_INT, ADDRESS);
    private static final FunctionDescriptor DEVICE_QUERY = FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS);
    private static final MethodHandle STRINGZILLA_CPU_CAPABILITIES_DETECTED =
            down("sz_cpu_capabilities_detected", QUERY);
    private static final MethodHandle STRINGZILLA_CPU_CAPABILITIES_COMPILED =
            down("sz_cpu_capabilities_compiled", QUERY);
    private static final MethodHandle STRINGZILLA_CPU_CAPABILITIES_ENABLED = down("sz_cpu_capabilities_enabled", QUERY);
    private static final MethodHandle STRINGZILLA_CPU_CONFIGURE_THREAD =
            down("sz_cpu_configure_thread", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
    private static final MethodHandle STRINGZILLA_CUDA_COUNT_DEVICES = down("sz_cuda_count_devices", QUERY);
    private static final MethodHandle STRINGZILLA_CUDA_CAPABILITIES_DETECTED =
            down("sz_cuda_capabilities_detected", DEVICE_QUERY);
    private static final MethodHandle STRINGZILLA_CUDA_CAPABILITIES_COMPILED =
            down("sz_cuda_capabilities_compiled", QUERY);
    private static final MethodHandle STRINGZILLA_CUDA_CAPABILITIES_ENABLED =
            down("sz_cuda_capabilities_enabled", DEVICE_QUERY);
    private static final MethodHandle STRINGZILLA_ROCM_COUNT_DEVICES = down("sz_rocm_count_devices", QUERY);
    private static final MethodHandle STRINGZILLA_ROCM_CAPABILITIES_DETECTED =
            down("sz_rocm_capabilities_detected", DEVICE_QUERY);
    private static final MethodHandle STRINGZILLA_ROCM_CAPABILITIES_COMPILED =
            down("sz_rocm_capabilities_compiled", QUERY);
    private static final MethodHandle STRINGZILLA_ROCM_CAPABILITIES_ENABLED =
            down("sz_rocm_capabilities_enabled", DEVICE_QUERY);
    private static final MethodHandle STRINGZILLA_METAL_COUNT_DEVICES = down("sz_metal_count_devices", QUERY);
    private static final MethodHandle STRINGZILLA_METAL_CAPABILITIES_DETECTED =
            down("sz_metal_capabilities_detected", DEVICE_QUERY);
    private static final MethodHandle STRINGZILLA_METAL_CAPABILITIES_COMPILED =
            down("sz_metal_capabilities_compiled", QUERY);
    private static final MethodHandle STRINGZILLA_METAL_CAPABILITIES_ENABLED =
            down("sz_metal_capabilities_enabled", DEVICE_QUERY);
    private static final MethodHandle STRINGZILLA_NAME_CAPABILITIES =
            downCritical("sz_capabilities_name", FunctionDescriptor.of(JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG));

    /** {@code STRINGZILLA_CAPABILITIES_NAME_CAPACITY}, which fits every capability name list. */
    private static final int CAPABILITIES_NAME_CAPACITY = 256;

    /** {@code sz_missing_gpu_k}, past the last device of a kind. */
    private static final int STATUS_MISSING_GPU = -16;

    /** {@code sz_missing_kernel_k}, for a CPU-only call on a GPU. */
    private static final int STATUS_MISSING_KERNEL = -20;

    private static final MethodHandle STRINGZILLA_VERSION_MAJOR =
            down("sz_version_major", FunctionDescriptor.of(JAVA_INT));
    private static final MethodHandle STRINGZILLA_VERSION_MINOR =
            down("sz_version_minor", FunctionDescriptor.of(JAVA_INT));
    private static final MethodHandle STRINGZILLA_VERSION_PATCH =
            down("sz_version_patch", FunctionDescriptor.of(JAVA_INT));

    // region UTF-8 Codepoints
    private static final MethodHandle STRINGZILLA_UTF8_COUNT = downCritical(
            "sz_utf8_count_best", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_UTF8_SEEK = down(
            "sz_utf8_seek_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_UTF8_DECODE = downCritical(
            "sz_utf8_decode_best",
            FunctionDescriptor.of(
                    JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));

    // endregion

    // region UTF-8 Case Folding and Uncased
    private static final MethodHandle STRINGZILLA_UTF8_UNCASED_FOLD = downCritical(
            "sz_utf8_uncased_fold_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_UTF8_UNCASED_NEEDLE_INIT = downCritical(
            "sz_utf8_uncased_needle_init_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_UTF8_UNCASED_SEARCH = downCritical(
            "sz_utf8_uncased_search_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_UTF8_UNCASED_ORDER = downCritical(
            "sz_utf8_uncased_order_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));

    /** {@code sz_utf8_uncased_needle_t}, which the binding allocates for the C side to prepare. */
    private static final MemoryLayout UNCASED_NEEDLE = MemoryLayout.structLayout(
            ADDRESS.withName("start"),
            JAVA_LONG.withName("length"),
            JAVA_LONG.withName("offset_in_unfolded"),
            JAVA_LONG.withName("length_in_unfolded"),
            MemoryLayout.sequenceLayout(16, JAVA_BYTE).withName("folded_slice"),
            JAVA_BYTE.withName("folded_slice_length"),
            JAVA_BYTE.withName("probe_second"),
            JAVA_BYTE.withName("probe_third"),
            JAVA_BYTE.withName("script"),
            MemoryLayout.paddingLayout(4));

    // endregion

    // region Hash Extras, SHA-256, Random and Lookup
    private static final MethodHandle STRINGZILLA_HASH_MULTISEED = downCritical(
            "sz_hash_multiseed_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_SHA256_INIT =
            down("sz_sha256_state_init_best", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_SHA256_UPDATE = downCritical(
            "sz_sha256_state_update_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_SHA256_DIGEST = downCritical(
            "sz_sha256_state_digest_best", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_FILL_RANDOM = downCritical(
            "sz_fill_random_best", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_LOOKUP = downCritical(
            "sz_lookup_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));

    // endregion

    // region Normalization and Collections
    private static final MethodHandle STRINGZILLA_UTF8_NORM = downCritical(
            "sz_utf8_norm_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_UTF8_FIND_DENORMALIZED = downCritical(
            "sz_utf8_find_denormalized_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_SEQUENCE_ARGSORT = down(
            "sz_sequence_argsort_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_SEQUENCE_ARGSORT_UNCASED = down(
            "sz_sequence_argsort_uncased_best",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
    private static final MethodHandle STRINGZILLA_SEQUENCE_INTERSECT = down(
            "sz_sequence_intersect_best",
            FunctionDescriptor.of(
                    JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));

    // endregion

    /** Every CPU capability, {@code sz_cap_cpus_k}, which every call passes and the library clamps to what this CPU
     *  runs. */
    private static final long CPUS = (1L << 48) - 1;

    private static MethodHandle down(String name, FunctionDescriptor desc, Linker.Option... opts) {
        MemorySegment addr = LOOKUP.find(name).orElseThrow(() -> new UnsatisfiedLinkError("missing symbol: " + name));
        return LINKER.downcallHandle(addr, desc, opts);
    }

    private static MethodHandle downCritical(String name, FunctionDescriptor desc) {
        return down(name, desc, Linker.Option.critical(true));
    }

    private static void check(String function, int status) {
        if (status != 0) throw new StatusException(function, status);
    }

    // region Search (MemorySegment)

    /** First byte offset of {@code needle} in the off-heap {@code haystack}, or -1. */
    public static long indexOf(MemorySegment haystack, MemorySegment needle) {
        long n = needle.byteSize();
        if (n == 0) return 0;
        if (n > haystack.byteSize()) return -1;
        requireNative(haystack);
        try {
            long[] match = new long[1];
            check("sz_find_best", (int) STRINGZILLA_FIND.invokeExact(
                    haystack,
                    haystack.byteSize(),
                    needle,
                    n,
                    MemorySegment.ofArray(match),
                    CPUS,
                    MemorySegment.NULL));
            return match[0] == 0 ? -1 : match[0] - haystack.address();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Last byte offset of {@code needle} in the off-heap {@code haystack}, or -1. */
    public static long lastIndexOf(MemorySegment haystack, MemorySegment needle) {
        long n = needle.byteSize();
        if (n == 0) return haystack.byteSize();
        if (n > haystack.byteSize()) return -1;
        requireNative(haystack);
        try {
            long[] match = new long[1];
            check("sz_rfind_best", (int) STRINGZILLA_RFIND.invokeExact(
                    haystack,
                    haystack.byteSize(),
                    needle,
                    n,
                    MemorySegment.ofArray(match),
                    CPUS,
                    MemorySegment.NULL));
            return match[0] == 0 ? -1 : match[0] - haystack.address();
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
            return findByteset(h.asSlice(0, haystack.length), set.toSegment(a));
        }
    }

    public static long lastIndexOfAny(byte[] haystack, Byteset set) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment h = a.allocate(haystack.length == 0 ? 1 : haystack.length);
            MemorySegment.copy(haystack, 0, h, JAVA_BYTE, 0, haystack.length);
            return rfindByteset(h.asSlice(0, haystack.length), set.toSegment(a));
        }
    }

    // endregion

    // region Comparison

    /** Byte-wise equality. Like {@link java.util.Arrays#equals(byte[], byte[])}, SIMD-accelerated. */
    public static boolean equal(byte[] a, byte[] b) {
        if (a.length != b.length) return false;
        if (a.length == 0) return true;
        try {
            int[] equal = new int[1];
            check("sz_equal_best", (int) STRINGZILLA_EQUAL.invokeExact(
                    MemorySegment.ofArray(a),
                    MemorySegment.ofArray(b),
                    (long) a.length,
                    MemorySegment.ofArray(equal),
                    CPUS,
                    MemorySegment.NULL));
            return equal[0] != 0;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Lexicographic byte comparison: -1, 0, or 1. Like {@link java.util.Arrays#compare(byte[], byte[])}. */
    public static int compare(byte[] a, byte[] b) {
        try {
            int[] ordering = new int[1];
            check("sz_order_best", (int) STRINGZILLA_ORDER.invokeExact(
                    MemorySegment.ofArray(a),
                    (long) a.length,
                    MemorySegment.ofArray(b),
                    (long) b.length,
                    MemorySegment.ofArray(ordering),
                    CPUS,
                    MemorySegment.NULL));
            return ordering[0];
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
        return hash(MemorySegment.ofArray(data), seed);
    }

    public static long hash(MemorySegment data, long seed) {
        try {
            long[] hash = new long[1];
            check("sz_hash_best", (int) STRINGZILLA_HASH.invokeExact(
                    data, data.byteSize(), seed, MemorySegment.ofArray(hash), CPUS, MemorySegment.NULL));
            return hash[0];
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static long byteSum(byte[] data) {
        try {
            long[] checksum = new long[1];
            check("sz_bytesum_best", (int) STRINGZILLA_BYTESUM.invokeExact(
                    MemorySegment.ofArray(data),
                    (long) data.length,
                    MemorySegment.ofArray(checksum),
                    CPUS,
                    MemorySegment.NULL));
            return checksum[0];
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
            long[] count = new long[1];
            check("sz_utf8_count_best", (int) STRINGZILLA_UTF8_COUNT.invokeExact(
                    text, text.byteSize(), MemorySegment.ofArray(count), CPUS, MemorySegment.NULL));
            return count[0];
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Byte offset of the {@code index}-th codepoint (0-based), or -1 if fewer exist. */
    public static long seekRune(byte[] text, long index) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment t = a.allocate(Math.max(text.length, 1));
            MemorySegment.copy(text, 0, t, JAVA_BYTE, 0, text.length);
            MemorySegment position = a.allocate(ADDRESS);
            check("sz_utf8_seek_best", (int) STRINGZILLA_UTF8_SEEK.invokeExact(
                    t, (long) text.length, index, position, CPUS, MemorySegment.NULL));
            long found = position.get(ADDRESS, 0).address();
            return found == 0 ? -1 : found - t.address();
        } catch (Throwable e) {
            throw rethrow(e);
        }
    }

    /** Decodes codepoints into {@code destination} (one Int32 scalar each; ill-formed or truncated
     *  → U+FFFD) and returns the count written, stopping early only when {@code destination} fills.
     *  Like {@code String.codePoints()}, SIMD-accelerated. */
    public static int decode(byte[] text, int[] destination) {
        return decode(MemorySegment.ofArray(text), destination);
    }

    public static int decode(MemorySegment text, int[] destination) {
        long[] unpacked = new long[1];
        long[] consumedIgnored = new long[1];
        try {
            check("sz_utf8_decode_best", (int) STRINGZILLA_UTF8_DECODE.invokeExact(
                    text,
                    text.byteSize(),
                    MemorySegment.ofArray(destination),
                    (long) destination.length,
                    MemorySegment.ofArray(unpacked),
                    MemorySegment.ofArray(consumedIgnored),
                    CPUS,
                    MemorySegment.NULL));
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
        int[] runes = new int[Math.toIntExact(text.byteSize())]; // a rune per byte at most
        return java.util.Arrays.copyOf(runes, decode(text, runes));
    }

    // endregion

    // region UTF-8 Segmentation

    /** A tiling segmentation: every byte lands in exactly one segment. */
    public enum SegmentKind {
        GRAPHEMES("sz_utf8_graphemes_best"),
        WORDS("sz_utf8_wordbreaks_best"),
        SENTENCES("sz_utf8_sentences_best"),
        LINE_BREAKS("sz_utf8_linebreaks_best");

        private final String function;
        private final MethodHandle kernel;

        SegmentKind(String function) {
            this.function = function;
            this.kernel = downCritical(
                    function,
                    FunctionDescriptor.of(
                            JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS));
        }
    }

    /** The separator tokens that {@link TokenSplits} splits on. */
    private enum TokenKind {
        NEWLINES("sz_utf8_newlines_best"),
        WHITESPACES("sz_utf8_whitespaces_best"),
        DELIMITERS("sz_utf8_delimiters_best");

        private final String function;
        private final MethodHandle kernel;

        TokenKind(String function) {
            this.function = function;
            this.kernel = downCritical(
                    function,
                    FunctionDescriptor.of(
                            JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG,
                            ADDRESS));
        }
    }

    /** Writes the byte lengths of the next segments of {@code text}, from {@code textOffset} for
     *  {@code textLength} bytes, into the caller's {@code lengths} and returns the count written.
     *  Allocation-free. Segments tile the text, each starting where the previous one ended, so
     *  resume at the sum of the lengths; only an empty text yields 0. Like
     *  {@link java.text.BreakIterator} / ICU4J, but deterministic, SIMD-accelerated, over UTF-8
     *  bytes (no locale, no transcode). By {@code GRAPHEMES}, the flag 🇺🇸 is one cluster spanning
     *  two codepoints; by {@code WORDS}, "Hello, 世界" splits the Latin run from the CJK run. */
    public static int segment(byte[] text, int textOffset, int textLength, SegmentKind kind, long[] lengths) {
        return segment(MemorySegment.ofArray(text), textOffset, textLength, kind, lengths);
    }

    public static int segment(MemorySegment text, long textOffset, long textLength, SegmentKind kind, long[] lengths) {
        try {
            long[] count = new long[1];
            check(kind.function, (int) kind.kernel.invokeExact(
                    text.asSlice(textOffset, textLength),
                    textLength,
                    MemorySegment.ofArray(lengths),
                    (long) lengths.length,
                    MemorySegment.ofArray(count),
                    CPUS,
                    MemorySegment.NULL));
            return (int) count[0];
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Writes the (offset, length) of the next separator tokens of {@code text} into
     *  {@code offsets} and {@code lengths} and returns the count. Only a full batch leaves text
     *  unscanned, and the scan resumes right after its last token. */
    private static int tokenize(MemorySegment text, TokenKind kind, long[] offsets, long[] lengths) {
        try {
            long[] count = new long[1];
            long[] consumed = new long[1];
            check(kind.function, (int) kind.kernel.invokeExact(
                    text,
                    text.byteSize(),
                    MemorySegment.ofArray(offsets),
                    MemorySegment.ofArray(lengths),
                    (long) offsets.length,
                    MemorySegment.ofArray(count),
                    MemorySegment.ofArray(consumed),
                    CPUS,
                    MemorySegment.NULL));
            return (int) count[0];
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    // endregion

    // region UTF-8 Case Folding and Uncased Matching

    /** Folds into the caller's {@code destination} (must hold ≥ text.length*3 bytes) and returns the
     *  bytes written. Allocation-free. */
    public static int caseFold(byte[] text, byte[] destination) {
        if (destination.length < (long) text.length * 3)
            throw new IllegalArgumentException("destination must hold at least text.length*3 bytes");
        try {
            long[] written = new long[1];
            check("sz_utf8_uncased_fold_best", (int) STRINGZILLA_UTF8_UNCASED_FOLD.invokeExact(
                    MemorySegment.ofArray(text),
                    (long) text.length,
                    MemorySegment.ofArray(destination),
                    MemorySegment.ofArray(written),
                    CPUS,
                    MemorySegment.NULL));
            return (int) written[0];
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Full Unicode case folding (UAX #21, one-to-many, e.g. ß → "ss"). Fills a gap: the JDK ships
     *  no case folding (only locale {@code toLowerCase}/{@code toUpperCase}). */
    public static byte[] caseFold(byte[] text) {
        if (text.length == 0) return new byte[0];
        byte[] dst = new byte[Math.multiplyExact(text.length, 3)]; // worst-case 3x expansion
        return java.util.Arrays.copyOf(dst, caseFold(text, dst));
    }

    /** Result of a case-insensitive search: byte {@code offset} (-1 if not found) and the matched
     *  byte length (may differ from the needle due to folding). */
    public record Match(long offset, long matchedLength) {}

    /** Case-insensitive (full-fold) search. Fills a gap: no JDK caseless UTF-8 substring search.
     *  For repeated searches with the same needle, prefer {@link UncasedNeedle}. */
    public static Match uncasedIndexOf(byte[] haystack, byte[] needle) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment prepared = uncasedNeedleInit(nativeView(MemorySegment.ofArray(needle), a), a);
            return uncasedMatch(nativeView(MemorySegment.ofArray(haystack), a), prepared);
        }
    }

    /** Case-insensitive (full-fold) lexicographic comparison: -1, 0, or 1. Fills a gap: no JDK
     *  Unicode case-folded ordinal comparison. */
    public static int uncasedCompare(byte[] a, byte[] b) {
        try {
            int[] ordering = new int[1];
            check("sz_utf8_uncased_order_best", (int) STRINGZILLA_UTF8_UNCASED_ORDER.invokeExact(
                    MemorySegment.ofArray(a),
                    (long) a.length,
                    MemorySegment.ofArray(b),
                    (long) b.length,
                    MemorySegment.ofArray(ordering),
                    CPUS,
                    MemorySegment.NULL));
            return ordering[0];
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Prepares the native {@code needle}, which must outlive the result, into an {@code arena}
     *  struct. */
    private static MemorySegment uncasedNeedleInit(MemorySegment needle, Arena arena) {
        MemorySegment prepared = arena.allocate(UNCASED_NEEDLE);
        try {
            check("sz_utf8_uncased_needle_init_best", (int) STRINGZILLA_UTF8_UNCASED_NEEDLE_INIT.invokeExact(
                    needle, needle.byteSize(), prepared, CPUS, MemorySegment.NULL));
            return prepared;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Offset of the first match of the {@code prepared} needle in the native {@code haystack}, or
     *  -1; {@code found} receives the match address and its byte length. */
    private static long uncasedSearch(MemorySegment haystack, MemorySegment prepared, long[] found) {
        try {
            check("sz_utf8_uncased_search_best", (int) STRINGZILLA_UTF8_UNCASED_SEARCH.invokeExact(
                    haystack,
                    haystack.byteSize(),
                    prepared,
                    MemorySegment.ofArray(found),
                    MemorySegment.ofArray(found).asSlice(JAVA_LONG.byteSize()),
                    CPUS,
                    MemorySegment.NULL));
            return found[0] == 0 ? -1 : found[0] - haystack.address();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static Match uncasedMatch(MemorySegment haystack, MemorySegment prepared) {
        long[] found = new long[2];
        long offset = uncasedSearch(haystack, prepared, found);
        return offset < 0 ? new Match(-1, 0) : new Match(offset, found[1]);
    }

    // endregion

    // region Hash Extras, Random and Lookup

    /** Hashes {@code data} under each seed, writing results into {@code hashes} (must hold ≥ seeds.length). */
    public static void hashMultiseed(byte[] data, long[] seeds, long[] hashes) {
        if (hashes.length < seeds.length)
            throw new IllegalArgumentException("hashes must hold at least seeds.length entries");
        try {
            check("sz_hash_multiseed_best", (int) STRINGZILLA_HASH_MULTISEED.invokeExact(
                    MemorySegment.ofArray(data),
                    (long) data.length,
                    MemorySegment.ofArray(seeds),
                    (long) seeds.length,
                    MemorySegment.ofArray(hashes),
                    CPUS,
                    MemorySegment.NULL));
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
            check("sz_fill_random_best", (int) STRINGZILLA_FILL_RANDOM.invokeExact(
                    MemorySegment.ofArray(buffer),
                    (long) buffer.length,
                    nonce,
                    CPUS,
                    MemorySegment.NULL));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Byte-wise table transform: {@code destination[i] = lut[source[i]]}. {@code lut} must be 256 bytes. */
    public static void lookup(byte[] destination, byte[] source, byte[] lut) {
        if (lut.length != 256) throw new IllegalArgumentException("LUT must be 256 bytes");
        if (destination.length < source.length) throw new IllegalArgumentException("destination too small");
        try {
            check("sz_lookup_best", (int) STRINGZILLA_LOOKUP.invokeExact(
                    MemorySegment.ofArray(destination),
                    MemorySegment.ofArray(source),
                    (long) source.length,
                    MemorySegment.ofArray(lut),
                    CPUS,
                    MemorySegment.NULL));
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
        if (destination.length < (long) text.length * 18)
            throw new IllegalArgumentException("destination must hold at least text.length*18 bytes");
        try {
            long[] written = new long[1];
            check("sz_utf8_norm_best", (int) STRINGZILLA_UTF8_NORM.invokeExact(
                    MemorySegment.ofArray(text),
                    (long) text.length,
                    formCode(form),
                    MemorySegment.ofArray(destination),
                    MemorySegment.ofArray(written),
                    CPUS,
                    MemorySegment.NULL));
            return (int) written[0];
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Normalizes {@code text} to the given form. Equivalent to {@link java.text.Normalizer#normalize},
     *  over UTF-8 bytes with no UTF-16 transcode. For example, "e"+U+0301 composes into a precomposed "é"
     *  under {@code NFC} (and back under {@code NFD}); the ligature "ﬁ" expands into "fi" under {@code NFKC}. */
    public static byte[] normalize(byte[] text, NormalForm form) {
        if (text.length == 0) return new byte[0];
        byte[] dst = new byte[Math.multiplyExact(text.length, 18)]; // worst-case per-codepoint expansion
        return java.util.Arrays.copyOf(dst, normalize(text, form, dst));
    }

    /** True if {@code text} is already in the given form. Like {@link java.text.Normalizer#isNormalized}. */
    public static boolean isNormalized(byte[] text, NormalForm form) {
        try {
            long[] violation = new long[1];
            check("sz_utf8_find_denormalized_best", (int) STRINGZILLA_UTF8_FIND_DENORMALIZED.invokeExact(
                    MemorySegment.ofArray(text),
                    (long) text.length,
                    formCode(form),
                    MemorySegment.ofArray(violation),
                    CPUS,
                    MemorySegment.NULL));
            return violation[0] == 0;
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
            MethodHandle mh = uncased ? STRINGZILLA_SEQUENCE_ARGSORT_UNCASED : STRINGZILLA_SEQUENCE_ARGSORT;
            check(uncased ? "sz_sequence_argsort_uncased_best" : "sz_sequence_argsort_best", (int) mh.invokeExact(
                    t.sequence,
                    top,
                    reverse ? 1 : 0,
                    MemorySegment.NULL,
                    orderSeg,
                    CPUS,
                    MemorySegment.NULL));
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
        // A range past the text would throw inside the FFM upcall, and that terminates the JVM.
        for (int i = 0; i < n; i++) java.util.Objects.checkFromIndexSize(starts[i], lengths[i], text.length);
        try (Arena a = Arena.ofConfined()) {
            SeqTable t = new SeqTable(text, starts, lengths, n, a);
            MemorySegment orderSeg = a.allocate(JAVA_LONG.byteSize() * n);
            MethodHandle mh = uncased ? STRINGZILLA_SEQUENCE_ARGSORT_UNCASED : STRINGZILLA_SEQUENCE_ARGSORT;
            check(uncased ? "sz_sequence_argsort_uncased_best" : "sz_sequence_argsort_best", (int) mh.invokeExact(
                    t.sequence,
                    top,
                    reverse ? 1 : 0,
                    MemorySegment.NULL,
                    orderSeg,
                    CPUS,
                    MemorySegment.NULL));
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
            check("sz_sequence_intersect_best", (int) STRINGZILLA_SEQUENCE_INTERSECT.invokeExact(
                    ta.sequence,
                    tb.sequence,
                    MemorySegment.NULL,
                    seed,
                    size,
                    fp,
                    sp,
                    CPUS,
                    MemorySegment.NULL));
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
        return new TokenSplits(MemorySegment.ofArray(text), TokenKind.NEWLINES, SplitParts.BETWEEN, false);
    }

    public static TokenSplits splitNewlines(MemorySegment text) {
        return new TokenSplits(text, TokenKind.NEWLINES, SplitParts.BETWEEN, false);
    }

    public static TokenSplits splitWhitespaces(byte[] text) {
        return new TokenSplits(MemorySegment.ofArray(text), TokenKind.WHITESPACES, SplitParts.BETWEEN, false);
    }

    public static TokenSplits splitWhitespaces(MemorySegment text) {
        return new TokenSplits(text, TokenKind.WHITESPACES, SplitParts.BETWEEN, false);
    }

    public static TokenSplits splitDelimiters(byte[] text) {
        return new TokenSplits(MemorySegment.ofArray(text), TokenKind.DELIMITERS, SplitParts.BETWEEN, false);
    }

    public static TokenSplits splitDelimiters(MemorySegment text) {
        return new TokenSplits(text, TokenKind.DELIMITERS, SplitParts.BETWEEN, false);
    }

    /** Lazily enumerates the byte offsets of every match of {@code needle} in {@code haystack}; chain
     *  {@code .overlapping()} for overlapping matches. Counting the result is the occurrence count. */
    public static Matches matches(byte[] haystack, byte[] needle) {
        return matches(MemorySegment.ofArray(haystack), MemorySegment.ofArray(needle));
    }

    public static Matches matches(MemorySegment haystack, MemorySegment needle) {
        return new Matches(haystack, needle, Math.max(needle.byteSize(), 1));
    }

    /** Lazily enumerates case-insensitive (full-fold) matches; chain {@code .overlapping()}. Each
     *  {@link Match} carries the offset and matched length (folding may make the latter differ). An
     *  empty needle matches at every codepoint boundary, the end included. */
    public static UncasedMatches uncasedMatches(byte[] haystack, byte[] needle) {
        return new UncasedMatches(MemorySegment.ofArray(haystack), MemorySegment.ofArray(needle), false);
    }

    public static UncasedMatches uncasedMatches(MemorySegment haystack, MemorySegment needle) {
        return new UncasedMatches(haystack, needle, false);
    }

    /** Splits at the first occurrence of {@code separator} into (before, separator, after) views; if
     *  absent, returns ({@code text}, empty, empty). */
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

    /** As {@link #partition} but at the last occurrence; if absent, returns (empty, empty, {@code text}). */
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

    /** Which runtime a device belongs to, as the {@code sz_<kind>_*} C functions name it. */
    public enum DeviceKind {
        CPU,
        CUDA,
        ROCM,
        METAL
    }

    /** One device StringZilla can run on: the host CPU, or a GPU by its runtime's own ordinal, the one
     *  {@code cudaSetDevice} or {@code hipSetDevice} takes, or the position in Metal's device list. Every call
     *  of this class dispatches over the CPU's {@link #capabilitiesEnabled()}; GPUs only report their masks.
     *
     *  <p>Prefer {@link #capabilitiesEnabled()} unless you specifically mean one of the raw axes:
     *  {@link #capabilitiesDetected()} describes the device and says nothing about whether a kernel was compiled
     *  into this binary, so selecting on it alone claims support for code that may not exist. */
    public record Device(DeviceKind kind, long ordinal) {

        /** Device {@code ordinal} of {@code kind}, throwing {@link StatusException} past the last one. */
        public Device {
            java.util.Objects.requireNonNull(kind);
            if (ordinal < 0 || ordinal >= count(kind))
                throw new StatusException("Device(" + kind + ", " + ordinal + ")", STATUS_MISSING_GPU);
        }

        /** The host CPU, which every build has. */
        public static Device cpu() {
            return new Device(DeviceKind.CPU, 0);
        }

        /** How many devices of {@code kind} the process sees: one CPU, or the GPUs its runtime counts. Throws
         *  {@link StatusException} without a GPU of {@code kind}. */
        public static long count(DeviceKind kind) {
            return switch (kind) {
                case CPU -> 1;
                case CUDA -> query(STRINGZILLA_CUDA_COUNT_DEVICES, "sz_cuda_count_devices");
                case ROCM -> query(STRINGZILLA_ROCM_COUNT_DEVICES, "sz_rocm_count_devices");
                case METAL -> query(STRINGZILLA_METAL_COUNT_DEVICES, "sz_metal_count_devices");
            };
        }

        /** What this device runs, as a bitmask, whether or not this binary holds kernels for it. */
        public long capabilitiesDetected() {
            return switch (kind) {
                case CPU -> query(STRINGZILLA_CPU_CAPABILITIES_DETECTED, "sz_cpu_capabilities_detected");
                case CUDA -> query(STRINGZILLA_CUDA_CAPABILITIES_DETECTED, "sz_cuda_capabilities_detected", ordinal);
                case ROCM -> query(STRINGZILLA_ROCM_CAPABILITIES_DETECTED, "sz_rocm_capabilities_detected", ordinal);
                case METAL -> query(STRINGZILLA_METAL_CAPABILITIES_DETECTED, "sz_metal_capabilities_detected", ordinal);
            };
        }

        /** What this binary holds kernels for on devices of this kind, as a bitmask, whether or not this one runs
         *  them. Decided at build time. */
        public long capabilitiesCompiled() {
            return switch (kind) {
                case CPU -> query(STRINGZILLA_CPU_CAPABILITIES_COMPILED, "sz_cpu_capabilities_compiled");
                case CUDA -> query(STRINGZILLA_CUDA_CAPABILITIES_COMPILED, "sz_cuda_capabilities_compiled");
                case ROCM -> query(STRINGZILLA_ROCM_CAPABILITIES_COMPILED, "sz_rocm_capabilities_compiled");
                case METAL -> query(STRINGZILLA_METAL_CAPABILITIES_COMPILED, "sz_metal_capabilities_compiled");
            };
        }

        /** What this device's calls run, as a bitmask: {@link #capabilitiesDetected()} and
         *  {@link #capabilitiesCompiled()} at once. On the CPU it always includes the serial bit, {@code 1}. */
        public long capabilitiesEnabled() {
            return switch (kind) {
                case CPU -> query(STRINGZILLA_CPU_CAPABILITIES_ENABLED, "sz_cpu_capabilities_enabled");
                case CUDA -> query(STRINGZILLA_CUDA_CAPABILITIES_ENABLED, "sz_cuda_capabilities_enabled", ordinal);
                case ROCM -> query(STRINGZILLA_ROCM_CAPABILITIES_ENABLED, "sz_rocm_capabilities_enabled", ordinal);
                case METAL -> query(STRINGZILLA_METAL_CAPABILITIES_ENABLED, "sz_metal_capabilities_enabled", ordinal);
            };
        }

        /** Prepares the calling thread for the kernels of {@code capabilities}, usually
         *  {@link #capabilitiesEnabled()}; call it once on every thread that runs them. Throws
         *  {@link StatusException} on a GPU, which has no thread state to configure. */
        public void configureThread(long capabilities) {
            if (kind != DeviceKind.CPU) throw new StatusException(this + ".configureThread", STATUS_MISSING_KERNEL);
            try {
                check("sz_cpu_configure_thread", (int) STRINGZILLA_CPU_CONFIGURE_THREAD.invokeExact(capabilities));
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }
    }

    /** Names the bits of {@code capabilities}, comma-separated, like {@code "serial,haswell,skylake"}. */
    public static String capabilitiesName(long capabilities) {
        byte[] names = new byte[CAPABILITIES_NAME_CAPACITY];
        try {
            long written = (long) STRINGZILLA_NAME_CAPABILITIES.invokeExact(
                    capabilities, MemorySegment.ofArray(names), (long) names.length);
            return new String(names, 0, (int) written, StandardCharsets.UTF_8);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Calls a device {@code query} for its one out-word, a count or a capability mask. */
    private static long query(MethodHandle query, String function) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment word = arena.allocate(JAVA_LONG);
            check(function, (int) query.invokeExact(word));
            return word.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Calls a device {@code query} of GPU {@code ordinal} for its capability mask. */
    private static long query(MethodHandle query, String function, long ordinal) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment word = arena.allocate(JAVA_LONG);
            check(function, (int) query.invokeExact(ordinal, word));
            return word.get(JAVA_LONG, 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    public static String version() {
        try {
            int major = (int) STRINGZILLA_VERSION_MAJOR.invokeExact();
            int minor = (int) STRINGZILLA_VERSION_MINOR.invokeExact();
            int patch = (int) STRINGZILLA_VERSION_PATCH.invokeExact();
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

    /** A native call returned a non-success {@code sz_status_t}, kept in {@link #status()}. */
    public static final class StatusException extends IllegalStateException {
        private final int status;

        StatusException(String function, int status) {
            super(function + " failed with " + statusName(status) + " (" + status + ")");
            this.status = status;
        }

        /** The raw {@code sz_status_t} code, e.g. -10 for {@code sz_bad_alloc_k}. */
        public int status() {
            return status;
        }

        /** The C enumerator name of {@link #status()}, e.g. {@code "sz_bad_alloc_k"}. */
        public String statusName() {
            return statusName(status);
        }

        private static String statusName(int status) {
            return switch (status) {
                case -10 -> "sz_bad_alloc_k";
                case -12 -> "sz_invalid_utf8_k";
                case -13 -> "sz_contains_duplicates_k";
                case -14 -> "sz_overflow_risk_k";
                case -15 -> "sz_unexpected_dimensions_k";
                case -16 -> "sz_missing_gpu_k";
                case -17 -> "sz_device_code_mismatch_k";
                case -18 -> "sz_device_memory_mismatch_k";
                case -19 -> "sz_authentication_failed_k";
                case -20 -> "sz_missing_kernel_k";
                case -21 -> "sz_missing_library_k";
                default -> "sz_status_unknown_k";
            };
        }
    }

    /** A 256-bit set of byte values. */
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
                check("sz_hash_state_init_best", (int)
                        STRINGZILLA_HASH_STATE_INIT.invokeExact(state, seed, CPUS, MemorySegment.NULL));
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        public Hasher update(byte[] data) {
            try {
                check("sz_hash_state_update_best", (int) STRINGZILLA_HASH_STATE_UPDATE.invokeExact(
                        state,
                        MemorySegment.ofArray(data),
                        (long) data.length,
                        CPUS,
                        MemorySegment.NULL));
                return this;
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        /** Finalize without mutating state (may be called repeatedly). */
        public long digest() {
            try {
                long[] hash = new long[1];
                check("sz_hash_state_digest_best", (int) STRINGZILLA_HASH_STATE_DIGEST.invokeExact(
                        state, MemorySegment.ofArray(hash), CPUS, MemorySegment.NULL));
                return hash[0];
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        @Override
        public void close() {
            arena.close();
        }
    }

    /** A case-folding search needle, prepared once and reusable from any thread. Close to free
     *  native memory. */
    public static final class UncasedNeedle implements AutoCloseable {
        private final Arena arena = Arena.ofShared();
        private final MemorySegment prepared; // points into the arena's copy of the needle

        public UncasedNeedle(byte[] needle) {
            prepared = uncasedNeedleInit(nativeView(MemorySegment.ofArray(needle), arena), arena);
        }

        /** First match of this needle in {@code haystack} (caseless), or offset -1. */
        public Match indexIn(byte[] haystack) {
            try (Arena a = Arena.ofConfined()) {
                return uncasedMatch(nativeView(MemorySegment.ofArray(haystack), a), prepared);
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
                check("sz_sha256_state_init_best", (int)
                        STRINGZILLA_SHA256_INIT.invokeExact(state, CPUS, MemorySegment.NULL));
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        public Sha256 update(byte[] data) {
            try {
                check("sz_sha256_state_update_best", (int) STRINGZILLA_SHA256_UPDATE.invokeExact(
                        state,
                        MemorySegment.ofArray(data),
                        (long) data.length,
                        CPUS,
                        MemorySegment.NULL));
                return this;
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }

        /** Writes the 32-byte digest into {@code destination} (must be ≥ 32 bytes; non-mutating). Allocation-free. */
        public void digest(byte[] destination) {
            if (destination.length < 32) throw new IllegalArgumentException("destination must hold at least 32 bytes");
            try {
                check("sz_sha256_state_digest_best", (int) STRINGZILLA_SHA256_DIGEST.invokeExact(
                        state, MemorySegment.ofArray(destination), CPUS, MemorySegment.NULL));
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

    /** A match inside a heap segment is an address the JVM never exposes, so it can't become an offset. */
    private static void requireNative(MemorySegment haystack) {
        if (!haystack.isNative()) throw new IllegalArgumentException("haystack must be an off-heap segment");
    }

    private static long findByteset(MemorySegment haystack, MemorySegment bytesetSeg) {
        if (haystack.byteSize() == 0) return -1;
        try {
            long[] match = new long[1];
            check("sz_find_byteset_best", (int) STRINGZILLA_FIND_BYTESET.invokeExact(
                    haystack,
                    haystack.byteSize(),
                    bytesetSeg,
                    MemorySegment.ofArray(match),
                    CPUS,
                    MemorySegment.NULL));
            return match[0] == 0 ? -1 : match[0] - haystack.address();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static long rfindByteset(MemorySegment haystack, MemorySegment bytesetSeg) {
        if (haystack.byteSize() == 0) return -1;
        try {
            long[] match = new long[1];
            check("sz_rfind_byteset_best", (int) STRINGZILLA_RFIND_BYTESET.invokeExact(
                    haystack,
                    haystack.byteSize(),
                    bytesetSeg,
                    MemorySegment.ofArray(match),
                    CPUS,
                    MemorySegment.NULL));
            return match[0] == 0 ? -1 : match[0] - haystack.address();
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

    /** Adapts a cursor to an {@link Iterator}, looking one element ahead. */
    private static <T> Iterator<T> iterator(BooleanSupplier advance, Supplier<T> current) {
        return new Iterator<>() {
            private boolean pulled;

            @Override
            public boolean hasNext() {
                return pulled || (pulled = advance.getAsBoolean());
            }

            @Override
            public T next() {
                if (!hasNext()) throw new NoSuchElementException();
                pulled = false;
                return current.get();
            }
        };
    }

    /** Zero-allocation forward cursor over tiling UTF-8 segments (words, graphemes, …). Reuses one 64-entry
     *  batch buffer; call {@link #next()} then read {@link #startOffset()}/{@link #length()}. */
    public static final class Segmenter {
        private final MemorySegment text;
        private final SegmentKind kind;
        private final long[] lengths = new long[64];
        private int count, index;
        private long start, length; // the current segment; the next one starts at its end

        Segmenter(MemorySegment text, SegmentKind kind) {
            this.text = text;
            this.kind = kind;
        }

        public boolean next() {
            long end = start + length;
            if (index == count) {
                if (end == text.byteSize()) return false;
                count = segment(text, end, text.byteSize() - end, kind, lengths);
                index = 0;
            }
            start = end;
            length = lengths[index++];
            return true;
        }

        public long startOffset() {
            return start;
        }

        public long length() {
            return length;
        }

        public MemorySegment current() {
            return text.asSlice(start, length);
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
            Segmenter cursor = cursor();
            return StringZilla.iterator(cursor::next, cursor::current);
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
        private final MemorySegment byteset; // null for a substring split
        private final boolean reverse, keepSeparator, skipEmpty;
        private long splitsLeft; // negative for unlimited
        private long cursor; // start of the remaining text, its end in reverse, or -1 once done
        private long start, length;

        Splitter(
                MemorySegment data,
                MemorySegment needle,
                MemorySegment byteset,
                boolean reverse,
                boolean keepSeparator,
                boolean skipEmpty,
                long maxSplit) {
            this.data = data;
            this.needle = needle;
            this.byteset = byteset;
            this.reverse = reverse;
            this.keepSeparator = keepSeparator;
            this.skipEmpty = skipEmpty;
            // An empty separator yields the whole text.
            this.splitsLeft = needle != null && needle.byteSize() == 0 ? 0 : maxSplit;
            this.cursor = reverse ? data.byteSize() : 0;
        }

        public boolean next() {
            while (cursor >= 0) {
                MemorySegment rest = reverse ? data.asSlice(0, cursor) : data.asSlice(cursor);
                long at = splitsLeft == 0 ? -1 : find(rest);
                if (at < 0) {
                    start = reverse ? 0 : cursor;
                    length = rest.byteSize();
                    cursor = -1;
                } else {
                    long separatorLength = needle != null ? needle.byteSize() : 1;
                    splitsLeft--;
                    if (reverse) {
                        start = keepSeparator ? at : at + separatorLength;
                        length = cursor - start;
                        cursor = at;
                    } else {
                        start = cursor;
                        length = at + (keepSeparator ? separatorLength : 0);
                        cursor += at + separatorLength;
                    }
                }
                if (!skipEmpty || length != 0) return true;
            }
            return false;
        }

        private long find(MemorySegment rest) {
            if (needle != null) return reverse ? lastIndexOf(rest, needle) : indexOf(rest, needle);
            return reverse ? rfindByteset(rest, byteset) : findByteset(rest, byteset);
        }

        public long startOffset() {
            return start;
        }

        public long length() {
            return length;
        }

        public MemorySegment current() {
            return data.asSlice(start, length);
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

        /** Stop after {@code limit} separators, leaving the rest as one final segment: 0 yields the
         *  whole text, and a negative limit, the default, splits at every separator. */
        public Splits withMaxSplit(long limit) {
            return new Splits(text, separator, byteset, reverse, keepSeparator, skipEmpty, limit);
        }

        public Splitter cursor() {
            Arena arena = Arena.ofAuto();
            MemorySegment data = nativeView(text, arena);
            MemorySegment needle = separator != null ? nativeView(separator, arena) : null;
            MemorySegment set = byteset != null ? byteset.toSegment(arena) : null;
            return new Splitter(data, needle, set, reverse, keepSeparator, skipEmpty, maxSplit);
        }

        @Override
        public Iterator<MemorySegment> iterator() {
            Splitter cursor = cursor();
            return StringZilla.iterator(cursor::next, cursor::current);
        }

        public Stream<MemorySegment> stream() {
            return StreamSupport.stream(spliterator(), false);
        }
    }

    /** Zero-allocation cursor over separator-token splits (newlines, whitespace, delimiters), batched
     *  64 separators per native call, applying the {@link SplitParts} policy. */
    public static final class TokenSplitter {
        private final MemorySegment text;
        private final TokenKind kind;
        private final SplitParts parts;
        private final boolean skipEmpty;
        private final long[] offsets = new long[64];
        private final long[] lengths = new long[64];
        // Step 2k is the gap before token k and 2k + 1 the token; past 2 * count once done.
        private int count, step;
        private long base, start, length; // base is where the batch's scan began

        TokenSplitter(MemorySegment text, TokenKind kind, SplitParts parts, boolean skipEmpty) {
            this.text = text;
            this.kind = kind;
            this.parts = parts;
            this.skipEmpty = skipEmpty;
            this.count = tokenize(text, kind, offsets, lengths);
        }

        public boolean next() {
            while (true) {
                if (step == 2 * count && count == offsets.length) {
                    base += offsets[count - 1] + lengths[count - 1];
                    count = tokenize(text.asSlice(base), kind, offsets, lengths);
                    step = 0;
                }
                if (step > 2 * count) return false;
                int token = step >> 1;
                boolean isToken = (step++ & 1) != 0;
                if (parts == (isToken ? SplitParts.BETWEEN : SplitParts.SEPARATORS)) continue;
                long from = isToken ? offsets[token] : token == 0 ? 0 : offsets[token - 1] + lengths[token - 1];
                long to = isToken ? from + lengths[token] : token == count ? text.byteSize() - base : offsets[token];
                if (skipEmpty && from == to) continue;
                start = base + from;
                length = to - from;
                return true;
            }
        }

        public long startOffset() {
            return start;
        }

        public long length() {
            return length;
        }

        public MemorySegment current() {
            return text.asSlice(start, length);
        }
    }

    /** Iterable / streamable view over separator-token splits, yielding zero-copy slices. */
    public static final class TokenSplits implements Iterable<MemorySegment> {
        private final MemorySegment text;
        private final TokenKind kind;
        private final SplitParts parts;
        private final boolean skipEmpty;

        TokenSplits(MemorySegment text, TokenKind kind, SplitParts parts, boolean skipEmpty) {
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
            TokenSplitter cursor = cursor();
            return StringZilla.iterator(cursor::next, cursor::current);
        }

        public Stream<MemorySegment> stream() {
            return StreamSupport.stream(spliterator(), false);
        }
    }

    /** Zero-allocation cursor over substring matches, yielding each match's byte offset. */
    public static final class Matcher {
        private final MemorySegment data;
        private final MemorySegment needle;
        private final long stride;
        private long cursor, offset; // cursor passes the end once exhausted

        Matcher(MemorySegment data, MemorySegment needle, long stride) {
            this.data = data;
            this.needle = needle;
            this.stride = stride;
        }

        public boolean next() {
            if (cursor > data.byteSize()) return false;
            long found = indexOf(data.asSlice(cursor), needle);
            if (found < 0) {
                cursor = data.byteSize() + 1;
                return false;
            }
            offset = cursor + found;
            cursor = offset + stride;
            return true;
        }

        public long offset() {
            return offset;
        }
    }

    /** Streamable view over substring matches; {@link #overlapping()} reports overlapping matches
     *  and {@link #cursor()} is the zero-allocation path. An empty needle matches at every offset,
     *  the end included. */
    public static final class Matches {
        private final MemorySegment haystack;
        private final MemorySegment needle;
        private final long stride;

        Matches(MemorySegment haystack, MemorySegment needle, long stride) {
            this.haystack = haystack;
            this.needle = needle;
            this.stride = stride;
        }

        /** Report overlapping matches, stepping one byte past each instead of the needle. */
        public Matches overlapping() {
            return new Matches(haystack, needle, 1);
        }

        public Matcher cursor() {
            Arena arena = Arena.ofAuto();
            return new Matcher(nativeView(haystack, arena), nativeView(needle, arena), stride);
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

    /** Zero-allocation cursor over case-insensitive matches of a needle prepared once. */
    public static final class UncasedMatcher {
        private final MemorySegment data;
        private final MemorySegment needle; // prepared
        private final boolean overlapping;
        private final long[] found = new long[2]; // the match address and length the search writes
        private long cursor, offset; // cursor passes the end once exhausted

        UncasedMatcher(MemorySegment data, MemorySegment needle, boolean overlapping) {
            this.data = data;
            this.needle = needle;
            this.overlapping = overlapping;
        }

        public boolean next() {
            long textLength = data.byteSize();
            if (cursor > textLength) return false;
            long at = uncasedSearch(data.asSlice(cursor), needle, found);
            if (at < 0) {
                cursor = textLength + 1;
                return false;
            }
            offset = cursor + at;
            // Overlapping and empty matches step a codepoint, keeping to rune boundaries.
            cursor = offset
                    + (found[1] != 0 && !overlapping
                            ? found[1]
                            : offset < textLength
                                    ? Math.min(utf8LeadWidth(data.get(JAVA_BYTE, offset)), textLength - offset)
                                    : 1);
            return true;
        }

        public long offset() {
            return offset;
        }

        /** The matched byte length, which folding may make differ from the needle's. */
        public long matchedLength() {
            return found[1];
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
            MemorySegment needle = uncasedNeedleInit(nativeView(this.needle, arena), arena);
            return new UncasedMatcher(nativeView(haystack, arena), needle, overlapping);
        }

        public java.util.List<Match> toList() {
            UncasedMatcher cursor = cursor();
            java.util.List<Match> out = new java.util.ArrayList<>();
            while (cursor.next()) out.add(new Match(cursor.offset(), cursor.matchedLength()));
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
