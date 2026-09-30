/**
 *  @brief Zero-copy .NET bindings for the StringZilla core C library.
 *  @file csharp/StringZilla.cs
 *  @author Ash Vardanian
 *
 *  Exposes the core single-string operations over UTF-8 byte spans:
 *
 *  - Search: `IndexOf`, `LastIndexOf`, and byte-set search
 *  - Comparison: `Equal` and `Compare`
 *  - Hashing: `Hash`, `ByteSum`, incremental `Hasher`, and `Sha256`
 *  - UTF-8: codepoint count/seek/decode, segmentation, case folding, and normalization
 *  - Collections: `ArgSort` and `Intersect`
 *
 *  All offsets are UTF-8 byte offsets. The zero-copy surface is `ReadOnlySpan<byte>`;
 *  see the README for details.
 */

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace StringZilla;

/// <summary>Core StringZilla operations: search, compare, hash, and memory primitives.</summary>
public static unsafe class Sz {
    #region Searching

    /// <summary>First byte offset of <paramref name="needle"/> in <paramref name="haystack"/>, or -1.</summary>
    /// <remarks>Like <see cref="System.MemoryExtensions.IndexOf{T}(System.ReadOnlySpan{T},System.ReadOnlySpan{T})"/>,
    /// SIMD-accelerated for large inputs. Keywords: find, search, substring, indexof.</remarks>
    public static long IndexOf(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle) {
        if (needle.Length == 0) return 0;
        if (needle.Length > haystack.Length) return -1;
        fixed (byte* h = haystack)
        fixed (byte* n = needle) {
            nint r;
            Check("sz_find_best", Native.sz_find_best(
                (nint)h, (nuint)haystack.Length, (nint)n, (nuint)needle.Length, &r, CpuEnabled, nint.Zero));
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    public static long LastIndexOf(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle) {
        if (needle.Length == 0) return haystack.Length;
        if (needle.Length > haystack.Length) return -1;
        fixed (byte* h = haystack)
        fixed (byte* n = needle) {
            nint r;
            Check("sz_rfind_best", Native.sz_rfind_best(
                (nint)h, (nuint)haystack.Length, (nint)n, (nuint)needle.Length, &r, CpuEnabled, nint.Zero));
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    public static long IndexOf(ReadOnlySpan<byte> haystack, byte needle) {
        fixed (byte* h = haystack) {
            nint r;
            Check("sz_find_byte_best", Native.sz_find_byte_best(
                (nint)h, (nuint)haystack.Length, (nint)(&needle), &r, CpuEnabled, nint.Zero));
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    public static long LastIndexOf(ReadOnlySpan<byte> haystack, byte needle) {
        fixed (byte* h = haystack) {
            nint r;
            Check("sz_rfind_byte_best", Native.sz_rfind_byte_best(
                (nint)h, (nuint)haystack.Length, (nint)(&needle), &r, CpuEnabled, nint.Zero));
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    /// <summary>First offset of any byte present in <paramref name="set"/>, or -1.</summary>
    public static long IndexOfAny(ReadOnlySpan<byte> haystack, ref Byteset set) {
        fixed (byte* h = haystack)
        fixed (Byteset* s = &set) {
            nint r;
            Check("sz_find_byteset_best", Native.sz_find_byteset_best(
                (nint)h, (nuint)haystack.Length, (nint)s, &r, CpuEnabled, nint.Zero));
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    /// <summary>Last offset of any byte present in <paramref name="set"/>, or -1.</summary>
    public static long LastIndexOfAny(ReadOnlySpan<byte> haystack, ref Byteset set) {
        fixed (byte* h = haystack)
        fixed (Byteset* s = &set) {
            nint r;
            Check("sz_rfind_byteset_best", Native.sz_rfind_byteset_best(
                (nint)h, (nuint)haystack.Length, (nint)s, &r, CpuEnabled, nint.Zero));
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    #endregion

    #region Comparison

    /// <summary>Byte-wise equality.</summary>
    /// <remarks>Like <see cref="System.MemoryExtensions.SequenceEqual{T}(System.ReadOnlySpan{T},System.ReadOnlySpan{T})"/>.</remarks>
    public static bool Equal(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b) {
        if (a.Length != b.Length) return false;
        if (a.Length == 0) return true;
        int equal;
        fixed (byte* pa = a)
        fixed (byte* pb = b)
            Check("sz_equal_best", Native.sz_equal_best((nint)pa, (nint)pb, (nuint)a.Length, &equal, CpuEnabled, nint.Zero));
        return equal != 0;
    }

    /// <summary>Lexicographic byte comparison: -1, 0, or 1.</summary>
    /// <remarks>Like <see cref="System.MemoryExtensions.SequenceCompareTo{T}(System.ReadOnlySpan{T},System.ReadOnlySpan{T})"/>.</remarks>
    public static int Compare(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b) {
        int ordering;
        fixed (byte* pa = a)
        fixed (byte* pb = b)
            Check("sz_order_best", Native.sz_order_best(
                (nint)pa, (nuint)a.Length, (nint)pb, (nuint)b.Length, &ordering, CpuEnabled, nint.Zero));
        return ordering;
    }

    #endregion

    #region Hashing

    /// <summary>Fast 64-bit AES-based hash (SMHasher-quality), with optional <paramref name="seed"/>.</summary>
    /// <remarks>Unlike <see cref="string.GetHashCode()"/>, this is stable across runs/processes and
    /// identical across all StringZilla bindings. For cryptographic hashing use <see cref="Sha256"/>.</remarks>
    public static ulong Hash(ReadOnlySpan<byte> data, ulong seed = 0) {
        ulong hash;
        fixed (byte* p = data)
            Check("sz_hash_best", Native.sz_hash_best((nint)p, (nuint)data.Length, seed, &hash, CpuEnabled, nint.Zero));
        return hash;
    }

    public static ulong ByteSum(ReadOnlySpan<byte> data) {
        ulong checksum;
        fixed (byte* p = data)
            Check("sz_bytesum_best", Native.sz_bytesum_best((nint)p, (nuint)data.Length, &checksum, CpuEnabled, nint.Zero));
        return checksum;
    }

    #endregion

    #region UTF-8 Codepoints

    /// <summary>Counts Unicode scalar values (codepoints) in UTF-8 <paramref name="text"/>.</summary>
    /// <remarks>Like counting <see cref="System.Text.Rune"/>s, but SIMD-accelerated over UTF-8 bytes
    /// with no UTF-16 transcode. Keywords: codepoint, rune count, length.</remarks>
    /// <example><c>CountRunes("你好世界"u8)</c> is 4; <c>CountRunes("Hello🌍"u8)</c> is 6 — the astral
    /// emoji is a single scalar.</example>
    public static long CountRunes(ReadOnlySpan<byte> text) {
        nuint count;
        fixed (byte* p = text)
            Check("sz_utf8_count_best", Native.sz_utf8_count_best((nint)p, (nuint)text.Length, &count, CpuEnabled, nint.Zero));
        return (long)count;
    }

    /// <summary>Byte offset of the <paramref name="index"/>-th codepoint (0-based), or -1 if fewer exist.</summary>
    /// <remarks>Analogous to <see cref="string.EnumerateRunes"/> positioning, on UTF-8 bytes.</remarks>
    public static long SeekRune(ReadOnlySpan<byte> text, long index) {
        fixed (byte* p = text) {
            nint r;
            Check("sz_utf8_seek_best", Native.sz_utf8_seek_best(
                (nint)p, (nuint)text.Length, (nuint)index, &r, CpuEnabled, nint.Zero));
            return r == 0 ? -1 : (long)((byte*)r - p);
        }
    }

    /// <summary>Decodes UTF-8 codepoints into <paramref name="destination"/> (one Int32 scalar each;
    /// ill-formed sequences become U+FFFD). Returns the count written.</summary>
    /// <remarks>Like <see cref="System.Text.Rune.DecodeFromUtf8"/> in bulk, SIMD-accelerated. Emitted
    /// values are valid Unicode scalars (incl. U+FFFD): wrap with <c>new Rune(cp)</c>.</remarks>
    public static int Decode(ReadOnlySpan<byte> text, Span<int> destination) => Decode(text, destination, out _);

    /// <summary>As <see cref="Decode(System.ReadOnlySpan{byte},System.Span{int})"/>, also reporting how
    /// many input bytes were consumed so callers can resume. Backs the streaming rune enumerator.</summary>
    public static int Decode(ReadOnlySpan<byte> text, Span<int> destination, out long bytesConsumed) {
        fixed (byte* p = text)
        fixed (int* d = destination) {
            nuint unpacked, consumed;
            Check("sz_utf8_decode_best", Native.sz_utf8_decode_best(
                (nint)p, (nuint)text.Length, (nint)d, (nuint)destination.Length, (nint)(&unpacked), &consumed,
                CpuEnabled, nint.Zero));
            bytesConsumed = (long)consumed;
            return (int)unpacked;
        }
    }

    /// <summary>Allocates and returns all codepoints of <paramref name="text"/> as Int32 scalars.</summary>
    public static int[] DecodeAll(ReadOnlySpan<byte> text) {
        var result = new int[CountRunes(text)];
        Decode(text, result);
        return result;
    }

    #endregion

    #region UTF-8 Segmentation

    public enum SegmentKind { Graphemes, Words, Sentences, LineBreaks, Newlines, Whitespaces, Delimiters }

    /// <summary>Writes the byte (start, length) ranges of the next segments into the caller's
    /// <paramref name="starts"/> / <paramref name="lengths"/> spans and returns the count written.
    /// Allocation-free. Resumable: advance the input by <paramref name="bytesConsumed"/> and call
    /// again until it returns 0. Starts are relative to <paramref name="text"/>.</summary>
    /// <remarks>Graphemes ≈ <see cref="System.Globalization.StringInfo"/> /
    /// <see cref="System.Globalization.TextElementEnumerator"/>; words/sentences/lines follow
    /// UAX-29 / UAX-14. SIMD-accelerated, deterministic, over UTF-8 bytes (no locale, no transcode).
    /// Keywords: grapheme, word break, sentence, line break, TextElement, BreakIterator.</remarks>
    /// <example>Segmentation tiles the text — every byte lands in exactly one segment. By
    /// <see cref="SegmentKind.Graphemes"/>, the flag 🇺🇸 is one cluster spanning two codepoints and "e"+U+0301
    /// (a decomposed é) is one cluster spanning two; by <see cref="SegmentKind.Words"/>, "Hello, 世界" splits
    /// the Latin run from the CJK run.</example>
    public static int Segment(ReadOnlySpan<byte> text, SegmentKind kind, Span<long> starts, Span<long> lengths, out long bytesConsumed) {
        int cap = Math.Min(starts.Length, lengths.Length);
        fixed (byte* t = text)
        fixed (long* s = starts)
        fixed (long* l = lengths) {
            nuint count, consumed;
            SegmentChunk(kind, (nint)t, (nuint)text.Length, (nint)s, (nint)l, (nuint)cap, &count, (nint)(&consumed));
            bytesConsumed = (long)consumed;
            return (int)count;
        }
    }

    private static void SegmentChunk(
        SegmentKind kind, nint text, nuint len, nint starts, nint lengths, nuint cap, nuint* count, nint consumed) {
        ulong caps = CpuEnabled;
        (string function, int status) = kind switch {
            SegmentKind.Graphemes => ("sz_utf8_graphemes_best", Native.sz_utf8_graphemes_best(text, len, starts, lengths, cap, count, consumed, caps, nint.Zero)),
            SegmentKind.Words => ("sz_utf8_wordbreaks_best", Native.sz_utf8_wordbreaks_best(text, len, starts, lengths, cap, count, consumed, caps, nint.Zero)),
            SegmentKind.Sentences => ("sz_utf8_sentences_best", Native.sz_utf8_sentences_best(text, len, starts, lengths, cap, count, consumed, caps, nint.Zero)),
            SegmentKind.LineBreaks => ("sz_utf8_linebreaks_best", Native.sz_utf8_linebreaks_best(text, len, starts, lengths, cap, count, consumed, caps, nint.Zero)),
            SegmentKind.Newlines => ("sz_utf8_newlines_best", Native.sz_utf8_newlines_best(text, len, starts, lengths, cap, count, consumed, caps, nint.Zero)),
            SegmentKind.Whitespaces => ("sz_utf8_whitespaces_best", Native.sz_utf8_whitespaces_best(text, len, starts, lengths, cap, count, consumed, caps, nint.Zero)),
            SegmentKind.Delimiters => ("sz_utf8_delimiters_best", Native.sz_utf8_delimiters_best(text, len, starts, lengths, cap, count, consumed, caps, nint.Zero)),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
        Check(function, status);
    }

    #endregion

    #region UTF-8 Case Folding and Uncased Matching

    /// <summary>Folds into the caller's <paramref name="destination"/> (must hold ≥ text.Length*3
    /// bytes) and returns the bytes written. Allocation-free.</summary>
    public static int CaseFold(ReadOnlySpan<byte> text, Span<byte> destination) {
        if (text.Length == 0) return 0;
        if (destination.Length < (long)text.Length * 3)
            throw new ArgumentException("destination must hold at least text.Length*3 bytes", nameof(destination));
        nuint written;
        fixed (byte* s = text)
        fixed (byte* d = destination)
            Check("sz_utf8_uncased_fold_best", Native.sz_utf8_uncased_fold_best(
                (nint)s, (nuint)text.Length, (nint)d, &written, CpuEnabled, nint.Zero));
        return (int)written;
    }

    /// <summary>Full Unicode case folding (UAX #21, one-to-many, e.g. ß → "ss"). Returns folded UTF-8.</summary>
    /// <remarks>Fills a gap: .NET has no public case-folding API (only culture-sensitive
    /// ToLower/ToUpper). Use for caseless matching/indexing. Keywords: case fold, caseless, casefold.</remarks>
    public static byte[] CaseFold(ReadOnlySpan<byte> text) {
        if (text.Length == 0) return Array.Empty<byte>();
        byte[] rent = ArrayPool<byte>.Shared.Rent(checked(text.Length * 3)); // worst-case 3x expansion
        try {
            fixed (byte* s = text)
            fixed (byte* d = rent) {
                nuint n;
                Check("sz_utf8_uncased_fold_best", Native.sz_utf8_uncased_fold_best(
                    (nint)s, (nuint)text.Length, (nint)d, &n, CpuEnabled, nint.Zero));
                return rent.AsSpan(0, (int)n).ToArray();
            }
        }
        finally { ArrayPool<byte>.Shared.Return(rent); }
    }

    /// <summary>Case-insensitive (full-fold) search: byte offset of the first match, or -1; sets
    /// <paramref name="matchedLength"/> (may differ from needle length due to folding).</summary>
    /// <remarks>Fills a gap: no .NET caseless UTF-8 substring search. For repeated searches with the
    /// same needle, prefer <see cref="UncasedNeedle"/>.</remarks>
    public static long UncasedIndexOf(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle, out long matchedLength) {
        // The native search dereferences the needle-metadata pointer unconditionally, so we pass a
        // zeroed scratch buffer (it is populated on use) rather than NULL. sz_utf8_uncased_needle_metadata_t
        // is 40 bytes; 64 is a safe over-allocation. stackalloc<byte> is zero-initialized.
        Span<byte> meta = stackalloc byte[64];
        fixed (byte* m = meta)
        fixed (byte* h = haystack)
        fixed (byte* n = needle) {
            nuint ml;
            nint r;
            Check("sz_utf8_uncased_search_best", Native.sz_utf8_uncased_search_best(
                (nint)h, (nuint)haystack.Length, (nint)n, (nuint)needle.Length, (nint)m, &r, (nint)(&ml),
                CpuEnabled, nint.Zero));
            matchedLength = (long)ml;
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    /// <summary>Case-insensitive (full-fold) lexicographic comparison: -1, 0, or 1.</summary>
    /// <remarks>Fills a gap: no .NET Unicode case-folded ordinal comparison.</remarks>
    public static int UncasedCompare(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b) {
        int ordering;
        fixed (byte* pa = a)
        fixed (byte* pb = b)
            Check("sz_utf8_uncased_order_best", Native.sz_utf8_uncased_order_best(
                (nint)pa, (nuint)a.Length, (nint)pb, (nuint)b.Length, &ordering, CpuEnabled, nint.Zero));
        return ordering;
    }

    #endregion

    #region Hashing Extras

    /// <summary>Hashes <paramref name="data"/> under each seed, writing results into
    /// <paramref name="hashes"/> (must hold ≥ seeds.Length entries). Allocation-free.</summary>
    public static void HashMultiseed(ReadOnlySpan<byte> data, ReadOnlySpan<ulong> seeds, Span<ulong> hashes) {
        if (hashes.Length < seeds.Length)
            throw new ArgumentException("hashes must hold at least seeds.Length entries", nameof(hashes));
        fixed (byte* p = data)
        fixed (ulong* s = seeds)
        fixed (ulong* o = hashes)
            Check("sz_hash_multiseed_best", Native.sz_hash_multiseed_best(
                (nint)p, (nuint)data.Length, (nint)s, (nuint)seeds.Length, (nint)o, CpuEnabled, nint.Zero));
    }

    /// <summary>Allocating convenience: returns a fresh array of one hash per seed.</summary>
    public static ulong[] HashMultiseed(ReadOnlySpan<byte> data, ReadOnlySpan<ulong> seeds) {
        var result = new ulong[seeds.Length];
        HashMultiseed(data, seeds, result);
        return result;
    }

    #endregion

    #region Memory

    /// <summary>Fills <paramref name="buffer"/> with deterministic pseudo-random bytes derived from
    /// <paramref name="nonce"/> (AES-CTR-style; reproducible for the same nonce).</summary>
    public static void FillRandom(Span<byte> buffer, ulong nonce) {
        fixed (byte* p = buffer)
            Check("sz_fill_random_best", Native.sz_fill_random_best((nint)p, (nuint)buffer.Length, nonce, CpuEnabled, nint.Zero));
    }

    /// <summary>Byte-wise table transform: destination[i] = lut[source[i]]. <paramref name="lut"/>
    /// must be exactly 256 bytes.</summary>
    public static void Lookup(Span<byte> destination, ReadOnlySpan<byte> source, ReadOnlySpan<byte> lut) {
        if (lut.Length != 256) throw new ArgumentException("LUT must be 256 bytes", nameof(lut));
        if (destination.Length < source.Length) throw new ArgumentException("destination too small", nameof(destination));
        fixed (byte* d = destination)
        fixed (byte* s = source)
        fixed (byte* l = lut)
            Check("sz_lookup_best", Native.sz_lookup_best(
                (nint)d, (nint)s, (nuint)source.Length, (nint)l, CpuEnabled, nint.Zero));
    }

    #endregion

    #region UTF-8 Normalization

    public enum NormalForm { Nfd = 0, Nfc = 1, Nfkd = 2, Nfkc = 3 }

    /// <summary>Normalizes into the caller's <paramref name="destination"/> (must hold ≥ text.Length*18
    /// bytes) and returns the bytes written. Allocation-free.</summary>
    public static int Normalize(ReadOnlySpan<byte> text, NormalForm form, Span<byte> destination) {
        if (text.Length == 0) return 0;
        if (destination.Length < (long)text.Length * 18)
            throw new ArgumentException("destination must hold at least text.Length*18 bytes", nameof(destination));
        nuint written;
        fixed (byte* s = text)
        fixed (byte* d = destination)
            Check("sz_utf8_norm_best", Native.sz_utf8_norm_best(
                (nint)s, (nuint)text.Length, (int)form, (nint)d, &written, CpuEnabled, nint.Zero));
        return (int)written;
    }

    /// <summary>Normalizes UTF-8 <paramref name="text"/> to the given form. Returns normalized UTF-8.</summary>
    /// <remarks>Equivalent to <see cref="string.Normalize(System.Text.NormalizationForm)"/>, but over
    /// UTF-8 bytes with no UTF-16 transcode.</remarks>
    /// <example>Composes "e"+U+0301 into a precomposed "é" under <see cref="NormalForm.Nfc"/> (and back under
    /// <see cref="NormalForm.Nfd"/>); expands the ligature "ﬁ" into "fi" under <see cref="NormalForm.Nfkc"/>.</example>
    public static byte[] Normalize(ReadOnlySpan<byte> text, NormalForm form) {
        if (text.Length == 0) return Array.Empty<byte>();
        byte[] rent = ArrayPool<byte>.Shared.Rent(checked(text.Length * 18)); // worst-case per-codepoint expansion
        try {
            fixed (byte* s = text)
            fixed (byte* d = rent) {
                nuint n;
                Check("sz_utf8_norm_best", Native.sz_utf8_norm_best(
                    (nint)s, (nuint)text.Length, (int)form, (nint)d, &n, CpuEnabled, nint.Zero));
                return rent.AsSpan(0, (int)n).ToArray();
            }
        }
        finally { ArrayPool<byte>.Shared.Return(rent); }
    }

    /// <summary>True if <paramref name="text"/> is already in the given normalization form.</summary>
    /// <remarks>Equivalent to <see cref="string.IsNormalized(System.Text.NormalizationForm)"/>.</remarks>
    public static bool IsNormalized(ReadOnlySpan<byte> text, NormalForm form) {
        nint violation;
        fixed (byte* s = text)
            Check("sz_utf8_find_denormalized_best", Native.sz_utf8_find_denormalized_best(
                (nint)s, (nuint)text.Length, (int)form, &violation, CpuEnabled, nint.Zero));
        return violation == 0;
    }

    #endregion

    #region Collections

    /// <summary>Writes the sorting permutation of <paramref name="items"/> into <paramref name="order"/>
    /// (must hold ≥ items.Count entries) and returns the count of sorted indices
    /// (<paramref name="top"/> if &gt; 0, else items.Count). Allocation-free for the result.</summary>
    /// <remarks>Like <see cref="System.Array.Sort(System.Array)"/> with
    /// <see cref="System.StringComparer.Ordinal"/>, but returns indices and runs over UTF-8 bytes.</remarks>
    public static int ArgSort(IReadOnlyList<byte[]> items, Span<long> order, bool reverse = false, long top = 0, bool uncased = false) {
        int n = items.Count;
        if (n == 0) return 0;
        if (order.Length < n) throw new ArgumentException($"order must hold at least {n} entries", nameof(order));
        long total = 0;
        foreach (var it in items) total += it.Length;
        byte* data = (byte*)NativeMemory.Alloc((nuint)Math.Max(total, 1));
        nuint* starts = (nuint*)NativeMemory.Alloc((nuint)n, (nuint)sizeof(nuint));
        nuint* lengths = (nuint*)NativeMemory.Alloc((nuint)n, (nuint)sizeof(nuint));
        try {
            FillTable(items, data, starts, lengths);
            SeqContext ctx; ctx.Data = data; ctx.Starts = starts; ctx.Lengths = lengths;
            SzSequence seq;
            seq.Handle = &ctx;
            seq.Count = (nuint)n;
            seq.GetStart = &SeqCallbacks.GetStart;
            seq.GetLength = &SeqCallbacks.GetLength;
            fixed (long* ord = order)
                ArgSortSequence(&seq, ord, top, reverse, uncased);
            return top > 0 ? (int)Math.Min(top, n) : n;
        }
        finally {
            NativeMemory.Free(data); NativeMemory.Free(starts); NativeMemory.Free(lengths);
        }
    }

    /// <summary>Allocating convenience: returns a fresh permutation array.</summary>
    public static long[] ArgSort(IReadOnlyList<byte[]> items, bool reverse = false, long top = 0, bool uncased = false) {
        int n = items.Count;
        if (n == 0) return Array.Empty<long>();
        var order = new long[n];
        int count = ArgSort(items, order, reverse, top, uncased);
        return count == n ? order : order.AsSpan(0, count).ToArray();
    }

    /// <summary>Sorts the segments of a single buffer in place — given per-segment <paramref name="starts"/>
    /// and <paramref name="lengths"/> (as produced by the split iterators), writes their sorting permutation
    /// into <paramref name="order"/> and returns the count. Zero-copy: no segment bytes are materialised.</summary>
    /// <remarks>Sort the lines of a file without building a <c>byte[][]</c>: collect ranges from a
    /// <see cref="SplitEnumerator"/>, then sort them here.</remarks>
    public static int ArgSort(
        ReadOnlySpan<byte> text,
        ReadOnlySpan<long> starts,
        ReadOnlySpan<long> lengths,
        Span<long> order,
        bool reverse = false,
        long top = 0,
        bool uncased = false) {
        int n = starts.Length;
        if (n == 0) return 0;
        if (lengths.Length < n) throw new ArgumentException($"lengths must hold at least {n} entries", nameof(lengths));
        if (order.Length < n) throw new ArgumentException($"order must hold at least {n} entries", nameof(order));
        for (int i = 0; i < n; i++)
            if (starts[i] < 0 || lengths[i] < 0 || starts[i] > text.Length - lengths[i])
                throw new ArgumentOutOfRangeException(nameof(starts), $"segment {i} falls outside the text");
        fixed (byte* basePtr = text)
        fixed (long* segmentStarts = starts)
        fixed (long* segmentLengths = lengths)
        fixed (long* ord = order) {
            SeqRangeContext ctx;
            ctx.Base = basePtr;
            ctx.Starts = segmentStarts;
            ctx.Lengths = segmentLengths;
            SzSequence seq;
            seq.Handle = &ctx;
            seq.Count = (nuint)n;
            seq.GetStart = &SeqCallbacks.GetRangeStart;
            seq.GetLength = &SeqCallbacks.GetRangeLength;
            ArgSortSequence(&seq, ord, top, reverse, uncased);
        }
        return top > 0 ? (int)Math.Min(top, n) : n;
    }

    private static void ArgSortSequence(SzSequence* seq, long* order, long top, bool reverse, bool uncased) {
        if (uncased)
            Check("sz_sequence_argsort_uncased_best", Native.sz_sequence_argsort_uncased_best(
                (nint)seq, (nuint)top, reverse ? 1 : 0, nint.Zero, (nint)order, CpuEnabled, nint.Zero));
        else
            Check("sz_sequence_argsort_best", Native.sz_sequence_argsort_best(
                (nint)seq, (nuint)top, reverse ? 1 : 0, nint.Zero, (nint)order, CpuEnabled, nint.Zero));
    }

    /// <summary>Inner-joins two de-duplicated sequences, writing matching index pairs into
    /// <paramref name="firstPositions"/> / <paramref name="secondPositions"/> (each must hold ≥
    /// min(a.Count, b.Count) entries) and returns the match count. Allocation-free for the result.</summary>
    public static int Intersect(IReadOnlyList<byte[]> a, IReadOnlyList<byte[]> b, Span<long> firstPositions, Span<long> secondPositions, ulong seed = 0) {
        int na = a.Count, nb = b.Count;
        if (na == 0 || nb == 0) return 0;
        int minN = Math.Min(na, nb);
        if (firstPositions.Length < minN || secondPositions.Length < minN)
            throw new ArgumentException($"position spans must hold at least {minN} entries");
        long ta = 0; foreach (var it in a) ta += it.Length;
        long tb = 0; foreach (var it in b) tb += it.Length;
        byte* dataA = (byte*)NativeMemory.Alloc((nuint)Math.Max(ta, 1));
        nuint* startsA = (nuint*)NativeMemory.Alloc((nuint)na, (nuint)sizeof(nuint));
        nuint* lengthsA = (nuint*)NativeMemory.Alloc((nuint)na, (nuint)sizeof(nuint));
        byte* dataB = (byte*)NativeMemory.Alloc((nuint)Math.Max(tb, 1));
        nuint* startsB = (nuint*)NativeMemory.Alloc((nuint)nb, (nuint)sizeof(nuint));
        nuint* lengthsB = (nuint*)NativeMemory.Alloc((nuint)nb, (nuint)sizeof(nuint));
        try {
            FillTable(a, dataA, startsA, lengthsA);
            FillTable(b, dataB, startsB, lengthsB);
            SeqContext ctxA; ctxA.Data = dataA; ctxA.Starts = startsA; ctxA.Lengths = lengthsA;
            SeqContext ctxB; ctxB.Data = dataB; ctxB.Starts = startsB; ctxB.Lengths = lengthsB;
            SzSequence seqA;
            seqA.Handle = &ctxA; seqA.Count = (nuint)na; seqA.GetStart = &SeqCallbacks.GetStart; seqA.GetLength = &SeqCallbacks.GetLength;
            SzSequence seqB;
            seqB.Handle = &ctxB; seqB.Count = (nuint)nb; seqB.GetStart = &SeqCallbacks.GetStart; seqB.GetLength = &SeqCallbacks.GetLength;
            nuint size;
            fixed (long* fp = firstPositions)
            fixed (long* sp = secondPositions)
                Check("sz_sequence_intersect_best", Native.sz_sequence_intersect_best(
                    (nint)(&seqA), (nint)(&seqB), nint.Zero, seed, (nint)(&size), (nint)fp, (nint)sp,
                    CpuEnabled, nint.Zero));
            return (int)size;
        }
        finally {
            NativeMemory.Free(dataA); NativeMemory.Free(startsA); NativeMemory.Free(lengthsA);
            NativeMemory.Free(dataB); NativeMemory.Free(startsB); NativeMemory.Free(lengthsB);
        }
    }

    private static void FillTable(IReadOnlyList<byte[]> items, byte* data, nuint* starts, nuint* lengths) {
        long off = 0;
        for (int i = 0; i < items.Count; i++) {
            byte[] it = items[i];
            starts[i] = (nuint)off;
            lengths[i] = (nuint)it.Length;
            if (it.Length > 0)
                fixed (byte* src = it) Buffer.MemoryCopy(src, data + off, it.Length, it.Length);
            off += it.Length;
        }
    }

    #endregion

    #region Iteration

    /// <summary>Lazily enumerates the Unicode codepoints of <paramref name="text"/> as
    /// <see cref="System.Text.Rune"/>s (ill-formed sequences yield U+FFFD).</summary>
    /// <remarks>Like <see cref="string.EnumerateRunes"/>, over UTF-8 bytes; batches 64 per native call.
    /// Allocation-free; synchronous <c>foreach</c> only (the enumerator is a ref struct).</remarks>
    public static RuneEnumerable EnumerateRunes(ReadOnlySpan<byte> text) => new(text);

    /// <summary>Lazily enumerates UAX-29 word segments as byte-slice views (tiling; no empty segments).</summary>
    /// <remarks>Like iterating <see cref="System.Globalization.StringInfo"/> word elements, but over UTF-8 bytes.</remarks>
    public static SegmentEnumerable EnumerateWords(ReadOnlySpan<byte> text) => new(text, SegmentKind.Words);

    /// <summary>Lazily enumerates UAX-29 grapheme clusters as byte-slice views (tiling).</summary>
    /// <remarks>Like <see cref="System.Globalization.StringInfo.GetTextElementEnumerator(string)"/>, over UTF-8 bytes.</remarks>
    public static SegmentEnumerable EnumerateGraphemes(ReadOnlySpan<byte> text) => new(text, SegmentKind.Graphemes);

    /// <summary>Lazily enumerates UAX-29 sentences as byte-slice views (tiling).</summary>
    public static SegmentEnumerable EnumerateSentences(ReadOnlySpan<byte> text) => new(text, SegmentKind.Sentences);

    /// <summary>Lazily enumerates UAX-14 line-break units as byte-slice views (tiling).</summary>
    public static SegmentEnumerable EnumerateLineBreaks(ReadOnlySpan<byte> text) => new(text, SegmentKind.LineBreaks);

    #endregion

    #region Splitting

    /// <summary>Lazily splits <paramref name="text"/> on each occurrence of <paramref name="separator"/>,
    /// yielding the segments between separators as byte-slice views. Empty segments are kept by default
    /// (like <see cref="string.Split(string[],System.StringSplitOptions)"/>); chain <c>.SkipEmpty()</c>,
    /// <c>.KeepSeparator()</c>, or <c>.WithMaxSplit(n)</c> to change behaviour.</summary>
    public static SplitEnumerable Split(ReadOnlySpan<byte> text, ReadOnlySpan<byte> separator) =>
        SplitEnumerable.Substring(text, separator, reverse: false);

    /// <summary>As <see cref="Split"/> but scanning from the end; yields the last segment first.</summary>
    public static SplitEnumerable RSplit(ReadOnlySpan<byte> text, ReadOnlySpan<byte> separator) =>
        SplitEnumerable.Substring(text, separator, reverse: true);

    /// <summary>Lazily splits on any byte in <paramref name="separators"/> (like splitting on a character set).</summary>
    public static SplitEnumerable SplitAny(ReadOnlySpan<byte> text, Byteset separators) =>
        SplitEnumerable.OfByteset(text, separators, reverse: false);

    /// <summary>As <see cref="SplitAny"/> but scanning from the end.</summary>
    public static SplitEnumerable RSplitAny(ReadOnlySpan<byte> text, Byteset separators) =>
        SplitEnumerable.OfByteset(text, separators, reverse: true);

    /// <summary>Lazily enumerates the byte offsets of every (non-overlapping) match of
    /// <paramref name="needle"/> in <paramref name="haystack"/>; chain <c>.Overlapping()</c> for overlaps.
    /// Counting the result is the occurrence count.</summary>
    public static MatchEnumerable EnumerateMatches(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle) =>
        new(haystack, needle, overlapping: false);

    /// <summary>Lazily splits on Unicode newline runs, yielding the lines between them. Chain
    /// <c>.WithSeparators()</c> for a lossless interleave or <c>.SkipEmpty()</c> to drop blanks.</summary>
    public static TokenSplitEnumerable SplitNewlines(ReadOnlySpan<byte> text) =>
        new(text, SegmentKind.Newlines, SplitParts.Between, skipEmpty: false);

    /// <summary>Lazily splits on Unicode whitespace runs, yielding the fields between them.</summary>
    public static TokenSplitEnumerable SplitWhitespaces(ReadOnlySpan<byte> text) =>
        new(text, SegmentKind.Whitespaces, SplitParts.Between, skipEmpty: false);

    /// <summary>Lazily splits on Unicode delimiter runs (punctuation/symbols/separators).</summary>
    public static TokenSplitEnumerable SplitDelimiters(ReadOnlySpan<byte> text) =>
        new(text, SegmentKind.Delimiters, SplitParts.Between, skipEmpty: false);

    /// <summary>Lazily enumerates case-insensitive (full-fold) matches of <paramref name="needle"/>,
    /// yielding each match's offset and matched length (which may differ from the needle length due to
    /// folding, e.g. "ß" matching "SS"). Chain <c>.Overlapping()</c> for overlapping matches.</summary>
    public static UncasedMatchEnumerable EnumerateUncasedMatches(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle) =>
        new(haystack, needle, overlapping: false);

    /// <summary>Splits at the first occurrence of <paramref name="separator"/> into (before, separator,
    /// after) views; if absent, returns (<paramref name="text"/>, empty, empty).</summary>
    public static Partition Partition(ReadOnlySpan<byte> text, ReadOnlySpan<byte> separator) {
        long at = IndexOf(text, separator);
        if (at < 0) return new Partition(text, default, default);
        return new Partition(text.Slice(0, (int)at), text.Slice((int)at, separator.Length), text.Slice((int)at + separator.Length));
    }

    /// <summary>As <see cref="Partition"/> but at the last occurrence; if absent, returns
    /// (empty, empty, <paramref name="text"/>).</summary>
    public static Partition RPartition(ReadOnlySpan<byte> text, ReadOnlySpan<byte> separator) {
        long at = LastIndexOf(text, separator);
        if (at < 0) return new Partition(default, default, text);
        return new Partition(text.Slice(0, (int)at), text.Slice((int)at, separator.Length), text.Slice((int)at + separator.Length));
    }

    #endregion

    #region Capabilities and Library Metadata

    /// <summary>The CPU mask every call passes, which <see cref="Device.CapabilitiesEnable"/> narrows.</summary>
    internal static ulong CpuEnabled {
        get => Volatile.Read(ref _cpuEnabled);
        set => Volatile.Write(ref _cpuEnabled, value);
    }

    private static ulong _cpuEnabled = CpuAvailable();

    /// <summary>The capabilities this CPU runs and this binary holds kernels for, before any narrowing.</summary>
    internal static ulong CpuAvailable() {
        ulong capabilities = 1;
        Check("sz_cpu_capabilities_enabled", Native.sz_cpu_capabilities_enabled(&capabilities));
        return capabilities;
    }

    /// <summary>Names the capabilities in <paramref name="capabilities"/>, comma-separated, like "serial,haswell".</summary>
    public static string CapabilitiesName(ulong capabilities) {
        const int capacity = 256; // `STRINGZILLA_CAPABILITIES_NAME_CAPACITY`
        byte* names = stackalloc byte[capacity];
        nuint written = Native.sz_capabilities_name(capabilities, names, capacity);
        return Marshal.PtrToStringUTF8((nint)names, (int)written);
    }

    public static Version Version =>
        new(Native.sz_version_major(), Native.sz_version_minor(), Native.sz_version_patch());

    /// <summary>Throws a <see cref="StatusException"/> naming <paramref name="function"/> unless <paramref name="status"/> succeeded.</summary>
    internal static void Check(string function, int status) {
        if (status != 0) throw new StatusException(function, status);
    }

    #endregion
}

/// <summary>Which runtime a device belongs to, as the <c>sz_&lt;kind&gt;_*</c> C functions name it.</summary>
public enum DeviceKind { Cpu, Cuda, Rocm, Metal }

/// <summary>One device StringZilla can run kernels on: the host CPU, or a GPU by its runtime's own ordinal,
/// the one <c>cudaSetDevice</c> or <c>hipSetDevice</c> takes, or the position in Metal's device list.</summary>
/// <remarks>Prefer <see cref="CapabilitiesEnabled"/> unless you specifically mean one of the raw axes:
/// <see cref="CapabilitiesDetected"/> describes the device and says nothing about whether a kernel was compiled
/// into this binary, so selecting on it alone claims support for code that may not exist. Every call of this
/// binding dispatches over the CPU's enabled set; a GPU device only reports its masks here.</remarks>
public readonly unsafe record struct Device {
    private const int MissingGpu = -16; // `sz_missing_gpu_k`
    private const int MissingKernel = -20; // `sz_missing_kernel_k`

    /// <summary>The runtime this device belongs to.</summary>
    public DeviceKind Kind { get; }

    /// <summary>The runtime's own index of this device, below <see cref="Count"/>.</summary>
    public ulong Ordinal { get; }

    /// <summary>Device <paramref name="ordinal"/> of <paramref name="kind"/>.</summary>
    /// <exception cref="StatusException">Without a device of <paramref name="kind"/>, or past the last one.</exception>
    public Device(DeviceKind kind, ulong ordinal = 0) {
        if (ordinal >= Count(kind)) throw new StatusException($"Device({kind}, {ordinal})", MissingGpu);
        Kind = kind;
        Ordinal = ordinal;
    }

    /// <summary>The host CPU, which every build has.</summary>
    public static Device Cpu => default;

    /// <summary>How many devices of <paramref name="kind"/> the process sees: one CPU, or the GPUs its runtime counts.</summary>
    /// <exception cref="StatusException">Without a GPU of <paramref name="kind"/>.</exception>
    public static ulong Count(DeviceKind kind) {
        nuint count = 1;
        switch (kind) {
            case DeviceKind.Cpu: break;
            case DeviceKind.Cuda: Sz.Check("sz_cuda_count_devices", Native.sz_cuda_count_devices(&count)); break;
            case DeviceKind.Rocm: Sz.Check("sz_rocm_count_devices", Native.sz_rocm_count_devices(&count)); break;
            case DeviceKind.Metal: Sz.Check("sz_metal_count_devices", Native.sz_metal_count_devices(&count)); break;
            default: throw new ArgumentOutOfRangeException(nameof(kind));
        }
        return count;
    }

    /// <summary>What this device runs, whether or not this binary holds kernels for it.</summary>
    /// <exception cref="StatusException">When the runtime can't be queried.</exception>
    public ulong CapabilitiesDetected {
        get {
            ulong mask = 0;
            nuint device = (nuint)Ordinal;
            switch (Kind) {
                case DeviceKind.Cpu: Sz.Check("sz_cpu_capabilities_detected", Native.sz_cpu_capabilities_detected(&mask)); break;
                case DeviceKind.Cuda: Sz.Check("sz_cuda_capabilities_detected", Native.sz_cuda_capabilities_detected(device, &mask)); break;
                case DeviceKind.Rocm: Sz.Check("sz_rocm_capabilities_detected", Native.sz_rocm_capabilities_detected(device, &mask)); break;
                case DeviceKind.Metal: Sz.Check("sz_metal_capabilities_detected", Native.sz_metal_capabilities_detected(device, &mask)); break;
            }
            return mask;
        }
    }

    /// <summary>What this binary holds kernels for on devices of this kind, whether or not this one runs them.</summary>
    public ulong CapabilitiesCompiled {
        get {
            ulong mask = 0;
            switch (Kind) {
                case DeviceKind.Cpu: Sz.Check("sz_cpu_capabilities_compiled", Native.sz_cpu_capabilities_compiled(&mask)); break;
                case DeviceKind.Cuda: Sz.Check("sz_cuda_capabilities_compiled", Native.sz_cuda_capabilities_compiled(&mask)); break;
                case DeviceKind.Rocm: Sz.Check("sz_rocm_capabilities_compiled", Native.sz_rocm_capabilities_compiled(&mask)); break;
                case DeviceKind.Metal: Sz.Check("sz_metal_capabilities_compiled", Native.sz_metal_capabilities_compiled(&mask)); break;
            }
            return mask;
        }
    }

    /// <summary>What this device's calls pass: <see cref="CapabilitiesDetected"/> and <see cref="CapabilitiesCompiled"/>
    /// at once. On the CPU it is what every call of this binding passes, narrowed by <see cref="CapabilitiesEnable"/>,
    /// and always contains the serial fallback, bit 0.</summary>
    /// <exception cref="StatusException">When the runtime can't be queried.</exception>
    public ulong CapabilitiesEnabled {
        get {
            ulong mask = 0;
            nuint device = (nuint)Ordinal;
            switch (Kind) {
                case DeviceKind.Cpu: return Sz.CpuEnabled;
                case DeviceKind.Cuda: Sz.Check("sz_cuda_capabilities_enabled", Native.sz_cuda_capabilities_enabled(device, &mask)); break;
                case DeviceKind.Rocm: Sz.Check("sz_rocm_capabilities_enabled", Native.sz_rocm_capabilities_enabled(device, &mask)); break;
                case DeviceKind.Metal: Sz.Check("sz_metal_capabilities_enabled", Native.sz_metal_capabilities_enabled(device, &mask)); break;
            }
            return mask;
        }
    }

    /// <summary>Makes <paramref name="wanted"/> the CPU's <see cref="CapabilitiesEnabled"/> set, clamped to what it
    /// detects and this binary compiled and keeping the serial fallback.</summary>
    /// <returns>The enabled set that took effect.</returns>
    /// <exception cref="StatusException">On a GPU, which keeps no such set.</exception>
    /// <example><c>Device.Cpu.CapabilitiesEnable(1)</c> runs every later call on the serial kernels.</example>
    public ulong CapabilitiesEnable(ulong wanted) {
        if (Kind != DeviceKind.Cpu) throw new StatusException($"CapabilitiesEnable on {this}", MissingKernel);
        ulong enabled = (wanted & Sz.CpuAvailable()) | 1;
        Sz.CpuEnabled = enabled;
        return enabled;
    }

    /// <summary>Prepares the calling thread for the kernels of <paramref name="capabilities"/>, usually
    /// <see cref="CapabilitiesEnabled"/>; call it once on every thread that runs them.</summary>
    /// <exception cref="StatusException">On a GPU, which has no thread state to configure.</exception>
    public void ConfigureThread(ulong capabilities) {
        if (Kind != DeviceKind.Cpu) throw new StatusException($"ConfigureThread on {this}", MissingKernel);
        Sz.Check("sz_cpu_configure_thread", Native.sz_cpu_configure_thread(capabilities));
    }
}

/// <summary>Stack-only batch buffer of 64 longs for the segmentation/split enumerators.</summary>
[System.Runtime.CompilerServices.InlineArray(64)]
internal struct LongBuf64 {
    private long _element0;
}

/// <summary>Stack-only batch buffer of 64 Int32 codepoints for the rune enumerator.</summary>
[System.Runtime.CompilerServices.InlineArray(64)]
internal struct IntBuf64 {
    private int _element0;
}

/// <summary>Allocation-free <c>foreach</c> source over UTF-8 codepoints. See <see cref="Sz.EnumerateRunes"/>.</summary>
public readonly ref struct RuneEnumerable {
    private readonly ReadOnlySpan<byte> _text;

    public RuneEnumerable(ReadOnlySpan<byte> text) => _text = text;

    public RuneEnumerator GetEnumerator() => new(_text);
}

/// <summary>Ref-struct enumerator yielding <see cref="Rune"/>s, batched 64 per native call.</summary>
public ref struct RuneEnumerator {
    private readonly ReadOnlySpan<byte> _text;
    private long _cursor;
    private IntBuf64 _buffer;
    private int _count;
    private int _index;

    internal RuneEnumerator(ReadOnlySpan<byte> text) {
        _text = text;
        _cursor = 0;
        _count = 0;
        _index = 0;
        _buffer = default;
        Current = default;
    }

    public Rune Current { get; private set; }

    public bool MoveNext() {
        while (true) {
            if (_index < _count) {
                Current = new Rune(_buffer[_index]);
                _index++;
                return true;
            }
            if (_cursor >= _text.Length) return false;
            _count = Sz.Decode(_text[(int)_cursor..], _buffer[..], out long consumed);
            if (_count == 0 || consumed == 0) return false;
            _index = 0;
            _cursor += consumed;
        }
    }
}

/// <summary>Allocation-free <c>foreach</c> source over UTF-8 segments. See <see cref="Sz.EnumerateWords"/>.</summary>
public readonly ref struct SegmentEnumerable {
    private readonly ReadOnlySpan<byte> _text;
    private readonly Sz.SegmentKind _kind;

    public SegmentEnumerable(ReadOnlySpan<byte> text, Sz.SegmentKind kind) {
        _text = text;
        _kind = kind;
    }

    public SegmentEnumerator GetEnumerator() => new(_text, _kind);
}

/// <summary>Ref-struct enumerator yielding byte-slice views of segments, batched 64 per native call.</summary>
public ref struct SegmentEnumerator {
    private readonly ReadOnlySpan<byte> _text;
    private readonly Sz.SegmentKind _kind;
    private long _cursor;
    private long _chunkBase;
    private LongBuf64 _starts;
    private LongBuf64 _lengths;
    private int _count;
    private int _index;

    internal SegmentEnumerator(ReadOnlySpan<byte> text, Sz.SegmentKind kind) {
        _text = text;
        _kind = kind;
        _cursor = 0;
        _chunkBase = 0;
        _count = 0;
        _index = 0;
        _starts = default;
        _lengths = default;
        Current = default;
    }

    public ReadOnlySpan<byte> Current { get; private set; }

    public bool MoveNext() {
        while (true) {
            if (_index < _count) {
                Current = _text.Slice((int)(_chunkBase + _starts[_index]), (int)_lengths[_index]);
                _index++;
                return true;
            }
            if (_cursor >= _text.Length) return false;
            _chunkBase = _cursor;
            _count = Sz.Segment(_text[(int)_cursor..], _kind, _starts[..], _lengths[..], out long consumed);
            if (_count == 0 || consumed == 0) return false;
            _index = 0;
            _cursor += consumed;
        }
    }
}

/// <summary>Allocation-free <c>foreach</c> source over split segments. See <see cref="Sz.Split"/>.</summary>
public readonly ref struct SplitEnumerable {
    private readonly ReadOnlySpan<byte> _text;
    private readonly ReadOnlySpan<byte> _separator;
    private readonly Byteset _byteset;
    private readonly bool _isByteset;
    private readonly bool _reverse;
    private readonly bool _keepSeparator;
    private readonly bool _skipEmpty;
    private readonly long _maxSplit;

    private SplitEnumerable(ReadOnlySpan<byte> text, ReadOnlySpan<byte> separator, Byteset byteset,
        bool isByteset, bool reverse, bool keepSeparator, bool skipEmpty, long maxSplit) {
        _text = text;
        _separator = separator;
        _byteset = byteset;
        _isByteset = isByteset;
        _reverse = reverse;
        _keepSeparator = keepSeparator;
        _skipEmpty = skipEmpty;
        _maxSplit = maxSplit;
    }

    internal static SplitEnumerable Substring(ReadOnlySpan<byte> text, ReadOnlySpan<byte> separator, bool reverse) =>
        new(text, separator, default, false, reverse, false, false, long.MaxValue);

    internal static SplitEnumerable OfByteset(ReadOnlySpan<byte> text, Byteset set, bool reverse) =>
        new(text, default, set, true, reverse, false, false, long.MaxValue);

    /// <summary>Drop zero-length segments.</summary>
    public SplitEnumerable SkipEmpty() =>
        new(_text, _separator, _byteset, _isByteset, _reverse, _keepSeparator, true, _maxSplit);

    /// <summary>Keep the separator attached to each segment: the trailing separator when splitting forward,
    /// the leading separator when splitting in reverse.</summary>
    public SplitEnumerable KeepSeparator() =>
        new(_text, _separator, _byteset, _isByteset, _reverse, true, _skipEmpty, _maxSplit);

    /// <summary>Stop after <paramref name="maxSplit"/> splits; the rest becomes one final segment.</summary>
    public SplitEnumerable WithMaxSplit(long maxSplit) =>
        new(_text, _separator, _byteset, _isByteset, _reverse, _keepSeparator, _skipEmpty, maxSplit);

    public SplitEnumerator GetEnumerator() =>
        new(_text, _separator, _byteset, _isByteset, _reverse, _keepSeparator, _skipEmpty, _maxSplit);
}

/// <summary>Ref-struct enumerator yielding split segments as byte-slice views.</summary>
public ref struct SplitEnumerator {
    private readonly ReadOnlySpan<byte> _text;
    private readonly ReadOnlySpan<byte> _separator;
    private Byteset _byteset;
    private readonly bool _isByteset;
    private readonly bool _reverse;
    private readonly bool _keepSeparator;
    private readonly bool _skipEmpty;
    private long _remainingSplits;
    private long _cursor; // forward: start of remaining text; reverse: exclusive end of remaining text
    private bool _done;

    internal SplitEnumerator(ReadOnlySpan<byte> text, ReadOnlySpan<byte> separator, Byteset byteset,
        bool isByteset, bool reverse, bool keepSeparator, bool skipEmpty, long maxSplit) {
        _text = text;
        _separator = separator;
        _byteset = byteset;
        _isByteset = isByteset;
        _reverse = reverse;
        _keepSeparator = keepSeparator;
        _skipEmpty = skipEmpty;
        _remainingSplits = maxSplit;
        _cursor = reverse ? text.Length : 0;
        _done = false;
        Current = default;
    }

    public ReadOnlySpan<byte> Current { get; private set; }

    public bool MoveNext() => _reverse ? MoveNextReverse() : MoveNextForward();

    private bool MoveNextForward() {
        while (true) {
            if (_done) return false;
            ReadOnlySpan<byte> remaining = _text.Slice((int)_cursor);
            long found = (_remainingSplits <= 0 || (!_isByteset && _separator.Length == 0))
                ? -1
                : _isByteset ? Sz.IndexOfAny(remaining, ref _byteset) : Sz.IndexOf(remaining, _separator);
            if (found < 0) {
                _done = true;
                if (_skipEmpty && remaining.Length == 0) return false;
                Current = remaining;
                return true;
            }
            int separatorLength = _isByteset ? 1 : _separator.Length;
            ReadOnlySpan<byte> segment = remaining.Slice(0, (int)found + (_keepSeparator ? separatorLength : 0));
            _cursor += found + separatorLength;
            _remainingSplits--;
            if (_skipEmpty && segment.Length == 0) continue;
            Current = segment;
            return true;
        }
    }

    private bool MoveNextReverse() {
        while (true) {
            if (_done) return false;
            ReadOnlySpan<byte> remaining = _text.Slice(0, (int)_cursor);
            long found = (_remainingSplits <= 0 || (!_isByteset && _separator.Length == 0))
                ? -1
                : _isByteset ? Sz.LastIndexOfAny(remaining, ref _byteset) : Sz.LastIndexOf(remaining, _separator);
            if (found < 0) {
                _done = true;
                if (_skipEmpty && remaining.Length == 0) return false;
                Current = remaining;
                return true;
            }
            int separatorLength = _isByteset ? 1 : _separator.Length;
            int segmentStart = _keepSeparator ? (int)found : (int)found + separatorLength;
            Current = _text.Slice(segmentStart, (int)_cursor - segmentStart);
            _cursor = found;
            _remainingSplits--;
            if (_skipEmpty && Current.Length == 0) continue;
            return true;
        }
    }

    public SplitEnumerator GetEnumerator() => this;
}

/// <summary>Allocation-free <c>foreach</c> source over match offsets. See <see cref="Sz.EnumerateMatches"/>.</summary>
public readonly ref struct MatchEnumerable {
    private readonly ReadOnlySpan<byte> _haystack;
    private readonly ReadOnlySpan<byte> _needle;
    private readonly bool _overlapping;

    internal MatchEnumerable(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle, bool overlapping) {
        _haystack = haystack;
        _needle = needle;
        _overlapping = overlapping;
    }

    /// <summary>Report overlapping matches (advance by one byte instead of the needle length).</summary>
    public MatchEnumerable Overlapping() => new(_haystack, _needle, true);

    public MatchEnumerator GetEnumerator() => new(_haystack, _needle, _overlapping);
}

/// <summary>Ref-struct enumerator yielding the byte offset of each match.</summary>
public ref struct MatchEnumerator {
    private readonly ReadOnlySpan<byte> _haystack;
    private readonly ReadOnlySpan<byte> _needle;
    private readonly bool _overlapping;
    private long _cursor;

    internal MatchEnumerator(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle, bool overlapping) {
        _haystack = haystack;
        _needle = needle;
        _overlapping = overlapping;
        _cursor = 0;
        Current = -1;
    }

    public long Current { get; private set; }

    public bool MoveNext() {
        if (_needle.Length == 0 || _cursor > _haystack.Length) return false;
        ReadOnlySpan<byte> remaining = _haystack.Slice((int)_cursor);
        long found = Sz.IndexOf(remaining, _needle);
        if (found < 0) return false;
        Current = _cursor + found;
        _cursor = Current + (_overlapping ? 1 : _needle.Length);
        return true;
    }

    public MatchEnumerator GetEnumerator() => this;
}

/// <summary>Which spans a separator-token split yields: the gaps, the separator runs, or both (lossless).</summary>
public enum SplitParts {
    Between,
    Separators,
    Both,
}

/// <summary>Allocation-free <c>foreach</c> source over separator-token splits. See <see cref="Sz.SplitWhitespaces"/>.</summary>
public readonly ref struct TokenSplitEnumerable {
    private readonly ReadOnlySpan<byte> _text;
    private readonly Sz.SegmentKind _kind;
    private readonly SplitParts _parts;
    private readonly bool _skipEmpty;

    internal TokenSplitEnumerable(ReadOnlySpan<byte> text, Sz.SegmentKind kind, SplitParts parts, bool skipEmpty) {
        _text = text;
        _kind = kind;
        _parts = parts;
        _skipEmpty = skipEmpty;
    }

    /// <summary>Also yield the separator runs, interleaved with the gaps (lossless reconstruction).</summary>
    public TokenSplitEnumerable WithSeparators() => new(_text, _kind, SplitParts.Both, _skipEmpty);

    /// <summary>Yield only the separator runs.</summary>
    public TokenSplitEnumerable OnlySeparators() => new(_text, _kind, SplitParts.Separators, _skipEmpty);

    /// <summary>Drop zero-length segments.</summary>
    public TokenSplitEnumerable SkipEmpty() => new(_text, _kind, _parts, true);

    public TokenSplitEnumerator GetEnumerator() => new(_text, _kind, _parts, _skipEmpty);
}

/// <summary>Ref-struct enumerator yielding separator-token split spans, batched 64 separators per call.</summary>
public ref struct TokenSplitEnumerator {
    private readonly ReadOnlySpan<byte> _text;
    private readonly Sz.SegmentKind _kind;
    private readonly SplitParts _parts;
    private readonly bool _skipEmpty;
    private LongBuf64 _starts;
    private LongBuf64 _lengths;
    private int _count;
    private int _index;
    private long _scanCursor;
    private long _chunkBase;
    private long _prevEnd; // absolute end of the previous separator run (start of the pending gap)
    private bool _atSeparator; // sticky only when a gap was emitted: resume by emitting the separator run
    private bool _exhausted;
    private bool _emittedTail;

    internal TokenSplitEnumerator(ReadOnlySpan<byte> text, Sz.SegmentKind kind, SplitParts parts, bool skipEmpty) {
        _text = text;
        _kind = kind;
        _parts = parts;
        _skipEmpty = skipEmpty;
        _starts = default;
        _lengths = default;
        _count = 0;
        _index = 0;
        _scanCursor = 0;
        _chunkBase = 0;
        _prevEnd = 0;
        _atSeparator = false;
        _exhausted = false;
        _emittedTail = false;
        Current = default;
    }

    public ReadOnlySpan<byte> Current { get; private set; }

    public bool MoveNext() {
        while (true) {
            if (!_exhausted && _index >= _count) {
                if (_scanCursor >= _text.Length) {
                    _exhausted = true;
                }
                else {
                    _chunkBase = _scanCursor;
                    _count = Sz.Segment(_text.Slice((int)_scanCursor), _kind, _starts[..], _lengths[..], out long consumed);
                    if (_count == 0 || consumed == 0)
                        _exhausted = true;
                    else {
                        _index = 0;
                        _scanCursor += consumed;
                    }
                }
            }
            if (_exhausted) {
                if (_emittedTail) return false;
                _emittedTail = true;
                if (_parts != SplitParts.Separators) {
                    ReadOnlySpan<byte> tail = _text.Slice((int)_prevEnd);
                    if (!(_skipEmpty && tail.Length == 0)) {
                        Current = tail;
                        return true;
                    }
                }
                return false;
            }
            long runStart = _chunkBase + _starts[_index];
            long runEnd = runStart + _lengths[_index];
            if (!_atSeparator) {
                _atSeparator = true;
                if (_parts != SplitParts.Separators) {
                    ReadOnlySpan<byte> gap = _text.Slice((int)_prevEnd, (int)(runStart - _prevEnd));
                    if (!(_skipEmpty && gap.Length == 0)) {
                        Current = gap;
                        return true;
                    }
                }
            }
            _atSeparator = false;
            _prevEnd = runEnd;
            _index++;
            if (_parts != SplitParts.Between) {
                ReadOnlySpan<byte> run = _text.Slice((int)runStart, (int)(runEnd - runStart));
                if (!(_skipEmpty && run.Length == 0)) {
                    Current = run;
                    return true;
                }
            }
        }
    }

    public TokenSplitEnumerator GetEnumerator() => this;
}

/// <summary>Stack-only 64-byte scratch buffer holding cached uncased-needle metadata across match steps.</summary>
[System.Runtime.CompilerServices.InlineArray(64)]
internal struct ByteBuf64 {
    private byte _element0;
}

/// <summary>A single split point: the spans before, of, and after a separator. See <see cref="Sz.Partition"/>.</summary>
public readonly ref struct Partition {
    public readonly ReadOnlySpan<byte> Before;
    public readonly ReadOnlySpan<byte> Separator;
    public readonly ReadOnlySpan<byte> After;

    public Partition(ReadOnlySpan<byte> before, ReadOnlySpan<byte> separator, ReadOnlySpan<byte> after) {
        Before = before;
        Separator = separator;
        After = after;
    }

    public void Deconstruct(out ReadOnlySpan<byte> before, out ReadOnlySpan<byte> separator, out ReadOnlySpan<byte> after) {
        before = Before;
        separator = Separator;
        after = After;
    }
}

/// <summary>One case-insensitive match: its byte offset and matched length (folding may make the latter
/// differ from the needle length).</summary>
public readonly struct UncasedMatch {
    public readonly long Offset;
    public readonly long Length;

    public UncasedMatch(long offset, long length) {
        Offset = offset;
        Length = length;
    }
}

/// <summary>Allocation-free <c>foreach</c> source over uncased matches. See <see cref="Sz.EnumerateUncasedMatches"/>.</summary>
public readonly ref struct UncasedMatchEnumerable {
    private readonly ReadOnlySpan<byte> _haystack;
    private readonly ReadOnlySpan<byte> _needle;
    private readonly bool _overlapping;

    internal UncasedMatchEnumerable(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle, bool overlapping) {
        _haystack = haystack;
        _needle = needle;
        _overlapping = overlapping;
    }

    /// <summary>Report overlapping matches (advance by one byte) instead of skipping past each match.</summary>
    public UncasedMatchEnumerable Overlapping() => new(_haystack, _needle, true);

    public UncasedMatchEnumerator GetEnumerator() => new(_haystack, _needle, _overlapping);
}

/// <summary>Ref-struct enumerator yielding <see cref="UncasedMatch"/>es, reusing one cached metadata buffer.</summary>
public unsafe ref struct UncasedMatchEnumerator {
    private readonly ReadOnlySpan<byte> _haystack;
    private readonly ReadOnlySpan<byte> _needle;
    private readonly bool _overlapping;
    private long _cursor;
    private ByteBuf64 _meta; // populated on first search, reused thereafter

    internal UncasedMatchEnumerator(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle, bool overlapping) {
        _haystack = haystack;
        _needle = needle;
        _overlapping = overlapping;
        _cursor = 0;
        _meta = default;
        Current = default;
    }

    public UncasedMatch Current { get; private set; }

    public bool MoveNext() {
        if (_needle.Length == 0 || _cursor > _haystack.Length) return false;
        Span<byte> meta = _meta[..];
        fixed (byte* hayBase = _haystack)
        fixed (byte* needlePtr = _needle)
        fixed (byte* metaPtr = meta) {
            nuint matchedLength;
            nint found;
            Sz.Check("sz_utf8_uncased_search_best", Native.sz_utf8_uncased_search_best(
                (nint)(hayBase + _cursor), (nuint)(_haystack.Length - _cursor),
                (nint)needlePtr, (nuint)_needle.Length,
                (nint)metaPtr, &found, (nint)(&matchedLength), Sz.CpuEnabled, nint.Zero));
            if (found == 0) {
                _cursor = _haystack.Length + 1;
                return false;
            }
            long offset = (long)((byte*)found - hayBase);
            Current = new UncasedMatch(offset, (long)matchedLength);
            // Overlapping advances one codepoint (not one byte): caseless folding is Unicode-aware, so the
            // next search must start on a UTF-8 boundary rather than mid-codepoint.
            _cursor = offset + (_overlapping ? Utf8LeadWidth(_haystack[(int)offset]) : Math.Max((long)matchedLength, 1));
            return true;
        }
    }

    private static int Utf8LeadWidth(byte lead) =>
        (lead & 0x80) == 0 ? 1 : (lead & 0xE0) == 0xC0 ? 2 : (lead & 0xF0) == 0xE0 ? 3 : (lead & 0xF8) == 0xF0 ? 4 : 1;

    public UncasedMatchEnumerator GetEnumerator() => this;
}

/// <summary>A native call returned a non-success <c>sz_status_t</c>, kept in <see cref="Status"/>.</summary>
public sealed class StatusException : InvalidOperationException {
    /// <summary>The raw <c>sz_status_t</c> code, e.g. -10 for <c>sz_bad_alloc_k</c>.</summary>
    public int Status { get; }

    /// <summary>The C enumerator name of <see cref="Status"/>, e.g. <c>"sz_bad_alloc_k"</c>.</summary>
    public string StatusName => NameOf(Status);

    internal StatusException(string function, int status)
        : base($"{function} failed with {NameOf(status)} ({status})") => Status = status;

    private static string NameOf(int status) =>
        status switch {
            -10 => "sz_bad_alloc_k",
            -12 => "sz_invalid_utf8_k",
            -13 => "sz_contains_duplicates_k",
            -14 => "sz_overflow_risk_k",
            -15 => "sz_unexpected_dimensions_k",
            -16 => "sz_missing_gpu_k",
            -17 => "sz_device_code_mismatch_k",
            -18 => "sz_device_memory_mismatch_k",
            -19 => "sz_authentication_failed_k",
            -20 => "sz_missing_kernel_k",
            -21 => "sz_missing_library_k",
            _ => "sz_status_unknown_k",
        };
}

/// <summary>A 256-bit set of byte values.</summary>
[StructLayout(LayoutKind.Sequential, Size = 32)]
public unsafe struct Byteset {
    private fixed ulong _w[4];

    public void Add(byte value) => _w[value >> 6] |= 1UL << (value & 63);

    public void AddRange(ReadOnlySpan<byte> values) {
        foreach (byte v in values) Add(v);
    }

    public bool Contains(byte value) => (_w[value >> 6] & (1UL << (value & 63))) != 0;

    public void Invert() {
        _w[0] = ~_w[0]; _w[1] = ~_w[1]; _w[2] = ~_w[2]; _w[3] = ~_w[3];
    }

    public static Byteset FromBytes(ReadOnlySpan<byte> values) {
        Byteset s = default;
        s.AddRange(values);
        return s;
    }
}

/// <summary>Incremental (streaming) StringZilla hash. Dispose to free the native state.</summary>
public sealed unsafe class Hasher : IDisposable {
    // sz_hash_state_t is 216 bytes; over-allocate and 64-byte align for SIMD safety.
    private void* _state;

    public Hasher(ulong seed = 0) {
        _state = NativeMemory.AlignedAlloc(256, 64);
        Sz.Check("sz_hash_state_init_best", Native.sz_hash_state_init_best((nint)_state, seed, Sz.CpuEnabled, nint.Zero));
    }

    public void Update(ReadOnlySpan<byte> data) {
        if (_state == null) throw new ObjectDisposedException(nameof(Hasher));
        fixed (byte* p = data)
            Sz.Check("sz_hash_state_update_best", Native.sz_hash_state_update_best(
                (nint)_state, (nint)p, (nuint)data.Length, Sz.CpuEnabled, nint.Zero));
    }

    /// <summary>Finalize without mutating state (may be called repeatedly).</summary>
    public ulong Digest() {
        if (_state == null) throw new ObjectDisposedException(nameof(Hasher));
        ulong hash;
        Sz.Check("sz_hash_state_digest_best", Native.sz_hash_state_digest_best((nint)_state, &hash, Sz.CpuEnabled, nint.Zero));
        return hash;
    }

    public void Dispose() {
        if (_state != null) { NativeMemory.AlignedFree(_state); _state = null; }
    }

    ~Hasher() => Dispose();
}

/// <summary>A reusable case-folding search needle: precompute once, search many haystacks.</summary>
/// <remarks>Fills a gap: no .NET caseless UTF-8 search. Backs <see cref="Sz.UncasedIndexOf"/>.</remarks>
public sealed unsafe class UncasedNeedle : IDisposable {
    private readonly byte[] _needle;
    private void* _meta; // sz_utf8_uncased_needle_metadata_t (40 bytes), populated on first search

    public UncasedNeedle(ReadOnlySpan<byte> needle) {
        _needle = needle.ToArray();
        _meta = NativeMemory.AllocZeroed(64, 1);
    }

    /// <summary>First byte offset of this needle in <paramref name="haystack"/> (caseless), or -1.</summary>
    public long IndexIn(ReadOnlySpan<byte> haystack, out long matchedLength) {
        if (_meta == null) throw new ObjectDisposedException(nameof(UncasedNeedle));
        fixed (byte* h = haystack)
        fixed (byte* n = _needle) {
            nuint ml;
            nint r;
            Sz.Check("sz_utf8_uncased_search_best", Native.sz_utf8_uncased_search_best(
                (nint)h, (nuint)haystack.Length, (nint)n, (nuint)_needle.Length, (nint)_meta, &r, (nint)(&ml),
                Sz.CpuEnabled, nint.Zero));
            matchedLength = (long)ml;
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    public void Dispose() {
        if (_meta != null) { NativeMemory.Free(_meta); _meta = null; }
    }

    ~UncasedNeedle() => Dispose();
}

/// <summary>Incremental SHA-256. Dispose to free the native state.</summary>
/// <remarks>Equivalent to <see cref="System.Security.Cryptography.IncrementalHash"/> with SHA-256,
/// SIMD-accelerated. One-shot: <see cref="HashData"/>.</remarks>
public sealed unsafe class Sha256 : IDisposable {
    private void* _state; // sz_sha256_state_t is 112 bytes

    public Sha256() {
        _state = NativeMemory.AlignedAlloc(128, 64);
        Sz.Check("sz_sha256_state_init_best", Native.sz_sha256_state_init_best((nint)_state, Sz.CpuEnabled, nint.Zero));
    }

    public void Update(ReadOnlySpan<byte> data) {
        if (_state == null) throw new ObjectDisposedException(nameof(Sha256));
        fixed (byte* p = data)
            Sz.Check("sz_sha256_state_update_best", Native.sz_sha256_state_update_best(
                (nint)_state, (nint)p, (nuint)data.Length, Sz.CpuEnabled, nint.Zero));
    }

    /// <summary>Writes the 32-byte digest into <paramref name="destination"/> (must be ≥ 32 bytes;
    /// non-mutating, may be called repeatedly). Allocation-free.</summary>
    public void Digest(Span<byte> destination) {
        if (_state == null) throw new ObjectDisposedException(nameof(Sha256));
        if (destination.Length < 32) throw new ArgumentException("destination must hold at least 32 bytes", nameof(destination));
        fixed (byte* o = destination)
            Sz.Check("sz_sha256_state_digest_best", Native.sz_sha256_state_digest_best(
                (nint)_state, (nint)o, Sz.CpuEnabled, nint.Zero));
    }

    /// <summary>Finalize into a fresh 32-byte digest (non-mutating; may be called repeatedly).</summary>
    public byte[] Digest() {
        var digest = new byte[32];
        Digest(digest);
        return digest;
    }

    /// <summary>One-shot SHA-256 of <paramref name="data"/> → 32-byte digest.</summary>
    /// <remarks>Equivalent to <see cref="System.Security.Cryptography.SHA256.HashData(System.ReadOnlySpan{byte})"/>.</remarks>
    public static byte[] HashData(ReadOnlySpan<byte> data) {
        using var h = new Sha256();
        h.Update(data);
        return h.Digest();
    }

    public void Dispose() {
        if (_state != null) { NativeMemory.AlignedFree(_state); _state = null; }
    }

    ~Sha256() => Dispose();
}

/// <summary>P/Invoke entry points and native-library resolution.</summary>
internal static unsafe partial class Native {
    private const string Lib = "stringzilla";

    static Native() => NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve);

    private static nint Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath) {
        if (name != Lib) return nint.Zero;
        string file = NativeFileName();
        string rid = Rid();
        foreach (string dir in CandidateDirs(rid)) {
            string full = Path.Combine(dir, file);
            if (File.Exists(full) && NativeLibrary.TryLoad(full, out nint handle)) return handle;
        }
        return NativeLibrary.TryLoad(file, assembly, searchPath, out nint fallback) ? fallback : nint.Zero;
    }

    private static IEnumerable<string> CandidateDirs(string rid) {
        string? env = Environment.GetEnvironmentVariable("STRINGZILLA_NATIVE_DIR");
        if (!string.IsNullOrEmpty(env)) yield return env;
        // AppContext.BaseDirectory is AOT/single-file safe (unlike Assembly.Location).
        string baseDir = AppContext.BaseDirectory;
        yield return Path.Combine(baseDir, "runtimes", rid, "native");
        yield return baseDir;
    }

    private static string Rid() {
        string arch = RuntimeInformation.OSArchitecture switch {
            Architecture.X64 => "x64",
            Architecture.Arm64 => "arm64",
            Architecture.X86 => "x86",
            // Ppc64le, LoongArch64, RiscV64, S390x, ... - the enum names lowercase into the matching RID arch,
            // so an unlisted architecture resolves to its own native directory instead of masquerading as x64.
            _ => RuntimeInformation.OSArchitecture.ToString().ToLowerInvariant(),
        };
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return $"win-{arch}";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return $"osx-{arch}";
        return $"linux-{arch}";
    }

    private static string NativeFileName() {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return "stringzilla.dll";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return "libstringzilla.dylib";
        return "libstringzilla.so";
    }

    #region Find
    [LibraryImport(Lib)] internal static partial int sz_find_best(nint haystack, nuint hLen, nint needle, nuint nLen, nint* match, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_rfind_best(nint haystack, nuint hLen, nint needle, nuint nLen, nint* match, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_find_byte_best(nint haystack, nuint hLen, nint needle, nint* match, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_rfind_byte_best(nint haystack, nuint hLen, nint needle, nint* match, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_find_byteset_best(nint haystack, nuint hLen, nint set, nint* match, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_rfind_byteset_best(nint haystack, nuint hLen, nint set, nint* match, ulong caps, nint stream);

    #endregion

    #region Compare
    [LibraryImport(Lib)] internal static partial int sz_equal_best(nint a, nint b, nuint len, int* equal, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_order_best(nint a, nuint aLen, nint b, nuint bLen, int* ordering, ulong caps, nint stream);

    #endregion

    #region Hash
    [LibraryImport(Lib)] internal static partial int sz_hash_best(nint text, nuint len, ulong seed, ulong* hash, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_bytesum_best(nint text, nuint len, ulong* checksum, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_hash_state_init_best(nint state, ulong seed, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_hash_state_update_best(nint state, nint text, nuint len, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_hash_state_digest_best(nint state, ulong* hash, ulong caps, nint stream);

    #endregion

    #region Capabilities and Metadata
    [LibraryImport(Lib)] internal static partial int sz_cpu_capabilities_detected(ulong* capabilities);
    [LibraryImport(Lib)] internal static partial int sz_cpu_capabilities_compiled(ulong* capabilities);
    [LibraryImport(Lib)] internal static partial int sz_cpu_capabilities_enabled(ulong* capabilities);
    [LibraryImport(Lib)] internal static partial int sz_cpu_configure_thread(ulong capabilities);
    [LibraryImport(Lib)] internal static partial int sz_cuda_count_devices(nuint* count);
    [LibraryImport(Lib)] internal static partial int sz_cuda_capabilities_detected(nuint device, ulong* capabilities);
    [LibraryImport(Lib)] internal static partial int sz_cuda_capabilities_compiled(ulong* capabilities);
    [LibraryImport(Lib)] internal static partial int sz_cuda_capabilities_enabled(nuint device, ulong* capabilities);
    [LibraryImport(Lib)] internal static partial int sz_rocm_count_devices(nuint* count);
    [LibraryImport(Lib)] internal static partial int sz_rocm_capabilities_detected(nuint device, ulong* capabilities);
    [LibraryImport(Lib)] internal static partial int sz_rocm_capabilities_compiled(ulong* capabilities);
    [LibraryImport(Lib)] internal static partial int sz_rocm_capabilities_enabled(nuint device, ulong* capabilities);
    [LibraryImport(Lib)] internal static partial int sz_metal_count_devices(nuint* count);
    [LibraryImport(Lib)] internal static partial int sz_metal_capabilities_detected(nuint device, ulong* capabilities);
    [LibraryImport(Lib)] internal static partial int sz_metal_capabilities_compiled(ulong* capabilities);
    [LibraryImport(Lib)] internal static partial int sz_metal_capabilities_enabled(nuint device, ulong* capabilities);
    [LibraryImport(Lib)] internal static partial nuint sz_capabilities_name(ulong capabilities, byte* buffer, nuint capacity);
    [LibraryImport(Lib)] internal static partial int sz_version_major();
    [LibraryImport(Lib)] internal static partial int sz_version_minor();
    [LibraryImport(Lib)] internal static partial int sz_version_patch();

    #endregion

    #region UTF-8 Codepoints
    [LibraryImport(Lib)] internal static partial int sz_utf8_count_best(nint text, nuint len, nuint* count, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_utf8_seek_best(nint text, nuint len, nuint n, nint* position, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_utf8_decode_best(nint text, nuint len, nint runes, nuint cap, nint unpacked, nuint* consumed, ulong caps, nint stream);

    #endregion

    #region UTF-8 Segmentation
    [LibraryImport(Lib)] internal static partial int sz_utf8_graphemes_best(nint text, nuint len, nint starts, nint lengths, nuint cap, nuint* count, nint consumed, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_utf8_wordbreaks_best(nint text, nuint len, nint starts, nint lengths, nuint cap, nuint* count, nint consumed, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_utf8_sentences_best(nint text, nuint len, nint starts, nint lengths, nuint cap, nuint* count, nint consumed, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_utf8_linebreaks_best(nint text, nuint len, nint starts, nint lengths, nuint cap, nuint* count, nint consumed, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_utf8_newlines_best(nint text, nuint len, nint starts, nint lengths, nuint cap, nuint* count, nint consumed, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_utf8_whitespaces_best(nint text, nuint len, nint starts, nint lengths, nuint cap, nuint* count, nint consumed, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_utf8_delimiters_best(nint text, nuint len, nint starts, nint lengths, nuint cap, nuint* count, nint consumed, ulong caps, nint stream);

    #endregion

    #region UTF-8 Case Folding and Uncased
    [LibraryImport(Lib)] internal static partial int sz_utf8_uncased_fold_best(nint src, nuint len, nint dst, nuint* written, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_utf8_uncased_search_best(nint hay, nuint hLen, nint needle, nuint nLen, nint meta, nint* match, nint matchedLen, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_utf8_uncased_order_best(nint a, nuint aLen, nint b, nuint bLen, int* ordering, ulong caps, nint stream);

    #endregion

    #region Hash Extras, SHA-256, Random and Lookup
    [LibraryImport(Lib)] internal static partial int sz_hash_multiseed_best(nint text, nuint len, nint seeds, nuint count, nint hashes, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_sha256_state_init_best(nint state, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_sha256_state_update_best(nint state, nint data, nuint len, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_sha256_state_digest_best(nint state, nint digest, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_fill_random_best(nint text, nuint len, ulong nonce, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_lookup_best(nint target, nint source, nuint len, nint lut, ulong caps, nint stream);

    #endregion

    #region UTF-8 Normalization
    [LibraryImport(Lib)] internal static partial int sz_utf8_norm_best(nint src, nuint len, int form, nint dst, nuint* written, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_utf8_find_denormalized_best(nint src, nuint len, int form, nint* violation, ulong caps, nint stream);

    #endregion

    #region Collections (Sequence Callbacks)
    [LibraryImport(Lib)] internal static partial int sz_sequence_argsort_best(nint sequence, nuint top, int reverse, nint allocator, nint order, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_sequence_argsort_uncased_best(nint sequence, nuint top, int reverse, nint allocator, nint order, ulong caps, nint stream);
    [LibraryImport(Lib)] internal static partial int sz_sequence_intersect_best(nint first, nint second, nint allocator, ulong seed, nint size, nint firstPos, nint secondPos, ulong caps, nint stream);

    #endregion
}

/// <summary>Off-heap string table backing a <c>sz_sequence_t</c>: concatenated bytes + per-item
/// (start, length). The callbacks index into this with trivial pointer arithmetic.</summary>
internal unsafe struct SeqContext {
    public byte* Data;
    public nuint* Starts;
    public nuint* Lengths;
}

/// <summary>Mirrors C <c>sz_sequence_t</c> { handle, count, get_start, get_length }.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct SzSequence {
    public void* Handle;
    public nuint Count;
    public delegate* unmanaged<void*, nuint, nint> GetStart;
    public delegate* unmanaged<void*, nuint, nuint> GetLength;
}

/// <summary>Unmanaged (AOT-safe) callbacks the native sort/intersect invokes per element.</summary>
internal static unsafe class SeqCallbacks {
    [UnmanagedCallersOnly]
    public static nint GetStart(void* handle, nuint index) {
        var c = (SeqContext*)handle;
        return (nint)(c->Data + c->Starts[index]);
    }

    [UnmanagedCallersOnly]
    public static nuint GetLength(void* handle, nuint index) {
        var c = (SeqContext*)handle;
        return c->Lengths[index];
    }

    [UnmanagedCallersOnly]
    public static nint GetRangeStart(void* handle, nuint index) {
        var c = (SeqRangeContext*)handle;
        return (nint)(c->Base + c->Starts[index]);
    }

    [UnmanagedCallersOnly]
    public static nuint GetRangeLength(void* handle, nuint index) {
        var c = (SeqRangeContext*)handle;
        return (nuint)c->Lengths[index];
    }
}

/// <summary>Backs a <c>sz_sequence_t</c> over (start, length) ranges into one caller-pinned buffer; the
/// callbacks index it without copying any segment bytes.</summary>
internal unsafe struct SeqRangeContext {
    public byte* Base;
    public long* Starts;
    public long* Lengths;
}
