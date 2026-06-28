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
            nint r = Native.sz_find((nint)h, (nuint)haystack.Length, (nint)n, (nuint)needle.Length);
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    public static long LastIndexOf(ReadOnlySpan<byte> haystack, ReadOnlySpan<byte> needle) {
        if (needle.Length == 0) return haystack.Length;
        if (needle.Length > haystack.Length) return -1;
        fixed (byte* h = haystack)
        fixed (byte* n = needle) {
            nint r = Native.sz_rfind((nint)h, (nuint)haystack.Length, (nint)n, (nuint)needle.Length);
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    public static long IndexOf(ReadOnlySpan<byte> haystack, byte needle) {
        fixed (byte* h = haystack) {
            nint r = Native.sz_find_byte((nint)h, (nuint)haystack.Length, (nint)(&needle));
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    public static long LastIndexOf(ReadOnlySpan<byte> haystack, byte needle) {
        fixed (byte* h = haystack) {
            nint r = Native.sz_rfind_byte((nint)h, (nuint)haystack.Length, (nint)(&needle));
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    /// <summary>First offset of any byte present in <paramref name="set"/>, or -1.</summary>
    public static long IndexOfAny(ReadOnlySpan<byte> haystack, ref Byteset set) {
        fixed (byte* h = haystack)
        fixed (Byteset* s = &set) {
            nint r = Native.sz_find_byteset((nint)h, (nuint)haystack.Length, (nint)s);
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    /// <summary>Last offset of any byte present in <paramref name="set"/>, or -1.</summary>
    public static long LastIndexOfAny(ReadOnlySpan<byte> haystack, ref Byteset set) {
        fixed (byte* h = haystack)
        fixed (Byteset* s = &set) {
            nint r = Native.sz_rfind_byteset((nint)h, (nuint)haystack.Length, (nint)s);
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
        fixed (byte* pa = a)
        fixed (byte* pb = b)
            return Native.sz_equal((nint)pa, (nint)pb, (nuint)a.Length) != 0;
    }

    /// <summary>Lexicographic byte comparison: -1, 0, or 1.</summary>
    /// <remarks>Like <see cref="System.MemoryExtensions.SequenceCompareTo{T}(System.ReadOnlySpan{T},System.ReadOnlySpan{T})"/>.</remarks>
    public static int Compare(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b) {
        fixed (byte* pa = a)
        fixed (byte* pb = b)
            return Native.sz_order((nint)pa, (nuint)a.Length, (nint)pb, (nuint)b.Length);
    }

    #endregion

    #region Hashing

    /// <summary>Fast 64-bit AES-based hash (SMHasher-quality), with optional <paramref name="seed"/>.</summary>
    /// <remarks>Unlike <see cref="string.GetHashCode()"/>, this is stable across runs/processes and
    /// identical across all StringZilla bindings. For cryptographic hashing use <see cref="Sha256"/>.</remarks>
    public static ulong Hash(ReadOnlySpan<byte> data, ulong seed = 0) {
        fixed (byte* p = data)
            return Native.sz_hash((nint)p, (nuint)data.Length, seed);
    }

    public static ulong ByteSum(ReadOnlySpan<byte> data) {
        fixed (byte* p = data)
            return Native.sz_bytesum((nint)p, (nuint)data.Length);
    }

    #endregion

    #region UTF-8 Codepoints

    /// <summary>Counts Unicode scalar values (codepoints) in UTF-8 <paramref name="text"/>.</summary>
    /// <remarks>Like counting <see cref="System.Text.Rune"/>s, but SIMD-accelerated over UTF-8 bytes
    /// with no UTF-16 transcode. Keywords: codepoint, rune count, length.</remarks>
    public static long CountRunes(ReadOnlySpan<byte> text) {
        fixed (byte* p = text)
            return (long)Native.sz_utf8_count((nint)p, (nuint)text.Length);
    }

    /// <summary>Byte offset of the <paramref name="index"/>-th codepoint (0-based), or -1 if fewer exist.</summary>
    /// <remarks>Analogous to <see cref="string.EnumerateRunes"/> positioning, on UTF-8 bytes.</remarks>
    public static long SeekRune(ReadOnlySpan<byte> text, long index) {
        fixed (byte* p = text) {
            nint r = Native.sz_utf8_seek((nint)p, (nuint)text.Length, (nuint)index);
            return r == 0 ? -1 : (long)((byte*)r - p);
        }
    }

    /// <summary>Decodes UTF-8 codepoints into <paramref name="destination"/> (one Int32 scalar each;
    /// ill-formed sequences become U+FFFD). Returns the count written.</summary>
    /// <remarks>Like <see cref="System.Text.Rune.DecodeFromUtf8"/> in bulk, SIMD-accelerated. Emitted
    /// values are valid Unicode scalars (incl. U+FFFD): wrap with <c>new Rune(cp)</c>.</remarks>
    public static int Decode(ReadOnlySpan<byte> text, Span<int> destination) {
        fixed (byte* p = text)
        fixed (int* d = destination) {
            nuint unpacked;
            Native.sz_utf8_decode((nint)p, (nuint)text.Length, (nint)d, (nuint)destination.Length, (nint)(&unpacked));
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
    public static int Segment(ReadOnlySpan<byte> text, SegmentKind kind, Span<long> starts, Span<long> lengths, out long bytesConsumed) {
        int cap = Math.Min(starts.Length, lengths.Length);
        fixed (byte* t = text)
        fixed (long* s = starts)
        fixed (long* l = lengths) {
            nuint consumed;
            nuint count = SegmentChunk(kind, (nint)t, (nuint)text.Length, (nint)s, (nint)l, (nuint)cap, (nint)(&consumed));
            bytesConsumed = (long)consumed;
            return (int)count;
        }
    }

    private static nuint SegmentChunk(SegmentKind kind, nint text, nuint len, nint starts, nint lengths, nuint cap, nint consumed) =>
        kind switch {
            SegmentKind.Graphemes => Native.sz_utf8_graphemes(text, len, starts, lengths, cap, consumed),
            SegmentKind.Words => Native.sz_utf8_words(text, len, starts, lengths, cap, consumed),
            SegmentKind.Sentences => Native.sz_utf8_sentences(text, len, starts, lengths, cap, consumed),
            SegmentKind.LineBreaks => Native.sz_utf8_linebreaks(text, len, starts, lengths, cap, consumed),
            SegmentKind.Newlines => Native.sz_utf8_newlines(text, len, starts, lengths, cap, consumed),
            SegmentKind.Whitespaces => Native.sz_utf8_whitespaces(text, len, starts, lengths, cap, consumed),
            SegmentKind.Delimiters => Native.sz_utf8_delimiters(text, len, starts, lengths, cap, consumed),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

    #endregion

    #region UTF-8 Case Folding and Uncased Matching

    /// <summary>Folds into the caller's <paramref name="destination"/> (must hold ≥ text.Length*3
    /// bytes) and returns the bytes written. Allocation-free.</summary>
    public static int CaseFold(ReadOnlySpan<byte> text, Span<byte> destination) {
        if (text.Length == 0) return 0;
        if (destination.Length < text.Length * 3)
            throw new ArgumentException("destination must hold at least text.Length*3 bytes", nameof(destination));
        fixed (byte* s = text)
        fixed (byte* d = destination)
            return (int)Native.sz_utf8_uncased_fold((nint)s, (nuint)text.Length, (nint)d);
    }

    /// <summary>Full Unicode case folding (UAX #21, one-to-many, e.g. ß → "ss"). Returns folded UTF-8.</summary>
    /// <remarks>Fills a gap: .NET has no public case-folding API (only culture-sensitive
    /// ToLower/ToUpper). Use for caseless matching/indexing. Keywords: case fold, caseless, casefold.</remarks>
    public static byte[] CaseFold(ReadOnlySpan<byte> text) {
        if (text.Length == 0) return Array.Empty<byte>();
        byte[] rent = ArrayPool<byte>.Shared.Rent(text.Length * 3); // worst-case 3x expansion
        try {
            fixed (byte* s = text)
            fixed (byte* d = rent) {
                nuint n = Native.sz_utf8_uncased_fold((nint)s, (nuint)text.Length, (nint)d);
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
            nint r = Native.sz_utf8_uncased_search((nint)h, (nuint)haystack.Length, (nint)n, (nuint)needle.Length, (nint)m, (nint)(&ml));
            matchedLength = (long)ml;
            return r == 0 ? -1 : (long)((byte*)r - h);
        }
    }

    /// <summary>Case-insensitive (full-fold) lexicographic comparison: -1, 0, or 1.</summary>
    /// <remarks>Fills a gap: no .NET Unicode case-folded ordinal comparison.</remarks>
    public static int UncasedCompare(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b) {
        fixed (byte* pa = a)
        fixed (byte* pb = b)
            return Native.sz_utf8_uncased_order((nint)pa, (nuint)a.Length, (nint)pb, (nuint)b.Length);
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
            Native.sz_hash_multiseed((nint)p, (nuint)data.Length, (nint)s, (nuint)seeds.Length, (nint)o);
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
            Native.sz_fill_random((nint)p, (nuint)buffer.Length, nonce);
    }

    /// <summary>Byte-wise table transform: destination[i] = lut[source[i]]. <paramref name="lut"/>
    /// must be exactly 256 bytes.</summary>
    public static void Lookup(Span<byte> destination, ReadOnlySpan<byte> source, ReadOnlySpan<byte> lut) {
        if (lut.Length != 256) throw new ArgumentException("LUT must be 256 bytes", nameof(lut));
        if (destination.Length < source.Length) throw new ArgumentException("destination too small", nameof(destination));
        fixed (byte* d = destination)
        fixed (byte* s = source)
        fixed (byte* l = lut)
            Native.sz_lookup((nint)d, (nuint)source.Length, (nint)s, (nint)l);
    }

    #endregion

    #region UTF-8 Normalization

    public enum NormalForm { Nfd = 0, Nfc = 1, Nfkd = 2, Nfkc = 3 }

    /// <summary>Normalizes into the caller's <paramref name="destination"/> (must hold ≥ text.Length*18
    /// bytes) and returns the bytes written. Allocation-free.</summary>
    public static int Normalize(ReadOnlySpan<byte> text, NormalForm form, Span<byte> destination) {
        if (text.Length == 0) return 0;
        if (destination.Length < text.Length * 18)
            throw new ArgumentException("destination must hold at least text.Length*18 bytes", nameof(destination));
        fixed (byte* s = text)
        fixed (byte* d = destination)
            return (int)Native.sz_utf8_norm((nint)s, (nuint)text.Length, (int)form, (nint)d);
    }

    /// <summary>Normalizes UTF-8 <paramref name="text"/> to the given form. Returns normalized UTF-8.</summary>
    /// <remarks>Equivalent to <see cref="string.Normalize(System.Text.NormalizationForm)"/>, but over
    /// UTF-8 bytes with no UTF-16 transcode.</remarks>
    public static byte[] Normalize(ReadOnlySpan<byte> text, NormalForm form) {
        if (text.Length == 0) return Array.Empty<byte>();
        byte[] rent = ArrayPool<byte>.Shared.Rent(text.Length * 18); // worst-case per-codepoint expansion
        try {
            fixed (byte* s = text)
            fixed (byte* d = rent) {
                nuint n = Native.sz_utf8_norm((nint)s, (nuint)text.Length, (int)form, (nint)d);
                return rent.AsSpan(0, (int)n).ToArray();
            }
        }
        finally { ArrayPool<byte>.Shared.Return(rent); }
    }

    /// <summary>True if <paramref name="text"/> is already in the given normalization form.</summary>
    /// <remarks>Equivalent to <see cref="string.IsNormalized(System.Text.NormalizationForm)"/>.</remarks>
    public static bool IsNormalized(ReadOnlySpan<byte> text, NormalForm form) {
        fixed (byte* s = text)
            return Native.sz_utf8_find_denormalized((nint)s, (nuint)text.Length, (int)form) == 0;
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
            fixed (long* ord = order) {
                int status = uncased
                    ? Native.sz_sequence_argsort_uncased((nint)(&seq), nint.Zero, (nint)ord, (nuint)top, reverse ? 1 : 0)
                    : Native.sz_sequence_argsort((nint)(&seq), nint.Zero, (nint)ord, (nuint)top, reverse ? 1 : 0);
                if (status != 0) throw new InvalidOperationException($"sz_sequence_argsort failed with status {status}");
            }
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
            fixed (long* sp = secondPositions) {
                int status = Native.sz_sequence_intersect((nint)(&seqA), (nint)(&seqB), nint.Zero, seed, (nint)(&size), (nint)fp, (nint)sp);
                if (status != 0) throw new InvalidOperationException($"sz_sequence_intersect failed with status {status}");
            }
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

    #region Library Metadata

    /// <summary>Active SIMD backend(s), e.g. "serial,haswell,skylake,icelake".</summary>
    public static string Backend =>
        Marshal.PtrToStringUTF8(Native.sz_capabilities_to_string(Native.sz_capabilities())) ?? "";

    public static Version Version =>
        new(Native.sz_version_major(), Native.sz_version_minor(), Native.sz_version_patch());

    #endregion
}

/// <summary>A 256-bit set of byte values (mirrors C <c>sz_byteset_t</c>).</summary>
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
        Native.sz_hash_state_init((nint)_state, seed);
    }

    public void Update(ReadOnlySpan<byte> data) {
        if (_state == null) throw new ObjectDisposedException(nameof(Hasher));
        fixed (byte* p = data)
            Native.sz_hash_state_update((nint)_state, (nint)p, (nuint)data.Length);
    }

    /// <summary>Finalize without mutating state (may be called repeatedly).</summary>
    public ulong Digest() {
        if (_state == null) throw new ObjectDisposedException(nameof(Hasher));
        return Native.sz_hash_state_digest((nint)_state);
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
            nint r = Native.sz_utf8_uncased_search((nint)h, (nuint)haystack.Length, (nint)n, (nuint)_needle.Length, (nint)_meta, (nint)(&ml));
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
        Native.sz_sha256_state_init((nint)_state);
    }

    public void Update(ReadOnlySpan<byte> data) {
        if (_state == null) throw new ObjectDisposedException(nameof(Sha256));
        fixed (byte* p = data)
            Native.sz_sha256_state_update((nint)_state, (nint)p, (nuint)data.Length);
    }

    /// <summary>Writes the 32-byte digest into <paramref name="destination"/> (must be ≥ 32 bytes;
    /// non-mutating, may be called repeatedly). Allocation-free.</summary>
    public void Digest(Span<byte> destination) {
        if (_state == null) throw new ObjectDisposedException(nameof(Sha256));
        if (destination.Length < 32) throw new ArgumentException("destination must hold at least 32 bytes", nameof(destination));
        fixed (byte* o = destination)
            Native.sz_sha256_state_digest((nint)_state, (nint)o);
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

    static Native() {
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve);
        sz_dispatch_table_init(); // idempotent; the lib also self-initializes on load
    }

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
            _ => "x64",
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
    [LibraryImport(Lib)] internal static partial nint sz_find(nint haystack, nuint hLen, nint needle, nuint nLen);
    [LibraryImport(Lib)] internal static partial nint sz_rfind(nint haystack, nuint hLen, nint needle, nuint nLen);
    [LibraryImport(Lib)] internal static partial nint sz_find_byte(nint haystack, nuint hLen, nint needle);
    [LibraryImport(Lib)] internal static partial nint sz_rfind_byte(nint haystack, nuint hLen, nint needle);
    [LibraryImport(Lib)] internal static partial nint sz_find_byteset(nint text, nuint len, nint set);
    [LibraryImport(Lib)] internal static partial nint sz_rfind_byteset(nint text, nuint len, nint set);

    #endregion

    #region Compare
    [LibraryImport(Lib)] internal static partial int sz_equal(nint a, nint b, nuint len);
    [LibraryImport(Lib)] internal static partial int sz_order(nint a, nuint aLen, nint b, nuint bLen);

    #endregion

    #region Hash
    [LibraryImport(Lib)] internal static partial ulong sz_hash(nint text, nuint len, ulong seed);
    [LibraryImport(Lib)] internal static partial ulong sz_bytesum(nint text, nuint len);
    [LibraryImport(Lib)] internal static partial void sz_hash_state_init(nint state, ulong seed);
    [LibraryImport(Lib)] internal static partial void sz_hash_state_update(nint state, nint text, nuint len);
    [LibraryImport(Lib)] internal static partial ulong sz_hash_state_digest(nint state);

    #endregion

    #region Memory
    [LibraryImport(Lib)] internal static partial void sz_copy(nint target, nint source, nuint len);
    [LibraryImport(Lib)] internal static partial void sz_move(nint target, nint source, nuint len);
    [LibraryImport(Lib)] internal static partial void sz_fill(nint target, nuint len, byte value);

    #endregion

    #region Metadata
    [LibraryImport(Lib)] internal static partial int sz_capabilities();
    [LibraryImport(Lib)] internal static partial nint sz_capabilities_to_string(int caps);
    [LibraryImport(Lib)] internal static partial int sz_version_major();
    [LibraryImport(Lib)] internal static partial int sz_version_minor();
    [LibraryImport(Lib)] internal static partial int sz_version_patch();
    [LibraryImport(Lib)] internal static partial void sz_dispatch_table_init();

    #endregion

    #region UTF-8 Codepoints
    [LibraryImport(Lib)] internal static partial nuint sz_utf8_count(nint text, nuint len);
    [LibraryImport(Lib)] internal static partial nint sz_utf8_seek(nint text, nuint len, nuint n);
    [LibraryImport(Lib)] internal static partial nint sz_utf8_decode(nint text, nuint len, nint runes, nuint cap, nint unpacked);

    #endregion

    #region UTF-8 Segmentation
    [LibraryImport(Lib)] internal static partial nuint sz_utf8_graphemes(nint text, nuint len, nint starts, nint lengths, nuint cap, nint consumed);
    [LibraryImport(Lib)] internal static partial nuint sz_utf8_words(nint text, nuint len, nint starts, nint lengths, nuint cap, nint consumed);
    [LibraryImport(Lib)] internal static partial nuint sz_utf8_sentences(nint text, nuint len, nint starts, nint lengths, nuint cap, nint consumed);
    [LibraryImport(Lib)] internal static partial nuint sz_utf8_linebreaks(nint text, nuint len, nint starts, nint lengths, nuint cap, nint consumed);
    [LibraryImport(Lib)] internal static partial nuint sz_utf8_newlines(nint text, nuint len, nint starts, nint lengths, nuint cap, nint consumed);
    [LibraryImport(Lib)] internal static partial nuint sz_utf8_whitespaces(nint text, nuint len, nint starts, nint lengths, nuint cap, nint consumed);
    [LibraryImport(Lib)] internal static partial nuint sz_utf8_delimiters(nint text, nuint len, nint starts, nint lengths, nuint cap, nint consumed);

    #endregion

    #region UTF-8 Case Folding and Uncased
    [LibraryImport(Lib)] internal static partial nuint sz_utf8_uncased_fold(nint src, nuint len, nint dst);
    [LibraryImport(Lib)] internal static partial nint sz_utf8_uncased_search(nint hay, nuint hLen, nint needle, nuint nLen, nint meta, nint matchedLen);
    [LibraryImport(Lib)] internal static partial int sz_utf8_uncased_order(nint a, nuint aLen, nint b, nuint bLen);

    #endregion

    #region Hash Extras, SHA-256, Random and Lookup
    [LibraryImport(Lib)] internal static partial void sz_hash_multiseed(nint text, nuint len, nint seeds, nuint count, nint hashes);
    [LibraryImport(Lib)] internal static partial void sz_sha256_state_init(nint state);
    [LibraryImport(Lib)] internal static partial void sz_sha256_state_update(nint state, nint data, nuint len);
    [LibraryImport(Lib)] internal static partial void sz_sha256_state_digest(nint state, nint digest);
    [LibraryImport(Lib)] internal static partial void sz_fill_random(nint text, nuint len, ulong nonce);
    [LibraryImport(Lib)] internal static partial void sz_lookup(nint target, nuint len, nint source, nint lut);

    #endregion

    #region UTF-8 Normalization
    [LibraryImport(Lib)] internal static partial nuint sz_utf8_norm(nint src, nuint len, int form, nint dst);
    [LibraryImport(Lib)] internal static partial nint sz_utf8_find_denormalized(nint src, nuint len, int form);

    #endregion

    #region Collections (Sequence Callbacks)
    [LibraryImport(Lib)] internal static partial int sz_sequence_argsort(nint sequence, nint alloc, nint order, nuint top, int reverse);
    [LibraryImport(Lib)] internal static partial int sz_sequence_argsort_uncased(nint sequence, nint alloc, nint order, nuint top, int reverse);
    [LibraryImport(Lib)] internal static partial int sz_sequence_intersect(nint first, nint second, nint alloc, ulong seed, nint size, nint firstPos, nint secondPos);

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
}
