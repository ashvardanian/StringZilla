/**
 *  @brief Unit tests for the StringZilla C# bindings.
 *  @file csharp/tests/StringZillaTests.cs
 *  @author Ash Vardanian
 */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using StringZilla;
using Xunit;

public class StringZillaTests {
    private static byte[] B(string s) => Encoding.UTF8.GetBytes(s);

    [Theory]
    [InlineData("the quick brown fox", "quick")]
    [InlineData("the quick brown fox", "fox")]
    [InlineData("the quick brown fox", "the")]
    [InlineData("the quick brown fox", "cat")]
    [InlineData("aaaaa", "aa")]
    [InlineData("héllo wörld", "wör")]
    [InlineData("", "x")]
    public void IndexOf_MatchesMemoryExtensions(string hay, string needle) {
        byte[] h = B(hay), n = B(needle);
        Assert.Equal(h.AsSpan().IndexOf(n), (int)Sz.IndexOf(h, n));
    }

    [Theory]
    [InlineData("the quick brown fox the", "the")]
    [InlineData("aaaaa", "aa")]
    [InlineData("abc", "z")]
    public void LastIndexOf_MatchesMemoryExtensions(string hay, string needle) {
        byte[] h = B(hay), n = B(needle);
        Assert.Equal(h.AsSpan().LastIndexOf(n), (int)Sz.LastIndexOf(h, n));
    }

    [Fact]
    public void IndexOf_Byte_MatchesMemoryExtensions() {
        byte[] h = B("the quick brown fox");
        Assert.Equal(h.AsSpan().IndexOf((byte)'q'), (int)Sz.IndexOf(h, (byte)'q'));
        Assert.Equal(h.AsSpan().LastIndexOf((byte)'o'), (int)Sz.LastIndexOf(h, (byte)'o'));
        Assert.Equal(-1, Sz.IndexOf(h, (byte)'Z'));
    }

    [Fact]
    public void Equal_MatchesSequenceEqual() {
        Assert.True(Sz.Equal(B("abc"), B("abc")));
        Assert.False(Sz.Equal(B("abc"), B("abd")));
        Assert.False(Sz.Equal(B("abc"), B("abcd")));
        Assert.True(Sz.Equal(B(""), B("")));
    }

    [Theory]
    [InlineData("abc", "abc")]
    [InlineData("abc", "abd")]
    [InlineData("abc", "ab")]
    [InlineData("abc", "abcd")]
    [InlineData("b", "a")]
    public void Compare_SignMatchesSequenceCompareTo(string a, string b) {
        Assert.Equal(Math.Sign(B(a).AsSpan().SequenceCompareTo(B(b))), Math.Sign(Sz.Compare(B(a), B(b))));
    }

    [Fact]
    public void Hash_IsDeterministicAndSeedSensitive() {
        byte[] data = B("the quick brown fox jumps over the lazy dog");
        Assert.Equal(Sz.Hash(data), Sz.Hash(data));
        Assert.Equal(Sz.Hash(data, 42), Sz.Hash(data, 42));
        Assert.NotEqual(Sz.Hash(data, 0), Sz.Hash(data, 42));
    }

    [Fact]
    public void Hasher_Streaming_EqualsOneShot() {
        byte[] data = B("the quick brown fox jumps over the lazy dog");
        using var h = new Hasher(seed: 7);
        h.Update(data.AsSpan(0, 10));
        h.Update(data.AsSpan(10));
        Assert.Equal(Sz.Hash(data, 7), h.Digest());
        Assert.Equal(h.Digest(), h.Digest()); // digest is non-mutating
    }

    [Fact]
    public void ByteSum_EqualsManualSum() {
        byte[] data = B("StringZilla ÿþ");
        ulong expected = 0;
        foreach (byte x in data) expected += x;
        Assert.Equal(expected, Sz.ByteSum(data));
    }

    [Fact]
    public void Byteset_AddContains_AndSearch() {
        var set = Byteset.FromBytes(B("aeiou"));
        Assert.True(set.Contains((byte)'e'));
        Assert.False(set.Contains((byte)'z'));

        byte[] h = B("rhythm xy z aei");
        Assert.Equal(h.AsSpan().IndexOfAny(B("aeiou")), (int)Sz.IndexOfAny(h, ref set));
        Assert.Equal(h.AsSpan().LastIndexOfAny(B("aeiou")), (int)Sz.LastIndexOfAny(h, ref set));
    }

    [Fact]
    public void Metadata_IsPopulated() {
        Assert.False(string.IsNullOrEmpty(Sz.Backend));
        Assert.True(Sz.Version.Major >= 4);
    }

    #region Codepoints

    [Fact]
    public void CountRunes_MatchesRuneCount() {
        string s = "héllo, 世界 🦖";
        long expected = 0;
        foreach (var _ in s.EnumerateRunes()) expected++;
        Assert.Equal(expected, Sz.CountRunes(B(s)));
    }

    [Fact]
    public void DecodeAll_MatchesRuneScalars() {
        string s = "AÉ世🦖";
        var expected = new List<int>();
        foreach (var r in s.EnumerateRunes()) expected.Add(r.Value);
        Assert.Equal(expected, Sz.DecodeAll(B(s)).ToList());
    }

    [Fact]
    public void SeekRune_FindsCodepointOffsets() {
        byte[] u = B("aé世"); // a=1B, é=2B, 世=3B
        Assert.Equal(0, Sz.SeekRune(u, 0));
        Assert.Equal(1, Sz.SeekRune(u, 1));
        Assert.Equal(3, Sz.SeekRune(u, 2));
        Assert.Equal(-1, Sz.SeekRune(u, 3));
    }

    #endregion

    #region Segmentation

    [Fact]
    public void Segment_Words_FindsWords() {
        byte[] u = B("the quick fox");
        Span<long> starts = stackalloc long[32];
        Span<long> lengths = stackalloc long[32];
        int count = Sz.Segment(u, Sz.SegmentKind.Words, starts, lengths, out _);
        var texts = new List<string>();
        for (int i = 0; i < count; i++)
            texts.Add(Encoding.UTF8.GetString(u, (int)starts[i], (int)lengths[i]));
        Assert.Contains("the", texts);
        Assert.Contains("quick", texts);
        Assert.Contains("fox", texts);
    }

    [Fact]
    public void Segment_Graphemes_CombiningMarkIsOneCluster() {
        byte[] u = B("éllo"); // e + combining acute, then l l o => 4 grapheme clusters
        Span<long> starts = stackalloc long[32];
        Span<long> lengths = stackalloc long[32];
        int count = Sz.Segment(u, Sz.SegmentKind.Graphemes, starts, lengths, out _);
        Assert.Equal(4, count); // four clusters regardless of NFC/NFD normalization
        Assert.Equal(0, starts[0]);
        long covered = 0;
        for (int i = 0; i < count; i++) covered += lengths[i];
        Assert.Equal((long)u.Length, covered); // segments cover the whole input
    }

    #endregion

    #region Case Folding and Uncased

    [Fact]
    public void CaseFold_FullFolding() {
        Assert.Equal("ss", Encoding.UTF8.GetString(Sz.CaseFold(B("ß"))));
        Assert.Equal("hello", Encoding.UTF8.GetString(Sz.CaseFold(B("HELLO"))));
    }

    [Fact]
    public void UncasedIndexOf_CaselessMatch() {
        byte[] hay = B("Hello WÖRLD");
        long off = Sz.UncasedIndexOf(hay, B("wörld"), out long ml);
        Assert.Equal(6, off);
        Assert.True(ml > 0);
        Assert.Equal(-1, Sz.UncasedIndexOf(hay, B("xyz"), out _));
    }

    [Fact]
    public void UncasedNeedle_ReusableSearch() {
        using var needle = new UncasedNeedle(B("fox"));
        Assert.Equal(4, needle.IndexIn(B("the FOX"), out _));
        Assert.Equal(0, needle.IndexIn(B("Fox!"), out _));
        Assert.Equal(-1, needle.IndexIn(B("cat"), out _));
    }

    [Fact]
    public void UncasedCompare_IgnoresCase() {
        Assert.Equal(0, Sz.UncasedCompare(B("HeLLo"), B("hello")));
        Assert.True(Sz.UncasedCompare(B("apple"), B("BANANA")) < 0);
    }

    #endregion

    #region Hash Extras, SHA-256 and Memory

    [Fact]
    public void HashMultiseed_MatchesPerSeedHash() {
        byte[] d = B("multiseed test");
        ulong[] seeds = { 0, 1, 42, 1000 };
        ulong[] multi = Sz.HashMultiseed(d, seeds);
        Assert.Equal(seeds.Length, multi.Length);
        for (int i = 0; i < seeds.Length; i++)
            Assert.Equal(Sz.Hash(d, seeds[i]), multi[i]);
    }

    [Fact]
    public void Sha256_MatchesBcl() {
        byte[] d = B("the quick brown fox");
        byte[] expected = System.Security.Cryptography.SHA256.HashData(d);
        Assert.Equal(expected, Sha256.HashData(d));
        using var h = new Sha256();
        h.Update(d.AsSpan(0, 5));
        h.Update(d.AsSpan(5));
        Assert.Equal(expected, h.Digest());
    }

    [Fact]
    public void FillRandom_IsDeterministic() {
        var a = new byte[64];
        var b = new byte[64];
        Sz.FillRandom(a, 12345);
        Sz.FillRandom(b, 12345);
        Assert.Equal(a, b);
        var c = new byte[64];
        Sz.FillRandom(c, 999);
        Assert.NotEqual(a, c);
    }

    [Fact]
    public void Lookup_AppliesTable() {
        var lut = new byte[256];
        for (int i = 0; i < 256; i++) lut[i] = (byte)(i + 1);
        byte[] src = { 1, 2, 3, 254 };
        var dst = new byte[src.Length];
        Sz.Lookup(dst, src, lut);
        Assert.Equal(new byte[] { 2, 3, 4, 255 }, dst);
    }

    #endregion

    #region Normalization

    [Fact]
    public void Normalize_Nfc_Composes() {
        byte[] decomposed = { (byte)'e', 0xCC, 0x81 }; // 'e' + U+0301
        Assert.Equal(new byte[] { 0xC3, 0xA9 }, Sz.Normalize(decomposed, Sz.NormalForm.Nfc)); // é U+00E9
    }

    [Fact]
    public void Normalize_Nfd_Decomposes() {
        byte[] precomposed = { 0xC3, 0xA9 }; // é U+00E9
        Assert.Equal(new byte[] { (byte)'e', 0xCC, 0x81 }, Sz.Normalize(precomposed, Sz.NormalForm.Nfd));
    }

    [Fact]
    public void IsNormalized_DetectsForm() {
        Assert.True(Sz.IsNormalized(new byte[] { 0xC3, 0xA9 }, Sz.NormalForm.Nfc));
        Assert.False(Sz.IsNormalized(new byte[] { (byte)'e', 0xCC, 0x81 }, Sz.NormalForm.Nfc));
    }

    #endregion

    #region Collections

    [Fact]
    public void ArgSort_SortsLexicographically() {
        var items = new[] { B("banana"), B("apple"), B("cherry") };
        Assert.Equal(new long[] { 1, 0, 2 }, Sz.ArgSort(items));
        Assert.Equal(new long[] { 2, 0, 1 }, Sz.ArgSort(items, reverse: true));
    }

    [Fact]
    public void ArgSort_Uncased() {
        var items = new[] { B("Banana"), B("apple") };
        Assert.Equal(new long[] { 1, 0 }, Sz.ArgSort(items, uncased: true));
    }

    [Fact]
    public void ArgSort_TopK() {
        var items = new[] { B("d"), B("a"), B("c"), B("b") };
        Assert.Equal(new long[] { 1, 3 }, Sz.ArgSort(items, top: 2));
    }

    [Fact]
    public void Intersect_FindsMatches() {
        var a = new[] { B("a"), B("b"), B("c") };
        var b = new[] { B("b"), B("c"), B("d") };
        Span<long> first = stackalloc long[3];
        Span<long> second = stackalloc long[3];
        int count = Sz.Intersect(a, b, first, second);
        var pairs = new HashSet<(long, long)>();
        for (int i = 0; i < count; i++) pairs.Add((first[i], second[i]));
        Assert.Equal(2, count);
        Assert.Contains((1L, 0L), pairs); // "b"
        Assert.Contains((2L, 1L), pairs); // "c"
    }
    #endregion

    #region Iteration

    [Fact]
    public void EnumerateRunes_MatchesEnumerateRunes() {
        string s = "héllo, 世界 🦖";
        var expected = new List<int>();
        foreach (var r in s.EnumerateRunes()) expected.Add(r.Value);
        var actual = new List<int>();
        foreach (var r in Sz.EnumerateRunes(B(s))) actual.Add(r.Value);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void EnumerateWords_TilesInput() {
        byte[] u = B("the quick fox");
        var words = new List<string>();
        long covered = 0;
        foreach (var w in Sz.EnumerateWords(u)) {
            words.Add(Encoding.UTF8.GetString(w));
            covered += w.Length;
        }
        Assert.Equal((long)u.Length, covered); // tiling covers the whole input
        Assert.Contains("the", words);
        Assert.Contains("quick", words);
        Assert.Contains("fox", words);
    }

    [Fact]
    public void EnumerateGraphemes_MatchesSegmentPrimitive() {
        byte[] u = B("éllo, world!");
        var viaEnumerator = new List<int>();
        foreach (var g in Sz.EnumerateGraphemes(u)) viaEnumerator.Add(g.Length);
        Span<long> starts = stackalloc long[64];
        Span<long> lengths = stackalloc long[64];
        int count = Sz.Segment(u, Sz.SegmentKind.Graphemes, starts, lengths, out _);
        Assert.Equal(count, viaEnumerator.Count);
        for (int i = 0; i < count; i++) Assert.Equal((int)lengths[i], viaEnumerator[i]);
    }

    [Fact]
    public void EnumerateWords_RefillsAcrossBatches() {
        var builder = new StringBuilder();
        for (int i = 0; i < 200; i++) builder.Append('w').Append(i).Append(' ');
        byte[] u = B(builder.ToString());
        int wordCount = 0;
        foreach (var word in Sz.EnumerateWords(u))
            if (word.Length > 0 && word[0] == (byte)'w') wordCount++;
        Assert.Equal(200, wordCount); // forces multiple 64-entry batch refills
    }

    [Fact]
    public void EnumerateRunes_RefillsBeyondBatch() {
        var builder = new StringBuilder();
        for (int i = 0; i < 200; i++) builder.Append('世'); // 200 multibyte codepoints > one 64-rune batch
        int count = 0;
        foreach (var rune in Sz.EnumerateRunes(B(builder.ToString()))) {
            Assert.Equal(0x4E16, rune.Value);
            count++;
        }
        Assert.Equal(200, count);
    }

    #endregion

    #region Splitting

    private static List<string> SplitToList(SplitEnumerable splits) {
        var parts = new List<string>();
        foreach (var part in splits) parts.Add(Encoding.UTF8.GetString(part));
        return parts;
    }

    [Fact]
    public void Split_KeepsEmptySegments() {
        Assert.Equal(new[] { "a", "bb", "", "c" }, SplitToList(Sz.Split(B("a,bb,,c"), ","u8)));
    }

    [Fact]
    public void Split_SkipEmpty() {
        Assert.Equal(new[] { "a", "b" }, SplitToList(Sz.Split(B("a,,b,"), ","u8).SkipEmpty()));
    }

    [Fact]
    public void Split_MaxSplit() {
        Assert.Equal(new[] { "a", "b", "c,d" }, SplitToList(Sz.Split(B("a,b,c,d"), ","u8).WithMaxSplit(2)));
    }

    [Fact]
    public void Split_KeepSeparator() {
        Assert.Equal(new[] { "a,", "b,", "c" }, SplitToList(Sz.Split(B("a,b,c"), ","u8).KeepSeparator()));
    }

    [Fact]
    public void RSplit_ReverseOrder() {
        Assert.Equal(new[] { "c", "b", "a" }, SplitToList(Sz.RSplit(B("a,b,c"), ","u8)));
    }

    [Fact]
    public void RSplit_MaxSplitKeepsLeftRemainder() {
        Assert.Equal(new[] { "d", "a,b,c" }, SplitToList(Sz.RSplit(B("a,b,c,d"), ","u8).WithMaxSplit(1)));
    }

    [Fact]
    public void SplitAny_OnByteset() {
        Assert.Equal(new[] { "a", "b", "c" }, SplitToList(Sz.SplitAny(B("a-b_c"), Byteset.FromBytes("-_"u8))));
    }

    [Fact]
    public void EnumerateMatches_FindsAllOffsets() {
        var offsets = new List<long>();
        foreach (long at in Sz.EnumerateMatches(B("abababab"), "ab"u8)) offsets.Add(at);
        Assert.Equal(new long[] { 0, 2, 4, 6 }, offsets);
    }

    [Fact]
    public void EnumerateMatches_OverlappingVsNot() {
        int nonOverlapping = 0;
        foreach (var _ in Sz.EnumerateMatches(B("aaaa"), "aa"u8)) nonOverlapping++;
        int overlapping = 0;
        foreach (var _ in Sz.EnumerateMatches(B("aaaa"), "aa"u8).Overlapping()) overlapping++;
        Assert.Equal(2, nonOverlapping);
        Assert.Equal(3, overlapping);
    }

    [Fact]
    public void SplitNewlines_YieldsLines() {
        var lines = new List<string>();
        foreach (var line in Sz.SplitNewlines(B("a\nbb\nc"))) lines.Add(Encoding.UTF8.GetString(line));
        Assert.Equal(new[] { "a", "bb", "c" }, lines);
    }

    [Fact]
    public void SplitWhitespaces_SkipEmptyDropsRuns() {
        var words = new List<string>();
        foreach (var word in Sz.SplitWhitespaces(B("  the   quick  ")).SkipEmpty())
            words.Add(Encoding.UTF8.GetString(word));
        Assert.Equal(new[] { "the", "quick" }, words);
    }

    [Fact]
    public void SplitWhitespaces_WithSeparatorsRoundTrips() {
        var rebuilt = new StringBuilder();
        foreach (var part in Sz.SplitWhitespaces(B("the quick fox")).WithSeparators())
            rebuilt.Append(Encoding.UTF8.GetString(part));
        Assert.Equal("the quick fox", rebuilt.ToString());
    }

    [Fact]
    public void RSplit_KeepSeparatorKeepsLeading() {
        Assert.Equal(new[] { ",c", ",b", "a" }, SplitToList(Sz.RSplit(B("a,b,c"), ","u8).KeepSeparator()));
    }

    [Fact]
    public void Split_MaxSplitZeroYieldsWhole() {
        Assert.Equal(new[] { "a,b,c" }, SplitToList(Sz.Split(B("a,b,c"), ","u8).WithMaxSplit(0)));
    }

    [Fact]
    public void SplitDelimiters_OnlySeparatorsYieldsEachDelimiter() {
        var separators = new List<string>();
        foreach (var run in Sz.SplitDelimiters(B("a-b.c")).OnlySeparators())
            separators.Add(Encoding.UTF8.GetString(run));
        Assert.Equal(new[] { "-", "." }, separators);
    }

    #endregion

    #region Uncased Matching, Partition, and Segment Sorting

    [Fact]
    public void EnumerateUncasedMatches_FindsCaseInsensitive() {
        var offsets = new List<long>();
        foreach (var match in Sz.EnumerateUncasedMatches(B("Hello hello HELLO"), "hello"u8))
            offsets.Add(match.Offset);
        Assert.Equal(new long[] { 0, 6, 12 }, offsets);
    }

    [Fact]
    public void EnumerateUncasedMatches_ReportsFoldedLength() {
        // "ß" (U+00DF, two bytes at offset 4) folds to "ss"; searching "SS" matches it.
        var matches = new List<(long Offset, long Length)>();
        foreach (var match in Sz.EnumerateUncasedMatches(B("Straße"), "SS"u8))
            matches.Add((match.Offset, match.Length));
        Assert.Single(matches);
        Assert.Equal(4, matches[0].Offset);
        Assert.Equal(2, matches[0].Length);
    }

    [Fact]
    public void EnumerateUncasedMatches_OverlappingStaysCodepointAligned() {
        int overlapping = 0;
        foreach (var _ in Sz.EnumerateUncasedMatches(B("AAAA"), "aa"u8).Overlapping()) overlapping++;
        Assert.Equal(3, overlapping); // "aa" at 0,1,2

        // Multibyte (each Ä is two bytes): overlapping advances by a codepoint, so offsets stay aligned.
        var offsets = new List<long>();
        foreach (var match in Sz.EnumerateUncasedMatches(B("ÄÄÄ"), "ä"u8).Overlapping()) offsets.Add(match.Offset);
        Assert.Equal(new long[] { 0, 2, 4 }, offsets);
    }

    [Fact]
    public void Partition_SplitsAtFirst() {
        var (before, separator, after) = Sz.Partition(B("key=value=tail"), "="u8);
        Assert.Equal("key", Encoding.UTF8.GetString(before));
        Assert.Equal("=", Encoding.UTF8.GetString(separator));
        Assert.Equal("value=tail", Encoding.UTF8.GetString(after));
    }

    [Fact]
    public void RPartition_SplitsAtLast() {
        var (before, separator, after) = Sz.RPartition(B("a/b/c"), "/"u8);
        Assert.Equal("a/b", Encoding.UTF8.GetString(before));
        Assert.Equal("/", Encoding.UTF8.GetString(separator));
        Assert.Equal("c", Encoding.UTF8.GetString(after));
    }

    [Fact]
    public void Partition_MissingSeparatorKeepsWhole() {
        var (before, separator, after) = Sz.Partition(B("nodelim"), "="u8);
        Assert.Equal("nodelim", Encoding.UTF8.GetString(before));
        Assert.True(separator.IsEmpty);
        Assert.True(after.IsEmpty);
    }

    [Fact]
    public void ArgSort_SortsBufferSegments() {
        byte[] text = B("banana\napple\ncherry");
        Span<long> starts = stackalloc long[] { 0, 7, 13 };
        Span<long> lengths = stackalloc long[] { 6, 5, 6 };
        Span<long> order = stackalloc long[3];
        int count = Sz.ArgSort(text, starts, lengths, order);
        Assert.Equal(3, count);
        Assert.Equal(new long[] { 1, 0, 2 }, order.ToArray()); // apple, banana, cherry
    }

    #endregion

}
