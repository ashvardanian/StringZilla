/**
 *  @brief Unit tests for the StringZilla Java bindings.
 *  @file java/src/test/java/com/stringzilla/StringZillaTest.java
 *  @author Ash Vardanian
 */
package com.stringzilla;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import org.apache.lucene.util.BytesRef;
import org.junit.jupiter.api.Test;

class StringZillaTest {

    private static byte[] b(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    /** Reference byte-level indexOf for cross-checking. */
    private static long byteIndexOf(byte[] hay, byte[] needle) {
        if (needle.length == 0) return 0;
        outer:
        for (int i = 0; i <= hay.length - needle.length; i++) {
            for (int j = 0; j < needle.length; j++) if (hay[i + j] != needle[j]) continue outer;
            return i;
        }
        return -1;
    }

    @Test
    void indexOf_heap_matchesReference() {
        for (String[] c : new String[][] {
            {"the quick brown fox", "quick"}, {"the quick brown fox", "fox"},
            {"the quick brown fox", "the"}, {"the quick brown fox", "cat"},
            {"aaaaa", "aa"}, {"héllo wörld", "wör"}
        }) {
            byte[] h = b(c[0]), n = b(c[1]);
            assertEquals(byteIndexOf(h, n), StringZilla.indexOf(h, n), c[0] + " / " + c[1]);
        }
    }

    @Test
    void indexOf_offHeap_isExactAndZeroCopy() {
        byte[] h = b("the quick brown fox"), n = b("brown");
        try (Arena a = Arena.ofConfined()) {
            MemorySegment hs = a.allocate(h.length);
            MemorySegment.copy(h, 0, hs, JAVA_BYTE, 0, h.length);
            MemorySegment ns = a.allocate(n.length);
            MemorySegment.copy(n, 0, ns, JAVA_BYTE, 0, n.length);
            assertEquals(byteIndexOf(h, n), StringZilla.indexOf(hs, ns));
        }
    }

    @Test
    void lastIndexOf_matchesReference() {
        byte[] h = b("the the the"), n = b("the");
        assertEquals(8, StringZilla.lastIndexOf(h, n));
        assertEquals(-1, StringZilla.lastIndexOf(h, b("zzz")));
    }

    @Test
    void equal_and_compare() {
        assertTrue(StringZilla.equal(b("abc"), b("abc")));
        assertFalse(StringZilla.equal(b("abc"), b("abd")));
        assertFalse(StringZilla.equal(b("abc"), b("abcd")));
        assertTrue(StringZilla.compare(b("abc"), b("abd")) < 0);
        assertTrue(StringZilla.compare(b("abd"), b("abc")) > 0);
        assertEquals(0, StringZilla.compare(b("abc"), b("abc")));
        assertTrue(StringZilla.compare(b("ab"), b("abc")) < 0);
    }

    @Test
    void hash_isDeterministicAndSeedSensitive() {
        byte[] d = b("the quick brown fox jumps over the lazy dog");
        assertEquals(StringZilla.hash(d), StringZilla.hash(d));
        assertEquals(StringZilla.hash(d, 42), StringZilla.hash(d, 42));
        assertNotEquals(StringZilla.hash(d, 0), StringZilla.hash(d, 42));
    }

    @Test
    void hasher_streaming_equalsOneShot() {
        byte[] d = b("the quick brown fox jumps over the lazy dog");
        try (StringZilla.Hasher h = new StringZilla.Hasher(7)) {
            h.update(java.util.Arrays.copyOfRange(d, 0, 10));
            h.update(java.util.Arrays.copyOfRange(d, 10, d.length));
            assertEquals(StringZilla.hash(d, 7), h.digest());
            assertEquals(h.digest(), h.digest());
        }
    }

    @Test
    void byteSum_matchesManual() {
        byte[] d = b("StringZilla ÿþ");
        long expected = 0;
        for (byte x : d) expected += (x & 0xFF);
        assertEquals(expected, StringZilla.byteSum(d));
    }

    @Test
    void byteset_addContainsAndSearch() {
        StringZilla.Byteset set = StringZilla.Byteset.of(b("aeiou"));
        assertTrue(set.contains((byte) 'e'));
        assertFalse(set.contains((byte) 'z'));
        byte[] h = b("rhythm xy z aei");
        assertEquals(12, StringZilla.indexOfAny(h, set)); // first vowel 'a'
        assertEquals(14, StringZilla.lastIndexOfAny(h, set)); // last vowel 'i'
    }

    @Test
    void metadata_isPopulated() {
        assertFalse(StringZilla.backend().isEmpty());
        assertTrue(Character.isDigit(StringZilla.version().charAt(0)));
    }

    /** Zero-copy adapter: hash a Lucene BytesRef's backing bytes with no copy (heap slice via critical). */
    @Test
    void luceneBytesRef_zeroCopyHashMatches() {
        byte[] data = b("lucene term value");
        BytesRef ref = new BytesRef(data);
        MemorySegment slice = MemorySegment.ofArray(ref.bytes).asSlice(ref.offset, ref.length);
        assertEquals(StringZilla.hash(data, 1), StringZilla.hash(slice, 1));
    }

    // region Codepoints

    @Test
    void countRunes_matchesCodePointCount() {
        String s = "héllo, 世界 🦖";
        assertEquals(s.codePointCount(0, s.length()), StringZilla.countRunes(b(s)));
    }

    @Test
    void decodeAll_matchesCodePoints() {
        String s = "AÉ世🦖";
        assertArrayEquals(s.codePoints().toArray(), StringZilla.decodeAll(b(s)));
    }

    @Test
    void seekRune_findsOffsets() {
        byte[] u = b("aé世"); // a=1B, é=2B, 世=3B
        assertEquals(0, StringZilla.seekRune(u, 0));
        assertEquals(1, StringZilla.seekRune(u, 1));
        assertEquals(3, StringZilla.seekRune(u, 2));
        assertEquals(-1, StringZilla.seekRune(u, 3));
    }

    // endregion

    // region Segmentation

    @Test
    void segment_words() {
        byte[] u = b("the quick fox");
        long[] starts = new long[32];
        long[] lengths = new long[32];
        long[] consumed = new long[1];
        int count = StringZilla.segment(u, 0, u.length, StringZilla.SegmentKind.WORDS, starts, lengths, consumed);
        var texts = new java.util.ArrayList<String>();
        for (int i = 0; i < count; i++)
            texts.add(new String(u, (int) starts[i], (int) lengths[i], StandardCharsets.UTF_8));
        assertTrue(texts.contains("the"));
        assertTrue(texts.contains("quick"));
        assertTrue(texts.contains("fox"));
    }

    @Test
    void segment_graphemes() {
        byte[] u = b("éllo"); // 'e' + U+0301 combining acute => 4 grapheme clusters
        long[] starts = new long[32];
        long[] lengths = new long[32];
        long[] consumed = new long[1];
        int count = StringZilla.segment(u, 0, u.length, StringZilla.SegmentKind.GRAPHEMES, starts, lengths, consumed);
        assertEquals(4, count); // four clusters regardless of NFC/NFD normalization
        assertEquals(0, starts[0]);
        long covered = 0;
        for (int i = 0; i < count; i++) covered += lengths[i];
        assertEquals(u.length, covered); // segments cover the whole input
    }

    // endregion

    // region Case Folding and Uncased

    @Test
    void caseFold_fullFolding() {
        assertEquals("ss", new String(StringZilla.caseFold(b("ß")), StandardCharsets.UTF_8));
        assertEquals("hello", new String(StringZilla.caseFold(b("HELLO")), StandardCharsets.UTF_8));
    }

    @Test
    void uncasedIndexOf_caseless() {
        var m = StringZilla.uncasedIndexOf(b("Hello WÖRLD"), b("wörld"));
        assertEquals(6, m.offset());
        assertTrue(m.matchedLength() > 0);
        assertEquals(-1, StringZilla.uncasedIndexOf(b("Hello"), b("xyz")).offset());
    }

    @Test
    void uncasedNeedle_reusable() {
        try (var needle = new StringZilla.UncasedNeedle(b("fox"))) {
            assertEquals(4, needle.indexIn(b("the FOX")).offset());
            assertEquals(0, needle.indexIn(b("Fox!")).offset());
            assertEquals(-1, needle.indexIn(b("cat")).offset());
        }
    }

    @Test
    void uncasedCompare_ignoresCase() {
        assertEquals(0, StringZilla.uncasedCompare(b("HeLLo"), b("hello")));
        assertTrue(StringZilla.uncasedCompare(b("apple"), b("BANANA")) < 0);
    }

    // endregion

    // region Hash Extras, SHA-256 and Memory

    @Test
    void hashMultiseed_matchesPerSeed() {
        byte[] d = b("multiseed test");
        long[] seeds = {0, 1, 42, 1000};
        long[] multi = StringZilla.hashMultiseed(d, seeds);
        assertEquals(seeds.length, multi.length);
        for (int i = 0; i < seeds.length; i++) assertEquals(StringZilla.hash(d, seeds[i]), multi[i]);
    }

    @Test
    void sha256_matchesJdk() throws Exception {
        byte[] d = b("the quick brown fox");
        byte[] expected = java.security.MessageDigest.getInstance("SHA-256").digest(d);
        assertArrayEquals(expected, StringZilla.Sha256.hashData(d));
        try (var h = new StringZilla.Sha256()) {
            h.update(java.util.Arrays.copyOfRange(d, 0, 5));
            h.update(java.util.Arrays.copyOfRange(d, 5, d.length));
            assertArrayEquals(expected, h.digest());
        }
    }

    @Test
    void fillRandom_deterministic() {
        byte[] a = new byte[64], bb = new byte[64], c = new byte[64];
        StringZilla.fillRandom(a, 12345);
        StringZilla.fillRandom(bb, 12345);
        StringZilla.fillRandom(c, 999);
        assertArrayEquals(a, bb);
        assertFalse(java.util.Arrays.equals(a, c));
    }

    @Test
    void lookup_appliesTable() {
        byte[] lut = new byte[256];
        for (int i = 0; i < 256; i++) lut[i] = (byte) (i + 1);
        byte[] src = {1, 2, 3, (byte) 254};
        byte[] dst = new byte[src.length];
        StringZilla.lookup(dst, src, lut);
        assertArrayEquals(new byte[] {2, 3, 4, (byte) 255}, dst);
    }

    // endregion

    // region Normalization

    @Test
    void normalize_nfc_composes() {
        byte[] decomposed = {'e', (byte) 0xCC, (byte) 0x81}; // 'e' + U+0301
        assertArrayEquals(
                new byte[] {(byte) 0xC3, (byte) 0xA9},
                StringZilla.normalize(decomposed, StringZilla.NormalForm.NFC)); // é U+00E9
    }

    @Test
    void normalize_nfd_decomposes() {
        byte[] precomposed = {(byte) 0xC3, (byte) 0xA9}; // é U+00E9
        assertArrayEquals(
                new byte[] {'e', (byte) 0xCC, (byte) 0x81},
                StringZilla.normalize(precomposed, StringZilla.NormalForm.NFD));
    }

    @Test
    void isNormalized_detectsForm() {
        assertTrue(StringZilla.isNormalized(new byte[] {(byte) 0xC3, (byte) 0xA9}, StringZilla.NormalForm.NFC));
        assertFalse(StringZilla.isNormalized(new byte[] {'e', (byte) 0xCC, (byte) 0x81}, StringZilla.NormalForm.NFC));
    }

    // endregion

    // region Collections

    @Test
    void argSort_lexicographic() {
        var items = java.util.List.of(b("banana"), b("apple"), b("cherry"));
        assertArrayEquals(new long[] {1, 0, 2}, StringZilla.argSort(items));
        assertArrayEquals(new long[] {2, 0, 1}, StringZilla.argSort(items, true, 0, false));
    }

    @Test
    void argSort_uncased() {
        var items = java.util.List.of(b("Banana"), b("apple"));
        assertArrayEquals(new long[] {1, 0}, StringZilla.argSort(items, false, 0, true));
    }

    @Test
    void argSort_topK() {
        var items = java.util.List.of(b("d"), b("a"), b("c"), b("b"));
        assertArrayEquals(new long[] {1, 3}, StringZilla.argSort(items, false, 2, false));
    }

    @Test
    void intersect_findsMatches() {
        var a = java.util.List.of(b("a"), b("b"), b("c"));
        var bb = java.util.List.of(b("b"), b("c"), b("d"));
        long[] first = new long[3];
        long[] second = new long[3];
        int count = StringZilla.intersect(a, bb, first, second);
        var pairs = new java.util.HashSet<java.util.List<Long>>();
        for (int i = 0; i < count; i++) pairs.add(java.util.List.of(first[i], second[i]));
        assertEquals(2, count);
        assertTrue(pairs.contains(java.util.List.of(1L, 0L))); // "b"
        assertTrue(pairs.contains(java.util.List.of(2L, 1L))); // "c"
    }

    @Test
    void argSort_bufferSegments() {
        byte[] text = b("banana\napple\ncherry");
        long[] starts = {0, 7, 13};
        long[] lengths = {6, 5, 6};
        long[] order = new long[3];
        int count = StringZilla.argSort(text, starts, lengths, order, false, 0, false);
        assertEquals(3, count);
        assertArrayEquals(new long[] {1, 0, 2}, order); // apple, banana, cherry
    }
    // endregion

    // region Iteration

    private static String str(MemorySegment segment) {
        return new String(segment.toArray(JAVA_BYTE), StandardCharsets.UTF_8);
    }

    private static java.util.List<String> toStrings(Iterable<MemorySegment> segments) {
        var out = new java.util.ArrayList<String>();
        for (MemorySegment segment : segments) out.add(str(segment));
        return out;
    }

    @Test
    void runes_matchesCodePoints() {
        String text = "héllo 世界 🦖";
        assertArrayEquals(
                text.codePoints().toArray(), StringZilla.runes(b(text)).toArray());
    }

    @Test
    void words_tilesInput() {
        byte[] text = b("the quick fox");
        long covered = 0;
        for (MemorySegment word : StringZilla.words(text)) covered += word.byteSize();
        assertEquals(text.length, covered); // tiling covers the whole input
        assertTrue(toStrings(StringZilla.words(text)).contains("quick"));
    }

    @Test
    void graphemes_matchCursorAndIterable() {
        byte[] text = b("éllo!");
        int viaIterable = toStrings(StringZilla.graphemes(text)).size();
        int viaCursor = 0;
        var cursor = StringZilla.graphemes(text).cursor();
        while (cursor.next()) viaCursor++;
        assertEquals(viaCursor, viaIterable);
    }

    // endregion

    // region Splitting

    @Test
    void split_keepsEmptySegments() {
        assertEquals(java.util.List.of("a", "bb", "", "c"), toStrings(StringZilla.split(b("a,bb,,c"), b(","))));
    }

    @Test
    void split_skipEmpty() {
        assertEquals(
                java.util.List.of("a", "b"),
                toStrings(StringZilla.split(b("a,,b,"), b(",")).skipEmpty()));
    }

    @Test
    void split_maxSplit() {
        assertEquals(
                java.util.List.of("a", "b", "c,d"),
                toStrings(StringZilla.split(b("a,b,c,d"), b(",")).withMaxSplit(2)));
    }

    @Test
    void split_keepSeparator() {
        assertEquals(
                java.util.List.of("a,", "b,", "c"),
                toStrings(StringZilla.split(b("a,b,c"), b(",")).keepSeparator()));
    }

    @Test
    void rsplit_reverseOrder() {
        assertEquals(java.util.List.of("c", "b", "a"), toStrings(StringZilla.rsplit(b("a,b,c"), b(","))));
    }

    @Test
    void rsplit_maxSplit() {
        assertEquals(
                java.util.List.of("d", "a,b,c"),
                toStrings(StringZilla.rsplit(b("a,b,c,d"), b(",")).withMaxSplit(1)));
    }

    @Test
    void splitAny_onByteset() {
        var separators = new StringZilla.Byteset().addAll(b("-_"));
        assertEquals(java.util.List.of("a", "b", "c"), toStrings(StringZilla.splitAny(b("a-b_c"), separators)));
    }

    @Test
    void splitNewlines_yieldsLines() {
        assertEquals(java.util.List.of("a", "bb", "c"), toStrings(StringZilla.splitNewlines(b("a\nbb\nc"))));
    }

    @Test
    void splitWhitespaces_withSeparatorsRoundTrips() {
        var rebuilt = new StringBuilder();
        for (MemorySegment part :
                StringZilla.splitWhitespaces(b("the quick fox")).withSeparators()) rebuilt.append(str(part));
        assertEquals("the quick fox", rebuilt.toString());
    }

    @Test
    void matches_findsAllOffsets() {
        assertArrayEquals(
                new long[] {0, 2, 4, 6},
                StringZilla.matches(b("abababab"), b("ab")).toArray());
    }

    @Test
    void matches_overlappingVsNot() {
        assertEquals(2, StringZilla.matches(b("aaaa"), b("aa")).toArray().length);
        assertEquals(3, StringZilla.matches(b("aaaa"), b("aa")).overlapping().toArray().length);
    }

    @Test
    void uncasedMatches_caseInsensitive() {
        var offsets = StringZilla.uncasedMatches(b("Hello hello HELLO"), b("hello")).stream()
                .map(StringZilla.Match::offset)
                .toList();
        assertEquals(java.util.List.of(0L, 6L, 12L), offsets);
    }

    @Test
    void partition_splitsAtFirst() {
        var partition = StringZilla.partition(b("key=value=tail"), b("="));
        assertEquals("key", str(partition.before()));
        assertEquals("=", str(partition.separator()));
        assertEquals("value=tail", str(partition.after()));
    }

    @Test
    void rpartition_splitsAtLast() {
        var partition = StringZilla.rpartition(b("a/b/c"), b("/"));
        assertEquals("a/b", str(partition.before()));
        assertEquals("/", str(partition.separator()));
        assertEquals("c", str(partition.after()));
    }

    @Test
    void rsplit_keepSeparatorKeepsLeading() {
        assertEquals(
                java.util.List.of(",c", ",b", "a"),
                toStrings(StringZilla.rsplit(b("a,b,c"), b(",")).keepSeparator()));
    }

    @Test
    void split_maxSplitZeroYieldsWhole() {
        assertEquals(
                java.util.List.of("a,b,c"),
                toStrings(StringZilla.split(b("a,b,c"), b(",")).withMaxSplit(0)));
    }

    @Test
    void splitWhitespaces_skipEmpty() {
        assertEquals(
                java.util.List.of("the", "quick"),
                toStrings(StringZilla.splitWhitespaces(b("  the   quick  ")).skipEmpty()));
    }

    @Test
    void splitDelimiters_onlySeparators() {
        assertEquals(
                java.util.List.of("-", "."),
                toStrings(StringZilla.splitDelimiters(b("a-b.c")).onlySeparators()));
    }

    @Test
    void words_refillsAcrossBatches() {
        var builder = new StringBuilder();
        for (int i = 0; i < 200; i++) builder.append('w').append(i).append(' ');
        byte[] text = b(builder.toString());
        int count = 0;
        var cursor = StringZilla.words(text).cursor();
        while (cursor.next()) if (cursor.length() > 0 && text[(int) cursor.startOffset()] == (byte) 'w') count++;
        assertEquals(200, count); // forces multiple 64-entry batch refills on the JVM side
    }

    @Test
    void split_overNativeSegmentIsZeroCopy() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment text = arena.allocate(6);
            MemorySegment.copy(b("a,bb,c"), 0, text, JAVA_BYTE, 0, 6);
            MemorySegment separator = arena.allocate(1);
            separator.set(JAVA_BYTE, 0, (byte) ',');

            var parts = new java.util.ArrayList<String>();
            for (MemorySegment part : StringZilla.split(text, separator)) parts.add(str(part));
            assertEquals(java.util.List.of("a", "bb", "c"), parts);

            // The yielded slices point into the caller's own native memory — no copy was made.
            var cursor = StringZilla.split(text, separator).cursor();
            assertTrue(cursor.next());
            assertEquals(text.address(), cursor.current().address());
        }
    }

    @Test
    void words_overNativeSegment() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment text = arena.allocate(13);
            MemorySegment.copy(b("the quick fox"), 0, text, JAVA_BYTE, 0, 13);
            var words = new java.util.ArrayList<String>();
            for (MemorySegment word : StringZilla.words(text)) words.add(str(word));
            assertTrue(words.contains("quick"));
        }
    }

    @Test
    void matches_overNativeSegment() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment haystack = arena.allocate(8);
            MemorySegment.copy(b("abababab"), 0, haystack, JAVA_BYTE, 0, 8);
            MemorySegment needle = arena.allocate(2);
            MemorySegment.copy(b("ab"), 0, needle, JAVA_BYTE, 0, 2);
            assertArrayEquals(
                    new long[] {0, 2, 4, 6},
                    StringZilla.matches(haystack, needle).toArray());
        }
    }

    @Test
    void uncasedMatches_overlappingStaysCodepointAligned() {
        assertEquals(
                3,
                StringZilla.uncasedMatches(b("AAAA"), b("aa")).overlapping().stream()
                        .count());
        var offsets = StringZilla.uncasedMatches(b("ÄÄÄ"), b("ä")).overlapping().stream()
                .map(StringZilla.Match::offset)
                .toList();
        assertEquals(java.util.List.of(0L, 2L, 4L), offsets); // each Ä is two bytes
    }

    // endregion

}
