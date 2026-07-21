# StringZilla 🦖 for Java

Zero-copy Java bindings for StringZilla via the Foreign Function & Memory API (FFM, JDK 22+), with no JNI.
SIMD-accelerated search, comparison, hashing, UTF-8 segmentation, case-folding, normalization, and sorting, operating directly on UTF-8 bytes.

## Installation

```xml
<dependency>
  <groupId>com.github.ashvardanian</groupId>
  <artifactId>stringzilla</artifactId>
  <version>4.6.2</version>
</dependency>
```

The bindings require JDK 22+ and native access enabled at launch.

```sh
java --enable-native-access=ALL-UNNAMED ...
```

```java
import com.stringzilla.StringZilla;
import static java.nio.charset.StandardCharsets.UTF_8;

byte[] text = "the quick brown fox".getBytes(UTF_8);
long position = StringZilla.indexOf(text, "brown".getBytes(UTF_8)); // first match, as a UTF-8 byte offset
long digest = StringZilla.hash(text);                               // fast 64-bit AES hash
```

All offsets are UTF-8 byte offsets, not UTF-16 char indices.
Value operations (`hash`, `equal`, `compare`) read on-heap `byte[]` in place via `critical` downcalls, while large off-heap buffers (memory-mapped files, columnar data) are fully zero-copy via `MemorySegment`.
Buffer-filling operations write into caller-provided arrays and return a count, so steady-state usage allocates nothing.

## Searching and Counting

```java
long position = StringZilla.indexOf(haystack, needle);  // first match, or -1 (cf. String.indexOf, on bytes)
long runeCount = StringZilla.countRunes(text);          // cf. String.codePointCount
```

## Comparison and Equality

```java
boolean same = StringZilla.equal(left, right);      // cf. Arrays.equals
int ordering = StringZilla.compare(left, right);    // -1 / 0 / 1
```

## Hashing and Checksums

```java
long digest = StringZilla.hash(data, 42);

try (StringZilla.Hasher hasher = new StringZilla.Hasher(42)) { // streaming
    hasher.update(firstChunk).update(secondChunk);
    long streamed = hasher.digest();
}

byte[] sha = StringZilla.Sha256.hashData(data); // cf. MessageDigest "SHA-256"
```

## UTF-8 Codepoints and Segmentation

The segmentation primitive is allocation-free: it fills the caller's arrays and returns the count.

```java
int decoded = StringZilla.decode(text, codepoints); // fill an int[]; ill-formed -> U+FFFD

long[] starts = new long[64];
long[] lengths = new long[64];
long[] consumed = new long[1];
int count = StringZilla.segment(text, 0, text.length, StringZilla.SegmentKind.WORDS, starts, lengths, consumed); // UAX-29
for (int segment = 0; segment < count; segment++)
    use(text, (int) starts[segment], (int) lengths[segment]);
```

Codepoints count scalar values, not bytes or UTF-16 chars.
Segmentation *tiles* the text — every byte belongs to exactly one segment — and a grapheme cluster can span several codepoints.

```java
StringZilla.countRunes("你好世界".getBytes(UTF_8)); // 4 codepoints
StringZilla.countRunes("Hello🌍".getBytes(UTF_8));  // 6 — the astral emoji is one scalar

for (MemorySegment cluster : StringZilla.graphemes("👍🏽🇺🇸".getBytes(UTF_8))) use(cluster);   // 👍🏽 (emoji + skin tone), 🇺🇸 (flag)
for (MemorySegment word : StringZilla.words("Hello, 世界".getBytes(UTF_8))) use(word);      // Latin run, then CJK run
```

## Splitting and Iteration

Every factory takes an on-heap `byte[]` or any `MemorySegment` (e.g. an off-heap Spark `UTF8String`, a direct `ByteBuffer`, or an mmap'd file).
The word, grapheme, sentence, line-break, and separator-token iterators are always zero-copy and batch 64 boundaries per native call.
The substring and byte-set splits (`split`/`rsplit`/`splitAny`) and the match iterators scan native segments in place; for on-heap inputs they copy the text off-heap once per `cursor()`, because pointer-returning finds need a stable address that the JVM does not expose for heap memory.
The `Iterable`/`Stream` adapters are the ergonomic path; a `cursor()` (a `boolean next()` plus `startOffset()`/`length()`) is the zero-allocation escape hatch.

```java
StringZilla.runes(text).forEach(this::use);                       // IntStream of codepoints
for (MemorySegment word : StringZilla.words(text)) use(word);     // also graphemes/sentences/lineBreaks

for (MemorySegment field : StringZilla.split(line, comma)) use(field);          // cf. String.split, zero-copy
for (MemorySegment part : StringZilla.rsplit(path, slash).withMaxSplit(1)) use(part); // from the end
for (MemorySegment token : StringZilla.splitAny(text, separators)) use(token);  // any byte in a Byteset
for (MemorySegment line : StringZilla.splitWhitespaces(text).skipEmpty()) use(line); // collapse runs

long[] offsets = StringZilla.matches(haystack, needle).toArray(); // every offset; .overlapping() for overlaps
StringZilla.uncasedMatches(haystack, needle).stream().forEach(this::use); // caseless; Match.offset/length

StringZilla.Partition parts = StringZilla.partition(text, equals); // cf. Python str.partition
```

The zero-allocation cursor, e.g. to sort a file's lines without materializing them:

```java
StringZilla.TokenSplitter splitter = StringZilla.splitNewlines(buffer).cursor();
while (splitter.next()) record(splitter.startOffset(), splitter.length());
```

## Case Folding and Normalization

Case folding fills a gap: the JDK ships no Unicode case folding.

```java
byte[] folded = StringZilla.caseFold("Straße".getBytes(UTF_8));         // → "strasse" (ß folds to ss)
StringZilla.Match match = StringZilla.uncasedIndexOf(haystack, needle); // caseless search

byte[] composed = StringZilla.normalize("é".getBytes(UTF_8), StringZilla.NormalForm.NFC);   // "e" + U+0301 → "é"
byte[] ascii = StringZilla.normalize("ﬁ".getBytes(UTF_8), StringZilla.NormalForm.NFKC);     // ligature "ﬁ" → "fi"
```

## Sorting and Set Operations

```java
long[] order = new long[items.size()];
int sorted = StringZilla.argSort(items, order, false, 0, false); // stable permutation over List<byte[]>

long[] lineOrder = new long[lineCount];
StringZilla.argSort(buffer, starts, lengths, lineOrder, false, 0, false); // sort one buffer's segments

long[] firstPositions = new long[Math.min(left.size(), right.size())];
long[] secondPositions = new long[firstPositions.length];
int matches = StringZilla.intersect(left, right, firstPositions, secondPositions); // matching index pairs
```

## Zero-Copy and Memory

Every operation accepts an on-heap `byte[]` or any `MemorySegment`.
Whether the input is read in place or copied once is determined by what the native kernel returns, not by the binding.

Value- and offset-returning operations are always zero-copy, on heap or off-heap.
They are linked with the FFM *critical* option, so the JVM lets the kernel read the array in place with no thread-state transition.
This covers `hash`, `byteSum`, `equal`, `compare`, and the `Hasher`; codepoints (`runes`, `decode`); segmentation (`words`, `graphemes`, `sentences`, `lineBreaks`) and the separator-token splits (`splitNewlines`, `splitWhitespaces`, `splitDelimiters`); `caseFold` and `normalize`; and `argSort` / `intersect`.

Pointer-returning search copies an on-heap input once.
`indexOf` / `lastIndexOf` / `indexOfAny`, the substring and byte-set splits (`split`, `rsplit`, `splitAny`), `matches`, `uncasedMatches`, and `partition` locate a match by pointer, and the JVM does not expose the address of pinned heap memory — so an on-heap `byte[]` (or a heap-backed `MemorySegment`, e.g. a Lucene `BytesRef`) is copied off-heap once before scanning.
That is one linear copy of the haystack per call, or per `cursor()` for the iterators, never per match.
Pass a native `MemorySegment` (an off-heap `Arena`, a direct `ByteBuffer`, an `MMapDirectory` slice, or a Spark Tungsten row) and these run fully in place with no copy.

| Input                                  | Value / offset ops | Pointer search |
| -------------------------------------- | ------------------ | -------------- |
| on-heap `byte[]`                       | zero-copy          | copies once    |
| heap `MemorySegment` (e.g. `BytesRef`) | zero-copy          | copies once    |
| native `MemorySegment` (off-heap)      | zero-copy          | zero-copy      |

The copy exists only because the JVM hides the address of pinned heap memory; .NET exposes it, so the C# binding scans `byte[]` with no copy.
For repeated pointer searches over the same large on-heap buffer, wrap it in a native `MemorySegment` once and reuse it.

## Zero-Copy with Apache Lucene

Lucene terms are UTF-8 already (`BytesRef`), so StringZilla reads the backing bytes directly.
Value operations such as `hash` and `compare` run with no copy; pointer searches copy a heap `BytesRef` once (see [Zero-Copy and Memory](#zero-copy-and-memory)).
This is ideal in custom codecs, comparators, or analysis components.

```java
import org.apache.lucene.util.BytesRef;
import java.lang.foreign.MemorySegment;

BytesRef term = termsEnum.term(); // {bytes, offset, length}
MemorySegment view = MemorySegment.ofArray(term.bytes).asSlice(term.offset, term.length); // zero-copy view
long termHash = StringZilla.hash(view, 0);
long position = StringZilla.indexOf(view, "prefix".getBytes(UTF_8));
```

For index data read through `MMapDirectory`, the bytes are already off-heap `MemorySegment`s, so they can be passed straight in for fully zero-copy scans.

## Zero-Copy with Apache Spark

Spark's `UTF8String` (Tungsten) is UTF-8 and exposes its backing memory.
Wrap it into a `MemorySegment` to run StringZilla inside Catalyst expressions or a native execution engine.

```java
import org.apache.spark.unsafe.types.UTF8String;
import java.lang.foreign.MemorySegment;

UTF8String value = row.getUTF8String(column);
MemorySegment view = (value.getBaseObject() == null)
        ? MemorySegment.ofAddress(value.getBaseOffset()).reinterpret(value.numBytes())  // off-heap (Tungsten)
        : MemorySegment.ofArray((byte[]) value.getBaseObject())                         // on-heap
                .asSlice(value.getBaseOffset() - 16, value.numBytes());                 // 16 = Unsafe.ARRAY_BYTE_BASE_OFFSET
long digest = StringZilla.hash(view, 0);
```

`UTF8String` lives in `org.apache.spark.unsafe`, a low-level surface best used at the Catalyst-expression or native-engine level rather than from ordinary UDFs.
A plain UDF receives a decoded `java.lang.String`, which is UTF-16 and would transcode.

## Runtime Dispatch

The bundled native library auto-selects the best SIMD backend for the host CPU at load time.

```java
System.out.println(StringZilla.backend()); // e.g. "serial,haswell,skylake,icelake"
System.out.println(StringZilla.version());
```
