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

## Case Folding and Normalization

Case folding fills a gap: the JDK ships no Unicode case folding.

```java
byte[] folded = StringZilla.caseFold(text);                             // ß → "ss"
StringZilla.Match match = StringZilla.uncasedIndexOf(haystack, needle); // caseless search
byte[] composed = StringZilla.normalize(text, StringZilla.NormalForm.NFC);  // cf. java.text.Normalizer
```

## Sorting and Set Operations

```java
long[] order = new long[items.size()];
int sorted = StringZilla.argSort(items, order, false, 0, false); // stable permutation over List<byte[]>

long[] firstPositions = new long[Math.min(left.size(), right.size())];
long[] secondPositions = new long[firstPositions.length];
int matches = StringZilla.intersect(left, right, firstPositions, secondPositions); // matching index pairs
```

## Zero-Copy with Apache Lucene

Lucene terms are UTF-8 already (`BytesRef`), so StringZilla operates on the backing bytes with no copy.
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
