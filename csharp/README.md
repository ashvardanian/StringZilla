# StringZilla 🦖 for C# / .NET

Zero-copy .NET bindings for StringZilla.
SIMD-accelerated search, comparison, hashing, UTF-8 segmentation, case-folding, normalization, and sorting, operating directly on UTF-8 `byte` spans with no UTF-16 transcode and no copy.

## Installation

```sh
dotnet add package StringZilla
```

The package targets .NET 8+ (`net8.0`) and is NativeAOT- and trimming-compatible.
Native libraries for `win-x64`, `linux-x64`, `osx-arm64`, and more are bundled in the package.

```csharp
using StringZilla;

ReadOnlySpan<byte> text = "the quick brown fox"u8;
long position = Sz.IndexOf(text, "brown"u8); // first match, as a UTF-8 byte offset
ulong digest = Sz.Hash(text);                // fast 64-bit AES hash
```

All offsets are UTF-8 byte offsets, not UTF-16 char indices.
The zero-copy surface is `ReadOnlySpan<byte>`; `string` overloads transcode and are a convenience, not the fast path.
Buffer-filling operations write into a caller-provided `Span<T>` and return a count, so steady-state usage allocates nothing.

## Searching and Counting

```csharp
long position = Sz.IndexOf(haystack, needle);     // first match, or -1 (cf. MemoryExtensions.IndexOf)
long last = Sz.LastIndexOf(haystack, needle);
Byteset vowels = Byteset.FromBytes("aeiou"u8);
long anyVowel = Sz.IndexOfAny(haystack, ref vowels); // first byte in the set (cf. SearchValues<byte>)
```

## Comparison and Equality

```csharp
bool same = Sz.Equal(left, right);    // cf. ReadOnlySpan<byte>.SequenceEqual
int ordering = Sz.Compare(left, right); // -1 / 0 / 1 (cf. SequenceCompareTo)
```

## Hashing and Checksums

```csharp
ulong digest = Sz.Hash(data, seed: 42);

using Hasher hasher = new(seed: 42); // streaming
hasher.Update(firstChunk);
hasher.Update(secondChunk);
ulong streamed = hasher.Digest();

byte[] sha = Sha256.HashData(data); // cf. System.Security.Cryptography.SHA256
```

## UTF-8 Codepoints and Segmentation

The segmentation primitive is allocation-free: it fills the caller's spans and returns the count.

```csharp
long runeCount = Sz.CountRunes(text);    // cf. counting System.Text.Rune
int decoded = Sz.Decode(text, codepoints); // fill a Span<int>; ill-formed -> U+FFFD

Span<long> starts = stackalloc long[64];
Span<long> lengths = stackalloc long[64];
int count = Sz.Segment(text, Sz.SegmentKind.Words, starts, lengths, out long consumed); // UAX-29
for (int segment = 0; segment < count; segment++)
    Use(text.Slice((int)starts[segment], (int)lengths[segment]));
// Kinds: Graphemes (cf. StringInfo), Words, Sentences, LineBreaks (UAX-14), Newlines, Whitespaces, Delimiters
```

Codepoints count scalar values, not bytes or UTF-16 chars.
Segmentation *tiles* the text — every byte belongs to exactly one segment — and a grapheme cluster can span several codepoints.

```csharp
Sz.CountRunes("你好世界"u8);                     // 4 codepoints
Sz.CountRunes("Hello🌍"u8);                      // 6 — the astral emoji is one scalar

foreach (var cluster in Sz.EnumerateGraphemes("👍🏽🇺🇸"u8)) Use(cluster); // 2 clusters: 👍🏽 (emoji + skin tone), 🇺🇸 (flag)
foreach (var word in Sz.EnumerateWords("Hello, 世界"u8)) Use(word);      // Latin run, then CJK run
```

## Splitting and Iteration

The iterators are lazy and allocation-free: each is a `ref struct` enumerator yielding zero-copy `ReadOnlySpan<byte>` views, so a `foreach` never touches the heap.
The codepoint and segmentation iterators batch 64 boundaries per native call; the substring and byte-set splits and the match iterators advance one `find` at a time.
Policies are fluent methods on the returned value.

```csharp
foreach (Rune rune in Sz.EnumerateRunes(text)) Use(rune);   // cf. string.EnumerateRunes
foreach (var word in Sz.EnumerateWords(text)) Use(word);    // also Graphemes/Sentences/LineBreaks

foreach (var field in Sz.Split(line, ","u8)) Use(field); // cf. string.Split, but zero-copy
foreach (var part in Sz.RSplit(path, "/"u8).WithMaxSplit(1)) Use(part); // from the end, at most one split
foreach (var token in Sz.SplitAny(text, separators)) Use(token); // split on any byte in a Byteset
foreach (var line in Sz.SplitWhitespaces(text).SkipEmpty()) Use(line); // collapse whitespace runs

foreach (long at in Sz.EnumerateMatches(haystack, "ab"u8)) Use(at); // every offset; .Overlapping() for overlaps
foreach (var m in Sz.EnumerateUncasedMatches(haystack, "ß"u8)) Use(m); // caseless; m.Offset, m.Length

var (before, separator, after) = Sz.Partition(text, "="u8); // cf. Python str.partition
```

## Case Folding and Normalization

Case folding fills a gap: .NET has no public Unicode case-folding API.

```csharp
byte[] folded = Sz.CaseFold("Straße"u8);              // -> "strasse"
long position = Sz.UncasedIndexOf(haystack, "WÖRLD"u8, out long matched); // caseless search

using UncasedNeedle needle = new("fox"u8); // reuse across haystacks
needle.IndexIn(document, out long matchedLength);

byte[] composed = Sz.Normalize(text, Sz.NormalForm.Nfc); // cf. string.Normalize
int written = Sz.Normalize(text, Sz.NormalForm.Nfc, destination); // allocation-free variant

Sz.Normalize("é"u8, Sz.NormalForm.Nfc);  // "e" + U+0301 -> precomposed "é"
Sz.Normalize("ﬁ"u8, Sz.NormalForm.Nfkc);       // ligature "ﬁ" -> "fi"
```

## Sorting and Set Operations

These take a caller-provided result buffer and return the count; allocating convenience overloads also exist.

```csharp
long[] order = new long[items.Count];
int sorted = Sz.ArgSort(items, order, top: 10, uncased: true); // cf. Array.Sort + StringComparer.Ordinal

Span<long> lineOrder = stackalloc long[lineCount];
Sz.ArgSort(buffer, starts, lengths, lineOrder); // sort one buffer's segments without a byte[][]

long[] firstPositions = new long[Math.Min(left.Count, right.Count)];
long[] secondPositions = new long[firstPositions.Length];
int matches = Sz.Intersect(left, right, firstPositions, secondPositions); // matching index pairs
```

## Zero-Copy in Unity

StringZilla's native byte kernels are a strong fit for Unity's `NativeArray<byte>` (loaded text assets, network buffers), searched and hashed without marshalling.

```csharp
using StringZilla;
using Unity.Collections;

NativeArray<byte> asset = LoadUtf8Asset();
ReadOnlySpan<byte> text = asset.AsReadOnlySpan(); // no copy
long position = Sz.IndexOf(text, "player_id"u8);
ulong identifier = Sz.Hash(text);
```

The managed package targets `net8.0`, which loads on Unity 6.2+ with the CoreCLR/.NET runtime.
On older Unity (Mono/IL2CPP), drop the platform native library into `Assets/Plugins/<platform>/` and call the same API.
The native binary itself is Unity-compatible; only the managed wrapper's target framework is the gate.

## Runtime Dispatch

The bundled library auto-selects the best SIMD backend for the host CPU at load time.

```csharp
Console.WriteLine(Sz.Backend); // e.g. "serial,haswell,skylake,icelake"
Console.WriteLine(Sz.Version);
```
