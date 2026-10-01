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

The segmentation primitive is allocation-free: it fills the caller's span with segment lengths and returns the count.
Segments tile the text, so each starts where the previous one ended.

```csharp
long runeCount = Sz.CountRunes(text);    // cf. counting System.Text.Rune
int decoded = Sz.Decode(text, codepoints); // fill a Span<int>; ill-formed or truncated -> U+FFFD

Span<long> lengths = stackalloc long[64];
int count = Sz.Segment(text, Sz.SegmentKind.Words, lengths); // UAX-29
ReadOnlySpan<byte> rest = text;
foreach (long length in lengths[..count]) {
    Use(rest[..(int)length]);
    rest = rest[(int)length..];
}
// Kinds: Graphemes (cf. StringInfo), Words, Sentences, LineBreaks (UAX-14)
```

Codepoints count scalar values, not bytes or UTF-16 chars.
Segmentation _tiles_ the text — every byte belongs to exactly one segment — and a grapheme cluster can span several codepoints.

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

var (before, separator, after) = Sz.Partition(text, "="u8); // split at the first "="
```

`WithMaxSplit(n)` stops after `n` separators: `0` yields the whole text, and a negative `n`, the default, splits at every separator.
An empty separator yields the whole text, a trailing separator leaves a final empty segment unless `.SkipEmpty()`, and an empty needle matches at every offset up to and including the end, or at every codepoint boundary when uncased.

## Case Folding and Normalization

Case folding fills a gap: .NET has no public Unicode case-folding API.

```csharp
byte[] folded = Sz.CaseFold("Straße"u8);              // -> "strasse"
long position = Sz.UncasedIndexOf(haystack, "WÖRLD"u8, out long matched); // caseless search

using UncasedNeedle needle = new("fox"u8); // prepared once, reusable across haystacks and threads
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

A failed native call throws `StatusException`, whose `Status` is the C `sz_status_t` code and `StatusName` its enumerator name.

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

## Capabilities and Runtime Selection

A `Device` is the host CPU or one GPU, by its runtime's own ordinal, and reports capability bitmasks with the serial fallback at bit 0.
Every call of this binding dispatches to the best kernel among the CPU's enabled capabilities; a GPU device only reports its masks here.

```csharp
Device cpu = Device.Cpu;
ulong enabled = cpu.CapabilitiesEnabled;                      // what dispatch uses
Console.WriteLine(Sz.CapabilitiesName(enabled));              // like "serial,neon,neonaes,neonsha"
cpu.CapabilitiesEnable(1);                                    // narrow dispatch to the serial kernels
cpu.CapabilitiesEnable(ulong.MaxValue);                       // back to everything this CPU and binary support
ulong gpus = Device.Count(DeviceKind.Metal);                  // throws StatusException without a Metal GPU
ulong metal = new Device(DeviceKind.Metal, 0).CapabilitiesEnabled;
Console.WriteLine(Sz.Version);
```

- `Device.Cpu` is the host CPU, which every build has, and `new Device(kind, ordinal)` is device `ordinal` of `DeviceKind.Cuda`, `Rocm` or `Metal`.
- `Device.Count(kind)` is how many devices of `kind` the process sees: one CPU, or the GPUs its runtime counts.
- `CapabilitiesDetected` is what the device can execute, from CPUID or HWCAP on the CPU.
- `CapabilitiesCompiled` is what this binary contains for devices of that kind, from the ISA probes at build time.
- `CapabilitiesEnabled` is what calls pass, both axes at once; on the CPU it is what dispatch uses, narrowed by `CapabilitiesEnable`, and always contains serial.
- `CapabilitiesEnable(wanted)` makes `wanted` the CPU's enabled set, clamped to both axes, and returns what took effect.
- `ConfigureThread(capabilities)` prepares the calling thread for the kernels of `capabilities`, usually `CapabilitiesEnabled`, once per thread that runs them.
- `Sz.CapabilitiesName(capabilities)` spells a mask as comma-separated capability names.

Reach for `CapabilitiesEnabled` unless you specifically mean one of the raw axes.
`CapabilitiesDetected` describes the machine and says nothing about whether a kernel was compiled in, so a build whose ISA probes failed still reports your CPU's full feature set while containing no SIMD kernels at all.
The CPU's enabled set is process-wide and shared by every thread.
A `StatusException` reports misuse: `sz_missing_gpu_k` for a GPU kind without devices or an ordinal past the last one, and `sz_missing_kernel_k` for `CapabilitiesEnable` or `ConfigureThread` on a GPU, which keeps no such set or thread state.
