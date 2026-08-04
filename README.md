# StringZilla 🦖

![StringZilla banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringZilla-v5.png?raw=true)

Strings are the first fundamental data type every programming language implements in software rather than hardware — the closest CPUs come to a "find substring" instruction is x86's `PCMPISTRI`, which is too slow and too narrow to build a library on, and nothing ships a "compute string hash" instruction at all.
So most string-processing code still looks like `for (i = 0; i < length; ++i) if (text[i] == 'x') …` — a tangle of loops, branches, and per-character lookups, where the surrounding control flow often costs more than the character-level logic itself, whether the text is ASCII or UTF-8 encoded Unicode.
Worse, chewing through one byte or codepoint at a time squanders the hardware: a modern CPU carries dozens of 16-64 byte architectural registers, and hundreds of physical ones to feed out-of-order execution.
StringZilla reaches for those [SIMD][faq-simd] and [SWAR][faq-swar] instructions directly, offering one of the widest, fastest, and most portable collections of text-processing primitives anywhere.

[![StringZilla Python installs](https://static.pepy.tech/personalized-badge/stringzilla?period=total&units=abbreviation&left_color=black&right_color=blue&left_text=StringZilla%20Python%20installs)](https://github.com/ashvardanian/stringzilla)
[![StringZilla Rust installs](https://img.shields.io/crates/d/stringzilla?logo=rust&label=Rust%20installs)](https://crates.io/crates/stringzilla)
![StringZilla code size](https://img.shields.io/github/languages/code-size/ashvardanian/stringzilla)

<!-- Those badges often stay in stale state - greyed out.
Consider enabling them later.
[![Ubuntu status](https://img.shields.io/github/checks-status/ashvardanian/StringZilla/main?checkName=Linux%20CI&label=Ubuntu)](https://github.com/ashvardanian/StringZilla/actions/workflows/release.yml)
[![Windows status](https://img.shields.io/github/checks-status/ashvardanian/StringZilla/main?checkName=Windows%20CI&label=Windows)](https://github.com/ashvardanian/StringZilla/actions/workflows/release.yml)
[![macOS status](https://img.shields.io/github/checks-status/ashvardanian/StringZilla/main?checkName=macOS%20CI&label=macOS)](https://github.com/ashvardanian/StringZilla/actions/workflows/release.yml)
-->

StringZilla is the GodZilla of string libraries, accelerating exact and fuzzy matching, hashing, edit distances, sorting, segmentation, and even random-string generation, with allocation-free lazily-evaluated iterators throughout.

- It can be __3x faster than LibC__ doing substring search on Arm servers, and __9x on Apple Silicon__, where the system `strstr` is weaker.
- It can be __10-70x faster than ICU__, both ICU4C and its Rust successor ICU4X, in UTF-8 handling, case folding, segmentation, and tokenization.
- It can be __over 10x faster than NVIDIA's own libraries__ for on-GPU Levenshtein, NW, and SW edit distances.
- It comes with built-in custom __WebAssembly__ backend for sandboxed browser, DBMS, & LLM environments, custom __RVV__ backend for RISC-V CPUs, __PowerPC__ backend for IBM Power servers, __LoongArch__ for Chinese domestic chips, and more!

Reach for it from your language of choice:

- 🐂 __[C](#c-and-c):__ Upgrade LibC's `<string.h>` to `<stringzilla/stringzilla.h>` in C 99
- 🐉 __[C++](#c-and-c):__ Upgrade STL's `<string>` to `<stringzilla/stringzilla.hpp>` in C++ 11
- 🧮 __[CUDA](include/stringzillas/README.md):__ Process in-bulk with `<stringzillas/stringzillas.cuh>` in CUDA C++ 17
- 🐍 __[Python](#python):__ Upgrade your `str` to faster `Str`
- 🦀 __[Rust](#rust):__ Use the `StringZilla` traits crate
- 🦫 __[Go](#go):__ Use the `StringZilla` cGo module
- 🍎 __[Swift](#swift):__ Use the `String+StringZilla` extension
- 🟨 __[JavaScript](#javascript):__ Use the `StringZilla` library
- 💜 __[C#](#c):__ Zero-copy over `ReadOnlySpan<byte>`, NativeAOT-friendly
- ☕ __[Java](#java):__ Pure FFM API over `MemorySegment`, no JNI
- 🐚 __[Shell][faq-shell]__: Accelerate common CLI tools with `sz-` prefix
- 📚 Researcher? Jump to [Algorithms & Design Decisions](#algorithms--design-decisions)
- 💡 Thinking to contribute? Look for ["good first issues"][first-issues]
- 🤝 And check the [guide](https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md) to set up the environment
- Want more bindings or features? Let [me](https://github.com/ashvardanian) know!

__Who is this for?__

- For data-engineers parsing large datasets, like the [CommonCrawl](https://commoncrawl.org/), [RedPajama](https://github.com/togethercomputer/RedPajama-Data), or [LAION](https://laion.ai/blog/laion-5b/).
- For software engineers optimizing strings in their apps and services.
- For bioinformaticians and search engineers looking for edit-distances for [USearch](https://github.com/unum-cloud/usearch).
- For [DBMS][faq-dbms] devs, optimizing `LIKE`, `ORDER BY`, and `GROUP BY` operations.
- For hardware designers, needing a SWAR baseline for string-processing functionality.
- For students studying SIMD/SWAR applications to non-data-parallel operations.

## Performance

Throughput and timings on two CPUs and one GPU, grouped by operation.
Only the languages that ship a counterpart appear under each heading.
`StringZilla.C` is the C kernel called directly; `StringZilla.Py` is the same kernel through the CPython binding, so the gap between them is the cost of crossing the interpreter boundary.

```
                                                      Xeon4    M5 Pro        H100
─────────────────────────────────────────────────────────────────────────────────
Unicode case-insensitive substring search  (GB/s)
  Python          icu.StringSearch                     0.06      0.15
  StringZilla.C   sz_utf8_uncased_search              13.16      9.30
  StringZilla.Py  sz.utf8_uncased_search              12.40      9.19

Find the first occurrence of a random word, ≅ 5 bytes  (GB/s)
  LibC            strstr                               21.3       3.5
  STL C++         std::string::find                     9.1      10.9
  Python          str.find                              2.5       3.1
  StringZilla.C   sz_find                              21.0      37.1
  StringZilla.Py  sz.find                              18.6      33.4

Split lines separated by \n or \r  (GB/s)
  LibC            strcspn                               9.2       3.8
  STL C++         std::string::find_first_of            1.1       4.1
  Python          re.finditer                          0.32      0.64
  StringZilla.C   sz_find_byteset                      13.8      23.7
  StringZilla.Py  sz.split_byteset_iter                11.2      21.7

Levenshtein distances, ≅ 100 byte DNA, one core  (MCUPS)
  Python          rapidfuzz.process.cdist             4,970    18,370
  StringZilla.C   szs_levenshtein_distances          15,680    22,844   5,980,110
  StringZilla.Py  szs.LevenshteinDistances           14,130    21,930   4,074,700

Needleman-Wunsch scores, ≅ 1 KB DNA, one core  (MCUPS)
  Python          Bio.Align.PairwiseAligner.score       430       870
  StringZilla.C   szs_needleman_wunsch_scores        12,000     1,267     701,760
  StringZilla.Py  szs.NeedlemanWunschScores          10,730     1,060     700,900
```

> Treat these as a first impression, not a benchmark suite.
> The Unicode numbers were obtained on a 128 MB slice of multilingual XLSum; the similarity rows on synthetic DNA strings.
> `Xeon4` is an Intel Sapphire Rapids with GCC and `glibc`, `M5 Pro` an 18-core Apple Silicon with Apple clang and `libc++`, `H100` an Nvidia Hopper GPU.
> The two CPUs therefore differ in standard library as much as in ISA, which is most of the gap in the `strstr`, `std::string::rfind`, and `bytes.translate` rows; the StringZilla rows build from the same source on both.
> These will not reproduce exactly; the links below carry the methodology and the per-library breakdowns.

Most StringZilla modules ship ready-to-run benchmarks for C, C++, Python, and more.
Grab them from `./scripts`, and see [`CONTRIBUTING.md`](CONTRIBUTING.md), [`test/README.md`](test/README.md), and [`bench/README.md`](bench/README.md) for instructions.
For wider head-to-heads against Rust and Python favorites, browse the __[StringWars][stringwars]__ repository.
To inspect collision resistance and distribution shapes for our hashers, see __[HashEvals][hashevals]__.

[stringwars]: https://github.com/ashvardanian/StringWars
[hashevals]: https://github.com/ashvardanian/HashEvals

## Why StringZilla

There are several other excellent libraries with overlapping subsets of operations and somewhat different design philosophies.
LibC obviously provides a good baseline for basic memory operations, but its APIs vary widely in quality, and its Arm implementations often trail its x86 ones.
ICU and ICU4X implement the Unicode standard to a letter, but don't exploit hidden invariants in the Unicode ruleset to vectorize those operations.
RapidFuzz comes with a very good set of string-similarity algorithms and is already well vectorized on CPUs, but leaves batched cross-products symmetries and massive GPU speedups on the table.
xxHash and aHash provide great non-cryptographic hashes, but may not cover all of the hashing use cases, or leverage the wider AES and predicated instructions available on modern CPUs.

Because StringZilla mirrors the familiar standard APIs, adoption is mostly a search-and-replace.

```
                           Standard                  StringZilla
─────────────────────────────────────────────────────────────────────────────────
Python
  Find a substring         "...".find(x)             sz.find("...", x)
  Sort strings             sorted(items)             sz.Strs(items).sorted()
  Split on a separator     "...".split(sep)          sz.Str("...").split(sep)
  Case-fold for matching   "...".casefold()          sz.utf8_uncased_fold("...")
  Streaming SHA-256        hashlib.sha256()          sz.Sha256()

C++
  Find a substring         std::string::find         sz::string::find
  Sort a collection        std::sort of indices      sz::argsort
  Intersect string sets    std::set_intersection     sz::try_intersect
  Hash map, string keys    std::unordered_map<K, V>  sz::hash + sz::equal_to
```

## Functionality

StringZilla is compatible with most modern CPUs, and provides a broad range of functionality.
It's split into 2 layers:

1. StringZilla: single-header C library and C++ wrapper for high-performance string operations.
2. StringZillas: parallel CPU/GPU backends used for large-batch operations and accelerators.

Having a second C++/CUDA layer greatly simplifies the implementation of similarity scoring and fingerprinting functions, which would otherwise require too much error-prone boilerplate code in pure C.
Both layers are designed to be extremely portable:

- [x] across both little-endian and big-endian architectures.
- [x] across 32-bit and 64-bit hardware architectures.
- [x] across operating systems and compilers.
- [x] across ASCII and UTF-8 encoded inputs.

Not all features are available across all bindings.
Consider contributing if you need a feature that's not yet implemented.

|                                | Maturity |   C   |  C++  | Python | Rust  |  JS   | Swift |  Go   |  C#   | Java  |
| :----------------------------- | :------: | :---: | :---: | :----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Substring Search               |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
| Character Set Search           |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
| Sorting & Sequence Operations  |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |   ✅   |   ✅   |
| Set Intersection & Joins       |    🧐     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |   ✅   |   ✅   |
| Lazy Ranges, Compressed Arrays |    🌳     |   ❌   |   ✅   |   ✅    |   ✅   |   ❌   |   ⚪   |   ⚪   |   ✅   |   ✅   |
| One-Shot & Streaming Hashes    |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
| Cryptographic Hashes           |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
| Small String Class             |    🧐     |   ✅   |   ✅   |   ❌    |   ⚪   |   ❌   |   ❌   |   ❌   |   ❌   |   ❌   |
| Random String Generation       |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |   ✅   |   ✅   |
|                                |          |       |       |        |       |       |       |       |       |       |
| Unicode Case Folding           |    🧐     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
| Uncased UTF-8 Search           |    🚧     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
| TR29 Word Boundary Detection   |    🚧     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ⚪   |   ✅   |   ✅   |
| TR29 Grapheme Segmentation     |    🚧     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ⚪   |   ⚪   |   ✅   |   ✅   |
| TR29 Sentence Segmentation     |    🚧     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ⚪   |   ⚪   |   ✅   |   ✅   |
| UAX14 Line-Break Detection     |    🚧     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ⚪   |   ⚪   |   ✅   |   ✅   |
| Unicode Normalization          |    🚧     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
| Codepoint Counting & Indexing  |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |   ✅   |
|                                |          |       |       |        |       |       |       |       |       |       |
| Parallel Similarity Scoring    |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |   ⚪   |   ⚪   |
| Parallel Rolling Fingerprints  |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |   ⚪   |   ⚪   |

> 🌳 parts are used in production.
> 🧐 parts are in beta.
> 🚧 parts are under active development, and are likely to break in subsequent releases.
> ✅ are implemented.
> ⚪ are considered.
> ❌ are not intended.

## Quick Start

Each binding has its own install command, import line, and dedicated guide, all collected in the per-language sections below.
The batch and GPU engines ship separately, as `stringzillas-cpus` and `stringzillas-cuda` on PyPI and the `cpus` and `cuda` crate features; each binding's guide covers the details.

### Python

`pip install stringzilla` &centerdot; guide: [`python/README.md`](python/README.md)

```python
import stringzilla as sz

text = sz.Str("the quick brown fox")
text.find("brown")          # 10
text.split()                # Strs(['the', 'quick', 'brown', 'fox'])
sz.hash("hello")            # fast 64-bit hash
```

The Python package upgrades `str` and `bytes` with SIMD search, sorting, hashing, UTF-8 segmentation, and Unicode case-folding, plus the batch-parallel `stringzillas` engines for edit distances and rolling fingerprints.

### C and C++

Header-only, or pull it in with CMake `FetchContent`, or `find_package(stringzilla)` an installed build &centerdot; guides: [`include/stringzilla/README.md`](include/stringzilla/README.md) and [`include/stringzillas/README.md`](include/stringzillas/README.md)

```c
#include <stringzilla/stringzilla.h>
sz_find(haystack, h_length, "brown", 5); // pointer to the match, or NULL
```

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;
sz::string_view("the quick brown fox").find("brown"); // 10
```

The header-only library covers search, hashing, sorting, comparison, set intersection, memory operations, and lazy UTF-8 segmentation; the bulk and GPU engines for edit distances, alignment scores, and fingerprints live in the companion `stringzillas` distribution.

### Rust

`cargo add stringzilla` &centerdot; guide: [`rust/README.md`](rust/README.md)

```rust
use stringzilla::sz;

assert_eq!(sz::find("the quick brown fox", "brown"), Some(10));
let digest = sz::hash("hello"); // fast 64-bit hash
```

The crate adds SIMD search, sorting, hashing, and UTF-8 segmentation to any `AsRef<[u8]>`, with the optional `stringzillas` engines for batch edit distances and rolling fingerprints.

### JavaScript

`npm install stringzilla` &centerdot; guide: [`javascript/README.md`](javascript/README.md)

```js
import sz from "stringzilla";

sz.find(Buffer.from("the quick brown fox"), Buffer.from("brown")); // => 10n
sz.hash(Buffer.from("hello"));                                     // 64-bit BigInt
```

The Node-API addon runs on Node, Bun, and Deno, exposing zero-copy search, hashing, SHA-256, and Unicode case-folding over `Buffer` objects.

### Swift

Add the Swift Package Manager dependency &centerdot; guide: [`swift/README.md`](swift/README.md)

```swift
import StringZilla

let i = "the quick brown fox".findFirst(substring: "brown") // Index of "brown"
let h = "hello".hash()                                       // fast 64-bit hash
```

The Foundation-free package extends `String` with SIMD search, comparison, hashing, Unicode case-folding, normalization, and word and line segmentation, on Linux and embedded targets as well as Apple platforms.

### Go

`go get github.com/ashvardanian/stringzilla/golang` &centerdot; guide: [`golang/README.md`](golang/README.md)

```go
import sz "github.com/ashvardanian/stringzilla/golang"

sz.Index("the quick brown fox", "brown") // 10
sz.Hash("hello", 0)                      // fast 64-bit hash
```

The cgo module exposes byte-level search, counting, checksums, SHA-256, and UTF-8 case-folding to Go.

### C#

Build from source &centerdot; guide: [`csharp/README.md`](csharp/README.md) &centerdot; __not yet on NuGet__

```csharp
using StringZilla;

Sz.IndexOf("the quick brown fox"u8, "brown"u8); // 10
Sz.Hash("hello"u8);                             // fast 64-bit hash
```

Zero-copy over `ReadOnlySpan<byte>` (and Unity's `NativeArray<byte>`); `net8.0`, NativeAOT-friendly.
Exposes search, hashing, SHA-256, UTF-8 segmentation, case-folding, normalization, sorting, and allocation-free splitting and iteration.

### Java

Build from source with `mvn` &centerdot; guide: [`java/README.md`](java/README.md) &centerdot; __not yet on Maven Central__

```java
import com.stringzilla.StringZilla;

StringZilla.indexOf("the quick brown fox".getBytes(), "brown".getBytes());  // 10
StringZilla.hash("hello".getBytes());                                       // fast 64-bit hash
```

Pure [Foreign Function & Memory API](https://openjdk.org/jeps/454) (JDK 22+), __no JNI__.
Zero-copy over `byte[]` and `MemorySegment` — including Lucene `BytesRef` and Spark `UTF8String` backing memory.
Lazy `Iterable`/`Stream` splitting and iteration, with zero-allocation cursors as the escape hatch.

## Algorithms & Design Decisions

StringZilla aims to optimize some of the slowest string operations.
Some popular operations, however, like equality comparisons and relative order checking, almost always complete on some of the very first bytes in either string.
In such operations vectorization is almost useless, unless huge and very similar strings are considered.
StringZilla implements those operations as well, but won't result in substantial speedups.
Where vectorization stops being effective, parallelism takes over, across two layers:

- StringZilla C library w/out dependencies
- StringZillas parallel extensions:
  - Parallel C++ algorithms built with [ForkUnion](https://github.com/ashvardanian/ForkUnion)
  - Parallel CUDA algorithms for Nvidia GPUs
  - Parallel ROCm algorithms for AMD GPUs 🔜

### Exact Substring Search

Substring search algorithms are generally divided into: comparison-based, automaton-based, and bit-parallel.
Different families are effective for different alphabet sizes and needle lengths.
The more operations are needed per-character - the more effective SIMD would be.
The longer the needle - the more effective the skip-tables are.
StringZilla uses different exact substring search algorithms for different needle lengths and backends:

- When no SIMD is available - SWAR (SIMD Within A Register) algorithms are used on 64-bit words.
- Boyer-Moore-Horspool (BMH) algorithm with Raita heuristic variation for longer needles.
- SIMD backends compare characters at multiple strategically chosen offsets within the needle to reduce degeneracy.

On very short needles, especially 1-4 characters long, brute force with SIMD is the fastest solution.
On mid-length needles, bit-parallel algorithms are effective, as the character masks fit into 32-bit or 64-bit words.
Either way, if the needle is under 64-bytes long, on haystack traversal we will still fetch every CPU cache line.
So the only way to improve performance is to reduce the number of comparisons.

> For 2-byte needles, see `sz_find_2byte_serial_` in `include/stringzilla/find/serial.h`:

https://github.com/ashvardanian/StringZilla/blob/a6402dd71d01a8967a62bcc45a900477e0232f60/include/stringzilla/find/serial.h#L231-L273

Going beyond that, to long needles, Boyer-Moore (BM) and its variants are often the best choice.
It has two tables: the good-suffix shift and the bad-character shift.
Common choice is to use the simplified BMH algorithm, which only uses the bad-character shift table, reducing the pre-processing time.
We do the same for mid-length needles up to 256 bytes long.
That way the stack-allocated shift table remains small.

> For mid-length needles (≤256 bytes), see `sz_find_horspool_upto_256bytes_serial_` in `include/stringzilla/find/serial.h`:

https://github.com/ashvardanian/StringZilla/blob/a6402dd71d01a8967a62bcc45a900477e0232f60/include/stringzilla/find/serial.h#L449-L500

In the C++ Standards Library, the `std::string::find` function uses the BMH algorithm with Raita's heuristic.
Before comparing the entire string, it matches the first, last, and the middle character.
Very practical, but can be slow for repetitive characters.
Both SWAR and SIMD backends of StringZilla have a cheap pre-processing step, where we locate unique characters.
This makes the library a lot more practical when dealing with non-English corpora.

> The offset selection heuristic is implemented in `sz_locate_needle_anomalies_` in `include/stringzilla/find/serial.h`:

https://github.com/ashvardanian/StringZilla/blob/a6402dd71d01a8967a62bcc45a900477e0232f60/include/stringzilla/find/serial.h#L35-L96

All those, still, have $O(hn)$ worst case complexity.
To guarantee $O(h)$ worst case time complexity, the Apostolico-Giancarlo (AG) algorithm adds an additional skip-table.
Preprocessing phase is $O(n + \sigma)$ in time and space.
On traversal, performs from $(h/n)$ to $(3h/2)$ comparisons.
It however, isn't practical on modern CPUs.
The Galil rule is a simpler and more relevant optimization, if many matches must be found.

Other algorithms previously considered and deprecated:

- Apostolico-Giancarlo algorithm for longer needles.
  _Control-flow is too complex for efficient vectorization._
- Shift-Or-based Bitap algorithm for short needles.
  _Slower than SWAR._
- Horspool-style bad-character check in SIMD backends.
  _Effective only for very long needles, and very uneven character distributions between the needle and the haystack._
  _Faster "character-in-set" check needed to generalize._

> § Reading materials.
> [Exact String Matching Algorithms in Java](https://www-igm.univ-mlv.fr/~lecroq/string).
> [SIMD-friendly algorithms for substring searching](http://0x80.pl/articles/simd-strfind.html).

### Exact Multiple Substring Search

Few algorithms for multiple substring search are known.
Most are based on the Aho-Corasick automaton, which is a generalization of the KMP algorithm.
The naive implementation, however:

- Allocates disjoint memory for each Trie node and Automaton state.
- Requires a lot of pointer chasing, limiting speculative execution.
- Has a lot of branches and conditional moves, which are hard to predict.
- Matches text a character at a time, which is slow on modern CPUs.

There are several ways to improve the original algorithm.
One is to use sparse DFA representation, which is more cache-friendly, but would require extra processing to navigate state transitions.

StringZilla does not ship an Aho-Corasick automaton today.
For multi-pattern workloads, the rolling-fingerprint machinery described below covers the near-duplicate and candidate-filtering cases, and [hyperscan](https://github.com/intel/hyperscan) or [pyahocorasick](https://github.com/WojciechMula/pyahocorasick) remain the better fit for large literal dictionaries.

### Levenshtein Edit Distance

Levenshtein distance is the best known edit-distance for strings, that checks, how many insertions, deletions, and substitutions are needed to transform one string to another.
It's extensively used in approximate string-matching, spell-checking, and bioinformatics.

The computational cost of the Levenshtein distance is $O(n * m)$, where $n$ and $m$ are the lengths of the string arguments.
To compute that, the naive approach requires $O(n * m)$ space to store the "Levenshtein matrix", the bottom-right corner of which will contain the Levenshtein distance.
The algorithm producing the matrix has been simultaneously studied/discovered by the Soviet mathematicians Vladimir Levenshtein in 1965, Taras Vintsyuk in 1968, and American computer scientists - Robert Wagner, David Sankoff, Michael J. Fischer in the following years.
Several optimizations are known:

1. __Space Optimization__: The matrix can be computed in $O(min(n,m))$ space, by only storing the last two rows of the matrix.
2. __Divide and Conquer__: Hirschberg's algorithm can be applied to decompose the computation into subtasks.
3. __Automata__: Levenshtein automata can be effective, if one of the strings doesn't change, and is a subject to many comparisons.
4. __Shift-Or__: Bit-parallel algorithms transpose the matrix into a bit-matrix, and perform bitwise operations on it.

The last approach is quite powerful and performant, and is used by the great [RapidFuzz][rapidfuzz] library.
It's less known, than the others, derived from the Baeza-Yates-Gonnet algorithm, extended to bounded edit-distance search by Manber and Wu in 1990s, and further extended by Gene Myers in 1999 and Heikki Hyyro between 2002 and 2004.

StringZilla focuses on a different approach, extensively used in Unum's internal combinatorial optimization libraries.
It doesn't change the number of trivial operations, but performs them in a different order, removing the data dependency, that occurs when computing the insertion costs.
StringZilla __evaluates diagonals instead of rows__, exploiting the fact that all cells within a diagonal are independent, and can be computed in parallel.
We'll store 3 diagonals instead of the 2 rows, and each consecutive diagonal will be computed from the previous two.
Substitution costs will come from the sooner diagonal, while insertion and deletion costs will come from the later diagonal.

<table>
<tr>
<td>
<strong>Row-by-Row Algorithm</strong><br>
Computing row 4:

<pre>
    ∅  A  B  C  D  E
 ∅  0  1  2  3  4  5
 P  1  ░  ░  ░  ░  ░
 Q  2  ■  ■  ■  ■  ■
 R  3  ■  ■  □  →  .
 S  4  .  .  .  .  .
 T  5  .  .  .  .  .
</pre>
</td>
<td>
<strong>Anti-Diagonal Algorithm</strong><br>
Computing diagonal 5:

<pre>
    ∅  A  B  C  D  E
 ∅  0  1  2  3  4  5
 P  1  ░  ░  ■  ■  □
 Q  2  ░  ■  ■  □  ↘
 R  3  ■  ■  □  ↘  .
 S  4  ■  □  ↘  .  .
 T  5  □  ↘  .  .  .
</pre>
</td>
</tr>
<tr>
<td colspan="2">
<strong>Legend:</strong><br>
<code>0,1,2,3...</code> = initialization constants &nbsp;&nbsp;
<code>░</code> = cells processed and forgotten &nbsp;&nbsp;
<code>■</code> = stored cells &nbsp;&nbsp;
<code>□</code> = computing in parallel &nbsp;&nbsp;
<code>→ ↘</code> = movement direction &nbsp;&nbsp;
<code>.</code> = cells to compute later
</td>
</tr>
</table>

This results in much better vectorization for intra-core parallelism and potentially multi-core evaluation of a single request.
Moreover, it's easy to generalize to weighted edit-distances, where the cost of a substitution between two characters may not be the same for all pairs, often used in bioinformatics.

> § Reading materials.
> [Faster Levenshtein Distances with a SIMD-friendly Traversal Order](https://ashvardanian.com/posts/levenshtein-diagonal).

[rapidfuzz]: https://github.com/rapidfuzz/RapidFuzz

### Needleman-Wunsch and Smith-Waterman Scores for Bioinformatics

The field of bioinformatics studies various representations of biological structures.
The "primary" representations are generally strings over sparse alphabets:

- [DNA][faq-dna] sequences, where the alphabet is {A, C, G, T}, ranging from ~100 characters for short reads to 3 billion for the human genome.
- [RNA][faq-rna] sequences, where the alphabet is {A, C, G, U}, ranging from ~50 characters for tRNA to thousands for mRNA.
- [Proteins][faq-protein], where the alphabet is made of 22 amino acids, ranging from 2 characters for [dipeptide][faq-dipeptide] to 35,000 for [Titin][faq-titin], the longest protein.

The shorter the representation, the more often researchers may want to use custom substitution matrices.
Meaning that the cost of a substitution between two characters may not be the same for all pairs.
In the general case the serial algorithm works for arbitrary substitution costs for each of 256×256 possible character pairs.
That lookup table, however, is too large to fit into CPU registers, so StringZilla ships a 32×32 substitution-matrix design, the `error_costs_32x32_t` type in `include/stringzillas/similarities.hpp`, which fits into 1 KB with single-byte "error costs" and stays resident across the diagonal sweep.
That said, most [BLOSUM][faq-blosum] and [PAM][faq-pam] substitution matrices only contain 4-bit values, so they can be packed even further.

[faq-dna]: https://en.wikipedia.org/wiki/DNA
[faq-rna]: https://en.wikipedia.org/wiki/RNA
[faq-protein]: https://en.wikipedia.org/wiki/Protein
[faq-blosum]: https://en.wikipedia.org/wiki/BLOSUM
[faq-pam]: https://en.wikipedia.org/wiki/Point_accepted_mutation
[faq-dipeptide]: https://en.wikipedia.org/wiki/Dipeptide
[faq-titin]: https://en.wikipedia.org/wiki/Titin

### Memory Copying, Fills, and Moves

A lot has been written about the time computers spend copying memory and how that operation is implemented in LibC.
Interestingly, the operation can still be improved, as most Assembly implementations use outdated instructions.
Even performance-oriented STL replacements, like Meta's [Folly v2024.09.23 focus on AVX2](https://github.com/facebook/folly/blob/main/folly/memset.S), and don't take advantage of the new masked instructions in AVX-512 or SVE.

In AVX-512, StringZilla uses non-temporal stores to avoid cache pollution, when dealing with very large strings.
Moreover, it handles the unaligned head and the tails of the `target` buffer separately, ensuring that writes in big copies are always aligned to cache-line boundaries.
That's true for both AVX2 and AVX-512 backends.

StringZilla also contains "drafts" of smarter, but less efficient algorithms, that minimize the number of unaligned loads, performing shuffles and permutations.
That's a topic for future research, as the performance gains are not yet satisfactory.

> § Reading materials.
> [`memset` benchmarks](https://github.com/nadavrot/memset_benchmark?tab=readme-ov-file) by Nadav Rotem.
> [Cache Associativity](https://en.algorithmica.org/hpc/cpu-cache/associativity/) by Sergey Slotin.


### Hashing

StringZilla implements a high-performance 64-bit hash function inspired by the "AquaHash", "aHash", and "GxHash" design and optimized for modern CPU architectures.
It passes the rigorous SMHasher test suite, including the `--extra` flag with no collisions.

The core algorithm operates on a __dual state__ that runs two independent mixers over the same bytes and folds them together at the end:

- __AES State__: Initialized with the seed `XOR`-ed against π constants and advanced with one AES encryption round per block, providing the strong avalanche behavior.
- __Sum State__: Initialized from a second slice of the π constants and advanced as an additive byte sum under a fixed permutation, cheap to compute and complementary to the AES mixing.

Because the AES round is the backbone, the per-ISA backends lean on each platform's cryptographic instructions rather than emulating them: AES-NI and VAES on x86, SHA-NI on Goldmont, NEON-AES, NEON-SHA, and SVE2-AES on Arm, and RVV-crypto on RISC-V, in the `include/stringzilla/hash/{neonaes,neonsha,sve2aes,rvvcrypto}.h` files.
To keep the streaming path off the stack, the incremental construction uses an __in-register streaming-state merge__: each incoming run is slid into a resident register with a masked load and a blend, so partial blocks never spill to memory.

For strings ≤64 bytes, a minimal state processes data in 16-byte blocks.
Longer strings employ a 4× wider state (512 bits) that processes 64-byte chunks, maximizing throughput on modern superscalar CPUs.
The algorithm can be expressed in pseudocode as:

```
function sz_hash(text: u8[], length: usize, seed: u64) -> u64:
    pi: u64[16] = [0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0, 0x082EFA98EC4E6C89, 0x452821E638D01377, 0xBE5466CF34E90C6C, 0xC0AC29B7C97C50DD, 0x3F84D5B5B5470917,
                   0x9216D5D98979FB1B, 0xD1310BA698DFB5AC, 0x2FFD72DBD01ADFB7, 0xB8E1AFED6A267E96, 0xBA7C9045F12C7F99, 0x24A19947B3916CF7, 0x0801F2E2858EFC16, 0x636920D871574E69]
    shuffle: u8[16] = [0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05, 0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02]   # Permutation order for the sum state

    # Both states are `lanes` × 128 bits wide: one lane for short inputs, four for long ones. The AES half seeds
    # from the low 512 bits of π, the sum half from the high 512 bits, each XOR-ed against the seed.
    lanes: usize = 1 if length ≤ 64 else 4
    aes: u128[lanes] = [seed ⊕ pi[2*lane], seed ⊕ pi[2*lane + 1] for lane in 0..lanes-1]
    sum: u128[lanes] = [seed ⊕ pi[2*lane + 8], seed ⊕ pi[2*lane + 9] for lane in 0..lanes-1]

    # One AES round and one shuffle-add per 16-byte block. Short inputs are zero-padded to 1-4 blocks;
    # long inputs stream 64-byte chunks, one block per lane, so the four lanes stay independent.
    for each chunk: u8[16 * lanes] in split_into_chunks(text, length, 16 * lanes):
        for lane in 0..lanes-1:
            aes[lane] = AESENC(aes[lane], chunk[lane])
            sum[lane] = SHUFFLE(sum[lane], shuffle) + chunk[lane]

    if lanes > 1: aes, sum = fold_to_one_lane(aes), fold_to_one_lane(sum)    # Collapse the 512-bit states back to 128

    # Finalization: mix the length into the key, then AES-mix the two states together for SMHasher compliance
    key: u128 = [seed + length, seed]
    mixed: u128 = AESENC(sum, aes)
    return low_u64(AESENC(AESENC(mixed, key), mixed))
```

This allows us to balance several design trade-offs.
First, it allows us to achieve a high port-level parallelism.
Looking at AVX-512 capable CPUs and their ZMM instructions, on each cycle, we'll have at least 2 ports busy when dealing with long strings:

- `VAESENC`: 5 cycles on port 0 on Intel Ice Lake, 4 cycles on ports 0/1 on AMD Zen4.
- `VPSHUFB_Z`: 3 cycles on port 5 on Intel Ice Lake, 2 cycles on ports 1/2 on AMD Zen4.
- `VPADDQ`: 1 cycle on ports 0/5 on Intel Ice Lake, 1 cycle on ports 0/1/2/3 on AMD Zen4.

When dealing with smaller strings, we design our approach to avoid large registers and maintain the CPU at the same energy state, thereby avoiding downclocking and expensive power-state transitions.

Unlike some AES-accelerated alternatives, the length of the input is not mixed into the AES block at the start to allow incremental construction, when the final length is not known in advance.
Also, unlike some alternatives, with "masked" AVX-512 and "predicated" SVE loads, we avoid expensive block-shuffling procedures on non-divisible-by-16 lengths.

> § Reading materials.
> [Stress-testing hash functions for avalance behaviour, collision bias, and distribution](https://github.com/ashvardanian/HashEvals).

### SHA-256 Checksums

In addition to the fast AES-based hash, StringZilla implements hardware-accelerated SHA-256 cryptographic checksums, following the [FIPS 180-4 specification](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf).
Where the AES hash leans on the AES round instructions, SHA-256 leans on the dedicated SHA extensions: `SHA256RNDS2` and `SHA256MSG1`/`SHA256MSG2` on x86 from Goldmont onward, and the `SHA256H`/`SHA256SU0` family on Arm, with SWAR, LASX, RVV, and WebAssembly fallbacks rounding out the set.

The API is a three-call streaming state — `sz_sha256_state_init`, `sz_sha256_state_update`, `sz_sha256_state_digest` — so arbitrarily long inputs can be absorbed in chunks without buffering the whole message.
Each backend is also exposed under its own suffix, like `sz_sha256_state_update_neonsha`, for the same manual-dispatch reasons as the rest of the library.

Digesting one message is a serial dependency chain — 32 dependent `SHA256RNDS2` at 4 cycles each — so no wider instruction set can accelerate it, and the SHA-NI path already sits within 15% of that hardware floor.
Independent messages are a different story: they compress in parallel lanes, sixteen at a time on AVX-512 and eight on AVX2, which is what `sz_sha256_multistate_update` exposes.
That lane-parallel form roughly doubles single-message SHA-NI throughput, and on an AVX-512 CPU without SHA-NI it is about ten times the scalar path.

### AES-256 Encryption

Most cipher libraries are built for a socket: a stream arrives in order, gets authenticated, and is consumed once.
Columnar and block-storage formats are not sockets.
They seek — a reader wants row group four hundred out of a million-row file and has no interest in the preceding gigabyte, so a cipher that can only start from byte zero forces the whole file through the AES units to answer a question about the middle of it.

StringZilla implements AES-256 in two modes, following [FIPS 197][faq-fips197] for the cipher and [NIST SP 800-38D][faq-gcm] for authentication, and treats seeking as the first-class case.
Counter mode is __seekable__: `sz_aes256_ctr_xor` takes an absolute `byte_offset`, so decryption starts wherever the reader is, at the cost of authentication.
Galois/counter mode adds a tag over the ciphertext and any associated data, and gives up seeking to get it.
Which trade a format wants is the format's decision, so both are exposed rather than one being wrapped in the other.

The two share a key schedule but not a key struct.
`sz_aes256_key_t` is the bare round-key schedule that counter mode needs, while `sz_aes256_gcm_key_t` adds the hash subkey powers `H^1` through `H^8`, so bulk encryption never carries a hundred and twenty-eight bytes it will not read.
Both one-shot and streaming forms exist, and the streaming state carries __two__ sixteen-byte rhythms across chunk boundaries — one for the keystream block and one for the hash block — so a five-byte chunk followed by an eleven-byte chunk produces the same bytes and the same tag as one sixteen-byte chunk.

Streaming splits by __type__ rather than by a flag: `sz_aes256_gcm_encryptor_t` seals and `sz_aes256_gcm_decryptor_t` opens.
The Galois hash absorbs ciphertext in both directions, so a shared state that could be pointed the wrong way would make that a runtime question; separate types make it a compile error instead.

Authenticated decryption returns `sz_authentication_failed_k` on a forged tag and zero-fills its output, so a caller who ignores the status finds nothing usable rather than forged plaintext.
The streaming decrypt is called `sz_aes256_gcm_decryptor_update_unverified` because it structurally emits bytes before anything is authenticated, and the name is the warning.

Backends cover AES-NI and PCLMUL, VAES and VPCLMULQDQ, Arm crypto extensions and SVE2-AES, RISC-V `Zvkned` with `Zvkg`, Power `vcipher` with `vpmsumd`, and a WebAssembly path that has no cipher instructions at all and emulates the round function through a constant-time tower-field substitution — where the 256-byte lookup table a serial implementation reaches for is itself the side channel.

Against OpenSSL the authenticated path is a fair fight, traded back and forth by message size.
Counter mode is not close, and that is an artifact of coverage rather than of arithmetic: OpenSSL ships hand-written AVX-512 assembly for Galois/counter mode and the cipher-block chaining variants, but not for counter mode, so the comparison is a vectorized kernel against a sixteen-byte one.
The honest summary is that we win where the competition has not bothered to vectorize, and draw where it has.
Per-backend numbers are in [`include/stringzilla/cipher/README.md`](include/stringzilla/cipher/README.md).

[faq-fips197]: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf
[faq-gcm]: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38d.pdf
### Random Generation

StringZilla implements a fast [Pseudorandom Number Generator][faq-prng] inspired by the "AES-CTR-128" algorithm, reusing the same AES primitives as the hash function.
Unlike "NIST SP 800-90A" which uses multiple AES rounds, StringZilla uses only one round of AES mixing for performance while maintaining reproducible output across platforms.
The generator operates in counter mode with `AESENC(nonce + lane_index, nonce ⊕ pi_constants)`, rotating through the first 512 bits of π for each 16-byte block.
The only state required to reproduce an output is a 64-bit `nonce`, which is much cheaper than a Mersenne Twister.

[faq-prng]: https://en.wikipedia.org/wiki/Pseudorandom_number_generator

### Sorting

For lexicographic sorting of string collections, StringZilla exports pointer-sized n‑grams ("pgrams") into a contiguous buffer to improve locality, then recursively QuickSorts those pgrams with a 3‑way partition and dives into equal pgrams to compare deeper characters.
Very small inputs fall back to insertion sort.

- Average time complexity: O(n log n)
- Worst-case time complexity: quadratic (due to QuickSort), mitigated in practice by 3‑way partitioning and the n‑gram staging

### Unicode 17, UTF-8, and Wide Characters

Most StringZilla operations are byte-level, so they work well with ASCII and UTF-8 content out of the box.
In some cases, like edit-distance computation, the result of byte-level evaluation and character-level evaluation may differ.

- `szs_levenshtein_distances_utf8("αβγδ", "αγδ") == 1` — one unicode symbol.
- `szs_levenshtein_distances("αβγδ", "αγδ") == 2` — one unicode symbol is two bytes long.

Java, JavaScript, Python 2, C#, and Objective-C, however, expose strings as UTF-16 — a variable-length encoding whose code units are two bytes, so anything outside the Basic Multilingual Plane takes two of them.
Because those languages index by code unit rather than by codepoint, this leads [to all kinds of offset-counting issues][wide-char-offsets] when facing four-byte long Unicode characters.
StringZilla's own bindings for those languages sidestep the problem entirely by operating on the UTF-8 bytes directly, and internally it uses proper 32-bit "runes" to represent unpacked Unicode codepoints, ensuring correct results in all operations.
If you need to transcode between UTF-8, UTF-16, and UTF-32 at the boundary, [simdutf](https://github.com/simdutf/simdutf) is the right tool.
Moreover, StringZilla implements the Unicode 17.0 standard, being practically the only library besides ICU and PCRE2 to do so, but with order(s) of magnitude better performance.

[wide-char-offsets]: https://josephg.com/blog/string-length-lies/

### Case Folding and Uncased Search

StringZilla provides Unicode-aware uncased substring search that handles the full complexity of Unicode case folding.
This includes multi-character expansions:

| Character | Codepoint | UTF-8 Bytes | Case-Folds To | Result Bytes |
| --------- | --------- | ----------- | ------------- | ------------ |
| `ß`       | U+00DF    | C3 9F       | `ss`          | 73 73        |
| `ﬃ`       | U+FB03    | EF AC 83    | `ffi`         | 66 66 69     |
| `İ`       | U+0130    | C4 B0       | `i` + `◌̇`     | 69 CC 87     |

The search returns byte offsets and lengths in the original haystack, correctly handling length differences.
For example, searching for `"STRASSE"` (7 bytes) in `"Straße"` (7 bytes: 53 74 72 61 C3 9F 65) succeeds because both case-fold to `"strasse"`.

Note that Turkish `İ` and ASCII `I` are distinct: `İstanbul` case-folds to `i̇stanbul` (with combining dot), while `ISTANBUL` case-folds to `istanbul` (without).
They will not match each other — this is correct Unicode behavior for Turkish locale handling.

Under the hood, folding uses register-resident lookup tables for the common single-codepoint folds and dedicated expansion paths for the multi-byte ones.
Uncased search folds on the fly, so the haystack is never pre-folded into a second buffer; the matcher tracks the byte-length mismatch whenever a folded form differs in length from its source, returning offsets into the original text.
The folding paths live in the `utf8_uncased*` files.

### UTF-8 Decoding and Unicode Segmentation

Codepoint decoding walks the input in register-wide chunks, classifying each lead byte by its continuation-byte count through a register-resident lookup table rather than branching byte by byte.
It runs under a __fill-and-drain__ contract: a decode-once step fills a buffer of runes, and a separate drain step emits them, which keeps the hot loop tight and lets callers consume runes at their own pace.
Ill-formed input is handled without derailing the stream, re-syncing on the maximal-subpart rule and substituting exactly one U+FFFD before continuing.
The decoder lives in `include/stringzilla/utf8_runes.h`.

UAX-29 word, grapheme, and sentence boundaries, together with UAX-14 line-break opportunities, are found in a single pass.
Each codepoint is classified against register-resident property tables and the boundary rules are applied directly, so combining marks, emoji zero-width-joiner sequences, and regional-indicator flags are all handled without a second traversal.
The segmenters live in the `utf8_wordbreaks*`, `utf8_graphemes*`, `utf8_sentences*`, and `utf8_linebreaks*` files.

### Unicode Normalization

Normalization implements the UAX-15 NFC, NFD, NFKC, and NFKD forms through canonical and compatibility decomposition, canonical-combining-class reordering, and recomposition.
Quick-check flags short-circuit the work, so text that is already in the requested form is passed through without the expensive decomposition and reordering passes.
The normalizers live in the `utf8_norm*` files.

### Set Intersection

StringZilla intersects two deduplicated string collections through a power-of-two open-addressing hash table.
The hash is seeded for adversarial resistance and the probe sequence runs under a bounded collision budget, so the intersection completes in linear time and space rather than degrading on crafted inputs.
The implementation lives in `include/stringzilla/intersect.h`.

### Rolling Fingerprints and MinHash

For near-duplicate detection and multi-pattern search at scale, StringZilla slides multiple Rabin-Karp rolling-hash windows of different widths over each document at once.
Each window tracks its running minimum to build MinHash sketches, and the same passes feed Count-Min-Sketch counters, so a single traversal yields both the similarity signatures and the frequency estimates.
The implementation lives in the `include/stringzillas/fingerprints*` files.

### GPU Edit Distances

On the GPU, short pairs are scored entirely in registers, one thread per pair, holding the anti-diagonal wavefront of the dynamic-programming matrix in registers instead of shared memory.
For short sequences this is several times faster than the classic shared-memory anti-diagonal kernel, which is dominated by shared-memory traffic at small sizes.
Hopper DPX instructions accelerate the min-plus recurrence, and a warp-per-pair path covers older GPUs that lack them.
The kernels live in `include/stringzillas/similarities/hopper.cuh` and `include/stringzillas/similarities/kepler.cuh`.

## Dynamic Dispatch

Due to the high-level of fragmentation of SIMD support in different CPUs, StringZilla names its backends after select CPU generations and instruction-set extensions.
The full v5 set spans the serial SWAR fallback, x86 (Westmere, Goldmont, Haswell, Skylake, Ice Lake), Arm (NEON, NEON-AES, NEON-SHA, SVE, SVE2, SVE2-AES), RISC-V (RVV, RVV-crypto), LoongArch (LASX), IBM Power (PowerVSX), and WebAssembly (v128 and relaxed v128).
You can query supported backends and use them manually.
Use it to guarantee constant performance, or to explore how different algorithms scale on your hardware.

```c
sz_find(text, length, pattern, 3);          // Auto-dispatch
sz_find_westmere(text, length, pattern, 3); // Intel Westmere+ SSE4.2
sz_find_haswell(text, length, pattern, 3);  // Intel Haswell+ AVX2
sz_find_skylake(text, length, pattern, 3);  // Intel Skylake+ AVX-512
sz_find_neon(text, length, pattern, 3);     // Arm NEON 128-bit
sz_find_sve(text, length, pattern, 3);      // Arm SVE 128/256/512/1024/2048-bit
```

StringZilla automatically picks the most advanced backend for the given CPU.
Similarly, in Python, you can log the auto-detected capabilities:

```python
python -c "import stringzilla; print(stringzilla.__capabilities__)"         # e.g. ('serial', 'westmere', 'goldmont', 'haswell', 'skylake', 'icelake')
python -c "import stringzilla; print(stringzilla.__capabilities_str__)"     # e.g. "serial, westmere, goldmont, haswell, skylake, icelake"
# Other targets report their own names: Arm "neon, neonaes, neonsha, sve, sve2, sve2aes",
# WebAssembly "v128, v128relaxed", RISC-V "rvv", LoongArch "lasx", IBM Power "powervsx".
```

You can also explicitly set the backend to use, or scope the backend to a specific function.

```python
import stringzilla as sz
sz.reset_capabilities(('serial',))          # Force SWAR backend
sz.reset_capabilities(('haswell',))         # Force AVX2 backend
sz.reset_capabilities(('neon',))            # Force NEON backend
sz.reset_capabilities(sz.__capabilities__)  # Reset to auto-dispatch
```

## Contributing 👾

Please check out the [contributing guide](https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md) for more details on how to set up the development environment and contribute to this project.
If you like this project, you may also enjoy [USearch][usearch], [UCall][ucall], [UForm][uform], and [SimSIMD][simsimd]. 🤗

[usearch]: https://github.com/unum-cloud/usearch
[ucall]: https://github.com/unum-cloud/ucall
[uform]: https://github.com/unum-cloud/uform
[simsimd]: https://github.com/ashvardanian/simsimd

If you like strings and value efficiency, you may also enjoy the following projects:

- [simdutf](https://github.com/simdutf/simdutf) - transcoding UTF-8, UTF-16, and UTF-32 LE and BE.
- [hyperscan](https://github.com/intel/hyperscan) - regular expressions with SIMD acceleration.
- [pyahocorasick](https://github.com/WojciechMula/pyahocorasick) - Aho-Corasick algorithm in Python.
- [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) - fast string matching in C++ and Python.
- [memchr](https://github.com/BurntSushi/memchr) - fast string search in Rust.

If you are looking for more reading materials on this topic, consider the following:

- [5x faster strings with SIMD & SWAR](https://ashvardanian.com/posts/stringzilla/).
- [The Painful Pitfalls of C++ STL Strings](https://ashvardanian.com/posts/painful-strings/).
- [Processing Strings 109x Faster than Nvidia on H100](https://ashvardanian.com/posts/stringwars-on-gpus/).
- [How a String Library Beat OpenCV at Image Processing by 4x](https://ashvardanian.com/posts/image-processing-with-strings/).
- [2x Faster Hashes on AWS Graviton: NEON → SVE2](https://ashvardanian.com/posts/aws-graviton-checksums-on-neon-vs-sve/).

## Citation

If StringZilla helps your research or product, please cite it:

```bibtex
@software{Vardanian_StringZilla,
  author = {Vardanian, Ash},
  title = {{StringZilla: Fast SIMD, SWAR, and GPGPU String Processing}},
  doi = {10.5281/zenodo.21472333},
  url = {https://github.com/ashvardanian/StringZilla},
  license = {Apache-2.0}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is provided at the repository root.

## License 📜

Feel free to use the project under Apache 2.0 or the Three-clause BSD license at your preference.

<!-- Link references -->

[faq-simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[faq-swar]: https://en.wikipedia.org/wiki/SWAR
[faq-shell]: https://github.com/ashvardanian/StringZilla-CLI
[first-issues]: https://github.com/ashvardanian/StringZilla/issues
[faq-dbms]: https://en.wikipedia.org/wiki/Database
