# StringZilla 🦖

![StringZilla banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringZilla-v5.png?raw=true)

Strings are the first fundamental data type every programming language implements in software rather than hardware — no CPU ships a dedicated "find substring" or "compute string hash" instruction.
So most string-processing code still looks like `for (i = 0; i < length; ++i) if (text[i] == 'x') …` — a tangle of loops, branches, and per-character lookups, where the surrounding control flow is easily 10-100x more expensive than the character-level logic itself, whether the text is ASCII or UTF-8 encoded Unicode.
Worse, chewing through one byte or codepoint at a time squanders the hardware: a modern CPU carries dozens of 16-64 byte architectural registers, and hundreds of physical ones to feed out-of-order execution.
StringZilla reaches for those [SIMD][faq-simd] and [SWAR][faq-swar] instructions directly, offering one of the widest, fastest, and most portable collections of text-processing primitives anywhere.

[![StringZilla Python installs](https://static.pepy.tech/personalized-badge/stringzilla?period=total&units=abbreviation&left_color=black&right_color=blue&left_text=StringZilla%20Python%20installs)](https://github.com/ashvardanian/stringzilla)
[![StringZilla Rust installs](https://img.shields.io/crates/d/stringzilla?logo=rust&label=Rust%20installs)](https://crates.io/crates/stringzilla)
![StringZilla code size](https://img.shields.io/github/languages/code-size/ashvardanian/stringzilla)

<!-- Those badges often stay in stale state - greyed out. Consider enabling them later.
[![Ubuntu status](https://img.shields.io/github/checks-status/ashvardanian/StringZilla/main?checkName=Linux%20CI&label=Ubuntu)](https://github.com/ashvardanian/StringZilla/actions/workflows/release.yml)
[![Windows status](https://img.shields.io/github/checks-status/ashvardanian/StringZilla/main?checkName=Windows%20CI&label=Windows)](https://github.com/ashvardanian/StringZilla/actions/workflows/release.yml)
[![macOS status](https://img.shields.io/github/checks-status/ashvardanian/StringZilla/main?checkName=macOS%20CI&label=macOS)](https://github.com/ashvardanian/StringZilla/actions/workflows/release.yml)
-->

StringZilla is the GodZilla of string libraries, accelerating exact and fuzzy matching, hashing, edit distances, sorting, segmentation, and even random-string generation, with allocation-free lazily-evaluated iterators throughout.

- It can be __3x faster than LibC__, the C standard library, doing substring search on Arm.
- It can be __10-70x faster than ICU__ for C and its C rewrite in UTF-8 handling, case folding, segmentation, and tokenization.
- It can be __100x faster than NVIDIA's own libraries__ for on-GPU Levenshtein, NW, and SW edit distances.
- It comes with built-in custom __WebAssembly__ backend for sandboxed browser, DBMS, & LLM environments, custom __RVV__ backend for RISC-V CPUs, __PowerPC__ backend for IBM mainframes, __LoongArch__ for Chinese domestic chips, and more!

Reach for it from your language of choice:

- 🐂 __[C](#c-and-c):__ Upgrade LibC's `<string.h>` to `<stringzilla/stringzilla.h>` in C 99
- 🐉 __[C++](#c-and-c):__ Upgrade STL's `<string>` to `<stringzilla/stringzilla.hpp>` in C++ 11
- 🧮 __[CUDA](include/stringzillas/README.md):__ Process in-bulk with `<stringzillas/stringzillas.cuh>` in CUDA C++ 17
- 🐍 __[Python](#python):__ Upgrade your `str` to faster `Str`
- 🦀 __[Rust](#rust):__ Use the `StringZilla` traits crate
- 🦫 __[Go](#go):__ Use the `StringZilla` cGo module
- 🍎 __[Swift](#swift):__ Use the `String+StringZilla` extension
- 🟨 __[JavaScript](#javascript):__ Use the `StringZilla` library
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

<table>
  <tr>
    <th align="center" width="25%">C</th>
    <th align="center" width="25%">C++</th>
    <th align="center" width="25%">Python</th>
    <th align="center" width="25%">StringZilla</th>
  </tr>
  <!-- Unicode case-folding -->
  <tr>
    <td colspan="4" align="center">Unicode case-folding, expanding characters like <code>ß</code> → <code>ss</code></td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">⚪</td>
    <td align="center">
      <code>.casefold</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.4</b> GB/s
    </td>
    <td align="center">
      <code>sz.utf8_uncased_fold</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>1.3</b> GB/s
    </td>
  </tr>
  <!-- Unicode uncased search -->
  <tr>
    <td colspan="4" align="center">Unicode uncased substring search</td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">⚪</td>
    <td align="center">
      <code>icu.StringSearch</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.02</b> GB/s
    </td>
    <td align="center">
      <code>utf8_uncased_find</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>3.0</b> GB/s
    </td>
  </tr>
  <!-- Substrings, normal order -->
  <tr>
    <td colspan="4" align="center">find the first occurrence of a random word from text, ≅ 5 bytes long</td>
  </tr>
  <tr>
    <td align="center">
      <code>strstr</code> <sup>1</sup><br/>
      <span style="color:#ABABAB;">x86:</span> <b>7.4</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>2.0</b> GB/s
    </td>
    <td align="center">
      <code>.find</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>2.9</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>1.6</b> GB/s
    </td>
    <td align="center">
      <code>.find</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>1.1</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.6</b> GB/s
    </td>
    <td align="center">
      <code>sz_find</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>10.6</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>7.1</b> GB/s
    </td>
  </tr>
  <!-- Substrings, reverse order -->
  <tr>
    <td colspan="4" align="center">find the last occurrence of a random word from text, ≅ 5 bytes long</td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">
      <code>.rfind</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.5</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.4</b> GB/s
    </td>
    <td align="center">
      <code>.rfind</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.9</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.5</b> GB/s
    </td>
    <td align="center">
      <code>sz_rfind</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>10.8</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>6.7</b> GB/s
    </td>
  </tr>
  <!-- Characters, normal order -->
  <tr>
    <td colspan="4" align="center">split lines separated by <code>\n</code> or <code>\r</code> <sup>2</sup></td>
  </tr>
  <tr>
    <td align="center">
      <code>strcspn</code> <sup>1</sup><br/>
      <span style="color:#ABABAB;">x86:</span> <b>5.42</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>2.19</b> GB/s
    </td>
    <td align="center">
      <code>.find_first_of</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.59</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.46</b> GB/s
    </td>
    <td align="center">
      <code>re.finditer</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.06</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.02</b> GB/s
    </td>
    <td align="center">
      <code>sz_find_byteset</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>4.08</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>3.22</b> GB/s
    </td>
  </tr>
  <!-- Characters, reverse order -->
  <tr>
    <td colspan="4" align="center">find the last occurrence of any of 6 whitespaces <sup>2</sup></td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">
      <code>.find_last_of</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.25</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.25</b> GB/s
    </td>
    <td align="center">⚪</td>
    <td align="center">
      <code>sz_rfind_byteset</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.43</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.23</b> GB/s
    </td>
  </tr>
  <!-- Random Generation -->
  <tr>
    <td colspan="4" align="center">Random string from a given alphabet, 20 bytes long <sup>3</sup></td>
  </tr>
  <tr>
    <td align="center">
      <code>rand() % n</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>18.0</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>9.4</b> MB/s
    </td>
    <td align="center">
      <code>uniform_int_distribution</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>47.2</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>20.4</b> MB/s
    </td>
    <td align="center">
      <code>join(random.choices(x))</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>13.3</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>5.9</b> MB/s
    </td>
    <td align="center">
      <code>sz_fill_random</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>56.2</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>25.8</b> MB/s
    </td>
  </tr>
  <!-- Mapping characters with lookup table transforms -->
  <tr>
    <td colspan="4" align="center">Mapping characters with lookup table transforms</td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">
      <code>std::transform</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>3.81</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>2.65</b> GB/s
    </td>
    <td align="center">
      <code>str.translate</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>260.0</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>140.0</b> MB/s
    </td>
    <td align="center">
      <code>sz_lookup</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>21.2</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>8.5</b> GB/s
    </td>
  </tr>
  <!-- Sorting -->
  <tr>
    <td colspan="4" align="center">Get sorted order, ≅ 8 million English words <sup>4</sup></td>
  </tr>
  <tr>
    <td align="center">
      <code>qsort_r</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>3.55</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>5.77</b> s
    </td>
    <td align="center">
      <code>std::sort</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>2.79</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>4.02</b> s
    </td>
    <td align="center">
      <code>numpy.argsort</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>7.58</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>13.00</b> s
    </td>
    <td align="center">
      <code>sz_sequence_argsort</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>1.91</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>2.37</b> s
    </td>
  </tr>
  <!-- Edit Distance -->
  <tr>
    <td colspan="4" align="center">Levenshtein edit distance, text lines ≅ 100 bytes long</td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">⚪</td>
    <td align="center">
      via <code>NLTK</code> <sup>5</sup> and <code>CuDF</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>1,615,306</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>1,349,980</b> &centerdot;
      <span style="color:#ABABAB;">cuda:</span> <b>6,532,411,354</b> CUPS
    </td>
    <td align="center">
      <code>szs_levenshtein_distances_t</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>3,434,427,548</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>1,605,340,403</b> &centerdot;
      <span style="color:#ABABAB;">cuda:</span> <b>93,662,026,653</b> CUPS
    </td>
  </tr>
  <!-- Alignment Score -->
  <tr>
    <td colspan="4" align="center">Needleman-Wunsch alignment scores, proteins ≅ 1 K amino acids long</td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">⚪</td>
    <td align="center">
      via <code>biopython</code> <sup>6</sup><br/>
      <span style="color:#ABABAB;">x86:</span> <b>575,981,513</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>436,350,732</b> CUPS
    </td>
    <td align="center">
      <code>szs_needleman_wunsch_scores_t</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>452,629,942</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>520,170,239</b> &centerdot;
      <span style="color:#ABABAB;">cuda:</span> <b>9,017,327,818</b> CUPS
    </td>
  </tr>
</table>

Most StringZilla modules ship ready-to-run benchmarks for C, C++, Python, and more.
Grab them from `./scripts`, and see [`CONTRIBUTING.md`](CONTRIBUTING.md), [`test/README.md`](test/README.md), and [`bench/README.md`](bench/README.md) for instructions.
On CPUs that permit misaligned loads, even the 64-bit SWAR baseline outruns both libc and the STL.
For wider head-to-heads against Rust and Python favorites, browse the __[StringWars][stringwars]__ repository.
To inspect collision resistance and distribution shapes for our hashers, see __[HashEvals][hashevals]__.

[stringwars]: https://github.com/ashvardanian/StringWars
[hashevals]: https://github.com/ashvardanian/HashEvals

> Most benchmarks were conducted on a 1 GB English text corpus, with an average word length of 6 characters.
> The code was compiled with GCC 12, using `glibc` v2.35.
> The benchmarks were performed on Arm-based Graviton3 AWS `c7g` instances and `r7iz` Intel Sapphire Rapids.
> Most modern Arm-based 64-bit CPUs will have similar relative speedups.
> Variance within x86 CPUs will be larger.
> For CUDA benchmarks, the Nvidia H100 GPUs were used.
> <sup>1</sup> Unlike other libraries, LibC requires strings to be NULL-terminated.
> <sup>2</sup> Six whitespaces in the ASCII set are: ` \t\n\v\f\r`. Python's and other standard libraries have specialized functions for those.
> <sup>3</sup> All modulo operations were conducted with `uint8_t` to allow compilers more optimization opportunities.
> The C++ STL and StringZilla benchmarks used a 64-bit [Mersenne Twister][faq-mersenne-twister] as the generator.
> For C, C++, and StringZilla, an in-place update of the string was used.
> In Python every string had to be allocated as a new object, which makes it less fair.
> <sup>4</sup> Contrary to the popular opinion, Python's default `sorted` function works faster than the C and C++ standard libraries.
> That holds for large lists or tuples of strings, but fails as soon as you need more complex logic, like sorting dictionaries by a string key, or producing the "sorted order" permutation.
> The latter is very common in database engines and is most similar to `numpy.argsort`.
> The current StringZilla solution can be at least 4x faster without loss of generality.
> <sup>5</sup> Most Python libraries for strings are also implemented in C.
> <sup>6</sup> Unlike the rest of BioPython, the alignment score computation is [implemented in C](https://github.com/biopython/biopython/blob/master/Bio/Align/_pairwisealigner.c).

[faq-mersenne-twister]: https://en.wikipedia.org/wiki/Mersenne_Twister

## Why StringZilla

StringZilla replaces a stack of specialized libraries with one portable dependency, and outpaces each on its own turf.

| Alternative                           | Scope It Covers                           | StringZilla Edge                                                                                                  |
| :------------------------------------ | :---------------------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| libc, via `strstr`/`memmem`/`memcpy`  | byte search and memory operations         | bidirectional search, hashing, sorting, and sets too, up to 3x faster on Arm                                      |
| ICU and ICU4X                         | Unicode case, segmentation, normalization | stable C ABI, no hidden allocations, 10-70x faster on folding and segmentation                                    |
| RapidFuzz and edit-distance libraries | fuzzy matching and Levenshtein            | batched on CPU cores, and 100x faster than NVIDIA's libraries on GPUs                                             |
| xxHash, aHash, and other fast hashers | non-cryptographic hashing                 | an AES-based hash on par or faster on single hashes, and far ahead with `hash_multiseed` for sketches and filters |
| `std::string` and `std::sort`         | general strings and sorting               | an SSO container, lazy allocation-free views, and allocator-routed sorting                                        |

Because StringZilla mirrors the familiar standard APIs, adoption is mostly a search-and-replace.

In Python:

| Operation              | Standard           | StringZilla                   |
| :--------------------- | :----------------- | :---------------------------- |
| Find a substring       | `"...".find(x)`    | `sz.find("...", x)`           |
| Sort strings           | `sorted(items)`    | `sz.Strs(items).sorted()`     |
| Split on a separator   | `"...".split(sep)` | `sz.Str("...").split(sep)`    |
| Case-fold for matching | `"...".casefold()` | `sz.utf8_uncased_fold("...")` |
| Streaming SHA-256      | `hashlib.sha256()` | `sz.Sha256()`                 |

In C++:

| Operation                 | Standard                             | StringZilla                                                  |
| :------------------------ | :----------------------------------- | :----------------------------------------------------------- |
| Find a substring          | `std::string::find`                  | `sz::string::find`                                           |
| Sort a collection         | `std::sort` of indices               | `sz::argsort`                                                |
| Hash map with string keys | `std::unordered_map<std::string, V>` | `std::unordered_map<std::string, V, sz::hash, sz::equal_to>` |
| Intersect two string sets | `std::set_intersection`              | `sz::try_intersect`                                          |

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

|                                | Maturity |   C   |  C++  | Python | Rust  |  JS   | Swift |  Go   |
| :----------------------------- | :------: | :---: | :---: | :----: | :---: | :---: | :---: | :---: |
| Substring Search               |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |
| Character Set Search           |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |
| Sorting & Sequence Operations  |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |
| Set Intersection & Joins       |    🧐     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |
| Lazy Ranges, Compressed Arrays |    🌳     |   ❌   |   ✅   |   ✅    |   ✅   |   ❌   |   ⚪   |   ⚪   |
| One-Shot & Streaming Hashes    |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |
| Cryptographic Hashes           |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |
| Small String Class             |    🧐     |   ✅   |   ✅   |   ❌    |   ⚪   |   ❌   |   ❌   |   ❌   |
| Random String Generation       |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |
|                                |          |       |       |        |       |       |       |       |
| Unicode Case Folding           |    🧐     |   ✅   |   ✅   |   ✅    |   ⚪   |   ⚪   |   ⚪   |   ⚪   |
| Uncased UTF-8 Search           |    🚧     |   ✅   |   ✅   |   ✅    |   ⚪   |   ⚪   |   ⚪   |   ⚪   |
| TR29 Word Boundary Detection   |    🚧     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ✅   |   ⚪   |
| TR29 Grapheme Segmentation     |    🚧     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |
| TR29 Sentence Segmentation     |    🚧     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |
| UAX14 Line-Break Detection     |    🚧     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |
| Unicode Normalization          |    🚧     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ✅   |   ⚪   |
| Codepoint Counting & Indexing  |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |
|                                |          |       |       |        |       |       |       |       |
| Parallel Similarity Scoring    |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |
| Parallel Rolling Fingerprints  |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |

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

Header-only, or pull it in with CMake `FetchContent` &centerdot; guides: [`include/stringzilla/README.md`](include/stringzilla/README.md) and [`include/stringzillas/README.md`](include/stringzillas/README.md)

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

`dotnet add package StringZilla` &centerdot; guide: [`csharp/README.md`](csharp/README.md)

```csharp
using StringZilla;

Sz.IndexOf("the quick brown fox"u8, "brown"u8); // 10
Sz.Hash("hello"u8);                             // fast 64-bit hash
```

Zero-copy over `ReadOnlySpan<byte>` (and Unity's `NativeArray<byte>`); `net8.0`, NativeAOT-friendly.
Exposes search, hashing, SHA-256, UTF-8 segmentation, case-folding, normalization, and sorting.

### Java

Maven `com.github.ashvardanian:stringzilla` &centerdot; guide: [`java/README.md`](java/README.md)

```java
import com.stringzilla.StringZilla;

StringZilla.indexOf("the quick brown fox".getBytes(), "brown".getBytes());  // 10
StringZilla.hash("hello".getBytes());                                       // fast 64-bit hash
```

Pure [Foreign Function & Memory API](https://openjdk.org/jeps/454) (JDK 22+), __no JNI__.
Zero-copy over `byte[]` and `MemorySegment` — including Lucene `BytesRef` and Spark `UTF8String` backing memory.

## Algorithms & Design Decisions

StringZilla aims to optimize some of the slowest string operations.
Some popular operations, however, like equality comparisons and relative order checking, almost always complete on some of the very first bytes in either string.
In such operations vectorization is almost useless, unless huge and very similar strings are considered.
StringZilla implements those operations as well, but won't result in substantial speedups.
Where vectorization stops being effective, parallelism takes over with the new layered cake architecture:

- StringZilla C library w/out dependencies
- StringZillas parallel extensions:
  - Parallel C++ algorithms built with [Fork Union](https://github.com/ashvardanian/fork_union)
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

> For 2-byte needles, see `sz_find_2byte_serial_` in `include/stringzilla/find.h`:

https://github.com/ashvardanian/StringZilla/blob/e1966de91600298d3c5cf4fe7be40d434f0f405e/include/stringzilla/find.h#L422-L463

Going beyond that, to long needles, Boyer-Moore (BM) and its variants are often the best choice.
It has two tables: the good-suffix shift and the bad-character shift.
Common choice is to use the simplified BMH algorithm, which only uses the bad-character shift table, reducing the pre-processing time.
We do the same for mid-length needles up to 256 bytes long.
That way the stack-allocated shift table remains small.

> For mid-length needles (≤256 bytes), see `sz_find_horspool_upto_256bytes_serial_` in `include/stringzilla/find.h`:

https://github.com/ashvardanian/StringZilla/blob/e1966de91600298d3c5cf4fe7be40d434f0f405e/include/stringzilla/find.h#L620-L667

In the C++ Standards Library, the `std::string::find` function uses the BMH algorithm with Raita's heuristic.
Before comparing the entire string, it matches the first, last, and the middle character.
Very practical, but can be slow for repetitive characters.
Both SWAR and SIMD backends of StringZilla have a cheap pre-processing step, where we locate unique characters.
This makes the library a lot more practical when dealing with non-English corpora.

> The offset selection heuristic is implemented in `sz_locate_needle_anomalies_` in `include/stringzilla/find.h`:

https://github.com/ashvardanian/StringZilla/blob/e1966de91600298d3c5cf4fe7be40d434f0f405e/include/stringzilla/find.h#L244-L305

All those, still, have $O(hn)$ worst case complexity.
To guarantee $O(h)$ worst case time complexity, the Apostolico-Giancarlo (AG) algorithm adds an additional skip-table.
Preprocessing phase is $O(n+sigma)$ in time and space.
On traversal, performs from $(h/n)$ to $(3h/2)$ comparisons.
It however, isn't practical on modern CPUs.
A simpler idea, the Galil-rule might be a more relevant optimizations, if many matches must be found.

Other algorithms previously considered and deprecated:

- Apostolico-Giancarlo algorithm for longer needles. _Control-flow is too complex for efficient vectorization._
- Shift-Or-based Bitap algorithm for short needles. _Slower than SWAR._
- Horspool-style bad-character check in SIMD backends. _Effective only for very long needles, and very uneven character distributions between the needle and the haystack. Faster "character-in-set" check needed to generalize._

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
    # 1024 bits worth of π constants
    pi: u64[16] = [
        0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0, 0x082EFA98EC4E6C89,
        0x452821E638D01377, 0xBE5466CF34E90C6C, 0xC0AC29B7C97C50DD, 0x3F84D5B5B5470917,
        0x9216D5D98979FB1B, 0xD1310BA698DFB5AC, 0x2FFD72DBD01ADFB7, 0xB8E1AFED6A267E96,
        0xBA7C9045F12C7F99, 0x24A19947B3916CF7, 0x0801F2E2858EFC16, 0x636920D871574E69]

    # Permutation order for the sum state
    shuffle_pattern: u8[16] = [
        0x04, 0x0b, 0x09, 0x06, 0x08, 0x0d, 0x0f, 0x05,
        0x0e, 0x03, 0x01, 0x0c, 0x00, 0x07, 0x0a, 0x02]

    # Initialize key and states
    keys_u64s: u64[2] = [seed, seed]
    aes_u64s: u64[2] = [seed ⊕ pi[0], seed ⊕ pi[1]]
    sum_u64s: u64[2] = [seed ⊕ pi[8], seed ⊕ pi[9]]

    if length ≤ 64:
        # Small input: process 1-4 zero-padded blocks of 16 bytes each
        blocks_u8s: u8[16][] = split_into_blocks(text, length, 16)
        for each block_u8s: u8[16] in blocks_u8s:
            aes_u64s = AESENC(aes_u64s, block_u8s)
            sum_u64s = SHUFFLE(sum_u64s, shuffle_pattern) + block_u8s
    else:
        # Large input: use 4× wider 512-bits states
        aes_u64s: u64[8] = [
            seed ⊕ pi[0], seed ⊕ pi[1], seed ⊕ pi[2], seed ⊕ pi[3],
            seed ⊕ pi[4], seed ⊕ pi[5], seed ⊕ pi[6], seed ⊕ pi[7]]
        sum_u64s: u64[8] = [
            seed ⊕ pi[8], seed ⊕ pi[9], seed ⊕ pi[10], seed ⊕ pi[11],
            seed ⊕ pi[12], seed ⊕ pi[13], seed ⊕ pi[14], seed ⊕ pi[15]]

        # Process 64-byte chunks (4×16-byte blocks)
        for each chunk_u8s: u8[64] in text:
            blocks_u8s: u8[16][4] = split_chunk_into_4_blocks(chunk_u8s)
            for i in 0..3:
                offset: usize = i * 2  # Each lane stores two u64s
                aes_u64s[offset:offset+1] = AESENC(aes_u64s[offset:offset+1], blocks_u8s[i])
                sum_u64s[offset:offset+1] = SHUFFLE(sum_u64s[offset:offset+1], shuffle_pattern) + blocks_u8s[i]

        # Fold 8×u64 state back to 2×u64 for finalization
        aes_u64s: u64[2] = fold_to_2u64(aes_u64s)
        sum_u64s: u64[2] = fold_to_2u64(sum_u64s)

    # Finalization: mix length into key
    key_with_length: u64[2] = [keys_u64s[0] + length, keys_u64s[1]]

    # Multiple AES rounds for SMHasher compliance
    mixed_u64s: u64[2] = AESENC(sum_u64s, aes_u64s)
    result_u64s: u64[2] = AESENC(AESENC(mixed_u64s, key_with_length), mixed_u64s)

    return result_u64s[0]  # Extract low 64 bits
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

In addition to the fast AES-based hash, StringZilla implements hardware-accelerated SHA-256 cryptographic checksums.
The implementation follows the [FIPS 180-4 specification](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf) and provides multiple backends.

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

Java, JavaScript, Python 2, C#, and Objective-C, however, use wide characters (`wchar`) - two byte long codes, instead of the more reasonable fixed-length UTF-32 or variable-length UTF-8.
This leads [to all kinds of offset-counting issues][wide-char-offsets] when facing four-byte long Unicode characters.
StringZilla uses proper 32-bit "runes" to represent unpacked Unicode codepoints, ensuring correct results in all operations.
Moreover, it implements the Unicode 17.0 standard, being practically the only library besides ICU and PCRE2 to do so, but with order(s) of magnitude better performance.

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

For wide-character environments (Java, JavaScript, Python 2, C#), consider transcoding with [simdutf](https://github.com/simdutf/simdutf).

### UTF-8 Decoding and Unicode Segmentation

Codepoint decoding walks the input in register-wide chunks, classifying each lead byte by its continuation-byte count through a register-resident lookup table rather than branching byte by byte.
It runs under a __fill-and-drain__ contract: a decode-once step fills a buffer of runes, and a separate drain step emits them, which keeps the hot loop tight and lets callers consume runes at their own pace.
Ill-formed input is handled without derailing the stream, re-syncing on the maximal-subpart rule and substituting exactly one U+FFFD before continuing.
The decoder lives in `include/stringzilla/utf8_runes.h`.

UAX-29 word, grapheme, and sentence boundaries, together with UAX-14 line-break opportunities, are found in a single pass.
Each codepoint is classified against register-resident property tables and the boundary rules are applied directly, so combining marks, emoji zero-width-joiner sequences, and regional-indicator flags are all handled without a second traversal.
The segmenters live in the `utf8_words*`, `utf8_graphemes*`, `utf8_sentences*`, and `utf8_linebreaks*` files.

### Unicode Normalization and Case Folding

Normalization implements the UAX-15 NFC, NFD, NFKC, and NFKD forms through canonical and compatibility decomposition, canonical-combining-class reordering, and recomposition.
Quick-check flags short-circuit the work, so text that is already in the requested form is passed through without the expensive decomposition and reordering passes.
The normalizers live in the `utf8_norm*` files.

Case folding uses register-resident lookup tables for the common single-codepoint folds and dedicated expansion paths for the multi-byte ones, such as `ß` folding to `ss` and the `ﬃ` ligature folding to `ffi`.
Uncased search folds on the fly, so the haystack is never pre-folded into a second buffer; the matcher tracks the byte-length mismatch whenever a folded form differs in length from its source, returning offsets into the original text.
The folding paths live in the `utf8_uncased*` files.

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
sz_find_westmere(text, length, pattern, 3);  // Intel Westmere+ SSE4.2
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
