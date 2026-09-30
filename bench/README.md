# StringZilla Benchmarks

Benchmarks that validate the SIMD-accelerated backends against serial baselines and measure throughput on real-world workloads.
This is the internal, cross-backend counterpart to [StringWars](https://github.com/ashvardanian/StringWars), which instead compares only the single best-available StringZilla backend against external libraries.

## CPU

`stringzilla_cpu_bench` is one executable over every family and every capability, linking the static library.
The family files time the dispatch points against the standard baselines, like `sz_find_best` against `strstr` and `std::boyer_moore_searcher`:

- `find.cpp` — bidirectional substring, byte, and byteset search.
- `token.cpp` — token-level hashing, checksums, equality, and ordering.
- `sequence.cpp` — sorting, partitioning, and set intersection of string arrays.
- `memory.cpp` — copies, moves, fills, and lookup-table transforms.
- `cipher.cpp` — AES-256 counter mode and Galois/counter mode throughput.
- `container.cpp` — STL associative containers with string keys.
- `levenshtein.cpp` — one-to-one and one-to-many edit distances, over bytes and over runes.
- `overlap.cpp` — window hashing, key extraction, B-tree probes, and whole-verb overlap scoring.
- `substrings.cpp` — multi-pattern Aho-Corasick counting, locating, rewriting, and BM25 scoring.
- `utf8_traverse.cpp` — codepoint counting, Nth-codepoint seeking, and codepoint iteration.
- `utf8_scan.cpp` — codepoint-class enumeration: newlines, whitespace, and delimiter runs.
- `utf8_segment.cpp` — UAX-29 and UAX-14 boundary segmentation: words, graphemes, sentences, and linebreaks.
- `utf8_norm.cpp` — Unicode normalization and quick-check scanning.
- `utf8_uncased.cpp` — case folding and uncased search.

The kernels are timed by name, each stress-tested against the serial kernel of the same operation and logged relative to it, like `sz_find_haswell` against `sz_find_serial`.
`cross.hpp` holds the adapters, and each `cross_<arch>.cpp` holds one section per capability, skipped when the CPU lacks it:

- `cross_serial.cpp` — the serial kernels, timed first.
- `cross_x8664.cpp` — Westmere, Goldmont, Haswell, Skylake, and Ice Lake.
- `cross_arm64.cpp` — NEON, NEON AES, NEON SHA, SVE, SVE2, and SVE2 AES.
- `cross_riscv64.cpp` — RVV and RVV Crypto.
- `cross_loongarch64.cpp` — Loongson ASX.
- `cross_ppc64.cpp` — Power VSX.
- `cross_wasm.cpp` — V128 and V128 Relaxed.

`stringzilla_cpu_header_bench` compiles `main.cpp` and the cross files header-only, with the kernels inline.
It has no dispatch points to time, but it reaches the private helpers only the tier headers define, like the SIMD sorts of pgrams and the Levenshtein and overlap steps.

## GPU

Each device gets its own executable, timing the engines' device-resident rounds against the widest CPU tier the build carries:

- `cross_simt.cuh` — the edit-distance, window-overlap, and multi-pattern search rows CUDA and ROCm share.
- `cross_cuda.cu` and `cross_rocm.hip` — `stringzilla_cuda_bench` and `stringzilla_rocm_bench` over those rows.
- `cross_metal.cpp` — `stringzilla_metal_bench`, window-overlap scoring on Apple GPUs.
- `substrings.cuh` — the vocabulary, corpus, and arms the CPU and GPU multi-pattern benchmarks share.

## Other Bindings

- `stringzilla.go` — Go binding benchmark.

## Configuration

`harness.hpp` is the common harness, and `main.cpp` runs the families and then the cross files.
All benchmarks read environment variables for configuration — dataset, backend filter, batch size, and stress mode — documented in each file's header.
Each family runs on its own corpus: English words or lines of `leipzig1M.txt`, or multilingual lines of `xlsum.csv`, each loaded on first use.
`STRINGWARS_DATASET` points every corpus at one file, and `STRINGWARS_DATASET_LIMIT` bounds the read, which keeps a short run short.
`STRINGWARS_FILTER` is a regex over benchmark names, and every backend runs when it is unset.
`STRINGWARS_SEED` shuffles the tokens when set to a positive integer, and leaves them in order when unset, which is the default.
`CONTRIBUTING.md` lists the rest, with their defaults.

```sh
cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
cmake --build build_release --config Release --target stringzilla_cpu_bench
STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_FILTER='find' build_release/stringzilla_cpu_bench
```
