# StringZilla Benchmarks

Benchmarks that validate the SIMD-accelerated backends against serial baselines and measure throughput on real-world workloads.
Each C++ benchmark compares the serial baseline against every available SIMD backend for the same operation, for example `sz_find_serial` against `sz_find_haswell` against `sz_find_skylake`.
This is the internal, cross-backend counterpart to [StringWars](https://github.com/ashvardanian/StringWars), which instead compares only the single best-available StringZilla backend against external libraries.

## CPU

- `find.cpp` — bidirectional substring, byte, and byteset search.
- `token.cpp` — token-level hashing, checksums, equality, and ordering.
- `sequence.cpp` — sorting, partitioning, and set intersection of string arrays.
- `memory.cpp` — copies, moves, fills, and lookup-table transforms.
- `container.cpp` — STL associative containers with string keys.
- `similarities.cpp` — Levenshtein, Needleman-Wunsch, and Smith-Waterman scoring.
- `fingerprints.cpp` — MinHash rolling fingerprints and multi-pattern search.
- `utf8_iterate.cpp` and `utf8_uncased.cpp` — UTF-8 iteration, segmentation, and case-folding throughput.

## CUDA

- `similarities.cu` — similarity scoring on CUDA GPUs.
- `fingerprints.cu` — fingerprinting on CUDA GPUs.

## Other Bindings

- `stringzilla.go` — Go binding benchmark.

`shared.hpp` is the common harness.
All benchmarks read environment variables for configuration — backend filter, batch size, and stress mode — documented in each file's header.
