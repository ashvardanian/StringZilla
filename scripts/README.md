# StringZilla Scripts

This directory contains benchmarks, tests, and exploratory scripts for the StringZilla library, focused on internal functionality, rather than third-party alternatives.

- For comparative performance analysis, please refer to [StringWars](https://github.com/ashvardanian/StringWars).
- To understand the distributional properties of hash functions, see [HashEvals](https://github.com/ashvardanian/HashEvals).

## Benchmark Programs

Benchmarks validate SIMD-accelerated backends against serial baselines and measure throughput on real-world workloads.

- `bench_find.cpp` - bidirectional substring search, byte search, and byteset search
- `bench_token.cpp` - token-level operations: hashing, checksums, equality, and ordering
- `bench_sequence.cpp` - sorting, partitioning, and set intersections of string arrays
- `bench_memory.cpp` - memory operations: copies, moves, fills, and lookup table transformations
- `bench_container.cpp` - STL associative containers (`std::map`, `std::unordered_map`) with string keys
- `bench_similarities.cpp` - Levenshtein, Needleman-Wunsch, Smith-Waterman scoring on CPU
- `bench_fingerprints.cpp` - MinHash rolling fingerprints and multi-pattern search on CPU
- `bench_similarities.cu` - similarity scoring algorithms on CUDA GPUs
- `bench_fingerprints.cu` - fingerprinting algorithms on CUDA GPUs

All benchmarks support environment variables for configuration.
Check file headers for details.

## Test Programs

Unit tests validate correctness across all backends and programming languages.

- `test_stringzilla.cpp` - C++ API tests against STL baselines
- `test_stringzilla.py` - Python API tests against native strings
- `test_stringzillas.cpp` - parallel CPU backend tests
- `test_stringzillas.cu` - CUDA backend tests
- `test.js` - JavaScript API tests

## Exploratory Notebooks

Jupyter notebooks for algorithm visualization and analysis.

- `explore_levenshtein.ipynb` - edit distance algorithms and diagonal traversal
- `explore_fingerprint.ipynb` - MinHash and rolling fingerprints
- `explore_unicode.ipynb` - UTF-8 handling and Unicode normalization

