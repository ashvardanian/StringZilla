# StringZilla Tests

Unit tests that validate correctness across every SIMD backend and language binding against serial and STL baselines.
Each C++ translation unit exercises one kernel family, and the Python suite mirrors it module-for-module.

## C++ on CPUs

The family files test the dispatch points and `stringzilla.hpp`, and the cross files hold every capability's kernels to known answers and to the serial kernels.

- `main_cpu.cpp` — the entry point, registering the family suites and the cross-checks of every architecture.
- `hash.cpp`, `cipher.cpp`, `sort.cpp`, `string.cpp`, `find.cpp` — hashing, ciphers, sorting and intersection, the string class and memory, search and comparison.
- `levenshtein.cpp`, `ngrams.cpp`, `substrings.cpp` — edit distances, N-gram overlaps and MinHash sketches, and multi-pattern search.
- `utf8_runes.cpp`, `utf8_tokens.cpp`, `utf8_wordbreaks.cpp`, `utf8_graphemes.cpp`, `utf8_sentences.cpp`, `utf8_linebreaks.cpp`, `utf8_norm.cpp`, `utf8_uncased.cpp` — UTF-8 decoding, segmentation, normalization and case folding.
- `cross.hpp` — the kernel-level checks, which take kernels by pointer and never reach a dispatch point.
- `cross_serial.cpp`, `cross_x8664.cpp`, `cross_arm64.cpp`, `cross_riscv64.cpp`, `cross_loongarch64.cpp`, `cross_ppc64.cpp`, `cross_wasm.cpp` — one per architecture, naming its kernels, one section per capability, skipping those the CPU lacks.
- `harness.hpp` and `utf8.hpp` are the shared harnesses.

The linked suite runs the CPU checks once, then checks each selected GPU. Two additional executables cover different linkage contracts:

- `stringzilla_test` links `stringzilla_static` and cross-checks CPU and enabled GPU kernels.
- `stringzilla_shared_test` links `stringzilla_shared`, testing CPU dispatch points and the same selected GPU backends.
- `stringzilla_cpu_header_test` compiles the kernels inline, header-only, and runs the cross-checks and the dispatch-point stubs, not the family suites.

## C++ on GPUs

The GPU tests check each vendor's engine kernels by name, and the dispatch points over them, against the serial answers.

- `cross_cuda.cu`, `cross_rocm.hip`, `cross_metal.cpp` — vendor entry points called by the common runner.
- `cross_device.hpp` — the device checks CUDA, ROCm and Metal share, over device-reachable memory, device-bound sequences, a caller's own stream, and the refusals that keep host pointers off the device.

All enabled vendors build into `stringzilla_test`; how the shards share one binary and how `STRINGZILLA_DEVICES` picks devices is under "Devices" in the "Test Suites" section of `CONTRIBUTING.md`.

## Running

The "Test Suites" section of `CONTRIBUTING.md` covers running every suite here, their environment variables and flags, the verdict, `rerun:` and summary lines, and how to write a test in each language.

## Python

The Python modules follow the C++ translation units, with `string_types.py` covering `string.cpp`, and run under pytest.

- `find.py`, `hash.py`, `sort.py`, `string_types.py`, `uncased.py`, `cipher.py`, `utf8_*.py`, and `doctests.py` — per-family tests.
- `base.py` holds the settings, gates and differential helpers, `utf8_helpers.py` the UTF-8 fixtures, and `conftest.py` the pytest hooks and fixtures.
- The modules are flat and import the helpers by basename, as `from base import ...`, so none may share a stdlib module's name.

## JavaScript

- `main.js` — Node test runner, invoked with `node --test`.
