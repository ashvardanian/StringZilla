# StringZilla Tests

Unit tests that validate correctness across every SIMD backend and language binding against serial and STL baselines.
Each C++ translation unit exercises one kernel family, and the Python suite mirrors it module-for-module.

## C++ on CPUs

The family files test the dispatch points and `stringzilla.hpp`, and the cross files hold every capability's kernels to known answers and to the serial kernels.

- `main_cpu.cpp` — the entry point, registering the family suites and the cross-checks of every architecture.
- `hash.cpp`, `cipher.cpp`, `sort.cpp`, `string.cpp`, `find.cpp` — hashing, ciphers, sorting and intersection, the string class and memory, search and comparison.
- `levenshtein.cpp`, `overlap.cpp`, `substrings.cpp` — edit distances, window overlap, and multi-pattern search.
- `utf8_runes.cpp`, `utf8_tokens.cpp`, `utf8_wordbreaks.cpp`, `utf8_graphemes.cpp`, `utf8_sentences.cpp`, `utf8_linebreaks.cpp`, `utf8_norm.cpp`, `utf8_uncased.cpp` — UTF-8 decoding, segmentation, normalization and case folding.
- `cross.hpp` — the kernel-level checks, which take kernels by pointer and never reach a dispatch point.
- `cross_serial.cpp`, `cross_x8664.cpp`, `cross_arm64.cpp`, `cross_riscv64.cpp`, `cross_loongarch64.cpp`, `cross_ppc64.cpp`, `cross_wasm.cpp` — one per architecture, naming its kernels, one section per capability, skipping those the CPU lacks.
- `harness.hpp` and `utf8.hpp` are the shared harnesses.

Three executables share these sources:

- `stringzilla_cpu_test` links `stringzilla_static` and cross-checks every kernel the library defines.
- `stringzilla_cpu_shared_test` links `stringzilla_shared` with every capability off, so only the dispatch points run.
- `stringzilla_cpu_header_test` compiles the kernels inline, header-only, and runs the cross-checks and the dispatch-point stubs, not the family suites.

## C++ on GPUs

The GPU tests check each vendor's engine kernels by name, and the dispatch points over them, against the serial answers.

- `main_cuda.cu`, `main_rocm.hip`, `main_metal.cpp` — the entry points, one per vendor.
- `cross_simt.cuh` — the device checks CUDA and ROCm share, over device-reachable memory, device-bound sequences, a caller's own stream, and the refusals that keep host pointers off the device.
- `cross_cuda.cu`, `cross_rocm.hip` — the CUDA and ROCm kernels by name, through `cross_simt.cuh`.
- `cross_metal.cpp` — the Metal kernels of edit distances, window overlap and multi-pattern search on Apple GPUs.

Each builds one executable over `stringzilla_static`: `stringzilla_cuda_test`, `stringzilla_rocm_test`, and `stringzilla_metal_test`.

## Running

- Cross-checks are named `test_<family>_<tier>_<capability>`, so `STRINGZILLA_FILTER='_haswell$'` runs one capability and `STRINGZILLA_FILTER='^test_find_'` one family across all of them.
- `STRINGZILLA_SEED` defaults to 42 and takes `random` to draw one, `STRINGZILLA_SCALE` defaults to 1.0, and `STRINGZILLA_FILTER` is a regex over test names that runs all of them when unset and matches as a substring when it does not compile.
- A failing test prints `rerun: STRINGZILLA_SEED=<n> STRINGZILLA_FILTER='^<name>$' <binary>` to stderr, which reruns exactly that test, and the run moves on to the next one.
- A crash stops the run, and the startup `- Rerun one test:` line holds the same command for the last test named.

## Python

The Python modules mirror the C++ translation units one-for-one and run under pytest.

- `find.py`, `hash.py`, `sort.py`, `string.py`, `uncased.py`, `cipher.py`, `utf8_*.py`, and `doctests.py` — per-family tests.
- `helpers.py` and `utf8_helpers.py` are shared helpers; `conftest.py` holds the pytest configuration.
- This directory is a Python package via `__init__.py`, so the prefix-less modules namespace as `test.*` and never shadow stdlib names.
- Run the suite with `pytest test/`.
- Each seeded test runs under the seeds 42, 0, 1 and 314159, a numeric `STRINGZILLA_SEED` replaces them with one, `random` adds a drawn one, and pytest's header prints the list.
- `STRINGZILLA_SCALE` applies as in C++, while `STRINGZILLA_FILTER` does not; select tests with `pytest -k` instead.

## JavaScript

- `main.js` — Node test runner, invoked with `node --test`.
