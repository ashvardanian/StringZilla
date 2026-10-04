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

The linked suite runs the CPU checks once, then checks each selected GPU. Two additional executables cover different linkage contracts:

- `stringzilla_test` links `stringzilla_static` and cross-checks CPU and enabled GPU kernels.
- `stringzilla_shared_test` links `stringzilla_shared`, testing CPU dispatch points and the same selected GPU backends.
- `stringzilla_cpu_header_test` compiles the kernels inline, header-only, and runs the cross-checks and the dispatch-point stubs, not the family suites.

## C++ on GPUs

The GPU tests check each vendor's engine kernels by name, and the dispatch points over them, against the serial answers.

- `cross_cuda.cu`, `cross_rocm.hip`, `cross_metal.cpp` — vendor entry points called by the common runner.
- `cross_simt.cuh` — the device checks CUDA and ROCm share, over device-reachable memory, device-bound sequences, a caller's own stream, and the refusals that keep host pointers off the device.
- `cross_cuda.cu`, `cross_rocm.hip` — the CUDA and ROCm kernels by name, through `cross_simt.cuh`.
- `cross_metal.cpp` — the Metal kernels of edit distances, window overlap and multi-pattern search on Apple GPUs.

All enabled vendors build into `stringzilla_test`; CUDA and ROCm can coexist in one build and run.
`STRINGZILLA_DEVICES=cuda:0,rocm:1` selects vendor-qualified ordinals. Unset, the runner visits device zero of each available compiled vendor; an explicit unavailable device is an error.

## Running

- Cross-checks are named `test_<family>_<tier>_<capability>`, so `STRINGZILLA_FILTER='_haswell$'` runs one capability and `STRINGZILLA_FILTER='^test_find_'` one family across all of them.
- `STRINGZILLA_SEED`, `STRINGZILLA_SCALE` and `STRINGZILLA_FILTER` are listed with their defaults in the header of `harness.hpp` and in `CONTRIBUTING.md`.
- A failing test prints `rerun: STRINGZILLA_SEED=<n> STRINGZILLA_FILTER='^<name>$' <binary>` to stderr, which reruns exactly that test, and the run moves on to the next one.
- A crash stops the run, and the startup `- Rerun one test:` line holds the same command for the last test named.

## Python

The Python modules follow the C++ translation units, with `string_types.py` covering `string.cpp`, and run under pytest.

- `find.py`, `hash.py`, `sort.py`, `string_types.py`, `uncased.py`, `cipher.py`, `utf8_*.py`, and `doctests.py` — per-family tests.
- `base.py` and `utf8_helpers.py` are shared helpers; `conftest.py` holds the pytest configuration.
- The modules are flat and import each other by basename, as `from base import ...`, so none may share a stdlib module's name.
- Install the `test` group and run the suite with `python -X faulthandler -m pytest -x`, as the CI does.
- Seeded tests take the `seed` fixture: 42, a numeric `STRINGZILLA_SEED` replaces it, `random` draws one, and pytest's header prints it as `- Seed: <n>`.
- `STRINGZILLA_SCALE` applies as in C++, and `STRINGZILLA_FILTER` keeps the tests whose node id it matches, alongside `pytest -k`.

## JavaScript

- `main.js` — Node test runner, invoked with `node --test`.
