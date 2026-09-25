# StringZilla Tests

Unit tests that validate correctness across every SIMD backend and language binding against serial and STL baselines.
Each C++ translation unit exercises one kernel family, and the Python suite mirrors it module-for-module.

## C++ and CUDA

- `main.cpp` / `main.cu` — C++ API against STL baselines, and the CUDA backends of the core families.
- `hash.cpp`, `find.cpp`, `sort.cpp`, `string.cpp`, `uncased.cpp`, `cipher.cpp` — per-family kernel tests.
- `levenshtein.cpp` / `levenshtein.cu`, `overlap.cpp` / `overlap.cu`, and `substrings.cpp` / `substrings.cu` — edit distances, window overlap, and multi-pattern search, each family's CPU backends beside its CUDA one.
- `utf8_runes.cpp`, `utf8_wordbreaks.cpp`, `utf8_graphemes.cpp`, `utf8_sentences.cpp`, `utf8_linebreaks.cpp`, `utf8_norm.cpp`, `utf8_tokens.cpp` — UTF-8 decode and segmentation tests.
- `harness.hpp` and `utf8.hpp` are the shared harnesses.
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
