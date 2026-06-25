# StringZilla Tests

Unit tests that validate correctness across every SIMD backend and language binding against serial and STL baselines.
Each C++ translation unit exercises one kernel family, and the Python suite mirrors it module-for-module.

## C++ and CUDA

- `stringzilla.cpp` — C++ API against STL baselines.
- `stringzillas.cpp` / `stringzillas.cu` — parallel CPU and CUDA backend tests.
- `hash.cpp`, `find.cpp`, `sort.cpp`, `string.cpp`, `uncased.cpp` — per-family kernel tests.
- `utf8_runes.cpp`, `utf8_words.cpp`, `utf8_graphemes.cpp`, `utf8_sentences.cpp`, `utf8_linewraps.cpp`, `utf8_norm.cpp`, `utf8_tokens.cpp` — UTF-8 decode and segmentation tests.
- `stringzilla.hpp` and `utf8.hpp` are the shared harnesses; `fingerprints.cuh` and `similarities.cuh` are the CUDA harnesses.

## Python

The Python modules mirror the C++ translation units one-for-one and run under pytest.

- `find.py`, `hash.py`, `sort.py`, `string.py`, `uncased.py`, `utf8_*.py`, `doctests.py`, `stringzilla.py`, `stringzillas.py` — per-family tests.
- `helpers.py` and `utf8_helpers.py` are shared helpers; `conftest.py` holds the pytest configuration.
- This directory is a Python package via `__init__.py`, so the prefix-less modules namespace as `test.*` and never shadow stdlib names.
- Run the suite with `pytest test/`.

## JavaScript

- `stringzilla.js` — Node test runner, invoked with `node --test`.
