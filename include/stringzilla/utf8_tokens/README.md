# UTF-8 Tokens: Whitespace and Newline Splitting

This directory holds the kernels behind `sz_utf8_whitespaces_best` and `sz_utf8_newlines_best`, the fast token splitters that cut a UTF-8 string at runs of Unicode whitespace or at line boundaries, recognizing the full set of UTF-8 space and newline codepoints rather than only the ASCII ones.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86, and each `_best` dispatch point runs the best kernel among the capabilities its caller passes.

## Methodology

Numbers are throughput in MB/s, measured with `bench/utf8_scan.cpp` over the full multilingual `xlsum.csv` corpus, reporting the median of repeated runs.
Each table fixes one input shape; its columns are the Whitespace Split and Newline Split operations and its rows are a backend on a chip, so reading down a column compares the same operation across the backend ladder while reading across a row compares operations on one backend.
Results are split into a Short Words workload (whitespace-delimited tokens averaging a few bytes) and a Long Lines workload (full text lines) to expose how each kernel scales with token length.
A `↑` cell means there is no dedicated kernel for that operation at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_whitespaces_best` | `sz_utf8_newlines_best` |
| :--------------- | -------------------------: | ----------------------: |
| Serial @ Xeon4   |                 125.6 MB/s |              137.0 MB/s |
| Haswell @ Xeon4  |                 143.4 MB/s |               90.5 MB/s |
| Ice Lake @ Xeon4 |                 197.9 MB/s |              202.8 MB/s |
| NEON @ Graviton4 |                          … |                       … |
| SVE2 @ Graviton4 |                          … |                       … |
| SVE @ Graviton3  |                          … |                       … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_utf8_whitespaces_best` | `sz_utf8_newlines_best` |
| :--------------- | -------------------------: | ----------------------: |
| Serial @ Xeon4   |                 256.8 MB/s |              252.7 MB/s |
| Haswell @ Xeon4  |                 300.9 MB/s |             2539.1 MB/s |
| Ice Lake @ Xeon4 |                1826.2 MB/s |             3955.1 MB/s |
| NEON @ Graviton4 |                          … |                       … |
| SVE2 @ Graviton4 |                          … |                       … |
| SVE @ Graviton3  |                          … |                       … |

> Measured June 26th, 2026.
