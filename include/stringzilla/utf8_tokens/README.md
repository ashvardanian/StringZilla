# UTF-8 Tokens: Whitespace and Newline Splitting

This directory holds the kernels behind `sz_utf8_whitespaces` and `sz_utf8_newlines`, the fast token splitters that cut a UTF-8 string at runs of Unicode whitespace or at line boundaries, recognizing the full set of UTF-8 space and newline codepoints rather than only the ASCII ones.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86, and the dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput in MB/s, measured with `bench/utf8_iterate.cpp` over the full multilingual `xlsum.csv` corpus, reporting the median of repeated runs.
Each table fixes one input shape; its columns are the Whitespace Split and Newline Split operations and its rows are a backend on a chip, so reading down a column compares the same operation across the backend ladder while reading across a row compares operations on one backend.
Results are split into a Short Words workload (whitespace-delimited tokens averaging a few bytes) and a Long Lines workload (full text lines) to expose how each kernel scales with token length.
A `↑` cell means there is no dedicated kernel for that operation at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_whitespaces` | `sz_utf8_newlines` |
| :--------------- | --------------------: | -----------------: |
| Serial @ Xeon4   |            131.7 MB/s |         143.7 MB/s |
| Haswell @ Xeon4  |            150.4 MB/s |          94.9 MB/s |
| Ice Lake @ Xeon4 |            207.5 MB/s |         212.7 MB/s |
| NEON @ Graviton4 |                     … |                  … |
| SVE2 @ Graviton4 |                     … |                  … |
| SVE @ Graviton3  |                     … |                  … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_utf8_whitespaces` | `sz_utf8_newlines` |
| :--------------- | --------------------: | -----------------: |
| Serial @ Xeon4   |            269.3 MB/s |         265.0 MB/s |
| Haswell @ Xeon4  |            315.5 MB/s |        2662.4 MB/s |
| Ice Lake @ Xeon4 |           1914.9 MB/s |        4147.2 MB/s |
| NEON @ Graviton4 |                     … |                  … |
| SVE2 @ Graviton4 |                     … |                  … |
| SVE @ Graviton3  |                     … |                  … |

> Measured June 26th, 2026.
