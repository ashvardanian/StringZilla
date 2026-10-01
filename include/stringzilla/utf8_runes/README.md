# UTF-8 Runes: Codepoint Counting, Decoding, and Indexing

This directory holds the kernels behind `sz_utf8_count_best`, `sz_utf8_decode_best`, and `sz_utf8_seek_best`, the low-level rune operations that count the codepoints in a UTF-8 string, decode each variable-length byte sequence into its 32-bit codepoint, and locate the byte offset of the Nth codepoint.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86, and each `_best` dispatch point runs the best kernel among the capabilities its caller passes.

## Methodology

Numbers are throughput measured with `bench/utf8_traverse.cpp` over the full multilingual `xlsum.csv` corpus, reporting the median of repeated runs.
Each table fixes one input shape; its columns are the rune operations and its rows are a backend on a chip, so reading down a column compares the same operation across the backend ladder while reading across a row compares operations on one backend.
Results are split into a Short Words workload (whitespace-delimited tokens averaging a few bytes, reported in MB/s) and a Long Lines workload (full text lines, reported in GB/s) to expose how each kernel scales with token length.
A `↑` cell means there is no dedicated kernel for that operation at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_count_best` | `sz_utf8_decode_best` | `sz_utf8_seek_best` |
| :--------------- | -------------------: | --------------------: | ------------------: |
| Serial @ Xeon4   |           164.5 MB/s |            212.8 MB/s |          97.56 MB/s |
| Haswell @ Xeon4  |           200.6 MB/s |            137.9 MB/s |          114.0 MB/s |
| Ice Lake @ Xeon4 |           237.3 MB/s |            244.1 MB/s |          125.6 MB/s |
| NEON @ Graviton4 |                    … |                     … |                   … |
| SVE2 @ Graviton4 |                    … |                     … |                   … |
| SVE @ Graviton3  |                    … |                     … |                   … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_utf8_count_best` | `sz_utf8_decode_best` | `sz_utf8_seek_best` |
| :--------------- | -------------------: | --------------------: | ------------------: |
| Serial @ Xeon4   |            1.10 GB/s |             0.38 GB/s |           0.51 GB/s |
| Haswell @ Xeon4  |            5.92 GB/s |             0.53 GB/s |           5.19 GB/s |
| Ice Lake @ Xeon4 |            4.86 GB/s |             1.49 GB/s |           5.17 GB/s |
| NEON @ Graviton4 |                    … |                     … |                   … |
| SVE2 @ Graviton4 |                    … |                     … |                   … |
| SVE @ Graviton3  |                    … |                     … |                   … |

> Measured June 26th, 2026.
