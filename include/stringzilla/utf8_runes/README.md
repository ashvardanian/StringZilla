# UTF-8 Runes: Codepoint Counting, Decoding, and Indexing

This directory holds the kernels behind `sz_utf8_count`, `sz_utf8_decode`, and `sz_utf8_find_nth`, the low-level rune operations that count the codepoints in a UTF-8 string, decode each variable-length byte sequence into its 32-bit codepoint, and locate the byte offset of the Nth codepoint.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86, and the dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput measured with `bench/utf8_iterate.cpp` over the full multilingual `xlsum.csv` corpus, reporting the median of repeated runs.
Each table fixes one input shape; its columns are the rune operations and its rows are a backend on a chip, so reading down a column compares the same operation across the backend ladder while reading across a row compares operations on one backend.
Results are split into a Short Words workload (whitespace-delimited tokens averaging a few bytes, reported in MB/s) and a Long Lines workload (full text lines, reported in GB/s) to expose how each kernel scales with token length.
A `↑` cell means there is no dedicated kernel for that operation at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_count` | `sz_utf8_decode` | `sz_utf8_find_nth` |
| :--------------- | --------------: | ---------------: | -----------------: |
| Serial @ Xeon4   |      172.5 MB/s |       223.1 MB/s |         102.3 MB/s |
| Haswell @ Xeon4  |      210.3 MB/s |       144.6 MB/s |         119.5 MB/s |
| Ice Lake @ Xeon4 |      248.8 MB/s |       256.0 MB/s |         131.7 MB/s |
| NEON @ Graviton4 |               … |                … |                  … |
| SVE2 @ Graviton4 |               … |                … |                  … |
| SVE @ Graviton3  |               … |                … |                  … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_utf8_count` | `sz_utf8_decode` | `sz_utf8_find_nth` |
| :--------------- | --------------: | ---------------: | -----------------: |
| Serial @ Xeon4   |       1.18 GB/s |        0.41 GB/s |          0.55 GB/s |
| Haswell @ Xeon4  |       6.36 GB/s |        0.57 GB/s |          5.57 GB/s |
| Ice Lake @ Xeon4 |       5.22 GB/s |        1.60 GB/s |          5.55 GB/s |
| NEON @ Graviton4 |               … |                … |                  … |
| SVE2 @ Graviton4 |               … |                … |                  … |
| SVE @ Graviton3  |               … |                … |                  … |

> Measured June 26th, 2026.
