# Uncased: Case-Insensitive UTF-8 Search

This directory holds the kernels behind `sz_utf8_uncased_search_best`, which locates a needle inside a haystack while ignoring case differences.
The match is done without pre-folding the haystack: each candidate position is compared under Unicode case folding on the fly, so the original bytes are never rewritten.
Every operation has a serial baseline plus per-ISA SIMD backends, and `sz_utf8_uncased_search_best` runs the best kernel among the capabilities its caller passes.
The needle is analysed once, by the serial `sz_utf8_uncased_needle_init_best`, and every search backend only reads that analysis.

## Methodology

Numbers are throughput in GB/s, measured with `bench/utf8_uncased.cpp` over the full multilingual `xlsum.csv` corpus, reporting the median of repeated runs.
Each table fixes one input shape; its single column is the `sz_utf8_uncased_search_best` operation and its rows are a backend on a chip, so reading down the column compares the backend ladder on one fixed input shape.
The Serial row is the reference; there is no Standard row here, since no standard library ships a Unicode case-insensitive substring search.
Results are split into a Short Words workload (tokens averaging a handful of bytes) and a Long Lines workload (full sentences).
A `↑` cell means there is no dedicated `sz_utf8_uncased_search_<isa>` kernel at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_uncased_search_best` |
| :--------------- | ----------------------------: |
| Serial @ Xeon4   |                     0.08 GB/s |
| Haswell @ Xeon4  |                     3.30 GB/s |
| Ice Lake @ Xeon4 |                     3.10 GB/s |
| NEON @ Graviton4 |                             … |
| SVE2 @ Graviton4 |                             … |
| SVE @ Graviton3  |                             … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_utf8_uncased_search_best` |
| :--------------- | ----------------------------: |
| Serial @ Xeon4   |                     0.10 GB/s |
| Haswell @ Xeon4  |                     2.20 GB/s |
| Ice Lake @ Xeon4 |                     4.28 GB/s |
| NEON @ Graviton4 |                             … |
| SVE2 @ Graviton4 |                             … |
| SVE @ Graviton3  |                             … |

> Measured June 26th, 2026.
