# Uncased: Case-Insensitive UTF-8 Search

This directory holds the kernels behind `sz_utf8_uncased_search_best`, which locates a needle inside a haystack while ignoring case differences.
The match is done without pre-folding the haystack: each candidate position is compared under Unicode case folding on the fly, so the original bytes are never rewritten.
Every operation has a serial baseline plus per-ISA SIMD backends, and `sz_utf8_uncased_search_best` runs the best kernel among the capabilities its caller passes.
The needle is analysed once, by the serial `sz_utf8_uncased_needle_init_best`, and every search backend only reads that analysis.

## Methodology

`bench/utf8_uncased.cpp` measures throughput over the first 64 MB of the multilingual `xlsum.csv` corpus by default.
`STRINGWARS_BYTES` controls the input size; `STRINGWARS_TOKENS` selects words, lines, or the file as one input.
The Serial row is the reference; there is no Standard row here, since no standard library ships a Unicode case-insensitive substring search.
Results are split into a Short Words workload (tokens averaging a handful of bytes) and a Long Lines workload (full sentences).
A `↑` cell means there is no dedicated `sz_utf8_uncased_search_<isa>` kernel at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend                   | `sz_utf8_uncased_search_best` |
| :------------------------ | ----------------------------: |
| Serial @ 1× Intel Xeon6   |                     0.20 GB/s |
| Haswell @ 1× Intel Xeon6  |                     2.35 GB/s |
| Ice Lake @ 1× Intel Xeon6 |                     3.76 GB/s |
| Serial @ 1× Apple M5 Pro  |                     0.45 GB/s |
| NEON @ 1× Apple M5 Pro    |                     2.47 GB/s |
| NEON @ 1× AWS Graviton4   |                             … |
| SVE2 @ 1× AWS Graviton4   |                             … |
| SVE @ 1× AWS Graviton3    |                             … |

## Long Lines

| Backend                   | `sz_utf8_uncased_search_best` |
| :------------------------ | ----------------------------: |
| Serial @ 1× Intel Xeon6   |                     0.20 GB/s |
| Haswell @ 1× Intel Xeon6  |                     3.21 GB/s |
| Ice Lake @ 1× Intel Xeon6 |                     7.09 GB/s |
| Serial @ 1× Apple M5 Pro  |                     0.46 GB/s |
| NEON @ 1× Apple M5 Pro    |                     4.06 GB/s |
| NEON @ 1× AWS Graviton4   |                             … |
| SVE2 @ 1× AWS Graviton4   |                             … |
| SVE @ 1× AWS Graviton3    |                             … |
