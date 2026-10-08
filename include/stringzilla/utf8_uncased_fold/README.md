# Uncased Fold: UTF-8 Case Folding

This directory holds the kernels behind `sz_utf8_uncased_fold_best`, which rewrites a UTF-8 string into its Unicode case-folded form.
Case folding maps every character to a canonical lowercase-like representative so that two strings that differ only in case become byte-identical, which is the preparation step for case-insensitive comparison and bucketing.
Every operation has a serial baseline plus per-ISA SIMD backends, and `sz_utf8_uncased_fold_best` runs the best kernel among the capabilities its caller passes.

## Methodology

`bench/utf8_uncased.cpp` measures throughput over the first 64 MB of the multilingual `xlsum.csv` corpus by default.
`STRINGWARS_BYTES` controls the input size; `STRINGWARS_TOKENS` selects words, lines, or the file as one input.
Metal rows use a standalone driver on the same corpus slice, reporting the median of three runs with a stream synchronization after each input.
The Serial row is the reference; there is no Standard row here, since no standard library ships Unicode case folding.
Results are split into a Short Words workload (tokens averaging a handful of bytes), a Long Lines workload (full sentences), and a Whole File workload (the entire corpus folded in one pass).
A `↑` cell means there is no dedicated `sz_utf8_uncased_fold_<isa>` kernel at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend                   | `sz_utf8_uncased_fold_best` |
| :------------------------ | --------------------------: |
| Serial @ 1× Intel Xeon6   |                  182.3 MB/s |
| Haswell @ 1× Intel Xeon6  |                  197.5 MB/s |
| Ice Lake @ 1× Intel Xeon6 |                  350.0 MB/s |
| Serial @ 1× Apple M5 Pro  |                  392.5 MB/s |
| NEON @ 1× Apple M5 Pro    |                  394.3 MB/s |
| NEON @ 1× AWS Graviton4   |                           … |
| SVE2 @ 1× AWS Graviton4   |                           … |
| SVE @ 1× AWS Graviton3    |                           … |

## Long Lines

| Backend                   | `sz_utf8_uncased_fold_best` |
| :------------------------ | --------------------------: |
| Serial @ 1× Intel Xeon6   |                  257.0 MB/s |
| Haswell @ 1× Intel Xeon6  |                1,193.0 MB/s |
| Ice Lake @ 1× Intel Xeon6 |                1,382.4 MB/s |
| Serial @ 1× Apple M5 Pro  |                  550.8 MB/s |
| NEON @ 1× Apple M5 Pro    |                1,233.6 MB/s |
| Metal @ Apple M5 Pro      |                   53.5 MB/s |
| NEON @ 1× AWS Graviton4   |                           … |
| SVE2 @ 1× AWS Graviton4   |                           … |
| SVE @ 1× AWS Graviton3    |                           … |

## Whole File

| Backend                   | `sz_utf8_uncased_fold_best` |
| :------------------------ | --------------------------: |
| Serial @ 1× Intel Xeon6   |                  273.7 MB/s |
| Haswell @ 1× Intel Xeon6  |                1,224.7 MB/s |
| Ice Lake @ 1× Intel Xeon6 |                1,388.5 MB/s |
| Serial @ 1× Apple M5 Pro  |                  551.5 MB/s |
| NEON @ 1× Apple M5 Pro    |                1,218.7 MB/s |
| Metal @ Apple M5 Pro      |               16,810.0 MB/s |
| CUDA @ 18× Nvidia SM103   |                4,384.8 MB/s |
| NEON @ 1× AWS Graviton4   |                           … |
| SVE2 @ 1× AWS Graviton4   |                           … |
| SVE @ 1× AWS Graviton3    |                           … |
