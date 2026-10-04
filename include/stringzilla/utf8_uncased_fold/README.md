# Uncased Fold: UTF-8 Case Folding

This directory holds the kernels behind `sz_utf8_uncased_fold_best`, which rewrites a UTF-8 string into its Unicode case-folded form.
Case folding maps every character to a canonical lowercase-like representative so that two strings that differ only in case become byte-identical, which is the preparation step for case-insensitive comparison and bucketing.
Every operation has a serial baseline plus per-ISA SIMD backends, and `sz_utf8_uncased_fold_best` runs the best kernel among the capabilities its caller passes.

## Methodology

`bench/utf8_uncased.cpp` measures throughput over the first 64 MiB of the multilingual `xlsum.csv` corpus by default.
`STRINGWARS_BYTES` controls the input size; `STRINGWARS_TOKENS` selects words, lines, or the file as one input.
The Serial row is the reference; there is no Standard row here, since no standard library ships Unicode case folding.
Results are split into a Short Words workload (tokens averaging a handful of bytes), a Long Lines workload (full sentences), and a Whole File workload (the entire corpus folded in one pass).
A `↑` cell means there is no dedicated `sz_utf8_uncased_fold_<isa>` kernel at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_uncased_fold_best` |
| :--------------- | --------------------------: |
| Serial @ Xeon4   |                  99.37 MB/s |
| Haswell @ Xeon4  |                   57.0 MB/s |
| Ice Lake @ Xeon4 |                  101.7 MB/s |
| Serial @ M5 Pro  |                 392.5 MiB/s |
| NEON @ M5 Pro    |                 394.3 MiB/s |
| NEON @ Graviton4 |                           … |
| SVE2 @ Graviton4 |                           … |
| SVE @ Graviton3  |                           … |

## Long Lines

| Backend          | `sz_utf8_uncased_fold_best` |
| :--------------- | --------------------------: |
| Serial @ Xeon4   |                  168.3 MB/s |
| Haswell @ Xeon4  |                  445.0 MB/s |
| Ice Lake @ Xeon4 |                  860.1 MB/s |
| Serial @ M5 Pro  |                 550.8 MiB/s |
| NEON @ M5 Pro    |               1,233.6 MiB/s |
| NEON @ Graviton4 |                           … |
| SVE2 @ Graviton4 |                           … |
| SVE @ Graviton3  |                           … |

## Whole File

| Backend          | `sz_utf8_uncased_fold_best` |
| :--------------- | --------------------------: |
| Serial @ Xeon4   |                  325.9 MB/s |
| Haswell @ Xeon4  |                  871.5 MB/s |
| Ice Lake @ Xeon4 |                 1289.4 MB/s |
| Serial @ M5 Pro  |                 551.5 MiB/s |
| NEON @ M5 Pro    |               1,218.7 MiB/s |
| NEON @ Graviton4 |                           … |
| SVE2 @ Graviton4 |                           … |
| SVE @ Graviton3  |                           … |
