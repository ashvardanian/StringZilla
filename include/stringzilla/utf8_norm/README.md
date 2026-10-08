# Norm: UTF-8 Unicode Normalization

This directory holds the kernels behind `sz_utf8_norm_best` and `sz_utf8_find_denormalized_best`, which bring UTF-8 text into a UAX-15 normalization form — NFC, NFD, NFKC, or NFKD.
Normalization reorders combining marks and composes or decomposes characters so that strings that look the same compare equal byte for byte.
The companion quick-check scan finds the first character that breaks the requested form, so callers can skip the rewrite when a string is already normalized.
Every operation has a serial baseline plus per-ISA SIMD backends, and each `_best` dispatch point runs the best kernel among the capabilities its caller passes.

## Methodology

`bench/utf8_norm.cpp` measures throughput over the first 64 MB of the multilingual `xlsum.csv` corpus by default.
`STRINGWARS_BYTES` controls the input size; `STRINGWARS_TOKENS` selects words, lines, or the file as one input.
Metal rows use a standalone driver on the same corpus slice, reporting the median of three runs with a stream synchronization after each input.
The Serial row is the reference; there is no Standard row here, since no standard library ships Unicode normalization.
For a comparison against ICU, see the main project README.
Results are split into Short Words, Long Lines, and Whole File workloads.
A `—` cell means the quick-check API has no GPU kernel.
A `↑` cell means there is no dedicated kernel for that operation at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend                   | `sz_utf8_norm_best` | `sz_utf8_find_denormalized_best` |
| :------------------------ | ------------------: | -------------------------------: |
| Serial @ 1× Intel Xeon6   |          188.8 MB/s |                       244.0 MB/s |
| Haswell @ 1× Intel Xeon6  |          233.5 MB/s |                       295.2 MB/s |
| Skylake @ 1× Intel Xeon6  |          232.6 MB/s |                       287.4 MB/s |
| Ice Lake @ 1× Intel Xeon6 |          230.7 MB/s |                       284.4 MB/s |
| Serial @ 1× Apple M5 Pro  |          414.4 MB/s |                       488.5 MB/s |
| NEON @ 1× Apple M5 Pro    |          466.4 MB/s |                       610.5 MB/s |
| NEON @ 1× AWS Graviton4   |                   … |                                … |
| SVE2 @ 1× AWS Graviton4   |                   … |                                … |
| SVE @ 1× AWS Graviton3    |                   … |                                … |

## Long Lines

| Backend                   | `sz_utf8_norm_best` | `sz_utf8_find_denormalized_best` |
| :------------------------ | ------------------: | -------------------------------: |
| Serial @ 1× Intel Xeon6   |          325.8 MB/s |                       381.1 MB/s |
| Haswell @ 1× Intel Xeon6  |          643.5 MB/s |                       773.5 MB/s |
| Skylake @ 1× Intel Xeon6  |          645.7 MB/s |                       776.0 MB/s |
| Ice Lake @ 1× Intel Xeon6 |          653.3 MB/s |                       786.4 MB/s |
| Serial @ 1× Apple M5 Pro  |          749.3 MB/s |                       819.3 MB/s |
| NEON @ 1× Apple M5 Pro    |        1,640.4 MB/s |                     1,857.8 MB/s |
| Metal @ Apple M5 Pro      |           24.4 MB/s |                                — |
| NEON @ 1× AWS Graviton4   |                   … |                                … |
| SVE2 @ 1× AWS Graviton4   |                   … |                                … |
| SVE @ 1× AWS Graviton3    |                   … |                                … |

## Whole File

The corpus slice normalized to NFC in one call.

| Backend                   | `sz_utf8_norm_best` |
| :------------------------ | ------------------: |
| Serial @ 1× Intel Xeon6   |          343.4 MB/s |
| Haswell @ 1× Intel Xeon6  |          663.9 MB/s |
| Skylake @ 1× Intel Xeon6  |          660.1 MB/s |
| Ice Lake @ 1× Intel Xeon6 |          669.2 MB/s |
| Serial @ 1× Apple M5 Pro  |          752.1 MB/s |
| NEON @ 1× Apple M5 Pro    |        1,647.4 MB/s |
| Metal @ Apple M5 Pro      |        5,566.1 MB/s |
| CUDA @ 18× Nvidia SM103   |        2,217.0 MB/s |
