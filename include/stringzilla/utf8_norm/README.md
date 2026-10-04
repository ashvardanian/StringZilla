# Norm: UTF-8 Unicode Normalization

This directory holds the kernels behind `sz_utf8_norm_best` and `sz_utf8_find_denormalized_best`, which bring UTF-8 text into a UAX-15 normalization form — NFC, NFD, NFKC, or NFKD.
Normalization reorders combining marks and composes or decomposes characters so that strings that look the same compare equal byte for byte.
The companion quick-check scan finds the first character that breaks the requested form, so callers can skip the rewrite when a string is already normalized.
Every operation has a serial baseline plus per-ISA SIMD backends, and each `_best` dispatch point runs the best kernel among the capabilities its caller passes.

## Methodology

`bench/utf8_norm.cpp` measures throughput over the first 64 MiB of the multilingual `xlsum.csv` corpus by default.
`STRINGWARS_BYTES` controls the input size; `STRINGWARS_TOKENS` selects words, lines, or the file as one input.
The Serial row is the reference; there is no Standard row here, since no standard library ships Unicode normalization.
For a comparison against ICU, see the main project README.
Results are split into Short Words, Long Lines, and Whole File workloads.
A `↑` cell means there is no dedicated kernel for that operation at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_norm_best` | `sz_utf8_find_denormalized_best` |
| :--------------- | ------------------: | -------------------------------: |
| Serial @ Xeon4   |          107.0 MB/s |                       125.4 MB/s |
| Haswell @ Xeon4  |          115.8 MB/s |                       201.8 MB/s |
| Skylake @ Xeon4  |          116.9 MB/s |                       207.7 MB/s |
| Ice Lake @ Xeon4 |          116.3 MB/s |                       202.8 MB/s |
| Serial @ M5 Pro  |         414.4 MiB/s |                      488.5 MiB/s |
| NEON @ M5 Pro    |         466.4 MiB/s |                      610.5 MiB/s |
| NEON @ Graviton4 |                   … |                                … |
| SVE2 @ Graviton4 |                   … |                                … |
| SVE @ Graviton3  |                   … |                                … |

## Long Lines

| Backend          | `sz_utf8_norm_best` | `sz_utf8_find_denormalized_best` |
| :--------------- | ------------------: | -------------------------------: |
| Serial @ Xeon4   |          205.5 MB/s |                       254.1 MB/s |
| Haswell @ Xeon4  |          359.1 MB/s |                       469.8 MB/s |
| Skylake @ Xeon4  |          355.4 MB/s |                       506.1 MB/s |
| Ice Lake @ Xeon4 |          362.3 MB/s |                       505.6 MB/s |
| Serial @ M5 Pro  |         749.3 MiB/s |                      819.3 MiB/s |
| NEON @ M5 Pro    |       1,640.4 MiB/s |                    1,857.8 MiB/s |
| NEON @ Graviton4 |                   … |                                … |
| SVE2 @ Graviton4 |                   … |                                … |
| SVE @ Graviton3  |                   … |                                … |

## Whole File

The corpus slice normalized to NFC in one call.

| Backend         | `sz_utf8_norm_best` |
| :-------------- | ------------------: |
| Serial @ M5 Pro |         752.1 MiB/s |
| NEON @ M5 Pro   |       1,647.4 MiB/s |
