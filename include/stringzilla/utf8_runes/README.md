# UTF-8 Runes: Codepoint Counting, Decoding, and Indexing

This directory holds the kernels behind `sz_utf8_count_best`, `sz_utf8_decode_best`, and `sz_utf8_seek_best`, the low-level rune operations that count the codepoints in a UTF-8 string, decode each variable-length byte sequence into its 32-bit codepoint, and locate the byte offset of the Nth codepoint.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86 plus `neon` and `sve2` on Arm, and each `_best` dispatch point runs the best kernel among the capabilities its caller passes.

## Methodology

`bench/utf8_traverse.cpp` measures throughput over the first 64 MiB of the multilingual `xlsum.csv` corpus by default.
`STRINGWARS_BYTES` controls the input size; `STRINGWARS_TOKENS` selects words, lines, or the file as one input.
Results are split into a Short Words workload (whitespace-delimited tokens) and a Long Lines workload (full text lines) to expose how each kernel scales with token length.
A `↑` cell means there is no dedicated kernel for that operation at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_count_best` | `sz_utf8_decode_best` | `sz_utf8_seek_best` |
| :--------------- | -------------------: | --------------------: | ------------------: |
| Serial @ Xeon4   |           164.5 MB/s |            212.8 MB/s |          97.56 MB/s |
| Haswell @ Xeon4  |           200.6 MB/s |            137.9 MB/s |          114.0 MB/s |
| Ice Lake @ Xeon4 |           237.3 MB/s |            244.1 MB/s |          125.6 MB/s |
| Serial @ M5 Pro  |         1016.0 MiB/s |           850.0 MiB/s |         609.5 MiB/s |
| NEON @ M5 Pro    |         1027.1 MiB/s |           688.5 MiB/s |         616.0 MiB/s |
| NEON @ Graviton4 |                    … |                     … |                   … |
| SVE2 @ Graviton4 |                    … |                     … |                   … |
| SVE @ Graviton3  |                    … |                     … |                   … |

## Long Lines

| Backend          | `sz_utf8_count_best` | `sz_utf8_decode_best` | `sz_utf8_seek_best` |
| :--------------- | -------------------: | --------------------: | ------------------: |
| Serial @ Xeon4   |            1.10 GB/s |             0.38 GB/s |           0.51 GB/s |
| Haswell @ Xeon4  |            5.92 GB/s |             0.53 GB/s |           5.19 GB/s |
| Ice Lake @ Xeon4 |            4.86 GB/s |             1.49 GB/s |           5.17 GB/s |
| Serial @ M5 Pro  |           8.20 GiB/s |            1.36 GiB/s |          1.51 GiB/s |
| NEON @ M5 Pro    |          76.48 GiB/s |            1.18 GiB/s |         26.39 GiB/s |
| NEON @ Graviton4 |                    … |                     … |                   … |
| SVE2 @ Graviton4 |                    … |                     … |                   … |
| SVE @ Graviton3  |                    … |                     … |                   … |
