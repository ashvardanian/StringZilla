# UTF-8 Runes: Codepoint Counting, Decoding, and Indexing

This directory holds the kernels behind `sz_utf8_count_best`, `sz_utf8_decode_best`, and `sz_utf8_seek_best`, the low-level rune operations that count the codepoints in a UTF-8 string, decode each variable-length byte sequence into its 32-bit codepoint, and locate the byte offset of the Nth codepoint.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86 plus `neon` and `sve2` on Arm, and each `_best` dispatch point runs the best kernel among the capabilities its caller passes.

## Methodology

`bench/utf8_traverse.cpp` measures throughput over the first 64 MB of the multilingual `xlsum.csv` corpus by default.
`STRINGWARS_BYTES` controls the input size; `STRINGWARS_TOKENS` selects words, lines, or the file as one input.
Results are split into a Short Words workload (whitespace-delimited tokens) and a Long Lines workload (full text lines) to expose how each kernel scales with token length.
A `↑` cell means there is no dedicated kernel for that operation at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend                   | `sz_utf8_count_best` | `sz_utf8_decode_best` | `sz_utf8_seek_best` |
| :------------------------ | -------------------: | --------------------: | ------------------: |
| Serial @ 1× Intel Xeon6   |           450.5 MB/s |            372.3 MB/s |          273.9 MB/s |
| Haswell @ 1× Intel Xeon6  |           474.3 MB/s |            236.0 MB/s |          285.8 MB/s |
| Ice Lake @ 1× Intel Xeon6 |         3,013.6 MB/s |            453.8 MB/s |          282.9 MB/s |
| Serial @ 1× Apple M5 Pro  |          1016.0 MB/s |            850.0 MB/s |          609.5 MB/s |
| NEON @ 1× Apple M5 Pro    |          1027.1 MB/s |            688.5 MB/s |          616.0 MB/s |
| NEON @ 1× AWS Graviton4   |                    … |                     … |                   … |
| SVE2 @ 1× AWS Graviton4   |                    … |                     … |                   … |
| SVE @ 1× AWS Graviton3    |                    … |                     … |                   … |

## Long Lines

| Backend                   | `sz_utf8_count_best` | `sz_utf8_decode_best` | `sz_utf8_seek_best` |
| :------------------------ | -------------------: | --------------------: | ------------------: |
| Serial @ 1× Intel Xeon6   |         2,065.4 MB/s |            681.5 MB/s |          776.7 MB/s |
| Haswell @ 1× Intel Xeon6  |        23,541.8 MB/s |          1,045.5 MB/s |       15,114.2 MB/s |
| Ice Lake @ 1× Intel Xeon6 |        28,190.7 MB/s |          1,999.9 MB/s |       16,240.6 MB/s |
| Serial @ 1× Apple M5 Pro  |            8.20 GB/s |             1.36 GB/s |           1.51 GB/s |
| NEON @ 1× Apple M5 Pro    |           76.48 GB/s |             1.18 GB/s |          26.39 GB/s |
| NEON @ 1× AWS Graviton4   |                    … |                     … |                   … |
| SVE2 @ 1× AWS Graviton4   |                    … |                     … |                   … |
| SVE @ 1× AWS Graviton3    |                    … |                     … |                   … |
