# UTF-8 Words: UAX-29 Word Boundary Iteration

This directory holds the kernels behind `sz_utf8_wordbreaks_best`, which walks a UTF-8 string and yields each word as defined by the Unicode UAX-29 word-boundary rules, so "don't" or a CJK run is split the way a human reader expects rather than on raw spaces.
Each call writes one byte length per word, and the words tile the input, so a caller that fills its buffer resumes from the sum of the lengths it got.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86 plus `neon` and `sve2` on Arm, and `sz_utf8_wordbreaks_best` runs the best kernel among the capabilities its caller passes.

## Methodology

`bench/utf8_segment.cpp` measures throughput over the first 64 MiB of the multilingual `xlsum.csv` corpus by default.
`STRINGWARS_BYTES` controls the input size; `STRINGWARS_TOKENS` selects words, lines, or the file as one input.
Results are split into a Short Words workload (whitespace-delimited tokens averaging a few bytes) and a Long Lines workload (full text lines) to expose how each kernel scales with token length.
A `↑` cell means there is no dedicated kernel at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_wordbreaks_best` |
| :--------------- | ------------------------: |
| Serial @ Xeon4   |                 69.2 MB/s |
| Haswell @ Xeon4  |                 13.5 MB/s |
| Ice Lake @ Xeon4 |                 34.5 MB/s |
| Serial @ M5 Pro  |               405.7 MiB/s |
| NEON @ M5 Pro    |               404.3 MiB/s |
| NEON @ Graviton4 |                         … |
| SVE2 @ Graviton4 |                         … |
| SVE @ Graviton3  |                         … |

## Long Lines

| Backend          | `sz_utf8_wordbreaks_best` |
| :--------------- | ------------------------: |
| Serial @ Xeon4   |                 43.8 MB/s |
| Haswell @ Xeon4  |                 41.1 MB/s |
| Ice Lake @ Xeon4 |                110.4 MB/s |
| Serial @ M5 Pro  |               302.9 MiB/s |
| NEON @ M5 Pro    |               305.0 MiB/s |
| NEON @ Graviton4 |                         … |
| SVE2 @ Graviton4 |                         … |
| SVE @ Graviton3  |                         … |
