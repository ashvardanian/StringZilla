# UTF-8 Line Wraps: UAX-14 Line Break Opportunity Iteration

This directory holds the kernels behind `sz_utf8_linebreaks_best`, which walks a UTF-8 string and yields each line-break opportunity as defined by the Unicode UAX-14 rules, marking the positions where a text layout engine is allowed to wrap rather than splitting on raw spaces or hard newlines alone.
Each call writes one byte length per line, and the lines tile the input, so a caller that fills its buffer resumes from the sum of the lengths it got.
Each operation has a serial baseline plus per-ISA SIMD backends, like `haswell` and `icelake` on x86, and `neon` and `sve2` on Arm.
`sz_utf8_linebreaks_best` runs the best kernel among the capabilities its caller passes.

## Methodology

`bench/utf8_segment.cpp` measures throughput over the first 64 MB of the multilingual `xlsum.csv` corpus by default.
`STRINGWARS_BYTES` controls the input size; `STRINGWARS_TOKENS` selects words, lines, or the file as one input.
Results are split into a Short Words workload (whitespace-delimited tokens averaging a few bytes) and a Long Lines workload (full text lines) to expose how each kernel scales with token length.

## Short Words

| Backend                   | `sz_utf8_linebreaks_best` |
| :------------------------ | ------------------------: |
| Serial @ 1× Intel Xeon6   |                 23.6 MB/s |
| Haswell @ 1× Intel Xeon6  |                 32.2 MB/s |
| Ice Lake @ 1× Intel Xeon6 |                 43.8 MB/s |
| Serial @ 1× Apple M5 Pro  |                 46.8 MB/s |
| NEON @ 1× Apple M5 Pro    |                 37.7 MB/s |
| NEON @ 1× AWS Graviton4   |                         … |
| SVE2 @ 1× AWS Graviton4   |                         … |
| SVE @ 1× AWS Graviton3    |                         … |

## Long Lines

| Backend                   | `sz_utf8_linebreaks_best` |
| :------------------------ | ------------------------: |
| Serial @ 1× Intel Xeon6   |                 19.4 MB/s |
| Haswell @ 1× Intel Xeon6  |                173.2 MB/s |
| Ice Lake @ 1× Intel Xeon6 |                230.0 MB/s |
| Serial @ 1× Apple M5 Pro  |                 37.5 MB/s |
| NEON @ 1× Apple M5 Pro    |                209.6 MB/s |
| NEON @ 1× AWS Graviton4   |                         … |
| SVE2 @ 1× AWS Graviton4   |                         … |
| SVE @ 1× AWS Graviton3    |                         … |
