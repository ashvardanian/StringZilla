# UTF-8 Sentences: UAX-29 Sentence Boundary Iteration

This directory holds the kernels behind `sz_utf8_sentences_best`, which walks a UTF-8 string and yields each sentence as defined by the Unicode UAX-29 sentence-boundary rules, distinguishing a real sentence terminator from an abbreviation dot or a decimal point rather than breaking on every period.
Each call writes one byte length per sentence, and the sentences tile the input, so a caller that fills its buffer resumes from the sum of the lengths it got.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86 plus `neon` and `sve2` on Arm, and `sz_utf8_sentences_best` runs the best kernel among the capabilities its caller passes.

## Methodology

`bench/utf8_segment.cpp` measures throughput over the first 64 MB of the multilingual `xlsum.csv` corpus by default.
`STRINGWARS_BYTES` controls the input size; `STRINGWARS_TOKENS` selects words, lines, or the file as one input.
Results are split into a Short Words workload (whitespace-delimited tokens averaging a few bytes) and a Long Lines workload (full text lines) to expose how each kernel scales with token length.
A `↑` cell means there is no dedicated kernel at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend                   | `sz_utf8_sentences_best` |
| :------------------------ | -----------------------: |
| Serial @ 1× Intel Xeon6   |                88.3 MB/s |
| Haswell @ 1× Intel Xeon6  |                53.3 MB/s |
| Ice Lake @ 1× Intel Xeon6 |                77.3 MB/s |
| Serial @ 1× Apple M5 Pro  |               160.1 MB/s |
| NEON @ 1× Apple M5 Pro    |                85.1 MB/s |
| NEON @ 1× AWS Graviton4   |                        … |
| SVE2 @ 1× AWS Graviton4   |                        … |
| SVE @ 1× AWS Graviton3    |                        … |

## Long Lines

| Backend                   | `sz_utf8_sentences_best` |
| :------------------------ | -----------------------: |
| Serial @ 1× Intel Xeon6   |                90.3 MB/s |
| Haswell @ 1× Intel Xeon6  |               407.1 MB/s |
| Ice Lake @ 1× Intel Xeon6 |               541.2 MB/s |
| Serial @ 1× Apple M5 Pro  |               159.8 MB/s |
| NEON @ 1× Apple M5 Pro    |               475.6 MB/s |
| NEON @ 1× AWS Graviton4   |                        … |
| SVE2 @ 1× AWS Graviton4   |                        … |
| SVE @ 1× AWS Graviton3    |                        … |
