# UTF-8 Graphemes: UAX-29 Grapheme Cluster Iteration

This directory holds the kernels behind `sz_utf8_graphemes_best`, which walks a UTF-8 string and yields each grapheme cluster as defined by the Unicode UAX-29 rules, so a base letter plus its combining marks, or an emoji with its modifiers, counts as one user-perceived character rather than several codepoints.
Each call writes one byte length per cluster, and the clusters tile the input, so a caller that fills its buffer resumes from the sum of the lengths it got.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86 and `neon` and `sve2` on Arm, and `sz_utf8_graphemes_best` runs the best kernel among the capabilities its caller passes.

## Methodology

Tables compare whitespace-delimited words and full text lines from `xlsum.csv`, measured with `bench/utf8_segment.cpp` using an output capacity of 16 clusters per call.
The benchmark reads the first 64 MiB by default; `STRINGWARS_BYTES` controls the input size.
`STRINGWARS_TOKENS` selects words, lines, or the file as one input.
A `↑` cell means there is no dedicated kernel at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_graphemes_best` |
| :--------------- | -----------------------: |
| Serial @ Xeon4   |                76.2 MB/s |
| Haswell @ Xeon4  |                16.7 MB/s |
| Ice Lake @ Xeon4 |                48.4 MB/s |
| Serial @ M5 Pro  |              244.2 MiB/s |
| NEON @ M5 Pro    |              257.4 MiB/s |
| SVE2 @ Graviton4 |                        … |
| SVE @ Graviton3  |                        … |

## Long Lines

| Backend          | `sz_utf8_graphemes_best` |
| :--------------- | -----------------------: |
| Serial @ Xeon4   |                58.9 MB/s |
| Haswell @ Xeon4  |                28.4 MB/s |
| Ice Lake @ Xeon4 |               104.6 MB/s |
| Serial @ M5 Pro  |              244.9 MiB/s |
| NEON @ M5 Pro    |              270.3 MiB/s |
| SVE2 @ Graviton4 |                        … |
| SVE @ Graviton3  |                        … |
