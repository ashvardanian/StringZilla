# UTF-8 Tokens: Whitespace and Newline Splitting

This directory holds the kernels behind `sz_utf8_whitespaces_best` and `sz_utf8_newlines_best`, the fast token splitters that cut a UTF-8 string at runs of Unicode whitespace or at line boundaries, recognizing the full set of UTF-8 space and newline codepoints rather than only the ASCII ones.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86, and each `_best` dispatch point runs the best kernel among the capabilities its caller passes.

## Methodology

Numbers are throughput in MB/s, measured with `bench/utf8_scan.cpp` over the full multilingual `xlsum.csv` corpus, reporting the median of repeated runs.
Results are split into a Short Words workload (whitespace-delimited tokens averaging a few bytes) and a Long Lines workload (full text lines) to expose how each kernel scales with token length.
A `↑` cell means there is no dedicated kernel for that operation at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend                   | `sz_utf8_whitespaces_best` | `sz_utf8_newlines_best` |
| :------------------------ | -------------------------: | ----------------------: |
| Serial @ 1× Intel Xeon6   |                 324.7 MB/s |              325.9 MB/s |
| Haswell @ 1× Intel Xeon6  |                 316.3 MB/s |              315.3 MB/s |
| Ice Lake @ 1× Intel Xeon6 |                 588.9 MB/s |              863.3 MB/s |
| NEON @ 1× AWS Graviton4   |                          … |                       … |
| SVE2 @ 1× AWS Graviton4   |                          … |                       … |
| SVE @ 1× AWS Graviton3    |                          … |                       … |

## Long Lines

| Backend                   | `sz_utf8_whitespaces_best` | `sz_utf8_newlines_best` |
| :------------------------ | -------------------------: | ----------------------: |
| Serial @ 1× Intel Xeon6   |                 420.8 MB/s |              998.0 MB/s |
| Haswell @ 1× Intel Xeon6  |                 626.6 MB/s |            4,319.2 MB/s |
| Ice Lake @ 1× Intel Xeon6 |               3,703.8 MB/s |           11,561.0 MB/s |
| NEON @ 1× AWS Graviton4   |                          … |                       … |
| SVE2 @ 1× AWS Graviton4   |                          … |                       … |
| SVE @ 1× AWS Graviton3    |                          … |                       … |
