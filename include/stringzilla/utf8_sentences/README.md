# UTF-8 Sentences: UAX-29 Sentence Boundary Iteration

This directory holds the kernels behind `sz_utf8_sentences`, which walks a UTF-8 string and yields each sentence as defined by the Unicode UAX-29 sentence-boundary rules, distinguishing a real sentence terminator from an abbreviation dot or a decimal point rather than breaking on every period.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86, and the dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput in MB/s, measured with `bench/utf8_iterate.cpp` over the full multilingual `xlsum.csv` corpus, reporting the median of repeated runs.
Each table fixes one input shape; its single column is the `sz_utf8_sentences` operation and its rows are a backend on a chip, so reading down the column compares the backend ladder on one fixed input shape.
Results are split into a Short Words workload (whitespace-delimited tokens averaging a few bytes) and a Long Lines workload (full text lines) to expose how each kernel scales with token length.
A `↑` cell means there is no dedicated kernel at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_sentences` |
| :--------------- | ------------------: |
| Serial @ Xeon4   |           46.0 MB/s |
| Haswell @ Xeon4  |            6.6 MB/s |
| Ice Lake @ Xeon4 |           59.7 MB/s |
| NEON @ Graviton4 |                   … |
| SVE2 @ Graviton4 |                   … |
| SVE @ Graviton3  |                   … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_utf8_sentences` |
| :--------------- | ------------------: |
| Serial @ Xeon4   |           54.1 MB/s |
| Haswell @ Xeon4  |           45.2 MB/s |
| Ice Lake @ Xeon4 |          270.9 MB/s |
| NEON @ Graviton4 |                   … |
| SVE2 @ Graviton4 |                   … |
| SVE @ Graviton3  |                   … |

> Measured June 26th, 2026.
