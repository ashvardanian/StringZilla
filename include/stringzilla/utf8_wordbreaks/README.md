# UTF-8 Words: UAX-29 Word Boundary Iteration

This directory holds the kernels behind `sz_utf8_wordbreaks`, which walks a UTF-8 string and yields each word as defined by the Unicode UAX-29 word-boundary rules, so "don't" or a CJK run is split the way a human reader expects rather than on raw spaces.
Each operation has a serial baseline plus `haswell` and `icelake` SIMD backends on x86, and the dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput in MB/s, measured with `bench/utf8_iterate.cpp` over the full multilingual `xlsum.csv` corpus, reporting the median of repeated runs.
Each table fixes one input shape; its single column is the `sz_utf8_wordbreaks` operation and its rows are a backend on a chip, so reading down the column compares the backend ladder on one fixed input shape.
Results are split into a Short Words workload (whitespace-delimited tokens averaging a few bytes) and a Long Lines workload (full text lines) to expose how each kernel scales with token length.
A `↑` cell means there is no dedicated kernel at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_wordbreaks` |
| :--------------- | --------------: |
| Serial @ Xeon4   |       72.6 MB/s |
| Haswell @ Xeon4  |       14.2 MB/s |
| Ice Lake @ Xeon4 |       36.2 MB/s |
| NEON @ Graviton4 |               … |
| SVE2 @ Graviton4 |               … |
| SVE @ Graviton3  |               … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_utf8_wordbreaks` |
| :--------------- | --------------: |
| Serial @ Xeon4   |       45.9 MB/s |
| Haswell @ Xeon4  |       43.1 MB/s |
| Ice Lake @ Xeon4 |      115.8 MB/s |
| NEON @ Graviton4 |               … |
| SVE2 @ Graviton4 |               … |
| SVE @ Graviton3  |               … |

> Measured June 26th, 2026.
