# UTF-8 Line Wraps: UAX-14 Line Break Opportunity Iteration

This directory holds the kernels behind `sz_utf8_linewraps`, which walks a UTF-8 string and yields each line-break opportunity as defined by the Unicode UAX-14 rules, marking the positions where a text layout engine is allowed to wrap rather than splitting on raw spaces or hard newlines alone.
Each operation has a serial baseline plus per-ISA SIMD backends, like `haswell`, `icelake` on x86.
The dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput in MB/s, measured with `bench/utf8_iterate.cpp` over the full multilingual `xlsum.csv` corpus, reporting the median of repeated runs.
Each table fixes one input shape; its single column is the `sz_utf8_linewraps` operation and its rows are a backend on a chip, so reading down the column compares the backend ladder on one fixed input shape.
Results are split into a Short Words workload (whitespace-delimited tokens averaging a few bytes) and a Long Lines workload (full text lines) to expose how each kernel scales with token length.

## Short Words

| Backend          | `sz_utf8_linewraps` |
| :--------------- | ------------------: |
| Serial @ Xeon4   |           20.9 MB/s |
| Haswell @ Xeon4  |            9.8 MB/s |
| Ice Lake @ Xeon4 |           39.8 MB/s |
| NEON @ Graviton4 |                   … |
| SVE2 @ Graviton4 |                   … |
| SVE @ Graviton3  |                   … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_utf8_linewraps` |
| :--------------- | ------------------: |
| Serial @ Xeon4   |            2.5 MB/s |
| Haswell @ Xeon4  |           25.3 MB/s |
| Ice Lake @ Xeon4 |           92.8 MB/s |
| NEON @ Graviton4 |                   … |
| SVE2 @ Graviton4 |                   … |
| SVE @ Graviton3  |                   … |

> Measured June 26th, 2026.
