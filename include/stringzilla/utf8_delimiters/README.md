# Delimiters: First UTF-8 Delimiter Scan

This directory holds the kernels behind `sz_find_delimiters_utf8`, which scans a UTF-8 string and returns the position of the first delimiter character — the boundary that ends a token, such as whitespace or punctuation.
The classifier works directly on UTF-8 bytes, decoding each character only as far as needed to decide whether it delimits.
Every operation has a serial baseline plus per-ISA SIMD backends, and the dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput in MB/s, measured with `bench/utf8_delimiters.cpp` over the full multilingual `xlsum.csv` corpus, reporting the median of repeated runs.
Each table fixes one input shape; its single column is the `sz_find_delimiters_utf8` operation and its rows are a backend on a chip, so reading down the column compares the backend ladder on one fixed input shape.
The Serial row is the reference; there is no Standard row here, since no standard library ships a Unicode-aware delimiter scan.
Results are split into a Short Words workload (tokens averaging a handful of bytes) and a Long Lines workload (full sentences).
A `↑` cell means there is no dedicated `sz_find_delimiters_utf8_<isa>` kernel at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_find_delimiters_utf8` |
| :--------------- | ------------------------: |
| Serial @ Xeon4   |                239.5 MB/s |
| Haswell @ Xeon4  |                 31.3 MB/s |
| Ice Lake @ Xeon4 |                120.0 MB/s |
| NEON @ Graviton4 |                         … |
| SVE2 @ Graviton4 |                         … |
| SVE @ Graviton3  |                         … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_find_delimiters_utf8` |
| :--------------- | ------------------------: |
| Serial @ Xeon4   |                276.3 MB/s |
| Haswell @ Xeon4  |                 18.1 MB/s |
| Ice Lake @ Xeon4 |                115.6 MB/s |
| NEON @ Graviton4 |                         … |
| SVE2 @ Graviton4 |                         … |
| SVE @ Graviton3  |                         … |

> Measured June 26th, 2026.
