# Intersect: Sequence Set Intersection

This directory holds the set-intersection kernel behind `sz_sequence_intersect`.
The operation has a serial baseline plus a dedicated AVX-512 VBMI `icelake` backend on x86.
The dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput in comparisons/s, rendered as Mcmp/s, measured with `bench/sequence.cpp` over the `leipzig1M_en.txt` corpus, reporting the median of repeated runs.
Each row is the library compiled with that single backend forced on one fixed chip, and the single column is the operation, so coverage and cross-chip comparison read down the column.
The Standard row is the platform's best stock equivalent, `std::unordered_map`.
Token length affects per-element cost, so results are split into a Short Words table (tokens averaging 5 bytes) and a Long Lines table (tokens averaging 130 bytes).
An empty cell is genuinely-missing data; `…` rows are placeholders awaiting an Arm run.

## Short Words

| Backend          | `sz_sequence_intersect` |
| :--------------- | ----------------------: |
| Standard @ Xeon4 |                5 Mcmp/s |
| Serial @ Xeon4   |               31 Mcmp/s |
| Ice Lake @ Xeon4 |               24 Mcmp/s |
| NEON @ Graviton4 |                       … |
| SVE @ Graviton3  |                       … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_sequence_intersect` |
| :--------------- | ----------------------: |
| Standard @ Xeon4 |                3 Mcmp/s |
| Serial @ Xeon4   |               10 Mcmp/s |
| Ice Lake @ Xeon4 |               10 Mcmp/s |
| NEON @ Graviton4 |                       … |
| SVE @ Graviton3  |                       … |

> Measured June 26th, 2026.
