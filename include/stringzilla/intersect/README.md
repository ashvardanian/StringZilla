# Intersect: Sequence Set Intersection

This directory holds the set-intersection kernel behind `sz_sequence_intersect_best`.
The operation has a serial baseline, its `westmere` and `neonaes` variants hashing with AES-NI and Arm AES, plus a dedicated AVX-512 VBMI `icelake` backend on x86.
The `_best` dispatch point runs the best kernel among the capabilities its caller passes.

## Methodology

Numbers are throughput in comparisons/s, rendered as Mcmp/s, measured with `bench/sequence.cpp` over the `leipzig1M.txt` corpus, reporting the median of repeated runs.
The Standard row is the platform's best stock equivalent, `std::unordered_map`.
Token length affects per-element cost, so results are split into a Short Words table (tokens averaging 5 bytes) and a Long Lines table (tokens averaging 130 bytes).
An empty cell or `…` is not measured yet.

## Short Words

| Backend                     | `sz_sequence_intersect_best` |
| :-------------------------- | ---------------------------: |
| Standard @ 1× Intel Xeon6   |                     9 Mcmp/s |
| Serial @ 1× Intel Xeon6     |                     7 Mcmp/s |
| Ice Lake @ 1× Intel Xeon6   |                    18 Mcmp/s |
| NEON AES @ 1× AWS Graviton4 |                            … |

## Long Lines

| Backend                     | `sz_sequence_intersect_best` |
| :-------------------------- | ---------------------------: |
| Standard @ 1× Intel Xeon6   |                     6 Mcmp/s |
| Serial @ 1× Intel Xeon6     |                     2 Mcmp/s |
| Ice Lake @ 1× Intel Xeon6   |                    13 Mcmp/s |
| NEON AES @ 1× AWS Graviton4 |                            … |
