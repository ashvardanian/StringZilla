# Sort: Argsort, Uncased Argsort, and Pgram Sort

This directory holds the sorting kernels behind `sz_sequence_argsort_best` and `sz_sequence_argsort_uncased_best`, and the integer pgram sort they build on, which is internal and has no dispatch point.
Each operation has a serial baseline plus `haswell` and `skylake` SIMD backends on x86, `neon` and `sve` on Arm, and `rvv` on RISC-V.
Each `_best` dispatch point runs the best kernel among the capabilities its caller passes.

## Methodology

Numbers are sorting throughput in comparisons/s, rendered as Mcmp/s, measured with `bench/sequence.cpp` over the `leipzig1M.txt` corpus, reporting the median of repeated runs.
Sorting throughput is reported as comparisons/s, with the operation count modeled as N·log₂ N, matching StringWars.
The Standard row is the platform's best stock equivalent per column — `std::sort` for Argsort and Pgram Sort, `std::stable_sort` for Uncased Argsort.
Token length matters for comparison cost, so results are split into a Short Words table (tokens averaging 5 bytes) and a Long Lines table (tokens averaging 130 bytes).
A `↑` cell means there is no dedicated kernel at that ISA level, so the dispatcher reuses the kernel from the tier above it; an empty cell is genuinely-missing data.

## Short Words

| Backend                   | `sz_sequence_argsort_best` | `sz_sequence_argsort_uncased_best` |
| :------------------------ | -------------------------: | ---------------------------------: |
| Standard @ 1× Intel Xeon6 |                  63 Mcmp/s |                          29 Mcmp/s |
| Serial @ 1× Intel Xeon6   |                 172 Mcmp/s |                         104 Mcmp/s |
| Haswell @ 1× Intel Xeon6  |                 230 Mcmp/s |                         125 Mcmp/s |
| Skylake @ 1× Intel Xeon6  |                 238 Mcmp/s |                         130 Mcmp/s |
| NEON @ 1× AWS Graviton4   |                          … |                                  … |
| SVE @ 1× AWS Graviton3    |                          … |                                  … |

## Long Lines

| Backend                   | `sz_sequence_argsort_best` | `sz_sequence_argsort_uncased_best` |
| :------------------------ | -------------------------: | ---------------------------------: |
| Standard @ 1× Intel Xeon6 |                  53 Mcmp/s |                          16 Mcmp/s |
| Serial @ 1× Intel Xeon6   |                 136 Mcmp/s |                          55 Mcmp/s |
| Haswell @ 1× Intel Xeon6  |                 181 Mcmp/s |                          60 Mcmp/s |
| Skylake @ 1× Intel Xeon6  |                 200 Mcmp/s |                          62 Mcmp/s |
| NEON @ 1× AWS Graviton4   |                          … |                                  … |
| SVE @ 1× AWS Graviton3    |                          … |                                  … |
