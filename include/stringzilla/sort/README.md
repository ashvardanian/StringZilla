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

| Backend          | `sz_sequence_argsort_best` | `sz_sequence_argsort_uncased_best` |
| :--------------- | -------------------------: | ---------------------------------: |
| Standard @ Xeon4 |                  22 Mcmp/s |                          27 Mcmp/s |
| Serial @ Xeon4   |                  96 Mcmp/s |                          28 Mcmp/s |
| Haswell @ Xeon4  |                 140 Mcmp/s |                          57 Mcmp/s |
| Skylake @ Xeon4  |                 114 Mcmp/s |                          64 Mcmp/s |
| NEON @ Graviton4 |                          … |                                  … |
| SVE @ Graviton3  |                          … |                                  … |

## Long Lines

| Backend          | `sz_sequence_argsort_best` | `sz_sequence_argsort_uncased_best` |
| :--------------- | -------------------------: | ---------------------------------: |
| Standard @ Xeon4 |                  64 Mcmp/s |                          70 Mcmp/s |
| Serial @ Xeon4   |                 148 Mcmp/s |                          30 Mcmp/s |
| Haswell @ Xeon4  |                 169 Mcmp/s |                          30 Mcmp/s |
| Skylake @ Xeon4  |                 165 Mcmp/s |                          34 Mcmp/s |
| NEON @ Graviton4 |                          … |                                  … |
| SVE @ Graviton3  |                          … |                                  … |
