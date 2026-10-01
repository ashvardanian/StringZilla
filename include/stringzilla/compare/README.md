# Compare: Equality and Lexicographic Ordering

This directory holds the comparison kernels behind `sz_equal_best` and `sz_order_best`.
Each operation has a serial baseline plus per-ISA SIMD backends, like `haswell`, `skylake`, `icelake` on x86.
`sz_equal_best` and `sz_order_best` run the best kernel the capability mask they are passed names.
For hot paths on short inputs, like a sort comparator, pick a capability's kernel, like `sz_order_neon`, or resolve one once through `sz_find_kernel_punned`, rather than paying the pick on every call.

## Methodology

Numbers are throughput in GB/s, measured with `bench/token.cpp` over the `leipzig1M.txt` corpus, reporting the median of repeated runs.
Each row is the library compiled with that single backend forced on one fixed chip, and each column is one operation, so coverage and cross-chip comparison read down a single column.
The Standard row is the platform's best stock equivalent per column, `std::memcmp` for both Equal and Order.
Comparison is decided in the first differing bytes, so a Short Words table (tokens averaging 5 bytes) and a Long Lines table (tokens averaging 130 bytes) are enough to show how token length shifts the balance.

## Short Words

| Backend          | `sz_equal_best` | `sz_order_best` |
| :--------------- | --------------: | --------------: |
| Standard @ Xeon4 |       0.37 GB/s |       0.33 GB/s |
| Serial @ Xeon4   |       0.53 GB/s |       0.49 GB/s |
| Haswell @ Xeon4  |       0.24 GB/s |       0.47 GB/s |
| Skylake @ Xeon4  |       0.44 GB/s |       0.32 GB/s |
| Ice Lake @ Xeon4 |               ↑ |               ↑ |
| NEON @ Graviton4 |               … |               … |
| SVE @ Graviton3  |               … |               … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_equal_best` | `sz_order_best` |
| :--------------- | --------------: | --------------: |
| Standard @ Xeon4 |       6.92 GB/s |       6.89 GB/s |
| Serial @ Xeon4   |      12.49 GB/s |      11.46 GB/s |
| Haswell @ Xeon4  |       5.38 GB/s |      12.94 GB/s |
| Skylake @ Xeon4  |       7.19 GB/s |       5.37 GB/s |
| Ice Lake @ Xeon4 |               ↑ |               ↑ |
| NEON @ Graviton4 |               … |               … |
| SVE @ Graviton3  |               … |               … |

> Measured June 26th, 2026.
