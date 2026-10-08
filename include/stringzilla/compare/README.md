# Compare: Equality and Lexicographic Ordering

This directory holds the comparison kernels behind `sz_equal_best` and `sz_order_best`.
Each operation has a serial baseline plus per-ISA SIMD backends, like `haswell`, `skylake`, `icelake` on x86.
`sz_equal_best` and `sz_order_best` run the best kernel the capability mask they are passed names.
For hot paths on short inputs, like a sort comparator, pick a capability's kernel, like `sz_order_neon`, or resolve one once through `sz_find_kernel_punned`, rather than paying the pick on every call.

## Methodology

Numbers are throughput in GB/s, measured with `bench/token.cpp` over the `leipzig1M.txt` corpus, reporting the median of repeated runs.
The Standard row is the platform's best stock equivalent per column, `std::memcmp` for both Equal and Order.
Comparison is decided in the first differing bytes, so a Short Words table (tokens averaging 5 bytes) and a Long Lines table (tokens averaging 130 bytes) are enough to show how token length shifts the balance.

## Short Words

| Backend                   | `sz_equal_best` | `sz_order_best` |
| :------------------------ | --------------: | --------------: |
| Standard @ 1× Intel Xeon6 |       1.32 GB/s |       1.32 GB/s |
| Serial @ 1× Intel Xeon6   |       0.34 GB/s |       0.37 GB/s |
| Haswell @ 1× Intel Xeon6  |       0.44 GB/s |       0.38 GB/s |
| Skylake @ 1× Intel Xeon6  |       1.19 GB/s |       0.89 GB/s |
| Ice Lake @ 1× Intel Xeon6 |               ↑ |               ↑ |
| NEON @ 1× AWS Graviton4   |               … |               … |
| SVE @ 1× AWS Graviton3    |               … |               … |

## Long Lines

| Backend                   | `sz_equal_best` | `sz_order_best` |
| :------------------------ | --------------: | --------------: |
| Standard @ 1× Intel Xeon6 |      12.69 GB/s |      13.38 GB/s |
| Serial @ 1× Intel Xeon6   |       5.10 GB/s |       5.37 GB/s |
| Haswell @ 1× Intel Xeon6  |       9.47 GB/s |       5.37 GB/s |
| Skylake @ 1× Intel Xeon6  |      12.35 GB/s |      10.33 GB/s |
| Ice Lake @ 1× Intel Xeon6 |               ↑ |               ↑ |
| NEON @ 1× AWS Graviton4   |               … |               … |
| SVE @ 1× AWS Graviton3    |               … |               … |
