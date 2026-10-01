# Memory: Copy, Move, Fill, and Lookup

This directory holds the memory kernels behind `sz_copy_best`, `sz_move_best`, `sz_fill_best`, and `sz_lookup_best`.
Each operation has a serial baseline plus per-ISA SIMD backends — `haswell`, `skylake`, and `icelake` on x86.
`sz_copy_best` and its siblings run the best kernel the capability mask they are passed names.
For hot paths on short inputs, pick a capability's kernel, like `sz_copy_neon`, or resolve one once through `sz_find_kernel_punned`, rather than paying the pick on every call.

## Methodology

Numbers are throughput in GB/s, measured with `bench/memory.cpp` over the `leipzig1M.txt` corpus, reporting the median of repeated runs.
Memory operations are bandwidth-bound and measured solo, so they are not tokenized and appear in a single table.
Each row is the library compiled with that single backend forced on one fixed chip, and each column is one operation, so coverage and cross-chip comparison read down a single column.
The Standard row is the platform's best stock equivalent per column — `std::memcpy`, `std::memmove`, `std::memset`, and `std::transform`.
A `↑` cell means there is no dedicated kernel at that ISA level, so `_best` reuses the kernel from the tier above it; an empty cell is genuinely-missing data.

## Memory

| Backend          | `sz_copy_best` | `sz_move_best` | `sz_fill_best` | `sz_lookup_best` |
| :--------------- | -------------: | -------------: | -------------: | ---------------: |
| Standard @ Xeon4 |      14.8 GB/s |      24.0 GB/s |      25.9 GB/s |         3.0 GB/s |
| Serial @ Xeon4   |       7.3 GB/s |      17.8 GB/s |      13.2 GB/s |         3.0 GB/s |
| Haswell @ Xeon4  |       4.8 GB/s |      23.2 GB/s |      14.7 GB/s |         6.8 GB/s |
| Skylake @ Xeon4  |      15.0 GB/s |      23.5 GB/s |      18.3 GB/s |                ↑ |
| Ice Lake @ Xeon4 |              ↑ |              ↑ |              ↑ |         7.7 GB/s |
| NEON @ Graviton4 |              … |              … |              … |                … |
| SVE @ Graviton3  |              … |              … |              … |                … |

> Measured June 26th, 2026.
