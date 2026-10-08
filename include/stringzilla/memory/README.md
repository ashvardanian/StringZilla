# Memory: Copy, Move, Fill, and Lookup

This directory holds the memory kernels behind `sz_copy_best`, `sz_move_best`, `sz_fill_best`, and `sz_lookup_best`.
Each operation has a serial baseline plus per-ISA SIMD backends — `haswell`, `skylake`, and `icelake` on x86.
`sz_copy_best` and its siblings run the best kernel the capability mask they are passed names.
For hot paths on short inputs, pick a capability's kernel, like `sz_copy_neon`, or resolve one once through `sz_find_kernel_punned`, rather than paying the pick on every call.

## Methodology

Numbers are throughput in GB/s, measured with `bench/memory.cpp` over the `leipzig1M.txt` corpus, reporting the median of repeated runs.
Memory operations are bandwidth-bound and measured solo, so they are not tokenized and appear in a single table.
The Standard row is the platform's best stock equivalent per column — `std::memcpy`, `std::memmove`, `std::memset`, and `std::transform`.
A `↑` cell means there is no dedicated kernel at that ISA level, so `_best` reuses the kernel from the tier above it; an empty cell is genuinely-missing data.

## Memory

| Backend                   | `sz_copy_best` | `sz_move_best` | `sz_fill_best` | `sz_lookup_best` |
| :------------------------ | -------------: | -------------: | -------------: | ---------------: |
| Standard @ 1× Intel Xeon6 |      11.2 GB/s |      16.6 GB/s |      11.8 GB/s |         2.6 GB/s |
| Serial @ 1× Intel Xeon6   |       4.1 GB/s |       5.6 GB/s |       4.0 GB/s |         2.5 GB/s |
| Haswell @ 1× Intel Xeon6  |       3.7 GB/s |       5.6 GB/s |       3.1 GB/s |         2.3 GB/s |
| Skylake @ 1× Intel Xeon6  |       9.2 GB/s |      13.3 GB/s |      10.2 GB/s |                ↑ |
| Ice Lake @ 1× Intel Xeon6 |              ↑ |              ↑ |              ↑ |         4.1 GB/s |
| NEON @ 1× AWS Graviton4   |              … |              … |              … |                … |
| SVE @ 1× AWS Graviton3    |              … |              … |              … |                … |
