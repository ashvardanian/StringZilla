# Memory: Copy, Move, Fill, and Lookup

This directory holds the memory kernels behind `sz_copy`, `sz_move`, `sz_fill`, and `sz_lookup`.
Each operation has a serial baseline plus per-ISA SIMD backends — `haswell`, `skylake`, and `icelake` on x86.
The dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput in GB/s, measured with `bench/memory.cpp` over the `leipzig1M_en.txt` corpus, reporting the median of repeated runs.
Memory operations are bandwidth-bound and measured solo, so they are not tokenized and appear in a single table.
Each row is the library compiled with that single backend forced on one fixed chip, and each column is one operation, so coverage and cross-chip comparison read down a single column.
The Standard row is the platform's best stock equivalent per column — `std::memcpy`, `std::memmove`, `std::memset`, and `std::transform`.
A `↑` cell means there is no dedicated kernel at that ISA level, so the dispatcher reuses the kernel from the tier above it; an empty cell is genuinely-missing data.

## Memory

| Backend          | `sz_copy` | `sz_move` | `sz_fill` | `sz_lookup` |
| :--------------- | --------: | --------: | --------: | ----------: |
| Standard @ Xeon4 | 15.9 GB/s | 25.8 GB/s | 27.8 GB/s |    3.2 GB/s |
| Serial @ Xeon4   |  7.8 GB/s | 19.1 GB/s | 14.2 GB/s |    3.2 GB/s |
| Haswell @ Xeon4  |  5.2 GB/s | 24.9 GB/s | 15.8 GB/s |    7.3 GB/s |
| Skylake @ Xeon4  | 16.1 GB/s | 25.2 GB/s | 19.7 GB/s |           ↑ |
| Ice Lake @ Xeon4 |         ↑ |         ↑ |         ↑ |    8.3 GB/s |
| NEON @ Graviton4 |         … |         … |         … |           … |
| SVE @ Graviton3  |         … |         … |         … |           … |

> Measured June 26th, 2026.
