# Compare: Equality and Lexicographic Ordering

This directory holds the comparison kernels behind `sz_equal` and `sz_order`.
Each operation has a serial baseline plus per-ISA SIMD backends, like `haswell`, `skylake`, `icelake` on x86.
The dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput in GB/s, measured with `bench/token.cpp` over the `leipzig1M_en.txt` corpus, reporting the median of repeated runs.
Each row is the library compiled with that single backend forced on one fixed chip, and each column is one operation, so coverage and cross-chip comparison read down a single column.
The Standard row is the platform's best stock equivalent per column, `std::memcmp` for both Equal and Order.
Comparison is decided in the first differing bytes, so a Short Words table (tokens averaging 5 bytes) and a Long Lines table (tokens averaging 130 bytes) are enough to show how token length shifts the balance.

## Short Words

| Backend          | `sz_equal` | `sz_order` |
| :--------------- | ---------: | ---------: |
| Standard @ Xeon4 |  0.40 GB/s |  0.35 GB/s |
| Serial @ Xeon4   |  0.57 GB/s |  0.53 GB/s |
| Haswell @ Xeon4  |  0.26 GB/s |  0.50 GB/s |
| Skylake @ Xeon4  |  0.47 GB/s |  0.34 GB/s |
| Ice Lake @ Xeon4 |          ↑ |          ↑ |
| NEON @ Graviton4 |          … |          … |
| SVE @ Graviton3  |          … |          … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_equal` | `sz_order` |
| :--------------- | ---------: | ---------: |
| Standard @ Xeon4 |  7.43 GB/s |  7.40 GB/s |
| Serial @ Xeon4   | 13.41 GB/s | 12.30 GB/s |
| Haswell @ Xeon4  |  5.78 GB/s | 13.89 GB/s |
| Skylake @ Xeon4  |  7.72 GB/s |  5.77 GB/s |
| Ice Lake @ Xeon4 |          ↑ |          ↑ |
| NEON @ Graviton4 |          … |          … |
| SVE @ Graviton3  |          … |          … |

> Measured June 26th, 2026.
