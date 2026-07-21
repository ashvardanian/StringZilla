# Find: Substring, Byte, and Byte Set Search

This directory holds the search kernels behind `sz_find`, `sz_rfind`, `sz_find_byte`, `sz_rfind_byte`, `sz_find_byteset`, and `sz_rfind_byteset`.
Each operation has a serial SWAR baseline plus per-ISA SIMD backends — `westmere`, `haswell`, `skylake`, `icelake` on x86, `neon`, `sve`, `sve2` on Arm, and `rvv`, `lasx`, `powervsx`, `v128` elsewhere.
The dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput, shown in GB/s in each cell, measured with `bench/find.cpp` over the `leipzig1M_en.txt` corpus, reporting the median of repeated runs.
Each row is the library compiled with that single backend forced on one fixed chip, and each column is one operation, so coverage and cross-chip comparison read down a single column.
The Standard row is the platform's best stock equivalent per column — `strstr` and `std::find_end` for substring search, `memchr` for the byte variants, and `strpbrk`/`strcspn` for the byte-set variants.
Substring search depends sharply on needle length, so results are split into a Short Words table with tokens averaging 5 bytes and a Long Lines table with tokens averaging 130 bytes.
A `↑` cell means there is no dedicated kernel at that ISA level, so the dispatcher reuses the kernel from the tier above it; an empty cell is genuinely-missing data.

## Short Words

| Backend          | `sz_find` | `sz_rfind` | `sz_find_byte` | `sz_rfind_byte` | `sz_find_byteset` | `sz_rfind_byteset` |
| :--------------- | --------: | ---------: | -------------: | --------------: | ----------------: | -----------------: |
| Standard @ Xeon4 | 2.75 GB/s |  0.26 GB/s |      0.24 GB/s |       0.24 GB/s |         0.08 GB/s |          0.10 GB/s |
| Serial @ Xeon4   | 2.67 GB/s |  0.65 GB/s |      0.35 GB/s |       0.39 GB/s |         0.28 GB/s |          0.29 GB/s |
| Westmere @ Xeon4 | 5.33 GB/s |  4.61 GB/s |      0.24 GB/s |       0.29 GB/s |                 ↑ |                  ↑ |
| Haswell @ Xeon4  | 5.94 GB/s |  5.24 GB/s |      0.34 GB/s |       0.30 GB/s |         0.27 GB/s |          0.28 GB/s |
| Skylake @ Xeon4  | 9.62 GB/s |  9.23 GB/s |      0.61 GB/s |       0.66 GB/s |                 ↑ |                  ↑ |
| Ice Lake @ Xeon4 |         ↑ |          ↑ |              ↑ |               ↑ |         0.28 GB/s |          0.31 GB/s |
| NEON @ Graviton4 |         … |          … |              … |               … |                 … |                  … |
| SVE @ Graviton3  |         … |          … |              … |               … |                 … |                  … |

> Measured June 26th, 2026.

## Long Lines

| Backend          |  `sz_find` | `sz_rfind` | `sz_find_byte` | `sz_rfind_byte` | `sz_find_byteset` | `sz_rfind_byteset` |
| :--------------- | ---------: | ---------: | -------------: | --------------: | ----------------: | -----------------: |
| Standard @ Xeon4 | 17.74 GB/s |  6.23 GB/s |      2.07 GB/s |       2.07 GB/s |         0.25 GB/s |          0.23 GB/s |
| Serial @ Xeon4   |  5.68 GB/s |  6.00 GB/s |      1.55 GB/s |       1.52 GB/s |         1.19 GB/s |          1.17 GB/s |
| Westmere @ Xeon4 | 12.85 GB/s | 10.62 GB/s |      2.09 GB/s |       2.03 GB/s |                 ↑ |                  ↑ |
| Haswell @ Xeon4  | 12.36 GB/s | 13.09 GB/s |      1.82 GB/s |       1.78 GB/s |         2.61 GB/s |          2.71 GB/s |
| Skylake @ Xeon4  | 18.78 GB/s | 18.91 GB/s |      1.86 GB/s |       1.90 GB/s |                 ↑ |                  ↑ |
| Ice Lake @ Xeon4 |          ↑ |          ↑ |              ↑ |               ↑ |         3.82 GB/s |          3.68 GB/s |
| NEON @ Graviton4 |          … |          … |              … |               … |                 … |                  … |
| SVE @ Graviton3  |          … |          … |              … |               … |                 … |                  … |

> Measured June 26th, 2026.
