# Hash: Byte Sum, AES Hash, and SHA-256

This directory holds the digest kernels behind `sz_bytesum`, `sz_hash`, the multi-seed `sz_hash` fan-out, and `sz_sha256`.
Each operation has a serial baseline plus per-ISA SIMD backends — `westmere`, `haswell`, `skylake`, `icelake` on x86, with a dedicated `goldmont` SHA-NI path for SHA-256.
The dispatcher picks the fastest one available on the running CPU.

## Methodology

Cells are dual: throughput in GB/s and hash rate in millions of hashes per second, measured with `bench/token.cpp` over the `leipzig1M_en.txt` corpus, reporting the median of repeated runs.
Each row is the library compiled with that single backend forced on one fixed chip, and each column is one operation, so coverage and cross-chip comparison read down a single column.
The Standard row is the platform's best stock equivalent per column — `std::accumulate` for Byte Sum and `std::hash` for the hashing columns.
Token length matters, so results are split into a Short Words table (tokens averaging 5 bytes) and a Long Lines table (tokens averaging 130 bytes).
Multi-seed hashing emits eight digests per input, so its hash rate is far higher at comparable throughput; SHA-256's accelerator is the SHA-NI path, shown as the Goldmont row.
A `↑` cell means there is no dedicated kernel at that ISA level, so the dispatcher reuses the kernel from the tier above it; an empty cell is genuinely-missing data.

## Short Words

| Backend          |               Byte Sum |                   Hash |         Multi-seed Hash |               SHA-256 |
| :--------------- | ---------------------: | ---------------------: | ----------------------: | --------------------: |
| Standard @ Xeon4 | 0.13 GB/s · 26 Mhash/s | 0.12 GB/s · 25 Mhash/s |  0.48 GB/s · 98 Mhash/s |                     - |
| Serial @ Xeon4   | 0.09 GB/s · 19 Mhash/s |  0.03 GB/s · 5 Mhash/s | 0.98 GB/s · 199 Mhash/s | 0.01 GB/s · 2 Mhash/s |
| Westmere @ Xeon4 |                      ↑ | 0.11 GB/s · 22 Mhash/s | 0.61 GB/s · 123 Mhash/s |                     ↑ |
| Goldmont @ Xeon4 |                      ↑ |                      ↑ |                       ↑ | 0.05 GB/s · 9 Mhash/s |
| Haswell @ Xeon4  | 0.09 GB/s · 19 Mhash/s |                      ↑ |                       ↑ |                     ↑ |
| Skylake @ Xeon4  | 0.18 GB/s · 37 Mhash/s | 0.14 GB/s · 29 Mhash/s |                       ↑ |                     ↑ |
| Ice Lake @ Xeon4 | 0.24 GB/s · 50 Mhash/s | 0.14 GB/s · 28 Mhash/s | 0.72 GB/s · 146 Mhash/s | 0.00 GB/s · 0 Mhash/s |
| NEON @ Graviton4 |                      … |                      … |                       … |                     … |
| SVE @ Graviton3  |                      … |                      … |                       … |                     … |

> Measured June 26th, 2026.

## Long Lines

| Backend          |               Byte Sum |                   Hash |        Multi-seed Hash |               SHA-256 |
| :--------------- | ---------------------: | ---------------------: | ---------------------: | --------------------: |
| Standard @ Xeon4 | 1.94 GB/s · 16 Mhash/s | 1.98 GB/s · 16 Mhash/s | 5.23 GB/s · 42 Mhash/s |                     - |
| Serial @ Xeon4   | 1.32 GB/s · 11 Mhash/s |  0.12 GB/s · 1 Mhash/s | 8.59 GB/s · 69 Mhash/s | 0.12 GB/s · 1 Mhash/s |
| Westmere @ Xeon4 |                      ↑ | 1.65 GB/s · 13 Mhash/s | 3.67 GB/s · 30 Mhash/s |                     ↑ |
| Goldmont @ Xeon4 |                      ↑ |                      ↑ |                      ↑ | 0.55 GB/s · 4 Mhash/s |
| Haswell @ Xeon4  | 1.37 GB/s · 11 Mhash/s |                      ↑ |                      ↑ |                     ↑ |
| Skylake @ Xeon4  | 2.08 GB/s · 17 Mhash/s | 1.45 GB/s · 12 Mhash/s |                      ↑ |                     ↑ |
| Ice Lake @ Xeon4 | 3.42 GB/s · 28 Mhash/s | 1.96 GB/s · 16 Mhash/s | 5.59 GB/s · 45 Mhash/s | 0.01 GB/s · 0 Mhash/s |
| NEON @ Graviton4 |                      … |                      … |                      … |                     … |
| SVE @ Graviton3  |                      … |                      … |                      … |                     … |

> Measured June 26th, 2026.
