# Cipher: AES-256 in Counter and Galois/Counter Modes

This directory holds the encryption kernels behind `sz_aes256_ctr_xor`, `sz_aes256_gcm_encrypt`, `sz_aes256_gcm_decrypt`, and the streaming `sz_aes256_gcm_encryptor` and `sz_aes256_gcm_decryptor` families.
Each operation has a serial baseline plus per-ISA SIMD backends — `westmere` and `icelake` on x86, `neonaes` and `sve2aes` on Arm, `rvvcrypto` on RISC-V, `powervsx` on Power, and `v128` with `v128relaxed` on WebAssembly.
The dispatcher picks the fastest one available on the running CPU.

## Methodology

Cells are throughput in GB/s, measured with `bench/cipher.cpp` on one core pinned away from the scheduler, reporting the median of nine calibrated samples and reproducible to about one and a half percent.
Each row is the library compiled with that single backend forced on one fixed chip, and each column is one message size, so coverage and cross-chip comparison read down a single column.
The Serial row is the reference; there is no Standard row here, since no standard library ships a block cipher.
Message size decides how much of a call is key schedule and tag arithmetic rather than bulk work, so results sweep four sizes from a short record to a page-sized buffer.
A `…` cell is genuinely-missing data, on a backend not yet measured on hardware that runs it.

## Counter Mode

| Backend          |     256 B |      1 KB |      4 KB |     16 KB |
| :--------------- | --------: | --------: | --------: | --------: |
| Serial @ Xeon4   | 0.04 GB/s | 0.04 GB/s | 0.04 GB/s | 0.04 GB/s |
| Westmere @ Xeon4 | 4.49 GB/s | 5.88 GB/s | 6.18 GB/s | 6.51 GB/s |
| Ice Lake @ Xeon4 | 8.18 GB/s | 12.2 GB/s | 14.6 GB/s | 14.8 GB/s |
| NEON @ Graviton4 |         … |         … |         … |         … |
| SVE2 @ Graviton4 |         … |         … |         … |         … |

> Measured August 4th, 2026.

## Galois/Counter Mode

| Backend          |      256 B |       1 KB |      4 KB |     16 KB |
| :--------------- | ---------: | ---------: | --------: | --------: |
| Serial @ Xeon4   | 0.004 GB/s | 0.004 GB/s | 0.004 GB/s | 0.004 GB/s |
| Westmere @ Xeon4 |  2.43 GB/s |  3.20 GB/s | 3.24 GB/s | 3.22 GB/s |
| Ice Lake @ Xeon4 |  3.42 GB/s |  6.37 GB/s | 7.59 GB/s | 8.31 GB/s |
| NEON @ Graviton4 |          … |          … |         … |         … |
| SVE2 @ Graviton4 |          … |          … |         … |         … |

> Measured August 4th, 2026.
