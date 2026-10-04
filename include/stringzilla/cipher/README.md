# Cipher: AES-256 in Counter and Galois/Counter Modes

This directory holds the encryption kernels behind `sz_aes256_ctr_xor_best`, `sz_aes256_gcm_encrypt_best`, `sz_aes256_gcm_decrypt_best`, and the streaming `sz_aes256_gcm_encryptor_*_best` and `sz_aes256_gcm_decryptor_*_best` families.
Each operation has a serial baseline plus per-ISA SIMD backends — `westmere` and `icelake` on x86, `neonaes` and `sve2aes` on Arm, `rvvcrypto` on RISC-V, `powervsx` on Power, and `v128` with `v128relaxed` on WebAssembly.
Each `_best` dispatch point runs the best kernel among the capabilities its caller passes.

## Methodology

Cells are throughput in GB/s, measured with `bench/cipher.cpp`, reporting the median of nine calibrated samples.
The Serial row is the reference; there is no Standard row here, since no standard library ships a block cipher.
Message size decides how much of a call is key schedule and tag arithmetic rather than bulk work, so results sweep four sizes from a short record to a page-sized buffer.
A `…` cell is not measured yet.

## Counter Mode

| Backend          |     256 B |      1 KB |      4 KB |     16 KB |
| :--------------- | --------: | --------: | --------: | --------: |
| Serial @ Xeon4   | 0.04 GB/s | 0.04 GB/s | 0.04 GB/s | 0.04 GB/s |
| Westmere @ Xeon4 | 4.18 GB/s | 5.48 GB/s | 5.76 GB/s | 6.06 GB/s |
| Ice Lake @ Xeon4 | 7.62 GB/s | 11.4 GB/s | 13.6 GB/s | 13.8 GB/s |
| NEON @ Graviton4 |         … |         … |         … |         … |
| SVE2 @ Graviton4 |         … |         … |         … |         … |

## Galois/Counter Mode

| Backend          |      256 B |       1 KB |       4 KB |      16 KB |
| :--------------- | ---------: | ---------: | ---------: | ---------: |
| Serial @ Xeon4   | 0.004 GB/s | 0.004 GB/s | 0.004 GB/s | 0.004 GB/s |
| Westmere @ Xeon4 |  2.26 GB/s |  2.98 GB/s |  3.02 GB/s |  3.00 GB/s |
| Ice Lake @ Xeon4 |  3.19 GB/s |  5.93 GB/s |  7.07 GB/s |  7.74 GB/s |
| NEON @ Graviton4 |          … |          … |          … |          … |
| SVE2 @ Graviton4 |          … |          … |          … |          … |
