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

| Backend                   |     256 B |      1 KB |      4 KB |     16 KB |
| :------------------------ | --------: | --------: | --------: | --------: |
| Serial @ 1× Intel Xeon6   | 0.04 GB/s | 0.04 GB/s | 0.04 GB/s | 0.04 GB/s |
| Westmere @ 1× Intel Xeon6 | 5.20 GB/s | 5.38 GB/s | 5.56 GB/s | 5.57 GB/s |
| Ice Lake @ 1× Intel Xeon6 | 12.0 GB/s | 13.2 GB/s | 13.3 GB/s | 13.3 GB/s |
| NEON @ 1× AWS Graviton4   |         … |         … |         … |         … |
| SVE2 @ 1× AWS Graviton4   |         … |         … |         … |         … |

## Galois/Counter Mode

| Backend                   |      256 B |       1 KB |       4 KB |      16 KB |
| :------------------------ | ---------: | ---------: | ---------: | ---------: |
| Serial @ 1× Intel Xeon6   | 0.007 GB/s | 0.007 GB/s | 0.007 GB/s | 0.006 GB/s |
| Westmere @ 1× Intel Xeon6 |  1.91 GB/s |  2.44 GB/s |  2.62 GB/s |  2.59 GB/s |
| Ice Lake @ 1× Intel Xeon6 |  4.11 GB/s |  6.02 GB/s |  6.87 GB/s |  7.04 GB/s |
| NEON @ 1× AWS Graviton4   |          … |          … |          … |          … |
| SVE2 @ 1× AWS Graviton4   |          … |          … |          … |          … |
