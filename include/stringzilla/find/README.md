# Find: Substring, Byte, and Byte Set Search

This directory holds the search kernels behind `sz_find_best`, `sz_rfind_best`, `sz_find_byte_best`, `sz_rfind_byte_best`, `sz_find_byteset_best`, and `sz_rfind_byteset_best`.
Each operation has a serial SWAR baseline plus per-ISA SIMD backends — `westmere`, `haswell`, `skylake`, `icelake` on x86, `neon`, `sve`, `sve2` on Arm, and `rvv`, `loongsonasx`, `powervsx`, `v128` elsewhere.
`sz_find_best` and its siblings run the best kernel the capability mask they are passed names.
For hot paths on short inputs, like a per-token `find_byte`, pick a capability's kernel, like `sz_find_byte_neon`, or resolve one once through `sz_find_kernel_punned`, rather than paying the pick on every call.

## Methodology

Numbers are throughput, shown in GB/s in each cell, measured with `bench/find.cpp` over the `leipzig1M.txt` corpus, reporting the median of repeated runs.
The Standard row is the platform's best stock equivalent per column — `strstr` and `std::find_end` for substring search, `memchr` for the byte variants, and `strpbrk`/`strcspn` for the byte-set variants.
Substring search depends sharply on needle length, so results are split into a Short Words table with tokens averaging 5 bytes and a Long Lines table with tokens averaging 130 bytes.
A `↑` cell means there is no dedicated kernel at that ISA level, so `_best` reuses the kernel from the tier above it; an empty cell is genuinely-missing data.

## Short Words

| Backend          | `sz_find_best` | `sz_rfind_best` | `sz_find_byte_best` | `sz_rfind_byte_best` | `sz_find_byteset_best` | `sz_rfind_byteset_best` |
| :--------------- | -------------: | --------------: | ------------------: | -------------------: | ---------------------: | ----------------------: |
| Standard @ Xeon4 |      2.56 GB/s |       0.24 GB/s |           0.22 GB/s |            0.22 GB/s |              0.07 GB/s |              0.093 GB/s |
| Serial @ Xeon4   |      2.49 GB/s |       0.61 GB/s |           0.33 GB/s |            0.36 GB/s |              0.26 GB/s |               0.27 GB/s |
| Westmere @ Xeon4 |      4.96 GB/s |       4.29 GB/s |           0.22 GB/s |            0.27 GB/s |                      ↑ |                       ↑ |
| Haswell @ Xeon4  |      5.53 GB/s |       4.88 GB/s |           0.32 GB/s |            0.28 GB/s |              0.25 GB/s |               0.26 GB/s |
| Skylake @ Xeon4  |      8.96 GB/s |       8.60 GB/s |           0.57 GB/s |            0.61 GB/s |                      ↑ |                       ↑ |
| Ice Lake @ Xeon4 |              ↑ |               ↑ |                   ↑ |                    ↑ |              0.26 GB/s |               0.29 GB/s |
| NEON @ Graviton4 |              … |               … |                   … |                    … |                      … |                       … |
| SVE @ Graviton3  |              … |               … |                   … |                    … |                      … |                       … |

## Long Lines

| Backend          | `sz_find_best` | `sz_rfind_best` | `sz_find_byte_best` | `sz_rfind_byte_best` | `sz_find_byteset_best` | `sz_rfind_byteset_best` |
| :--------------- | -------------: | --------------: | ------------------: | -------------------: | ---------------------: | ----------------------: |
| Standard @ Xeon4 |     16.52 GB/s |       5.80 GB/s |           1.93 GB/s |            1.93 GB/s |              0.23 GB/s |               0.21 GB/s |
| Serial @ Xeon4   |      5.29 GB/s |       5.59 GB/s |           1.44 GB/s |            1.42 GB/s |              1.11 GB/s |               1.09 GB/s |
| Westmere @ Xeon4 |     11.97 GB/s |      9.891 GB/s |           1.95 GB/s |            1.89 GB/s |                      ↑ |                       ↑ |
| Haswell @ Xeon4  |     11.51 GB/s |      12.19 GB/s |           1.70 GB/s |            1.66 GB/s |              2.43 GB/s |               2.52 GB/s |
| Skylake @ Xeon4  |     17.49 GB/s |      17.61 GB/s |           1.73 GB/s |            1.77 GB/s |                      ↑ |                       ↑ |
| Ice Lake @ Xeon4 |              ↑ |               ↑ |                   ↑ |                    ↑ |              3.56 GB/s |               3.43 GB/s |
| NEON @ Graviton4 |              … |               … |                   … |                    … |                      … |                       … |
| SVE @ Graviton3  |              … |               … |                   … |                    … |                      … |                       … |
