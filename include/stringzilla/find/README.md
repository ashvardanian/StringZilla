# Find: Substring, Byte, and Byte Set Search

This directory holds the search kernels behind `sz_find_best`, `sz_rfind_best`, `sz_find_byte_best`, `sz_rfind_byte_best`, `sz_find_byteset_best`, and `sz_rfind_byteset_best`.
Each operation has a serial SWAR baseline plus per-ISA SIMD backends — `westmere`, `haswell`, `skylake`, `icelake` on x86, `neon`, `sve`, `sve2` on Arm, and `rvv`, `loongsonasx`, `powervsx`, `v128` elsewhere.
`sz_find_best` and its siblings run the best kernel the capability mask they are passed names.
For hot paths on short inputs, like a per-token `find_byte`, pick a capability's kernel, like `sz_find_byte_neon`, or resolve one once through `sz_find_kernel_punned`, rather than paying the pick on every call.

## Methodology

Numbers are throughput, shown in GB/s in each cell, measured with `bench/find.cpp` over the `leipzig1M.txt` corpus, reporting the median of repeated runs.
The Standard row is the platform's best stock equivalent per column — `strstr`, `memmem` and the `std::boyer_moore` searchers for substring search, `memchr` and `std::find` for the forward byte search, which has no stock reverse twin, and `strcspn` and `std::string_view::find_first_of`/`find_last_of` for the byte-set variants.
Substring search depends sharply on needle length, so results are split into a Short Words table with tokens averaging 5 bytes and a Long Lines table with tokens averaging 130 bytes.
A `↑` cell means there is no dedicated kernel at that ISA level, so `_best` reuses the kernel from the tier above it; an empty cell is genuinely-missing data.

## Short Words

| Backend                   | `sz_find_best` | `sz_rfind_best` | `sz_find_byte_best` | `sz_rfind_byte_best` | `sz_find_byteset_best` | `sz_rfind_byteset_best` |
| :------------------------ | -------------: | --------------: | ------------------: | -------------------: | ---------------------: | ----------------------: |
| Standard @ 1× Intel Xeon6 |      7.11 GB/s |       0.74 GB/s |           1.72 GB/s |                    … |              0.28 GB/s |               0.28 GB/s |
| Serial @ 1× Intel Xeon6   |      2.40 GB/s |       0.60 GB/s |           0.50 GB/s |            0.55 GB/s |              0.52 GB/s |               0.51 GB/s |
| Westmere @ 1× Intel Xeon6 |      7.41 GB/s |       6.23 GB/s |           0.49 GB/s |            0.51 GB/s |                      ↑ |                       ↑ |
| Haswell @ 1× Intel Xeon6  |      8.34 GB/s |       8.12 GB/s |           0.49 GB/s |            0.51 GB/s |              0.48 GB/s |               0.46 GB/s |
| Skylake @ 1× Intel Xeon6  |      8.87 GB/s |       8.70 GB/s |           1.48 GB/s |            1.56 GB/s |                      ↑ |                       ↑ |
| Ice Lake @ 1× Intel Xeon6 |              ↑ |               ↑ |                   ↑ |                    ↑ |              0.72 GB/s |               0.75 GB/s |
| NEON @ 1× AWS Graviton4   |              … |               … |                   … |                    … |                      … |                       … |
| SVE @ 1× AWS Graviton3    |              … |               … |                   … |                    … |                      … |                       … |

## Long Lines

| Backend                   | `sz_find_best` | `sz_rfind_best` | `sz_find_byte_best` | `sz_rfind_byte_best` | `sz_find_byteset_best` | `sz_rfind_byteset_best` |
| :------------------------ | -------------: | --------------: | ------------------: | -------------------: | ---------------------: | ----------------------: |
| Standard @ 1× Intel Xeon6 |     20.87 GB/s |       6.36 GB/s |           2.06 GB/s |                    … |              2.11 GB/s |               0.58 GB/s |
| Serial @ 1× Intel Xeon6   |      5.44 GB/s |       5.99 GB/s |           1.51 GB/s |            1.47 GB/s |              1.41 GB/s |               1.39 GB/s |
| Westmere @ 1× Intel Xeon6 |     16.43 GB/s |      12.58 GB/s |           2.02 GB/s |            1.89 GB/s |                      ↑ |                       ↑ |
| Haswell @ 1× Intel Xeon6  |     17.77 GB/s |      16.63 GB/s |           1.74 GB/s |            1.77 GB/s |              3.16 GB/s |               3.32 GB/s |
| Skylake @ 1× Intel Xeon6  |     20.71 GB/s |      19.89 GB/s |           1.68 GB/s |            1.84 GB/s |                      ↑ |                       ↑ |
| Ice Lake @ 1× Intel Xeon6 |              ↑ |               ↑ |                   ↑ |                    ↑ |              4.92 GB/s |               4.68 GB/s |
| NEON @ 1× AWS Graviton4   |              … |               … |                   … |                    … |                      … |                       … |
| SVE @ 1× AWS Graviton3    |              … |               … |                   … |                    … |                      … |                       … |
