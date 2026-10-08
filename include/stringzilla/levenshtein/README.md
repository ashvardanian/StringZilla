# Levenshtein: Edit Distances Under Unit Costs

This directory holds the kernels behind `sz_levenshtein_distances`, which scores a prepared batch of queries against a batch of candidates into a strided `[queries, candidates]` matrix, on the host or enqueued on a device's stream.
A batch is prepared by `sz_levenshtein_engine_init` for the best capability of a mask and the device its stream names, over bytes or over UTF-8 runes as `sz_levenshtein_symbol_t` spells it, and released by `sz_levenshtein_engine_free`.
The operation has a serial baseline plus per-ISA SIMD backends — `haswell`, `skylake`, `icelake` on x86 and `neon` on Arm — and `cuda`, `rocm` and `metal` backends on the device.
The CUDA and ROCm kernels share `simt.cuh`, each vendor launches them from its own host code in `cuda.cuh` and `rocm.cuh`, which `c/target/cuda.cu` and `c/target/rocm.hip` compile into the library, and the Metal ones live in `metal.h` with the `metal.metal` shaders, which `c/target/metal.c` compiles.
Each capability has its own init kernel, which records that capability in the engine, and every round runs the same capability's kernel on the device of its own stream; Ice Lake has no rune arm, so its kernel scores a rune batch with Skylake's.
The `cuda`, `rocm` and `metal` collection backends use an internal tiled wavefront for long byte pairs and dynamic Myers state for long rune queries.
Empty and mixed-length queries use the same engine and scoring API.
Long queries and candidates are grouped by length, and each bucket cross-product is split to fit its workspace and launch limits while preserving the original output order.
Temporary workspace belongs to each queued round, so independent streams can score one engine concurrently.

The batched kernels run Myers' bit-parallel algorithm: every query is a pattern, packed 64 symbols per machine word, and every candidate streams one symbol per step.
The engine prepares the whole batch's match masks once and advances several candidates per step — one per scalar state on `serial`, two per NEON register on `neon`, four per YMM on `haswell`, eight per ZMM on `skylake`, and sixty-four byte lanes per ZMM on `icelake` when the query is eight symbols or fewer.
A byte is its own mask class and a rune takes the class the query assigned it through a two-level page table over Unicode, so the rune alphabet only swaps the query preparation and the candidate transpose.
Each backend also exports its building blocks: a `state` per register of candidates, a `vertical` per query word, the `init`, `step`, `any_active` and `score` verbs over them, and the `transpose` that feeds a step its class ids.

## Methodology

Cells are GCUPS, billions of cell updates per second, over words and lines from `xlsum.csv`.
A `↑` cell reuses the kernel of the tier above, and a `…` cell is not measured yet.

The benchmark reads the whole `xlsum.csv` by default; `STRINGWARS_BYTES` controls the input size.
That slice averages 8.55 bytes per word and 4,976.32 bytes per line, with median query byte limits of 6 and 3,460 respectively.
UTF-8 rows use the harness's byte-based cell-update count, rather than a count of decoded rune pairs.

## Batches Over Byte Strings

| Backend                   | Short Words |  Long Lines |
| :------------------------ | ----------: | ----------: |
| Serial @ 1× Intel Xeon6   |  0.36 GCUPS | 29.72 GCUPS |
| Haswell @ 1× Intel Xeon6  |  0.79 GCUPS | 49.21 GCUPS |
| Skylake @ 1× Intel Xeon6  |  0.60 GCUPS | 52.80 GCUPS |
| Ice Lake @ 1× Intel Xeon6 |  0.76 GCUPS |           ↑ |
| Serial @ 1× AWS Graviton4 |           … |           … |
| Serial @ 1× Apple M5 Pro  |  1.25 GCUPS | 39.88 GCUPS |
| NEON @ 1× Apple M5 Pro    |  1.78 GCUPS | 44.55 GCUPS |
| Metal @ Apple M5 Pro      | 50.01 GCUPS | 1,309 GCUPS |
| CUDA @ Nvidia SM90        |           … |           … |
| CUDA @ 18× Nvidia SM103   | 27.87 GCUPS | 3,603 GCUPS |
| CUDA @ Nvidia SM120       |  3.59 GCUPS |           … |

## Batches Over UTF-8 Strings

| Backend                   | Short Words |   Long Lines |
| :------------------------ | ----------: | -----------: |
| Serial @ 1× Intel Xeon6   |  0.29 GCUPS |  58.48 GCUPS |
| Haswell @ 1× Intel Xeon6  |  0.37 GCUPS |  98.78 GCUPS |
| Skylake @ 1× Intel Xeon6  |  0.32 GCUPS | 108.51 GCUPS |
| Ice Lake @ 1× Intel Xeon6 |           ↑ |            ↑ |
| Serial @ 1× AWS Graviton4 |           … |            … |
| Serial @ 1× Apple M5 Pro  |  0.37 GCUPS |  73.66 GCUPS |
| NEON @ 1× Apple M5 Pro    |  0.40 GCUPS |  92.13 GCUPS |
| Metal @ Apple M5 Pro      | 44.05 GCUPS | 915.15 GCUPS |
| CUDA @ Nvidia SM90        |           … |            … |
| CUDA @ 18× Nvidia SM103   | 31.23 GCUPS |  4,264 GCUPS |
| CUDA @ Nvidia SM120       |           … |            … |

## Long Byte Collections

The internal tiled path uses a wavefront recurrence for long byte pairs within a collection.
Each measurement below submits a 1 × 1 cross-product through `sz_levenshtein_distances`.
Cells are milliseconds for equal-length strings over four byte values, reporting the median of seven runs and including device queue synchronization.

| Backend                  | 16,384 Bytes | 32,768 Bytes |
| :----------------------- | -----------: | -----------: |
| Serial @ 1× Apple M5 Pro |         8.40 |        27.50 |
| Metal @ Apple M5 Pro     |         7.43 |        15.01 |

The collection keeps short pairs on the Myers kernels because tiled launch costs outweigh their parallelism at shorter lengths.
