# Levenshtein: Edit Distances Under Unit Costs

This directory holds the kernels behind `sz_levenshtein_distances`, which scores a prepared batch of queries against a batch of candidates into a strided `[queries, candidates]` matrix, on the host or enqueued on a device's stream.
A batch is prepared by `sz_levenshtein_engine_init` for the best capability of a mask and the device its stream names, over bytes or over UTF-8 runes as `sz_levenshtein_symbol_t` spells it, and released by `sz_levenshtein_engine_free`.
The operation has a serial baseline plus per-ISA SIMD backends — `haswell`, `skylake`, `icelake` on x86 — and `cuda`, `rocm` and `metal` backends on the device.
The CUDA and ROCm kernels share `simt.cuh`, each vendor launches them from its own host code in `cuda.cuh` and `rocm.cuh`, which `c/target/cuda.cu` and `c/target/rocm.hip` compile into the library, and the Metal ones live in `metal.h` with the `metal.metal` shaders, which `c/target/metal.c` compiles.
Each capability has its own init kernel, which records that capability in the engine, and every round runs the same capability's kernel on the device of its own stream; Ice Lake has no rune arm, so its kernel scores a rune batch with Skylake's.
The `cuda` and `rocm` backends also hold the tiled wavefront behind `sz_levenshtein_distance_tiled_best`, for one pair too long for a single thread's recurrence: it runs in caller-owned device scratch of `sz_levenshtein_distance_tiled_scratch_bytes`, allocates and joins nothing, and leaves the distance in device-reachable memory once the caller joins the stream.

All of them run Myers' bit-parallel algorithm: every query is a pattern, packed 64 symbols per machine word, and every candidate streams one symbol per step.
The engine prepares the whole batch's match masks once and advances several candidates per step — one per scalar state on `serial`, four per YMM on `haswell`, eight per ZMM on `skylake`, and sixty-four byte lanes per ZMM on `icelake` when the query is eight symbols or fewer.
A byte is its own mask class and a rune takes the class the query assigned it through a two-level page table over Unicode, so the rune alphabet only swaps the query preparation and the candidate transpose.
Each backend also exports its building blocks: a `state` per register of candidates, a `vertical` per query word, the `init`, `step`, `any_active` and `score` verbs over them, and the `transpose` that feeds a step its class ids.

## Methodology

Cells are GCUPS, billions of cell updates per second, over `xlsum.csv` words averaging 9 bytes and lines averaging 3 KB.
A `↑` cell reuses the kernel of the tier above, and a `…` cell is not measured yet.

## Batches Over Byte Strings

| Backend            | Short Words |  Long Lines |
| :----------------- | ----------: | ----------: |
| Serial @ Xeon6     |  0.58 GCUPS | 29.72 GCUPS |
| Haswell @ Xeon6    |  1.07 GCUPS | 49.21 GCUPS |
| Skylake @ Xeon6    |  0.79 GCUPS | 52.80 GCUPS |
| Ice Lake @ Xeon6   |  1.04 GCUPS |           ↑ |
| Serial @ Graviton4 |           … |           … |
| CUDA @ SM90        |           … |           … |
| CUDA @ SM103 MIG   | 33.88 GCUPS | 3,603 GCUPS |
| CUDA @ SM120       |  3.59 GCUPS |           … |

> Measured September 22nd, 2026, and October 2nd, 2026, for SM103.

## Batches Over UTF-8 Strings

| Backend            | Short Words |   Long Lines |
| :----------------- | ----------: | -----------: |
| Serial @ Xeon6     |  0.19 GCUPS |  58.48 GCUPS |
| Haswell @ Xeon6    |  0.23 GCUPS |  98.78 GCUPS |
| Skylake @ Xeon6    |  0.27 GCUPS | 108.51 GCUPS |
| Ice Lake @ Xeon6   |           ↑ |            ↑ |
| Serial @ Graviton4 |           … |            … |
| CUDA @ SM90        |           … |            … |
| CUDA @ SM103 MIG   | 36.72 GCUPS |  4,264 GCUPS |
| CUDA @ SM120       |           … |            … |

> Measured September 22nd, 2026, and October 2nd, 2026, for SM103.
