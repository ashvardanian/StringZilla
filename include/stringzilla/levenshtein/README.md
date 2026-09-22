# Levenshtein: Edit Distances Under Unit Costs

This directory holds the kernels behind `sz_levenshtein_distance` and `sz_levenshtein_distances` over byte strings, and `sz_levenshtein_distance_utf8` and `sz_levenshtein_distances_utf8` over UTF-8 strings counted in runes.
Each operation has a serial baseline plus per-ISA SIMD backends — `haswell`, `skylake`, `icelake` on x86 — and a `cuda` backend on the GPU.
The dispatcher picks the fastest one the running machine carries.

All four run Myers' bit-parallel algorithm: the query is the pattern, packed 64 symbols per machine word, and every candidate streams one symbol per step.
The one-to-many forms prepare the query's match masks once and advance several candidates per step — one per scalar state on `serial`, four per YMM on `haswell`, eight per ZMM on `skylake`, and sixty-four byte lanes per ZMM on `icelake` when the query is eight symbols or fewer.
A byte is its own mask class and a rune takes the class the query assigned it through a two-level page table over Unicode, so the UTF-8 forms only swap the query preparation and the candidate transpose.
Each backend also exports its building blocks: a `state` per register of candidates, a `vertical` per query word, the `init`, `step`, `any_active` and `score` verbs over them, and the `transpose` that feeds a step its class ids.

## Methodology

Numbers are throughput in cell updates per second, shown in GCUPS in each cell, one cell per query symbol per candidate symbol, measured with `bench/levenshtein.cpp` over the `xlsum.csv` corpus on one pinned core.
Each row is the library compiled with that single backend forced on one fixed chip, and each column is one operation, so coverage and cross-chip comparison read down a single column.
There is no Standard row here, since no standard library ships an edit distance, so the Serial row is the reference.
Token length matters, so results are split into a Short Words column with tokens averaging 9 bytes and a Long Lines column with tokens averaging 3 KB, and the one-to-one table reports the Long Lines run.
The GPU rows come from `bench/levenshtein.cu` and score one residency wave of one candidate per thread.
A `↑` cell means there is no dedicated kernel at that ISA level, so the dispatcher reuses the kernel from the tier above it; a `…` cell is genuinely-missing data.

## One to One

| Backend            | `sz_levenshtein_distance` | `sz_levenshtein_distance_utf8` |
| :----------------- | ------------------------: | -----------------------------: |
| Serial @ Xeon6     |               30.64 GCUPS |                    59.20 GCUPS |
| Serial @ Graviton4 |                         … |                              … |

> Measured September 22nd, 2026.

## One to Many, Byte Strings

| Backend            | Short Words |  Long Lines |
| :----------------- | ----------: | ----------: |
| Serial @ Xeon6     |  0.58 GCUPS | 29.72 GCUPS |
| Haswell @ Xeon6    |  1.07 GCUPS | 49.21 GCUPS |
| Skylake @ Xeon6    |  0.79 GCUPS | 52.80 GCUPS |
| Ice Lake @ Xeon6   |  1.04 GCUPS |           ↑ |
| Serial @ Graviton4 |           … |           … |
| CUDA @ SM90        |           … |           … |
| CUDA @ SM120       |  3.59 GCUPS |           … |

> Measured September 22nd, 2026.

## One to Many, UTF-8 Strings

| Backend            | Short Words |   Long Lines |
| :----------------- | ----------: | -----------: |
| Serial @ Xeon6     |  0.19 GCUPS |  58.48 GCUPS |
| Haswell @ Xeon6    |  0.23 GCUPS |  98.78 GCUPS |
| Skylake @ Xeon6    |  0.27 GCUPS | 108.51 GCUPS |
| Ice Lake @ Xeon6   |           ↑ |            ↑ |
| Serial @ Graviton4 |           … |            … |
| CUDA @ SM90        |           … |            … |
| CUDA @ SM120       |           … |            … |

> Measured September 22nd, 2026.
