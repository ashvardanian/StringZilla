# Levenshtein: Edit Distances Under Unit Costs

This directory holds the kernels behind `sz_levenshtein_distance` and `sz_levenshtein_distances` over byte strings, and `sz_levenshtein_distance_utf8` and `sz_levenshtein_distances_utf8` over UTF-8 strings counted in runes.
The one-to-many operations have a serial baseline plus per-ISA SIMD backends — `haswell` and `icelake` on x86 — and the dispatcher picks the fastest one available on the running CPU.
The one-to-one operations fill a single candidate, so every backend runs the serial walk.

All four run Myers' bit-parallel algorithm: the query is the pattern, packed 64 symbols per machine word, and every candidate streams one symbol per step.
The one-to-many forms prepare the query's match masks once and advance several candidates per step — one per scalar state on `serial`, four per YMM on `haswell`, eight per ZMM on `icelake` — so the table is built once per query rather than once per pair.
A byte is its own mask class; a rune takes the class the query assigned it through a two-level page table over Unicode, or class zero when the query lacks it, so the UTF-8 forms only swap the query preparation and the candidate stripe.
Each backend also exports its building blocks for callers that own the loop nest: a `state` per register of candidates, a `vertical` per query word, and the `init`, `step`, `any_active`, and `score` verbs over them, named by candidates per step — `sz_levenshtein_u64x1_*_serial`, `sz_levenshtein_u64x4_*_haswell`, `sz_levenshtein_u64x8_*_icelake` — beside the byte stripes `sz_levenshtein_u8x4_stripe_haswell` and `sz_levenshtein_u8x8_stripe_icelake`.

## Methodology

Cells are throughput in cell updates per second, rendered as GCUPS, one cell per query symbol per candidate symbol, measured with `bench/levenshtein.cpp` over the `xlsum.csv` corpus on one pinned core, reporting the median of repeated runs.
Each row is the library compiled with that single backend forced on one fixed chip, and each column is one operation, so coverage and cross-chip comparison read down a single column.
There is no Standard row here, since no standard library ships an edit distance, so the Serial row is the reference.
The one-to-one table is keyed by verb rather than by token length, and reports the Long Lines run.
Token length matters, so results are split into a Short Words column (tokens averaging 9 bytes) and a Long Lines column (tokens averaging 3 KB), and every query is a token clamped to 1024 bytes — the widest the device's per-thread verticals hold.
The GPU rows come from `bench/levenshtein.cu`, score one residency wave of one candidate per thread, and carry only the one-to-many columns.
The CUDA backend holds sixteen Myers words, so it refuses a query past 1024 bytes and its Long Lines cell stays empty until that ceiling rises.
A `…` cell is genuinely-missing data, on a backend not yet measured on hardware that runs it.

## One to One

| Backend            | `sz_levenshtein_distance` | `sz_levenshtein_distance_utf8` |
| :----------------- | ------------------------: | -----------------------------: |
| Serial @ Xeon6     |                     27.91 |                          48.84 |
| Serial @ Graviton4 |                         … |                              … |

## One to Many, Byte Strings

| Backend            | Short Words | Long Lines |
| :----------------- | ----------: | ---------: |
| Serial @ Xeon6     |        0.99 |      27.62 |
| Haswell @ Xeon6    |        2.12 |      45.66 |
| Ice Lake @ Xeon6   |        1.79 |      48.42 |
| Serial @ Graviton4 |           … |          … |
| CUDA @ SM90        |           … |          … |
| CUDA @ SM120       |        3.59 |          … |

## One to Many, UTF-8 Strings

| Backend            | Short Words | Long Lines |
| :----------------- | ----------: | ---------: |
| Serial @ Xeon6     |        0.42 |      39.93 |
| Haswell @ Xeon6    |        0.45 |      74.86 |
| Ice Lake @ Xeon6   |        0.59 |      83.99 |
| Serial @ Graviton4 |           … |          … |
| CUDA @ SM90        |           … |          … |
| CUDA @ SM120       |           … |          … |
