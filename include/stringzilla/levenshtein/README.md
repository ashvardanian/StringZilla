# Levenshtein: Edit Distances Under Unit Costs

This directory holds the kernels behind `sz_levenshtein_distance` and `sz_levenshtein_distances` over byte strings, and `sz_levenshtein_distance_utf8` and `sz_levenshtein_distances_utf8` over UTF-8 strings counted in runes.
The one-to-many operations have a serial baseline plus per-ISA SIMD backends — `haswell` and `icelake` on x86 — and the dispatcher picks the fastest one available on the running CPU.
The one-to-one operations fill a single candidate, so every backend runs the serial walk.

All four run Myers' bit-parallel algorithm: the query is the pattern, packed 64 symbols per machine word, and every candidate streams one symbol per step.
The one-to-many forms prepare the query's match masks once and advance several candidates per step — one per scalar state on `serial`, four per YMM on `haswell`, eight per ZMM on `icelake` — so the table is built once per query rather than once per pair.
A byte is its own mask class; a rune takes the class the query assigned it through a two-level page table over Unicode, or class zero when the query lacks it, so the UTF-8 forms only swap the query preparation and the candidate stripe.
Each backend also exports its building blocks for callers that own the loop nest: a `state` per register of candidates, a `vertical` per query word, and the `init`, `step`, `any_active`, and `score` verbs over them, named by candidates per step — `sz_levenshtein_u64x1_*_serial`, `sz_levenshtein_u64x4_*_haswell`, `sz_levenshtein_u64x8_*_icelake` — beside the byte stripes `sz_levenshtein_u8x4_stripe_haswell` and `sz_levenshtein_u8x8_stripe_icelake`.

## Methodology

Cells are throughput in billions of cell updates per second, one cell per query symbol per candidate symbol, measured with `bench/levenshtein.cpp` over the lines of the `leipzig1M.txt` corpus on one core pinned away from the scheduler, reporting the median of repeated runs.
Each row is the library compiled with that single backend forced on one fixed chip, and each column is one operation, so coverage and cross-chip comparison read down a single column.
The Serial row is the reference; there is no Standard row here, since no standard library ships an edit distance.
The one-to-one columns score a line against its successor, and the one-to-many columns score a line against the 64 lines after it, at two query widths — clamped to 64 bytes, one Myers word, and to 512 bytes, up to eight words.
A `…` cell is genuinely-missing data, on a backend not yet measured on hardware that runs it.

## One to One

| Backend            | `sz_levenshtein_distance` | `sz_levenshtein_distance_utf8` |
| :----------------- | ------------------------: | -----------------------------: |
| Serial @ Xeon4     |                         … |                              … |
| Serial @ Graviton4 |                         … |                              … |

## One to Many, Byte Strings

| Backend            | 64 B queries | 512 B queries |
| :----------------- | -----------: | ------------: |
| Serial @ Xeon4     |            … |             … |
| Haswell @ Xeon4    |            … |             … |
| Ice Lake @ Xeon4   |            … |             … |
| Serial @ Graviton4 |            … |             … |

## One to Many, UTF-8 Strings

| Backend            | 64 B queries | 512 B queries |
| :----------------- | -----------: | ------------: |
| Serial @ Xeon4     |            … |             … |
| Haswell @ Xeon4    |            … |             … |
| Ice Lake @ Xeon4   |            … |             … |
| Serial @ Graviton4 |            … |             … |
