# Strings: Tapes, Rows, Escaping, Lookups, and Normalization

This directory holds the collection kernels behind `sz_strings_to_tape_best`, `sz_strings_to_rows_best`, `sz_strings_escape_best`, `sz_strings_unescape_best`, `sz_strings_lookup_best` and `sz_strings_normalize_best`.
Every one reads a `sz_strings_view_t` in any layout, optionally reordered through indices, and writes a tape or NUL-padded rows.
Every one that writes a tape is atomic: it succeeds and leaves the tape holding the whole result, or fails and leaves the tape, its blocks and their bytes exactly as they were.
Calls measure first and finish every fallible step, like a refused allocation or a malformed string, before the first write into a block the tape holds, and the allocator's `reallocate` is an optional fast path.
Each operation has a serial baseline plus per-ISA backends, like `haswell` and `icelake` on x86.
The backends share one set of drivers and differ only in their byte-set search, which skips the bytes needing no escaping, their lookup-table kernel, and their normalization kernels.
`sz_strings_lookup_best` and its siblings run the best kernel the capability mask they are passed names.
For hot paths on short inputs, pick a capability's kernel, like `sz_strings_lookup_neon`, or resolve one once through `sz_find_kernel_punned`, rather than paying the pick on every call.

## Methodology

Every table reports throughput in GB/s of input, where a GB is 2³⁰ bytes, measured with `bench/strings.cpp` for 5 seconds per row after a 1-second warm-up.
Lookups, escaping and tapes run over the first 4 MiB of `leipzig1M_en.txt`, and normalization over the first 4 MiB of the multilingual XL-Sum corpus.
Each call converts one batch of lines, as many as fill the first-level cache at the median line length: a few hundred English sentences, or a few XL-Sum articles.
The Standard row is the platform's best stock equivalent, `std::transform` for lookups.
The Loop rows convert one line at a time: a byte-by-byte escaping loop, or `sz_utf8_norm_best` into scratch sized for the worst case.
A `↑` cell means there is no dedicated kernel at that ISA level, so `_best` reuses the kernel from the tier above it, and a `…` cell is not measured yet.

## Lookups

| Backend                    | `sz_strings_lookup_best` |
| :------------------------- | -----------------------: |
| Standard @ 1× Intel Xeon6  |                 2.6 GB/s |
| Serial @ 1× Intel Xeon6    |                 2.5 GB/s |
| Haswell @ 1× Intel Xeon6   |                 2.3 GB/s |
| Skylake @ 1× Intel Xeon6   |                        ↑ |
| Ice Lake @ 1× Intel Xeon6  |                 4.1 GB/s |
| Standard @ 1× Apple M5 Pro |                 3.4 GB/s |
| Serial @ 1× Apple M5 Pro   |                 4.1 GB/s |
| NEON @ 1× Apple M5 Pro     |                 7.5 GB/s |
| SVE @ 1× AWS Graviton3     |                        … |

The Xeon6 rows predate the batches of lines and were measured over the corpus as one contiguous buffer.

## Escaping

| Backend                   |     JSON |      CSV |     HTML |       URL |
| :------------------------ | -------: | -------: | -------: | --------: |
| Loop @ 1× Apple M5 Pro    | 0.5 GB/s | 0.6 GB/s |        … |         … |
| Serial @ 1× Apple M5 Pro  | 1.2 GB/s | 1.2 GB/s | 1.2 GB/s | 0.66 GB/s |
| NEON @ 1× Apple M5 Pro    | 3.4 GB/s | 2.8 GB/s | 3.0 GB/s | 0.63 GB/s |
| Haswell @ 1× Intel Xeon6  |        … |        … |        … |         … |
| Ice Lake @ 1× Intel Xeon6 |        … |        … |        … |         … |

## Unescaping

| Backend                   |     JSON |      CSV |     HTML |       URL |
| :------------------------ | -------: | -------: | -------: | --------: |
| Serial @ 1× Apple M5 Pro  | 1.0 GB/s | 1.5 GB/s | 1.2 GB/s | 0.82 GB/s |
| NEON @ 1× Apple M5 Pro    | 3.3 GB/s | 4.0 GB/s | 2.8 GB/s | 0.76 GB/s |
| Haswell @ 1× Intel Xeon6  |        … |        … |        … |         … |
| Ice Lake @ 1× Intel Xeon6 |        … |        … |        … |         … |

URL escaping and unescaping run at serial speed on NEON for now: an escape every few bytes keeps the drivers in their per-escape path, so the byte-set search barely runs.

## Tapes

| Backend                  | `sz_strings_to_tape_best` |
| :----------------------- | ------------------------: |
| Serial @ 1× Apple M5 Pro |                 14.6 GB/s |
| NEON @ 1× Apple M5 Pro   |                 14.8 GB/s |

Gathering lines in order is one `memcpy` per batch, so every tier runs at the speed of `memcpy`.

## Normalization

| Backend                  |       NFC |      NFKC | NFKC_Casefold |
| :----------------------- | --------: | --------: | ------------: |
| Loop @ 1× Apple M5 Pro   |  1.7 GB/s |         … |     0.62 GB/s |
| Serial @ 1× Apple M5 Pro |  0.8 GB/s |  0.7 GB/s |     0.34 GB/s |
| NEON @ 1× Apple M5 Pro   |  1.7 GB/s |  1.6 GB/s |     0.61 GB/s |

A whole collection normalizes as fast as the loop over its lines while writing one tape, and NFKC_Casefold folds clean runs through the case-folding kernels.
