# Strings: Tapes, Rows, Escaping, Lookups, Normalization, Splitting, and Joining

This directory holds the collection kernels behind `sz_strings_to_tape_best`, `sz_strings_to_rows_best`, `sz_strings_escape_best`, `sz_strings_unescape_best`, `sz_strings_lookup_best`, `sz_strings_normalize_best`, `sz_strings_split_best`, `sz_strings_split_lines_best` and `sz_strings_join_best`.
Every one reads a `sz_strings_view_t` in any layout, optionally reordered through indices, or one text, and writes a tape, NUL-padded rows, or one buffer.
Every one that writes a tape is atomic: it succeeds and leaves the tape holding the whole result, or fails and leaves the tape, its blocks and their bytes exactly as they were.
Calls measure first and finish every fallible step, like a refused allocation or a malformed string, before the first write into a block the tape holds, and the allocator's `reallocate` is an optional fast path.
Each operation has a serial baseline plus per-ISA backends, like `haswell` and `icelake` on x86.
The backends share one set of drivers and differ only in their byte-set search, which skips the bytes needing no escaping, their lookup-table kernel, and their normalization kernels, while NEON and Ice Lake split a text in loops of their own.
`sz_strings_lookup_best` and its siblings run the best kernel the capability mask they are passed names.
For hot paths on short inputs, pick a capability's kernel, like `sz_strings_lookup_neon`, or resolve one once through `sz_find_kernel_punned`, rather than paying the pick on every call.

## Methodology

`bench/strings.cpp` reports GB/s of input over the lines of `leipzig1M_en.txt`, or of XL-Sum for normalization.
Each call converts one batch of lines that fits the first-level cache, while splits take the text whole.
Loop rows handle one line or part at a time, and Standard rows call the platform's stock equivalent, like `std::transform`.
A `↑` cell reuses the tier above, and a `…` cell is not measured yet.

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

## Tapes

| Backend                  | `sz_strings_to_tape_best` |
| :----------------------- | ------------------------: |
| Serial @ 1× Apple M5 Pro |                 14.6 GB/s |
| NEON @ 1× Apple M5 Pro   |                 14.8 GB/s |

## Normalization

| Backend                  |       NFC |      NFKC | NFKC_Casefold |
| :----------------------- | --------: | --------: | ------------: |
| Loop @ 1× Apple M5 Pro   |  1.7 GB/s |         … |     0.62 GB/s |
| Serial @ 1× Apple M5 Pro |  0.8 GB/s |  0.7 GB/s |     0.34 GB/s |
| NEON @ 1× Apple M5 Pro   |  1.7 GB/s |  1.6 GB/s |     0.61 GB/s |

## Splitting and Joining

| Backend                   | `split` at `' '` | `split_lines` |     `join` |
| :------------------------ | ---------------: | ------------: | ---------: |
| Loop @ 1× Apple M5 Pro    |        0.41 GB/s |      1.5 GB/s |   6.4 GB/s |
| Serial @ 1× Apple M5 Pro  |                … |             … |          … |
| NEON @ 1× Apple M5 Pro    |                … |             … |          ↑ |
| Haswell @ 1× Intel Xeon6  |                ↑ |             ↑ |          ↑ |
| Ice Lake @ 1× Intel Xeon6 |                … |             … |          ↑ |

The vector kernels classify each 64-byte block into the bytes they remove and the ones that end parts, then set the offset of every end from the bytes kept before it.
NEON slides each kept segment down with overlapping 16-byte stores, and Ice Lake packs the whole block with one `vpcompressb`.
