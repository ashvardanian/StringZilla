# Norm: UTF-8 Unicode Normalization

This directory holds the kernels behind `sz_utf8_norm` and `sz_utf8_find_denormalized`, which bring UTF-8 text into a UAX-15 normalization form — NFC, NFD, NFKC, or NFKD.
Normalization reorders combining marks and composes or decomposes characters so that strings that look the same compare equal byte for byte.
The companion quick-check scan finds the first character that breaks the requested form, so callers can skip the rewrite when a string is already normalized.
Every operation has a serial baseline plus per-ISA SIMD backends, and the dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput in MB/s, measured with `bench/utf8_norm.cpp` over the full multilingual `xlsum.csv` corpus, reporting the median of repeated runs.
Each table fixes one input shape; its columns are the Normalize NFC and Quick Check operations and its rows are a backend on a chip, so reading down a column compares the same operation across the backend ladder while reading across a row compares operations on one backend.
The Serial row is the reference; there is no Standard row here, since no standard library ships Unicode normalization.
For a comparison against ICU, see the main project README.
Results are split into a Short Words workload (tokens averaging a handful of bytes) and a Long Lines workload (full sentences).
A `↑` cell means there is no dedicated kernel for that operation at that backend, so the dispatcher reuses the tier above it.

## Short Words

| Backend          | `sz_utf8_norm` | `sz_utf8_find_denormalized` |
| :--------------- | -------------: | -----------------------: |
| Serial @ Xeon4   |     112.2 MB/s |               131.5 MB/s |
| Haswell @ Xeon4  |     121.4 MB/s |               211.6 MB/s |
| Skylake @ Xeon4  |     122.6 MB/s |               217.8 MB/s |
| Ice Lake @ Xeon4 |     122.0 MB/s |               212.6 MB/s |
| NEON @ Graviton4 |              … |                        … |
| SVE2 @ Graviton4 |              … |                        … |
| SVE @ Graviton3  |              … |                        … |

> Measured June 26th, 2026.

## Long Lines

| Backend          | `sz_utf8_norm` | `sz_utf8_find_denormalized` |
| :--------------- | -------------: | -----------------------: |
| Serial @ Xeon4   |     215.5 MB/s |               266.4 MB/s |
| Haswell @ Xeon4  |     376.5 MB/s |               492.6 MB/s |
| Skylake @ Xeon4  |     372.7 MB/s |               530.7 MB/s |
| Ice Lake @ Xeon4 |     379.9 MB/s |               530.2 MB/s |
| NEON @ Graviton4 |              … |                        … |
| SVE2 @ Graviton4 |              … |                        … |
| SVE @ Graviton3  |              … |                        … |

> Measured June 26th, 2026.
