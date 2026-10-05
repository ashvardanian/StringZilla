# Overlap: Window Hashing and Prepared-Query Match Counting

This directory holds the kernels behind `sz_overlap_engine_init` and `sz_overlap_scores`, which prepare query windows and score candidate strings.
Each operation has a serial baseline plus `haswell` and `skylake` SIMD backends on x86 and `neon` on Arm, and `cuda`, `rocm` and `metal` backends on the device.
The CUDA and ROCm kernels share `simt.cuh`, each vendor launches them from its own host code in `cuda.cuh` and `rocm.cuh`, which `c/target/cuda.cu` and `c/target/rocm.hip` compile into the library, and the Metal ones live in `metal.h` with the `metal.metal` shaders, which `c/target/metal.c` compiles.
The init picks the best capability of a mask once, when the batch of queries is prepared on one device, and every later round scores with that capability's kernel alone.
A device engine keeps no stream: on CUDA it keeps no per-round state either, so any number of streams may score it at once, while on Metal `candidates_budget` sizes one round block at init and a round past it is refused with `sz_unexpected_dimensions_k`.

## Methodology

Cells are Mwin/s, millions of windows per second, one window per byte offset at the scored width, over words and lines from `xlsum.csv`.
A `…` cell is not measured yet.

The benchmark reads the first 64 MiB of `xlsum.csv` by default; `STRINGWARS_BYTES` controls the input size.
That slice averages 8.55 bytes per word and 4,976.32 bytes per line, with scored window widths of 2 and 5 respectively.

## Short Words

| Backend          | `sz_overlap_scores` |
| :--------------- | ------------------: |
| Serial @ Xeon6   |                   … |
| Haswell @ Xeon6  |                   … |
| Skylake @ Xeon6  |                   … |
| Serial @ M5 Pro  |               180.8 |
| NEON @ M5 Pro    |               201.7 |
| Metal @ M5 Pro   |               1,051 |
| CUDA @ SM90      |                   … |
| CUDA @ SM103 MIG |               545.1 |
| CUDA @ SM120     |                   … |

## Long Lines

| Backend          | `sz_overlap_scores` |
| :--------------- | ------------------: |
| Serial @ Xeon6   |                   … |
| Haswell @ Xeon6  |                   … |
| Skylake @ Xeon6  |                   … |
| Serial @ M5 Pro  |               133.1 |
| NEON @ M5 Pro    |               201.7 |
| Metal @ M5 Pro   |              25,020 |
| CUDA @ SM90      |                   … |
| CUDA @ SM103 MIG |               8,919 |
| CUDA @ SM120     |                   … |

## Window Hashes

A window hash is the window's own big-endian value reduced by a 32-bit prime, taken as the difference of two prefix hashes:

$$H(i, w) = P(i + w) - P(i) \cdot 256^{w} \bmod p, \qquad p = 4026525731.$$

One modular multiply-add serves any width and any offset, so the window widths need not form a doubling chain, and the window hash is full-width rather than a residue with spare bits above it.
Two 32-bit residues do not multiply exactly inside a 53-bit mantissa, so every product splits into the rounded head $ab$ and the exact tail $\mathrm{fma}(a, b, -ab)$, and the two reduce together.
Serial takes the same identity through a 64-bit integer product and one remainder by a compile-time constant, needing no fused multiply-add.

The prime is not interchangeable with another of its size.
Two windows collide exactly when their byte difference, read as a signed base-256 expansion, is a multiple of $p$, so what matters is the cheapest such multiple over the eight digits a machine word spans:

$$\mathrm{weight}(p) = \min \sum_{i=0}^{7} |c_i| \quad \text{over } c \neq 0 \text{ with } \sum_{i=0}^{7} c_i \cdot 256^{i} \equiv 0 \pmod p.$$

The largest prime below $2^{32}$ is $2^{32} - 5 = 256^{4} - 5$, whose weight is six: two texts differing by $+5$ at one byte and $-1$ four bytes along already collide.
This prime's weight is twenty-four, which `test/overlap.cpp` certifies by the same search that rediscovers the six.

## Window Widths

Every width's query window hashes share one B-tree: the step primitives are told a width and know nothing about why, while the engine takes the whole `window_widths` list at construction and `sz_overlap_scores` answers one score per query per candidate per width.
A candidate window hash is produced at a known width, so a hit is attributed to that width, and a coincidence with another width's window hash runs at $2^{-32}$.
What to do with those scores — weight them, or pick one per pair from the two texts' lengths — stays with the caller.

The width above which spurious window matches become rare follows from the alphabet's collision entropy $H_2$, the Rényi entropy of order two, in bits per symbol:

$$w^{\ast} = \frac{\log_2 \left( n_\text{query} \cdot n_\text{candidate} \right)}{H_2}.$$

That lands near 6 bytes for English sentences, 4 for protein, and 14 for DNA — the nominal alphabet size is the wrong input, since English text has a byte collision entropy near 2.2 bits, an effective alphabet near 4.6 rather than 27.
