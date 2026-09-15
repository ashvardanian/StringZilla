# Overlap: Window Hashing and Prepared-Query Match Counting

This directory holds the kernels behind `sz_overlap_score` and `sz_overlap_scores`, plus the prefix-hash, window-hash, key-sort and B-tree probe primitives every backend shares.
Each operation has a serial baseline plus `haswell` and `skylake` SIMD backends on x86.
The dispatcher picks the fastest one available on the running CPU.

## Methodology

Numbers are throughput in windows per second, rendered as Mwin/s, measured with `bench/overlap.cpp` over the `leipzig1M.txt` corpus, reporting the median of repeated runs.
One window is one byte offset at the scored width.
Each row is the library compiled with that single backend forced on one fixed chip, and each column is one stage.
Query length decides whether the B-tree fits L1, so results split into a Short Queries table at the corpus's median token length and a Long Queries table at the query whose keys fill L1.
The Preparation column is the query's key sort and tree layout, paid once per query; at long queries it dominates a round.

## Short Queries

| Backend         | `sz_overlap_score` | `sz_overlap_scores` | Prefix hashes | Window hashes | Preparation | B-tree probe |
| :-------------- | -----------------: | ------------------: | ------------: | ------------: | ----------: | -----------: |
| Serial @ Xeon4  |                  … |                   … |             … |             … |           … |            … |
| Haswell @ Xeon4 |                  … |                   … |             … |             … |           … |            … |
| Skylake @ Xeon4 |                  … |                   … |             … |             … |           … |            … |

## Long Queries

| Backend         | `sz_overlap_score` | `sz_overlap_scores` | Prefix hashes | Window hashes | Preparation | B-tree probe |
| :-------------- | -----------------: | ------------------: | ------------: | ------------: | ----------: | -----------: |
| Serial @ Xeon4  |                  … |                   … |             … |             … |           … |            … |
| Haswell @ Xeon4 |                  … |                   … |             … |             … |           … |            … |
| Skylake @ Xeon4 |                  … |                   … |             … |             … |           … |            … |

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

Every width's query window hashes share one B-tree: the step primitives are told a width and know nothing about why, while `sz_overlap_score` and `sz_overlap_scores` take the whole `window_widths` list and answer one score per width.
A candidate window hash is produced at a known width, so a hit is attributed to that width, and a coincidence with another width's window hash runs at $2^{-32}$.
What to do with those scores — weight them, or pick one per pair from the two texts' lengths — stays with the caller.

The width above which spurious window matches become rare follows from the alphabet's collision entropy $H_2$, the Rényi entropy of order two, in bits per symbol:

$$w^{\ast} = \frac{\log_2 \left( n_\text{query} \cdot n_\text{candidate} \right)}{H_2}.$$

That lands near 6 bytes for English sentences, 4 for protein, and 14 for DNA — the nominal alphabet size is the wrong input, since English text has a byte collision entropy near 2.2 bits, an effective alphabet near 4.6 rather than 27.
