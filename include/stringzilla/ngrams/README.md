# N-grams: Overlaps & MinHash Sketches

This directory holds the kernels behind the `ngrams` family: one engine prepares a batch of queries once, scores candidate strings against it, and sketches any texts over the same N-grams.
An overlap is the share of the query's distinct N-grams the candidate holds, each counted once, and a sketch turns a text into as many one-byte slots as the caller asks, so that the share of slots two texts agree at estimates the share of N-grams they have in common.
Each operation has a serial baseline plus `haswell` and `skylake` SIMD backends on x86, `neon` on Arm and `powervsx` on Power, and `cuda`, `rocm` and `metal` backends on the device.
The CUDA and ROCm kernels share `simt.cuh`, each vendor launches them from its own host code in `cuda.cuh` and `rocm.cuh`, which `src/capabilities/cuda.cu` and `src/capabilities/rocm.hip` compile into the library, and the Metal ones live in `metal.h` with the `metal.metal` shaders, which `src/capabilities/metal.c` compiles.
The `ampere` and `blackwell` tiers over CUDA and the `cdna3`, `cdna4` and `cdna5` tiers over ROCm compile the same walks into kernels of their own, in `ampere.cuh`, `blackwell.cuh`, `cdna3.cuh`, `cdna4.cuh` and `cdna5.cuh`, each built by the unit of the same name in `src/capabilities/`.
On `ampere` a bounded walk sums its lanes' counts with `redux.sync` rather than ten shuffles.
The init picks the best capability of a mask once, when the batch of queries is prepared on one device, and every later call runs that capability's kernel alone.
A device engine keeps no stream: on CUDA it keeps no per-round state either, so any number of streams may score it at once, while on Metal `candidates_budget` sizes one round block at init and a round past it is refused with `sz_unexpected_dimensions_k`.
An engine of no queries only sketches.

Two verbs serve callers, and two lower levels serve those driving their own loops, with the capability last in the name.
The N-gram residues of one text at every width come from one call per CPU capability, a sketch of one text from another that takes its slot state from the caller, and the device walks are public too, so a kernel of another library can roll, probe and sketch inside its own launch.

> `sz_ngrams_overlaps`, `sz_ngrams_sketches`, `sz_ngrams_residues_haswell`, `sz_ngrams_sketch_haswell`, `sz_ngrams_roll_simt`, `sz_ngrams_sketch_simt`, `sz_ngrams_sketch_metal`

## Methodology

Overlap cells are Mwin/s, millions of N-grams per second, one window per byte offset at the scored width, over words and lines from `xlsum.csv`.
Sketch cells are texts per second over lines from `xlsum.csv`, over N-grams of 5 and 8 bytes.
A `…` cell is not measured yet.

The benchmark reads the whole `xlsum.csv` by default; `STRINGWARS_BYTES` controls the input size.
That slice averages 8.55 bytes per word and 4,976.32 bytes per line, with scored window widths of 2 and 5 respectively.

## Short Words

| Backend                       | `sz_ngrams_overlaps` |
| :---------------------------- | --------------------: |
| Serial @ 1× Intel Xeon6       |                  61.9 |
| Haswell @ 1× Intel Xeon6      |                  73.1 |
| Skylake @ 1× Intel Xeon6      |                  71.5 |
| Serial @ 1× Intel Xeon5       |                  75.5 |
| Haswell @ 1× Intel Xeon5      |                  96.5 |
| Skylake @ 1× Intel Xeon5      |                 103.3 |
| Ice Lake @ 1× Intel Xeon5     |                     ↑ |
| Serial @ 1× Apple M5 Pro      |                 180.8 |
| NEON @ 1× Apple M5 Pro        |                 201.7 |
| Metal @ Apple M5 Pro          |                 1,051 |
| Serial @ 1× IBM POWER9        |                     … |
| Power VSX @ 1× IBM POWER9     |                     … |
| CUDA @ Nvidia SM90            |                     … |
| CUDA @ 148× Nvidia SM100      |                 3,064 |
| Hopper @ 148× Nvidia SM100    |                     ↑ |
| Blackwell @ 148× Nvidia SM100 |                 3,076 |
| CUDA @ 18× Nvidia SM103       |                 1,163 |
| Blackwell @ 18× Nvidia SM103  |                     … |
| CUDA @ Nvidia SM120           |                     … |
| ROCm @ AMD gfx942             |                     … |
| CDNA3 @ AMD gfx942            |                     … |
| ROCm @ AMD gfx950             |                     … |
| CDNA4 @ AMD gfx950            |                     … |
| ROCm @ AMD gfx1250            |                     … |
| CDNA5 @ AMD gfx1250           |                     … |

## Long Lines

| Backend                       | `sz_ngrams_overlaps` |
| :---------------------------- | --------------------: |
| Serial @ 1× Intel Xeon6       |                  57.8 |
| Haswell @ 1× Intel Xeon6      |                 118.2 |
| Skylake @ 1× Intel Xeon6      |                 191.7 |
| Serial @ 1× Intel Xeon5       |                  69.8 |
| Haswell @ 1× Intel Xeon5      |                 140.7 |
| Skylake @ 1× Intel Xeon5      |                 254.1 |
| Ice Lake @ 1× Intel Xeon5     |                     ↑ |
| Serial @ 1× Apple M5 Pro      |                 133.1 |
| NEON @ 1× Apple M5 Pro        |                 201.7 |
| Metal @ Apple M5 Pro          |                25,020 |
| Serial @ 1× IBM POWER9        |                     … |
| Power VSX @ 1× IBM POWER9     |                     … |
| CUDA @ Nvidia SM90            |                     … |
| CUDA @ 148× Nvidia SM100      |                22,190 |
| Hopper @ 148× Nvidia SM100    |                     ↑ |
| Blackwell @ 148× Nvidia SM100 |                22,810 |
| CUDA @ 18× Nvidia SM103       |                 8,696 |
| Blackwell @ 18× Nvidia SM103  |                10,700 |
| CUDA @ Nvidia SM120           |                     … |
| ROCm @ AMD gfx942             |                     … |
| CDNA3 @ AMD gfx942            |                     … |
| ROCm @ AMD gfx950             |                     … |
| CDNA4 @ AMD gfx950            |                     … |
| ROCm @ AMD gfx1250            |                     … |
| CDNA5 @ AMD gfx1250           |                     … |

## Sketches of Lines

| Backend                       | 128 slots | 512 slots | 2,048 slots |
| :---------------------------- | --------: | --------: | ----------: |
| Serial @ 1× Intel Xeon6       |         … |         … |           … |
| Haswell @ 1× Intel Xeon6      |         … |         … |           … |
| Skylake @ 1× Intel Xeon6      |         … |         … |           … |
| Serial @ 1× Apple M5 Pro      |         … |         … |           … |
| NEON @ 1× Apple M5 Pro        |         … |         … |           … |
| Metal @ Apple M5 Pro          |         … |         … |           … |
| Serial @ 1× IBM POWER9        |         … |         … |           … |
| Power VSX @ 1× IBM POWER9     |         … |         … |           … |
| CUDA @ Nvidia SM120           |         … |         … |           … |
| Ampere @ Nvidia SM120         |         … |         … |           … |
| Blackwell @ Nvidia SM120      |         … |         … |           … |
| ROCm @ AMD gfx942             |         … |         … |           … |
| CDNA3 @ AMD gfx942            |         … |         … |           … |
| ROCm @ AMD gfx950             |         … |         … |           … |
| CDNA4 @ AMD gfx950            |         … |         … |           … |
| ROCm @ AMD gfx1250            |         … |         … |           … |
| CDNA5 @ AMD gfx1250           |         … |         … |           … |

## Window Hashes

A window hash is the window's own big-endian value reduced by a 32-bit prime, taken as the difference of two prefix hashes:

$$H(i, w) = P(i + w) - P(i) \cdot 256^{w} \bmod p, \qquad p = 4026525731.$$

One modular multiply-add serves any width and any offset, so the window widths need not form a doubling chain, and the window hash is full-width rather than a residue with spare bits above it.
Two 32-bit residues do not multiply exactly inside a 53-bit mantissa, so every product splits into the rounded head $ab$ and the exact tail $\mathrm{fma}(a, b, -ab)$, and the two reduce together.
Serial takes the same identity through a 64-bit integer product and one remainder by a compile-time constant, needing no fused multiply-add.

The prime is not interchangeable with another of its size.
Two N-grams collide exactly when their byte difference, read as a signed base-256 expansion, is a multiple of $p$, so what matters is the cheapest such multiple over the eight digits a machine word spans:

$$\mathrm{weight}(p) = \min \sum_{i=0}^{7} |c_i| \quad \text{over } c \neq 0 \text{ with } \sum_{i=0}^{7} c_i \cdot 256^{i} \equiv 0 \pmod p.$$

The largest prime below $2^{32}$ is $2^{32} - 5 = 256^{4} - 5$, whose weight is six: two texts differing by $+5$ at one byte and $-1$ four bytes along already collide.
This prime's weight is twenty-four, which `test/ngrams.cpp` certifies by the same search that rediscovers the six.

## Window Widths

Every width's distinct query N-grams share one table: the step primitives are told a width and know nothing about why, while the engine takes the whole `window_widths` list at construction and `sz_ngrams_overlaps` answers one `f32` share per query per candidate per width.
A slot keys a window by its hash and its width's index, so a candidate window hash only ever meets the query's N-grams of its own width.
What to do with those shares — weight them, or pick one per pair from the two texts' lengths — stays with the caller.

The GPU engines take every width the host ones do.
CUDA and ROCm lanes take whole aligned words of a candidate and hash each window starting in them straight from those words: up to eight bytes are one integer reduced once, and a wider window folds four more bytes per reduction, so no lane rolls a chain or primes one.
Metal rolls every width's window hash through each lane's stretch in registers, eight widths together.

A width's count is the query's distinct N-grams the candidate holds, each counted once however often either side repeats it.
Its share is that count over the query's distinct N-grams at the width, or zero where the query holds none, and every backend writes it at the store as one correctly rounded `f32` division: both operands are integers of at most 2²⁴, exact in `f32`, and Metal divides with `precise::divide`, as its shaders compile with fast math.
A pair's distance is its misses, the query's distinct N-grams the candidate lacks summed over every width, over the query's distinct N-grams in all.
A radius on `sz_ngrams_overlaps` is an `f32` share in `[0, 1]` that gives each pair its own budget of misses, so a long pair never loosens a short one's.
Each call rounds the share up to 24 fractional bits once, and a pair's budget is that times the query's distinct N-grams, shifted down by 24, so pruning stays in integers on GPUs without FP64.
Pairs within the radius share exactly and pairs past it share zero at every width, on every backend, except that a pair less than 2⁻²⁴ past it may still share exactly.
Every backend rounds alike, so the shares match bit for bit and a caller ranking by distance gets the same answer whatever ran it.
The host sweeps probe every window of a candidate once per query, a hit marking the query window's first byte so repeats count once, and the lengths alone rule out a candidate with too few N-grams to reach the budget before a window is probed.
The GPU engines take every width in one bounded launch and give each candidate a warp, zeroing a pair past its budget once it is counted.
Metal skips the walk of a pair its lengths rule out and zeroes the rest past their budgets as it copies the shares out.

The width above which spurious window matches become rare follows from the alphabet's collision entropy $H_2$, the Rényi entropy of order two, in bits per symbol:

$$w^{\ast} = \frac{\log_2 \left( n_\text{query} \cdot n_\text{candidate} \right)}{H_2}.$$

That lands near 6 bytes for English sentences, 4 for protein, and 14 for DNA — the nominal alphabet size is the wrong input, since English text has a byte collision entropy near 2.2 bits, an effective alphabet near 4.6 rather than 27.

## Sketch Slots

A sketch takes one permutation of the N-grams rather than one per slot, and densifies the slots it leaves empty:

$$a = \text{mix}(r + w), \qquad s = \left\lfloor \frac{a \cdot n}{2^{32}} \right\rfloor, \qquad \ell_s = \min \text{mix}(a \oplus \texttt{0x9E3779B9}).$$

Every window of residue $r$ at width $w$ lands in one of the $n$ slots by the high half of a product and keeps the least rank there, so a window costs one residue, two mixes, a multiply-high and one minimum, however many slots the caller asks for.
A slot no window landed in takes the rank of the next filled slot mixed with its distance to it, so two texts still agree there exactly when they agree at that neighbor, and slot $s$ stores the low byte of $\text{mix}(\ell_s)$.
The state is one 32-bit rank a slot, which the engine keeps in scratch it grows and never shrinks; a device keeps it in shared or threadgroup memory, its lanes splitting the N-grams and taking an atomic minimum.
Nothing is a float, so every backend hashes the same integers and a sketch is the same on every capability.
A sketch of $n$ slots is not a prefix of a wider one, so a caller changing the slot count sketches its texts again.

In 32-bit integers alone, where `mix` is the bijective finalizer every backend shares and all ones marks a slot no window reached:

```text
ranks[c] = 0xFFFFFFFF for every slot c
for each window at each width:   a = mix(residue + width)
                                 slot = (a × slots) >> 32,  rank = mix(a ^ 0x9E3779B9)
                                 ranks[slot] = min(ranks[slot], rank)
for each slot c still all ones:  ranks[c] = mix(ranks[(c + step) mod slots] + step),
                                 step ≥ 1 the distance to the nearest reached slot after c
sketch[c] = low byte of mix(ranks[c]); a text with no N-grams: every byte 0x4A
```

## Estimates

Two texts whose window sets are $A$ and $B$ agree at one slot when the least window of $A \cup B$ in it lies in both, which has probability close to $J = |A \cap B| / |A \cup B|$, or by chance of one byte in 256:

$$\Pr[\text{agree}] = J + \frac{1 - J}{256}.$$

The count of unequal bytes of two sketches, a Hamming distance, therefore estimates $1 - J$ after a rescale by $256/255$.
Over $n$ slots the share of agreeing slots has a standard error of $\sqrt{p (1 - p) / n}$ for $p = \Pr[\text{agree}]$, never more than $1 / (2 \sqrt{n})$: 4.4% at 128 slots, 2.2% at 512 and 1.1% at 2,048.
Widths add N-grams of their own to both sets, so a text sketched at 5 and 8 bytes estimates the Jaccard index of the union of its 5-byte and 8-byte N-grams.

## Symbols

Every byte is a symbol, so a sketch is exact about bytes and folds nothing.
A caller who wants case folded or UTF-8 normalized text sketches the folded bytes, which the Unicode kernels produce.
A text shorter than every width has no N-grams, and every slot of its sketch holds `0x4A`, so two such texts are at distance zero.
A sketch of no slots or a stride under the slots is refused.
