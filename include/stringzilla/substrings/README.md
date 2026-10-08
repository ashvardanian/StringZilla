# Substrings: Multi-Pattern Search Over One Automaton

This directory holds the kernels behind `sz_substrings_engine_init`, `sz_substrings_counts`, `sz_substrings_find`, `sz_substrings_replace` and `sz_substrings_bm25_scores`, on the host and on the device alike.
One engine owns the compiled vocabulary, the overlap policy it was sized for, the round's arena and the report every verb writes, so a compute verb allocates nothing and joins nothing.
A vocabulary of needles compiles once into a byte-level Aho-Corasick automaton, and every haystack then streams through it in one pass whatever the needle count, since the alternative is one search per needle per haystack.
Nothing here assumes what the bytes mean: the same automaton serves nucleotides, binary records and multilingual UTF-8, and adapts to each through the classes its own vocabulary spells.

## The Automaton

The automaton is two-tiered, split by how often a state is visited rather than by anything about the vocabulary.
Text keeps resetting the walk toward the root, so a small set of states absorbs most byte steps however large the dictionary grows, and those states get a dense goto-completed row: one load, no branch, no failure chase.
A row holds one target per byte __class__ rather than per byte.
A class is a byte some needle spells, or the one shared class of every byte none does, because those bytes behave identically from every state.
A row is therefore as wide as the vocabulary's own alphabet: five targets for nucleotides, a few hundred for multilingual text, never more than 256.
The class depends on the byte alone, so its lookup sits off the chain of transitions.
The hot tier is sized in bytes rather than states, so a narrow alphabet keeps far more of the automaton hot: a thousand nucleotide needles fit whole.
Everything else lives in a double array, where `base[state] + byte` addresses the target and `check` confirms ownership, so a collision reads as a missing edge and the walk hops to `fail` and retries.

## The Walks

A transition is a data-dependent load, so one chain leaves the load ports idle for its whole latency.
Byte-exact walks step eight disjoint slices at once, each primed by the bytes before it.
A walk that must report in haystack order does so in rounds of windows, buffering each window's accepting positions and flushing them in window order.
A leftmost cover is settled during the walk, through a ring of undecided starts whose claimed slots a bitmap tracks, so draining jumps between claims rather than visiting every byte.

Case folding lives in the stream rather than in the automaton.
A needle is folded once at insertion, and a case-insensitive haystack is folded as the walk consumes it, one codepoint at a time, so both sides meet in one canonical byte stream and the trie stays a tree.
A folded length brackets rather than fixes a source span: needle `k` matches both the one-byte `k` and the three-byte Kelvin sign, so a match's span is recovered per match.

## SIMD Tiers

The transition has no vector form that beats eight scalar chains: stepping the chains with AVX2 or AVX-512 gathers measured at 0.3× to 0.5× their scalar throughput.
The Haswell, Ice Lake and NEON tiers replace a text-side stage instead.
While a walk stands on the root, the tier's byte-set search jumps to the next byte that leaves the root, drawn from the automaton's own set of live root bytes.
Whether that pays depends on the text as much as on the vocabulary, so each haystack samples it: the skip runs once fewer than one byte in eight leaves the root.
Over nucleotides every byte is live and the skip switches itself off, while a sparse vocabulary over large alphabets runs about four times faster.

## The Device

The device backends are `cuda` and `rocm`, whose kernels share `simt.cuh`, each launched from its own host code in `cuda.cuh` and `rocm.cuh`, which `c/target/cuda.cu` and `c/target/rocm.hip` compile into the library, and `metal`, whose kernels live in `metal.h` with the `metal.metal` shaders, which `c/target/metal.c` compiles.
All three parallelize over haystack __chunks__ rather than over haystacks, so a corpus of one long document and a corpus of a million short ones fill the device the same way.
A chunk reports every match ending inside it and primes itself from the bytes before its own start, clamped to its own haystack, so every match is found exactly once.
A leftmost cover is settled after the walk instead, since inside it would cost every thread a ring wide enough for the longest match.
On CUDA and ROCm each block stages the class map and as much of the hot tier as fits beside its own scratch into shared memory.
BM25 gives each block one haystack and a shared-memory tally sized by the vocabulary, hashing a wider one and spilling past it, and sums in fixed point so thread order cannot move a score.
A batch of fewer haystacks than the device holds clusters of blocks gives each haystack a whole cluster instead, every block counting into the first one's tally through distributed shared memory.
A device engine's arena is sized once, by `sz_substrings_engine_init` from `matches_budget` and `haystacks_budget`, so a round carrying more haystacks is refused with `sz_unexpected_dimensions_k` and matches past the budget surface as the report's `shortfall`.
Every round of one engine shares that arena and its report, so the caller orders them, on one stream or with events between two, and reads the report only after joining the round that wrote it.
A device round over no haystacks returns at once and leaves the report as the round before it wrote it.
On Metal a round is one command buffer on the device's queue, nothing is staged in threadgroup memory, and BM25 runs on 64 threadgroups whose tallies hold 2048 needles before hashing, its terms computed in `f32`, as Apple GPUs have no `f64`.
Case-insensitive Metal walks reuse the folding tables shared with CUDA and ROCm and report matches in the haystack's original byte offsets.

## Methodology

Cells are haystack MB/s over the first 64 MB of `xlsum.csv` text and of random `ACGT` in 4,096-byte lines, set with `STRINGWARS_BYTES=64MB`.
Vocabularies are the Frequent and Rare one percent of the corpus's words, and 1,000 Sampled substrings of 4 to 16 bytes.
A `…` cell is not measured yet.

## Overlapping Matches

Every match of every needle, including nested ones, over text.

| Backend                   | Count, Frequent | Count, Rare | Find, Frequent | Find, Rare |
| :------------------------ | --------------: | ----------: | -------------: | ---------: |
| Serial @ 1× Intel Xeon6   |           384.8 |       746.4 |          288.5 |      591.4 |
| Haswell @ 1× Intel Xeon6  |           417.9 |     3,506.2 |          319.7 |    3,191.8 |
| Ice Lake @ 1× Intel Xeon6 |           411.7 |     3,757.1 |          309.0 |    3,281.9 |
| NEON @ 1× AWS Graviton4   |               … |           … |              … |          … |
| CUDA @ Nvidia SM90        |               … |           … |              … |          … |
| CUDA @ 18× Nvidia SM103   |        17,674.2 |    33,454.1 |        9,125.9 |   19,025.9 |
| CUDA @ Nvidia SM120       |        22,077.4 |    26,101.8 |       13,240.3 |   25,569.3 |
| Serial @ 1× Apple M5 Pro  |           905.2 |     1,421.4 |          695.9 |    1,152.0 |
| NEON @ 1× Apple M5 Pro    |           894.2 |     7,010.1 |          692.9 |    6,233.3 |
| Metal @ Apple M5 Pro      |        18,086.8 |    33,734.4 |        8,220.8 |   17,799.2 |

## Leftmost Cover

Matches sharing no bytes, under the leftmost-longest policy, over text.

| Backend                   | Count, Frequent | Count, Rare | Find, Frequent | Find, Rare |
| :------------------------ | --------------: | ----------: | -------------: | ---------: |
| Serial @ 1× Intel Xeon6   |           230.7 |       505.3 |          225.3 |      500.0 |
| Haswell @ 1× Intel Xeon6  |           239.3 |     2,740.2 |          236.9 |    2,736.1 |
| Ice Lake @ 1× Intel Xeon6 |           241.5 |     2,890.8 |          236.3 |    2,891.8 |
| NEON @ 1× AWS Graviton4   |               … |           … |              … |          … |
| CUDA @ Nvidia SM90        |               … |           … |              … |          … |
| CUDA @ 18× Nvidia SM103   |         8,457.2 |    18,544.6 |        7,997.4 |   18,227.2 |
| CUDA @ Nvidia SM120       |         7,936.0 |    16,332.8 |        8,151.0 |   23,808.0 |
| Serial @ 1× Apple M5 Pro  |           524.5 |     1,284.9 |          519.8 |    1,295.2 |
| NEON @ 1× Apple M5 Pro    |           520.9 |     5,498.2 |          516.7 |    5,461.8 |
| Metal @ Apple M5 Pro      |         7,761.8 |    17,952.5 |        7,525.6 |   17,737.4 |

## Rewriting

One replacement per needle, substituted over the leftmost-longest cover; an overlapping policy admits no rewrite.

| Backend                   | Replace, Frequent | Replace, Rare |
| :------------------------ | ----------------: | ------------: |
| Serial @ 1× Intel Xeon6   |             200.4 |         481.8 |
| Haswell @ 1× Intel Xeon6  |             205.3 |       2,263.0 |
| Ice Lake @ 1× Intel Xeon6 |             204.2 |       2,338.8 |
| NEON @ 1× AWS Graviton4   |                 … |             … |
| CUDA @ Nvidia SM90        |                 … |             … |
| CUDA @ 18× Nvidia SM103   |           5,529.6 |       9,059.3 |
| CUDA @ Nvidia SM120       |           7,024.6 |      15,923.2 |
| Serial @ 1× Apple M5 Pro  |             461.5 |       1,244.7 |
| NEON @ 1× Apple M5 Pro    |             444.3 |       4,818.3 |
| Metal @ Apple M5 Pro      |           5,386.3 |      12,808.1 |

## Scoring

BM25 with the vocabulary as the query, one score per haystack, over raw overlapping term frequencies.
A CPU sums each haystack's terms in ascending needle order, while device backends accumulate in fixed point.
Metal evaluates each term in `f32` before conversion to fixed point.

| Backend                   | BM25, Frequent | BM25, Rare |
| :------------------------ | -------------: | ---------: |
| Serial @ 1× Intel Xeon6   |          269.3 |      594.6 |
| Haswell @ 1× Intel Xeon6  |          303.1 |    2,808.8 |
| Ice Lake @ 1× Intel Xeon6 |          264.6 |    2,751.5 |
| NEON @ 1× AWS Graviton4   |              … |          … |
| CUDA @ Nvidia SM90        |              … |          … |
| CUDA @ 18× Nvidia SM103   |        3,530.8 |   12,748.8 |
| CUDA @ Nvidia SM120       |        8,140.8 |   31,037.4 |
| Serial @ 1× Apple M5 Pro  |          640.3 |    1,104.2 |
| NEON @ 1× Apple M5 Pro    |          637.2 |    4,989.1 |
| Metal @ Apple M5 Pro      |        7,404.1 |   18,186.1 |

## Nucleotides

The sampled slice over uniform `ACGT`, where rows are five classes wide and the whole automaton is hot.
Short needles over four letters match about 0.4 times per byte, so every column past the count is bound by reporting rather than by the walk.

| Backend                   |    Count |    Find | Count, Leftmost |    BM25 |
| :------------------------ | -------: | ------: | --------------: | ------: |
| Serial @ 1× Intel Xeon6   |    767.9 |   108.4 |            50.7 |    97.2 |
| Haswell @ 1× Intel Xeon6  |    801.2 |   109.3 |            50.8 |    97.1 |
| Ice Lake @ 1× Intel Xeon6 |    804.0 |   109.0 |            51.6 |   100.8 |
| NEON @ 1× AWS Graviton4   |        … |       … |               … |       … |
| CUDA @ Nvidia SM90        |        … |       … |               … |       … |
| CUDA @ 18× Nvidia SM103   | 23,644.2 | 5,979.1 |         1,443.8 | 4,365.3 |
| CUDA @ Nvidia SM120       | 22,732.8 | 1,566.7 |           908.2 | 7,833.6 |

## Case Folding

The frequent slice with both sides folded, which is the cost of matching a vocabulary against text that does not share its case.

| Backend                   |   Count |    Find | Replace |
| :------------------------ | ------: | ------: | ------: |
| Serial @ 1× Intel Xeon6   |   138.6 |   144.3 |   120.1 |
| Haswell @ 1× Intel Xeon6  |   143.2 |   139.4 |   119.2 |
| Ice Lake @ 1× Intel Xeon6 |   143.0 |   148.3 |   117.4 |
| Serial @ 1× Apple M5 Pro  |   314.7 |   317.3 |   264.4 |
| NEON @ 1× Apple M5 Pro    |   314.8 |   314.0 |   263.7 |
| Metal @ Apple M5 Pro      | 5,558.3 | 2,812.8 | 2,259.4 |
| NEON @ 1× AWS Graviton4   |       … |       … |       … |
| CUDA @ Nvidia SM90        |       … |       … |       … |
| CUDA @ 18× Nvidia SM103   | 5,289.0 | 2,628.6 | 2,120.7 |
| CUDA @ Nvidia SM120       | 9,523.2 | 4,587.5 | 3,215.4 |

## Compilation

Building the engine is its own cost, paid once per vocabulary, and it is reported separately because a pipeline that rebuilds per query is bound by this rather than by the walk.
Cells are needle bytes per second in MB/s, over 3,462 needles per word slice and 1,000 per sampled slice.

| Backend                   | Cased, Frequent | Cased, Rare | Folded, Frequent | Sampled, Text | Sampled, Nucleotides |
| :------------------------ | --------------: | ----------: | ---------------: | ------------: | -------------------: |
| Serial @ 1× Intel Xeon6   |           11.99 |       20.31 |            11.93 |         11.04 |                25.89 |
| Serial @ 1× AWS Graviton4 |               … |           … |                … |             … |                    … |
