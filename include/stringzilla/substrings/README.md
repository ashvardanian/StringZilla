# Substrings: Multi-Pattern Search Over One Automaton

This directory holds the kernels behind `sz_substrings_build`, `sz_substrings_counts`, `sz_substrings_find`, `sz_substrings_replace` and `sz_substrings_bm25_scores`.
A vocabulary of needles compiles once into a byte-level Aho-Corasick automaton, and every haystack then streams through it in a single pass, whatever the needle count — which is the whole reason the family exists, since the alternative is one search per needle per haystack.
There is a serial baseline and a CUDA backend, and no per-ISA SIMD tier yet, which makes this the one family in the library without one.
A transition is a data-dependent load rather than arithmetic over a vector, and the serial walk already runs eight independent chains to hide that load's latency; stepping the same chains with AVX2 or AVX-512 gathers measured at about half the scalar throughput.
The folded walk has a nearer opportunity still, since it reaches the serial rune iterator while `utf8_uncased_fold` already ships a vectorized fold on every ISA.

The automaton is two-tiered, split by how often a state is visited rather than by anything about the vocabulary.
Text keeps resetting the walk toward the root, so a small set of states absorbs most byte steps however large the dictionary grows, and those states get a dense goto-completed row of 256 targets each — one load, no branch, no failure chase.
Everything else lives in a double array, where `base[state] + byte` addresses the target and `check` confirms ownership, so a collision reads as a missing edge rather than a wrong one and the walk hops to `fail` and retries.
States are numbered so the hot ones come first, which makes the tier test `state < hot_count` with no lookup of its own.

Case folding lives in the stream rather than in the automaton.
A needle is folded once at insertion and inserted byte for byte, and an uncased haystack is folded as the walk consumes it, so both sides meet in one canonical byte stream.
That is what keeps the trie a tree with single-valued failure links, and what keeps every CJK, Arabic and emoji sequence at byte-exact speed — nothing under those lead bytes folds, so their own bytes already are their folded image.
The price is that a folded length brackets rather than fixes a source span: needle `k` matches both the one-byte `k` and the three-byte Kelvin sign, so a match's span is carried per match and recovered by a backward walk when a fold has broken a byte boundary.

The three overlap policies are one decision taken in one place.
`sz_substrings_overlapping_k` reports every match of every needle, including nested ones; the two leftmost policies report a cover that shares no bytes, breaking ties by span or by needle index.
On the CPU a cover is settled during the walk, through a ring of undecided starts the walk drains as it passes them.
On the device it is settled after the walk instead, because inside the walk it would cost every thread a ring wide enough for the longest match and a second walk to find a safe place to start.

The CUDA backend parallelizes over haystack __chunks__ rather than over haystacks, so a corpus of one long document and a corpus of a million short ones fill the device the same way.
A chunk reports every match ending inside it and primes itself from the `max_source_match_bytes - 1` bytes before its own start, clamped to its own haystack, so every match is found exactly once and no chunk reads a neighbour's text.
The head of the hot tier is staged into shared memory once per block, which is the mitigation for the one cost the algorithm cannot shed: a single cold lane makes its whole warp pay that lane's failure-chase depth.

## Methodology

Numbers are haystack throughput in MB/s, measured with `bench/substrings.cpp` over the `xlsum.csv` corpus tokenized by lines, reporting the median of repeated runs.
Every backend registers its own row in a single binary, and each column is one operation, so coverage and cross-chip comparison read down a single column.
There is no Standard row, since no standard library ships a multi-pattern search.
Needles are corpus words with the most frequent one percent and every word occurring once removed, and results are split by which end of that ranking a dictionary is drawn from — a Frequent column that stresses match reporting and a Rare column that measures the transition alone.
The GPU rows come from `bench/substrings.cu` and walk at least one residency wave of haystack chunks.
A `…` cell is genuinely-missing data, on a backend not yet measured on hardware that runs it.

## Overlapping Matches

Every match of every needle, including nested ones, which is the shape a count answers without materializing anything.

| Backend                  | Count, Frequent | Count, Rare | Find, Frequent | Find, Rare |
| :----------------------- | --------------: | ----------: | -------------: | ---------: |
| Serial @ Xeon 6776P      |           198.6 |       757.7 |          182.0 |      766.0 |
| Serial @ Graviton4       |               … |           … |              … |          … |
| CUDA @ SM90              |               … |           … |              … |          … |
| CUDA @ SM120             |        18,200.0 |    24,350.0 |       11,310.0 |   21,260.0 |

## Leftmost Cover

Matches sharing no bytes, resolved during the walk on a CPU and after it on a device.

| Backend                  | Count, Frequent | Count, Rare | Find, Frequent | Find, Rare |
| :----------------------- | --------------: | ----------: | -------------: | ---------: |
| Serial @ Xeon 6776P      |           138.2 |       382.3 |          138.6 |      389.5 |
| Serial @ Graviton4       |               … |           … |              … |          … |
| CUDA @ SM90              |               … |           … |              … |          … |
| CUDA @ SM120             |         6,900.0 |    12,670.0 |        7,000.0 |   17,070.0 |

## Rewriting

One replacement per needle, substituted over the leftmost cover; an overlapping policy admits no rewrite.

| Backend                  | Replace, Frequent | Replace, Rare |
| :----------------------- | ----------------: | ------------: |
| Serial @ Xeon 6776P      |             125.8 |         355.3 |
| Serial @ Graviton4       |                 … |             … |
| CUDA @ SM90              |                 … |             … |
| CUDA @ SM120             |           6,180.0 |      14,230.0 |

## Scoring

BM25 with the vocabulary as the query, one score per haystack, over raw overlapping term frequencies.
A CPU sums each haystack's terms in ascending needle order and a device in fixed point, so each backend is bit-stable across runs and the two agree to rounding.
The device scores one haystack per block into a shared-memory tally, which a vocabulary wider than the tally hashes into and spills past.

| Backend                  | BM25, Frequent | BM25, Rare |
| :----------------------- | -------------: | ---------: |
| Serial @ Xeon 6776P      |              … |          … |
| Serial @ Graviton4       |              … |          … |
| CUDA @ SM90              |              … |          … |
| CUDA @ SM120             |              … |          … |

## Case Folding

The frequent slice again, with both sides folded, which is the cost of matching a vocabulary against text that does not share its case.

| Backend                  | Count | Find | Replace |
| :----------------------- | ----: | ---: | ------: |
| Serial @ Xeon 6776P      | 111.9 | 104.5 |    93.3 |
| Serial @ Graviton4       |     … |    … |       … |
| CUDA @ SM90              |     … |    … |       … |
| CUDA @ SM120             | 9,000.0 | 3,620.0 | 3,270.0 |

## Compilation

Building the automaton is its own cost, paid once per vocabulary, and it is reported separately because a pipeline that recompiles per query is bound by this rather than by the walk.
Cells are needle bytes per second, over a vocabulary of 5,631 needles.

| Backend                  | Cased, Frequent | Cased, Rare | Uncased, Frequent |
| :----------------------- | --------------: | ----------: | ----------------: |
| Serial @ Xeon 6776P      |            9.76 |       16.01 |             10.54 |
| Serial @ Graviton4       |               … |           … |                 … |
