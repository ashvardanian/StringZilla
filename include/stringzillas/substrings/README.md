# Substrings for StringZillas

The substrings engine matches a __whole dictionary of needles__ against a __whole collection of haystacks__ in a single pass, the workhorse of __log scanning__, __protocol dispatch__, __content filtering__, and __signature matching__.
It compiles the needle set once into an Aho-Corasick automaton and reuses it across every later call, so the dictionary is paid for once rather than per haystack.
The automaton is __goto-completed__ and split into two tiers: a dense 256-wide row per frequently-visited state, and a double array for the rest, so a step is one load with no failure-following at runtime.
`substrings_cased_k` matches raw bytes and accepts any needle, while `substrings_uncased_k` bakes __full Unicode case folding__ into the transitions themselves, matching case variants of the original haystack bytes without folding the haystack at runtime.
Every engine runs across a slice of CPU cores or a CUDA GPU, advancing many haystacks concurrently rather than making any single haystack faster.

Throughput is reported in __GB/s__ of haystack bytes consumed, the rate at which the automaton advances over the corpus.

## Methodology

Each table fixes one input shape: the `leipzig1M.txt` corpus split into lines as haystacks, searched with one slice of that corpus's own frequency-ordered vocabulary as the dictionary.
Slices are taken by term count, so the frequent and the rare slice of the same percentage hold the same number of needles and differ only in byte totals, since Zipf makes frequent terms short and their automata correspondingly shallower.
Cells carry __GB/s__ parsed from the benchmark's `Throughput` line, for the counting pass that only tallies matches and the finding pass that materializes each one.
A `-` cell is a measurement not yet taken.

The tier split dominates these numbers more than the backend does, so `bench/substrings.cpp` prints the automaton's state count, how many of those states fit the hot tier, and the double array's byte size beside every result.
Rows name a threading tier rather than an ISA because there are no per-ISA kernels: a transition is one data-dependent load on a serial dependency chain, so throughput comes from independent scalar chains over many haystacks, and an AVX-512 `vpgatherdd` formulation measured slower than plain scalar chains at every dictionary size.
These numbers should not be compared against `sz_find`, which searches for one needle and solves a strictly easier problem.

Uncased matching follows `CaseFolding.txt` status codes `C` and `F`, excluding `S` and `T`, which is the same contract `sz_utf8_uncased_find` implements and is what keeps results locale-independent.
No normalization is applied, so precomposed `é` does not match `e` followed by a combining acute; callers who need canonical equivalence run `sz_utf8_norm` first.
Uncased needles must be well-formed UTF-8, and needles of either mode must be non-empty.

## Most Frequent 1% of the Vocabulary

| Backend          | Cased Count | Cased Find | Uncased Count | Uncased Find |
| :--------------- | ----------: | ---------: | ------------: | -----------: |
| Serial @ Xeon4   |           - |          - |             - |            - |
| Parallel @ Xeon4 |           - |          - |             - |            - |
| CUDA @ H100      |           - |          - |             - |            - |

> Not yet measured.

## Most Frequent 10% of the Vocabulary

| Backend          | Cased Count | Cased Find | Uncased Count | Uncased Find |
| :--------------- | ----------: | ---------: | ------------: | -----------: |
| Serial @ Xeon4   |           - |          - |             - |            - |
| Parallel @ Xeon4 |           - |          - |             - |            - |
| CUDA @ H100      |           - |          - |             - |            - |

> Not yet measured.

## Least Frequent 1% of the Vocabulary

| Backend          | Cased Count | Cased Find | Uncased Count | Uncased Find |
| :--------------- | ----------: | ---------: | ------------: | -----------: |
| Serial @ Xeon4   |           - |          - |             - |            - |
| Parallel @ Xeon4 |           - |          - |             - |            - |
| CUDA @ H100      |           - |          - |             - |            - |

> Not yet measured.

## Least Frequent 10% of the Vocabulary

| Backend          | Cased Count | Cased Find | Uncased Count | Uncased Find |
| :--------------- | ----------: | ---------: | ------------: | -----------: |
| Serial @ Xeon4   |           - |          - |             - |            - |
| Parallel @ Xeon4 |           - |          - |             - |            - |
| CUDA @ H100      |           - |          - |             - |            - |

> Not yet measured.

## Entire Vocabulary

| Backend          | Cased Count | Cased Find | Uncased Count | Uncased Find |
| :--------------- | ----------: | ---------: | ------------: | -----------: |
| Serial @ Xeon4   |           - |          - |             - |            - |
| Parallel @ Xeon4 |           - |          - |             - |            - |
| CUDA @ H100      |           - |          - |             - |            - |

> Not yet measured.
