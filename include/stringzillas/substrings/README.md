# Substrings for StringZillas

The substrings engine matches a __whole dictionary of needles__ against a __whole collection of haystacks__ in a single pass, the workhorse of __log scanning__, __protocol dispatch__, __content filtering__, and __signature matching__.
It compiles the needle set once into an Aho-Corasick automaton and reuses it across every later call, so the dictionary is paid for once rather than per haystack.
The automaton is __goto-completed__ and split into two tiers: a dense 256-wide row per frequently-visited state, and a double array for the rest, so a step is one load with no failure-following at runtime.
`substrings_cased_k` matches raw bytes and accepts any needle, while `substrings_uncased_k` bakes __full Unicode case folding__ into the transitions themselves, matching case variants of the original haystack bytes without folding the haystack at runtime.
Every engine runs across a slice of CPU cores or a CUDA GPU, advancing many haystacks concurrently rather than making any single haystack faster.

Throughput is reported in __MB/s__ of haystack bytes consumed, the rate at which the automaton advances over the corpus.

## Methodology

Each table fixes one input shape: the `xlsum.csv` corpus split into lines as haystacks, searched with one slice of that corpus's own frequency-ordered vocabulary as the dictionary.
Slices are taken by term count, so the frequent and the rare slice of the same percentage hold the same number of needles and differ only in byte totals, since Zipf makes frequent terms short and their automata correspondingly shallower.
Cells carry the benchmark's `Throughput` line, for the counting pass that only tallies matches and the finding pass that materializes each one, with the fastest backend of each column in bold.
A `-` cell is a measurement not yet taken.
The run behind these numbers read 64 MiB of the corpus, `STRINGWARS_DATASET_LIMIT=64mb` with `STRINGWARS_TOKENS=lines`, on one H100 80GB HBM3 and one Xeon Platinum 8468.

The tier split dominates these numbers more than the backend does, so `bench/substrings.cpp` prints the automaton's state count, how many of those states fit the hot tier, and the double array's byte size beside every result.
Rows name a threading tier rather than an ISA because there are no per-ISA kernels: a transition is one data-dependent load on a serial dependency chain, so throughput comes from independent scalar chains over many haystacks, and an AVX-512 `vpgatherdd` formulation measured slower than plain scalar chains at every dictionary size.
These numbers should not be compared against `sz_find`, which searches for one needle and solves a strictly easier problem.

Uncased matching follows `CaseFolding.txt` status codes `C` and `F`, excluding `S` and `T`, which is the same contract `sz_utf8_uncased_find` implements and is what keeps results locale-independent.
No normalization is applied, so precomposed `é` does not match `e` followed by a combining acute; callers who need canonical equivalence run `sz_utf8_norm` first.
Uncased needles must be well-formed UTF-8, and needles of either mode must be non-empty.

## Most Frequent 1% of the Vocabulary

| Backend          |      Cased Count |       Cased Find |    Uncased Count |     Uncased Find |
| :--------------- | ---------------: | ---------------: | ---------------: | ---------------: |
| Serial @ Xeon4   |       166.5 MB/s |        78.8 MB/s |       156.9 MB/s |        75.0 MB/s |
| Parallel @ Xeon4 |      2330.0 MB/s |      1106.0 MB/s |      2287.1 MB/s |      1073.7 MB/s |
| CUDA @ H100      | __16170.6 MB/s__ | __17169.1 MB/s__ | __15827.0 MB/s__ | __16428.2 MB/s__ |

## Most Frequent 10% of the Vocabulary

| Backend          |      Cased Count |       Cased Find |    Uncased Count |     Uncased Find |
| :--------------- | ---------------: | ---------------: | ---------------: | ---------------: |
| Serial @ Xeon4   |        61.1 MB/s |        26.7 MB/s |        62.2 MB/s |        29.1 MB/s |
| Parallel @ Xeon4 |       920.6 MB/s |       441.5 MB/s |       921.9 MB/s |       417.8 MB/s |
| CUDA @ H100      | __12852.7 MB/s__ | __13668.7 MB/s__ | __12433.9 MB/s__ | __10694.5 MB/s__ |

## Least Frequent 1% of the Vocabulary

| Backend          |      Cased Count |       Cased Find |    Uncased Count |     Uncased Find |
| :--------------- | ---------------: | ---------------: | ---------------: | ---------------: |
| Serial @ Xeon4   |       335.2 MB/s |       174.1 MB/s |       346.2 MB/s |       174.8 MB/s |
| Parallel @ Xeon4 |      4949.9 MB/s |      2426.7 MB/s |      5186.2 MB/s |      2405.2 MB/s |
| CUDA @ H100      | __14044.5 MB/s__ | __25565.8 MB/s__ | __19638.7 MB/s__ | __25254.4 MB/s__ |

## Least Frequent 10% of the Vocabulary

| Backend          |      Cased Count |       Cased Find |    Uncased Count |     Uncased Find |
| :--------------- | ---------------: | ---------------: | ---------------: | ---------------: |
| Serial @ Xeon4   |       130.6 MB/s |        64.9 MB/s |       131.6 MB/s |        63.2 MB/s |
| Parallel @ Xeon4 |      1943.5 MB/s |       978.3 MB/s |      1975.7 MB/s |       983.1 MB/s |
| CUDA @ H100      | __12799.0 MB/s__ | __16814.8 MB/s__ | __13851.3 MB/s__ | __18071.1 MB/s__ |

## Entire Vocabulary

| Backend          |     Cased Count |      Cased Find |   Uncased Count |    Uncased Find |
| :--------------- | --------------: | --------------: | --------------: | --------------: |
| Serial @ Xeon4   |       39.9 MB/s |       17.7 MB/s |       31.4 MB/s |       13.7 MB/s |
| Parallel @ Xeon4 |      581.2 MB/s |      263.1 MB/s |      493.6 MB/s |      205.6 MB/s |
| CUDA @ H100      | __9524.1 MB/s__ | __8031.6 MB/s__ | __8332.2 MB/s__ | __5937.8 MB/s__ |
