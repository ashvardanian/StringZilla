# Substrings for StringZillas

The substrings engine matches a __whole dictionary of needles__ against a __whole collection of haystacks__ in a single pass, the workhorse of __log scanning__, __protocol dispatch__, __content filtering__, and __signature matching__.
It compiles the needle set once into an Aho-Corasick automaton and reuses it across every later call, so the dictionary is paid for once rather than per haystack.
The automaton is __goto-completed__ and split into two tiers: a dense 256-wide row per frequently-visited state, and a double array for the rest, so a step is one load with no failure-following at runtime.
`substrings_cased_k` matches raw bytes and accepts any needle, while `substrings_uncased_k` applies __full Unicode case folding__ to both sides of the comparison: the needle is folded once at build time, and the haystack one codepoint at a time as the walk consumes it, never into a buffer.
A match is therefore any contiguous run of the folded haystack, so it may begin or end part-way through an expansion - `"s"` matches inside `"ß"` - with both ends reported in original haystack bytes, snapped outward to the codepoint they fall in.
That is the same rule `sz_utf8_uncased_search` follows, so the single-pattern and multi-pattern engines agree.
Every engine runs across a slice of CPU cores or a CUDA GPU, advancing many haystacks concurrently rather than making any single haystack faster.

Throughput is reported in __MB/s__ of haystack bytes consumed, the rate at which the automaton advances over the corpus.

## Methodology

Each table fixes one input shape: the `xlsum.csv` corpus split into lines as haystacks, searched with one slice of that corpus's own frequency-ordered vocabulary as the dictionary.
Slices are taken by term count, so the frequent and the rare slice of the same percentage hold the same number of needles and differ only in byte totals, since Zipf makes frequent terms short and their automata correspondingly shallower.
Cells carry the benchmark's `Throughput` line for each of the four capabilities — counting, which only tallies matches; finding, which materializes each one; replacing, which rewrites every haystack; and scoring, which reduces each haystack to one BM25 float — with the fastest backend of each column in bold.
A `-` cell is a measurement not yet taken.
The replace columns resolve a __leftmost__ cover, since a rewrite must, while counting and finding are __overlapping__ passes; the two are not the same walk and should not be differenced.
The score columns weight every needle uniformly, so each slice heading doubles as a query size — scoring carries a cost proportional to the dictionary that counting does not, since a document is walked once per byte but scored once per needle.
One run produced every cell: 64 MiB of the corpus, `STRINGWARS_DATASET_LIMIT=64mb` with `STRINGWARS_TOKENS=lines`, on one H100 80GB HBM3 and one Xeon Platinum 8468, built through the `cuda_clang` preset.
That preset is not a preference — GCC miscompiles the engines' return-by-value in these translation units, handing the caller the kernel timing where the status belongs, so a GCC-hosted benchmark aborts on its first CUDA cell.

The tier split dominates these numbers more than the backend does, so `bench/substrings.cpp` prints the automaton's state count, how many of those states fit the hot tier, and the double array's byte size beside every result.
Rows name a threading tier rather than an ISA because there are no per-ISA kernels: a transition is one data-dependent load on a serial dependency chain, so throughput comes from independent scalar chains over many haystacks, and an AVX-512 `vpgatherdd` formulation measured slower than plain scalar chains at every dictionary size.
These numbers should not be compared against `sz_find`, which searches for one needle and solves a strictly easier problem.

Uncased matching follows `CaseFolding.txt` status codes `C` and `F`, excluding `S` and `T`, which is the same contract `sz_utf8_uncased_find` implements and is what keeps results locale-independent.
No normalization is applied, so precomposed `é` does not match `e` followed by a combining acute; callers who need canonical equivalence run `sz_utf8_norm` first.
Uncased needles must be well-formed UTF-8, and needles of either mode must be non-empty.

## Most Frequent 1% of the Vocabulary

| Backend          |      Cased Count |       Cased Find |    Cased Replace |      Cased Score |   Uncased Count |    Uncased Find | Uncased Replace |   Uncased Score |
| :--------------- | ---------------: | ---------------: | ---------------: | ---------------: | --------------: | --------------: | --------------: | --------------: |
| Serial @ Xeon4   |       186.4 MB/s |        89.4 MB/s |        58.8 MB/s |       168.7 MB/s |       86.6 MB/s |       42.9 MB/s |       34.2 MB/s |       85.3 MB/s |
| Parallel @ Xeon4 |      2555.5 MB/s |      1224.1 MB/s |       524.2 MB/s |      2265.6 MB/s |     1031.6 MB/s |      502.9 MB/s |      374.9 MB/s |      995.6 MB/s |
| CUDA @ H100      | __18371.7 MB/s__ | __21732.5 MB/s__ | __12401.7 MB/s__ | __23837.1 MB/s__ | __8568.5 MB/s__ | __4992.9 MB/s__ | __3962.1 MB/s__ | __5476.1 MB/s__ |

## Most Frequent 10% of the Vocabulary

| Backend          |      Cased Count |      Cased Find |   Cased Replace |     Cased Score |   Uncased Count |    Uncased Find | Uncased Replace |   Uncased Score |
| :--------------- | ---------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| Serial @ Xeon4   |        66.2 MB/s |       31.1 MB/s |       21.8 MB/s |       52.1 MB/s |       43.1 MB/s |       20.5 MB/s |       16.6 MB/s |       37.0 MB/s |
| Parallel @ Xeon4 |       958.0 MB/s |      447.4 MB/s |      245.3 MB/s |      744.3 MB/s |      612.8 MB/s |      285.2 MB/s |      221.1 MB/s |      513.2 MB/s |
| CUDA @ H100      | __12809.7 MB/s__ | __8826.2 MB/s__ | __5647.9 MB/s__ | __6141.8 MB/s__ | __6646.5 MB/s__ | __3833.3 MB/s__ | __2834.7 MB/s__ | __2555.5 MB/s__ |

## Least Frequent 1% of the Vocabulary

| Backend          |      Cased Count |       Cased Find |    Cased Replace |      Cased Score |   Uncased Count |    Uncased Find | Uncased Replace |   Uncased Score |
| :--------------- | ---------------: | ---------------: | ---------------: | ---------------: | --------------: | --------------: | --------------: | --------------: |
| Serial @ Xeon4   |       411.3 MB/s |       206.1 MB/s |       173.1 MB/s |       410.0 MB/s |      106.3 MB/s |       53.0 MB/s |       50.0 MB/s |      105.9 MB/s |
| Parallel @ Xeon4 |      6399.5 MB/s |      2802.5 MB/s |      2018.6 MB/s |      4584.9 MB/s |     1116.7 MB/s |      561.1 MB/s |      524.0 MB/s |     1127.4 MB/s |
| CUDA @ H100      | __25555.1 MB/s__ | __29442.0 MB/s__ | __15096.8 MB/s__ | __36335.4 MB/s__ | __8922.8 MB/s__ | __5562.0 MB/s__ | __4649.3 MB/s__ | __6249.2 MB/s__ |

## Least Frequent 10% of the Vocabulary

| Backend          |      Cased Count |       Cased Find |   Cased Replace |     Cased Score |   Uncased Count |    Uncased Find | Uncased Replace |   Uncased Score |
| :--------------- | ---------------: | ---------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| Serial @ Xeon4   |       142.4 MB/s |        68.2 MB/s |       53.5 MB/s |      132.0 MB/s |       74.7 MB/s |       36.7 MB/s |       32.3 MB/s |       71.5 MB/s |
| Parallel @ Xeon4 |      2040.1 MB/s |       972.8 MB/s |      627.9 MB/s |     1868.3 MB/s |      955.2 MB/s |      476.0 MB/s |      409.2 MB/s |      942.6 MB/s |
| CUDA @ H100      | __16503.4 MB/s__ | __11081.0 MB/s__ | __7258.5 MB/s__ | __8160.4 MB/s__ | __7795.4 MB/s__ | __4434.6 MB/s__ | __3575.6 MB/s__ | __3865.5 MB/s__ |

## Entire Vocabulary

| Backend          |     Cased Count |      Cased Find |   Cased Replace |    Cased Score |   Uncased Count |    Uncased Find | Uncased Replace |  Uncased Score |
| :--------------- | --------------: | --------------: | --------------: | -------------: | --------------: | --------------: | --------------: | -------------: |
| Serial @ Xeon4   |       40.0 MB/s |       18.6 MB/s |       11.8 MB/s |      24.8 MB/s |       25.9 MB/s |       12.2 MB/s |        8.8 MB/s |      15.0 MB/s |
| Parallel @ Xeon4 |      596.2 MB/s |      271.9 MB/s |      155.5 MB/s |     340.1 MB/s |      394.2 MB/s |      179.2 MB/s |      132.0 MB/s |     207.9 MB/s |
| CUDA @ H100      | __8289.3 MB/s__ | __4703.0 MB/s__ | __2577.0 MB/s__ | __944.9 MB/s__ | __5046.6 MB/s__ | __2587.7 MB/s__ | __1385.1 MB/s__ | __703.6 MB/s__ |
