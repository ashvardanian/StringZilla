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
The score columns weight every needle uniformly, so each slice heading doubles as a query size — scoring counts into a table sized by the document rather than by the dictionary, so its cost tracks the walk, and the widest slice no longer collapses the way a counter per needle made it.
One run produced every cell: 64 MiB of the corpus, `STRINGWARS_DATASET_LIMIT=64mb` with `STRINGWARS_TOKENS=lines`, on one idle H100 80GB HBM3 and one Xeon Platinum 8468, built through the `cuda_clang` preset.
The box is shared, so the run waits for it to go quiet; the counting columns are the control, and they reproduce the previous table's within 1.6x end to end, which is what makes the score columns comparable against it.
That preset is not a preference — GCC miscompiles the engines' return-by-value in these translation units, handing the caller the kernel timing where the status belongs, so a GCC-hosted benchmark aborts on its first CUDA cell.

The tier split dominates these numbers more than the backend does, so `bench/substrings.cpp` prints the automaton's state count, how many of those states fit the hot tier, and the double array's byte size beside every result.
Rows name a threading tier rather than an ISA because there are no per-ISA kernels: a transition is one data-dependent load on a serial dependency chain, so throughput comes from independent scalar chains over many haystacks, and an AVX-512 `vpgatherdd` formulation measured slower than plain scalar chains at every dictionary size.
These numbers should not be compared against `sz_find`, which searches for one needle and solves a strictly easier problem.

Uncased matching follows `CaseFolding.txt` status codes `C` and `F`, excluding `S` and `T`, which is the same contract `sz_utf8_uncased_find` implements and is what keeps results locale-independent.
No normalization is applied, so precomposed `é` does not match `e` followed by a combining acute; callers who need canonical equivalence run `sz_utf8_norm` first.
Uncased needles must be well-formed UTF-8, and needles of either mode must be non-empty.

## Most Frequent 1% of the Vocabulary

| Backend          |      Cased Count |       Cased Find |   Cased Replace |      Cased Score |   Uncased Count |    Uncased Find | Uncased Replace |   Uncased Score |
| :--------------- | ---------------: | ---------------: | --------------: | ---------------: | --------------: | --------------: | --------------: | --------------: |
| Serial @ Xeon4   |       210.2 MB/s |       100.1 MB/s |       55.1 MB/s |       184.0 MB/s |       90.2 MB/s |       44.7 MB/s |       36.2 MB/s |       91.9 MB/s |
| Parallel @ Xeon4 |      2974.3 MB/s |      1256.3 MB/s |      467.8 MB/s |      2448.1 MB/s |     1065.9 MB/s |      512.9 MB/s |      390.7 MB/s |     1038.7 MB/s |
| CUDA @ H100      | __19198.5 MB/s__ | __17995.9 MB/s__ | __8504.0 MB/s__ | __15461.9 MB/s__ | __7688.0 MB/s__ | __4949.9 MB/s__ | __4359.4 MB/s__ | __4273.5 MB/s__ |

## Most Frequent 10% of the Vocabulary

| Backend          |     Cased Count |      Cased Find |   Cased Replace |     Cased Score |   Uncased Count |    Uncased Find | Uncased Replace |   Uncased Score |
| :--------------- | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| Serial @ Xeon4   |       63.1 MB/s |       29.3 MB/s |       20.7 MB/s |       55.8 MB/s |       45.3 MB/s |       20.6 MB/s |       17.1 MB/s |       41.9 MB/s |
| Parallel @ Xeon4 |      901.4 MB/s |      423.4 MB/s |      248.4 MB/s |      796.7 MB/s |      610.5 MB/s |      292.2 MB/s |      226.9 MB/s |      568.4 MB/s |
| CUDA @ H100      | __9051.6 MB/s__ | __8654.4 MB/s__ | __5798.2 MB/s__ | __7913.5 MB/s__ | __7108.2 MB/s__ | __3908.4 MB/s__ | __2856.2 MB/s__ | __2652.1 MB/s__ |

## Least Frequent 1% of the Vocabulary

| Backend          |      Cased Count |       Cased Find |    Cased Replace |      Cased Score |   Uncased Count |    Uncased Find | Uncased Replace |   Uncased Score |
| :--------------- | ---------------: | ---------------: | ---------------: | ---------------: | --------------: | --------------: | --------------: | --------------: |
| Serial @ Xeon4   |       353.3 MB/s |       177.9 MB/s |       146.3 MB/s |       356.7 MB/s |      107.6 MB/s |       53.9 MB/s |       50.7 MB/s |      107.2 MB/s |
| Parallel @ Xeon4 |      5486.8 MB/s |      2491.1 MB/s |      1975.7 MB/s |      4541.9 MB/s |     1127.4 MB/s |      560.7 MB/s |      524.2 MB/s |     1116.7 MB/s |
| CUDA @ H100      | __26274.5 MB/s__ | __29893.0 MB/s__ | __16256.5 MB/s__ | __34971.8 MB/s__ | __8300.0 MB/s__ | __5379.4 MB/s__ | __4617.1 MB/s__ | __5991.5 MB/s__ |

## Least Frequent 10% of the Vocabulary

| Backend          |      Cased Count |       Cased Find |   Cased Replace |      Cased Score |   Uncased Count |    Uncased Find | Uncased Replace |   Uncased Score |
| :--------------- | ---------------: | ---------------: | --------------: | ---------------: | --------------: | --------------: | --------------: | --------------: |
| Serial @ Xeon4   |       140.3 MB/s |        71.2 MB/s |       54.4 MB/s |       122.8 MB/s |       76.5 MB/s |       38.0 MB/s |       32.5 MB/s |       75.5 MB/s |
| Parallel @ Xeon4 |      2083.1 MB/s |       991.4 MB/s |      589.9 MB/s |      1675.0 MB/s |      958.7 MB/s |      476.3 MB/s |      405.9 MB/s |      941.3 MB/s |
| CUDA @ H100      | __11220.6 MB/s__ | __10930.7 MB/s__ | __7065.2 MB/s__ | __13636.5 MB/s__ | __7086.7 MB/s__ | __4348.7 MB/s__ | __3425.2 MB/s__ | __4380.9 MB/s__ |

## Entire Vocabulary

| Backend          |     Cased Count |      Cased Find |   Cased Replace |     Cased Score |   Uncased Count |    Uncased Find | Uncased Replace |   Uncased Score |
| :--------------- | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| Serial @ Xeon4   |       34.7 MB/s |       17.5 MB/s |       12.0 MB/s |       31.9 MB/s |       27.2 MB/s |       12.0 MB/s |        9.7 MB/s |       23.6 MB/s |
| Parallel @ Xeon4 |      563.7 MB/s |      269.3 MB/s |      150.4 MB/s |      442.9 MB/s |      396.5 MB/s |      166.3 MB/s |      133.6 MB/s |      324.7 MB/s |
| CUDA @ H100      | __7698.7 MB/s__ | __4649.3 MB/s__ | __2652.1 MB/s__ | __3972.8 MB/s__ | __4831.8 MB/s__ | __2566.2 MB/s__ | __1395.9 MB/s__ | __1567.7 MB/s__ |
