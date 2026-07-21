# Similarities for StringZillas

The similarities engines score __large collections__ of strings against each other as a cross-product matrix, the workhorse of __fuzzy matching__ and __bioinformatics__ alignment.
Levenshtein computes the minimum-cost __edit distance__ byte by byte, and Levenshtein UTF-8 does the same __codepoint by codepoint__ for correct results on multibyte text.
Needleman-Wunsch maximizes a signed __global__ alignment score end-to-end, while Smith-Waterman finds the best-scoring __local__ subsequence; both fold biological alphabets into a compact class-based substitution matrix.
Every engine runs across a slice of CPU cores or a CUDA GPU, routing string pairs into size-tiered kernels.

Throughput is reported in __GCUPS__ (billions of cell updates per second), the standard alignment metric where one cell update is one dynamic-programming step.

## Methodology

Each table fixes one input shape: a corpus of equal-length DNA-like (`acgt`) sequences, scored all-pairs.
Cells carry __GCUPS__ for the `_unit` cost scheme (match/mismatch/gap of 0/1/1), parsed from the benchmark's `Efficiency: <X> GOps/s` line, where one GOp equals one billion cell updates.
The all-pairs tile is the __balanced square auto-sized per device__: the per-call pair budget is `STRINGWARS_BATCH_PER_CORE × parallelism` (256 × 16 cores → `q64×c64` on the CPU; 256 × 132 SMs → `q183×c184` on the H100), and each cell reports that balanced tile.
A `↑` cell means there is no dedicated kernel at that ISA level, so the dispatcher reuses the kernel from the tier above it; an empty cell is genuinely-missing data.

## 100-byte Sequences

| Backend          |  Levenshtein | Levenshtein UTF-8 | Needleman-Wunsch | Smith-Waterman |
| :--------------- | -----------: | ----------------: | ---------------: | -------------: |
| Serial @ Xeon4   |   81.2 GCUPS |        10.6 GCUPS |        1.9 GCUPS |      1.5 GCUPS |
| Haswell @ Xeon4  |            ↑ |                 ↑ |       15.2 GCUPS |     13.4 GCUPS |
| Ice Lake @ Xeon4 |  114.6 GCUPS |         6.7 GCUPS |       16.0 GCUPS |     15.8 GCUPS |
| Hopper @ H100    | 1499.8 GCUPS |                 ↑ |      187.3 GCUPS |    164.8 GCUPS |

> Measured June 27th, 2026.

## 1000-byte Sequences

| Backend          |  Levenshtein | Levenshtein UTF-8 | Needleman-Wunsch | Smith-Waterman |
| :--------------- | -----------: | ----------------: | ---------------: | -------------: |
| Serial @ Xeon4   |  175.4 GCUPS |        13.8 GCUPS |        2.7 GCUPS |      2.4 GCUPS |
| Haswell @ Xeon4  |            ↑ |                 ↑ |       66.9 GCUPS |     50.9 GCUPS |
| Ice Lake @ Xeon4 |  339.4 GCUPS |        36.6 GCUPS |       97.1 GCUPS |     85.0 GCUPS |
| Hopper @ H100    | 5910.8 GCUPS |                 ↑ |      711.3 GCUPS |    615.8 GCUPS |

> Measured June 27th, 2026.
