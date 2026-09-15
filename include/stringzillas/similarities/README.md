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
The codepoint engine has no Hopper-specific kernel, so its H100 figure is the one the base-CUDA codepoint family delivers, which is what an H100 dispatches to.

## 100-byte Sequences

| Backend          | Levenshtein | Levenshtein UTF-8 | Needleman-Wunsch | Smith-Waterman |
| :--------------- | ----------: | ----------------: | ---------------: | -------------: |
| Serial @ Xeon4   |  32.5 GCUPS |         4.1 GCUPS |        3.6 GCUPS |      3.0 GCUPS |
| Haswell @ Xeon4  |           ↑ |                 ↑ |       50.7 GCUPS |     44.8 GCUPS |
| Ice Lake @ Xeon4 |  38.8 GCUPS |        11.0 GCUPS |       67.7 GCUPS |     53.6 GCUPS |
| Hopper @ H100    | 982.1 GCUPS |       619.1 GCUPS |      145.1 GCUPS |    132.6 GCUPS |

> Measured August 23rd, 2026.

## 1000-byte Sequences

| Backend          |  Levenshtein | Levenshtein UTF-8 | Needleman-Wunsch | Smith-Waterman |
| :--------------- | -----------: | ----------------: | ---------------: | -------------: |
| Serial @ Xeon4   |  156.5 GCUPS |        13.9 GCUPS |        5.6 GCUPS |      4.5 GCUPS |
| Haswell @ Xeon4  |            ↑ |                 ↑ |       78.1 GCUPS |     55.3 GCUPS |
| Ice Lake @ Xeon4 |  139.8 GCUPS |        10.3 GCUPS |       77.0 GCUPS |     70.1 GCUPS |
| Hopper @ H100    | 3663.4 GCUPS |      3255.6 GCUPS |      590.1 GCUPS |    501.3 GCUPS |

> Measured August 23rd, 2026.
