# Fingerprinting

The fingerprints engine sketches each string into a fixed-width __MinHash__ signature by rolling several hash windows of different widths over the text and keeping the minimum hash per dimension.
Two near-duplicate documents land on overlapping hash dimensions even after small edits, which powers __near-duplicate detection__ and __multi-pattern search__ across web-scale corpora.
A single pass over a tape yields one compact signature per document that you can index, cluster, or compare cheaply.
The engine runs across a slice of CPU cores or a CUDA GPU, saturating wide SIMD registers and full CUDA warps with at least 64 dimensions per window width.

Each cell is dual: __MB/s__ of text consumed and __hashes/s__ of rolling hashes emitted, since one operation equals one hash dimension produced.

## Methodology

Each table fixes one input shape: a corpus of equal-length DNA-like (`acgt`) sequences sketched into MinHash signatures.
The throughput value is parsed from the benchmark's `Throughput` line, and the hash-rate value from its `Efficiency` line, where one operation already equals one emitted hash (inputs × dimensions).
CPU throughput is per-byte and so stays roughly flat across document lengths, while the emitted hash-rate falls ~10× per 10× length (fewer, longer documents for the same bytes); the GPU instead scales up with length as longer documents fill more of each warp.

## 100-byte Sequences

| Backend         |                    MinHash |
| :-------------- | -------------------------: |
| Serial @ Xeon4  |   41.1 MB/s · 27.5 Mhash/s |
| Haswell @ Xeon4 |   64.3 MB/s · 43.2 Mhash/s |
| Skylake @ Xeon4 | 193.6 MB/s · 129.9 Mhash/s |
| Hopper @ H100   | 438.7 MB/s · 294.4 Mhash/s |

> Measured June 27th, 2026.

## 1000-byte Sequences

| Backend         |                   MinHash |
| :-------------- | ------------------------: |
| Serial @ Xeon4  |  38.6 MB/s ·  2.6 Mhash/s |
| Haswell @ Xeon4 |  61.8 MB/s ·  4.1 Mhash/s |
| Skylake @ Xeon4 | 192.4 MB/s · 12.9 Mhash/s |
| Hopper @ H100   |  3.4 GB/s · 236.2 Mhash/s |

> Measured June 27th, 2026.

## 10000-byte Sequences

| Backend         |                  MinHash |
| :-------------- | -----------------------: |
| Serial @ Xeon4  |  39.6 MB/s · 266 Khash/s |
| Haswell @ Xeon4 |  64.9 MB/s · 436 Khash/s |
| Skylake @ Xeon4 | 199.2 MB/s · 1.3 Mhash/s |
| Hopper @ H100   | 11.2 GB/s · 77.2 Mhash/s |

> Measured June 27th, 2026.

## 100000-byte Sequences

| Backend         |                  MinHash |
| :-------------- | -----------------------: |
| Serial @ Xeon4  |  38.9 MB/s ·  26 Khash/s |
| Haswell @ Xeon4 |  64.6 MB/s ·  43 Khash/s |
| Skylake @ Xeon4 | 196.2 MB/s · 132 Khash/s |
| Hopper @ H100   |   5.6 GB/s · 3.9 Mhash/s |

> Measured June 27th, 2026.
