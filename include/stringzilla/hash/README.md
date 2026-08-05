# Hash: Byte Sum, AES Hash, and SHA-256

This directory holds the digest kernels behind `sz_bytesum`, `sz_hash`, the multi-seed `sz_hash` fan-out, `sz_sha256`, and the batched `sz_sha256_multistate`.
Each operation has a serial baseline plus per-ISA SIMD backends — `westmere`, `haswell`, `skylake`, `icelake` on x86, with a dedicated `goldmont` SHA-NI path for SHA-256.
SHA-256 has no Ice Lake form: SHA-NI is 128-bit only and has no VEX or EVEX encoding, so `goldmont` is the widest single-message kernel there will be.
The dispatcher picks the fastest one available on the running CPU.

## Multi-state SHA-256

Hashing one message is a serial dependency chain: every block feeds the next, so no amount of vector width makes a single digest faster, which is why the single-message column tops out at whatever the SHA-NI instructions deliver.
Independent messages have no such dependency between them.
`sz_sha256_multistate_update` takes one message per lane and runs the ordinary SHA-256 arithmetic across sixteen of them at once on AVX-512, or eight on AVX2, holding the state word-major so each vector register carries the same hash word from every lane.

Lanes are grouped by vector width, and a group advances one block per turn until its longest member is done.
Shorter lanes retire as they finish and are simply left out of the remaining turns, so a batch of ragged lengths is correct and stays vectorized throughout — but a group still costs as much as its longest message, and a lane that retires early leaves its slot idle.
Throughput therefore depends on how evenly lengths are distributed inside a group, which is entirely the caller's to arrange: feeding messages in length order puts similar lengths together and keeps groups full.
`sz_sequence_argsort` produces that ordering.

The two ends of that behaviour are visible in the benchmark, which carries `_one_short` and `_one_long` variants of every multi-state row: trimming one lane of sixteen barely moves the numbers, while stretching one lane to eight times its neighbours drags every backend down to roughly the serial rate, batched or not.

Message length decides how much the width is worth, because each call gathers the lane states into word-major registers and scatters them back afterwards, and that cost is fixed per call rather than per block.
At line length the sixteen-wide kernel is only about a quarter faster than the single-message SHA-NI path; by several kilobytes per lane it is more than twice as fast.

## Methodology

Cells are dual: throughput in GB/s and hash rate in millions of hashes per second, measured with `bench/token.cpp` over the `leipzig1M_en.txt` corpus, reporting the median of repeated runs.
Every backend registers its own row in a single binary, so a row is one kernel rather than one build, and each column is one operation — coverage and cross-chip comparison read down a single column.
The Standard row is the platform's best stock equivalent per column — `std::accumulate` for Byte Sum and `std::hash` for the hashing columns.
Token length matters, so results are split into a Short Words table (tokens averaging 5 bytes) and a Long Lines table (tokens averaging 130 bytes).
Multi-seed hashing emits eight digests per input, so its hash rate is far higher at comparable throughput; SHA-256's accelerator is the SHA-NI path, shown as the Goldmont row.
Each multi-state row pairs its own tier's update with its own tier's digest, which matters more than it sounds: Goldmont is legacy-SSE SHA-NI while Skylake and above are AVX-512, and letting every row borrow one shared wide digest both lends the narrow tiers a kernel they do not have and makes the SHA-NI ones pay an AVX-SSE transition on every instruction.
A `↑` cell means there is no dedicated kernel at that ISA level, so the dispatcher reuses whichever tier does supply one — usually the tier above, but SHA-256 on Ice Lake reaches back down to Goldmont.
An empty cell is genuinely-missing data.

## Short Words

| Backend          |               Byte Sum |                   Hash |         Multi-seed Hash |               SHA-256 |    Multi-state SHA-256 |
| :--------------- | ---------------------: | ---------------------: | ----------------------: | --------------------: | ---------------------: |
| Standard @ Xeon4 | 0.20 GB/s · 38 Mhash/s | 0.18 GB/s · 35 Mhash/s |                       - |                     - |                      - |
| Serial @ Xeon4   | 0.19 GB/s · 36 Mhash/s |  0.03 GB/s · 6 Mhash/s |  0.03 GB/s ·  6 Mhash/s | 0.02 GB/s · 3 Mhash/s |  0.02 GB/s · 3 Mhash/s |
| Westmere @ Xeon4 |                      ↑ | 0.18 GB/s · 35 Mhash/s | 0.91 GB/s · 175 Mhash/s |                     ↑ |                      ↑ |
| Goldmont @ Xeon4 |                      ↑ |                      ↑ |                       ↑ | 0.05 GB/s · 9 Mhash/s | 0.06 GB/s · 11 Mhash/s |
| Haswell @ Xeon4  | 0.18 GB/s · 34 Mhash/s |                      ↑ |                       ↑ |                     ↑ | 0.07 GB/s · 13 Mhash/s |
| Skylake @ Xeon4  | 0.37 GB/s · 71 Mhash/s | 0.34 GB/s · 65 Mhash/s |                       ↑ |                     ↑ | 0.10 GB/s · 20 Mhash/s |
| Ice Lake @ Xeon4 | 0.37 GB/s · 71 Mhash/s | 0.35 GB/s · 68 Mhash/s | 2.02 GB/s · 391 Mhash/s |                     ↑ |                      ↑ |
| NEON @ Graviton4 |                      … |                      … |                       … |                     … |                      … |
| SVE @ Graviton3  |                      … |                      … |                       … |                     … |                      … |

> Measured August 4th, 2026.

## Long Lines

| Backend          |               Byte Sum |                   Hash |          Multi-seed Hash |               SHA-256 |   Multi-state SHA-256 |
| :--------------- | ---------------------: | ---------------------: | -----------------------: | --------------------: | --------------------: |
| Standard @ Xeon4 | 2.74 GB/s · 21 Mhash/s | 3.00 GB/s · 23 Mhash/s |                        - |                     - |                     - |
| Serial @ Xeon4   | 1.94 GB/s · 15 Mhash/s |  0.17 GB/s · 1 Mhash/s |   0.18 GB/s ·  1 Mhash/s | 0.18 GB/s · 1 Mhash/s | 0.18 GB/s · 1 Mhash/s |
| Westmere @ Xeon4 |                      ↑ | 2.94 GB/s · 23 Mhash/s |   6.36 GB/s · 49 Mhash/s |                     ↑ |                     ↑ |
| Goldmont @ Xeon4 |                      ↑ |                      ↑ |                        ↑ | 0.71 GB/s · 5 Mhash/s | 0.81 GB/s · 6 Mhash/s |
| Haswell @ Xeon4  | 2.80 GB/s · 22 Mhash/s |                      ↑ |                        ↑ |                     ↑ | 0.54 GB/s · 4 Mhash/s |
| Skylake @ Xeon4  | 4.54 GB/s · 35 Mhash/s | 2.88 GB/s · 22 Mhash/s |                        ↑ |                     ↑ | 0.90 GB/s · 7 Mhash/s |
| Ice Lake @ Xeon4 | 4.51 GB/s · 35 Mhash/s | 4.84 GB/s · 37 Mhash/s | 13.55 GB/s · 104 Mhash/s |                     ↑ |                     ↑ |
| NEON @ Graviton4 |                      … |                      … |                        … |                     … |                     … |
| SVE @ Graviton3  |                      … |                      … |                        … |                     … |                     … |

> Measured August 4th, 2026.

