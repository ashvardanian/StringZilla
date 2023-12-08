# Benchmark for Levenshtein distance computation for most popular Python libraries.
# Prior to benchmarking, downloads a file with tokens and runs a small fuzzy test,
# comparing the outputs of different libraries.
#
# Downloading commonly used datasets:
# !wget --no-clobber -O ./leipzig1M.txt https://introcs.cs.princeton.edu/python/42sort/leipzig1m.txt
#
# Install the libraries:
# !pip install python-levenshtein  # 4.8 M/mo: https://github.com/maxbachmann/python-Levenshtein
# !pip install levenshtein # 4.2 M/mo: https://github.com/maxbachmann/Levenshtein
# !pip install jellyfish # 2.3 M/mo: https://github.com/jamesturk/jellyfish/
# !pip install editdistance # 700 k/mo: https://github.com/roy-ht/editdistance
# !pip install distance # 160 k/mo: https://github.com/doukremt/distance
# !pip install polyleven # 34 k/mo: https://github.com/fujimotos/polyleven

import time
import random
import multiprocessing as mp

import fire

import stringzilla as sz
import polyleven as pl
import editdistance as ed
import jellyfish as jf
import Levenshtein as le


def log(name: str, bytes_length: int, operator: callable):
    a = time.time_ns()
    checksum = operator()
    b = time.time_ns()
    secs = (b - a) / 1e9
    gb_per_sec = bytes_length / (1e9 * secs)
    print(
        f"{name}: took {secs:.2f} seconds ~ {gb_per_sec:.3f} GB/s - checksum is {checksum:,}"
    )


def compute_distances(func, words, sample_words) -> int:
    result = 0
    for word in sample_words:
        for other in words:
            result += func(word, other)
    return result


def log_distances(name, func, words, sample_words) -> int:
    total_bytes = sum(len(w) for w in words) * len(sample_words)
    log(name, total_bytes, lambda: compute_distances(func, words, sample_words))


def bench(text_path: str = None, threads: int = 0):
    text: str = open(text_path, "r").read()
    words: list = text.split(" ")

    targets = (
        ("levenshtein", le.distance),
        ("stringzilla", sz.levenshtein),
        ("polyleven", pl.levenshtein),
        ("editdistance", ed.eval),
        ("jellyfish", jf.levenshtein_distance),
    )

    # Fuzzy Test
    for _ in range(100):  # Test 100 random pairs
        word1, word2 = random.sample(words, 2)
        results = [func(word1, word2) for _, func in targets]
        assert all(
            r == results[0] for r in results
        ), f"Inconsistent results for pair {word1}, {word2}"

    print("Fuzzy test passed. All libraries returned consistent results.")

    # Run the Benchmark
    sample_words = random.sample(words, 100)  # Sample 100 words for benchmarking

    if threads == 1:
        for name, func in targets:
            log_distances(name, func, words, sample_words)
    else:
        processes = []
        for name, func in targets:
            p = mp.Process(target=log_distances, args=(name, func, words, sample_words))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()


if __name__ == "__main__":
    fire.Fire(bench)
