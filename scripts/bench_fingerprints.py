# /// script
# dependencies = [
#   "stringzilla",
#   "datasketch",
#   "scikit-learn",
#   "numpy",
#   "tqdm",
# ]
# ///
"""
StringZilla fingerprinting benchmark script.

This script benchmarks MinHash fingerprinting operations using specialized sketching libraries:
- datasketch: MinHash, HyperLogLog, and LSH implementations
- sklearn: Feature hashing and MinHash variants

Example usage via UV:

    # Benchmark with a file
    uv run --no-project scripts/bench_fingerprints.py --dataset leipzig1M.txt

    # Benchmark with limited docs
    uv run --no-project scripts/bench_fingerprints.py --dataset leipzig1M.txt --max-docs 1000

    # Benchmark with custom parameters
    uv run --no-project scripts/bench_fingerprints.py --dataset leipzig1M.txt --dimensions 32
"""

import argparse
import time
from pathlib import Path
from typing import List, Callable, Iterable

from tqdm import tqdm
import numpy as np

from datasketch import MinHash
from sklearn.feature_extraction.text import HashingVectorizer

# Global state for MinHash to avoid repeated initialization
_datasketch_min_hash_state = None
_sklearn_feature_hasher = None
_sklearn_words_vectorizer = None
_sklearn_ngram_vectorizer = None


def log(
    name: str,
    docs: Iterable[str],
    docs_sizes: Iterable[int],
    operation_func: Callable,
    timeout_seconds: int = 10,
):
    """Benchmark an operation with timeout and progress tracking."""
    processed_docs = 0
    processed_bytes = 0
    start_time = time.time_ns()

    try:
        with tqdm(desc=name, unit="docs", leave=False, total=len(docs)) as progress_bar:
            for doc, doc_size in zip(docs, docs_sizes):

                # Check timeout (convert seconds to nanoseconds)
                if time.time_ns() - start_time > timeout_seconds * 1e9:
                    break

                try:
                    operation_func(doc)
                    processed_docs += 1
                    processed_bytes += doc_size

                    # Update progress bar with custom rate
                    elapsed_ns = time.time_ns() - start_time
                    elapsed_s = elapsed_ns / 1e9
                    if elapsed_s > 0:
                        docs_per_sec = processed_docs / elapsed_s
                        bytes_per_sec = processed_bytes / elapsed_s
                        progress_bar.set_postfix(
                            {
                                "docs/s": f"{docs_per_sec:.0f}",
                                "MB/s": f"{bytes_per_sec/1e6:.1f}",
                            }
                        )
                    progress_bar.update(1)

                except Exception as e:
                    # Skip failed operations but continue
                    continue

    except KeyboardInterrupt:
        print(f"\n{name}: SKIPPED (interrupted by user)")
        return

    total_time_ns = time.time_ns() - start_time
    total_time_s = total_time_ns / 1e9
    if processed_docs > 0:
        docs_per_sec = processed_docs / total_time_s
        mb_per_sec = processed_bytes / (1e6 * total_time_s)
        print(
            f"{name}: {processed_docs:,} docs in {total_time_s:.2f}s ~ {mb_per_sec:.3f} MB/s, {docs_per_sec:.0f} docs/s"
        )
    else:
        print(f"{name}: No documents processed")


def log_fingerprinting_functionality(
    docs: Iterable[str],
    docs_sizes: Iterable[int],
    dimensions: int,
    timeout_seconds: int = 10,
):
    """Benchmark fingerprinting and sketching implementations."""
    global _datasketch_min_hash_state, _sklearn_feature_hasher, _sklearn_words_vectorizer, _sklearn_ngram_vectorizer

    binary_docs = [doc.encode("utf-8") for doc in docs]
    if _datasketch_min_hash_state is None:
        _datasketch_min_hash_state = MinHash(num_perm=dimensions)

    def datasketch_minhash_update(doc: bytes) -> np.ndarray:
        _datasketch_min_hash_state.update(doc)
        digest = _datasketch_min_hash_state.digest()
        _datasketch_min_hash_state.clear()
        return digest

    log(
        "datasketch.MinHash.update",
        binary_docs,
        docs_sizes,
        datasketch_minhash_update,
        timeout_seconds,
    )

    def datasketch_minhash_update_batch(
        doc: bytes,
        window_width: int = 3,
    ) -> np.ndarray:
        ngrams = (doc[i : i + window_width] for i in range(len(doc) - window_width + 1))
        _datasketch_min_hash_state.update_batch(ngrams)
        digest = _datasketch_min_hash_state.digest()
        _datasketch_min_hash_state.clear()
        return digest

    log(
        "datasketch.MinHash.update_batch(ngrams)",
        binary_docs,
        docs_sizes,
        datasketch_minhash_update_batch,
        timeout_seconds,
    )

    _sklearn_words_vectorizer = HashingVectorizer(
        n_features=dimensions,
        analyzer="word",
        decode_error="ignore",
        norm=None,
    )

    def sklearn_words_vectorizer(doc: bytes) -> list:
        return _sklearn_words_vectorizer.transform([doc]).toarray()

    log(
        "sklearn.HashingVectorizer(word)",
        docs,
        docs_sizes,
        sklearn_words_vectorizer,
        timeout_seconds,
    )

    _sklearn_ngrams_vectorizer = HashingVectorizer(
        n_features=dimensions,
        analyzer="char",
        ngram_range=(3, 17),  # trigrams and larger, up to 17-grams
        decode_error="ignore",
        norm=None,
    )

    def sklearn_ngrams_vectorizer(doc: bytes) -> list:
        return _sklearn_ngrams_vectorizer.transform([doc]).toarray()

    log(
        "sklearn.HashingVectorizer(ngram)",
        docs,
        docs_sizes,
        sklearn_ngrams_vectorizer,
        timeout_seconds,
    )


def bench(
    dataset_path: str,
    max_docs: int = None,
    dimensions: int = 64,
    timeout_seconds: int = 10,
):
    """Run fingerprinting benchmarks."""

    # Load dataset
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        docs = [doc.strip() for doc in f if doc.strip()]

    if max_docs:
        docs = docs[:max_docs]

    docs_sizes = [len(doc.encode("utf-8")) for doc in docs]

    print(
        f"Prepared {len(docs):,} docs of {sum(docs_sizes)/len(docs_sizes):.1f} mean byte length!"
    )
    print(f"Total bytes: {sum(docs_sizes):,}")
    print(f"Num hashes: {dimensions}")
    print()

    print("=== Fingerprinting & Sketching Benchmarks ===")
    log_fingerprinting_functionality(docs, docs_sizes, dimensions, timeout_seconds)


_main_epilog = """
Examples:

  # Benchmark with a file
  %(prog)s --dataset leipzig1M.txt

  # Benchmark with limited docs
  %(prog)s --dataset leipzig1M.txt --max-docs 1000

  # Custom parameters
  %(prog)s --dataset leipzig1M.txt --dimensions 32
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla fingerprinting operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    parser.add_argument("--dataset", required=True, help="Path to text dataset file")
    parser.add_argument(
        "--max-docs", type=int, help="Maximum number of docs to process"
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=64,
        help="Number of hash functions for MinHash (default: 64)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds for each benchmark (default: 10)",
    )

    args = parser.parse_args()
    bench(args.dataset, args.max_docs, args.dimensions, args.timeout)


if __name__ == "__main__":
    main()
