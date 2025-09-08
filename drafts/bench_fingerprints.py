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

Benchmarks document fingerprinting/sketching across libraries:
- datasketch: MinHash and update_batch
- scikit-learn: HashingVectorizer (word / char n-grams)
- stringzillas: SIMD/GPU-accelerated fingerprints (CPU/GPU, single + batch)

Highlights (aligned with bench_similarities.py):
- Monotonic timing and progress with pairs/s, MB/s
- Optional batching for functions that can benefit
- Regex filter to select benchmarks

Examples:
  uv run --no-project scripts/bench_fingerprints.py --dataset leipzig1M.txt
  uv run --no-project scripts/bench_fingerprints.py --dataset leipzig1M.txt -n 10000 -d 256 -b 1024 -k "(sklearn|szs)"
"""

import os
import time
import argparse
import itertools
import re
from pathlib import Path
from typing import Callable, Any, List, Optional

from tqdm import tqdm
import numpy as np

from datasketch import MinHash
from sklearn.feature_extraction.text import HashingVectorizer
import stringzillas as szs
import stringzilla as sz

# Global state for MinHash to avoid repeated initialization
_datasketch_min_hash_state = None
_sklearn_feature_hasher = None
_sklearn_words_vectorizer = None
_sklearn_ngrams_vectorizer = None


def _name_matches(name: str, pattern: Optional[re.Pattern]) -> bool:
    return True if pattern is None else bool(pattern.search(name))


def _checksum_from_results(result) -> int:
    """Normalize different return types to an integer checksum.

    - numpy arrays → sum of elements
    - tuple of arrays (hashes, counts) → sum both
    - iterable of numerics → sum
    - scalar → int(result)
    """
    try:
        if isinstance(result, tuple) and len(result) == 2:
            a, b = result
            sa = int(np.asarray(a).sum())
            sb = int(np.asarray(b).sum())
            return sa + sb
        if isinstance(result, np.ndarray):
            return int(result.sum())
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            return int(sum(int(x) for x in result))
        return int(result)
    except Exception:
        return 0


def log(
    name: str,
    documents: List[str],
    document_sizes: List[int],
    single_doc: Optional[Callable[[str], Any]] = None,
    batch_docs: Optional[Callable[[List[str]], Iterable[Any]]] = None,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    ops_counter: Optional[Callable[[List[str], Iterable[Any]], int]] = None,
):
    """Benchmark an operation with timeout, batching, progress, and checksum.

    Provide one or both callables. If both are given and batch_size > 1, uses batch_docs.
    """
    processed_docs = 0
    processed_bytes = 0
    checksum = 0
    start_ns = time.monotonic_ns()

    try:
        bar = tqdm(desc=name, unit="docs", leave=False, total=len(documents))
        for batch_indices in itertools.batched(range(len(documents)), max(1, batch_size)):
            if (time.monotonic_ns() - start_ns) > int(timeout_seconds * 1e9):
                break

            batch_documents = [documents[i] for i in batch_indices]
            batch_bytes = [document_sizes[i] for i in batch_indices]

            # Choose batch vs single path explicitly
            results_iterable: Iterable[Any]
            if batch_docs is not None and batch_size > 1:
                results = batch_docs(batch_documents)
                if hasattr(results, "__iter__") and not isinstance(results, (str, bytes)):
                    results_iterable = results
                else:
                    results_iterable = [results]
            else:
                if single_doc is None:
                    raise ValueError("single_doc callable is required when batch_docs is not provided")
                results_iterable = (single_doc(doc) for doc in batch_documents)

            for result in results_iterable:
                checksum += _checksum_from_results(result)

            processed_docs += len(batch_documents)
            processed_bytes += sum(batch_bytes)

            # Count operations (hashes computed) if provided
            if ops_counter is not None:
                try:
                    # Recompute results for ops counting if generator was consumed
                    if batch_docs is not None and batch_size > 1:
                        results_for_ops = batch_docs(batch_documents)
                    else:
                        results_for_ops = [single_doc(doc) for doc in batch_documents]  # type: ignore[arg-type]
                    total_ops = total_ops + ops_counter(batch_documents, results_for_ops)
                except Exception:
                    pass

            elapsed_s = (time.monotonic_ns() - start_ns) / 1e9
            if elapsed_s > 0:
                docs_per_sec = processed_docs / elapsed_s
                mb_per_sec = processed_bytes / (1e6 * elapsed_s)
                hashes_per_sec = (total_ops / elapsed_s) if "total_ops" in locals() else 0
                bar.set_postfix(
                    {
                        "docs/s": f"{docs_per_sec:.0f}",
                        "MB/s": f"{mb_per_sec:.1f}",
                        "hashes/s": f"{hashes_per_sec:.0f}",
                        "chk": checksum,
                    }
                )
            bar.update(len(batch_documents))
        bar.close()
    except KeyboardInterrupt:
        print(f"\n{name}: SKIPPED (interrupted by user)")
        return

    total_time_s = (time.monotonic_ns() - start_ns) / 1e9
    if processed_docs:
        docs_per_sec = processed_docs / total_time_s
        mb_per_sec = processed_bytes / (1e6 * total_time_s)
        hashes_per_sec = (total_ops / total_time_s) if "total_ops" in locals() else 0
        extra = f", {hashes_per_sec:.0f} hashes/s" if hashes_per_sec else ""
        print(
            f"{name}: {processed_docs:,} docs in {total_time_s:.2f}s ~ {mb_per_sec:.3f} MB/s, {docs_per_sec:.0f} docs/s{extra}, checksum={checksum}"
        )
    else:
        print(f"{name}: No documents processed")


def benchmark_third_party_fingerprints(
    docs: List[str],
    docs_sizes: List[int],
    dimensions: int,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    filter_pattern: Optional[re.Pattern] = None,
):
    """Benchmark third-party fingerprinting/sketching implementations."""
    global _datasketch_min_hash_state, _sklearn_feature_hasher, _sklearn_words_vectorizer, _sklearn_ngram_vectorizer

    binary_docs = [doc.encode("utf-8") for doc in docs]
    if _datasketch_min_hash_state is None:
        _datasketch_min_hash_state = MinHash(num_perm=dimensions)

    def datasketch_minhash_update(doc: bytes) -> np.ndarray:
        _datasketch_min_hash_state.update(doc)
        digest = _datasketch_min_hash_state.digest()
        _datasketch_min_hash_state.clear()
        return digest

    if _name_matches("datasketch.MinHash.update", filter_pattern):
        log(
            "datasketch.MinHash.update",
            binary_docs,
            docs_sizes,
            datasketch_minhash_update,
            timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_docs, _res: len(batch_docs) * dimensions,
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

    if _name_matches("datasketch.MinHash.update_batch(ngrams)", filter_pattern):
        ww = 3
        log(
            "datasketch.MinHash.update_batch(ngrams)",
            binary_docs,
            docs_sizes,
            datasketch_minhash_update_batch,
            timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_docs, _res, ww=ww: sum(max(len(d) - ww + 1, 0) for d in batch_docs) * dimensions,
        )

    _sklearn_words_vectorizer = HashingVectorizer(
        n_features=dimensions,
        analyzer="word",
        decode_error="ignore",
        norm=None,
    )

    def sklearn_words_vectorizer(doc: str):
        return _sklearn_words_vectorizer.transform([doc])

    if _name_matches("sklearn.HashingVectorizer(word)", filter_pattern):
        log(
            name="sklearn.HashingVectorizer(word)",
            documents=docs,
            document_sizes=docs_sizes,
            single_doc=sklearn_words_vectorizer,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_documents, results: sum(getattr(r, "nnz", 0) for r in results),
        )

    _sklearn_ngrams_vectorizer = HashingVectorizer(
        n_features=dimensions,
        analyzer="char",
        ngram_range=(3, 17),  # trigrams and larger, up to 17-grams
        decode_error="ignore",
        norm=None,
    )

    def sklearn_ngrams_vectorizer(doc: str):
        return _sklearn_ngrams_vectorizer.transform([doc])

    if _name_matches("sklearn.HashingVectorizer(ngram)", filter_pattern):
        log(
            name="sklearn.HashingVectorizer(ngram)",
            documents=docs,
            document_sizes=docs_sizes,
            single_doc=sklearn_ngrams_vectorizer,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_documents, results: sum(getattr(r, "nnz", 0) for r in results),
        )


def benchmark_szs_fingerprints(
    docs: List[str],
    docs_sizes: List[int],
    dimensions: int,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    filter_pattern: Optional[re.Pattern] = None,
):
    """Benchmark StringZillas Fingerprints across device scopes and modes."""
    cpu_cores = os.cpu_count()
    default_scope = szs.DeviceScope()
    cpu_scope = szs.DeviceScope(cpu_cores=cpu_cores)
    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:
        gpu_scope = None

    # Per-doc kernels (single doc per call) for parity with scalar libs
    if _name_matches("szs.Fingerprints(1xCPU)", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=default_scope)

        def kernel(doc: str) -> int:
            hashes, counts = engine(sz.Strs([doc]), device=default_scope)
            return _checksum_from_results((hashes, counts))

        log(
            name="szs.Fingerprints(1xCPU)",
            documents=docs,
            document_sizes=docs_sizes,
            single_doc=kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_documents, _results: len(batch_documents) * dimensions,
        )

    if _name_matches(f"szs.Fingerprints({cpu_cores}xCPU)", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=cpu_scope)

        def kernel(doc: str) -> int:
            hashes, counts = engine(sz.Strs([doc]), device=cpu_scope)
            return _checksum_from_results((hashes, counts))

        log(
            name=f"szs.Fingerprints({cpu_cores}xCPU)",
            documents=docs,
            document_sizes=docs_sizes,
            single_doc=kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_documents, _results, d=dimensions: len(batch_documents) * d,
        )

    if gpu_scope is not None and _name_matches("szs.Fingerprints(1xGPU)", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=gpu_scope)

        def kernel(doc: str) -> int:
            hashes, counts = engine(sz.Strs([doc]), device=gpu_scope)
            return _checksum_from_results((hashes, counts))

        log(
            name="szs.Fingerprints(1xGPU)",
            documents=docs,
            document_sizes=docs_sizes,
            single_doc=kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_documents, _results, d=dimensions: len(batch_documents) * d,
        )

    # Batched kernels (list[str] → list[int] checksums)
    if _name_matches(f"szs.Fingerprints(1xCPU,batch={batch_size})", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=default_scope)

        def kernel_batch(batch_docs: List[str]) -> List[int]:
            hashes, counts = engine(sz.Strs(batch_docs), device=default_scope)
            # Reduce per document: sum of row in both arrays
            hashes = np.asarray(hashes)
            counts = np.asarray(counts)
            per_doc = hashes.sum(axis=1).astype(np.int64) + counts.sum(axis=1).astype(np.int64)
            return [int(x) for x in per_doc]

        log(
            name=f"szs.Fingerprints(1xCPU,batch={batch_size})",
            documents=docs,
            document_sizes=docs_sizes,
            batch_docs=kernel_batch,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            ops_counter=lambda batch_documents, _results, d=dimensions: len(batch_documents) * d,
        )

    if _name_matches(f"szs.Fingerprints({cpu_cores}xCPU,batch={batch_size})", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=cpu_scope)

        def kernel_batch(batch_docs: List[str]) -> List[int]:
            hashes, counts = engine(sz.Strs(batch_docs), device=cpu_scope)
            hashes = np.asarray(hashes)
            counts = np.asarray(counts)
            per_doc = hashes.sum(axis=1).astype(np.int64) + counts.sum(axis=1).astype(np.int64)
            return [int(x) for x in per_doc]

        log(
            name=f"szs.Fingerprints({cpu_cores}xCPU,batch={batch_size})",
            documents=docs,
            document_sizes=docs_sizes,
            batch_docs=kernel_batch,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            ops_counter=lambda batch_documents, _results, d=dimensions: len(batch_documents) * d,
        )

    if gpu_scope is not None and _name_matches(f"szs.Fingerprints(1xGPU,batch={batch_size})", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=gpu_scope)

        def kernel_batch(batch_docs: List[str]) -> List[int]:
            hashes, counts = engine(sz.Strs(batch_docs), device=gpu_scope)
            hashes = np.asarray(hashes)
            counts = np.asarray(counts)
            per_doc = hashes.sum(axis=1).astype(np.int64) + counts.sum(axis=1).astype(np.int64)
            return [int(x) for x in per_doc]

        log(
            name=f"szs.Fingerprints(1xGPU,batch={batch_size})",
            documents=docs,
            document_sizes=docs_sizes,
            batch_docs=kernel_batch,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            ops_counter=lambda batch_documents, _results, d=dimensions: len(batch_documents) * d,
        )


def bench(
    dataset_path: str,
    max_docs: int = None,
    dimensions: int = 1024,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    filter_pattern: Optional[re.Pattern] = None,
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

    print(f"Prepared {len(docs):,} docs of {sum(docs_sizes)/len(docs_sizes):.1f} mean byte length!")
    print(f"Total bytes: {sum(docs_sizes):,}")
    print(f"Num hashes: {dimensions}")
    print()

    print("=== Fingerprinting & Sketching Benchmarks ===")
    benchmark_third_party_fingerprints(docs, docs_sizes, dimensions, timeout_seconds, batch_size, filter_pattern)
    benchmark_szs_fingerprints(docs, docs_sizes, dimensions, timeout_seconds, batch_size, filter_pattern)


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
    parser.add_argument("-n", "--max-docs", type=int, help="Maximum number of docs to process")
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        default=1024,
        help="Number of hash functions for MinHash (default: 1024)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds for each benchmark (default: 10)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for batch-capable APIs (default: 1)",
    )
    parser.add_argument(
        "-k",
        "--filter",
        metavar="REGEX",
        help="Regex to select which benchmarks to run by name",
    )

    args = parser.parse_args()
    pattern = None
    if args.filter:
        try:
            pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter/-k: {e}")

    bench(args.dataset, args.max_docs, args.dimensions, args.timeout, args.batch_size, pattern)


if __name__ == "__main__":
    main()
