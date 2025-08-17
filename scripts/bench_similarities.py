# /// script
# dependencies = [
#   "stringzilla",
#   "rapidfuzz",
#   "python-Levenshtein",
#   "levenshtein",
#   "jellyfish",
#   "editdistance",
#   "distance",
#   "polyleven",
#   "edlib",
#   "nltk",
#   "biopython",
#   "numpy",
#   "tqdm",
# ]
# ///
"""
StringZilla similarity benchmark script.

This script benchmarks string similarity operations using various libraries:
- stringzilla: Fast edit distance and alignment scoring
- rapidfuzz: Fast fuzzy string matching
- python-Levenshtein: Classic Levenshtein distance
- jellyfish: Multiple string distance metrics
- editdistance: Pure Python edit distance
- nltk: Natural language toolkit distances
- edlib: Fast sequence alignment
- biopython: Needleman-Wunsch alignment with BLOSUM matrices

Example usage via UV:

    # Benchmark with a file
    uv run --no-project scripts/bench_similarities.py --dataset leipzig1M.txt

    # Benchmark with limited pairs
    uv run --no-project scripts/bench_similarities.py --dataset leipzig1M.txt --max-pairs 1000

    # Benchmark with custom timeout
    uv run --no-project scripts/bench_similarities.py --dataset leipzig1M.txt --timeout 30

    # Benchmark protein sequences
    uv run --no-project scripts/bench_similarities.py --protein-mode --protein-length 500

    # Benchmark protein sequences with a custom file
    uv run --no-project scripts/bench_similarities.py --protein-mode --dataset acgt_1k.txt
"""

import os
import time
import random
import argparse
from pathlib import Path
from typing import List, Callable, Tuple

from tqdm import tqdm
import numpy as np

# String similarity libraries
import stringzilla as sz
import stringzillas as szs
import jellyfish as jf
import Levenshtein as le
import editdistance as ed
from rapidfuzz.distance import Levenshtein as rf
from nltk.metrics.distance import edit_distance as nltk_ed
import edlib

try:
    import polyleven

    POLYLEVEN_AVAILABLE = True
except ImportError:
    POLYLEVEN_AVAILABLE = False

# For Needleman-Wunsch alignment
try:
    from Bio import Align
    from Bio.Align import substitution_matrices

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# Global state for initialized models
_biopython_aligner = None
_blosum_matrix = None


def log_similarity_operation(
    name: str,
    string_pairs: List[Tuple[str, str]],
    similarity_func: Callable,
    timeout_seconds: int = 10,
):
    """Benchmark a similarity operation with timeout and progress tracking."""
    processed_pairs = 0
    processed_bytes = 0
    checksum = 0
    start_time = time.time_ns()

    try:
        with tqdm(desc=name, unit="pairs", leave=False, total=len(string_pairs)) as progress_bar:
            for str_a, str_b in string_pairs:
                # Check timeout (convert seconds to nanoseconds)
                if time.time_ns() - start_time > timeout_seconds * 1e9:
                    break

                try:
                    distance = similarity_func(str_a, str_b)
                    checksum += distance
                    processed_pairs += 1
                    processed_bytes += len(str_a.encode("utf-8")) + len(str_b.encode("utf-8"))

                    # Update progress bar with custom rate
                    elapsed_ns = time.time_ns() - start_time
                    elapsed_s = elapsed_ns / 1e9
                    if elapsed_s > 0:
                        pairs_per_sec = processed_pairs / elapsed_s
                        bytes_per_sec = processed_bytes / elapsed_s
                        progress_bar.set_postfix(
                            {
                                "pairs/s": f"{pairs_per_sec:.0f}",
                                "MB/s": f"{bytes_per_sec/1e6:.1f}",
                                "checksum": f"{checksum}",
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
    if processed_pairs > 0:
        pairs_per_sec = processed_pairs / total_time_s
        mb_per_sec = processed_bytes / (1e6 * total_time_s)
        print(
            f"{name}: {processed_pairs:,} pairs in {total_time_s:.2f}s ~ {mb_per_sec:.3f} MB/s, {pairs_per_sec:.0f} pairs/s, checksum={checksum}"
        )
    else:
        print(f"{name}: No pairs processed")


def benchmark_edit_distances(string_pairs: List[Tuple[str, str]], timeout_seconds: int = 10):
    """Benchmark various edit distance implementations."""

    # StringZilla
    log_similarity_operation(
        "stringzilla.edit_distance",
        string_pairs,
        sz.edit_distance,
        timeout_seconds,
    )

    log_similarity_operation(
        "stringzilla.edit_distance_unicode",
        string_pairs,
        sz.edit_distance_unicode,
        timeout_seconds,
    )

    # RapidFuzz
    log_similarity_operation(
        "rapidfuzz.Levenshtein.distance",
        string_pairs,
        rf.distance,
        timeout_seconds,
    )

    # python-Levenshtein
    log_similarity_operation(
        "Levenshtein.distance",
        string_pairs,
        le.distance,
        timeout_seconds,
    )

    # Jellyfish
    log_similarity_operation(
        "jellyfish.levenshtein_distance",
        string_pairs,
        jf.levenshtein_distance,
        timeout_seconds,
    )

    # EditDistance
    log_similarity_operation(
        "editdistance.eval",
        string_pairs,
        ed.eval,
        timeout_seconds,
    )

    # NLTK
    log_similarity_operation(
        "nltk.edit_distance",
        string_pairs,
        nltk_ed,
        timeout_seconds,
    )

    # Edlib
    def edlib_distance(a: str, b: str) -> int:
        return edlib.align(a, b, mode="NW", task="distance")["editDistance"]

    log_similarity_operation(
        "edlib.align",
        string_pairs,
        edlib_distance,
        timeout_seconds,
    )

    # Polyleven (if available)
    if POLYLEVEN_AVAILABLE:
        log_similarity_operation(
            "polyleven.levenshtein",
            string_pairs,
            polyleven.levenshtein,
            timeout_seconds,
        )

    # StringZillas batch processing
    def benchmark_stringzillas_batch(engine_name, engine_class, device_scope):
        try:
            engine = engine_class()

            # Prepare data for batch processing
            strings_a = sz.Strs([pair[0] for pair in string_pairs])
            strings_b = sz.Strs([pair[1] for pair in string_pairs])

            start_time = time.time_ns()
            results = engine(strings_a, strings_b, device_scope)
            end_time = time.time_ns()

            total_time_s = (end_time - start_time) / 1e9
            processed_bytes = sum(len(a.encode("utf-8")) + len(b.encode("utf-8")) for a, b in string_pairs)
            mb_per_sec = processed_bytes / (1e6 * total_time_s)
            pairs_per_sec = len(string_pairs) / total_time_s
            checksum = sum(results) if hasattr(results, "__iter__") else 0

            print(
                f"{engine_name}: {len(string_pairs):,} pairs in {total_time_s:.2f}s ~ {mb_per_sec:.3f} MB/s, {pairs_per_sec:.0f} pairs/s, checksum={checksum}"
            )
        except Exception as e:
            print(f"{engine_name}: FAILED - {e}")

    # StringZillas Levenshtein distances (batch)
    cpu_scope = szs.DeviceScope(cpu_cores=os.cpu_count())
    benchmark_stringzillas_batch("stringzillas.LevenshteinDistances(CPU)", szs.LevenshteinDistances, cpu_scope)

    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
        benchmark_stringzillas_batch("stringzillas.LevenshteinDistances(GPU)", szs.LevenshteinDistances, gpu_scope)
    except:
        pass  # GPU may not be available

    # StringZillas UTF-8 Levenshtein distances (batch)
    benchmark_stringzillas_batch("stringzillas.LevenshteinDistancesUTF8(CPU)", szs.LevenshteinDistancesUTF8, cpu_scope)

    try:
        benchmark_stringzillas_batch(
            "stringzillas.LevenshteinDistancesUTF8(GPU)", szs.LevenshteinDistancesUTF8, gpu_scope
        )
    except:
        pass  # GPU may not be available


def benchmark_alignment_scores(string_pairs: List[Tuple[str, str]], timeout_seconds: int = 10):
    """Benchmark alignment scoring with substitution matrices."""
    global _biopython_aligner, _blosum_matrix

    if not BIOPYTHON_AVAILABLE:
        print("BioPython not available, skipping alignment benchmarks")
        return

    # Initialize BioPython aligner
    if _biopython_aligner is None:
        _biopython_aligner = Align.PairwiseAligner()
        _biopython_aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        _biopython_aligner.open_gap_score = 1
        _biopython_aligner.extend_gap_score = 1

        # Convert BLOSUM matrix to dense 256x256 for StringZilla
        subs_packed = np.array(_biopython_aligner.substitution_matrix).astype(np.int8)
        _blosum_matrix = np.zeros((256, 256), dtype=np.int8)
        _blosum_matrix.fill(127)  # Large penalty for invalid characters

        for packed_row, packed_row_aminoacid in enumerate(_biopython_aligner.substitution_matrix.alphabet):
            for packed_column, packed_column_aminoacid in enumerate(_biopython_aligner.substitution_matrix.alphabet):
                reconstructed_row = ord(packed_row_aminoacid)
                reconstructed_column = ord(packed_column_aminoacid)
                _blosum_matrix[reconstructed_row, reconstructed_column] = subs_packed[packed_row, packed_column]

    # StringZilla alignment score
    def sz_alignment_score(a: str, b: str) -> int:
        return sz.alignment_score(a, b, substitution_matrix=_blosum_matrix, gap_score=1)

    log_similarity_operation("stringzilla.alignment_score", string_pairs, sz_alignment_score, timeout_seconds)

    # BioPython alignment score
    log_similarity_operation(
        "biopython.PairwiseAligner.score",
        string_pairs,
        _biopython_aligner.score,
        timeout_seconds,
    )

    # StringZillas alignment functions (batch)
    def benchmark_stringzillas_alignment_batch(engine_name, engine_class, substitution_matrix, device_scope):
        try:
            engine = engine_class(substitution_matrix=substitution_matrix)

            # Prepare data for batch processing
            strings_a = sz.Strs([pair[0] for pair in string_pairs])
            strings_b = sz.Strs([pair[1] for pair in string_pairs])

            start_time = time.time_ns()
            results = engine(strings_a, strings_b, device_scope)
            end_time = time.time_ns()

            total_time_s = (end_time - start_time) / 1e9
            processed_bytes = sum(len(a.encode("utf-8")) + len(b.encode("utf-8")) for a, b in string_pairs)
            mb_per_sec = processed_bytes / (1e6 * total_time_s)
            pairs_per_sec = len(string_pairs) / total_time_s
            checksum = sum(results) if hasattr(results, "__iter__") else 0

            print(
                f"{engine_name}: {len(string_pairs):,} pairs in {total_time_s:.2f}s ~ {mb_per_sec:.3f} MB/s, {pairs_per_sec:.0f} pairs/s, checksum={checksum}"
            )
        except Exception as e:
            print(f"{engine_name}: FAILED - {e}")

    # StringZillas Needleman-Wunsch (global alignment)
    cpu_scope = szs.DeviceScope(cpu_cores=os.cpu_count())
    benchmark_stringzillas_alignment_batch(
        "stringzillas.NeedlemanWunsch(CPU)", szs.NeedlemanWunsch, _blosum_matrix, cpu_scope
    )

    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
        benchmark_stringzillas_alignment_batch(
            "stringzillas.NeedlemanWunsch(GPU)", szs.NeedlemanWunsch, _blosum_matrix, gpu_scope
        )
    except:
        pass  # GPU may not be available

    # StringZillas Smith-Waterman (local alignment)
    benchmark_stringzillas_alignment_batch(
        "stringzillas.SmithWaterman(CPU)", szs.SmithWaterman, _blosum_matrix, cpu_scope
    )

    try:
        benchmark_stringzillas_alignment_batch(
            "stringzillas.SmithWaterman(GPU)", szs.SmithWaterman, _blosum_matrix, gpu_scope
        )
    except:
        pass  # GPU may not be available


def generate_random_pairs(strings: List[str], num_pairs: int) -> List[Tuple[str, str]]:
    """Generate random string pairs from a list of strings."""
    return [(random.choice(strings), random.choice(strings)) for _ in range(num_pairs)]


def generate_protein_sequences(num_sequences: int, length: int) -> List[str]:
    """Generate random protein sequences using ACGT alphabet."""
    return ["".join(random.choices("ACGT", k=length)) for _ in range(num_sequences)]


def bench(
    dataset_path: str,
    max_pairs: int = None,
    timeout_seconds: int = 10,
    protein_mode: bool = False,
    protein_length: int = 1000,
):
    """Run similarity benchmarks."""

    if protein_mode:
        print("=== Protein Sequence Benchmarks ===")
        print(f"Generating {protein_length}-length protein sequences...")
        proteins = generate_protein_sequences(1000, protein_length)
        pairs = generate_random_pairs(proteins, max_pairs or 1000)

        print(f"Generated {len(pairs):,} protein sequence pairs")
        print(f"Average sequence length: {protein_length} chars")
        print(f"Timeout per benchmark: {timeout_seconds}s")
        print()

        print("=== Edit Distance Benchmarks ===")
        benchmark_edit_distances(pairs, timeout_seconds)
        print()

        print("=== Alignment Score Benchmarks ===")
        benchmark_alignment_scores(pairs, timeout_seconds)

    else:
        # Load dataset
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
            strings = [line.strip() for line in f if line.strip()]

        # Generate random pairs
        num_pairs = max_pairs or min(100000, len(strings) * 10)
        pairs = generate_random_pairs(strings, num_pairs)

        total_chars = sum(len(a) + len(b) for a, b in pairs)
        avg_length = total_chars / (2 * len(pairs))

        print(f"Prepared {len(pairs):,} string pairs from {len(strings):,} unique strings")
        print(f"Average string length: {avg_length:.1f} chars")
        print(f"Total characters: {total_chars:,}")
        print(f"Timeout per benchmark: {timeout_seconds}s")
        print()

        print("=== Edit Distance Benchmarks ===")
        benchmark_edit_distances(pairs, timeout_seconds)


_main_epilog = """
Examples:

  # Benchmark with a file
  %(prog)s --dataset leipzig1M.txt

  # Benchmark with limited pairs
  %(prog)s --dataset leipzig1M.txt --max-pairs 1000

  # Benchmark protein sequences
  %(prog)s --protein-mode --protein-length 5000 --max-pairs 500

  # Custom timeout
  %(prog)s --dataset leipzig1M.txt --timeout 30
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla similarity operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    parser.add_argument("--dataset", help="Path to text dataset file")
    parser.add_argument("--max-pairs", type=int, help="Maximum number of string pairs to process")
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout in seconds for each benchmark (default: 10)",
    )
    parser.add_argument(
        "--protein-mode",
        action="store_true",
        help="Generate random protein sequences instead of using dataset",
    )
    parser.add_argument(
        "--protein-length",
        type=int,
        default=1000,
        help="Length of generated protein sequences (default: 1000)",
    )

    args = parser.parse_args()

    if not args.protein_mode and not args.dataset:
        parser.error("Either --dataset or --protein-mode is required")

    bench(
        args.dataset,
        args.max_pairs,
        args.timeout,
        args.protein_mode,
        args.protein_length,
    )


if __name__ == "__main__":
    main()
