"""
StringZilla sequence operations benchmark script.

This script benchmarks string splitting operations using different backends:
- Python's built-in `str.split()`
- StringZilla's `Str.split()`
- Python's built-in `sort()` on the result of `str.split()`
- StringZilla's `Str.split().sort()`

Example usage vi UV:

    # Benchmark with a file
    uv run --no-project scripts/bench_sequence.py --haystack-path leipzig1M.txt --needle "\n"

    # Benchmark with synthetic data
    uv run --no-project scripts/bench_sequence.py --haystack-pattern "hello world " --haystack-length 1000000 --needle " "
"""

import argparse
import time

from stringzilla import Str, File


def log(name: str, bytes_length: int, operator: callable):
    a = time.time_ns()
    operator()
    b = time.time_ns()
    secs = (b - a) / 1e9
    gb_per_sec = bytes_length / (1e9 * secs)
    print(f"{name}: took {secs:.4f} seconds ~ {gb_per_sec:.3f} GB/s")


def log_functionality(
    pattern: str,
    bytes_length: int,
    pythonic_str: str,
    stringzilla_str: Str,
    stringzilla_file: File,
):
    log("str.split", bytes_length, lambda: pythonic_str.split(pattern))
    log("Str.split", bytes_length, lambda: stringzilla_str.split(pattern))
    log("str.split.sort", bytes_length, lambda: pythonic_str.split(pattern).sort())
    log("Str.split.sort", bytes_length, lambda: stringzilla_str.split(pattern).sort())


def bench(
    haystack_path: str = None,
    haystack_pattern: str = None,
    haystack_length: int = None,
    needle: str = None,
):
    """Run string splitting benchmarks."""
    if haystack_path:
        pythonic_str: str = open(haystack_path, "r").read()
        stringzilla_file = File(haystack_path)
    else:
        haystack_length = int(haystack_length)
        repetitions = haystack_length // len(haystack_pattern)
        pythonic_str: str = haystack_pattern * repetitions
        stringzilla_file = None

    stringzilla_str = Str(pythonic_str)
    print(f"Prepared input of {len(pythonic_str):,} length!")

    log_functionality(
        needle,
        len(stringzilla_str),
        pythonic_str,
        stringzilla_str,
        stringzilla_file,
    )


_main_epilog = """
Examples:

  # Benchmark with a file
  %(prog)s --haystack-path leipzig1M.txt --needle "\\n"
  
  # Benchmark with synthetic data  
  %(prog)s --haystack-pattern "hello world " --haystack-length 1000000 --needle " "
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla sequence operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    parser.add_argument("--haystack-path", help="Path to input file")
    parser.add_argument(
        "--haystack-pattern", help="Pattern to repeat for synthetic data"
    )
    parser.add_argument(
        "--haystack-length", type=int, help="Length of synthetic haystack"
    )
    parser.add_argument("--needle", required=True, help="Pattern to split on")

    args = parser.parse_args()

    if args.haystack_path:
        if args.haystack_pattern or args.haystack_length:
            parser.error("Cannot specify both --haystack-path and synthetic options")
    else:
        if not (args.haystack_pattern and args.haystack_length):
            parser.error(
                "Must specify either --haystack-path or both --haystack-pattern and --haystack-length"
            )

    bench(args.haystack_path, args.haystack_pattern, args.haystack_length, args.needle)


if __name__ == "__main__":
    main()
