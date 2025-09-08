# /// script
# dependencies = [
#   "stringzilla"
# ]
# ///
"""
StringZilla find operations benchmark script.

This script benchmarks string search operations using different backends:
- Python's built-in `str.find()` and `str.rfind()`
- StringZilla's `Str.find()` and `Str.rfind()`
- Regular expressions with `re.finditer()`
- StringZilla's `Str.find_first_of()` for character sets
- String translation operations

Example usage via UV:

    # Benchmark with a file
    uv run --no-project scripts/bench_find.py --haystack-path leipzig1M.txt

    # Benchmark with synthetic data
    uv run --no-project scripts/bench_find.py --haystack-pattern "hello world " --haystack-length 1000000
"""

import argparse
import re
import random
import time
from typing import List

from stringzilla import Str


def log(name: str, haystack, patterns, operator: callable):
    a = time.time_ns()
    for pattern in patterns:
        operator(haystack, pattern)
    b = time.time_ns()
    bytes_length = len(haystack) * len(patterns)
    secs = (b - a) / 1e9
    gb_per_sec = bytes_length / (1e9 * secs)
    print(f"{name}: took {secs:.4f} seconds ~ {gb_per_sec:.3f} GB/s")


def find_all(haystack, pattern) -> int:
    count, start = 0, 0
    while True:
        index = haystack.find(pattern, start)
        if index == -1:
            break
        count += 1
        start = index + 1
    return count


def rfind_all(haystack, pattern) -> int:
    count, start = 0, len(haystack) - 1
    while True:
        index = haystack.rfind(pattern, 0, start + 1)
        if index == -1:
            break
        count += 1
        start = index - 1
    return count


def find_all_regex(haystack: str, characters: str) -> int:
    regex_matcher = re.compile(f"[{characters}]")
    count = 0
    for _ in re.finditer(regex_matcher, haystack):
        count += 1
    return count


def find_all_sets(haystack: Str, characters: str) -> int:
    count, start = 0, 0
    while True:
        index = haystack.find_first_of(characters, start)
        if index == -1:
            break
        count += 1
        start = index + 1
    return count


def translate(haystack: Str, look_up_table) -> str:
    return haystack.translate(look_up_table)


def log_functionality(
    tokens: List[str],
    pythonic_str: str,
    stringzilla_str: Str,
):
    # Read-only Search
    log("str.find", pythonic_str, tokens, find_all)
    log("Str.find", stringzilla_str, tokens, find_all)
    log("str.rfind", pythonic_str, tokens, rfind_all)
    log("Str.rfind", stringzilla_str, tokens, rfind_all)
    log("re.finditer", pythonic_str, [r" \t\n\r"], find_all_regex)
    log("Str.find_first_of", stringzilla_str, [r" \t\n\r"], find_all_sets)

    # Search & Modify
    identity = bytes(range(256))
    reverse = bytes(reversed(identity))
    repeated = bytes(range(64)) * 4
    hex = b"0123456789abcdef" * 16
    log(
        "str.translate",
        pythonic_str,
        [
            bytes.maketrans(identity, reverse),
            bytes.maketrans(identity, repeated),
            bytes.maketrans(identity, hex),
        ],
        translate,
    )
    log("Str.translate", stringzilla_str, [reverse, repeated, hex], translate)


def bench(
    haystack_path: str = None,
    haystack_pattern: str = None,
    haystack_length: int = None,
):
    """Run string search benchmarks."""
    if haystack_path:
        pythonic_str: str = open(haystack_path, "r").read()
    else:
        haystack_length = int(haystack_length)
        repetitions = haystack_length // len(haystack_pattern)
        pythonic_str: str = haystack_pattern * repetitions

    stringzilla_str = Str(pythonic_str)
    tokens = pythonic_str.split()
    total_tokens = len(tokens)
    mean_token_length = sum(len(t) for t in tokens) / total_tokens

    print(f"Prepared {total_tokens:,} tokens of {mean_token_length:.2f} mean length!")

    tokens = random.sample(tokens, 100)
    log_functionality(
        tokens,
        pythonic_str,
        stringzilla_str,
    )


_main_epilog = """
Examples:

  # Benchmark with a file
  %(prog)s --haystack-path leipzig1M.txt

  # Benchmark with synthetic data
  %(prog)s --haystack-pattern "hello world " --haystack-length 1000000
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla find operations",
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

    args = parser.parse_args()

    if args.haystack_path:
        if args.haystack_pattern or args.haystack_length:
            parser.error("Cannot specify both --haystack-path and synthetic options")
    else:
        if not (args.haystack_pattern and args.haystack_length):
            parser.error(
                "Must specify either --haystack-path or both --haystack-pattern and --haystack-length"
            )

    bench(args.haystack_path, args.haystack_pattern, args.haystack_length)


if __name__ == "__main__":
    main()
