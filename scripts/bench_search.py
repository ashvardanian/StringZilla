import time
import re
import random
from typing import List

import fire

from stringzilla import Str


def log(name: str, haystack, patterns, operator: callable):
    a = time.time_ns()
    for pattern in patterns:
        operator(haystack, pattern)
    b = time.time_ns()
    bytes_length = len(haystack) * len(patterns)
    secs = (b - a) / 1e9
    gb_per_sec = bytes_length / (1e9 * secs)
    print(f"{name}: took {secs:} seconds ~ {gb_per_sec:.3f} GB/s")


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


def log_functionality(
    tokens: List[str],
    pythonic_str: str,
    stringzilla_str: Str,
):
    log("str.find", pythonic_str, tokens, find_all)
    log("Str.find", stringzilla_str, tokens, find_all)
    log("str.rfind", pythonic_str, tokens, rfind_all)
    log("Str.rfind", stringzilla_str, tokens, rfind_all)
    log("re.finditer", pythonic_str, [r" \t\n\r"], find_all_regex)
    log("Str.find_first_of", stringzilla_str, [r" \t\n\r"], find_all_sets)


def bench(
    haystack_path: str = None,
    haystack_pattern: str = None,
    haystack_length: int = None,
):
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

    print(
        f"Parsed the file with {total_tokens:,} words of {mean_token_length:.2f} mean length!"
    )

    tokens = random.sample(tokens, 100)
    log_functionality(
        tokens,
        pythonic_str,
        stringzilla_str,
    )


if __name__ == "__main__":
    fire.Fire(bench)
