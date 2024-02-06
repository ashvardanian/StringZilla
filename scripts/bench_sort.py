import time

import fire

from stringzilla import Str, File


def log(name: str, bytes_length: int, operator: callable):
    a = time.time_ns()
    operator()
    b = time.time_ns()
    secs = (b - a) / 1e9
    gb_per_sec = bytes_length / (1e9 * secs)
    print(f"{name}: took {secs:} seconds ~ {gb_per_sec:.3f} GB/s")


def log_functionality(
    pattern: str,
    bytes_length: int,
    pythonic_str: str,
    stringzilla_str: Str,
    stringzilla_file: File,
):
    log("str.split", bytes_length, lambda: pythonic_str.split(pattern))
    log("Str.split", bytes_length, lambda: stringzilla_str.split(pattern))
    if stringzilla_file:
        log("File.split", bytes_length, lambda: stringzilla_file.split(pattern))

    log("str.split.sort", bytes_length, lambda: pythonic_str.split(pattern).sort())
    log("Str.split.sort", bytes_length, lambda: stringzilla_str.split(pattern).sort())
    if stringzilla_file:
        log("File.split", bytes_length, lambda: stringzilla_file.split(pattern).sort())


def bench(
    haystack_path: str = None,
    haystack_pattern: str = None,
    haystack_length: int = None,
    needle: str = None,
):
    if haystack_path:
        pythonic_str: str = open(haystack_path, "r").read()
        stringzilla_file = File(haystack_path)
    else:
        haystack_length = int(haystack_length)
        repetitions = haystack_length // len(haystack_pattern)
        pythonic_str: str = haystack_pattern * repetitions
        stringzilla_file = None

    stringzilla_str = Str(pythonic_str)

    log_functionality(
        needle, len(stringzilla_str), pythonic_str, stringzilla_str, stringzilla_file
    )


if __name__ == "__main__":
    fire.Fire(bench)
