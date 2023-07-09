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
    log("str.contains", bytes_length, lambda: pattern in pythonic_str)
    log("Str.contains", bytes_length, lambda: pattern in stringzilla_str)
    if stringzilla_file:
        log("File.contains", bytes_length, lambda: pattern in stringzilla_file)

    log("str.count", bytes_length, lambda: pythonic_str.count(pattern))
    log("Str.count", bytes_length, lambda: stringzilla_str.count(pattern))
    if stringzilla_file:
        log("File.count", bytes_length, lambda: stringzilla_file.count(pattern))

    log("str.split", bytes_length, lambda: pythonic_str.split(pattern))
    log("Str.split", bytes_length, lambda: stringzilla_str.split(pattern))
    if stringzilla_file:
        log("File.split", bytes_length, lambda: stringzilla_file.split(pattern))


def bench(
    needle: str,
    haystack_path: str = None,
    haystack_pattern: str = None,
    haystack_length: int = None,
):
    if haystack_path:
        pythonic_str: str = open(haystack_path, "r").read()
        stringzilla_file = File(haystack_path)
    else:
        haystack_length = int(haystack_length)
        repretitions = haystack_length // len(haystack_pattern)
        pythonic_str: str = haystack_pattern * repretitions
        stringzilla_file = None

    stringzilla_str = Str(pythonic_str)

    log_functionality(
        needle, len(stringzilla_str), pythonic_str, stringzilla_str, stringzilla_file
    )


if __name__ == "__main__":
    fire.Fire(bench)
