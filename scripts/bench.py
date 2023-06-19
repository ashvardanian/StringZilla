import time

import fire

from stringzilla.compiled import Str, File


def log_duration(name: str, func: callable):
    a = time.time_ns()
    func()
    b = time.time_ns()
    secs = (b - a) / 1e9
    print(f"{name}: took {secs:} seconds")


def log_functionality(
    pattern: str, pythonic_str: str, stringzilla_str: Str, stringzilla_file: File
):
    log_duration("Find match in Python", lambda: pattern in pythonic_str)
    log_duration("Find match in Stringzilla", lambda: pattern in stringzilla_str)
    log_duration("Find match in Stringzilla File", lambda: pattern in stringzilla_file)

    log_duration("Count matches in Python", lambda: pythonic_str.count(pattern))
    log_duration("Count matches in Stringzilla", lambda: stringzilla_str.count(pattern))
    log_duration(
        "Count matches in Stringzilla File", lambda: stringzilla_file.count(pattern)
    )

    log_duration("Split Python", lambda: pythonic_str.split(pattern))
    log_duration("Split Stringzilla", lambda: stringzilla_str.split(pattern))
    log_duration("Split Stringzilla File", lambda: stringzilla_file.split(pattern))


def bench(path: str, pattern: str):
    pythonic_str: str = open(path, "r").read()
    stringzilla_str = Str(pythonic_str)
    stringzilla_file = File(path)

    log_functionality(pattern, pythonic_str, stringzilla_str, stringzilla_file)


if __name__ == "__main__":
    fire.Fire(bench)
