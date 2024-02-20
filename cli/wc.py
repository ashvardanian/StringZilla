#!/usr/bin/env python3

import sys

from stringzilla import File, Str


def wc(file_path):
    try:
        mapped_file = File(file_path)
        mapped_bytes = Str(mapped_file)
        line_count = mapped_bytes.count("\n")
        word_count = mapped_bytes.count(" ")
        char_count = mapped_bytes.__len__()

        return line_count, word_count, char_count
    except FileNotFoundError:
        return f"No such file: {file_path}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python wc.py <file>")
        sys.exit(1)

    file_path = sys.argv[1]
    counts = wc(file_path)

    if isinstance(counts, tuple):
        line_count, word_count, char_count = counts
        print(f"{line_count} {word_count} {char_count} {file_path}")
    else:
        print(counts)


if __name__ == "__main__":
    main()
