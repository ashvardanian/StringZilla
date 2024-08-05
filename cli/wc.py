#!/usr/bin/env python3

import sys, os
import argparse
import stringzilla
from stringzilla import File, Str


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Print newline, word, and byte counts for each FILE, and a total line if more than one FILE is \
        specified. A word is a non-zero-length sequence of acters delimited by white space."
    )
    parser.add_argument("files", nargs="*", default=["-"], help="Files to process")
    parser.add_argument(
        "-c", "--bytes", action="store_true", help="print the byte counts"
    )
    parser.add_argument(
        "-m", "--chars", action="store_true", help="print the character counts"
    )
    parser.add_argument(
        "-l", "--lines", action="store_true", help="print the newline counts"
    )
    parser.add_argument(
        "-L",
        "--max-line-length",
        action="store_true",
        help="print the maximum display width",
    )
    parser.add_argument(
        "-w", "--words", action="store_true", help="print the word counts"
    )
    parser.add_argument(
        "--files0-from",
        metavar="filename",
        help="Read input from the files specified by NUL-terminated names in file F;"
        " If F is - then read names from standard input",
    )

    parser.add_argument("--version", action="version", version=stringzilla.__version__)
    return parser.parse_args()


def wc(file_path, args):
    if file_path == "-":  # read from stdin
        content = sys.stdin.read()
        mapped_bytes = Str(content)
    else:
        try:
            mapped_file = File(file_path)
            mapped_bytes = Str(mapped_file)
        except RuntimeError:  # File gives a RuntimeError if the file does not exist
            return f"No such file: {file_path}", False

    counts = {}
    if args.lines:
        counts["line_count"] = mapped_bytes.count("\n")
    if args.words:
        counts["word_count"] = mapped_bytes.count(" ") + 1
    if args.chars:
        counts["char_count"] = mapped_bytes.__len__()

    if args.max_line_length:
        max_line_length = max(len(line) for line in mapped_bytes.split("\n"))
        counts["max_line_length"] = max_line_length

    if args.bytes:
        if args.chars:
            counts["byte_count"] = counts["char_count"]
        else:
            counts["byte_count"] = mapped_bytes.__len__()

    return counts, True


def format_output(counts, args, just):
    selected_counts = []
    if args.lines:
        selected_counts.append(counts["line_count"])
    if args.words:
        selected_counts.append(counts["word_count"])
    if args.chars:
        selected_counts.append(counts["char_count"])
    if args.bytes:
        selected_counts.append(counts["byte_count"])
    if args.max_line_length:
        selected_counts.append(counts.get("max_line_length", 0))

    return " ".join(str(count).rjust(just) for count in selected_counts)


def get_files_from(fn):
    f = open(fn, "r")
    s = f.read()
    f.close()
    return [x for x in s.split("\0") if os.path.isfile(x)]


def main():
    args = parse_arguments()
    total_counts = {
        "line_count": 0,
        "word_count": 0,
        "char_count": 0,
        "max_line_length": 0,
        "byte_count": 0,
    }
    if not any([args.lines, args.words, args.chars, args.bytes, args.max_line_length]):
        args.lines = 1
        args.words = 1
        args.bytes = 1

    # wc uses the file size to determine column width when printing
    if args.files0_from:
        if args.files[0] == "-":
            args.files = get_files_from(args.files0_from)
            if len(args.files) == 0:
                # print("  No filenames found in ", args.files0_from)
                exit(0)

    just = max(len(str(os.stat(fn).st_size)) for fn in args.files)

    for file_path in args.files:
        counts, success = wc(file_path, args)
        if success:
            for key in total_counts.keys():
                total_counts[key] += counts.get(key, 0)
            output = format_output(counts, args, just) + f" {file_path}"
            print(output)
        else:
            print(counts)

    if len(args.files) > 1:
        total_output = format_output(total_counts, args, just) + " total"
        print(total_output)


if __name__ == "__main__":
    main()
