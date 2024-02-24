#!/usr/bin/env python3

import argparse
import sys
import stringzilla
from stringzilla import File, Str


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Output pieces of FILE to PREFIXaa, PREFIXab, ...; default size is 1000 lines, and default PREFIX is 'x'."
    )
    parser.add_argument(
        "file", nargs="?", default="-", help='File to process, "-" for standard input'
    )
    parser.add_argument(
        "prefix", nargs="?", default="X", help='Output file prefix, default is "x"'
    )
    parser.add_argument(
        "-l",
        "--lines",
        type=int,
        default=1000,
        help="Number of lines per output file, default is 1000",
    )
    parser.add_argument(
        "-t",
        "--separator",
        default="\n",
        help="Use SEP instead of newline as the record separator; '\\0' (zero) specifies the NUL character",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=None,
        help="Generate N output files based on size of input",
    )
    parser.add_argument("--version", action="version", version=stringzilla.__version__)
    return parser.parse_args()


def split_file(file_path, lines_per_file, output_prefix, separator, number_of_files):
    try:
        if separator == "\\0":
            separator = "\0"
        if file_path == "-":
            file_contents = Str(sys.stdin.read())
        else:
            file_mapped = File(file_path)
            file_contents = Str(file_mapped)

        if number_of_files is not None:
            total_length = len(file_contents)
            chunk_size = total_length // number_of_files
            for file_part in range(number_of_files):
                start = file_part * chunk_size
                end = (
                    start + chunk_size
                    if file_part < number_of_files - 1
                    else total_length
                )
                current_slice = file_contents[start:end]
                output_path = f"{output_prefix}{file_part}"
                current_slice.write_to(output_path)
            return
        current_position = 0
        file_part = 0
        newline_position = -1

        while current_position < len(file_contents):
            for _ in range(lines_per_file):
                newline_position = file_contents.find(separator, newline_position + 1)
                if newline_position == -1:
                    break

            if newline_position == -1 and current_position < len(file_contents):
                newline_position = len(file_contents)

            section_length = (
                newline_position - current_position if newline_position != -1 else 0
            )

            if section_length > 0:
                current_slice = file_contents[current_position : newline_position + 1]
                output_path = f"{output_prefix}{file_part}"
                current_slice.write_to(output_path)

                file_part += 1
                current_position = newline_position + 1

    except FileNotFoundError:
        print(f"No such file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Usage example: split.py [-l LINES] [file] [prefix]")


def main():
    args = parse_arguments()
    split_file(args.file, args.lines, args.prefix, args.separator, args.number)


if __name__ == "__main__":
    main()
