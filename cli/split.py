#!/usr/bin/env python3

import sys

from stringzilla import File, Str


def split_file(file_path, lines_per_file, output_prefix):
    try:
        # 1. Memory-map the large file
        file_mapped = File(file_path)
        file_contents = Str(file_mapped)

        # Variables to keep track of the current position and file part number
        current_position = 0
        file_part = 0
        newline_position = (
            -1
        )  # Start before file begins to find the first newline correctly

        # Loop until the end of the file
        while current_position < len(file_contents):
            # 2. Loop to skip `lines_per_file` lines
            for _ in range(lines_per_file):
                newline_position = file_contents.find("\n", newline_position + 1)
                if newline_position == -1:  # No more newlines
                    break

            # If no newlines were found and we're not at the start, process the rest of the file
            if newline_position == -1 and current_position < len(file_contents):
                newline_position = len(file_contents)

            # 3. Use offset_within to get the length of the current section
            # Assuming offset_within gives you the length from the current position
            section_length = (
                newline_position - current_position if newline_position != -1 else 0
            )

            # Extract the current section to write out
            if section_length > 0:  # Prevent creating empty files
                current_slice = file_contents[current_position : newline_position + 1]

                # 4. Save the current slice to file
                output_path = f"{output_prefix}{file_part}"
                current_slice.write_to(output_path)

                # Prepare for the next slice
                file_part += 1
                current_position = newline_position + 1

    except FileNotFoundError:
        print(f"No such file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python split_file.py <lines_per_file> <input_file> <output_prefix>"
        )
        sys.exit(1)

    lines_per_file = int(sys.argv[1])
    file_path = sys.argv[2]
    output_prefix = sys.argv[3]

    split_file(file_path, lines_per_file, output_prefix)


if __name__ == "__main__":
    main()
