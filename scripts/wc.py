import argparse
from stringzilla.compiled import Str, File, Slices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lines", nargs="*", help="Count lines in files")
    args = parser.parse_args()
    if args.lines:
        for filename in args.lines:
            print(File(filename).count("\n"))
