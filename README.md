# Stringzilla

## Crunch 100+ GB Strings in Python with ease, leveraging SIMD Assembly

Stringzilla was born many years ago as a tutorial for SIMD accelerated string-processing.
But one day, processing 100 GB+ Chemistry and AI single-file datasets, I have had enough.

| Benchmark                |   IoT    |  Laptop  |  Server   |
| :----------------------- | :------: | :------: | :-------: |
| Python: `str.find`       |  4 MB/s  | 14 MB/s  |  11 MB/s  |
| C++: `std::string::find` | 560 MB/s | 1,2 GB/s | 1,3 GB/s  |
| Stringzilla              | 4,3 Gb/s | 12 GB/s  | 12.1 GB/s |

The native `open(...).readlines()` and `str().splitlines()` are ridiculously slow.
I have looked for a solution in PyPi and found nothing.
So I took my SIMD tutorial and transformed into a Python package.

## Installation

- For Python: `pip install stringzilla`
- For Conan C++ users.

## Usage


## Development

```sh
pip install -e . && pytest test.py -s -x
```