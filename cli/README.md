# SIMD-accelerated CLI utilities based on StringZilla

This section of the project is pretty much a work in progress.
The goal is to provide a set of command-line utilities that:

- ✅ that benefit the most from SIMD instructions, 
- ✅ rely solely on core StringZilla functionality,
- ✅ works the same on Linux, MacOS, and Windows.

Other utilities are, of course, welcome to use StringZilla but may not be good candidates for this repository.
To install, pull the Python package from PyPi:

```bash
pip install stringzilla
```

Currently implemented:

- `sz_wc`: 3x faster `wc` word count.
- `sz_split`: 4x faster `splt` file splitting.

What other interfaces should be added?
Levenshtein distances?
Fuzzy search?
Are there common alternatives to mimic?

## `wc`: Word Count

The `wc` utility on Linux can be used to count the number of lines, words, and bytes in a file.
Using SIMD-accelerated character and character-set search, StringZilla, even with slow SSDs, it can be noticeably faster.

```bash
$ time wc enwik9.txt
  13147025 129348346 1000000000 enwik9.txt

real    0m3.562s
user    0m3.470s
sys     0m0.092s

$ time sz_wc enwik9.txt
  13147025 139132610 1000000000 enwik9.txt # Note: different word count, WIP

real    0m1.165s
user    0m1.121s
sys     0m0.044s
```

## `split`: Split File into Smaller Ones

The `split` utility on Linux can be used to split a file into smaller ones.
The current prototype only splits by line counts.

```bash
$ time split -l 100000 enwik9.txt ...

real    0m6.424s
user    0m0.179s
sys     0m0.663s

$ time sz_split -l 100000 enwik9.txt ...

real    0m1.482s
user    0m1.020s
sys     0m0.460s
```
