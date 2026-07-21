# StringZilla for Python

StringZilla for Python wraps SIMD- and SWAR-accelerated native kernels behind `str`- and `bytes`-shaped types that avoid copies wherever possible.
The `stringzilla` module covers single-string search, slicing, splitting, trimming, translation, hashing, checksums, sorting, sampling, random generation, UTF-8 segmentation, Unicode case-folding, and Unicode normalization.
The companion `stringzillas` module covers batch-parallel edit distances, alignment scores, and rolling fingerprints spread across CPU cores and CUDA devices.

Every kernel selects the fastest backend for the running CPU at import time, and the batched engines release the GIL around native work.
The module is also marked safe for free-threaded `Py_GIL_DISABLED` CPython builds.

Throughout this document the `stringzilla` package is imported as `sz`, the `stringzillas` package as `szs`, and NumPy as `np`:

```python
import stringzilla as sz
import stringzillas as szs
import numpy as np
```

Almost every `sz` operation is available both as a method on a `Str` object and as a module-level function that accepts a `str`, `bytes`, `bytearray`, or `Str`.
For example `sz.Str("hello").find("l")` and `sz.find("hello", "l")` are equivalent.

## Installation

StringZilla ships on PyPI as the single-string module plus an optional batch-engine package:

```sh
pip install stringzilla        # single-string kernels: the `stringzilla` module
pip install stringzillas-cpus  # batch engines on CPU cores: the `stringzillas` module
pip install stringzillas-cuda  # batch engines on NVIDIA GPUs: the `stringzillas` module
```

Install `stringzilla` for the single-string API.
Install exactly one of `stringzillas-cpus` or `stringzillas-cuda` for the batch engines; both expose the same `import stringzillas` package, differing only in whether CUDA kernels are compiled in.
The batch engines accept `sz.Strs` collections and NumPy arrays, so NumPy is required to use them.

## Types

The `stringzilla` module exposes five public types: `Str`, `Strs`, `File`, the incremental hashers `Hasher` and `Sha256`, plus a family of UTF-8 iterator types.

### `Str`

`Str(source)` is an immutable, zero-copy view over a `str`, `bytes`, `bytearray`, another `Str`, or a memory-mapped `File`.
It implements the read-only half of the `str`/`bytes` protocol — indexing, slicing, iteration, `len`, `in`, rich comparison, hashing, and the buffer protocol — without materializing new Python objects.
Slicing returns a new `Str` view into the same backing buffer rather than copying bytes.

Two read-only properties expose its memory layout for zero-copy interop with PyArrow, `ctypes`, and similar:

- `Str.address` — integer memory address of the first byte.
- `Str.nbytes` — length of the view in bytes.

```python
import stringzilla as sz

text = sz.Str("the quick brown fox")
assert str(text[4:9]) == "quick"   # a zero-copy view
assert str(text[10:15]) == "brown"
assert "quick" in text
assert len(text) == 19
assert text.nbytes == 19
assert isinstance(text.address, int)
```

### `Strs`

`Strs(sequence, view=False)` is an ordered, indexable, sliceable collection of `Str` views.
It is produced by the splitting and sorting methods, and can also be built directly from a `list`, `tuple`, generator, or a `pyarrow.Array`.
With `view=True` it references the original data instead of copying it.


`Strs` keeps its parts on a single contiguous "tape" with an offsets array, so `sorted`, `argsort`, `sample`, and `shuffled` reorder offsets rather than bytes.
This layout is also dramatically more memory-efficient than a Python `list` of `str`.
Every part lives on one shared byte tape alongside a compact offsets array, so N substrings cost roughly one allocation, whereas a `list` of `str` holds N separately heap-allocated objects, each carrying a PyObject header, a hash cache, and per-object allocator overhead.
Splitting a large document into millions of tokens therefore stays compact and cache-friendly instead of fragmenting the heap.
The lazy `split_iter`, `rsplit_iter`, and `*_byteset_iter` forms go a step further and build no collection at all, yielding one `Str` view at a time.
It carries one of three internal storage layouts:

- `TAPE` — owns a contiguous data buffer with an offsets array; StringTape-compatible.
- `TAPE_VIEW` — a zero-copy view into existing data, such as an Arrow or StringTape slice.
- `FRAGMENTED` — non-contiguous strings with individual pointers.

Several read-only properties expose the tape for Apache Arrow / StringTape interop:

| Property                 | Meaning                                    |
| ------------------------ | ------------------------------------------ |
| `Strs.tape`              | In-place transform to the Arrow layout.    |
| `Strs.tape_address`      | Address of the tape buffer's first byte.   |
| `Strs.tape_nbytes`       | Total tape length in bytes.                |
| `Strs.offsets_address`   | Address of the offsets array's first byte. |
| `Strs.offsets_nbytes`    | Offsets array length in bytes.             |
| `Strs.offsets_are_large` | `True` when 64-bit offsets are needed.     |
| `Strs.__layout__`        | Internal layout name.                      |

`Strs.tape` rewrites the collection in place into the Arrow string layout — a contiguous buffer plus an offsets array.
`Strs.offsets_are_large` is `True` when the tape is too large for 32-bit offsets and needs 64-bit ones for Arrow export.
`Strs.__layout__` is a debug string reporting the internal storage: `TAPE`, `TAPE_VIEW`, or `FRAGMENTED`.

```python
import stringzilla as sz

names = sz.Strs(["banana", "apple", "cherry", "date"])
assert len(names) == 4
assert str(names[0]) == "banana"
assert [str(x) for x in names[1:3]] == ["apple", "cherry"]   # a sliced view
names.__layout__                 # e.g. 'TAPE'
```

### `File`

`File(path, mode='r')` memory-maps a path so a `Str` can view data larger than RAM without reading it into memory.
The default `mode` is `'r'`, for read-only access.

```python
import os, tempfile, pathlib
import stringzilla as sz

path = os.path.join(tempfile.mkdtemp(), "log.txt")
pathlib.Path(path).write_text("error: disk full\nok\nerror: timeout\n")

mapped = sz.Str(sz.File(path))   # scan in place, no copy into RAM
assert mapped.count("error") == 2
```

## Searching and Counting

All search methods accept optional `start` and `end` indices that bound the search window; `start` defaults to `0` and `end` to the string length.
They are available both as `Str` methods and as module-level functions.

### Substring Search

- `find(substring, start=0, end=len)` — index of the first occurrence, or `-1` if absent.
- `rfind(substring, start=0, end=len)` — index of the last occurrence, or `-1`.
- `index(substring, start=0, end=len)` — like `find`, but raises `ValueError` if absent.
- `rindex(substring, start=0, end=len)` — like `rfind`, but raises `ValueError` if absent.
- `contains(substring, start=0, end=len)` — `True`/`False` membership test.
- `startswith(prefix, start=0, end=len)` — `True` if the windowed string starts with `prefix`.
- `endswith(suffix, start=0, end=len)` — `True` if the windowed string ends with `suffix`.
- `count(substring, start=0, end=len, allowoverlap=False)` — number of occurrences; pass `allowoverlap=True` to count overlapping matches.

```python
import stringzilla as sz

text = sz.Str("the quick brown fox")
assert text.find("brown") == 10
assert text.rfind("o") == 17
assert text.index("quick") == 4
assert text.contains("quick")
assert text.startswith("the")
assert text.endswith("fox")
assert sz.Str("banana").count("a") == 3
assert sz.Str("aaaa").count("aa") == 2                     # non-overlapping
assert sz.Str("aaaa").count("aa", allowoverlap=True) == 3  # overlapping
```

### Byteset Search

The `*_of` / `*_not_of` family searches for any byte from a set, the byteset-accelerated counterpart of substring search.
`count_byteset` counts how many bytes fall in the set.

- `find_first_of(chars, start=0, end=len)` — index of the first byte that is in `chars`, or `-1`.
- `find_last_of(chars, start=0, end=len)` — index of the last byte that is in `chars`, or `-1`.
- `find_first_not_of(chars, start=0, end=len)` — index of the first byte not in `chars`, or `-1`.
- `find_last_not_of(chars, start=0, end=len)` — index of the last byte not in `chars`, or `-1`.
- `count_byteset(chars, start=0, end=len)` — number of bytes that are in `chars`.

```python
import stringzilla as sz

text = sz.Str("hello world")
assert text.find_first_of("aeiou") == 1
assert text.find_last_of("aeiou") == 7
assert text.find_first_not_of("he") == 2
assert text.find_last_not_of("lo") == 10
assert text.count_byteset("lo") == 5
```

### Equality and Offsets

- `sz.equal(first, second)` — module-level fast byte-wise equality, returning `bool`.
- `offset_within(larger)` — returns the byte offset of this `Str` view inside a larger `Str` it was sliced from, or `-1` if it is not a sub-view.

```python
import stringzilla as sz

assert sz.equal("abc", "abc")
big = sz.Str("hello world")
assert big[6:].offset_within(big) == 6
```

## Splitting and Partitioning

### Eager Splits Returning `Strs`

- `split(separator, maxsplit=∞, keepseparator=False, skip_empty=False)` — split on a non-empty substring separator; raises `ValueError` on an empty separator.
- `rsplit(separator, maxsplit=∞, keepseparator=False, skip_empty=False)` — same, scanning from the right.
- `split_byteset(separators, maxsplit=∞, keepseparator=False, skip_empty=False)` — split on any single byte from the `separators` set.
- `rsplit_byteset(separators, maxsplit=∞, keepseparator=False, skip_empty=False)` — same, from the right.
- `splitlines(keeplinebreaks=False, maxsplit=∞)` — split on line breaks.

`maxsplit` caps the number of splits and defaults to unlimited.
`keepseparator` keeps the matched separator attached to the parts.
`skip_empty` drops empty segments between adjacent separators.

```python
import stringzilla as sz

assert list(map(str, sz.Str("a,b,c").split(","))) == ["a", "b", "c"]
assert list(map(str, sz.Str("a,b,c").rsplit(",", maxsplit=1))) == ["a,b", "c"]
assert list(map(str, sz.Str("a,b;c").split_byteset(",;"))) == ["a", "b", "c"]
assert list(map(str, sz.Str("a\nb\nc").splitlines())) == ["a", "b", "c"]
```

### Lazy Split Iterators

Each eager split has a lazy counterpart that yields `Str` views one at a time without building the whole `Strs`.
These take `keepseparator=False` and `skip_empty=False` but no `maxsplit`.

- `split_iter(separator, keepseparator=False, skip_empty=False)`
- `rsplit_iter(separator, keepseparator=False, skip_empty=False)`
- `split_byteset_iter(separators, keepseparator=False, skip_empty=False)`
- `rsplit_byteset_iter(separators, keepseparator=False, skip_empty=False)`

The reverse iterators yield the last field first.

```python
import stringzilla as sz

for field in sz.Str("2024-01-15").split_iter("-"):
    print(field)                                       # 2024, then 01, then 15

assert str(next(iter(sz.Str("a/b/c").rsplit_iter("/")))) == "c"
assert sum(1 for _ in sz.Str("a,b,c").split_iter(",")) == 3
```

### Partitioning

- `partition(separator)` — return a 3-tuple `(head, separator, tail)` split at the first occurrence; if not found returns `(self, '', '')`.
- `rpartition(separator)` — same, at the last occurrence; if not found returns `('', '', self)`.

```python
import stringzilla as sz

assert tuple(map(str, sz.Str("a=b=c").partition("="))) == ("a", "=", "b=c")
assert tuple(map(str, sz.Str("a=b=c").rpartition("="))) == ("a=b", "=", "c")
```

## Trimming and Translating

### Trimming

- `strip(chars=whitespace)` — remove leading and trailing bytes that are in `chars`.
- `lstrip(chars=whitespace)` — remove only leading bytes.
- `rstrip(chars=whitespace)` — remove only trailing bytes.

With no argument, the default character set is ASCII whitespace.
Pass a byteset to trim an arbitrary set of bytes.

```python
import stringzilla as sz

assert sz.Str("  hi  ").strip() == "hi"
assert sz.Str("xxhi").lstrip("x") == "hi"
assert sz.Str("hi!!").rstrip("!") == "hi"
```

### Translating

`translate(table, inplace=False, start=0, end=len)` applies a byte-to-byte mapping in a single pass.
`table` is either a 256-byte string/bytes lookup table or a `dict` mapping bytes to bytes.
With `inplace=True` the buffer is rewritten in place and `None` is returned; otherwise a new `bytes`/`str` is returned.
A non-256-byte table raises `ValueError`; a non-string, non-dict table raises `TypeError`.

```python
import stringzilla as sz

assert sz.Str("abc").translate({"a": "A"}) == b"Abc"

table = bytes(range(256)).translate(bytes.maketrans(b"abc", b"ABC"))
assert sz.Str("cabbage").translate(table) == b"CABBAge"
```

### Decoding

`decode(encoding='utf-8', errors='strict')` decodes the bytes to a Python `str`, mirroring `bytes.decode`.

```python
import stringzilla as sz

assert sz.Str("abc").decode() == "abc"
```

## Hashing and Checksums

### One Shot Hashes and Checksums

- `sz.hash(text, seed=0)` — seeded 64-bit AES-accelerated hash, returned as an unsigned `int`. This differs from Python's built-in `hash()`, which returns a platform-dependent `Py_hash_t`.
- `sz.hash_multiseed(text, seeds, out=None)` — hash one string under many seeds at once. `seeds` is a contiguous buffer of `uint64`, such as a `numpy.uint64` array or `array('Q', ...)`; plain `int` lists are not accepted. Returns a tuple of ints, or fills the optional contiguous `uint64` `out` buffer in place and returns `None`. This is much faster than looping `hash` for short strings under many seeds, and is useful for feature hashing, Count-Min sketches, Bloom/cuckoo filters, and MinHash/LSH.
- `sz.bytesum(text)` — additive checksum of the individual byte values, as an `int`.
- `sz.sha256(text)` — 32-byte SHA-256 digest as `bytes`.
- `sz.hmac_sha256(key, message)` — 32-byte HMAC-SHA256 digest as `bytes`.

`hash`, `hash_multiseed`, `bytesum`, and `sha256` are also available as `Str` methods.

```python
import stringzilla as sz
import numpy as np

sz.hash("hello")                  # seeded 64-bit value
sz.hash("hello", seed=42)         # a different value
assert sz.bytesum("abc") == 294
assert sz.sha256("abc").hex()[:8] == "ba7816bf"
assert len(sz.hmac_sha256(b"key", b"message")) == 32

seeds = np.arange(8, dtype=np.uint64)
assert len(sz.hash_multiseed("hello", seeds)) == 8   # tuple of 8 ints
```

### Incremental 64-bit `Hasher`

`Hasher(seed=0)` is an incremental, AES-accelerated hasher producing the same 64-bit value as `sz.hash`.

- `update(data)` — absorb a `str`/`bytes` chunk; returns `self` for chaining.
- `digest()` — current hash as an unsigned 64-bit `int`, without consuming state.
- `hexdigest()` — current hash as a 16-character lowercase hex `str`.
- `reset()` — reset to the initial seed; returns `self`.

```python
import stringzilla as sz

h = sz.Hasher()
h.update(b"hello").update(b" world")
assert h.digest() == sz.hash(b"hello world")
assert len(h.hexdigest()) == 16
```

### Incremental `Sha256` Compatible with `hashlib`

`Sha256()` is an incremental SHA-256 hasher with a `hashlib`-compatible interface, hardware-accelerated where available.

- `update(data)` — absorb a `str`/`bytes` chunk; returns `self`.
- `digest()` — current 32-byte digest as `bytes`.
- `hexdigest()` — current digest as a 64-character lowercase hex `str`.
- `reset()` — reset to the initial SHA-256 constants; returns `self`.
- `copy()` — return an independent copy with identical internal state.

```python
import stringzilla as sz
import hashlib

assert sz.Sha256().update(b"abc").hexdigest() == hashlib.sha256(b"abc").hexdigest()

a = sz.Sha256().update(b"ab")
assert a.copy().update(b"c").hexdigest() == sz.Sha256().update(b"abc").hexdigest()
```

## Sorting, Sampling, and Shuffling

These operate on a `Strs` collection and return new collections, leaving the original unchanged.

### Sorting

- `sorted(reverse=False, uncased=False, top=None)` — return a new, stably sorted `Strs`. `reverse` sorts descending, `uncased` orders by Unicode case-folding, and `top` keeps only the leading `top` elements (a partial top-k, cheaper than a full sort).
- `argsort(reverse=False, uncased=False, top=None, out=None)` — return the stable permutation of indices that sorts the collection, as a tuple of ints. `out` is an optional writable, C-contiguous 64-bit-unsigned buffer such as `numpy.uintp` or `array('Q')`, receiving the indices with zero allocation; when given, `out` itself is returned.

```python
import stringzilla as sz

names = sz.Strs(["banana", "apple", "cherry"])
assert [str(x) for x in names.sorted()] == ["apple", "banana", "cherry"]
assert [str(x) for x in names.sorted(reverse=True)] == ["cherry", "banana", "apple"]
assert [str(x) for x in names.sorted(top=2)] == ["apple", "banana"]
assert names.argsort() == (1, 0, 2)
```

### Sampling and Shuffling

- `sample(size, seed=None)` — return a new `Strs` of `size` elements drawn at random with replacement; `seed` makes it reproducible.
- `shuffled(seed=None)` — return a new `Strs` with the elements randomly permuted; `seed` makes it reproducible.

```python
import stringzilla as sz

pool = sz.Strs(["a", "b", "c", "d"])
assert len(pool.sample(2)) == 2
assert [str(x) for x in sorted(pool.shuffled(seed=42))] == ["a", "b", "c", "d"]
```

### Random Byte Generation

These module-level functions generate random bytes and are not tied to `Strs`:

- `sz.random(length, nonce=0, alphabet=None)` — return a fresh random `bytes` of `length`; if `alphabet` is given, each byte `b` is mapped to `alphabet[b % len(alphabet)]`.
- `sz.fill_random(buffer, nonce=0, alphabet=None, start=0, end=len)` — fill a writable, contiguous byte buffer such as a `bytearray`, `memoryview`, or `Str` in place with pseudo-random bytes, optionally remapped to `alphabet`, optionally limited to the `[start, end)` slice. Returns `None`. Also available as the `Str.fill_random` method.

```python
import stringzilla as sz

assert len(sz.random(16)) == 16
sz.random(8, alphabet="ACGT")                 # e.g. b'GCTAACGT'

buf = bytearray(16)
sz.fill_random(buf)                           # mutates buf in place
assert len(buf) == 16
```

## Edit Distances and Alignment Scores

The `stringzillas` module scores many string pairs at once across CPU cores or a CUDA device.
Each engine is constructed once with its cost model, then _called_ over two collections to produce a dense cross-product result matrix.

### Execution Model: `DeviceScope`

`DeviceScope(cpu_cores=None, gpu_device=None)` controls where work runs.
Pass `cpu_cores=N` to restrict to `N` CPU cores, where `0` means all cores, or `gpu_device=ID` to target a CUDA device; you cannot pass both.
A `DeviceScope` may be supplied at construction time as the `capabilities` argument, and/or per call as the `device` argument.

```python
import stringzillas as szs

cpu_scope = szs.DeviceScope(cpu_cores=4)
gpu_scope = szs.DeviceScope(gpu_device=0)   # requires stringzillas-cuda
```

### Call Convention

Every engine is called as `engine(queries, candidates=None, device=None, out=None)`:

- `queries` — the collection forming the matrix rows, either a `sz.Strs` or any string sequence.
- `candidates` — the collection forming the matrix columns; when omitted or `None`, the engine computes the symmetric self-similarity of `queries`.
- `device` — an optional `DeviceScope` overriding the constructor's.
- `out` — an optional pre-allocated 2-D NumPy output buffer of shape `(len(queries), len(candidates))`.

The result `result[i, j]` is the distance or score between `queries[i]` and `candidates[j]`.

### `LevenshteinDistances` and `LevenshteinDistancesUTF8`

`LevenshteinDistances(match=0, mismatch=1, open=1, extend=1, capabilities=None)` computes byte-level edit distances with affine gap penalties.
`LevenshteinDistancesUTF8(...)` takes the same arguments but counts edits over UTF-8 codepoints instead of raw bytes.
The result matrix is `uint64`.

| Argument       | Default | Meaning                                                                                         |
| -------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `match`        | `0`     | Cost of a matching character.                                                                   |
| `mismatch`     | `1`     | Cost of a substitution.                                                                         |
| `open`         | `1`     | Cost of opening a gap, meaning an insertion or deletion run.                                    |
| `extend`       | `1`     | Cost of extending a gap.                                                                        |
| `capabilities` | `None`  | A capabilities tuple like `('serial', 'parallel')`, or a `DeviceScope` for automatic inference. |

```python
import stringzilla as sz
import stringzillas as szs

engine = szs.LevenshteinDistances()
a = sz.Strs(["hello", "world"])
b = sz.Strs(["hallo", "word"])
distances = engine(a, b)              # 2x2 uint64 matrix
assert distances.shape == (2, 2)
assert distances[0, 0] == 1           # hello -> hallo

# UTF-8 codepoint distances
utf8_engine = szs.LevenshteinDistancesUTF8(mismatch=5)
utf8_engine(sz.Strs(["café", "naïve"]), sz.Strs(["caffe", "naive"]))
```

### `NeedlemanWunschScores` and `SmithWatermanScores`

`NeedlemanWunschScores(byte_to_class, class_substitution_costs, open=-1, extend=-1, capabilities=None)` computes global alignment scores.
`SmithWatermanScores(...)` takes identical arguments and computes local alignment scores.
Both score from a class-based substitution matrix, and the result matrix is `int64`.

| Argument                   | Meaning                                                                   |
| -------------------------- | ------------------------------------------------------------------------- |
| `byte_to_class`            | A 256-element `uint8` NumPy array mapping each byte to one of 32 classes. |
| `class_substitution_costs` | A `32x32` `int8` NumPy matrix of costs between classes.                   |
| `open`                     | Gap-open cost, defaulting to `-1`.                                        |
| `extend`                   | Gap-extend cost, defaulting to `-1`.                                      |
| `capabilities`             | A capabilities tuple or a `DeviceScope`.                                  |

```python
import stringzilla as sz
import stringzillas as szs
import numpy as np

classes = (np.arange(256) % 32).astype(np.uint8)
costs = np.eye(32, dtype=np.int8)          # reward exact class matches

global_engine = szs.NeedlemanWunschScores(classes, costs)
global_engine(sz.Strs(["ACGT", "TGCA"]), sz.Strs(["ACCT", "TGAA"]))

local_engine = szs.SmithWatermanScores(classes, costs, open=-3, extend=-1)
local_engine(sz.Strs(["ACGTACGT"]), sz.Strs(["CGTACGTA"]))
```

### Selecting a GPU

Pass a GPU `DeviceScope` at construction or per call to run on CUDA, which requires `stringzillas-cuda`:

```python
gpu = szs.DeviceScope(gpu_device=0)
engine = szs.LevenshteinDistances(match=0, mismatch=2, open=3, extend=1, capabilities=gpu)
distances = engine(a, b, device=gpu)
```

Each engine also exposes a read-only `__capabilities__` property reporting the backends it selected at runtime.
The module-level `szs.to_device(strs)` converts a `sz.Strs` to use a unified/device-accessible allocator, forcing the allocator swap that normally happens during GPU kernel execution.

## Rolling Fingerprints

`Fingerprints(ndim, window_widths=None, alphabet_size=256, seed=0, capabilities=None)` computes MinHash-style rolling sketches over many documents in parallel.
The sketches drive near-duplicate detection, clustering, and multi-pattern search, and share the same device model as the alignment engines.

| Argument        | Default  | Meaning                                      |
| --------------- | -------- | -------------------------------------------- |
| `ndim`          | required | Dimensions per fingerprint.                  |
| `window_widths` | `None`   | 1-D `uint64` array of rolling-window widths. |
| `alphabet_size` | `256`    | Alphabet size.                               |
| `seed`          | `0`      | Reproducibility seed.                        |
| `capabilities`  | `None`   | Capabilities tuple or `DeviceScope`.         |

`window_widths` is a 1-D contiguous `uint64` NumPy array; built-in defaults are used when `None`.
`alphabet_size` is `256` for arbitrary binary strings.
`seed` derives independent per-dimension multipliers and moduli for MinHash independence; `0` is just the default seed, not a special mode.
The `capabilities` tuple may include `'cuda'`.

The engine is called as `engine(texts, device=None, out=None)` and returns a tuple `(hashes, counts)` of two `uint32` NumPy matrices, each of shape `(len(texts), ndim)`.
`hashes` holds the minimal hash per dimension; `counts` holds the corresponding match counts.
`Fingerprints` exposes its selected backends through a read-only `capabilities` property — note that this name lacks the leading and trailing underscores used by the alignment engines.

```python
import stringzilla as sz
import stringzillas as szs
import numpy as np

engine = szs.Fingerprints(ndim=128)
docs = sz.Strs(["document one", "document two", "document three"])
hashes, counts = engine(docs)
assert hashes.shape == (3, 128)

# Custom window widths and a fixed seed
widths = np.array([4, 8, 16], dtype=np.uint64)
engine = szs.Fingerprints(ndim=256, window_widths=widths, seed=7)
hashes, counts = engine(docs)
```

## UTF-8 Segmentation

`Str`, along with the matching module-level functions, exposes lazy iterators over Unicode boundaries.
Each yields `Str` views into the original buffer, so segmentation stays allocation-free, except `utf8_codepoints`, which yields `int` code points.

| Method / function                                      | Standard                         | Yields                                                                                 |
| ------------------------------------------------------ | -------------------------------- | -------------------------------------------------------------------------------------- |
| `utf8_codepoints(string)`                              | scalar values                    | `int` code points; ill-formed bytes decode to `U+FFFD`, so iteration never raises.     |
| `utf8_graphemes(string, skip_empty=False)`             | TR29 grapheme clusters           | user-perceived characters such as a base plus combining marks, or emoji ZWJ sequences. |
| `utf8_wordbreaks(string, skip_empty=False)`            | TR29 word boundaries             | all UAX-29 word segments (words and the separators between them; they tile).            |
| `utf8_sentences(string, skip_empty=False)`             | TR29 sentence boundaries         | sentences.                                                                             |
| `utf8_linebreaks(string, skip_empty=False)`             | UAX14 line-break opportunities   | soft-wrap segments.                                                                    |
| `utf8_split_newlines(string, skip_empty=False, with_separators=False)` | 7 Unicode newlines + CRLF | content BETWEEN hard newlines (LF, VT, FF, CR, NEL, `U+2028`, `U+2029`, CRLF).          |
| `utf8_newlines(string, skip_empty=False)`              | 7 Unicode newlines + CRLF        | the newline runs themselves (the separators).                                          |
| `utf8_split_whitespaces(string, skip_empty=False, with_separators=False)` | 25 Unicode `"White_Space"` | content BETWEEN whitespace runs, like `str.split()` with no separator.            |
| `utf8_whitespaces(string, skip_empty=False)`           | 25 Unicode `"White_Space"` chars | the whitespace runs themselves (the separators).                                       |
| `utf8_split_delimiters(string, skip_empty=False, with_separators=False)` | punctuation/symbol/separator | content BETWEEN any Unicode delimiter (superset of whitespace).                  |
| `utf8_delimiters(string, skip_empty=False)`            | punctuation/symbol/separator     | the delimiter runs themselves (the separators).                                        |

Naming follows one rule: the bare name (`newlines`/`whitespaces`/`delimiters`) yields the **separators**, while
`split_*` yields the content **between** them. `skip_empty` drops empty segments; `with_separators=True` interleaves
both losslessly (concatenation reproduces the input), replacing the old `keepends`.

```python
import stringzilla as sz

assert list(sz.utf8_codepoints("AB")) == [65, 66]
assert [str(g) for g in sz.Str("a👍🏽b").utf8_graphemes()] == ["a", "👍🏽", "b"]
[str(w) for w in sz.utf8_wordbreaks("Hi, world")]    # all UAX-29 segments
assert sum(1 for _ in sz.Str("first\nsecond\nthird").utf8_split_newlines()) == 3
assert [str(t) for t in sz.utf8_split_whitespaces("foo  bar baz", skip_empty=True)] == ["foo", "bar", "baz"]
assert "".join(str(s) for s in sz.utf8_split_newlines("a\nb", with_separators=True)) == "a\nb"
```

### Counting Code Points

`utf8_count(string)` counts Unicode characters rather than bytes, unlike `len`, which counts bytes:

```python
import stringzilla as sz

assert sz.utf8_count("hello") == 5
assert sz.utf8_count("é") == 1
assert len(sz.Str("é")) == 2      # bytes
```

### Unicode Case Folding for Uncased Matching

These apply Unicode case folding, correctly handling one-to-many expansions such as German `ß` matching `SS`.

- `utf8_uncased_fold(text, validate=False)` — return the case-folded UTF-8 string as `bytes`.
- `utf8_uncased_search(haystack, needle, start=0, end=len, validate=False)` — index of the first uncased match, or `-1`. For `str` inputs `start`/`end` and the result are codepoint offsets; for `bytes` inputs they are byte offsets.
- `utf8_uncased_order(a, b, validate=False)` — uncased lexicographic comparison: negative, zero, or positive `int`.
- `utf8_uncased_matches(haystack, needle, include_overlapping=False)` — iterate over all uncased matches, yielding each matched region as a `Str` view whose length may differ from `needle` due to folding expansions.

Pass `validate=True` to validate UTF-8 before processing.

```python
import stringzilla as sz

assert sz.utf8_uncased_fold("HELLO") == b"hello"
assert sz.utf8_uncased_fold("Straße") == b"strasse"
assert sz.utf8_uncased_search("Hello World", "WORLD") == 6
assert sz.utf8_uncased_order("hello", "HELLO") == 0
assert [str(m) for m in sz.utf8_uncased_matches("Hello HELLO hello", "hello")] == ["Hello", "HELLO", "hello"]
```

### Unicode Normalization

These mirror `unicodedata.normalize` but operate on raw UTF-8 bytes.

- `utf8_norm(text, form, validate=False)` — normalize to one of `'NFC'`, `'NFD'`, `'NFKC'`, `'NFKD'`; returns `bytes`.
- `utf8_find_denormalized(text, form)` — return the byte offset of the first codepoint that breaks the given normalization form, or `None` if the string is already fully normalized.

```python
import stringzilla as sz

assert sz.utf8_norm("café", "NFD") == b"cafe\xcc\x81"   # decomposed
assert sz.utf8_norm("ﬁ", "NFKD") == b"fi"   # ligature expanded
assert sz.utf8_find_denormalized("café", "NFC") is None   # already NFC
```

## Runtime Dispatch and Capabilities

StringZilla detects the running CPU's SIMD features at import time and routes every kernel to the fastest available backend without recompilation.

- `sz.__version__` — the package version string.
- `sz.__capabilities__` — a tuple of the detected backends, e.g. `('serial', 'haswell', 'skylake', 'ice')`.
- `sz.reset_capabilities(names)` — restrict the active backends to `names`, intersected with the hardware's actual capabilities; if the intersection is empty it falls back to `'serial'`. This updates `sz.__capabilities__` and re-points the dispatch table, which is useful for testing, benchmarking one backend, or reproducibility.

```python
import stringzilla as sz

sz.__capabilities__               # e.g. ('serial', 'haswell', 'skylake', 'ice')
sz.reset_capabilities(["serial"]) # force the scalar backend for this module
```

The `stringzillas` module has the same controls.
`szs.reset_capabilities(names)` sets its default backends, intersected with hardware and falling back to `'serial'`, and each batch engine reports the backends it selected through a read-only property — `__capabilities__` on the four alignment engines and `capabilities` on `Fingerprints`.

```python
import stringzillas as szs

szs.reset_capabilities(("serial",))   # restrict batch dispatch to the scalar backend
szs.LevenshteinDistances().__capabilities__   # backends this engine will use
```
