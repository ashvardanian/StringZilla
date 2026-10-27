# StringZilla for Python

StringZilla for Python wraps SIMD- and SWAR-accelerated native kernels behind `str`- and `bytes`-shaped types that avoid copies wherever possible.
The `stringzilla` module covers single-string search, slicing, splitting, trimming, translation, hashing, checksums, sorting, sampling, random generation, UTF-8 segmentation, Unicode case-folding, Unicode normalization, escaping, and dumping JSON and CSV.
The same module also carries the batch engines, each preparing one set of queries once and scoring it against many collections of candidates, on the host or on a CUDA or ROCm device.

Every call runs the best kernel of the capabilities it is given, by default every CPU capability, which the library clamps to `sz.cpu_capabilities_enabled()`, and the batch engines release the GIL around native work.
The module is also marked safe for free-threaded `Py_GIL_DISABLED` CPython builds.

Throughout this document the `stringzilla` package is imported as `sz`, and NumPy as `np`:

```python
import stringzilla as sz
import numpy as np
```

Almost every `sz` operation is available both as a method on a `String` object and as a module-level function that accepts a `str`, `bytes`, `bytearray`, or `String`.
For example `sz.String("hello").find("l")` and `sz.find("hello", "l")` are equivalent.

## Installation

StringZilla ships on PyPI as one package, carrying the single-string kernels and the batch engines alike:

```sh
pip install stringzilla
```

NumPy is not required, since every engine writes into any writable buffer the caller supplies, including a plain `memoryview`.
It is the convenient way to produce one, though, so most of the examples below reach for it.

Building from source goes through CMake, which scikit-build-core drives, so it needs a C compiler and pulls CMake and Ninja from PyPI when the system has none recent enough.
The build probes the compiler for every SIMD kit it can emit and compiles them all, and every call picks among them by its capability mask, as [Runtime Dispatch](#runtime-dispatch) describes.
Build parallelism follows `CMAKE_BUILD_PARALLEL_LEVEL`, which should be lowered in memory-constrained containers.
`STRINGZILLA_TARGET_ARCH=native` in the environment tunes the build for the machine it runs on, and `-C cmake.define.STRINGZILLA_BUILD_CUDA=ON` adds the CUDA backends of the batch engines, as `STRINGZILLA_BUILD_ROCM=ON` does the ROCm ones.

```sh
CMAKE_BUILD_PARALLEL_LEVEL=2 pip install .
```

## Types

The `stringzilla` module exposes `String`, `Strings`, and `File`, the incremental hashers `Hasher`, `Sha256` and `Sha256s`, the batch engines `LevenshteinEngine`, `OverlapEngine` and `SubstringsEngine`, plus a family of UTF-8 iterator types.

### `String`

`String(source)` is an immutable, zero-copy view over a `str`, `bytes`, `bytearray`, another `String`, or a memory-mapped `File`.
It implements the read-only half of the `str`/`bytes` protocol — indexing, slicing, iteration, `len`, `in`, rich comparison, hashing, and the buffer protocol — without materializing new Python objects.
Slicing returns a new `String` view into the same backing buffer rather than copying bytes.

Two read-only properties expose its memory layout for zero-copy interop with PyArrow, `ctypes`, and similar:

- `String.address` — integer memory address of the first byte.
- `String.nbytes` — length of the view in bytes.

```python
import stringzilla as sz

text = sz.String("the quick brown fox")
assert str(text[4:9]) == "quick"   # a zero-copy view
assert str(text[10:15]) == "brown"
assert "quick" in text
assert len(text) == 19
assert text.nbytes == 19
assert isinstance(text.address, int)
```

### `Strings`

`Strings(sequence=())` is an immutable, indexable, sliceable sequence of byte strings, like a `tuple` of `bytes`, kept on one tape: the bytes back to back and an offsets array, the Apache Arrow `string` or `large_string` layout.
A `pyarrow.Array` of strings or binaries is borrowed in place, while a `list`, `tuple`, generator, or any other iterable of `str` and bytes-like objects is copied once.
The eager splits return one too, with the parts copied onto a tape of their own.
`strings.copy(capabilities, stream=None)` copies the strings into one tape in the unified memory of `capabilities`, as [Engines on a GPU](#engines-on-a-gpu) describes; the host reads it as a normal read-only `Strings`.

Indexing yields `String` views into the tape, and slicing with a step of one borrows it, so neither copies any bytes.
Any other step, `sorted`, `shuffled`, and `sample` gather into a new tape, while `argsort` and `intersect` return indices and copy nothing.
One tape is also dramatically more memory-efficient than a Python `list` of `str`.
N strings cost two allocations, whereas a `list` of `str` holds N separately heap-allocated objects, each carrying a PyObject header, a hash cache, and per-object allocator overhead.
Splitting a large document into millions of tokens therefore stays compact and cache-friendly instead of fragmenting the heap.
The lazy `split_iter`, `rsplit_iter`, and `*_byteset_iter` forms go a step further and build no collection at all, yielding one `String` view at a time.

Five read-only properties expose the tape for Apache Arrow and StringTape interop:

- `Strings.tape_address` — address of the tape's first byte, which the offsets count from.
- `Strings.tape_nbytes` — bytes from that address to the end of the last string.
- `Strings.offsets_address` — address of the first offset.
- `Strings.offsets_nbytes` — length of the offsets array in bytes, one offset more than there are strings.
- `Strings.offsets_are_large` — `True` for the 64-bit offsets a tape past 4 GiB needs, `False` for 32-bit ones.

A borrowed slice shares `tape_address` with its source, while its offsets start past zero.

```python
import stringzilla as sz

names = sz.Strings(["banana", "apple", "cherry", "date"])
assert len(names) == 4
assert str(names[0]) == "banana"
assert [str(x) for x in names[1:3]] == ["apple", "cherry"]   # a borrowed slice
assert names[1:3].tape_address == names.tape_address
```

### `File`

`File(path, mode='r')` memory-maps a path so a `String` can view data larger than RAM without reading it into memory.
The default `mode` is `'r'`, for read-only access.

```python
import os, tempfile, pathlib
import stringzilla as sz

path = os.path.join(tempfile.mkdtemp(), "log.txt")
pathlib.Path(path).write_text("error: disk full\nok\nerror: timeout\n")

mapped = sz.String(sz.File(path))   # scan in place, no copy into RAM
assert mapped.count("error") == 2
```

## Searching and Counting

All search methods accept optional `start` and `end` indices that bound the search window; `start` defaults to `0` and `end` to the string length.
They are available both as `String` methods and as module-level functions.

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

text = sz.String("the quick brown fox")
assert text.find("brown") == 10
assert text.rfind("o") == 17
assert text.index("quick") == 4
assert text.contains("quick")
assert text.startswith("the")
assert text.endswith("fox")
assert sz.String("banana").count("a") == 3
assert sz.String("aaaa").count("aa") == 2                     # non-overlapping
assert sz.String("aaaa").count("aa", allowoverlap=True) == 3  # overlapping
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

text = sz.String("hello world")
assert text.find_first_of("aeiou") == 1
assert text.find_last_of("aeiou") == 7
assert text.find_first_not_of("he") == 2
assert text.find_last_not_of("lo") == 10
assert text.count_byteset("lo") == 5
```

### Equality and Offsets

- `sz.equal(first, second)` — module-level fast byte-wise equality, returning `bool`.
- `offset_within(larger)` — returns the byte offset of this `String` view inside a larger `String` it was sliced from, or `-1` if it is not a sub-view.

```python
import stringzilla as sz

assert sz.equal("abc", "abc")
big = sz.String("hello world")
assert big[6:].offset_within(big) == 6
```

## Splitting and Partitioning

### Eager Splits Returning `Strings`

- `split(separator, maxsplit=∞, keepseparator=False, skip_empty=False)` — split on a substring separator; unlike `str.split`, an empty separator leaves the text whole instead of raising.
- `rsplit(separator, maxsplit=∞, keepseparator=False, skip_empty=False)` — same, scanning from the right.
- `split_byteset(separators, maxsplit=∞, keepseparator=False, skip_empty=False)` — split on any single byte from the `separators` set.
- `rsplit_byteset(separators, maxsplit=∞, keepseparator=False, skip_empty=False)` — same, from the right.
- `splitlines(keeplinebreaks=False, maxsplit=∞)` — split on line breaks.

`maxsplit` caps the number of separators consumed, so `0` leaves the text whole, while a negative value or the default means unlimited.
`keepseparator` keeps the matched separator attached to the parts.
`skip_empty` drops zero-length parts, so a part keeping its separator is never dropped.
Without `maxsplit` and `keepseparator`, a one-byte separator, a byteset, and `splitlines` split the whole text in one native pass straight into the tape.

```python
import stringzilla as sz

assert list(map(str, sz.String("a,b,c").split(","))) == ["a", "b", "c"]
assert list(map(str, sz.String("a,b,c").rsplit(",", maxsplit=1))) == ["a,b", "c"]
assert list(map(str, sz.String("a,b;c").split_byteset(",;"))) == ["a", "b", "c"]
assert list(map(str, sz.String("a\nb\nc").splitlines())) == ["a", "b", "c"]
```

### Lazy Split Iterators

Each eager split has a lazy counterpart that yields `String` views one at a time without building the whole `Strings`.
They take the same arguments.

- `split_iter(separator, maxsplit=∞, keepseparator=False, skip_empty=False)`
- `rsplit_iter(separator, maxsplit=∞, keepseparator=False, skip_empty=False)`
- `split_byteset_iter(separators, maxsplit=∞, keepseparator=False, skip_empty=False)`
- `rsplit_byteset_iter(separators, maxsplit=∞, keepseparator=False, skip_empty=False)`

The reverse iterators yield the last field first.

```python
import stringzilla as sz

for field in sz.String("2024-01-15").split_iter("-"):
    print(field)                                       # 2024, then 01, then 15

assert str(next(iter(sz.String("a/b/c").rsplit_iter("/")))) == "c"
assert sum(1 for _ in sz.String("a,b,c").split_iter(",")) == 3
```

### Joining

`join(strings)` concatenates `strings` with the `String` between every two, as `str.join` does, into a new `String`.
A `Strings` is read in place, any other iterable of `str` and bytes-like objects through a snapshot of it, and the result is measured first and written once.

```python
import stringzilla as sz

assert str(sz.String(", ").join(["Ada", "Grace", "Linus"])) == "Ada, Grace, Linus"
assert str(sz.String("\n").join(sz.String("a b c").split())) == "a\nb\nc"
```

### Partitioning

- `partition(separator)` — return a 3-tuple `(head, separator, tail)` split at the first occurrence; if not found returns `(self, '', '')`.
- `rpartition(separator)` — same, at the last occurrence; if not found returns `('', '', self)`.

```python
import stringzilla as sz

assert tuple(map(str, sz.String("a=b=c").partition("="))) == ("a", "=", "b=c")
assert tuple(map(str, sz.String("a=b=c").rpartition("="))) == ("a=b", "=", "c")
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

assert sz.String("  hi  ").strip() == "hi"
assert sz.String("xxhi").lstrip("x") == "hi"
assert sz.String("hi!!").rstrip("!") == "hi"
```

### Translating

`translate(table, inplace=False, start=0, end=len)` applies a byte-to-byte mapping in a single pass.
`table` is either a 256-byte string/bytes lookup table or a `dict` mapping bytes to bytes, spelled as characters below U+0100.
A `str` input is translated as its UTF-8 bytes and decoded back, so a table that breaks the encoding raises `UnicodeDecodeError`.
With `inplace=True` the buffer is rewritten in place and `None` is returned; otherwise a new `bytes`/`str` is returned.
A non-256-byte table raises `ValueError`; a non-string, non-dict table raises `TypeError`.

```python
import stringzilla as sz

assert sz.String("abc").translate({"a": "A"}) == b"Abc"

table = bytes(range(256)).translate(bytes.maketrans(b"abc", b"ABC"))
assert sz.String("cabbage").translate(table) == b"CABBAge"
```

`Strings.translate(table)` takes the same tables and maps every string of a collection in one call, returning a new `Strings`, as [Escaping and Normalizing Collections](#escaping-and-normalizing-collections) describes.

### Decoding

`decode(encoding='utf-8', errors='strict')` decodes the bytes to a Python `str`, mirroring `bytes.decode`.

```python
import stringzilla as sz

assert sz.String("abc").decode() == "abc"
```

## Hashing and Checksums

### One Shot Hashes and Checksums

- `sz.hash(text, seed=0)` — seeded 64-bit AES-accelerated hash, returned as an unsigned `int`.
  This differs from Python's built-in `hash()`, which returns a platform-dependent `Py_hash_t`.
- `sz.hash_multiseed(text, seeds, out=None)` — hash one string under many seeds at once.
  `seeds` is a contiguous buffer of `uint64`, such as a `numpy.uint64` array or `array('Q', ...)`; plain `int` lists are not accepted.
  Returns a tuple of ints, or fills the optional contiguous `uint64` `out` buffer in place and returns `None`.
  This is much faster than looping `hash` for short strings under many seeds, and is useful for feature hashing, Count-Min sketches, Bloom/cuckoo filters, and MinHash/LSH.
- `sz.bytesum(text)` — additive checksum of the individual byte values, as an `int`.
- `sz.sha256(text)` — 32-byte SHA-256 digest as `bytes`.
- `sz.hmac_sha256(key, message)` — 32-byte HMAC-SHA256 digest as `bytes`.

`hash`, `hash_multiseed`, `bytesum`, and `sha256` are also available as `String` methods.

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

### `Sha256s` for Many Messages at Once

Digesting one message is a serial dependency chain, but independent messages compress in parallel lanes — sixteen at a time on AVX-512, eight on AVX2.
`Sha256s(lanes)` holds one hasher per lane, with `update(chunks)` taking exactly one chunk per lane and `digest(out=None)`, `hexdigest()`, `reset()`, `copy()` and `len()` behaving as on `Sha256`.
Lanes are workers rather than messages: `lanes[i]` reads one as a `Sha256`, and `lanes[i] = sz.Sha256()` retires a finished lane while the others keep streaming.

`sz.hmac_sha256(key, message, out=None)` takes either one message or a collection, authenticating a batch of tokens sharing a secret through the same kernels.

Both accept a `Strings`, read in place, and an `out` buffer of bytes — a `(count, 32)` `numpy.uint8` matrix receives one digest per row with no allocation, since that is byte-for-byte the layout the kernel writes.
The GIL is released around the kernel, so several threads digest concurrently.

Lanes advance in lockstep within a group, so a group costs as much as its longest message.
Every length is handled correctly, but throughput is best when messages of similar length share a group, which `Strings.argsort` arranges.

```python
import numpy as np
import stringzilla as sz

texts = sz.Strings(["alpha", "beta", "gamma"])
lanes = sz.Sha256s(len(texts))
digests = np.empty((len(lanes), sz.Sha256.digest_length), np.uint8)
lanes.update(texts).digest(out=digests)

tags = sz.hmac_sha256(b"secret", texts)   # one 32-byte tag per message
assert tags[0] == sz.hmac_sha256(b"secret", "alpha")
```

## Sorting and Sets

These operate on a `Strings` collection and return new collections or tuples of numbers, leaving the original unchanged.

### Sorting

- `sorted(reverse=False, uncased=False, top=None)` — return a new, stably sorted `Strings`.
  `reverse` sorts descending, `uncased` orders by Unicode case-folding, and `top` keeps only the leading `top` elements (a partial top-k, cheaper than a full sort).
- `argsort(reverse=False, uncased=False, top=None, out=None)` — return the stable permutation of indices that sorts the collection, as a tuple of ints.
  `out` is an optional writable, C-contiguous 64-bit-unsigned buffer such as `numpy.uintp` or `array('Q')`, receiving the indices with zero allocation; when given, `out` itself is returned.

```python
import stringzilla as sz

names = sz.Strings(["banana", "apple", "cherry"])
assert [str(x) for x in names.sorted()] == ["apple", "banana", "cherry"]
assert [str(x) for x in names.sorted(reverse=True)] == ["cherry", "banana", "apple"]
assert [str(x) for x in names.sorted(top=2)] == ["apple", "banana"]
assert names.argsort() == (1, 0, 2)
```

### Intersection

- `intersect(other, seed=0)` — return the positions of strings present in both collections, as a pair of parallel index tuples: `result[0][i]` in this collection and `result[1][i]` in `other` point to equal strings.
  Each distinct shared value is matched exactly once, even if either side holds duplicates; `seed` reshuffles the underlying hash table to resist adversarial inputs.

```python
import stringzilla as sz

ours, theirs = sz.Strings(["banana", "apple", "cherry"]).intersect(sz.Strings(["cherry", "orange", "banana"]))
assert sorted(ours) == [0, 2]                 # "banana" and "cherry" on our side
assert len(ours) == len(theirs) == 2
```

### Searching and Hashing Every String

- `find(needle, out=None)` — return the byte offset of the first `needle` in every string, or `-1` where a string holds none, like `str.find` over each; a `str`, `bytes`, or `String` needle is accepted.
  `out` is an optional writable, C-contiguous pointer-width signed buffer such as `numpy.intp`.
- `hashes(seed=0, out=None)` — return one unsigned 64-bit hash per string, each equal to `sz.hash(text, seed)`, for group-by keys, joins, and sharding.
  `out` is an optional writable, C-contiguous 64-bit unsigned buffer such as `numpy.uint64`.
- `utf8_count(out=None)` — return the codepoints of every string, like `len(text.decode())` over each, validating every string as UTF-8 on the way and raising `ValueError` that names the index of the first ill-formed one.
  `out` is an optional writable, C-contiguous pointer-width unsigned buffer such as `numpy.uintp`.

All three run once over the whole tape and, like `argsort`, return `out` itself when it is given.

```python
import stringzilla as sz

names = sz.Strings(["hello", "world", "help"])
assert names.find("el") == (1, -1, 1)
assert names.hashes(seed=7) == tuple(sz.hash(name, 7) for name in ["hello", "world", "help"])
assert sz.Strings(["hello", "café", ""]).utf8_count() == (5, 4, 0)
```

### Sampling and Shuffling

- `sample(size, seed=None)` — return a new `Strings` of `size` elements drawn at random with replacement; `seed` makes it reproducible.
- `shuffled(seed=None)` — return a new `Strings` with the elements randomly permuted; `seed` makes it reproducible.

```python
import stringzilla as sz

pool = sz.Strings(["a", "b", "c", "d"])
assert len(pool.sample(2)) == 2
assert [str(x) for x in sorted(pool.shuffled(seed=42))] == ["a", "b", "c", "d"]
```

### Random Byte Generation

These module-level functions generate random bytes and are not tied to `Strings`:

- `sz.random(length, nonce=0, alphabet=None)` — return a fresh random `bytes` of `length`; if `alphabet` is given, each byte `b` is mapped to `alphabet[b % len(alphabet)]`.
- `sz.fill_random(buffer, nonce=0, alphabet=None, start=0, end=len)` — fill a writable, contiguous byte buffer such as a `bytearray`, `memoryview`, or `String` in place with pseudo-random bytes, optionally remapped to `alphabet`, optionally limited to the `[start, end)` slice.
  Returns `None`.
  Also available as the `String.fill_random` method.

```python
import stringzilla as sz

assert len(sz.random(16)) == 16
sz.random(8, alphabet="ACGT")                 # e.g. b'GCTAACGT'

buf = bytearray(16)
sz.fill_random(buf)                           # mutates buf in place
assert len(buf) == 16
```

## Escaping and Normalizing Collections

Every verb below is one native call over the whole collection and returns a new `Strings` owning one tape.

- `translate(table)`: maps every byte through a 256-byte table or a `dict`, as `bytes.translate` does for each string.
- `escape(kind='json')`: escapes every string for a text format.
- `unescape(kind='json')`: reverses `escape`, raising `ValueError` that names the index of the first string holding a malformed escape.
- `normalize(form='NFC')`: brings every string into a Unicode form, as `unicodedata.normalize` does, taking the forms `utf8_norm` takes.

The escape kinds follow the standard library's own writers and readers:

- `json`: a string body as `json.dumps(text, ensure_ascii=False)[1:-1]` writes it, decoded as `json.loads` decodes it.
- `json_ascii`: the same with every non-ASCII codepoint as `\uXXXX`, as `json.dumps(text)` writes it.
- `hex`: two lowercase digits per byte, as `bytes.hex` writes them and `bytes.fromhex` reads them.
- `html`: `&`, `<`, `>`, `"` and `'` as entities, as `html.escape` writes them but with `&#39;` for `'`, decoding the five XML entities and numeric references as `html.unescape` does.
- `url`: every byte outside `A-Z a-z 0-9 - . _ ~` percent-encoded, as `urllib.parse.quote(text, safe="")` writes it and `urllib.parse.unquote_to_bytes` reads it.
- `console`: controls, DEL, ill-formed UTF-8 and the bidirectional overrides as `\n`, `\xHH` or `\uXXXX`, so a logged string can neither hide nor reorder what a terminal shows.
- `csv`: a field quoted only around a comma, a quote, CR or LF, with every quote doubled, as `csv.writer` writes it and `csv.reader` reads it.

```python
import stringzilla as sz

names = sz.Strings(["Ada", 'Grace "Amazing" Hopper', "Linus\nTorvalds"])
assert [str(x) for x in names.escape("csv")] == ["Ada", '"Grace ""Amazing"" Hopper"', '"Linus\nTorvalds"']
assert [str(x) for x in names.escape("url").unescape("url")] == [str(x) for x in names]
assert [str(x) for x in names.translate({"a": "A"})][0] == "AdA"
assert [str(x) for x in sz.Strings(["Straße", "ﬁle"]).normalize("NFKC_Casefold")] == ["strasse", "file"]
```

## Dumping JSON and CSV

Writing a column of strings out as JSON or CSV usually costs a Python object per cell: `json.dumps` walks a `list` of `str`, `csv.writer` formats one row at a time, and pandas first builds a `DataFrame` of objects.
`sz.to_json` and `sz.to_csv` instead escape each column once, in one native pass over the whole column, then join the framed rows with one more native call into a single `bytes` object of exactly the right size.

- `Strings.to_json(ensure_ascii=True)`: one compact JSON array, byte for byte what `json.dumps(strings, ensure_ascii=ensure_ascii, separators=(",", ":"))` writes.
- `sz.to_json(columns, ensure_ascii=True)`: an array of records, one object per row, byte for byte what `json.dumps` writes for those records, and what `pandas.DataFrame(columns).to_json(orient="records")` writes apart from pandas spelling `/` as `\/`.
- `sz.to_csv(columns, lineterminator="\n")`: a header line of the names, then a line per row, byte for byte what `csv.writer` and `pandas.DataFrame(columns).to_csv(index=False)` write.

`columns` maps every name to a `Strings` or to any sequence of `str`, all of the same length, and the names are escaped the same way as the cells.

```python
import csv, io, json
import pandas as pd
import stringzilla as sz

names = ["Ada", 'Grace "Amazing" Hopper', "Linus\nTorvalds"]
cities = ["London", "New York, NY", "Helsinki"]
columns = {"name": sz.Strings(names), "city": cities}
frame = pd.DataFrame({"name": names, "city": cities})

records = [{"name": name, "city": city} for name, city in zip(names, cities)]
assert sz.to_json(columns) == json.dumps(records, separators=(",", ":")).encode()
assert sz.to_json(columns) == frame.to_json(orient="records").encode()

buffer = io.StringIO()
writer = csv.writer(buffer, lineterminator="\n")
writer.writerow(columns)
writer.writerows(zip(names, cities))
assert sz.to_csv(columns) == buffer.getvalue().encode()
assert sz.to_csv(columns) == frame.to_csv(index=False, lineterminator="\n").encode()
```

On an Apple M5 Pro with CPython 3.12 and pandas 3.0, the input is 200,000 English sentences from Leipzig 1M plus a second column holding the first word of each, 27.1 MB of UTF-8 in all.
Times are the best of 5 runs, each producing the whole `bytes` object:

- A JSON array of the sentences with `ensure_ascii=True`: 39.4 ms for `json.dumps` and 12.0 ms for `Strings.to_json`, 3.3x faster.
- The same array with `ensure_ascii=False`: 75.8 ms and 10.9 ms, 7.0x faster.
- CSV of both columns: 174.5 ms for `csv.writer` and 188.5 ms for `DataFrame.to_csv`, against 20.8 ms for `sz.to_csv` over `list`s and 17.1 ms over `Strings`, 8.4x and 10.2x faster than `csv.writer`.
- JSON records of both columns: 94.1 ms for `json.dumps` with the `dict`s built inline and 68.2 ms for `DataFrame.to_json`, against 15.5 ms for `sz.to_json` over `Strings`, 6.1x and 4.4x faster.

## Batch Engines

Three engines prepare one set of queries once and score it against as many collections of candidates as a caller has.
Preparation fixes the alphabet or the vocabulary, resolves the SIMD tier, and takes the memory the rounds reuse; every later call only streams candidates past what is already there.
Each engine writes into a writable buffer the caller supplies as `out`, so nothing is allocated per round and a round can fill one slice of a larger matrix.

| Engine                     | Constructed from                                  | Scores with                                                     |
| :------------------------- | :------------------------------------------------ | :--------------------------------------------------------------- |
| `LevenshteinEngine`        | a batch of query strings and a symbol alphabet    | `distances(candidates, out, bound=None)`                         |
| `OverlapEngine`            | a batch of query strings and window widths        | `scores(candidates, out, radius=None)`                           |
| `SubstringsEngine`         | a vocabulary of needles and an overlap policy     | `counts`, `find`, `replace`, and `bm25_scores` over haystacks    |

Queries, candidates, and haystacks all arrive as `sz.Strings`, and every `out` buffer is any writable object supporting the buffer protocol — a NumPy array, an `array.array`, or a plain `memoryview`.
Engine calls release the GIL, and calls into one engine take turns on its lock, so build one engine per thread to score in parallel.

### Edit Distances

`LevenshteinEngine(queries, symbol='bytes')` computes unit-cost Levenshtein distances from every prepared query to every candidate.
`symbol='bytes'` counts byte edits, `symbol='runes'` counts UTF-8 rune edits, and the choice is fixed at construction because it picks the mask layout as well as the tier.

`distances(candidates, out, *, bound=None)` fills a `[queries, candidates]` block of pointer-width unsigned integers, contiguous along its candidate axis.
A `bound` keeps a distance exact up to it and reports `bound + 1` for every candidate past it, on every backend, so a round can stop early; `None` keeps every distance exact.

```python
import stringzilla as sz

engine = sz.LevenshteinEngine(sz.Strings(["hello", "world"]))
out = memoryview(bytearray(4 * 8)).cast("Q", (2, 2))
engine.distances(sz.Strings(["hallo", "word"]), out)
assert out[0, 0] == 1           # hello -> hallo

# Bounded at 0, any candidate that differs reports exactly 1.
engine.distances(sz.Strings(["hallo", "word"]), out, bound=0)
assert out[0, 0] == 1 and out[0, 1] == 1

# UTF-8 rune distances, where one accented codepoint is one edit rather than two.
runes = sz.LevenshteinEngine(sz.Strings(["café"]), symbol="runes")
out = memoryview(bytearray(8)).cast("Q", (1, 1))
runes.distances(sz.Strings(["cafe"]), out)
assert out[0, 0] == 1
```

### Window Overlap

`OverlapEngine(queries, widths)` hashes every query's fixed-width byte n-grams into one B-tree per query, at every width in `widths`, and probes that forest with each candidate's own windows.
A score is the share of a candidate's windows the query also spells, in `[0, 1]`, so it is asymmetric: swapping the two sides changes the answer when either repeats a window.
Widths need not form a doubling chain, and a width past a text scores zero for every pair it spans.

`scores(candidates, out, *, radius=None)` fills a `[queries, candidates, widths]` block of 32-bit floats, contiguous along its width axis.
A pair's distance is the windows it misses over its longer side's, summed over every width, and a `radius` in `[0, 1]` keeps every pair within it exact; a pair past it scores `0.0` at every width, on every backend, and `None` keeps every score exact.

```python
import stringzilla as sz

engine = sz.OverlapEngine(sz.Strings(["the quick brown fox"]), [4, 8])
out = memoryview(bytearray(1 * 2 * 2 * 4)).cast("f", (1, 2, 2))
engine.scores(sz.Strings(["the quick brown cat", "nothing alike"]), out)
assert all(0.0 <= out[0, c, w] <= 1.0 for c in range(2) for w in range(2))
```

### Multi-Pattern Matching

`SubstringsEngine(needles, case_sensitivity='cased', overlap_policy='overlapping', hot_states=None, matches_budget=0)` compiles a whole vocabulary into one Aho-Corasick automaton, so every haystack answers every needle in a single pass whatever the vocabulary size.

| Argument           | Default           | Meaning                                                                        |
| :----------------- | :---------------- | :------------------------------------------------------------------------------ |
| `needles`          | required          | `sz.Strings` vocabulary; an empty needle is refused, valid UTF-8 needed when folding. |
| `case_sensitivity` | `'cased'`         | `'cased'` matches bytes, `'uncased'` folds both sides under full Unicode folding. |
| `overlap_policy`   | `'overlapping'`   | `'overlapping'`, `'leftmost-longest'`, or `'leftmost-first'`.                    |
| `hot_states`       | `None`            | States kept in the dense hot rows; `None` sizes them to a fixed byte budget.     |
| `matches_budget`   | `0`               | Matches one round may emit, read by a device tier only; `0` takes its default.   |

The policy belongs to the engine rather than to a call, because the round's arena is sized for one cover and holds no other.
Four verbs share the compiled automaton:

| Call                                                  | Writes                                                                               |
| :---------------------------------------------------- | :------------------------------------------------------------------------------------ |
| `counts(haystacks, out)`                              | One pointer-width count per haystack.                                                 |
| `find(haystacks, matches, offsets)`                   | One `(haystack, needle, offset, length)` row per match, plus one boundary per haystack. |
| `replace(haystacks, replacements, target, offsets)`   | The rewritten haystacks into one target buffer, plus one boundary per haystack.        |
| `bm25_scores(haystacks, needle_weights, out, ...)`    | One 32-bit score per haystack, the vocabulary being the query.                          |

A capacity too small is not an error for `find` or `replace`: the boundaries and the `report` are filled either way, which is what sizes the next call.
Passing `None` for `matches` or `target` makes the call a pure size query.
`engine.report` is a dict of four counts from the last round — `matches_emitted` is the truth whatever the outputs could hold, `matches_stored` what was written, `target_length` what a rewrite needs, and `shortfall` what did not fit.

```python
import stringzilla as sz

engine = sz.SubstringsEngine(sz.Strings(["he", "she", "his", "hers"]))
haystacks = sz.Strings(["ushers", "hershey"])

counts = memoryview(bytearray(2 * 8)).cast("Q")
engine.counts(haystacks, counts)
assert counts[0] + counts[1] == 7

matches = memoryview(bytearray(8 * 4 * 8)).cast("Q", (8, 4))
offsets = memoryview(bytearray(3 * 8)).cast("Q")
engine.find(haystacks, matches, offsets)

cover = sz.SubstringsEngine(sz.Strings(["he", "she"]), overlap_policy="leftmost-longest")
target = memoryview(bytearray(32))
offsets = memoryview(bytearray(3 * 8)).cast("Q")
cover.replace(haystacks, sz.Strings(["HE", "SHE"]), target, offsets)
assert bytes(target[offsets[1]:offsets[2]]) == b"HErSHEy"
```

`bm25_scores` treats the vocabulary itself as the query, `needle_weights[i]` carrying needle `i`'s IDF or boost, and writes one score per haystack without ever materializing a per-term frequency row.
Term frequencies are raw overlapping counts, so the engine's own overlap policy does not apply there.
`document_lengths` defaults to each haystack's byte length, and `average_document_length` is required whenever `length_normalization` is positive.

```python
import stringzilla as sz

engine = sz.SubstringsEngine(sz.Strings(["cat", "dog"]))
weights = memoryview(bytearray(2 * 4)).cast("f")
weights[0], weights[1] = 1.0, 1.0
out = memoryview(bytearray(2 * 4)).cast("f")
engine.bm25_scores(sz.Strings(["cat and dog", "nothing here"]), weights, out, average_document_length=10.0)
assert out[1] == 0.0
```

### Engines on a GPU

Every engine constructor takes keyword-only `capabilities=` and `stream=`.
`capabilities=` defaults to every CPU capability, clamped to what this CPU runs, and a GPU producer's mask, like `sz.cuda_capabilities_enabled(0)`, builds the engine on that vendor.
It prepares the batch where a kernel reaches it and resolves the launch geometry once, and every later round of that engine runs there.
`stream=` is a `cudaStream_t`, a `hipStream_t` or an `id<MTLCommandQueue>` as an integer, which names the device, or `None` for the default stream of the default device, and it carries only the preparation: the engine does not keep it; a CPU engine takes no stream.
`SubstringsEngine` also takes a `haystacks_budget` and `OverlapEngine` a `candidates_budget`, the most one round may carry on a device, which the CPU ignores.
The device arena is sized once from those budgets and no verb allocates, so a substrings round past its budget is refused, while CUDA overlap keeps no per-candidate arena and ignores its budget.

Every verb then takes its own keyword-only `stream`, so one engine can be scored from several streams.
`LevenshteinEngine` and `OverlapEngine` keep no round state on the device, so their rounds may run on any number of streams at once.
`SubstringsEngine` rounds share one arena and one report, so the caller orders them, on one stream or with events between two.

A device engine reads its inputs in place, so they come from `sz.Strings(texts).copy(gpu)`, and a host `Strings` raises `BufferError`, while a host engine reads a copy made for any capabilities.
`sz.cuda_stream_init(ordinal)` makes a stream on any device, and `sz.cuda_stream_free(stream)` frees it once synchronized, with `rocm_` and `metal_` twins.

```python
import stringzilla as sz

gpu = sz.cuda_capabilities_enabled(0)
stream = sz.cuda_stream_init(0)
engine = sz.LevenshteinEngine(sz.Strings(["kitten", "saturday"]), capabilities=gpu, stream=stream)
candidates = sz.Strings(["sitting", "sunday"]).copy(gpu, stream)
engine.distances(candidates, unified_out, stream=stream)
sz.synchronize(gpu, stream)  # before `unified_out` is read or `candidates` dropped
sz.cuda_stream_free(stream)
```

A device round enqueues and returns, so its `out` buffer has to be memory the device reaches, has to outlive the launch, and must not be read before the caller joins the stream.
A plain NumPy array is host memory and is refused; a NumPy array over a `cudaMallocManaged` allocation works.
`SubstringsEngine.report` is written by the device too, so it obeys the same rule.
A host engine ignores `stream`, imposes no such requirement, and has always finished writing by the time it answers.

## Unicode Segmentation

`String`, along with the matching module-level functions, exposes lazy iterators over Unicode boundaries.
Each yields `String` views into the original buffer, so segmentation stays allocation-free, except `utf8_codepoints`, which yields `int` code points.

| Method / function                                                         | Standard                         | Yields                                                                                 |
| :------------------------------------------------------------------------ | :------------------------------- | :------------------------------------------------------------------------------------- |
| `utf8_codepoints(string)`                                                 | scalar values                    | `int` code points; ill-formed bytes decode to `U+FFFD`, so iteration never raises.     |
| `utf8_graphemes(string, skip_empty=False)`                                | UAX-29 grapheme clusters         | user-perceived characters such as a base plus combining marks, or emoji ZWJ sequences. |
| `utf8_wordbreaks(string, skip_empty=False)`                               | UAX-29 word boundaries           | all UAX-29 word segments (words and the separators between them; they tile).           |
| `utf8_sentences(string, skip_empty=False)`                                | UAX-29 sentence boundaries       | sentences.                                                                             |
| `utf8_linebreaks(string, skip_empty=False)`                               | UAX-14 line-break opportunities  | soft-wrap segments.                                                                    |
| `utf8_split_newlines(string, skip_empty=False, with_separators=False)`    | 7 Unicode newlines + CRLF        | content BETWEEN hard newlines (LF, VT, FF, CR, NEL, `U+2028`, `U+2029`, CRLF).         |
| `utf8_newlines(string, skip_empty=False)`                                 | 7 Unicode newlines + CRLF        | the newline runs themselves (the separators).                                          |
| `utf8_split_whitespaces(string, skip_empty=False, with_separators=False)` | 25 Unicode `"White_Space"`       | content BETWEEN whitespace runs, like `str.split()` with no separator.                 |
| `utf8_whitespaces(string, skip_empty=False)`                              | 25 Unicode `"White_Space"` chars | the whitespace runs themselves (the separators).                                       |
| `utf8_split_delimiters(string, skip_empty=False, with_separators=False)`  | punctuation/symbol/separator     | content BETWEEN any Unicode delimiter (superset of whitespace).                        |
| `utf8_delimiters(string, skip_empty=False)`                               | punctuation/symbol/separator     | the delimiter runs themselves (the separators).                                        |

Naming follows one rule: the bare name (`newlines`/`whitespaces`/`delimiters`) yields the __separators__, while `split_*` yields the content __between__ them.
`skip_empty` drops empty segments, and `with_separators=True` interleaves both losslessly, so concatenating the result reproduces the input.

```python
import stringzilla as sz

assert list(sz.utf8_codepoints("AB")) == [65, 66]
assert [str(g) for g in sz.String("a👍🏽b").utf8_graphemes()] == ["a", "👍🏽", "b"]
[str(w) for w in sz.utf8_wordbreaks("Hi, world")]    # all UAX-29 segments
assert sum(1 for _ in sz.String("first\nsecond\nthird").utf8_split_newlines()) == 3
assert [str(t) for t in sz.utf8_split_whitespaces("foo  bar baz", skip_empty=True)] == ["foo", "bar", "baz"]
assert "".join(str(s) for s in sz.utf8_split_newlines("a\nb", with_separators=True)) == "a\nb"
```

### Counting Code Points

`utf8_count(string)` counts Unicode characters rather than bytes, unlike `len`, which counts bytes, and `Strings.utf8_count()` counts every string of a collection in one pass:

```python
import stringzilla as sz

assert sz.utf8_count("hello") == 5
assert sz.utf8_count("é") == 1
assert len(sz.String("é")) == 2      # bytes
```

### Unicode Case Folding for Uncased Matching

These apply Unicode case folding, correctly handling one-to-many expansions such as German `ß` matching `SS`.

- `utf8_uncased_fold(text, validate=False)` — return the case-folded UTF-8 string as `bytes`.
- `utf8_uncased_search(haystack, needle, start=0, end=len, validate=False)` — index of the first uncased match, or `-1`.
  For `str` inputs `start`/`end` and the result are codepoint offsets; for `bytes` inputs they are byte offsets.
- `utf8_uncased_order(a, b, validate=False)` — uncased lexicographic comparison: negative, zero, or positive `int`.
- `utf8_uncased_matches(haystack, needle, include_overlapping=False)` — iterate over all uncased matches, yielding each matched region as a `String` view whose length may differ from `needle` due to folding expansions; an empty `needle` matches at every codepoint boundary, the end included.

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

- `utf8_norm(text, form, validate=False)` — normalize to one of `'NFC'`, `'NFD'`, `'NFKC'`, `'NFKD'`, `'NFKC_Casefold'`; returns `bytes`.
  `'NFKC_Casefold'` is the Unicode toNFKC_Casefold, the keys search engines match on: NFKC with full case folding and without default-ignorable codepoints.
- `utf8_find_denormalized(text, form)` — return the byte offset of the first codepoint that breaks the given normalization form, or `None` if the string is already fully normalized.

```python
import stringzilla as sz

assert sz.utf8_norm("café", "NFD") == b"cafe\xcc\x81"   # decomposed
assert sz.utf8_norm("ﬁ", "NFKD") == b"fi"   # ligature expanded
assert sz.utf8_find_denormalized("café", "NFC") is None   # already NFC
assert sz.utf8_norm("Straße­", "NFKC_Casefold") == b"strasse"   # folded, soft hyphen dropped
```

## Runtime Dispatch

StringZilla detects the running CPU's SIMD features and routes every kernel to the fastest capability both the CPU and this build have, without recompilation.
Capabilities are `sz.Capability` flags, one member per CPU capability, like `sz.Capability.HASWELL` or `sz.Capability.NEON`; GPU bits sit above every member, so a GPU mask prints as a number.

The producers report capabilities for the CPU, or for one GPU by its runtime's ordinal, or make a stream on that GPU, and are the only calls taking an ordinal.

- `sz.__version__` — the package version string.
- `sz.cpu_capabilities_detected()`, `sz.cuda_capabilities_detected(ordinal)` — what the device can execute, whether or not its kernels were compiled in.
- `sz.cpu_capabilities_compiled()`, `sz.cuda_capabilities_compiled()` — what this build holds kernels for, whether or not the device runs them.
- `sz.cpu_capabilities_enabled()`, `sz.cuda_capabilities_enabled(ordinal)` — what calls dispatch with, both of the above; on the CPU it always has `SERIAL`.
- `sz.cuda_count_devices()` — the number of CUDA devices; it raises `RuntimeError` when none answers, as when this build lacks its kernels, and so does an ordinal past the last device.
- `sz.cpu_configure_thread(capabilities)` — prepare the calling thread for the kernels of `capabilities`; call it on every thread that runs kernels.
- `sz.cuda_stream_init(ordinal)`, `sz.cuda_stream_free(stream)` — make a stream on one device, as the integer every `stream=` takes, and free it once nothing queued on it remains.
- `sz.synchronize(capabilities, stream=None)` — wait for one stream of the vendor `capabilities` names, or its default one.

ROCm and Metal have the same `rocm_` and `metal_` producers.
Every function that runs kernels also takes a keyword-only `capabilities=`, narrowing that one call and leaving every other call alone.
The keyword keeps no serial fallback, so a mask with no kernel for the call raises `LookupError`.

```python
import stringzilla as sz

sz.cpu_capabilities_enabled()                                 # e.g. <Capability.SERIAL|NEON|NEONAES|NEONSHA: 449>
sz.find("haystack", "st", capabilities=sz.Capability.SERIAL)  # one call on the scalar kernel
sz.cpu_configure_thread(sz.cpu_capabilities_enabled())        # on a thread that runs kernels
[sz.cuda_capabilities_enabled(ordinal) for ordinal in range(sz.cuda_count_devices())]  # raises without CUDA
```

Engines and the stateful objects, like `Hasher`, `Sha256` and the AES keys, dispatch every call with the capabilities they were constructed with.
