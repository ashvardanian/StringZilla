# StringZilla for Python

StringZilla for Python wraps SIMD- and SWAR-accelerated native kernels behind `str`- and `bytes`-shaped types that avoid copies wherever possible.
The `stringzilla` module covers single-string search, slicing, splitting, trimming, translation, hashing, checksums, sorting, sampling, random generation, UTF-8 segmentation, Unicode case-folding, and Unicode normalization.
The same module also carries the batch engines, each preparing one set of queries once and scoring it against many collections of candidates, on the host or on a CUDA device.

Every kernel selects the fastest backend for the running CPU at import time, and the batch engines release the GIL around native work.
The module is also marked safe for free-threaded `Py_GIL_DISABLED` CPython builds.

Throughout this document the `stringzilla` package is imported as `sz`, and NumPy as `np`:

```python
import stringzilla as sz
import numpy as np
```

Almost every `sz` operation is available both as a method on a `Str` object and as a module-level function that accepts a `str`, `bytes`, `bytearray`, or `Str`.
For example `sz.Str("hello").find("l")` and `sz.find("hello", "l")` are equivalent.

## Installation

StringZilla ships on PyPI as one package, carrying the single-string kernels and the batch engines alike:

```sh
pip install stringzilla
```

NumPy is not required, since every engine writes into any writable buffer the caller supplies, including a plain `memoryview`.
It is the convenient way to produce one, though, so most of the examples below reach for it.

## Types

The `stringzilla` module exposes `Str`, `Strs`, and `File`, the incremental hashers `Hasher`, `Sha256` and `Sha256s`, the batch engines `LevenshteinEngine`, `OverlapEngine` and `SubstringsEngine`, plus a family of UTF-8 iterator types.

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


`Strs` keeps its parts on a single contiguous "tape" with an offsets array, so `sorted`, `argsort`, `intersect`, `sample`, and `shuffled` reorder offsets rather than bytes.
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
| :----------------------- | :----------------------------------------- |
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

- `sz.hash(text, seed=0)` — seeded 64-bit AES-accelerated hash, returned as an unsigned `int`.
  This differs from Python's built-in `hash()`, which returns a platform-dependent `Py_hash_t`.
- `sz.hash_multiseed(text, seeds, out=None)` — hash one string under many seeds at once.
  `seeds` is a contiguous buffer of `uint64`, such as a `numpy.uint64` array or `array('Q', ...)`; plain `int` lists are not accepted.
  Returns a tuple of ints, or fills the optional contiguous `uint64` `out` buffer in place and returns `None`.
  This is much faster than looping `hash` for short strings under many seeds, and is useful for feature hashing, Count-Min sketches, Bloom/cuckoo filters, and MinHash/LSH.
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

### `Sha256s` for Many Messages at Once

Digesting one message is a serial dependency chain, but independent messages compress in parallel lanes — sixteen at a time on AVX-512, eight on AVX2.
`Sha256s(lanes)` holds one hasher per lane, with `update(chunks)` taking exactly one chunk per lane and `digest(out=None)`, `hexdigest()`, `reset()`, `copy()` and `len()` behaving as on `Sha256`.
Lanes are workers rather than messages: `lanes[i]` reads one as a `Sha256`, and `lanes[i] = sz.Sha256()` retires a finished lane while the others keep streaming.

`sz.hmac_sha256(key, message, out=None)` takes either one message or a collection, authenticating a batch of tokens sharing a secret through the same kernels.

Both accept a `Strs`, read in place whatever its layout, and an `out` buffer of bytes — a `(count, 32)` `numpy.uint8` matrix receives one digest per row with no allocation, since that is byte-for-byte the layout the kernel writes.
The GIL is released around the kernel, so several threads digest concurrently.

Lanes advance in lockstep within a group, so a group costs as much as its longest message.
Every length is handled correctly, but throughput is best when messages of similar length share a group, which `Strs.argsort` arranges.

```python
import numpy as np
import stringzilla as sz

texts = sz.Strs(["alpha", "beta", "gamma"])
lanes = sz.Sha256s(len(texts))
digests = np.empty((len(lanes), sz.Sha256.digest_length), np.uint8)
lanes.update(texts).digest(out=digests)

tags = sz.hmac_sha256(b"secret", texts)   # one 32-byte tag per message
assert tags[0] == sz.hmac_sha256(b"secret", "alpha")
```

## Sorting, Intersecting, and Sampling

These operate on a `Strs` collection and return new collections or index tuples, leaving the original unchanged.

### Sorting

- `sorted(reverse=False, uncased=False, top=None)` — return a new, stably sorted `Strs`.
  `reverse` sorts descending, `uncased` orders by Unicode case-folding, and `top` keeps only the leading `top` elements (a partial top-k, cheaper than a full sort).
- `argsort(reverse=False, uncased=False, top=None, out=None)` — return the stable permutation of indices that sorts the collection, as a tuple of ints.
  `out` is an optional writable, C-contiguous 64-bit-unsigned buffer such as `numpy.uintp` or `array('Q')`, receiving the indices with zero allocation; when given, `out` itself is returned.

```python
import stringzilla as sz

names = sz.Strs(["banana", "apple", "cherry"])
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

ours, theirs = sz.Strs(["banana", "apple", "cherry"]).intersect(sz.Strs(["cherry", "orange", "banana"]))
assert sorted(ours) == [0, 2]                 # "banana" and "cherry" on our side
assert len(ours) == len(theirs) == 2
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
- `sz.fill_random(buffer, nonce=0, alphabet=None, start=0, end=len)` — fill a writable, contiguous byte buffer such as a `bytearray`, `memoryview`, or `Str` in place with pseudo-random bytes, optionally remapped to `alphabet`, optionally limited to the `[start, end)` slice.
  Returns `None`.
  Also available as the `Str.fill_random` method.

```python
import stringzilla as sz

assert len(sz.random(16)) == 16
sz.random(8, alphabet="ACGT")                 # e.g. b'GCTAACGT'

buf = bytearray(16)
sz.fill_random(buf)                           # mutates buf in place
assert len(buf) == 16
```

## Batch Engines

Three engines prepare one set of queries once and score it against as many collections of candidates as a caller has.
Preparation fixes the alphabet or the vocabulary, resolves the SIMD tier, and takes the memory the rounds reuse; every later call only streams candidates past what is already there.
Each engine writes into a writable buffer the caller supplies as `out`, so nothing is allocated per round and a round can fill one slice of a larger matrix.

| Engine                     | Constructed from                                  | Scores with                                                     |
| :------------------------- | :------------------------------------------------ | :--------------------------------------------------------------- |
| `LevenshteinEngine`        | a batch of query strings and a symbol alphabet    | `distances(candidates, out)`                                     |
| `OverlapEngine`            | a batch of query strings and window widths        | `scores(candidates, out)`                                        |
| `SubstringsEngine`         | a vocabulary of needles and an overlap policy     | `counts`, `find`, `replace`, and `bm25_scores` over haystacks    |

Queries, candidates, and haystacks all arrive as `sz.Strs`, and every `out` buffer is any writable object supporting the buffer protocol — a NumPy array, an `array.array`, or a plain `memoryview`.
An engine holds its own state and grows its round scratch, so one engine is not safe to call from two threads at once; build one per worker and shard the candidates between them.

### Edit Distances

`LevenshteinEngine(queries, symbol='bytes')` computes unit-cost Levenshtein distances from every prepared query to every candidate.
`symbol='bytes'` counts byte edits, `symbol='runes'` counts UTF-8 rune edits, and the choice is fixed at construction because it picks the mask layout as well as the tier.

`distances(candidates, out)` fills a `[queries, candidates]` block of pointer-width unsigned integers, contiguous along its candidate axis.

```python
import stringzilla as sz

engine = sz.LevenshteinEngine(sz.Strs(["hello", "world"]))
out = memoryview(bytearray(4 * 8)).cast("Q", (2, 2))
engine.distances(sz.Strs(["hallo", "word"]), out)
assert out[0, 0] == 1           # hello -> hallo

# UTF-8 rune distances, where one accented codepoint is one edit rather than two.
runes = sz.LevenshteinEngine(sz.Strs(["café"]), symbol="runes")
out = memoryview(bytearray(8)).cast("Q", (1, 1))
runes.distances(sz.Strs(["cafe"]), out)
assert out[0, 0] == 1
```

### Window Overlap

`OverlapEngine(queries, widths)` hashes every query's fixed-width byte n-grams into one B-tree per query, at every width in `widths`, and probes that forest with each candidate's own windows.
A score is the share of a candidate's windows the query also spells, in `[0, 1]`, so it is asymmetric: swapping the two sides changes the answer when either repeats a window.
Widths need not form a doubling chain, and a width past a text scores zero for every pair it spans.

`scores(candidates, out)` fills a `[queries, candidates, widths]` block of 32-bit floats, contiguous along its width axis.

```python
import stringzilla as sz

engine = sz.OverlapEngine(sz.Strs(["the quick brown fox"]), [4, 8])
out = memoryview(bytearray(1 * 2 * 2 * 4)).cast("f", (1, 2, 2))
engine.scores(sz.Strs(["the quick brown cat", "nothing alike"]), out)
assert all(0.0 <= out[0, c, w] <= 1.0 for c in range(2) for w in range(2))
```

### Multi-Pattern Matching

`SubstringsEngine(needles, case_sensitivity='cased', overlap_policy='overlapping', hot_states=None, matches_budget=0)` compiles a whole vocabulary into one Aho-Corasick automaton, so every haystack answers every needle in a single pass whatever the vocabulary size.

| Argument           | Default           | Meaning                                                                        |
| :----------------- | :---------------- | :------------------------------------------------------------------------------ |
| `needles`          | required          | `sz.Strs` vocabulary; an empty needle is refused, valid UTF-8 needed when folding. |
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
| `replace(haystacks, replacements, tape, offsets)`     | The rewritten haystacks onto one tape, plus one boundary per haystack.                 |
| `bm25_scores(haystacks, needle_weights, out, ...)`    | One 32-bit score per haystack, the vocabulary being the query.                          |

A capacity too small is not an error for `find` or `replace`: the boundaries and the `report` are filled either way, which is what sizes the next call.
Passing `None` for `matches` or `tape` makes the call a pure size query.
`engine.report` is a dict of four counts from the last round — `matches_emitted` is the truth whatever the outputs could hold, `matches_stored` what was written, `tape_bytes` what a rewrite needs, and `shortfall` what did not fit.

```python
import stringzilla as sz

engine = sz.SubstringsEngine(sz.Strs(["he", "she", "his", "hers"]))
haystacks = sz.Strs(["ushers", "hershey"])

counts = memoryview(bytearray(2 * 8)).cast("Q")
engine.counts(haystacks, counts)
assert counts[0] + counts[1] == 7

matches = memoryview(bytearray(8 * 4 * 8)).cast("Q", (8, 4))
offsets = memoryview(bytearray(3 * 8)).cast("Q")
engine.find(haystacks, matches, offsets)

cover = sz.SubstringsEngine(sz.Strs(["he", "she"]), overlap_policy="leftmost-longest")
tape = memoryview(bytearray(32))
offsets = memoryview(bytearray(3 * 8)).cast("Q")
cover.replace(haystacks, sz.Strs(["HE", "SHE"]), tape, offsets)
assert bytes(tape[offsets[1]:offsets[2]]) == b"HErSHEy"
```

`bm25_scores` treats the vocabulary itself as the query, `needle_weights[i]` carrying needle `i`'s IDF or boost, and writes one score per haystack without ever materializing a per-term frequency row.
Term frequencies are raw overlapping counts, so the engine's own overlap policy does not apply there.
`document_lengths` defaults to each haystack's byte length, and `average_document_length` is required whenever `length_normalization` is positive.

```python
import stringzilla as sz

engine = sz.SubstringsEngine(sz.Strs(["cat", "dog"]))
weights = memoryview(bytearray(2 * 4)).cast("f")
weights[0], weights[1] = 1.0, 1.0
out = memoryview(bytearray(2 * 4)).cast("f")
engine.bm25_scores(sz.Strs(["cat and dog", "nothing here"]), weights, out, average_document_length=10.0)
assert out[1] == 0.0
```

### Engines on a GPU

Every engine has an `on_gpu` classmethod taking the same arguments as its constructor plus a `stream`, which is a `cudaStream_t` as an integer, or `0` for the current device's default stream.
Choosing a device is choosing that constructor: it prepares the batch where a kernel reaches it and resolves the launch geometry once, and every later round of that engine runs there.

```python
import stringzilla as sz

engine = sz.LevenshteinEngine.on_gpu(sz.Strs(["kitten", "saturday"]), stream=handle)
engine.distances(sz.Strs(["sitting", "sunday"]), device_out)
# cudaStreamSynchronize(handle) belongs here, before `device_out` is read.
```

A device round enqueues and returns, so its `out` buffer has to be memory the device reaches, has to outlive the launch, and must not be read before the caller joins the stream itself.
`SubstringsEngine.report` is written by the device too, so it obeys the same rule.
A host engine imposes no such requirement, and has always finished writing by the time it answers.

## UTF-8 Segmentation

`Str`, along with the matching module-level functions, exposes lazy iterators over Unicode boundaries.
Each yields `Str` views into the original buffer, so segmentation stays allocation-free, except `utf8_codepoints`, which yields `int` code points.

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
- `utf8_uncased_search(haystack, needle, start=0, end=len, validate=False)` — index of the first uncased match, or `-1`.
  For `str` inputs `start`/`end` and the result are codepoint offsets; for `bytes` inputs they are byte offsets.
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
- `sz.reset_capabilities(names)` — restrict the active backends to `names`, intersected with the hardware's actual capabilities; if the intersection is empty it falls back to `'serial'`.
  This updates `sz.__capabilities__` and re-points the dispatch table, which is useful for testing, benchmarking one backend, or reproducibility.

```python
import stringzilla as sz

sz.__capabilities__               # e.g. ('serial', 'haswell', 'skylake', 'ice')
sz.reset_capabilities(["serial"]) # force the scalar backend for this module
```

An engine resolves its tier once, when it is constructed, and records it, so `sz.reset_capabilities` affects the engines built after the call rather than the ones already holding a batch.
