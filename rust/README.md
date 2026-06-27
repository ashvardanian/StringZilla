# StringZilla for Rust

`stringzilla` is a Rust crate for fast string processing with SWAR, SIMD, and GPGPU acceleration.
It exposes two modules, each re-exported under a short alias for convenience:

- `stringzilla`, aliased `sz` — single-string operations: search, counting, splitting, hashing, sorting, UTF-8 segmentation, normalization, and case folding.
- `stringzillas`, aliased `szs` — batch/parallel engines: byte and UTF-8 Levenshtein distances, Needleman-Wunsch and Smith-Waterman alignment scores, and Min-Hash fingerprints, with CPU multi-threading and optional CUDA/ROCm offload.

The single-string module is `no_std`-friendly and SIMD-accelerated; most of its surface is exposed both as free functions in `sz` and as ergonomic extension-trait methods on any `AsRef<[u8]>` — `&str`, `String`, `&[u8]`, `Vec<u8>`, `Cow<str>`, and more.
The batch module is gated behind the parallel-backend features and operates over slices of strings, returning dense result matrices in unified memory.

## Installation

Add the crate with Cargo:

```sh
cargo add stringzilla
```

Or declare it in `Cargo.toml`:

```toml
[dependencies]
stringzilla = "4"
```

The crate ships the C/C++ sources and compiles them through a `build.rs` via `cc`, so no system StringZilla install is required.

### Feature Flags

| Feature            | Default | Effect                                             |
| ------------------ | ------- | -------------------------------------------------- |
| `std`              | yes     | `std` support, else `no_std`                       |
| `dynamic-dispatch` | yes     | Runtime SIMD dispatch; disable to bake in one tier |
| `cpus`             | no      | Multi-threaded CPU backend                         |
| `cuda`             | no      | CUDA GPU backend; implies `cpus`                   |
| `rocm`             | no      | ROCm GPU backend; implies `cpus`                   |

Without `std` the crate is `no_std`; `std` is also required for the `BuildSzHasher` integration with `HashMap`/`HashSet`.
The `cpus` backend compiles the `stringzillas` module and pulls in `allocator-api2` and `stringtape`.
The entire `stringzillas` module is compiled only when at least one of `cpus`, `cuda`, or `rocm` is enabled:

```toml
[dependencies]
stringzilla = { version = "4", features = ["cpus"] }   # CPU batch engines
# stringzilla = { version = "4", features = ["cuda"] } # CUDA-accelerated batch engines
```

Import either by full module name or by alias:

```rust
use stringzilla::sz;                       // single-string free functions
use stringzilla::sz::StringZillableBinary; // search/split extension methods
use stringzilla::sz::StringZillableUnary;  // hash/segmentation extension methods
use stringzilla::szs;                      // batch engines (needs cpus/cuda/rocm)
```

### Dynamic vs Compile-Time Dispatch

The `dynamic-dispatch` feature (on by default) controls how the C kernels select a SIMD backend, mirroring the C library's `SZ_DYNAMIC_DISPATCH` macro:

- __On (default):__ every ISA tier is compiled and the best one is chosen _at load_ through a dispatch table — the same model as the precompiled `stringzilla_shared` C library.
  One binary runs optimally on any CPU of the target architecture, at the cost of one indirect call per operation.
- __Off:__ each function is resolved _at compile time_ to the single newest tier the toolchain supports (for WebAssembly: `v128relaxed` when available, else `v128`).
  There is no table and no constructor, the unused tiers are dead-code-stripped, and the call goes straight to the kernel — the analog of including the header-only `stringzilla_header` in your own translation unit.

Removing the indirection trades flexibility for speed.
The resulting binary __bakes in one ISA tier__, so it is not portable across CPUs (a build that selected AVX-512 will fault on an older host) — it is intended for WebAssembly and for "build where you run" deployments.
In exchange, call-bound operations get faster: a short-input `sz_find` microbenchmark on this machine ran ~15% more calls per second without the table (≈205 → ≈240 Mcalls/s for 8–64 B inputs).
The win shrinks as inputs grow and the SIMD kernel, rather than the call overhead, dominates.

Disable it by opting out of default features (re-adding the ones you still want):

```toml
[dependencies]
# Compile-time dispatch: smaller, faster, but pinned to this machine's best ISA.
stringzilla = { version = "4", default-features = false, features = ["std"] }
```

`sz::dynamic_dispatch()` reports which mode the crate was built with.
This setting applies to the single-string `sz` library; the batch `stringzillas` engines (`cpus`/`cuda`/`rocm`) always use runtime dispatch.

## Types

The `sz` module exposes a handful of public value types used throughout the API.

- `Byteset` — a 256-bit set of bytes used by the byteset search and replace functions.
- `IndexSpan` — a `{ offset, length }` byte span, returned by uncased match iterators.
- `Hasher` — an incremental AES-based 64-bit hasher implementing `core::hash::Hasher`.
- `Sha256` — an incremental SHA-256 hasher with a one-shot convenience constructor.
- `BuildSzHasher` — a `std::hash::BuildHasher`, gated on feature `std`, for using `Hasher` with `HashMap`/`HashSet`.
- `Utf8View`, `Utf8Runes`, `Utf8Lines`, `Utf8Tokens`, `Utf8Words`, `Utf8Graphemes`, `Utf8Sentences`, `Utf8Linewraps` — lazy UTF-8 views and iterators.
- `Utf8UncasedNeedle`, `Utf8UncasedMatches`, `Utf8NormalForm` — uncased search and Unicode normalization helpers.
- `SemVer`, `Status`, `SmallCString` — version, error, and capability-string types.
- `ArgsortOptions` — knobs for sorting.

`Byteset` is constructed from bytes and supports inversion, useful for "not from" semantics:

```rust
use stringzilla::sz::{self, Byteset};

let vowels = Byteset::from("aeiou");
let mut punct = Byteset::new();
punct.add(',');
punct.add('.');
let everything = Byteset::new_ascii();         // all ASCII bytes set
let not_vowels = vowels.inverted();            // leaves `vowels` unchanged
let _ = (everything, not_vowels, punct);
```

`Byteset` API: `new()`, `new_ascii()`, `from_bytes(&[u8])` / `From<T: AsRef<[u8]>>`, `add(char)`, `add_u8(u8)`, `invert(&mut self)`, `inverted(&self) -> Byteset`.

`IndexSpan` API: `new(offset, length)`, `range() -> Range<usize>`, `extract<'a>(&self, &'a [u8]) -> &'a [u8]`, `end() -> usize`, plus public `offset` and `length` fields.

```rust
use stringzilla::sz::IndexSpan;

let span = IndexSpan::new(6, 5);
assert_eq!(span.range(), 6..11);
assert_eq!(span.end(), 11);
assert_eq!(span.extract(b"Hello World"), b"World");
```

## Searching and Counting

All search functions accept any `AsRef<[u8]>` and return byte offsets as `Option<usize>`.
They are available both as `sz::` free functions and as `StringZillableBinary` trait methods.

Free functions:

```rust
pub fn find<H: AsRef<[u8]>, N: AsRef<[u8]>>(haystack: H, needle: N) -> Option<usize>;
pub fn rfind<H: AsRef<[u8]>, N: AsRef<[u8]>>(haystack: H, needle: N) -> Option<usize>;

pub fn find_byteset<H: AsRef<[u8]>>(haystack: H, needles: Byteset) -> Option<usize>;
pub fn rfind_byteset<H: AsRef<[u8]>>(haystack: H, needles: Byteset) -> Option<usize>;

pub fn find_byte_from<H, N>(haystack: H, needles: N) -> Option<usize>;      // first byte in the set
pub fn rfind_byte_from<H, N>(haystack: H, needles: N) -> Option<usize>;     // last  byte in the set
pub fn find_byte_not_from<H, N>(haystack: H, needles: N) -> Option<usize>;  // first byte not in the set
pub fn rfind_byte_not_from<H, N>(haystack: H, needles: N) -> Option<usize>; // last  byte not in the set
```

```rust
use stringzilla::sz;

assert_eq!(sz::find("Hello, world!", "world"), Some(7));
assert_eq!(sz::rfind("Hello, world, world!", "world"), Some(14));

// First/last byte from a set of delimiters, or the first/last byte NOT in a set.
assert_eq!(sz::find_byte_from("Hello, world!", "aeiou"), Some(1));      // 'e'
assert_eq!(sz::rfind_byte_from("Hello, world!", "aeiou"), Some(8));     // 'o'
assert_eq!(sz::find_byte_not_from("Hello, world!", "aeiou"), Some(0));  // 'H'
assert_eq!(sz::rfind_byte_not_from("Hello, world!", "aeiou"), Some(12));// '!'
```

The `StringZillableBinary<'a, N>` trait provides the same operations as methods on the haystack.
The blanket impl covers every `T: AsRef<[u8]> + ?Sized`, so they work on `&str`, `String`, `&[u8]`, `Cow<str>`, and more:

```rust
use stringzilla::sz::StringZillableBinary;

let haystack = "Hello, world!";
assert_eq!(haystack.sz_find("world".as_bytes()), Some(7));
assert_eq!(haystack.sz_rfind("world".as_bytes()), Some(7));
assert_eq!(haystack.sz_find_byte_from("aeiou".as_bytes()), Some(1));
assert_eq!(haystack.sz_rfind_byte_from("aeiou".as_bytes()), Some(8));
assert_eq!(haystack.sz_find_byte_not_from("aeiou".as_bytes()), Some(0));
assert_eq!(haystack.sz_rfind_byte_not_from("aeiou".as_bytes()), Some(12));
```

Trait methods: `sz_find`, `sz_rfind`, `sz_find_byte_from`, `sz_rfind_byte_from`, `sz_find_byte_not_from`, `sz_rfind_byte_not_from`.

### Counting UTF-8 Characters

These two functions reason about codepoint boundaries without fully decoding the string.

```rust
pub fn count_utf8<T: AsRef<[u8]>>(text: T) -> usize;
pub fn find_nth_utf8<T: AsRef<[u8]>>(text: T, n: usize) -> Option<usize>;
```

`count_utf8` counts codepoint start bytes with SIMD; `find_nth_utf8` returns the byte offset of the Nth, zero-indexed, codepoint without decoding the whole string:

```rust
use stringzilla::sz;

assert_eq!(sz::count_utf8("你好世界"), 4);
assert_eq!(sz::count_utf8("Hello🌍"), 6);
assert_eq!(sz::find_nth_utf8("Hello🌍", 5), Some(5)); // 🌍 starts at byte 5
assert_eq!(sz::find_nth_utf8("Hello", 5), None);
```

### Comparison and Equality

`order` and `equal` are SIMD byte comparisons, while `utf8_uncased_order` compares under Unicode case folding.

```rust
pub fn order<A: AsRef<[u8]>, B: AsRef<[u8]>>(a: A, b: B) -> core::cmp::Ordering; // SIMD sz_order
pub fn equal<A: AsRef<[u8]>, B: AsRef<[u8]>>(a: A, b: B) -> bool;               // SIMD sz_equal
pub fn utf8_uncased_order<A, B>(a: A, b: B) -> core::cmp::Ordering;             // Unicode case-folded
```

```rust
use stringzilla::sz;
use std::cmp::Ordering;

assert_eq!(sz::order("apple", "banana"), Ordering::Less);
assert!(sz::equal("abc", "abc"));
assert_eq!(sz::utf8_uncased_order("Hello", "HELLO"), Ordering::Equal);
```

## Splitting and Partitioning

The crate exposes iterator-based match and split operations.
The `StringZillableBinary` trait yields these iterators; the underlying `FindMatches`, `RFindMatches`, `FindSplits`, and `RFindSplits` structs can also be constructed directly from a `MatcherType`.

Trait methods, each taking a borrowed needle `&'a N`:

```rust
fn sz_matches(&'a self, needle: &'a N) -> FindMatches<'a>;
fn sz_rmatches(&'a self, needle: &'a N) -> RFindMatches<'a>;
fn sz_splits(&'a self, needle: &'a N) -> FindSplits<'a>;
fn sz_rsplits(&'a self, needle: &'a N) -> RFindSplits<'a>;
fn sz_find_first_of(&'a self, needles: &'a N) -> FindMatches<'a>;
fn sz_find_last_of(&'a self, needles: &'a N) -> RFindMatches<'a>;
fn sz_find_first_not_of(&'a self, needles: &'a N) -> FindMatches<'a>;
fn sz_find_last_not_of(&'a self, needles: &'a N) -> RFindMatches<'a>;
```

Matching yields the matched sub-slices; splitting yields the segments between matches:

```rust
use stringzilla::sz::StringZillableBinary;

let haystack = b"hello world hello universe";
let needle = b"hello";
let fwd: Vec<&[u8]> = haystack.sz_matches(needle).collect();
let rev: Vec<&[u8]> = haystack.sz_rmatches(needle).collect();
assert_eq!(fwd, vec![b"hello", b"hello"]);
assert_eq!(rev, vec![b"hello", b"hello"]);

let csv = b"a,b,c,d";
let splits: Vec<&[u8]> = csv.sz_splits(b",").collect();
let rsplits: Vec<&[u8]> = csv.sz_rsplits(b",").collect();
assert_eq!(splits, vec![b"a", b"b", b"c", b"d"]);
assert_eq!(rsplits, vec![b"d", b"c", b"b", b"a"]);
```

`find_first_of` / `find_last_of` match any byte present in the set; the `not_of` variants match any byte absent from the set:

```rust
use stringzilla::sz::StringZillableBinary;

let haystack = b"hello world";
let any: Vec<&[u8]> = haystack.sz_find_first_of(b"or").collect();
assert_eq!(any, vec![b"o", b"o", b"r"]);

let other: Vec<&[u8]> = b"aabbbcccd".sz_find_first_not_of(b"ab").collect();
assert_eq!(other, vec![b"c", b"c", b"c", b"d"]);
```

By default the split iterators __keep__ empty segments, mirroring `str::split` / `str::rsplit`; chain `.skip_empty()` to drop zero-length segments:

```rust
use stringzilla::sz::StringZillableBinary;

let kept: Vec<&[u8]> = b"a,,b,".sz_splits(b",").collect();
assert_eq!(kept, vec![b"a", &b""[..], b"b", &b""[..]]);

let nonempty: Vec<&[u8]> = b"a,,b,".sz_splits(b",").skip_empty().collect();
assert_eq!(nonempty, vec![b"a", b"b"]);
```

For direct construction, `MatcherType<'a>` selects the search mode and is paired with `FindMatches::new(haystack, matcher, include_overlaps)`, `RFindMatches::new(...)`, `FindSplits::new(haystack, matcher)`, or `RFindSplits::new(...)`:

```rust
use stringzilla::sz::{MatcherType, FindMatches, FindSplits};

let overlapping: Vec<&[u8]> = FindMatches::new(b"aaaa", MatcherType::Find(b"aa"), true).collect();
assert_eq!(overlapping, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);

let split: Vec<&[u8]> = FindSplits::new(b",a;;b,", MatcherType::FindFirstOf(b",;")).skip_empty().collect();
assert_eq!(split, vec![b"a", b"b"]);
```

`MatcherType` variants: `Find`, `RFind`, `FindFirstOf`, `FindLastOf`, `FindFirstNotOf`, `FindLastNotOf`.

## Trimming and Translating

The crate provides byte-level buffer transforms, lookup-table translation, in-place replacement, and Unicode case folding / normalization.

### Lookup Table Translation

`lookup` maps every byte of a source through a 256-entry table into a destination, while `lookup_inplace` rewrites a buffer in place.

```rust
pub fn lookup<T: AsMut<[u8]>, S: AsRef<[u8]>>(target: &mut T, source: &S, table: [u8; 256]);
pub fn lookup_inplace<T: AsMut<[u8]>>(buffer: &mut T, table: [u8; 256]);
```

`lookup` maps every byte through a 256-entry table; for example to lowercase ASCII:

```rust
use stringzilla::sz;

let mut to_lower: [u8; 256] = core::array::from_fn(|i| i as u8);
for (upper, lower) in ('A'..='Z').zip('a'..='z') {
    to_lower[upper as usize] = lower as u8;
}
let mut text = *b"HELLO WORLD!";
sz::lookup_inplace(&mut text, to_lower);
assert_eq!(&text, b"hello world!");
```

### Buffer Fill, Copy, and Move

`fill` sets every byte of a buffer to one value, `copy` writes a source into a target, and `move_` does the same for overlapping regions.

```rust
pub fn fill<T: AsMut<[u8]>>(target: &mut T, value: u8);
pub fn copy<T: AsMut<[u8]>, S: AsRef<[u8]>>(target: &mut T, source: &S);
pub fn move_<T: AsMut<[u8]>, S: AsRef<[u8]>>(target: &mut T, source: &S);
```

`copy` and `move_` assert the target is at least as long as the source; `move_` tolerates overlapping regions.

### In Place Replacement

These rewrite a `Vec<u8>` in place, replacing either a literal needle or any byte from a `Byteset` with a replacement slice.

```rust
pub fn try_replace_all(buffer: &mut Vec<u8>, needle: &[u8], replacement: &[u8]) -> Result<usize, Status>;
pub fn try_replace_all_byteset(buffer: &mut Vec<u8>, byteset: Byteset, replacement: &[u8]) -> Result<usize, Status>;
```

Both replace all non-overlapping occurrences in place, returning the replacement count.
Equal-length replacements overwrite, shorter ones compact forward without allocating, and longer ones resize once and rewrite from the back:

```rust
use stringzilla::sz::{self, Byteset};

let mut buffer = b"a-b-c".to_vec();
let n = sz::try_replace_all(&mut buffer, b"-", b"__").unwrap();
assert_eq!(n, 2);
assert_eq!(buffer, b"a__b__c");

let mut spaced = b"a, b ,c".to_vec();
sz::try_replace_all_byteset(&mut spaced, Byteset::from(", "), b"").unwrap();
assert_eq!(spaced, b"abc");
```

### Case Folding and Normalization

`utf8_uncased_fold` case-folds into a destination buffer, `utf8_norm` applies a Unicode normal form, and `utf8_find_denormalized` checks conformance without rewriting.

```rust
pub fn utf8_uncased_fold<T: AsRef<[u8]>, D: AsMut<[u8]>>(source: T, destination: &mut D) -> usize;
pub fn utf8_norm<T: AsRef<[u8]>, D: AsMut<[u8]>>(source: T, form: Utf8NormalForm, destination: &mut D) -> usize;
pub fn utf8_find_denormalized<T: AsRef<[u8]>>(source: T, form: Utf8NormalForm) -> Option<usize>;
```

`utf8_uncased_fold` applies Unicode case folding, for example `ß` → `ss`, returning the number of bytes written.
`utf8_norm` normalizes to one of the four `Utf8NormalForm` variants: `Nfd`, `Nfc`, `Nfkd`, and `Nfkc`.
`utf8_find_denormalized` is a fast check returning the byte offset of the first non-conforming byte, or `None` if already normalized:

```rust
use stringzilla::sz::{self, Utf8NormalForm};

let mut dest = [0u8; 32];
let len = sz::utf8_uncased_fold("HELLO WORLD", &mut dest);
assert_eq!(&dest[..len], b"hello world");

// NFC check: a decomposed "café" (e + combining acute) violates NFC.
assert!(sz::utf8_find_denormalized("cafe\u{0301}", Utf8NormalForm::Nfc).is_some());
assert!(sz::utf8_find_denormalized("caf\u{00E9}", Utf8NormalForm::Nfc).is_none());

let mut out = vec![0u8; "cafe\u{0301}".len() * 18];
let n = sz::utf8_norm("cafe\u{0301}", Utf8NormalForm::Nfc, &mut out);
assert_eq!(&out[..n], "caf\u{00E9}".as_bytes());
```

Destination buffers must be sized for worst-case expansion: `source.len() * 3` for folding and `source.len() * 18` for normalization.

### Uncased UTF-8 Search

`utf8_uncased_search` locates a needle in a haystack under Unicode case folding.

```rust
pub fn utf8_uncased_search<H: AsRef<[u8]>, N: Utf8UncasedNeedleArg>(haystack: H, needle: N) -> Option<(usize, usize)>;
```

Returns `Some((offset, matched_length))`, where the matched length may differ from the needle length due to case folding — `ß` matching `SS`, for instance.
A reusable `Utf8UncasedNeedle` caches needle metadata across searches, and `Utf8UncasedMatches` iterates all matches as `IndexSpan`s:

```rust
use stringzilla::sz::{self, Utf8UncasedNeedle, Utf8UncasedMatches, IndexSpan};

assert_eq!(sz::utf8_uncased_search("Hello WORLD", "world"), Some((6, 5)));

let needle = Utf8UncasedNeedle::new(b"hello");
assert_eq!(sz::utf8_uncased_search(b"HELLO there", &needle), Some((0, 5)));

let spans: Vec<IndexSpan> = Utf8UncasedMatches::new(b"Hello hello", b"hello").collect();
assert_eq!(spans, vec![IndexSpan::new(0, 5), IndexSpan::new(6, 5)]);
```

## Hashing and Checksums

These free functions cover a byte checksum, seeded and unseeded 64-bit hashes, and a multi-seed hash that fills an output slice in one pass.

```rust
pub fn bytesum<T: AsRef<[u8]>>(text: T) -> u64;
pub fn hash<T: AsRef<[u8]>>(text: T) -> u64;
pub fn hash_with_seed<T: AsRef<[u8]>>(text: T, seed: u64) -> u64;
pub fn hash_multiseed_into<T: AsRef<[u8]>>(text: T, seeds: &[u64], out: &mut [u64]);
```

`bytesum` is an order-insensitive byte sum; `hash` / `hash_with_seed` are order-sensitive AES-based 64-bit hashes.
`hash_multiseed_into` hashes one input under many seeds in a single pass, handy for MinHash, Count-Min sketches, and Bloom/cuckoo filters; it panics if `out.len() != seeds.len()`:

```rust
use stringzilla::sz;

assert_eq!(sz::bytesum("hi"), 209);
assert_ne!(sz::hash("Hello"), sz::hash("World"));
assert_eq!(sz::hash_with_seed("Hello", 42), sz::hash_with_seed("Hello", 42));

let seeds = [1u64, 2, 3, 4];
let mut out = [0u64; 4];
sz::hash_multiseed_into("token", &seeds, &mut out);
```

These are also available as `StringZillableUnary` methods on any `AsRef<[u8]>`:

```rust
use stringzilla::sz::StringZillableUnary;

assert_eq!(b"Hello".sz_bytesum(), 500);
assert_ne!(b"Hello".sz_hash(), b"World".sz_hash());
```

### Incremental Hashing

`Hasher` is an incremental AES-based hasher implementing `core::hash::Hasher`:

```rust
use stringzilla::sz::Hasher;

let mut hasher = Hasher::new(123);
hasher.update(b"Hello, ").update(b"world!");
assert_eq!(hasher.digest(), Hasher::new(123).update(b"Hello, world!").digest());
```

`Hasher` API: `new(seed)`, `update(&mut self, &[u8]) -> &mut Self`, `digest(&self) -> u64`, plus the full `core::hash::Hasher` trait.

With the `std` feature, `BuildSzHasher` plugs the AES-based hash into the standard `HashMap` and `HashSet` as their hasher, replacing the default SipHash.
For string keys this swaps a scalar hash for StringZilla's vectorized one while keeping the entire collection API unchanged.
The seed defaults to `0` and is deterministic across runs, so pass a random seed through `with_seed` when you want per-process resistance to hash-flooding:

```rust
use std::collections::{HashMap, HashSet};
use stringzilla::sz::BuildSzHasher;

let mut counts: HashMap<&str, i32, BuildSzHasher> = HashMap::with_hasher(BuildSzHasher::with_seed(0));
for word in ["apple", "banana", "apple"] { *counts.entry(word).or_insert(0) += 1; }
assert_eq!(counts["apple"], 2);

let mut seen: HashSet<&str, BuildSzHasher> = HashSet::with_hasher(BuildSzHasher::with_seed(42));
seen.insert("apple");
seen.insert("banana");
assert!(seen.contains("apple"));
assert_eq!(seen.len(), 2);
```

### SHA-256 and HMAC

`hmac_sha256` computes a keyed HMAC-SHA-256 over a message, returning a 32-byte tag.

```rust
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32];
```

`Sha256` supports one-shot and incremental hashing, returning a 32-byte digest:

```rust
use stringzilla::sz::{Sha256, hmac_sha256};

let digest = Sha256::hash(b"Hello, world!");
assert_eq!(digest.len(), 32);

let mut hasher = Sha256::new();
hasher.update(b"Hello, ").update(b"world!");
assert_eq!(hasher.digest(), digest);

let mac = hmac_sha256(b"secret_key", b"important message");
assert_eq!(mac.len(), 32);
```

`Sha256` API: `new()`, `update(&mut self, &[u8]) -> &mut Self`, `digest(&self) -> [u8; 32]`, `hash(&[u8]) -> [u8; 32]`.

## Sorting, Sampling, and Shuffling

### Argsort

`argsort` writes the sorted permutation of a slice into a caller-provided `order` slice, while `argsort_by` sorts by a byte-slice key extracted from each index.

```rust
pub fn argsort<T: AsRef<[u8]>>(data: &[T], order: &mut [SortedIdx], options: ArgsortOptions) -> Result<(), Status>;
pub fn argsort_by<F, A>(mapper: F, order: &mut [SortedIdx], options: ArgsortOptions) -> Result<(), Status>
where F: Fn(usize) -> A, A: AsRef<[u8]>;
```

`argsort` writes the sorting permutation of `data` into a caller-supplied `order` buffer of length at least `data.len()`.
`argsort_by` infers the element count from the `order` slice and sorts by a caller-provided byte-slice key, ideal for sorting structs by a field.
`SortedIdx` is an alias for `usize`.

`ArgsortOptions` is a builder; the default is a full, ascending, byte-lexicographic, __stable__ sort:

- `reversed()` / field `reverse: bool` — descending order, still stable on ties.
- `uncased()` / field `uncased: bool` — order under Unicode case folding instead of raw bytes.
- `top(k)` / field `top: Option<usize>` — only fully order the leading `k` elements, a top-K or partial sort.

```rust
use stringzilla::sz::{self, ArgsortOptions};

let fruits = ["banana", "apple", "cherry"];
let mut order = [0; 3];
sz::argsort(&fruits, &mut order, Default::default()).unwrap();
assert_eq!(&order, &[1, 0, 2]); // apple, banana, cherry

// Descending + uncased
let labels = ["beta", "Alpha", "BETA"];
let mut order = [0; 3];
sz::argsort(&labels, &mut order, ArgsortOptions::default().reversed().uncased()).unwrap();

// Sort structs by a key
struct Person { name: &'static str }
let people = [Person { name: "Charlie" }, Person { name: "Alice" }, Person { name: "Bob" }];
let mut order = [0; 3];
sz::argsort_by(|i| people[i].name.as_bytes(), &mut order, Default::default()).unwrap();
assert_eq!(&order, &[1, 2, 0]); // Alice, Bob, Charlie
```

### Intersection, an Inner Join

`intersection` matches two collections directly, while `intersection_by` matches by a byte-slice key extracted from each index, both writing the matched positions of each side.

```rust
pub fn intersection<T: AsRef<[u8]>>(data1: &[T], data2: &[T], seed: u64,
    positions1: &mut [SortedIdx], positions2: &mut [SortedIdx]) -> Result<usize, Status>;
pub fn intersection_by<F, G, A, B>(mapper1: F, mapper2: G, seed: u64,
    positions1: &mut [SortedIdx], positions2: &mut [SortedIdx]) -> Result<usize, Status>
where F: Fn(usize) -> A, A: AsRef<[u8]>, G: Fn(usize) -> B, B: AsRef<[u8]>;
```

Both compute the intersection of two collections, writing the matching positions into output buffers each sized at least `min(len1, len2)`, and returning the intersection size:

```rust
use stringzilla::sz;

let set1 = ["banana", "apple", "cherry"];
let set2 = ["cherry", "orange", "pineapple", "banana"];
let (mut p1, mut p2) = ([0; 3], [0; 3]);
let n = sz::intersection(&set1, &set2, 0, &mut p1, &mut p2).unwrap();
assert_eq!(n, 2); // "banana" and "cherry"
```

### Random Fill

`fill_random` overwrites a buffer with deterministic pseudo-random bytes derived from a `nonce`.

```rust
pub fn fill_random<T: AsMut<[u8]>>(buffer: &mut T, nonce: u64);
```

`fill_random` overwrites a buffer with deterministic pseudo-random bytes for a given `nonce` — the same nonce always yields the same output — useful for generating test data:

```rust
use stringzilla::sz;

let mut a = vec![0u8; 10];
let mut b = vec![1u8; 10];
sz::fill_random(&mut a, 42);
sz::fill_random(&mut b, 42);
assert_eq!(a, b); // identical nonce → identical bytes
```

## Edit Distances and Alignment Scores

The `szs` module, gated behind feature `cpus`, `cuda`, or `rocm`, provides batch engines that compute dense cross-product matrices over slices of strings.
Every engine is created against a `DeviceScope` and exposes `compute`, `compute_symmetric`, and `compute_into`.

`compute<T, S>` and `compute_symmetric<T, S>` are generic over `T: AsRef<[S]>` and `S: AsRef<[u8]>`, so they accept `Vec<&str>`, `&[String]`, `Vec<Vec<u8>>`, and similar.
Results are returned as `UnifiedMat<usize>` for distances or `UnifiedMat<isize>` for scores, a row-major `queries × candidates` matrix indexable with `matrix[(query, candidate)]`, with `dimensions()`, `queries_count()`, `candidates_count()`, `row(i)`, `row_stride()`, and `as_slice()` accessors.

### Levenshtein Distances, Byte and UTF-8

Each engine is constructed with match, mismatch, and gap open/extend costs, then exposes `compute` for a cross-product matrix and `compute_symmetric` for self-similarity.

```rust
LevenshteinDistances::new(device, match_cost: i8, mismatch_cost: i8, open_cost: i8, extend_cost: i8)
    -> Result<Self, Error>;
LevenshteinDistancesUtf8::new(device, match_cost: i8, mismatch_cost: i8, open_cost: i8, extend_cost: i8)
    -> Result<Self, Error>;

fn compute<T, S>(&self, device: &DeviceScope, queries: T, candidates: T) -> Result<UnifiedMat<usize>, Error>;
fn compute_symmetric<T, S>(&self, device: &DeviceScope, sequences: T) -> Result<UnifiedMat<usize>, Error>;
```

`LevenshteinDistances` works at the byte level; `LevenshteinDistancesUtf8` operates on Unicode codepoints and is correct for international text.
Costs are: `match_cost`, typically `0`, then `mismatch_cost`, gap `open_cost`, and gap `extend_cost`.

```rust
use stringzilla::szs::{DeviceScope, LevenshteinDistances, LevenshteinDistancesUtf8};

let device = DeviceScope::default().unwrap();
let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();

let queries = vec!["cat", "dog"];
let candidates = vec!["bat", "fog", "word"];
let matrix = engine.compute(&device, &queries, &candidates).unwrap();
assert_eq!(matrix.dimensions(), (2, 3));
assert_eq!(matrix[(0, 0)], 1); // cat vs bat

// Symmetric self-similarity (square matrix, symmetric, zero diagonal).
let words = vec!["cat", "bat", "rat"];
let sym = engine.compute_symmetric(&device, &words).unwrap();
assert_eq!(sym[(0, 1)], sym[(1, 0)]);

// Unicode-aware variant.
let utf8 = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1).unwrap();
let a = vec!["Hello", "こんにちは"];
let b = vec!["Hallo", "こんばんは"];
let _ = utf8.compute(&device, &a, &b).unwrap();
```

### Needleman-Wunsch and Smith-Waterman Scores

Each engine is built from a substitution scheme and gap costs, then exposes `compute` and `compute_symmetric` returning signed-score matrices.

```rust
NeedlemanWunschScores::new(device, byte_to_class: &[u8; 256], class_substitution_costs: &[[i8; 32]; 32],
    open_cost: i8, extend_cost: i8) -> Result<Self, Error>;
SmithWatermanScores::new(device, byte_to_class: &[u8; 256], class_substitution_costs: &[[i8; 32]; 32],
    open_cost: i8, extend_cost: i8) -> Result<Self, Error>;

fn compute<T, S>(&self, device, queries: T, candidates: T) -> Result<UnifiedMat<isize>, Error>;
fn compute_symmetric<T, S>(&self, device, sequences: T) -> Result<UnifiedMat<isize>, Error>;
```

Both take a substitution scheme: a 256-entry `byte_to_class` map sending each byte to one of 32 classes, plus a `32 × 32` `class_substitution_costs` matrix, then gap `open_cost` and `extend_cost`, typically negative.
Needleman-Wunsch is global alignment; Smith-Waterman is local alignment.
Scores are returned as `UnifiedMat<isize>`, where higher is more similar.

Two helpers build a compact diagonal scheme:

```rust
pub fn error_costs_classes_diagonal(match_score: i8, mismatch_score: i8) -> ([u8; 256], [[i8; 32]; 32]);
pub fn error_costs_classes_unary() -> ([u8; 256], [[i8; 32]; 32]); // == diagonal(0, -1)
```

```rust
use stringzilla::szs::{DeviceScope, NeedlemanWunschScores, SmithWatermanScores, error_costs_classes_diagonal};

let device = DeviceScope::default().unwrap();
let (byte_to_class, class_costs) = error_costs_classes_diagonal(2, -1);

let nw = NeedlemanWunschScores::new(&device, &byte_to_class, &class_costs, -2, -1).unwrap();
let queries = vec!["ATCGATCG", "GGCCTTAA"];
let candidates = vec!["ATCGATCC", "GGCCTTAA"];
let scores = nw.compute(&device, &queries, &candidates).unwrap();
assert_eq!(scores.dimensions(), (2, 2));

let sw = SmithWatermanScores::new(&device, &byte_to_class, &class_costs, -3, -1).unwrap();
let _local = sw.compute(&device, &queries, &candidates).unwrap();
```

### Computing into Preallocated Tapes

For GPU work or to avoid reallocation, every engine also offers `compute_into`, which accepts an `AnyBytesTape<'a>` — an owned `BytesTape` or a zero-copy view, with 32- or 64-bit offsets — an optional candidates tape that is `None` for symmetric runs, and a `&mut UnifiedMat<...>`.
The `compute` / `compute_symmetric` wrappers handle tape construction automatically, so most callers never touch `compute_into` directly.

## Rolling Fingerprints

`Fingerprints` is created through a builder and its `compute` returns the Min-Hash signatures and Count-Min-Sketch counts for a batch of strings.

```rust
Fingerprints::builder() -> FingerprintsBuilder;
fn compute<T, S>(&self, device: &DeviceScope, strings: T, dimensions: usize)
    -> Result<(UnifiedVec<u32>, UnifiedVec<u32>), Error>;
```

`Fingerprints` computes Min-Hash signatures and Count-Min-Sketch frequencies for locality-sensitive similarity, deduplication, and clustering.
It is configured through `FingerprintsBuilder`:

- `new()` — defaults: `dimensions = 1024`, `seed = 0`, hardware-optimized window widths.
- `binary()` — 256-symbol alphabet of arbitrary bytes.
- `ascii()` — 128-symbol alphabet.
- `dna()` — 4-symbol alphabet of nucleotides.
- `protein()` — amino-acid alphabet.
- `alphabet_size(size: usize)` — explicit alphabet size.
- `window_widths(widths: &[usize])` — n-gram window widths to hash.
- `dimensions(dimensions: usize)` — number of hash functions per fingerprint.
- `seed(seed: u64)` — reproducibility seed.
- `build(device: &DeviceScope) -> Result<Fingerprints, Error>`.

`compute` returns `(min_hashes, min_counts)`, each a `UnifiedVec<u32>` laid out as `num_strings × dimensions`, where entry `i * dimensions + j` is the j-th hash/count of string `i`:

```rust
use stringzilla::szs::{Fingerprints, DeviceScope};

let device = DeviceScope::default().unwrap();
let dimensions = 256;
let engine = Fingerprints::builder()
    .ascii()
    .dimensions(dimensions)
    .build(&device)
    .unwrap();

let documents = vec![
    "The quick brown fox jumps over the lazy dog",
    "A quick brown fox leaps over a lazy dog",
];
let (hashes, _counts) = engine.compute(&device, &documents, dimensions).unwrap();

// Estimate Jaccard similarity between the two documents.
let matches = (0..dimensions).filter(|&i| hashes[i] == hashes[dimensions + i]).count();
let similarity = matches as f64 / dimensions as f64;
println!("Estimated Jaccard similarity: {similarity:.3}");
```

`Fingerprints` also exposes `compute_into` for writing into caller-provided buffers via an `AnyBytesTape`.

## UTF-8 Segmentation

A single emoji such as the flag `🇺🇸` is 8 bytes and 2 codepoints, yet a reader sees one character — one grapheme cluster.
StringZilla draws every one of these boundaries: `sz_utf8_runes` (`Utf8View`) walks codepoints, while `sz_utf8_graphemes` (`Utf8Graphemes`) walks user-perceived clusters.
The `StringZillableUnary` trait turns any `AsRef<[u8]>` into a broad family of lazy UTF-8 iterators and views, each yielding borrowed byte sub-slices or `char`s on demand:

```rust
fn sz_utf8_runes(&self) -> Utf8View<'_>;       // codepoints / random codepoint access
fn sz_utf8_lines(&self) -> Utf8Lines<'_>;      // split on hard newlines (CRLF = one delimiter)
fn sz_utf8_tokens(&self) -> Utf8Tokens<'_>;    // split on Unicode whitespace
fn sz_utf8_words(&self) -> Utf8Words<'_>;      // UAX-29 words
fn sz_utf8_graphemes(&self) -> Utf8Graphemes<'_>;  // UAX-29 grapheme clusters
fn sz_utf8_sentences(&self) -> Utf8Sentences<'_>;  // UAX-29 sentences
fn sz_utf8_linewraps(&self) -> Utf8Linewraps<'_>;  // UAX-14 line-break opportunities
```

Every member of this family is lazy and zero-copy.
`Utf8View`, `Utf8Runes`, `Utf8Lines`, `Utf8Tokens`, `Utf8Words`, `Utf8Graphemes`, `Utf8Sentences`, and `Utf8Linewraps`, together with the `sz_splits` / `sz_rsplits` iterators, all borrow from the source string and yield `&[u8]` or `&str` slices on demand without allocating.
There is no backing vector and no per-element heap buffer.
Contrast this with the standard library, where collecting into a `Vec<String>` allocates the vector and a fresh heap buffer for every element, and even a `Vec<&str>` allocates the backing vector up front.
Because these iterators borrow from the input, you can stream over millions of words or grapheme clusters of a large document with effectively zero per-element allocation, and the borrow lifetimes keep the yielded slices zero-copy.

`Utf8View` offers O(1) construction with lazy, cached `len()` for the codepoint count, `offset_of(n)` for the byte offset of the Nth codepoint, and `iter()` for batched `char` iteration:

```rust
use stringzilla::sz::StringZillableUnary;

let view = "Hello🌍".sz_utf8_runes();
assert_eq!(view.len(), 6);
assert_eq!(view.offset_of(5), Some(5));
let chars: Vec<char> = view.iter().collect();
assert_eq!(chars, vec!['H', 'e', 'l', 'l', 'o', '🌍']);
```

The boundary iterators `words`, `graphemes`, `sentences`, and `linewraps` __tile__ the input — every byte belongs to exactly one segment, so consecutive segments are contiguous and no empty slices appear:

```rust
use stringzilla::sz::StringZillableUnary;

let words: Vec<&[u8]> = b"Hi, world".sz_utf8_words().collect();
assert_eq!(words, vec![&b"Hi"[..], &b","[..], &b" "[..], &b"world"[..]]);

let sentences: Vec<&[u8]> = b"Hi. Bye.".sz_utf8_sentences().collect();
assert_eq!(sentences, vec![&b"Hi. "[..], &b"Bye."[..]]);
```

`lines` splits on the 8 Unicode hard newlines, treating CRLF as a single delimiter and excluding the newline from each segment, while `tokens` splits on the 25 Unicode whitespace characters and __keeps__ empty segments by default; chain `.skip_empty()` for `str::split_whitespace`-style behavior:

```rust
use stringzilla::sz::StringZillableUnary;

let lines: Vec<&str> = "Hello\nWorld\r\nRust"
    .sz_utf8_lines()
    .map(|l| std::str::from_utf8(l).unwrap())
    .collect();
assert_eq!(lines, vec!["Hello", "World", "Rust"]);

let tokens: Vec<&[u8]> = b"  hi  ".sz_utf8_tokens().skip_empty().collect();
assert_eq!(tokens, vec![&b"hi"[..]]);
```

Every iterator is also constructible directly, for example `Utf8Words::new(text)`.
Each exposes a `STEPS` const-generic batch-size knob via `with_steps`, written with a turbofish such as `Utf8Runes::<256>::with_steps(bytes)`, that trades buffer size for FFI-call amortization without changing the yielded results.
The default batch size is the public constant `ITERATORS_DEFAULT_STEPS = 64`.

### Low Level Decoding

`utf8_decode` unpacks UTF-8 bytes into a UTF-32 `runes` buffer, returning how many bytes were consumed and codepoints written.

```rust
pub fn utf8_decode(text: &[u8], runes: &mut [u32]) -> (usize, usize); // (bytes_consumed, runes_unpacked)
```

`utf8_decode` decodes UTF-8 into UTF-32 codepoints, filling the output buffer or draining the input per call.
For inputs larger than the buffer, loop and resume at `bytes_consumed`.
It is total: ill-formed bytes decode to U+FFFD, so every value written is a valid Unicode scalar:

```rust
use stringzilla::sz;

let mut runes = [0u32; 16];
let (bytes, count) = sz::utf8_decode("Hello".as_bytes(), &mut runes);
assert_eq!((bytes, count), (5, 5));
assert_eq!(runes[0], 'H' as u32);
```

## Runtime Dispatch and Capabilities

The `sz` module reports the compiled version and the SIMD capabilities chosen at runtime:

```rust
pub fn dynamic_dispatch() -> bool;        // was the library built with runtime dispatch?
pub fn version() -> SemVer;               // { major, minor, patch }
pub fn capabilities() -> SmallCString;    // human-readable list of active backends
```

```rust
use stringzilla::sz;

let v = sz::version();
println!("StringZilla {}.{}.{}", v.major, v.minor, v.patch);
println!("dynamic dispatch: {}", sz::dynamic_dispatch());
println!("capabilities: {}", sz::capabilities().as_str());
```

The `szs` module reports the same metadata for the batch engines, plus a one-line backend summary and per-`DeviceScope` introspection:

```rust
pub fn version() -> SemVer;
pub fn capabilities() -> SmallCString;
pub fn backend_info() -> &'static str; // e.g. "Multi-threaded CPU backend enabled"
```

A `DeviceScope` selects the execution backend.
`DeviceScope::default()` auto-detects the best available hardware, `cpu_cores(n)` forces `n` CPU threads where `0` means all cores, and `gpu_device(i)` targets GPU `i` and requires the `cuda` or `rocm` feature:

```rust
use stringzilla::szs::{self, DeviceScope};

let device = DeviceScope::default().unwrap();
let cpu = DeviceScope::cpu_cores(4).unwrap();
assert_eq!(cpu.get_cpu_cores().unwrap(), 4);
assert!(!cpu.is_gpu());

println!("backend: {}", szs::backend_info());
println!("capabilities: {}", szs::capabilities().as_str());

match DeviceScope::gpu_device(0) {
    Ok(gpu) => println!("using GPU {}", gpu.get_gpu_device().unwrap()),
    Err(e)  => println!("GPU unavailable: {e:?}"),
}
```

`DeviceScope` API: `default()`, `cpu_cores(usize)`, `gpu_device(usize)`, `get_capabilities() -> Result<Capability, Error>`, `get_cpu_cores() -> Result<usize, Error>`, `get_gpu_device() -> Result<usize, Error>`, `is_gpu() -> bool`.
Batch-engine failures surface as `szs::Error`, a `status` plus an optional message, which converts from the shared `Status` enum.
