# StringZilla for Rust

`stringzilla` is a Rust crate for fast string processing with SWAR, SIMD, and GPGPU acceleration.
It exposes one module, `stringzilla`, re-exported under the short alias `sz`, covering two kinds of work:

- __single-string operations__ — search, counting, splitting, hashing, sorting, UTF-8 segmentation, normalization, and case folding.
- __batch engines__ — Levenshtein distances, window overlap, and multi-pattern substring search, each a stateful cross-product engine prepared once from a batch of queries and reused across every later round.

The crate is `no_std`-friendly and SIMD-accelerated; most of the single-string surface is exposed both as free functions in `sz` and as ergonomic extension-trait methods on any `AsRef<[u8]>` — `&str`, `String`, `&[u8]`, `Vec<u8>`, `Cow<str>`, and more.
The engines take slices of anything `AsRef<[u8]>` and write into caller-owned buffers at a caller-chosen stride, so a pipeline allocates once and reuses those buffers across every batch.

## Installation

Add the crate with Cargo:

```sh
cargo add stringzilla
```

Or declare it in `Cargo.toml`:

```toml
[dependencies]
stringzilla = "5"
```

The crate ships the C sources, and its `build.rs` builds them into `stringzilla_static` through CMake, so no system StringZilla install is required, but CMake and a C and C++ compiler are.
`STRINGZILLA_LIBRARY_DIR` names a directory holding a `stringzilla_static` CMake already built, like a parent project's build tree, and skips that build.

### Feature Flags

- `std`, on by default: `std` support, without which the crate is `no_std`.
- `cuda`: the CUDA backend, so `Capabilities::cuda_count_devices` counts devices and engines can be built on a `Stream` of `Capabilities::cuda_enabled(ordinal)`.
- `rocm`: the ROCm backend, compiled through HIP, so the `Capabilities::rocm_*` producers answer the same way.
- `metal`: the Metal backend on Apple platforms, linking the Metal and Foundation frameworks, so the `Capabilities::metal_*` producers answer the same way.

Every GPU feature implies `std`, which is also required for the `BuildSzHasher` integration with `HashMap`/`HashSet`.
Without one, the engines are host-only and every other verb is unchanged:

```toml
[dependencies]
stringzilla = { version = "5", features = ["cuda"] } # CUDA-accelerated engines
```

Import by full module name or by alias:

```rust
use stringzilla::sz;                       // free functions and batch engines
use stringzilla::sz::StringZillableBinary; // search/split extension methods
use stringzilla::sz::StringZillableUnary;  // hash/segmentation extension methods
```

### SIMD Dispatch

Every call picks its kernel from the capability mask it passes, `Capabilities::cpu_enabled()` (see [Runtime Dispatch and Capabilities](#runtime-dispatch-and-capabilities)) — the same `_best` dispatch points as the precompiled `stringzilla_shared` C library.
One binary runs optimally on any CPU of the target architecture, at the cost of one pick per operation.
CMake decides which capabilities to compile in, as for every other binding, by try-compiling `probes/<capability>.c` — tiny programs calling one of the capability's real kernels header-only at the baseline flags, so broken or old toolchains are caught up front.
Every capability the toolchain can emit is built, and the enabled mask leaves out whatever the CPU lacks.
Where the OS cannot be asked, as on OS-less targets, the CPU reports only the capabilities the C compiler's own flags guarantee.
A WebAssembly module carries the one SIMD kit the Rust target declares: `+relaxed-simd` gives `v128relaxed`, `+simd128` gives `v128`, and neither gives `serial`.

The build script forwards these environment variables to CMake, and rebuilds when one changes:

- `STRINGZILLA_TARGET_<CAPABILITY>`, like `STRINGZILLA_TARGET_SVE2=0 cargo build`, forces one capability on or off, though never past what the toolchain can compile.
- `STRINGZILLA_TARGET_ARCH=native` tunes the library for the building machine.
- `STRINGZILLA_CUDA_ARCHITECTURES` and `STRINGZILLA_ROCM_ARCHITECTURES` narrow the GPU code the features compile.

An engine fixes its capability once, when it is constructed, and every round dispatches on that.

## Types

The `sz` module exposes a handful of public value types used throughout the API.

- `Byteset` — a 256-bit set of bytes used by the byteset search and replace functions.
- `IndexSpan` — a `{ offset, length }` byte span, returned by uncased match iterators.
- `Hasher` — an incremental AES-based 64-bit hasher implementing `core::hash::Hasher`.
- `Sha256` — an incremental SHA-256 hasher with a one-shot convenience constructor.
- `BuildSzHasher` — a `std::hash::BuildHasher`, gated on feature `std`, for using `Hasher` with `HashMap`/`HashSet`.
- `Utf8View`, `Utf8Runes`, `Utf8SplitNewlines`, `Utf8SplitWhitespaces`, `Utf8Wordbreaks`, `Utf8Graphemes`, `Utf8Sentences`, `Utf8Linebreaks` — lazy UTF-8 views and iterators.
- `Utf8UncasedNeedle`, `Utf8UncasedMatches`, `Utf8NormalForm` — uncased search and Unicode normalization helpers.
- `SemVer`, `Status` — version and error types.
- `Capability`, `Capabilities` — one CPU or GPU capability, and a set of them printed as their names, with the producers that report a device's set by its ordinal.
- `Stream`, `Scope` — a stream of one device, which every engine, tape and allocator takes, and the span its engine verbs queue in before one join.
- `Sequence`, `UnifiedAllocator` — a batch copied into one tape, and the memory the host and a GPU both address.
- `ArgsortOptions` — knobs for sorting.

`Status` is the error of every fallible call: one variant per failure code of C's `sz_status_t`, plus `Unrecognized` for a code this crate's header does not list.
Success has no variant, being the `Ok` side of the `Result`, and `Status` implements `Display` and `core::error::Error`.
A buffer of the wrong length is reported as `Status::UnexpectedDimensions` rather than asserted.

`Byteset` is constructed from bytes and supports inversion, useful for "not from" semantics:

```rust
use stringzilla::sz::{self, Byteset};

let vowels = Byteset::from("aeiou");
let mut punct = Byteset::new();
punct.add(b',');
punct.add(b'.');
let everything = Byteset::new_ascii();         // all ASCII bytes set
let not_vowels = vowels.inverted();            // leaves `vowels` unchanged
let _ = (everything, not_vowels, punct);
```

`Byteset` API: `new()`, `new_ascii()`, `from_bytes(&[u8])` / `From<T: AsRef<[u8]>>`, `add(u8)`, `invert(&mut self)`, `inverted(&self) -> Byteset`.

`IndexSpan` API: `new(offset, length)`, `range() -> Range<usize>`, `extract<'a>(&self, &'a [u8]) -> Option<&'a [u8]>`, `end() -> usize`, plus public `offset` and `length` fields.
`end` saturates rather than overflowing, and `extract` returns `None` for a span that runs past the text.

```rust
use stringzilla::sz::IndexSpan;

let span = IndexSpan::new(6, 5);
assert_eq!(span.range(), 6..11);
assert_eq!(span.end(), 11);
assert_eq!(span.extract(b"Hello World"), Some(&b"World"[..]));
assert_eq!(span.extract(b"Hello"), None);
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
pub fn order<A: AsRef<[u8]>, B: AsRef<[u8]>>(a: A, b: B) -> core::cmp::Ordering; // SIMD sz_order_best
pub fn equal<A: AsRef<[u8]>, B: AsRef<[u8]>>(a: A, b: B) -> bool;               // SIMD sz_equal_best
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
The `StringZillableBinary` trait yields these iterators; the underlying `FindMatches`, `RFindMatches`, `FindSplits`, and `RFindSplits` structs can also be constructed directly from a `Matcher`.

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

For direct construction, a `Matcher<'a>` names what to look for, `Matcher::Substring(needle)` or `Matcher::Bytes(set)`, and the iterator type picks the direction: `FindMatches` and `FindSplits` search front to back, `RFindMatches` and `RFindSplits` back to front.
An inverted `Byteset` matches the bytes outside the set.
Policies are compile-time markers, opted into with builder methods: `.overlapping()` (default `NonOverlapping`, like `str::matches`) and `.skip_empty()` (default `KeepEmpty`, like `str::split`):

```rust
use stringzilla::sz::{Byteset, FindMatches, FindSplits, Matcher};

let non_overlapping: Vec<&[u8]> = FindMatches::new(b"aaaa", Matcher::Substring(b"aa")).collect();
assert_eq!(non_overlapping, vec![&b"aa"[..], &b"aa"[..]]);
let overlapping: Vec<&[u8]> = FindMatches::new(b"aaaa", Matcher::Substring(b"aa")).overlapping().collect();
assert_eq!(overlapping, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);

let separators = Matcher::Bytes(Byteset::from(b",;"));
let split: Vec<&[u8]> = FindSplits::new(b",a;;b,", separators).skip_empty().collect();
assert_eq!(split, vec![b"a", b"b"]);
```

An empty needle matches at every offset, `n + 1` times in an `n`-byte haystack, like `"abc".matches("")`.
An empty separator never splits, so the whole haystack comes back as one segment.

## Trimming and Translating

The crate provides byte-level buffer transforms, lookup-table translation, in-place replacement, and Unicode case folding / normalization.

### Lookup Table Translation

`lookup` maps every byte of a source through a 256-entry table into a target, while `lookup_inplace` rewrites a buffer in place.

```rust
pub fn lookup<T: AsMut<[u8]>, S: AsRef<[u8]>>(target: &mut T, source: &S, table: [u8; 256]) -> Result<(), Status>;
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
pub fn copy<T: AsMut<[u8]>, S: AsRef<[u8]>>(target: &mut T, source: &S) -> Result<(), Status>;
pub fn move_<T: AsMut<[u8]>, S: AsRef<[u8]>>(target: &mut T, source: &S) -> Result<(), Status>;
```

`copy`, `move_` and `lookup` return `Status::UnexpectedDimensions`, writing nothing, when the target is shorter than the source; `move_` tolerates overlapping regions.

### In Place Replacement

These rewrite a `Vec<u8>` in place, replacing either a literal needle or any byte from a `Byteset` with a replacement slice.

```rust
pub fn replace_all(buffer: &mut Vec<u8>, needle: &[u8], replacement: &[u8]) -> Result<usize, Status>;
pub fn replace_all_byteset(buffer: &mut Vec<u8>, byteset: Byteset, replacement: &[u8]) -> Result<usize, Status>;
```

Both replace all non-overlapping occurrences in place, returning the replacement count.
Equal-length replacements overwrite, shorter ones compact forward without allocating, and longer ones resize once and rewrite from the back:

```rust
use stringzilla::sz::{self, Byteset};

let mut buffer = b"a-b-c".to_vec();
let n = sz::replace_all(&mut buffer, b"-", b"__").unwrap();
assert_eq!(n, 2);
assert_eq!(buffer, b"a__b__c");

let mut spaced = b"a, b ,c".to_vec();
sz::replace_all_byteset(&mut spaced, Byteset::from(", "), b"").unwrap();
assert_eq!(spaced, b"abc");
```

### Case Folding and Normalization

`utf8_uncased_fold` case-folds into a target buffer, `utf8_norm` applies a Unicode normal form, and `utf8_find_denormalized` checks conformance without rewriting.

```rust
pub fn utf8_uncased_fold<S: AsRef<[u8]>, T: AsMut<[u8]>>(source: S, target: &mut T) -> Result<usize, Status>;
pub fn utf8_norm<S: AsRef<[u8]>, T: AsMut<[u8]>>(source: S, form: Utf8NormalForm, target: &mut T) -> Result<usize, Status>;
pub fn utf8_find_denormalized<T: AsRef<[u8]>>(source: T, form: Utf8NormalForm) -> Option<usize>;
```

`utf8_uncased_fold` applies Unicode case folding, for example `ß` → `ss`, returning the number of bytes written.
`utf8_norm` normalizes to one of the four `Utf8NormalForm` variants: `Nfd`, `Nfc`, `Nfkd`, and `Nfkc`.
`utf8_find_denormalized` is a fast check returning the byte offset of the first non-conforming byte, or `None` if already normalized:

```rust
use stringzilla::sz::{self, Utf8NormalForm};

let mut dest = [0u8; 33];
let len = sz::utf8_uncased_fold("HELLO WORLD", &mut dest).unwrap();
assert_eq!(&dest[..len], b"hello world");

// NFC check: a decomposed "café" (e + combining acute) violates NFC.
assert!(sz::utf8_find_denormalized("cafe\u{0301}", Utf8NormalForm::Nfc).is_some());
assert!(sz::utf8_find_denormalized("caf\u{00E9}", Utf8NormalForm::Nfc).is_none());

let mut out = vec![0u8; "cafe\u{0301}".len() * 18];
let n = sz::utf8_norm("cafe\u{0301}", Utf8NormalForm::Nfc, &mut out).unwrap();
assert_eq!(&out[..n], "caf\u{00E9}".as_bytes());
```

The C kernels write without a capacity, so target buffers must hold the worst-case expansion whatever the input: `source.len() * 3` for folding and `source.len() * 18` for normalization.
A shorter one is refused with `Status::UnexpectedDimensions` before anything is written.

### Uncased UTF-8 Search

`utf8_uncased_search` locates a needle in a haystack under Unicode case folding.

```rust
pub fn utf8_uncased_search<H: AsRef<[u8]>, N: Utf8UncasedNeedleArg>(haystack: H, needle: N) -> Option<(usize, usize)>;
```

Returns `Some((offset, match_length))`, where the matched length may differ from the needle length due to case folding — `ß` matching `SS`, for instance.
A `Utf8UncasedNeedle` is prepared once and only read by every search, so one needle serves many haystacks and threads, and `Utf8UncasedMatches` iterates all matches as `IndexSpan`s.
Matches start on codepoint boundaries: overlapping iteration and any zero-length match advance one codepoint, so an empty needle matches at every codepoint boundary, the end included, three times in `"aé"`:

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
pub fn hash_multiseed_into<T: AsRef<[u8]>>(text: T, seeds: &[u64], out: &mut [u64]) -> Result<(), Status>;
```

`bytesum` is an order-insensitive byte sum; `hash` / `hash_with_seed` are order-sensitive AES-based 64-bit hashes.
`hash_multiseed_into` hashes one input under many seeds in a single pass, handy for MinHash, Count-Min sketches, and Bloom/cuckoo filters; it returns `Status::UnexpectedDimensions` if `out.len() != seeds.len()`:

```rust
use stringzilla::sz;

assert_eq!(sz::bytesum("hi"), 209);
assert_ne!(sz::hash("Hello"), sz::hash("World"));
assert_eq!(sz::hash_with_seed("Hello", 42), sz::hash_with_seed("Hello", 42));

let seeds = [1u64, 2, 3, 4];
let mut out = [0u64; 4];
sz::hash_multiseed_into("token", &seeds, &mut out).unwrap();
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
`hmac_sha256_multistate` authenticates many messages under one key at once — a batch of tokens or webhook bodies sharing a secret — running both the message pass and HMAC's outer wrap through the lane-parallel kernels.
It allocates nothing, so it works under `no_std`: `states` is scratch of one state per message, reused for both passes, and only `tags` is written for keeps.

```rust
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> Sha256Digest;
pub fn hmac_sha256_multistate<Element: AsRef<[u8]>>(
    key: &[u8],
    messages: &[Element],
    states: &mut [Sha256],
    tags: &mut [Sha256Digest],
) -> Result<(), Status>;
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

`Sha256` API: `new()`, `update(&mut self, &[u8]) -> &mut Self`, `digest(&self) -> Sha256Digest`, `hash(&[u8]) -> Sha256Digest`, where `Sha256Digest` is `[u8; SHA256_DIGEST_LENGTH]`.

Digesting one message is a serial dependency chain, but independent messages compress in parallel lanes — sixteen at a time on AVX-512, eight on AVX2.
`sha256_multistate_update` advances one hasher per message, taking anything that dereferences to bytes, so a `Vec<String>` needs no intermediate slice of slices.
Use `sha256_multistate_update_by(&mut states, &records, |record| record.payload.as_bytes())` when each message is a field of a larger element.

Lanes advance in lockstep within a group, so a group costs as much as its longest message.
Every length is handled correctly, but throughput is best when messages of similar length share a group, which `argsort` arranges.

```rust
use stringzilla::sz;

let texts: Vec<String> = vec!["alpha".into(), "beta".into()];
let mut states = vec![sz::Sha256::new(); texts.len()];
let mut digests = vec![[0u8; 32]; texts.len()];

sz::sha256_multistate_update(&mut states, &texts).unwrap();
sz::sha256_multistate_digest(&states, &mut digests).unwrap();
assert_eq!(digests[0], sz::Sha256::hash(b"alpha"));
```

## Sorting, Sampling, and Shuffling

### Argsort

`argsort` writes the sorted permutation of a slice into a caller-provided `order` slice, while `argsort_by` sorts by a byte-slice key extracted from each element.

```rust
pub fn argsort<T: AsRef<[u8]>>(data: &[T], order: &mut [SortedIdx], options: ArgsortOptions) -> Result<(), Status>;
pub fn argsort_by<T, K>(data: &[T], key: K, order: &mut [SortedIdx], options: ArgsortOptions) -> Result<(), Status>
where K: Fn(&T) -> &[u8];
```

Both write the sorting permutation of `data` into a caller-supplied `order` buffer of length at least `data.len()`, or return `Status::UnexpectedDimensions`.
`argsort_by` sorts by a byte-slice key of each element, ideal for sorting structs by a field.
The binding indexes `data` itself, so the key only projects an element onto its bytes; it runs inside the C call, where a panic cannot unwind and aborts the process.
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
sz::argsort_by(&people, |person| person.name.as_bytes(), &mut order, Default::default()).unwrap();
assert_eq!(&order, &[1, 2, 0]); // Alice, Bob, Charlie
```

### Intersection, an Inner Join

`intersection` matches two collections directly, while `intersection_by` matches by a byte-slice key extracted from each element, both writing the matched positions of each side.

```rust
pub fn intersection<T: AsRef<[u8]>>(data1: &[T], data2: &[T], seed: u64,
    positions1: &mut [SortedIdx], positions2: &mut [SortedIdx]) -> Result<usize, Status>;
pub fn intersection_by<T, U, K, L>(data1: &[T], key1: K, data2: &[U], key2: L, seed: u64,
    positions1: &mut [SortedIdx], positions2: &mut [SortedIdx]) -> Result<usize, Status>
where K: Fn(&T) -> &[u8], L: Fn(&U) -> &[u8];
```

Both compute the intersection of two collections, which may differ in length and, for `intersection_by`, in element type.
They write the matching positions into output buffers each sized at least `min(data1.len(), data2.len())`, or return `Status::UnexpectedDimensions`, and return the intersection size:

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

## Edit Distances

`LevenshteinEngine` prepares a batch of queries once — Myers' bit-parallel masks, the capability, and on a device the launch geometry — then scores as many batches of candidates against it as a caller has.
Preparation is the expensive half, so a long-lived engine amortizes it across every later round, and the round's scratch grows to fit the widest batch it has seen and is never shrunk.

```rust
fn new<Q: AsRef<[u8]>>(queries: &[Q], symbol: LevenshteinSymbol,
    stream: &Stream) -> Result<LevenshteinEngine, Status>;

fn distances<'scope, C: Strings + ?Sized>(&'scope mut self, scope: &'scope Scope<'scope, '_>,
    candidates: &'scope C, distances: &'scope mut [usize], distances_stride: usize) -> Result<(), Status>;
```

`LevenshteinSymbol::Bytes` counts bytes and `LevenshteinSymbol::Runes` counts UTF-8 runes, an ill-formed byte decoding to `U+FFFD`; the two alphabets are one engine and one verb rather than two spellings.
Costs are unit — one per substitution, insertion and deletion — and there are no gap costs and no substitution matrix.

The output is a `[queries, candidates]` block the caller owns: query `q` against candidate `c` lands at `distances[q * distances_stride + c]`, the stride counting entries rather than bytes and being at least the candidate count.
A stride wider than the candidate count is what lets one round fill a sub-block of a larger matrix.

```rust
use stringzilla::sz::{Capabilities, LevenshteinEngine, LevenshteinSymbol, Stream};

let cpu = Stream::default(Capabilities::cpu_enabled());
let mut engine = LevenshteinEngine::new(&["kitten", "saturday"], LevenshteinSymbol::Bytes, &cpu).unwrap();

let mut distances = [0usize; 4];
cpu.scope(|scope| engine.distances(scope, &["sitting", "sunday"], &mut distances, 2)).unwrap();
assert_eq!(distances[0], 3); // kitten vs sitting
assert_eq!(distances[3], 3); // saturday vs sunday

// The same engine, a second round, no preparation repeated.
let mut again = [0usize; 2];
cpu.scope(|scope| engine.distances(scope, &["mitten"], &mut again, 1)).unwrap();

// Runes rather than bytes, for text where a codepoint is the unit.
let mut unicode = LevenshteinEngine::new(&["café"], LevenshteinSymbol::Runes, &cpu).unwrap();
let mut one = [0usize; 1];
cpu.scope(|scope| unicode.distances(scope, &["cafe"], &mut one, 1)).unwrap();
assert_eq!(one[0], 1);
```

## Window Overlap

`OverlapEngine` hashes and sorts a batch of queries into one B-tree forest over every window width, then streams candidates through it.
A window is a fixed-width byte n-gram, and the overlap of two texts at that width is the count of one's windows that occur in the other, over whichever of the two holds more.
Nothing is stored per candidate, so the candidates may change round to round while the queries and the widths stay.

```rust
fn new<Q: AsRef<[u8]>>(queries: &[Q], window_widths: &[usize], candidates_budget: usize,
    stream: &Stream) -> Result<OverlapEngine, Status>;

fn scores<'scope, C: Strings + ?Sized>(&'scope mut self, scope: &'scope Scope<'scope, '_>,
    candidates: &'scope C, scores: &'scope mut [f32],
    scores_query_stride: usize, scores_candidate_stride: usize) -> Result<(), Status>;
```

Widths are in bytes and need not form a doubling chain, since a window hash comes from a prefix difference that costs one modular multiply-add at any width; a width past a text scores zero for every pair it spans.
The output is a `[queries, candidates, widths]` block with the width axis unit-strided: share `[q, c, w]` lands at `scores[q * query_stride + c * candidate_stride + w]`, each in `[0, 1]`.
The score is asymmetric — a candidate's window occurrences count against a query's distinct windows — so swapping the two sides changes the answer whenever either repeats a window.

```rust
use stringzilla::sz::{Capabilities, OverlapEngine, Stream};

let cpu = Stream::default(Capabilities::cpu_enabled());
let mut engine = OverlapEngine::new(&["the quick brown fox"], &[4, 8], 0, &cpu).unwrap();

let mut scores = [0.0f32; 2];
cpu.scope(|scope| engine.scores(scope, &["the quick brown cat"], &mut scores, 2, 2)).unwrap();
assert!(scores.iter().all(|share| (0.0..=1.0).contains(share)));
```

## Multi-Pattern Search

`SubstringsEngine` compiles a whole needle set into one Aho-Corasick automaton and walks every haystack against all of them at once, so a dictionary of thousands of terms costs one pass rather than thousands.
Building it is the expensive half and the engine is reusable, so a long-lived one amortizes that across every later batch.

```rust
fn new<N: AsRef<[u8]>>(needles: &[N], case_sensitivity: CaseSensitivity,
    overlap_policy: SubstringsOverlapPolicy, hot_states: usize, matches_budget: usize,
    haystacks_budget: usize, stream: &Stream) -> Result<SubstringsEngine, Status>;

fn counts<'scope, H: Strings + ?Sized>(&'scope mut self, scope: &'scope Scope<'scope, '_>,
    haystacks: &'scope H, counts: &'scope mut [usize], counts_stride: usize) -> Result<(), Status>;
fn find<'scope, H: Strings + ?Sized>(&'scope mut self, scope: &'scope Scope<'scope, '_>,
    haystacks: &'scope H, matches: &'scope mut [SubstringsMatch],
    matches_offsets: &'scope mut [usize]) -> Result<(), Status>;
fn replace<'scope, H: Strings + ?Sized, R: Strings + ?Sized>(&'scope mut self,
    scope: &'scope Scope<'scope, '_>, haystacks: &'scope H, replacements: &'scope R,
    target: &'scope mut [u8], offsets: &'scope mut [usize]) -> Result<(), Status>;
fn bm25_scores<'scope, H: Strings + ?Sized>(&'scope mut self, scope: &'scope Scope<'scope, '_>,
    haystacks: &'scope H, document_lengths: Option<&'scope [f32]>, parameters: &'scope Bm25Params,
    needle_weights: &'scope [f32], scores: &'scope mut [f32], scores_stride: usize) -> Result<(), Status>;

fn report(&self) -> SubstringsReport;
```

`hot_states` sizes the automaton's dense tier, or takes `SUBSTRINGS_HOT_STATES_AUTO` to fill a fixed byte budget instead, which holds more states the fewer byte classes the vocabulary spells.
`matches_budget` bounds what one round may emit and `haystacks_budget` how many haystacks it may carry, and both are read by a device tier alone, so a host engine takes `SUBSTRINGS_MATCHES_BUDGET_AUTO` and zero and walks straight into the caller's output.

`CaseSensitivity::Cased` matches bytes exactly and accepts arbitrary needles, while `CaseSensitivity::Uncased` folds both sides under full Unicode case folding and requires valid UTF-8.
Folding is not a byte-length-preserving operation, so a 1-byte needle can match a 3-byte span — the Kelvin sign `U+212A` folds to `k` — which is why every `SubstringsMatch` carries its own `byte_length` rather than borrowing the needle's.

`SubstringsOverlapPolicy` decides what a walk reports, and it sizes the engine's arena, so it is fixed at construction and one engine runs exactly one of the three:

- `Overlapping` — every match of every needle, nested and overlapping ones included.
- `LeftmostLongest` — a cover taking the widest match at the earliest start.
- `LeftmostFirst` — a cover taking the lowest needle index at the earliest start.

A capacity shortfall is not an error.
The sizing walk always runs, so `report()` names `matches_emitted`, the true total; `matches_stored`, what was written; `target_length`, what a rewrite needs; and `shortfall`, what did not fit.
That is what lets one call with an empty output size the next one, with no walk in between.

```rust
use stringzilla::sz::{
    Capabilities, CaseSensitivity, Stream, SubstringsEngine, SubstringsMatch, SubstringsOverlapPolicy,
    SUBSTRINGS_HOT_STATES_AUTO, SUBSTRINGS_MATCHES_BUDGET_AUTO,
};

let cpu = Stream::default(Capabilities::cpu_enabled());
let mut engine = SubstringsEngine::new(
    &["cat", "catalog"],
    CaseSensitivity::Cased,
    SubstringsOverlapPolicy::Overlapping,
    SUBSTRINGS_HOT_STATES_AUTO,
    SUBSTRINGS_MATCHES_BUDGET_AUTO,
    0,
    &cpu,
)
.unwrap();

let documents = ["a catalog of cats", "nothing here"];
let mut counts = [0usize; 2];
cpu.scope(|scope| engine.counts(scope, &documents, &mut counts, 1)).unwrap();
assert_eq!(counts, [3, 0]); // "catalog", the "cat" inside it, and the "cat" of "cats"

// `find` fills one boundary per haystack plus a final total, whether or not the matches fit.
let mut matches = [SubstringsMatch::default(); 3];
let mut offsets = [0usize; 3];
cpu.scope(|scope| engine.find(scope, &documents, &mut matches, &mut offsets)).unwrap();
assert_eq!(offsets, [0, 3, 3]);
assert_eq!(engine.report().matches_emitted, 3);
assert_eq!(engine.report().shortfall, 0);
```

### Scoring and Rewriting

The same automaton scores documents with BM25 and rewrites them, each in a single walk.
`bm25_scores` treats the dictionary itself as the query — `needle_weights[i]` is needle `i`'s IDF or boost — and writes one score per haystack, so a many-term query over a large corpus never materializes per-term frequency rows.
Term frequencies are raw overlapping counts, which is classic BM25, so the engine's own policy does not apply here, and `document_lengths` falls back to each haystack's byte length when `None`.

`Bm25Params` has no `Default`, because a corpus mean has no correct default value.
Its two constructors name the two configurations that exist: `Bm25Params::normalized(mean)` is the literature's `k1 = 1.2` and `b = 0.75` against a corpus whose mean document length you know, and `Bm25Params::unnormalized()` switches length normalization off and leaves `document_lengths` unread.

`replace` takes one replacement per needle, inserted verbatim, an empty one deleting its match, and writes every rewritten haystack into one `target` buffer beside their boundaries.
Rewriting is defined only under a cover, so an engine built with `SubstringsOverlapPolicy::Overlapping` is refused there — an overlapping rewrite is not a function.
A `target` too small is not an error either: `report().target_length` names the bytes the rewrite needed and the buffer's contents are then unspecified, so an empty `target` is how the next call is sized.

```rust
use stringzilla::sz::{Bm25Params, Capabilities, CaseSensitivity, Stream, SubstringsEngine, SubstringsOverlapPolicy,
                      SUBSTRINGS_HOT_STATES_AUTO, SUBSTRINGS_MATCHES_BUDGET_AUTO};

let cpu = Stream::default(Capabilities::cpu_enabled());
let mut engine = SubstringsEngine::new(
    &["cat", "dog"],
    CaseSensitivity::Cased,
    SubstringsOverlapPolicy::LeftmostLongest,
    SUBSTRINGS_HOT_STATES_AUTO,
    SUBSTRINGS_MATCHES_BUDGET_AUTO,
    0,
    &cpu,
)
.unwrap();

let documents = ["cat and dog", "nothing here"];
let mut scores = [0.0f32; 2];
let parameters = Bm25Params::normalized(10.0);
cpu.scope(|scope| engine.bm25_scores(scope, &documents, None, &parameters, &[1.0, 1.0], &mut scores, 1))
    .unwrap();
assert_eq!(scores[1], 0.0);

// One replacement per needle: an empty target sizes the rewrite, a second scope performs it.
let replacements = ["feline", "canine"];
let mut offsets = [0usize; 3];
cpu.scope(|scope| engine.replace(scope, &documents, &replacements, &mut [], &mut offsets)).unwrap();
let mut target = vec![0u8; engine.report().target_length];
cpu.scope(|scope| engine.replace(scope, &documents, &replacements, &mut target, &mut offsets)).unwrap();
assert_eq!(&target[offsets[0]..offsets[1]], b"feline and canine");
```

## UTF-8 Segmentation

A single emoji such as the flag `🇺🇸` is 8 bytes and 2 codepoints, yet a reader sees one character — one grapheme cluster.
StringZilla draws every one of these boundaries: `sz_utf8_runes` (`Utf8View`) walks codepoints, while `sz_utf8_graphemes` (`Utf8Graphemes`) walks user-perceived clusters.
The `StringZillableUnary` trait turns any `AsRef<[u8]>` into a broad family of lazy UTF-8 iterators and views, each yielding borrowed byte sub-slices or `char`s on demand:

```rust
fn sz_utf8_runes(&self) -> Utf8View<'_>;                         // codepoints / random codepoint access
fn sz_utf8_split_newlines(&self) -> Utf8SplitNewlines<'_>;       // content BETWEEN hard newlines (CRLF = one)
fn sz_utf8_newlines(&self) -> Utf8Newlines<'_>;                  // the newline runs themselves
fn sz_utf8_split_whitespaces(&self) -> Utf8SplitWhitespaces<'_>; // content BETWEEN Unicode whitespace
fn sz_utf8_whitespaces(&self) -> Utf8Whitespaces<'_>;            // the whitespace runs themselves
fn sz_utf8_split_delimiters(&self) -> Utf8SplitDelimiters<'_>;   // content BETWEEN whitespace/punctuation
fn sz_utf8_delimiters(&self) -> Utf8Delimiters<'_>;              // the delimiter runs themselves
fn sz_utf8_wordbreaks(&self) -> Utf8Wordbreaks<'_>;              // all UAX-29 word segments (tiling)
fn sz_utf8_graphemes(&self) -> Utf8Graphemes<'_>;                // UAX-29 grapheme clusters
fn sz_utf8_sentences(&self) -> Utf8Sentences<'_>;                // UAX-29 sentences
fn sz_utf8_linebreaks(&self) -> Utf8Linebreaks<'_>;              // UAX-14 line-break opportunities
```

The naming follows one rule: the bare name (`newlines`/`whitespaces`/`delimiters`) yields the __separators__ the kernel finds, while `split_*` yields the content __between__ them.
Chain `.with_separators()` on a `split_*` iterator to interleave both losslessly.

Every member of this family is lazy and zero-copy.
`Utf8View`, `Utf8Runes`, `Utf8SplitNewlines`, `Utf8SplitWhitespaces`, `Utf8Wordbreaks`, `Utf8Graphemes`, `Utf8Sentences`, and `Utf8Linebreaks`, together with the `sz_splits` / `sz_rsplits` iterators, all borrow from the source string and yield `&[u8]` or `&str` slices on demand without allocating.
There is no backing vector and no per-element heap buffer.
Contrast this with the standard library, where collecting into a `Vec<String>` allocates the vector and a fresh heap buffer for every element, and even a `Vec<&str>` allocates the backing vector up front.
Because these iterators borrow from the input, you can stream over millions of words or grapheme clusters of a large document with effectively zero per-element allocation, and the borrow lifetimes keep the yielded slices zero-copy.

`Utf8View` offers O(1) construction, `len()` counting the codepoints on demand, `offset_of(n)` for the byte offset of the Nth codepoint, and `iter()` for batched `char` iteration:

```rust
use stringzilla::sz::StringZillableUnary;

let view = "Hello🌍".sz_utf8_runes();
assert_eq!(view.len(), 6);
assert_eq!(view.offset_of(5), Some(5));
let chars: Vec<char> = view.iter().collect();
assert_eq!(chars, vec!['H', 'e', 'l', 'l', 'o', '🌍']);
```

The boundary iterators `wordbreaks`, `graphemes`, `sentences`, and `linebreaks` __tile__ the input — every byte belongs to exactly one segment, so consecutive segments are contiguous and no empty slices appear:

```rust
use stringzilla::sz::StringZillableUnary;

let words: Vec<&[u8]> = b"Hi, world".sz_utf8_wordbreaks().collect();
assert_eq!(words, vec![&b"Hi"[..], &b","[..], &b" "[..], &b"world"[..]]);

let sentences: Vec<&[u8]> = b"Hi. Bye.".sz_utf8_sentences().collect();
assert_eq!(sentences, vec![&b"Hi. "[..], &b"Bye."[..]]);
```

`split_newlines` splits on the 8 Unicode hard newlines, treating CRLF as a single delimiter and excluding the newline from each segment, while `split_whitespaces` splits on the 25 Unicode whitespace characters and __keeps__ empty segments by default; chain `.skip_empty()` for `str::split_whitespace`-style behavior:

```rust
use stringzilla::sz::StringZillableUnary;

let lines: Vec<&str> = "Hello\nWorld\r\nRust"
    .sz_utf8_split_newlines()
    .map(|l| std::str::from_utf8(l).unwrap())
    .collect();
assert_eq!(lines, vec!["Hello", "World", "Rust"]);

let tokens: Vec<&[u8]> = b"  hi  ".sz_utf8_split_whitespaces().skip_empty().collect();
assert_eq!(tokens, vec![&b"hi"[..]]);
```

Every iterator is also constructible directly, for example `Utf8Wordbreaks::new(text)`.
Each exposes a `STEPS` const-generic batch-size knob via `with_steps`, written with a turbofish such as `Utf8Runes::<256>::with_steps(bytes)`, that trades buffer size for FFI-call amortization without changing the yielded results.
The default batch size is the public constant `ITERATORS_DEFAULT_STEPS = 64`.

### Low Level Decoding

`utf8_decode` unpacks UTF-8 bytes into a UTF-32 `runes` buffer, returning how many bytes were consumed and codepoints written.

```rust
pub fn utf8_decode(text: &[u8], runes: &mut [u32]) -> (usize, usize); // (bytes_consumed, runes_count)
```

`utf8_decode` decodes UTF-8 into UTF-32 codepoints, filling the output buffer or draining the input per call.
For inputs larger than the buffer, loop and resume at `bytes_consumed`.
It is total: ill-formed bytes decode to U+FFFD, so every value written is a valid Unicode scalar.
The end of `text` is the end of the input, so a truncated final sequence also decodes to one U+FFFD and is consumed; cut a streamed input into chunks at rune boundaries:

```rust
use stringzilla::sz;

let mut runes = [0u32; 16];
assert_eq!(sz::utf8_decode(b"a\xE2\x82", &mut runes), (3, 2));
assert_eq!(&runes[..2], &['a' as u32, 0xFFFD]);
let (bytes, count) = sz::utf8_decode("Hello".as_bytes(), &mut runes);
assert_eq!((bytes, count), (5, 5));
assert_eq!(runes[0], 'H' as u32);
```

## Runtime Dispatch and Capabilities

The producers report `Capabilities`, a set of `Capability` bits, for the CPU or for one GPU by its runtime's own ordinal.

```rust
use stringzilla::sz::{self, Capabilities, Capability};

let v = sz::version();
println!("StringZilla {}.{}.{}", v.major, v.minor, v.patch);

let enabled = Capabilities::cpu_enabled();
enabled.configure_thread()?;
println!("dispatching to {enabled}"); // like "serial,neon,neonaes,neonsha"

for ordinal in 0..Capabilities::cuda_count_devices().unwrap_or(0) {
    println!("CUDA device {ordinal} runs {}", Capabilities::cuda_enabled(ordinal)?); // like "cuda"
}
```

Each group, `cpu`, `cuda`, `rocm` and `metal`, reports along two independent axes, plus the set dispatch uses:

- `cpu_detected()`, `cuda_detected(ordinal)`: what the device can execute, from CPUID or HWCAP on the CPU and from the runtime on a GPU
- `cpu_compiled()`, `cuda_compiled()`: what this binary contains, from the probes at build time
- `cpu_enabled()`, `cuda_enabled(ordinal)`: what dispatch uses, i.e. both axes at once

Reach for the enabled set unless you specifically mean one of the raw axes.
The detected set describes the machine and says nothing about whether a kernel was compiled in, so a build whose ISA probes failed still reports your CPU's full feature set while containing no SIMD kernels at all.
The crate keeps no process state: every call passes `Capabilities::CPUS`, which the library clamps to the enabled set, while an engine keeps the capability it was built with.
`cuda_count_devices()` fails with `Status::MissingGpu` for a vendor this build or machine has no device of, and `cuda_enabled(ordinal)` fails the same way past the last device.

Call `configure_thread` at the start of every thread that runs kernels, to prepare it for the capabilities it passes.
In a thread-pool setting, each worker thread needs its own call.
The function is idempotent and cheap to call more than once on the same thread.

### Engines on a GPU

Every engine constructor takes a `Stream`, whose capabilities it is built with, and a GPU stream builds it on that GPU, fixing the launch geometry once and sizing a round's device memory from `candidates_budget` for `OverlapEngine` and `haystacks_budget` for `SubstringsEngine`.
The capabilities name the vendor and the stream names the device: `Stream::default(capabilities)` is the default stream of the default device, while `Stream::new(capabilities, ordinal)` makes one on any device by its ordinal and frees it on drop.
`Stream::from_raw` borrows a `cudaStream_t`, a `hipStream_t` or an `id<MTLCommandQueue>` the caller made, and is `unsafe` because no type system checks that the handle is live and belongs to the vendor.

Every verb only queues its round on the stream of the `Scope` it takes, which `Stream::scope` hands its body and joins once the body returns, even when it fails.
The verb borrows the engine, its inputs and its outputs for the whole scope, so none of them can be read or dropped before the device is done with them, and an engine runs one verb per scope.
A stream of another group than the engine's is refused with `Status::DeviceMemoryMismatch`.

A device engine reads its rounds where they lie, so its candidates and its outputs have to sit in memory the device reaches.
`UnifiedAllocator::new(&stream)` is that memory on the stream's device, behind the standard `Allocator` trait, so `Vec::new_in` and any other allocator-aware collection lays an output out there.
`Sequence::copy` copies a batch on the host into one tape of offsets and bytes in it, queuing only its migration to the device, and every engine reads that tape in place, while a batch of host strings is refused with `Status::DeviceMemoryMismatch`.

```rust
use stringzilla::sz::{Capabilities, LevenshteinEngine, LevenshteinSymbol, Sequence, Stream, UnifiedAllocator};

let stream = Stream::new(Capabilities::cuda_enabled(0)?, 0)?;
let unified = UnifiedAllocator::new(&stream);
let candidates = Sequence::copy(&["sitting", "kitten"], &unified, &stream)?;
let mut distances = Vec::new_in(unified);
distances.resize(2, 0usize);

let mut engine = LevenshteinEngine::new(&["kitten"], LevenshteinSymbol::Bytes, &stream)?;
stream.scope(|scope| engine.distances(scope, &candidates, &mut distances, 2))?;
assert_eq!(distances[..], [3, 0]);
```

`stream.synchronize()` joins a stream outside any scope.
A host engine imposes no such requirement: plain `Vec` and stack buffers are exactly what its verbs expect, and a `Sequence` copied for any capabilities reads there too.
