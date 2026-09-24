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

The crate ships the C/C++ sources and compiles them through a `build.rs` via `cc`, so no system StringZilla install is required.

### Feature Flags

| Feature            | Default | Effect                                             |
| :----------------- | :-----: | :------------------------------------------------- |
| `std`              |   yes   | `std` support, else `no_std`                       |
| `dynamic-dispatch` |   yes   | Runtime SIMD dispatch; disable to bake in one tier |
| `cuda`             |   no    | CUDA GPU backend, behind every `new_on_gpu`        |

Without `std` the crate is `no_std`; `std` is also required for the `BuildSzHasher` integration with `HashMap`/`HashSet`.
The `cuda` feature compiles the CUDA backend and unlocks each engine's `new_on_gpu` constructor; without it the engines are host-only and every other verb is unchanged:

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

### Dynamic vs Compile-Time Dispatch

The `dynamic-dispatch` feature (on by default) controls how the C kernels select a SIMD backend, mirroring the C library's `SZ_DYNAMIC_DISPATCH` macro.
The build script decides which ISA tiers to enable from two independently probed facts, using the same checked-in `probes/` sources as the CMake build:

- the __compile set__ — tiers this toolchain can emit, learned by try-compiling `probes/<arch>_<tier>.c` (tiny programs reusing the real kernels' `target` pragmas and intrinsics, so broken or old toolchains are caught up front);
- the __run set__ — tiers this machine can execute, learned by compiling and _running_ `probes/run_capabilities.c` on native builds.
  When cross-compiling, the target description (`-C target-cpu=…` / `-C target-feature=+…`, surfaced as `CARGO_CFG_TARGET_FEATURE`) stands in for the machine.

The two sets are independent — an old compiler on a new CPU misses tiers the machine could run, a new compiler on an old CPU can emit tiers the machine would trap on — and the dispatch mode picks the gate:

- __On (default):__ every tier in the _compile set_ is built, and the best one is chosen _at load_ through a dispatch table — the same model as the precompiled `stringzilla_shared` C library.
  One binary runs optimally on any CPU of the target architecture, at the cost of one indirect call per operation.
  Two constraints bound the optimism, both expressed in the probes rather than in build-system code: the SVE probes refuse Apple targets outright (no Apple CPU implements SVE, so the kernels could compile but never dispatch), and targets whose built library performs no runtime CPU detection — WebAssembly and OS-less exotica, inferred by compile-probing the header's own `SZ_CAPABILITIES_RUNTIME_DETECTABLE_` — stay within the target description, since no load-time masking exists there.
- __Off:__ each function is resolved _at compile time_ to the newest tier in the _intersection_ of the compile and run sets — the analog of including the header-only `stringzilla_header` in your own translation unit with `-march` describing the deployment CPU.
  There is no table and no constructor, the unused tiers are dead-code-stripped, and the call goes straight to the kernel.

Removing the indirection trades flexibility for speed.
A compile-time binary built natively is __tuned to the build machine__ ("build where you run"): the run probe may enable tiers beyond the declared target features, and the build prints a warning when it does, because the result is not portable to older CPUs.
In exchange, call-bound operations get faster: a short-input `sz_find` microbenchmark on this machine ran ~15% more calls per second without the table (≈205 → ≈240 Mcalls/s for 8–64 B inputs).
The win shrinks as inputs grow and the SIMD kernel, rather than the call overhead, dominates.

Disable it by opting out of default features (re-adding the ones you still want):

```toml
[dependencies]
# Compile-time dispatch: smaller, faster, but pinned to the build machine's best ISA.
stringzilla = { version = "5", default-features = false, features = ["std"] }
```

Every tier can be forced on or off with its `SZ_USE_*` environment variable (`SZ_USE_SVE2=0 cargo build`), overriding the run gate but never the compile gate; the CMake build honors the same names as cache options (`-D SZ_USE_SVE2=0`).
For a portable compile-time build, cross-describe the floor instead of probing the machine: pin `-C target-feature=…` (or `-C target-cpu=…`) to the oldest deployment CPU.
`sz::dynamic_dispatch()` reports which mode the crate was built with.
An engine resolves its ISA tier once, when it is constructed, under either mode — so the table costs it one branch per round rather than one per call.

## Types

The `sz` module exposes a handful of public value types used throughout the API.

- `Byteset` — a 256-bit set of bytes used by the byteset search and replace functions.
- `IndexSpan` — a `{ offset, length }` byte span, returned by uncased match iterators.
- `Hasher` — an incremental AES-based 64-bit hasher implementing `core::hash::Hasher`.
- `Sha256` — an incremental SHA-256 hasher with a one-shot convenience constructor.
- `BuildSzHasher` — a `std::hash::BuildHasher`, gated on feature `std`, for using `Hasher` with `HashMap`/`HashSet`.
- `Utf8View`, `Utf8Runes`, `Utf8SplitNewlines`, `Utf8SplitWhitespaces`, `Utf8Wordbreaks`, `Utf8Graphemes`, `Utf8Sentences`, `Utf8Linebreaks` — lazy UTF-8 views and iterators.
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

For direct construction, `MatcherType<'a>` selects the search mode and is paired with `FindMatches::new(haystack, matcher)`, `RFindMatches::new(...)`, `FindSplits::new(haystack, matcher)`, or `RFindSplits::new(...)`.
Policies are compile-time markers, opted into with builder methods: `.overlapping()` (default `NonOverlapping`, like `str::matches`) and `.skip_empty()` (default `KeepEmpty`, like `str::split`):

```rust
use stringzilla::sz::{MatcherType, FindMatches, FindSplits};

let non_overlapping: Vec<&[u8]> = FindMatches::new(b"aaaa", MatcherType::Find(b"aa")).collect();
assert_eq!(non_overlapping, vec![&b"aa"[..], &b"aa"[..]]);
let overlapping: Vec<&[u8]> = FindMatches::new(b"aaaa", MatcherType::Find(b"aa")).overlapping().collect();
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
Use `sha256_multistate_update_by` when the messages are not in one contiguous slice.

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

## Edit Distances

`LevenshteinEngine` prepares a batch of queries once — Myers' bit-parallel masks, the ISA tier, and on a device the launch geometry — then scores as many batches of candidates against it as a caller has.
Preparation is the expensive half, so a long-lived engine amortizes it across every later round, and the round's scratch grows to fit the widest batch it has seen and is never shrunk.

```rust
fn new<Q: AsRef<[u8]>>(queries: &[Q], symbol: LevenshteinSymbol) -> Result<LevenshteinEngine, Status>;
unsafe fn new_on_gpu<Q: AsRef<[u8]>>(queries: &[Q], symbol: LevenshteinSymbol, stream: *mut c_void)
    -> Result<LevenshteinEngine, Status>;                                      // needs `cuda`

fn distances<C: AsRef<[u8]>>(&mut self, candidates: &[C], distances: &mut [usize],
    distances_stride: usize) -> Result<(), Status>;
```

`LevenshteinSymbol::Bytes` counts bytes and `LevenshteinSymbol::Runes` counts UTF-8 runes, an ill-formed byte decoding to `U+FFFD`; the two alphabets are one engine and one verb rather than two spellings.
Costs are unit — one per substitution, insertion and deletion — and there are no gap costs and no substitution matrix.

The output is a `[queries, candidates]` block the caller owns: query `q` against candidate `c` lands at `distances[q * distances_stride + c]`, the stride counting entries rather than bytes and being at least the candidate count.
A stride wider than the candidate count is what lets one round fill a sub-block of a larger matrix.

```rust
use stringzilla::sz::{LevenshteinEngine, LevenshteinSymbol};

let mut engine = LevenshteinEngine::new(&["kitten", "saturday"], LevenshteinSymbol::Bytes).unwrap();

let mut distances = [0usize; 4];
engine.distances(&["sitting", "sunday"], &mut distances, 2).unwrap();
assert_eq!(distances[0], 3); // kitten vs sitting
assert_eq!(distances[3], 3); // saturday vs sunday

// The same engine, a second round, no preparation repeated.
let mut again = [0usize; 2];
engine.distances(&["mitten"], &mut again, 1).unwrap();

// Runes rather than bytes, for text where a codepoint is the unit.
let mut unicode = LevenshteinEngine::new(&["café"], LevenshteinSymbol::Runes).unwrap();
let mut one = [0usize; 1];
unicode.distances(&["cafe"], &mut one, 1).unwrap();
assert_eq!(one[0], 1);
```

## Window Overlap

`OverlapEngine` hashes and sorts a batch of queries into one B-tree forest over every window width, then streams candidates through it.
A window is a fixed-width byte n-gram, and the overlap of two texts at that width is the count of one's windows that occur in the other, over whichever of the two holds more.
Nothing is stored per candidate, so the candidates may change round to round while the queries and the widths stay.

```rust
fn new<Q: AsRef<[u8]>>(queries: &[Q], window_widths: &[usize]) -> Result<OverlapEngine, Status>;
unsafe fn new_on_gpu<Q: AsRef<[u8]>>(queries: &[Q], window_widths: &[usize], stream: *mut c_void)
    -> Result<OverlapEngine, Status>;                                          // needs `cuda`

fn scores<C: AsRef<[u8]>>(&mut self, candidates: &[C], scores: &mut [f32],
    scores_query_stride: usize, scores_candidate_stride: usize) -> Result<(), Status>;
```

Widths are in bytes and need not form a doubling chain, since a window hash comes from a prefix difference that costs one modular multiply-add at any width; a width past a text scores zero for every pair it spans.
The output is a `[queries, candidates, widths]` block with the width axis unit-strided: share `[q, c, w]` lands at `scores[q * query_stride + c * candidate_stride + w]`, each in `[0, 1]`.
The score is asymmetric — a candidate's window occurrences count against a query's distinct windows — so swapping the two sides changes the answer whenever either repeats a window.

```rust
use stringzilla::sz::OverlapEngine;

let mut engine = OverlapEngine::new(&["the quick brown fox"], &[4, 8]).unwrap();

let mut scores = [0.0f32; 2];
engine.scores(&["the quick brown cat"], &mut scores, 2, 2).unwrap();
assert!(scores.iter().all(|share| (0.0..=1.0).contains(share)));
```

## Multi-Pattern Search

`SubstringsEngine` compiles a whole needle set into one Aho-Corasick automaton and walks every haystack against all of them at once, so a dictionary of thousands of terms costs one pass rather than thousands.
Building it is the expensive half and the engine is reusable, so a long-lived one amortizes that across every later batch.

```rust
fn new<N: AsRef<[u8]>>(needles: &[N], case_sensitivity: CaseSensitivity,
    overlap_policy: SubstringsOverlapPolicy, hot_states: usize, matches_budget: usize)
    -> Result<SubstringsEngine, Status>;
unsafe fn new_on_gpu<N: AsRef<[u8]>>(/* the same, plus */ stream: *mut c_void)
    -> Result<SubstringsEngine, Status>;                                       // needs `cuda`

fn counts<H: AsRef<[u8]>>(&mut self, haystacks: &[H], counts: &mut [usize],
    counts_stride: usize) -> Result<(), Status>;
fn find<H: AsRef<[u8]>>(&mut self, haystacks: &[H], matches: &mut [SubstringsMatch],
    matches_offsets: &mut [usize]) -> Result<(), Status>;
fn replace<H: AsRef<[u8]>, R: AsRef<[u8]>>(&mut self, haystacks: &[H], replacements: &[R],
    tape: &mut [u8], offsets: &mut [usize]) -> Result<(), Status>;
fn bm25_scores<H: AsRef<[u8]>>(&mut self, haystacks: &[H], document_lengths: Option<&[f32]>,
    parameters: &Bm25Params, needle_weights: &[f32], scores: &mut [f32],
    scores_stride: usize) -> Result<(), Status>;

fn report(&self) -> SubstringsReport;
```

`hot_states` sizes the automaton's dense tier, or takes `SUBSTRINGS_HOT_STATES_AUTO` to fill a fixed byte budget instead, which holds more states the fewer byte classes the vocabulary spells.
`matches_budget` bounds what one round may emit and is read by a device tier alone, so a host engine takes `SUBSTRINGS_MATCHES_BUDGET_AUTO` and walks straight into the caller's output.

`CaseSensitivity::Cased` matches bytes exactly and accepts arbitrary needles, while `CaseSensitivity::Uncased` folds both sides under full Unicode case folding and requires valid UTF-8.
Folding is not a byte-length-preserving operation, so a 1-byte needle can match a 3-byte span — the Kelvin sign `U+212A` folds to `k` — which is why every `SubstringsMatch` carries its own `byte_length` rather than borrowing the needle's.

`SubstringsOverlapPolicy` decides what a walk reports, and it sizes the engine's arena, so it is fixed at construction and one engine runs exactly one of the three:

- `Overlapping` — every match of every needle, nested and overlapping ones included.
- `LeftmostLongest` — a cover taking the widest match at the earliest start.
- `LeftmostFirst` — a cover taking the lowest needle index at the earliest start.

A capacity shortfall is not an error.
The sizing walk always runs, so `report()` names `matches_emitted`, the true total; `matches_stored`, what was written; `tape_bytes`, what a rewrite needs; and `shortfall`, what did not fit.
That is what lets one call with an empty output size the next one, with no walk in between.

```rust
use stringzilla::sz::{
    CaseSensitivity, SubstringsEngine, SubstringsMatch, SubstringsOverlapPolicy,
    SUBSTRINGS_HOT_STATES_AUTO, SUBSTRINGS_MATCHES_BUDGET_AUTO,
};

let mut engine = SubstringsEngine::new(
    &["cat", "catalog"],
    CaseSensitivity::Cased,
    SubstringsOverlapPolicy::Overlapping,
    SUBSTRINGS_HOT_STATES_AUTO,
    SUBSTRINGS_MATCHES_BUDGET_AUTO,
)
.unwrap();

let documents = ["a catalog of cats", "nothing here"];
let mut counts = [0usize; 2];
engine.counts(&documents, &mut counts, 1).unwrap();
assert_eq!(counts, [3, 0]); // "catalog", the "cat" inside it, and the "cat" of "cats"

// `find` fills one boundary per haystack plus a final total, whether or not the matches fit.
let mut matches = [SubstringsMatch::default(); 3];
let mut offsets = [0usize; 3];
engine.find(&documents, &mut matches, &mut offsets).unwrap();
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

`replace` takes one replacement per needle, inserted verbatim, an empty one deleting its match, and writes one output tape beside its boundaries.
Rewriting is defined only under a cover, so an engine built with `SubstringsOverlapPolicy::Overlapping` is refused there — an overlapping rewrite is not a function.
A tape too small is not an error either: `report().tape_bytes` names the bytes the rewrite needed and the tape's contents are then unspecified, so an empty tape is how the next call is sized.

```rust
use stringzilla::sz::{Bm25Params, CaseSensitivity, SubstringsEngine, SubstringsOverlapPolicy,
                      SUBSTRINGS_HOT_STATES_AUTO, SUBSTRINGS_MATCHES_BUDGET_AUTO};

let mut engine = SubstringsEngine::new(
    &["cat", "dog"],
    CaseSensitivity::Cased,
    SubstringsOverlapPolicy::LeftmostLongest,
    SUBSTRINGS_HOT_STATES_AUTO,
    SUBSTRINGS_MATCHES_BUDGET_AUTO,
)
.unwrap();

let documents = ["cat and dog", "nothing here"];
let mut scores = [0.0f32; 2];
let parameters = Bm25Params::normalized(10.0);
engine
    .bm25_scores(&documents, None, &parameters, &[1.0, 1.0], &mut scores, 1)
    .unwrap();
assert_eq!(scores[1], 0.0);

// One replacement per needle: an empty tape sizes the rewrite, a second call performs it.
let replacements = ["feline", "canine"];
let mut offsets = [0usize; 3];
engine.replace(&documents, &replacements, &mut [], &mut offsets).unwrap();
let mut tape = vec![0u8; engine.report().tape_bytes];
engine.replace(&documents, &replacements, &mut tape, &mut offsets).unwrap();
assert_eq!(&tape[offsets[0]..offsets[1]], b"feline and canine");
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

`Utf8View` offers O(1) construction with lazy, cached `len()` for the codepoint count, `offset_of(n)` for the byte offset of the Nth codepoint, and `iter()` for batched `char` iteration:

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

### Engines on a GPU

With the `cuda` feature every engine gains a `new_on_gpu` constructor taking a `cudaStream_t` — or null for the current device's default stream — beside the arguments its host constructor takes.
That constructor is the only place a stream is ever named: it prepares the batch where a kernel reaches it and resolves the launch geometry once, and every compute verb of that engine then enqueues on the same stream and returns without joining.

It is `unsafe` for three reasons no type system checks.
The stream has to be live and belong to the current context.
Every output buffer has to be device-reachable memory — unified or plain device memory, never page-locked host memory, and a host buffer is refused with `Status::DeviceMemoryMismatch` rather than copied behind your back — and it has to outlive the launch.
And because a verb returns before the device has written anything, nothing it produced, `SubstringsEngine::report` included, may be read until the caller has joined the stream itself.

```rust
use stringzilla::sz::{LevenshteinEngine, LevenshteinSymbol};

// SAFETY: `stream` is live, `distances` is unified memory, and the caller joins before reading it.
let mut engine = unsafe {
    LevenshteinEngine::new_on_gpu(&["kitten", "saturday"], LevenshteinSymbol::Bytes, stream)?
};
engine.distances(&["sitting", "sunday"], distances, 2)?;
// cudaStreamSynchronize(stream) belongs here, before `distances` is read.
```

__One gap worth naming.__
A device engine's compute verbs also need a candidate sequence whose accessors run on the device, over device-resident texts, and the C tier builds one only inside a CUDA translation unit rather than exporting a symbol for it — so this crate can construct a device engine but cannot yet drive it, and a verb handed the host-side sequence the safe methods build answers `Status::DeviceMemoryMismatch` rather than letting a kernel read host memory.
The device allocators are unexported for the same reason, so nothing here hands back the unified memory those output buffers would have to live in.

A host engine imposes no such requirement: plain `Vec` and stack buffers are exactly what its verbs expect, and every one of them has returned by the time it answers.
