# StringZilla for JavaScript

StringZilla is a native __Node-API__ (N-API) addon that exposes SIMD-accelerated string kernels to JavaScript.
Every function operates directly on Node `Buffer` objects, zero-copy, so no decoding or re-allocation happens on the JS side.
Byte offsets, counts, hashes, and sums are returned as `BigInt` values, suffixed with `n`, because they are 64-bit on the native side.

## Installation

Install the package from npm:

```sh
npm install stringzilla
```

```js
import sz from "stringzilla";
```

The addon requires Node.js 22 or newer and is compiled on install via `node-gyp`.
You can inspect which SIMD backends were selected at load time:

```js
console.log(sz.capabilities); // e.g. "serial,haswell,skylake"
```

## Runtimes

The binding is a native __Node-API__ addon, built from `javascript/lib.c` against `node_api.h`.
Node-API is a stable, runtime-agnostic ABI, so the same compiled `.node` addon runs beyond Node.js.
It also loads on __Bun__ and __Deno__ through their Node-API compatibility layers, which implement the `napi_*` interface that this addon links against.
No Bun- or Deno-specific build is required; the addon and its `bindings`-based loader are shared across all three runtimes.

Separately, StringZilla's C/C++ core __compiles to WebAssembly__.
The `cmake/toolchain-wasm32.cmake` toolchain targets `wasm32-wasip1` with `-msimd128` and `-mrelaxed-simd`, which auto-enables the core's `SZ_USE_V128` and `SZ_USE_V128RELAXED` SIMD backends (the same v128 kernels under `include/stringzilla/`).
This is a capability of the C core, exercised by the WebAssembly test builds; the npm package itself ships the N-API native addon and does not bundle a prebuilt `.wasm` artifact.

## Searching and Counting

`find` and `findLast` locate a needle `Buffer` inside a haystack `Buffer`, returning the byte offset as a `BigInt`, or `-1n` when absent.
An empty needle matches at `0n` for `find` and at the haystack length for `findLast`.

```js
import assert from "node:assert";

const haystack = Buffer.from("hello world, hello node");
const needle = Buffer.from("hello");

assert.strictEqual(sz.find(haystack, needle), 0n);
assert.strictEqual(sz.findLast(haystack, needle), 13n);
```

`findByte` and `findLastByte` search for a single byte value, a number in `0`–`255`, returning the first or last offset, or `-1n`.

```js
assert.strictEqual(sz.findByte(haystack, 0x6f), 4n);      // first 'o'
assert.strictEqual(sz.findLastByte(haystack, 0x6f), 19n); // last 'o'
```

`findByteFrom` and `findLastByteFrom` search for the first or last byte that belongs to a set.
The set is passed as a `Buffer` listing the allowed byte values.

```js
const vowels = Buffer.from("aeiou");
assert.strictEqual(sz.findByteFrom(haystack, vowels), 1n);      // first 'e'
assert.strictEqual(sz.findLastByteFrom(haystack, vowels), 22n); // last 'e' in "node"
```

`count` returns how many times a needle occurs in a haystack as a `BigInt`.
Pass `true` as the third argument to count overlapping matches; the default is non-overlapping.

```js
assert.strictEqual(sz.count(Buffer.from("aaaa"), Buffer.from("aa")), 2n);       // non-overlapping
assert.strictEqual(sz.count(Buffer.from("aaaa"), Buffer.from("aa"), true), 3n); // overlapping
```

## Comparing

`equal` reports whether two buffers hold identical bytes, returning a JavaScript boolean.
`compare` orders two buffers lexicographically, returning `-1`, `0`, or `1` as a plain number.

```js
assert.strictEqual(sz.equal(Buffer.from("abc"), Buffer.from("abc")), true);
assert.strictEqual(sz.compare(Buffer.from("abc"), Buffer.from("abd")), -1);
```

## Hashing and Checksums

`hash` computes StringZilla's fast 64-bit hash of a buffer, returned as a `BigInt`.
An optional second argument seeds the hash and accepts a `BigInt` or a number, defaulting to `0`.

```js
sz.hash(Buffer.from("hello"));        // => 64-bit BigInt
sz.hash(Buffer.from("hello"), 42n);   // seeded
```

`byteSum` returns the arithmetic sum of all byte values in a buffer as a `BigInt`, a cheap checksum.

```js
assert.strictEqual(sz.byteSum(Buffer.from([1, 2, 3])), 6n);
```

The `Hasher` class computes the same fast hash incrementally over chunks.
Construct it with an optional seed, feed data with `update`, read the running result with `digest`, and rewind to the initial seed with `reset`.
Both `update` and `reset` return the hasher, so calls can be chained.

```js
const h = new sz.Hasher(42n);
h.update(Buffer.from("hel")).update(Buffer.from("lo"));
assert.strictEqual(h.digest(), sz.hash(Buffer.from("hello"), 42n)); // 64-bit BigInt
h.reset(); // back to the seed state
```

`sha256` computes the SHA-256 digest of a buffer in one call, returning a 32-byte `Buffer`.

```js
sz.sha256(Buffer.from("hello")).toString("hex");
```

The `Sha256` class streams the same digest over chunks.
Use `update` to add data, then `digest` for a 32-byte `Buffer`, `hexdigest` for the 64-character lowercase hex string, or `reset` to start over.

```js
const s = new sz.Sha256();
s.update(Buffer.from("hel")).update(Buffer.from("lo"));
assert.strictEqual(s.hexdigest(), "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
```

## Unicode Case Insensitive Operations

These functions operate on UTF-8 encoded buffers and apply full Unicode case folding.
Each accepts an optional trailing `validate` boolean; when `true`, the input is checked for valid UTF-8 and an error is thrown otherwise.

`utf8UncasedFold` returns a new `Buffer` with the input case-folded.
The result may be longer than the input because some code points expand when folded.

```js
assert.strictEqual(sz.utf8UncasedFold(Buffer.from("Straße")).toString(), "strasse");
```

`utf8UncasedFind` performs a case-insensitive substring search.
It returns an object `{ index, length }`, both `BigInt`, where `index` is the matched byte offset, or `-1n` if not found, and `length` is the matched byte length, which can differ from the needle's length under folding.

```js
const { index, length } = sz.utf8UncasedFind(Buffer.from("Hello WÖRLD"), Buffer.from("wörld"));
```

The `Utf8UncasedNeedle` class precompiles a needle for repeated searches, amortizing the folding setup.
Construct it with the needle buffer and an optional `validate` flag, then call `findIn(haystack, validate?)`, which returns the same `{ index, length }` object.

```js
const needle = new sz.Utf8UncasedNeedle(Buffer.from("wörld"));
needle.findIn(Buffer.from("Hello WÖRLD")); // => { index, length }
```
