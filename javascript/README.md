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

The addon requires Node.js 22 or newer.
Supported platforms get a prebuilt one from their `@stringzilla/<platform>-<arch>` package, and elsewhere the install compiles it through `cmake-js`, which needs CMake and a C compiler.
A checkout builds it with `npm run prebuild`, which stages the addon under `prebuilds/` where the loader finds it first.

## Capabilities and Runtime Selection

Every call runs on the CPU and dispatches to the best kernel of its enabled capabilities, a `BigInt` bitmask a `Device` reports:

```js
const cpu = sz.Device.cpu();
console.log(cpu.capabilitiesEnabled()); // what dispatch uses
console.log((cpu.capabilitiesEnabled() & sz.Capability.haswell) !== 0n);
cpu.capabilitiesEnable(cpu.capabilitiesEnabled() & ~sz.Capability.skylake); // stop dispatching to AVX-512
```

`capabilitiesEnabled()` is the one you usually want, and derives from two independent axes:

- `capabilitiesDetected()`: what this device can execute.
- `capabilitiesCompiled()`: what this build contains for devices of its kind, from the ISA probes at build time.
- `capabilitiesEnabled()`: what dispatch uses, both axes at once unless narrowed.
- `capabilitiesEnable(wanted)`: makes `wanted` the CPU's enabled set, clamped to both axes, and returns what took effect.

`capabilitiesDetected()` describes the machine and says nothing about whether a kernel was compiled in, so a build whose ISA probes failed still reports your CPU's full feature set while containing no SIMD kernels at all.
The enabled set always keeps the `serial` fallback.

A `Device` is the host CPU or one GPU of a runtime, named by that runtime's own ordinal:

```js
sz.Device.count("cpu");            // 1
const gpu = new sz.Device("cuda"); // throws without a CUDA device, or past the last one
gpu.capabilitiesEnabled();         // what that GPU runs, like `Capability.cuda`
```

- `Device.cpu()` is the host CPU, which every build has.
- `Device.count(kind)` counts the devices of `"cpu"`, `"cuda"`, `"rocm"` or `"metal"`, and throws without a GPU of that kind.
- `new Device(kind, ordinal)` throws a `RangeError` past the last device.
- `capabilitiesEnable` throws on a GPU, which keeps no enabled set of its own; this binding only reports GPU masks.

`Capability` maps each lowercase capability name, like `haswell`, `neon` or `cuda`, to its bit, and the `cpus`, `devices` and `any` groups to theirs.
It is built at load from the C library's own names.

There is no `configure_thread` call in the JS binding.
Thread configuration is managed internally by the native addon.

## Runtimes

The binding is a native __Node-API__ addon, compiled from C against `node_api.h`.
Node-API is a stable, runtime-agnostic ABI, so the same compiled `.node` addon runs beyond Node.js.
It also loads on __Bun__ and __Deno__ through their Node-API compatibility layers, which implement the `napi_*` interface that this addon links against.
No Bun- or Deno-specific build is required; the addon and its `node-gyp-build`-based loader are shared across all three runtimes.

Separately, StringZilla's C/C++ core __compiles to WebAssembly__.
Targeting `wasm32-wasip1` with `STRINGZILLA_TARGET_ARCH` set to `v128` or `v128relaxed` enables the core's `STRINGZILLA_TARGET_V128` and `STRINGZILLA_TARGET_V128RELAXED` SIMD backends — the same v128 kernels the native addon uses.
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
A seed that is not a non-negative integer within 64 bits throws a `RangeError`, here and in the `Hasher` constructor.

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

## Unicode Normalization

`utf8Norm` converts a UTF-8 buffer into one of the four Unicode normal forms, passed as a `Utf8NormalForm` constant: `NFD`, `NFC`, `NFKD`, or `NFKC`.
It returns a new `Buffer` and accepts the same optional trailing `validate` boolean as the case-folding functions.

```js
const decomposed = Buffer.from("café"); // "e" followed by a combining acute
assert.strictEqual(sz.utf8Norm(decomposed, sz.Utf8NormalForm.NFC).toString(), "café");
```

`utf8FindDenormalized` checks a buffer against a normal form without rewriting it.
It returns the byte offset of the first violation as a `BigInt`, or `-1n` when the input is already normalized.

```js
assert.strictEqual(sz.utf8FindDenormalized(Buffer.from("café"), sz.Utf8NormalForm.NFC), -1n);
assert.strictEqual(sz.utf8FindDenormalized(decomposed, sz.Utf8NormalForm.NFC), 3n);
```

## Unicode Segmentation

Four iterable classes lazily split a UTF-8 buffer into TR29 and UAX14 segments: `Utf8Wordbreaks` for words, `Utf8Graphemes` for grapheme clusters, `Utf8Sentences` for sentences, and `Utf8Linebreaks` for line-break opportunities.
Each is constructed with the source buffer and an optional `validate` flag, and yields zero-copy `subarray` views into it, so the buffer must outlive the iteration.

```js
const words = [...new sz.Utf8Wordbreaks(Buffer.from("Hello world!"))].map((b) => b.toString());
assert.deepStrictEqual(words, ["Hello", " ", "world", "!"]);

const graphemes = [...new sz.Utf8Graphemes(Buffer.from("a👩‍👩‍👧‍👦b"))].map((b) => b.toString());
assert.deepStrictEqual(graphemes, ["a", "👩‍👩‍👧‍👦", "b"]); // the ZWJ family is one cluster
```
