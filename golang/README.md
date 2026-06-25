# StringZilla for Go

StringZilla is a SIMD-accelerated string library for modern CPUs, written in C 99 and using AVX2, AVX-512, Arm NEON, and SVE intrinsics to accelerate processing.
This package is a thin `cgo` binding, the `sz` package, over a precompiled StringZilla shared library.

Unlike the standard `strings` package, StringZilla primarily targets byte-level binary data processing, with less emphasis on UTF-8 and locale-specific tasks.
Where it does expose UTF-8 helpers, they are documented as such below.

It requires Go 1.24 or newer to leverage the `cgo` `noescape` and `nocallback` directives.
Without those the latency of calling C functions from Go would be too high to be useful for string processing.

## Installation

Add the module to your project with `go get`:

```sh
go get github.com/ashvardanian/stringzilla/golang
```

The binding links against the StringZilla shared library `libstringzilla_shared`.
Build it first, so that the linker can find it via the configured `LDFLAGS` search paths: `.`, `/usr/local/lib`, `../build_golang`, `../build_release`, and `../build_shared`.

Import it as:

```go
import sz "github.com/ashvardanian/stringzilla/golang"
```

The dispatch table is initialized automatically on package load.
To inspect which SIMD backend was selected at runtime:

```go
fmt.Println(sz.Capabilities()) // e.g. "serial,haswell,skylake,ice"
```

```go
func Capabilities() string
```

## Searching and Counting

These functions mirror the corresponding helpers in the standard `strings` package, but operate on raw bytes.

```go
func Contains(str string, substr string) bool
func Index(str string, substr string) int64
func LastIndex(str string, substr string) int64
func IndexByte(str string, c byte) int64
func LastIndexByte(str string, c byte) int64
func IndexAny(str string, substr string) int64
func LastIndexAny(str string, substr string) int64
func Count(str string, substr string, overlap bool) int64
```

`Index` / `LastIndex` return the byte offset of the first / last match, or `-1` if absent.
`IndexByte` / `LastIndexByte` do the same for a single byte.
`IndexAny` / `LastIndexAny` find the first / last byte that belongs to the byte set `substr`; note this is byte-set based, not Unicode-rune based like `strings.IndexAny`.
`Count` returns the number of matches, with `overlap` selecting overlapping versus non-overlapping counting; an empty `substr` yields `1 + len(str)`.

```go
haystack := "the quick brown fox"
sz.Contains(haystack, "brown")   // true
sz.Index(haystack, "brown")      // 10
sz.LastIndexByte(haystack, 'o')  // 17
sz.IndexAny(haystack, "aeiou")   // 2
sz.Count(haystack, "o", false)   // 2
```

## Hashing and Checksums

A fast 64-bit additive checksum and a seeded 64-bit non-cryptographic hash:

```go
func Bytesum(str string) uint64
func Hash(str string, seed uint64) uint64
```

```go
sum := sz.Bytesum("hello")
h := sz.Hash("hello", 42)
```

For incremental hashing, `Hasher` implements `hash.Hash64` and `io.Writer`:

```go
func NewHasher(seed uint64) *Hasher
func (h *Hasher) Write(p []byte) (n int, err error)
func (h *Hasher) Sum(b []byte) []byte
func (h *Hasher) Sum64() uint64
func (h *Hasher) Digest() uint64
func (h *Hasher) Reset()
func (h *Hasher) Size() int
func (h *Hasher) BlockSize() int
```

`Sum64` and its alias `Digest` return the current digest without consuming the state.

```go
h := sz.NewHasher(0)
h.Write([]byte("hello "))
h.Write([]byte("world"))
digest := h.Sum64()
```

### SHA-256

A cryptographic SHA-256 is available both as a one-shot and as a streaming hasher implementing `hash.Hash` and `io.Writer`:

```go
func HashSha256(data []byte) [32]byte

func NewSha256() *Sha256
func (h *Sha256) Write(p []byte) (n int, err error)
func (h *Sha256) Sum(b []byte) []byte
func (h *Sha256) Digest() [32]byte
func (h *Sha256) Hexdigest() string
func (h *Sha256) Reset()
func (h *Sha256) Size() int
func (h *Sha256) BlockSize() int
```

`Digest` returns the raw 32-byte hash and `Hexdigest` returns its lowercase hex string, both without consuming the state.

```go
sum := sz.HashSha256([]byte("hello"))

h := sz.NewSha256()
h.Write([]byte("hello"))
fmt.Println(h.Hexdigest())
```

## UTF-8 Case Operations

These helpers apply full Unicode case folding and are the only UTF-8-aware functions in the binding.
Each takes a `validate` flag; when set, inputs are checked and `ErrInvalidUTF8` is returned for malformed UTF-8.

```go
var ErrInvalidUTF8 = errors.New("invalid UTF-8")

func Utf8CaseFold(str string, validate bool) (string, error)
func Utf8CaseInsensitiveFind(haystack, needle string, validate bool) (index int64, length int64, err error)
```

`Utf8CaseFold` returns the case-folded string; folding may expand the output, so `"ß"` becomes `"ss"`.
`Utf8CaseInsensitiveFind` returns the byte `index` and matched byte `length` of the first case-insensitive match, or `index == -1` when no match is found; an empty needle returns `0, 0`.

```go
folded, _ := sz.Utf8CaseFold("Straße", false) // "strasse"
idx, length, _ := sz.Utf8CaseInsensitiveFind("Hello WÖRLD", "wörld", false)
// idx == 6, length == 6
```

For repeated searches with the same needle, `Utf8CaseInsensitiveNeedle` caches the precomputed needle metadata.
It is not safe for concurrent use, since the metadata is computed lazily and mutated on first search.

```go
type Utf8CaseInsensitiveNeedle struct { /* ... */ }

func NewUtf8CaseInsensitiveNeedle(needle string, validate bool) (*Utf8CaseInsensitiveNeedle, error)
func (n *Utf8CaseInsensitiveNeedle) FindIn(haystack string, validate bool) (index int64, length int64, err error)
```

```go
needle, _ := sz.NewUtf8CaseInsensitiveNeedle("wörld", false)
idx, length, _ := needle.FindIn("Hello WÖRLD", false)
```
