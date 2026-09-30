# StringZilla for Go

StringZilla is a SIMD-accelerated string library for modern CPUs, written in C 99 and using AVX2, AVX-512, Arm NEON, and SVE intrinsics to accelerate processing.
This package is a thin `cgo` binding, the `sz` package, over the precompiled StringZilla static library.

Unlike the standard `strings` package, StringZilla primarily targets byte-level binary data processing, with less emphasis on UTF-8 and locale-specific tasks.
Where it does expose UTF-8 helpers, they are documented as such below.

It requires Go 1.24 or newer to leverage the `cgo` `noescape` and `nocallback` directives.
Without those the latency of calling C functions from Go would be too high to be useful for string processing.

## Installation

Add the module to your project with `go get`:

```sh
go get github.com/ashvardanian/stringzilla/golang
```

The binding links `libstringzilla_static` through `cgo`, so programs carry it whole and need no shared library at runtime.
Build it with the `release_shared` preset and copy it next to the Go sources:

```sh
cmake --preset release_shared
cmake --build --preset release_shared --target stringzilla_static
cp build_release_shared/libstringzilla_static.a golang/
```

An archive elsewhere, like the one `STRINGZILLA_LIBRARY_DIR` names for the Rust crate, links through `CGO_LDFLAGS="-L$STRINGZILLA_LIBRARY_DIR"`.

Import it as:

```go
import sz "github.com/ashvardanian/stringzilla/golang"
```

## Devices and Capabilities

Every call runs on the CPU and dispatches to the best kernel of its enabled capabilities, those it executes and this binary contains:

```go
cpu := sz.CPU()
enabled, _ := cpu.CapabilitiesEnabled()       // what dispatch uses: detected on this CPU and compiled in
fmt.Println(enabled)                          // like "serial,neon,neonaes,neonsha"
fmt.Println(enabled.Has(sz.CapNeon))          // test one capability
cpu.CapabilitiesEnable(enabled &^ sz.CapNeon) // narrow dispatch, returns what took effect
```

```go
func CPU() Device
func CountDevices(kind DeviceKind) (int, error)
func NewDevice(kind DeviceKind, ordinal int) (Device, error)
func (d Device) CapabilitiesDetected() (Capability, error)
func (d Device) CapabilitiesCompiled() Capability
func (d Device) CapabilitiesEnabled() (Capability, error)
func (d Device) CapabilitiesEnable(wanted Capability) (Capability, error)
func (d Device) ConfigureThread(capabilities Capability) (func(), error)
func (c Capability) Has(capability Capability) bool
func (c Capability) String() string
```

A `Device` is the host CPU or one GPU of a runtime, `DeviceCUDA`, `DeviceROCm` or `DeviceMetal`, named by that runtime's own ordinal.
`CountDevices` counts them and fails without a GPU of that kind, and `NewDevice` fails past the last one.
`CapabilitiesDetected` and `CapabilitiesCompiled` report the two raw axes, what the device executes and what this binary contains for its kind.
`CapabilitiesEnable` makes its argument the CPU's enabled set, clamped to both axes and always keeping `CapSerial`, and returns what took effect.
It fails on a GPU, which keeps no enabled set of its own: this package only reports GPU capabilities.
`ConfigureThread` pins the goroutine to an OS thread and prepares it for the kernels of its argument, returning the function that unpins it; it fails on a GPU, which has no thread state to configure.
Every capability is a typed `Capability` constant, like `CapSerial`, `CapHaswell`, `CapNeon`, `CapSve2` or `CapCuda`, and `CapCpus`, `CapDevices` and `CapAny` group them.
A call that reports a failure status returns it as an `error` where the function has one, and panics otherwise.

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
Create a `Hasher` with `NewHasher(seed)`; its zero value is not ready to use.

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
Create a `Sha256` with `NewSha256()`; its zero value is not ready to use.

```go
sum := sz.HashSha256([]byte("hello"))

h := sz.NewSha256()
h.Write([]byte("hello"))
fmt.Println(h.Hexdigest())
```

## UTF-8 Operations

The binding exposes codepoint counting, Unicode normalization, and full case folding over UTF-8 bytes.
Counts are scalar values, not bytes; normalization covers all four forms (`NFC`, `NFD`, `NFKC`, `NFKD`).

```go
sz.Utf8Count("你好世界")    // 4 codepoints
sz.Utf8Count("Hello🌍")     // 6 — the astral emoji is one scalar

sz.Utf8Normalize("é", sz.NFC)   // "e" + U+0301 → precomposed "é"
sz.Utf8Normalize("ﬁ", sz.NFKC)  // ligature "ﬁ" → "fi"
```

### Case Folding

These helpers apply full Unicode case folding.
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

For repeated searches with the same needle, `Utf8CaseInsensitiveNeedle` prepares it once, over its own copy of the bytes.
It is safe for concurrent use, since searches only read it.

```go
type Utf8CaseInsensitiveNeedle struct { /* ... */ }

func NewUtf8CaseInsensitiveNeedle(needle string, validate bool) (*Utf8CaseInsensitiveNeedle, error)
func (n *Utf8CaseInsensitiveNeedle) FindIn(haystack string, validate bool) (index int64, length int64, err error)
```

```go
needle, _ := sz.NewUtf8CaseInsensitiveNeedle("wörld", false)
idx, length, _ := needle.FindIn("Hello WÖRLD", false)
```
