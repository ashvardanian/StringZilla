// StringZilla is a SIMD-accelerated string library modern CPUs, written in C 99,
// and using AVX2, AVX512, Arm NEON, and SVE intrinsics to accelerate processing.
//
// The GoLang binding is intended to provide a simple interface to a precompiled
// shared library, available on GitHub: https://github.com/ashvardanian/stringzilla
//
// It requires Go 1.24 or newer to leverage the `cGo` `noescape` and `nocallback`
// directives. Without those the latency of calling C functions from Go is too high
// to be useful for string processing.
//
// Unlike the native Go `strings` package, StringZilla primarily targets byte-level
// binary data processing, with less emphasis on UTF-8 and locale-specific tasks.
//
// For some functions we are avoiding `noescape` and `nocallback`, assuming they use
// too much stack space:
// - sz_hash_state_init, sz_hash_state_update, sz_hash_state_digest
// - sz_sha256_state_init, sz_sha256_state_update, sz_sha256_state_digest
package sz

// #cgo CFLAGS: -O3 -mno-red-zone -I../include -DSZ_DYNAMIC_DISPATCH=1
// #cgo LDFLAGS: -L. -L/usr/local/lib -L../build_golang -L../build_release -L../build_shared -lstringzilla_shared
// #cgo noescape sz_find
// #cgo nocallback sz_find
// #cgo noescape sz_find_byte
// #cgo nocallback sz_find_byte
// #cgo noescape sz_rfind
// #cgo nocallback sz_rfind
// #cgo noescape sz_rfind_byte
// #cgo nocallback sz_rfind_byte
// #cgo noescape sz_find_byte_from
// #cgo nocallback sz_find_byte_from
// #cgo noescape sz_rfind_byte_from
// #cgo nocallback sz_rfind_byte_from
// #cgo noescape sz_bytesum
// #cgo nocallback sz_bytesum
// #cgo noescape sz_hash
// #cgo nocallback sz_hash
// #define SZ_DYNAMIC_DISPATCH 1
// #include <stringzilla/stringzilla.h>
import "C"
import (
	"fmt"
	"io"
	"unsafe"
)

// Explicitly initialize the dynamic dispatch table.
func init() {
	// The `__attribute__((constructor))` in the C library may not be called
	// by CGO's internal linker (see golang/go#28909), so we call it manually
	// to ensure the dispatch table is populated before any functions are used.
	C.sz_dispatch_table_init()
}

// Contains reports whether `substr` is within `str`.
// https://pkg.go.dev/strings#Contains
func Contains(str string, substr string) bool {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := len(substr)
	matchPtr := unsafe.Pointer(C.sz_find(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
	return matchPtr != nil
}

// Index returns the index of the first instance of `substr` in `str`, or -1 if `substr` is not present.
// https://pkg.go.dev/strings#Index
func Index(str string, substr string) int64 {
	substrLen := len(substr)
	if substrLen == 0 {
		return 0
	}
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	matchPtr := unsafe.Pointer(C.sz_find(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Index returns the index of the last instance of `substr` in `str`, or -1 if `substr` is not present.
// https://pkg.go.dev/strings#LastIndex
func LastIndex(str string, substr string) int64 {
	substrLen := len(substr)
	strLen := int64(len(str))
	if substrLen == 0 {
		return strLen
	}
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	matchPtr := unsafe.Pointer(C.sz_rfind(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Index returns the index of the first instance of a byte in `str`, or -1 if a byte is not present.
// https://pkg.go.dev/strings#IndexByte
func IndexByte(str string, c byte) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	cPtr := (*C.char)(unsafe.Pointer(&c))
	matchPtr := unsafe.Pointer(C.sz_find_byte(strPtr, C.ulong(strLen), cPtr))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Index returns the index of the last instance of a byte in `str`, or -1 if a byte is not present.
// https://pkg.go.dev/strings#LastIndexByte
func LastIndexByte(str string, c byte) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	cPtr := (*C.char)(unsafe.Pointer(&c))
	matchPtr := unsafe.Pointer(C.sz_rfind_byte(strPtr, C.ulong(strLen), cPtr))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Index returns the index of the first instance of any byte from `substr` in `str`, or -1 if none are present.
// Note: This is byte-set based (ASCII/bytes), not Unicode rune semantics like strings.IndexAny.
// https://pkg.go.dev/strings#IndexAny
func IndexAny(str string, substr string) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := len(substr)
	matchPtr := unsafe.Pointer(C.sz_find_byte_from(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Index returns the index of the last instance of any byte from `substr` in `str`, or -1 if none are present.
// Note: This is byte-set based (ASCII/bytes), not Unicode rune semantics like strings.LastIndexAny.
// https://pkg.go.dev/strings#LastIndexAny
func LastIndexAny(str string, substr string) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := len(substr)
	matchPtr := unsafe.Pointer(C.sz_rfind_byte_from(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Bytesum computes a simple 64-bit checksum by summing bytes.
func Bytesum(str string) uint64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := C.ulong(len(str))
	return uint64(C.sz_bytesum(strPtr, strLen))
}

// Hash computes a 64-bit non-cryptographic hash with a seed.
func Hash(str string, seed uint64) uint64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := C.ulong(len(str))
	return uint64(C.sz_hash(strPtr, strLen, (C.sz_u64_t)(seed)))
}

// Hasher is a streaming 64-bit non-cryptographic hasher that implements hash.Hash64 and io.Writer.
type Hasher struct {
	state C.sz_hash_state_t
	seed  uint64
}

// Compile-time interface checks
var _ io.Writer = (*Hasher)(nil)

// NewHasher creates a new streaming hasher with the given seed.
func NewHasher(seed uint64) *Hasher {
	h := &Hasher{seed: seed}
	C.sz_hash_state_init(&h.state, (C.sz_u64_t)(seed))
	return h
}

// Write adds data to the streaming hasher. Implements io.Writer.
func (h *Hasher) Write(p []byte) (n int, err error) {
	if len(p) > 0 {
		C.sz_hash_state_update(&h.state, (*C.char)(unsafe.Pointer(&p[0])), C.ulong(len(p)))
	}
	return len(p), nil
}

// Sum appends the current hash to b and returns the resulting slice.
// It does not change the underlying hash state. Implements hash.Hash.
func (h *Hasher) Sum(b []byte) []byte {
	digest := h.Sum64()
	return append(b,
		byte(digest>>56), byte(digest>>48), byte(digest>>40), byte(digest>>32),
		byte(digest>>24), byte(digest>>16), byte(digest>>8), byte(digest))
}

// Reset resets the hasher to its initial state. Implements hash.Hash.
func (h *Hasher) Reset() {
	C.sz_hash_state_init(&h.state, (C.sz_u64_t)(h.seed))
}

// Size returns the number of bytes Sum will return. Implements hash.Hash.
func (h *Hasher) Size() int {
	return 8
}

// BlockSize returns the hash's underlying block size. Implements hash.Hash.
func (h *Hasher) BlockSize() int {
	return 1 // No specific block size for this hash
}

// Sum64 returns the current 64-bit hash without consuming the state. Implements hash.Hash64.
func (h *Hasher) Sum64() uint64 {
	return uint64(C.sz_hash_state_digest(&h.state))
}

// Digest returns the current 64-bit hash without consuming the state.
// This is an alias for Sum64() for consistency with other bindings.
func (h *Hasher) Digest() uint64 {
	return h.Sum64()
}

// HashSha256 computes the SHA-256 cryptographic hash of the input data.
func HashSha256(data []byte) [32]byte {
	var state C.sz_sha256_state_t
	C.sz_sha256_state_init(&state)
	if len(data) > 0 {
		C.sz_sha256_state_update(&state, (*C.char)(unsafe.Pointer(&data[0])), C.ulong(len(data)))
	}
	var digest [32]byte
	C.sz_sha256_state_digest(&state, (*C.uchar)(unsafe.Pointer(&digest[0])))
	return digest
}

// Sha256 is a streaming SHA-256 hasher that implements hash.Hash and io.Writer.
type Sha256 struct {
	state C.sz_sha256_state_t
}

// Compile-time interface checks
var _ io.Writer = (*Sha256)(nil)

// NewSha256 creates a new streaming SHA-256 hasher.
func NewSha256() *Sha256 {
	h := &Sha256{}
	C.sz_sha256_state_init(&h.state)
	return h
}

// Write adds data to the streaming SHA-256 hasher. Implements io.Writer.
func (h *Sha256) Write(p []byte) (n int, err error) {
	if len(p) > 0 {
		C.sz_sha256_state_update(&h.state, (*C.char)(unsafe.Pointer(&p[0])), C.ulong(len(p)))
	}
	return len(p), nil
}

// Sum appends the current hash to b and returns the resulting slice.
// It does not change the underlying hash state. Implements hash.Hash.
func (h *Sha256) Sum(b []byte) []byte {
	digest := h.Digest()
	return append(b, digest[:]...)
}

// Reset resets the hasher to its initial state. Implements hash.Hash.
func (h *Sha256) Reset() {
	C.sz_sha256_state_init(&h.state)
}

// Size returns the number of bytes Sum will return. Implements hash.Hash.
func (h *Sha256) Size() int {
	return 32
}

// BlockSize returns the hash's underlying block size. Implements hash.Hash.
func (h *Sha256) BlockSize() int {
	return 64
}

// Digest returns the current SHA-256 hash as a 32-byte array without consuming the state.
// This is a convenience method in addition to the standard hash.Hash interface.
func (h *Sha256) Digest() [32]byte {
	var digest [32]byte
	C.sz_sha256_state_digest(&h.state, (*C.uchar)(unsafe.Pointer(&digest[0])))
	return digest
}

// Hexdigest returns the current SHA-256 hash as a lowercase hexadecimal string.
// This is a convenience method matching Python's hashlib interface.
func (h *Sha256) Hexdigest() string {
	digest := h.Digest()
	return fmt.Sprintf("%x", digest)
}

// Count returns the number of overlapping or non-overlapping instances of `substr` in `str`.
// If `substr` is an empty string, returns 1 + the length of the `str`.
// https://pkg.go.dev/strings#Count
func Count(str string, substr string, overlap bool) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := int64(len(str))
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := int64(len(substr))

	if substrLen == 0 {
		return 1 + strLen
	}
	if strLen == 0 || strLen < substrLen {
		return 0
	}

	count := int64(0)
	if overlap == true {
		for strLen > 0 {
			matchPtr := unsafe.Pointer(C.sz_find(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
			if matchPtr == nil {
				break
			}
			count += 1
			strLen -= (1 + int64(uintptr(matchPtr)-uintptr(unsafe.Pointer(strPtr))))
			strPtr = (*C.char)(unsafe.Add(matchPtr, 1))
		}
	} else {
		for strLen > 0 {
			matchPtr := unsafe.Pointer(C.sz_find(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
			if matchPtr == nil {
				break
			}
			count += 1
			strLen -= (substrLen + int64(uintptr(matchPtr)-uintptr(unsafe.Pointer(strPtr))))
			strPtr = (*C.char)(unsafe.Add(matchPtr, substrLen))
		}
	}

	return count
}
