// Byte sums and seeded 64-bit hashes, one-shot and streaming, over the dispatched sz_hash family.
//
// File: golang/hash.go
// Author: Ash Vardanian

package sz

// #include <stringzilla/stringzilla.h>
import "C"
import (
	"io"
	"unsafe"
)

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
// This is an alias for [Hasher.Sum64].
func (h *Hasher) Digest() uint64 {
	return h.Sum64()
}
