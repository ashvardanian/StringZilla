// SHA-256 digests, one-shot and streaming, backed by the dispatched sz_sha256 state.
//
// File: golang/sha256.go
// Author: Ash Vardanian

package sz

// #include <stringzilla/stringzilla.h>
import "C"
import (
	"fmt"
	"io"
	"unsafe"
)

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
// This is a convenience wrapper over [Sha256.Digest].
func (h *Sha256) Hexdigest() string {
	digest := h.Digest()
	return fmt.Sprintf("%x", digest)
}
