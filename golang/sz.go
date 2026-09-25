// Package sz binds StringZilla, a SIMD-accelerated string library for modern CPUs, written in
// C 99 and using AVX2, AVX512, Arm NEON, and SVE intrinsics to accelerate processing.
//
// The GoLang binding is intended to provide a simple interface to a precompiled
// shared library, available on GitHub: https://github.com/ashvardanian/StringZilla
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
//
// File: golang/sz.go
// Author: Ash Vardanian
package sz

// #cgo CFLAGS: -O3 -I../include -DSTRINGZILLA_RUNTIME_DISPATCH=1
// #cgo LDFLAGS: -L. -L/usr/local/lib -L../build_golang -L../build_release -L../build_shared
// #cgo LDFLAGS: -lstringzilla_shared
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
// #cgo noescape sz_utf8_uncased_fold
// #cgo nocallback sz_utf8_uncased_fold
// #cgo noescape sz_utf8_uncased_search
// #cgo nocallback sz_utf8_uncased_search
// #cgo noescape sz_utf8_count
// #cgo nocallback sz_utf8_count
// #cgo noescape sz_utf8_norm
// #cgo nocallback sz_utf8_norm
// #define STRINGZILLA_RUNTIME_DISPATCH 1
// #include <stringzilla/stringzilla.h>
import "C"

// Explicitly initialize the dynamic dispatch table.
func init() {
	// The `__attribute__((constructor))` in the C library may not be called by CGO's internal linker
	// (see golang/go#28909), so we call it manually to ensure the dispatch table is populated before
	// any functions are used.
	C.sz_dispatch_cpu_table_init()
}
