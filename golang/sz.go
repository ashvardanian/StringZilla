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
// - sz_hash_state_init_best, sz_hash_state_update_best, sz_hash_state_digest_best
// - sz_sha256_state_init_best, sz_sha256_state_update_best, sz_sha256_state_digest_best
//
// File: golang/sz.go
// Author: Ash Vardanian
package sz

// #cgo CFLAGS: -O3 -I${SRCDIR}/../include
// #cgo LDFLAGS: -L${SRCDIR} -lstringzilla_static
// #cgo noescape sz_find_best
// #cgo nocallback sz_find_best
// #cgo noescape sz_find_byte_best
// #cgo nocallback sz_find_byte_best
// #cgo noescape sz_rfind_best
// #cgo nocallback sz_rfind_best
// #cgo noescape sz_rfind_byte_best
// #cgo nocallback sz_rfind_byte_best
// #cgo noescape sz_find_byteset_best
// #cgo nocallback sz_find_byteset_best
// #cgo noescape sz_rfind_byteset_best
// #cgo nocallback sz_rfind_byteset_best
// #cgo noescape sz_bytesum_best
// #cgo nocallback sz_bytesum_best
// #cgo noescape sz_hash_best
// #cgo nocallback sz_hash_best
// #cgo noescape sz_utf8_uncased_fold_best
// #cgo nocallback sz_utf8_uncased_fold_best
// #cgo noescape sz_utf8_uncased_search_best
// #cgo nocallback sz_utf8_uncased_search_best
// #cgo noescape sz_utf8_count_best
// #cgo nocallback sz_utf8_count_best
// #cgo noescape sz_utf8_norm_best
// #cgo nocallback sz_utf8_norm_best
// #include <stringzilla/stringzilla.h>
import "C"
