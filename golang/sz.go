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
