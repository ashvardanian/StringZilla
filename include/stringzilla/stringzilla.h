/**
 *  @file include/stringzilla/stringzilla.h
 *  @author Ash Vardanian
 *  @date August 14, 2020
 *  @brief StringZilla is a collection of advanced string algorithms, designed for Big Data.
 *
 *  It is generally faster than LibC, and has a broader and cleaner interface for safer
 *  @b length-bounded strings. On modern CPUs it uses AVX2, AVX-512, NEON, SVE, SVE2, WebAssembly,
 *  RISC-V, LoongArch and Power @b SIMD, with a SWAR fallback for older CPUs. On @b CUDA-capable
 *  GPUs it also provides C++ kernels for bulk processing.
 *
 *  @see StringZilla docs: https://github.com/ashvardanian/StringZilla/blob/main/README.md
 *  @see LibC string docs: https://pubs.opengroup.org/onlinepubs/009695399/basedefs/string.h.html
 *
 *  @section sz_introduction Introduction
 *
 *  StringZilla is multi-language project designed for high-throughput string processing,
 *  differentiating the low-level "embeddable" mostly-C core implementation, containing:
 *
 *  - @c capabilities.h - capability bits, kernel kinds and signatures, the pick of a kernel, and
 *    CPU and GPU detection.
 *  - @c compare.h - byte-level comparison functions.
 *  - @c memory.h - copying, moving, and filling raw memory.
 *  - @c hash.h - hash functions and checksum algorithms.
 *  - @c cipher.h - AES-256 encryption in counter and Galois/counter modes.
 *  - @c find.h - searching for substrings and byte sets.
 *  - @c sort.h - single-threaded sorting algorithms.
 *  - @c intersect.h - intersections of unordered string sets.
 *  - @c levenshtein.h - edit distances between batches of queries and candidates, on a CPU or GPU.
 *  - @c overlap.h - window overlap between batches of queries and candidates, on a CPU or GPU.
 *  - @c substrings.h - multi-pattern search over a compiled vocabulary, on a CPU or a CUDA device.
 *  - @c small_string.h - "Small String Optimization" in C 99.
 *  - @c stringzilla.h - umbrella header for the core C API.
 *  - @c stringzilla.hpp - umbrella header for the core C++ API.
 *
 *  @section sz_compilation_settings Compilation Settings
 *
 *  Consider overriding the following macros to customize the library:
 *
 *  - `STRINGZILLA_DEBUG=0` - whether to enable debug assertions and logging.
 *  - `STRINGZILLA_WITH_LIBC=1` - whether to include the standard C library headers.
 *  - `STRINGZILLA_HEADER_ONLY=0` - whether to inline every kernel instead of linking the library,
 *    whose dispatch points then report @c sz_missing_library_k.
 *  - `STRINGZILLA_ALLOW_MISALIGNED_LOADS=0` - whether to use misaligned loads where supported.
 *  - `STRINGZILLA_WITH_METAL=0` - whether to compile the Metal host API, which links Metal.
 *
 *  Performance tuning:
 *
 *  - `STRINGZILLA_SWAR_THRESHOLD=24` - length from which SWAR replaces serial byte-level loops.
 *  - `STRINGZILLA_CACHE_LINE_BYTES=?` - cache-line width, derived from the target, that affects
 *    some algorithms and the first heap buffer of a growing string.
 *
 *  Different generations of CPUs and SIMD capabilities can be toggled with the following macros:
 *
 *  - `STRINGZILLA_TARGET_WESTMERE=?` - whether to use SSE4.2 and AES-NI instructions on x86_64.
 *  - `STRINGZILLA_TARGET_GOLDMONT=?` - whether to use SHA-NI instructions on x86_64.
 *  - `STRINGZILLA_TARGET_HASWELL=?` - whether to use AVX2 instructions on x86_64.
 *  - `STRINGZILLA_TARGET_SKYLAKE=?` - whether to use AVX-512 instructions on x86_64.
 *  - `STRINGZILLA_TARGET_ICELAKE=?` - whether to use AVX-512 VBMI and wider AES instructions on
 *    x86_64.
 *  - `STRINGZILLA_TARGET_NEON=?` - whether to use NEON instructions on Arm.
 *  - `STRINGZILLA_TARGET_NEONAES=?` - whether to use NEON AES instructions on Arm.
 *  - `STRINGZILLA_TARGET_NEONSHA=?` - whether to use NEON SHA-2 instructions on Arm.
 *  - `STRINGZILLA_TARGET_SVE=?` - whether to use SVE instructions on Arm.
 *  - `STRINGZILLA_TARGET_SVE2=?` - whether to use SVE2 instructions on Arm.
 *  - `STRINGZILLA_TARGET_SVE2AES=?` - whether to use SVE2 AES instructions on Arm.
 *  - `STRINGZILLA_TARGET_V128=?` - whether to use WebAssembly SIMD128 instructions.
 *  - `STRINGZILLA_TARGET_V128RELAXED=?` - whether to use WebAssembly relaxed-SIMD instructions.
 *  - `STRINGZILLA_TARGET_RVV=?` - whether to use RISC-V Vector (RVV 1.0) instructions.
 *  - `STRINGZILLA_TARGET_LOONGSONASX=?` - whether to use LoongArch LASX instructions.
 *  - `STRINGZILLA_TARGET_POWERVSX=?` - whether to use IBM Power VSX instructions.
 *  - `STRINGZILLA_TARGET_CUDA=?`, `STRINGZILLA_TARGET_ROCM=?`, `STRINGZILLA_TARGET_METAL=?` -
 *    whether to define the GPU kernels of a vendor, on by default wherever that vendor's compiler
 *    or runtime is.
 */
#ifndef STRINGZILLA_H_
#define STRINGZILLA_H_

#include "stringzilla/types.h"        // `sz_size_t`, `sz_bool_t`, `sz_ordering_t`, `STRINGZILLA_H_VERSION_MAJOR`
#include "stringzilla/capabilities.h" // `sz_capability_t`, `sz_cpu_capabilities_enabled`, `sz_capabilities_name`
#include "stringzilla/compare.h"      // `sz_equal_best`, `sz_order_best`
#include "stringzilla/memory.h"       // `sz_copy_best`, `sz_move_best`, `sz_fill_best`
#include "stringzilla/hash.h"         // `sz_bytesum_best`, `sz_hash_best`, `sz_hash_state_init_best`
#include "stringzilla/cipher.h"       // `sz_aes256_key_init_best`, `sz_aes256_ctr_xor_best`
#include "stringzilla/find.h"         // `sz_find_best`, `sz_find_byteset_best`, `sz_rfind_best`

#include "stringzilla/sort.h"        // `sz_sequence_argsort_best`, `sz_sequence_argsort_uncased_best`
#include "stringzilla/intersect.h"   // `sz_sequence_intersect_best`
#include "stringzilla/levenshtein.h" // `sz_levenshtein_engine_init`, `sz_levenshtein_distances`
#include "stringzilla/overlap.h"     // `sz_overlap_engine_init`, `sz_overlap_scores`
#include "stringzilla/substrings.h"  // `sz_substrings_engine_init`, `sz_substrings_find`

#include "stringzilla/utf8_runes.h"        // `sz_utf8_count_best`, `sz_utf8_seek_best`, `sz_utf8_decode_best`
#include "stringzilla/utf8_tokens.h"       // `sz_utf8_{newlines,whitespaces,delimiters}_best`
#include "stringzilla/utf8_wordbreaks.h"   // `sz_utf8_wordbreaks_best`, `sz_rune_word_break_property`
#include "stringzilla/utf8_graphemes.h"    // `sz_utf8_graphemes_best`
#include "stringzilla/utf8_sentences.h"    // `sz_utf8_sentences_best`
#include "stringzilla/utf8_linebreaks.h"   // `sz_utf8_linebreaks_best`
#include "stringzilla/utf8_uncased_fold.h" // `sz_utf8_uncased_fold_best`
#include "stringzilla/utf8_uncased.h"      // `sz_utf8_uncased_{search,order}_best`, `sz_utf8_find_cased_best`
#include "stringzilla/utf8_norm.h"         // `sz_utf8_norm_best`, `sz_utf8_find_denormalized_best`

#include "stringzilla/small_string.h" // `sz_string_t`, `sz_string_init`, `sz_string_free`

#ifdef __cplusplus
extern "C" {
#endif

STRINGZILLA_API int sz_version_major(void);
STRINGZILLA_API int sz_version_minor(void);
STRINGZILLA_API int sz_version_patch(void);

#if STRINGZILLA_HEADER_ONLY
STRINGZILLA_API int sz_version_major(void) { return STRINGZILLA_H_VERSION_MAJOR; }
STRINGZILLA_API int sz_version_minor(void) { return STRINGZILLA_H_VERSION_MINOR; }
STRINGZILLA_API int sz_version_patch(void) { return STRINGZILLA_H_VERSION_PATCH; }
#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // STRINGZILLA_H_
