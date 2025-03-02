/**
 *  @brief  StringZilla is a collection of advanced string algorithms, designed to be used in Big Data applications.
 *          It is generally faster than LibC, and has a broader & cleaner interface, and targets modern x86 CPUs
 *          with AVX-512 and Arm NEON and older CPUs with SWAR and auto-vectorization.
 *  @file   stringzilla.h
 *  @author Ash Vardanian
 *
 *  @see    StringZilla docs: https://github.com/ashvardanian/StringZilla/blob/main/README.md
 *  @see    LibC string docs: https://pubs.opengroup.org/onlinepubs/009695399/basedefs/string.h.html
 *
 *  @section    Introduction
 *
 *
 *  @section    Compilation Settings
 *
 *  Consider overriding the following macros to customize the library:
 *
 *  - `SZ_DEBUG=0` - whether to enable debug assertions and logging.
 *  - `SZ_AVOID_LIBC=0` - whether to avoid including the standard C library headers.
 *  - `SZ_DYNAMIC_DISPATCH=0` - whether to use runtime dispatching of the most advanced SIMD backend.
 *  - `SZ_USE_MISALIGNED_LOADS=0` - whether to use misaligned loads on platforms that support them.
 *
 *  Performance tuning:
 *
 *  - `SZ_SWAR_THRESHOLD=24` - threshold for switching to SWAR backend over serial byte-level for-loops.
 *  - `SZ_CACHE_LINE_WIDTH=64` - cache-line width that affects the execution of some algorithms.
 *  - `SZ_CACHE_SIZE=1048576` - the combined size of L1d and L2 caches in bytes, affecting temporal loads.
 *
 *  Different generations of CPUs and SIMD capabilities can be enabled or disabled with the following macros:
 *
 *  - `SZ_USE_HASWELL=?` - whether to use AVX2 instructions on x86_64.
 *  - `SZ_USE_SKYLAKE=?` - whether to use AVX-512 instructions on x86_64.
 *  - `SZ_USE_ICE=?` - whether to use AVX-512 VBMI & wider AES instructions on x86_64.
 *  - `SZ_USE_NEON=?` - whether to use NEON instructions on ARM.
 *  - `SZ_USE_SVE=?` - whether to use SVE and SVE2 instructions on ARM.
 */
#ifndef STRINGZILLA_H_
#define STRINGZILLA_H_

#define STRINGZILLA_H_VERSION_MAJOR 3
#define STRINGZILLA_H_VERSION_MINOR 11
#define STRINGZILLA_H_VERSION_PATCH 3

#include "types.h"        // `sz_size_t`, `sz_bool_t`, `sz_ordering_t`
#include "compare.h"      // `sz_equal`, `sz_order`
#include "memory.h"       // `sz_copy`, `sz_move`, `sz_fill`
#include "hash.h"         // `sz_bytesum`, `sz_hash`, `sz_state_init`, `sz_state_stream`, `sz_state_fold`
#include "find.h"         // `sz_find`, `sz_find_byteset`, `sz_rfind`
#include "small_string.h" // `sz_string_t`, `sz_string_init`, `sz_string_free`
#include "similarity.h"   // `sz_levenshtein_distance`, `sz_needleman_wunsch_score`
#include "sort.h"         // `sz_sequence_argsort`, `sz_pgrams_sort`, `sz_pgrams_sort_stable`

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Enumeration of SIMD capabilities of the target architecture.
 *          Used to introspect the supported functionality of the dynamic library.
 */
typedef enum {
    sz_cap_serial_k = 1,       ///< Serial (non-SIMD) capability
    sz_cap_any_k = 0x7FFFFFFF, ///< Mask representing any capability with `INT_MAX`

    sz_cap_haswell_k = 1 << 10, ///< x86 AVX2 capability with FMA and F16C extensions
    sz_cap_skylake_k = 1 << 11, ///< x86 AVX512 baseline capability
    sz_cap_ice_k = 1 << 12,     ///< x86 AVX512 capability with advanced integer algos and AES extensions

    sz_cap_neon_k = 1 << 20,     ///< ARM NEON baseline capability
    sz_cap_neon_aes_k = 1 << 21, ///< ARM NEON baseline capability with AES extensions
    sz_cap_sve_k = 1 << 24,      ///< ARM SVE baseline capability
    sz_cap_sve2_k = 1 << 25,     ///< ARM SVE2 capability
    sz_cap_sve2_aes_k = 1 << 26, ///< ARM SVE2 capability with AES extensions

} sz_capability_t;

/**
 *  @brief  Function to determine the SIMD capabilities of the current machine @b only at @b runtime.
 *  @return A bitmask of the SIMD capabilities represented as a `sz_capability_t` enum value.
 */
SZ_DYNAMIC sz_capability_t sz_capabilities(void);

/**
 *  @brief Internal helper function to convert SIMD capabilities to a string.
 *  @sa    sz_capabilities_to_string, sz_capabilities
 */
SZ_INTERNAL sz_cptr_t _sz_capabilities_to_string_implementation(sz_capability_t caps) {

    static char buf[256];
    char *p = buf;
    char *const end = buf + sizeof(buf);

    // Mapping each flag to its string literal.
    struct {
        sz_capability_t flag;
        char const *name;
    } capability_map[] = {
        {sz_cap_serial_k, "serial"}, {sz_cap_haswell_k, "haswell"}, {sz_cap_skylake_k, "skylake"},
        {sz_cap_ice_k, "ice"},       {sz_cap_neon_k, "neon"},       {sz_cap_neon_aes_k, "neon+aes"},
        {sz_cap_sve_k, "sve"},       {sz_cap_sve2_k, "sve2"},       {sz_cap_sve2_aes_k, "sve2+aes"},
    };
    int const capabilities_count = sizeof(capability_map) / sizeof(capability_map[0]);

    // Iterate over each capability flag.
    for (int i = 0; i < capabilities_count; i++) {
        if (caps & capability_map[i].flag) {
            int const is_first = p == buf;
            // Add separator if this is not the first capability.
            if (!is_first) {
                char const sep[3] = {',', ' ', '\0'};
                char const *s = sep;
                while (*s && p < end - 1) *p++ = *s++;
            }
            // Append the capability name character by character.
            char const *s = capability_map[i].name;
            while (*s && p < end - 1) *p++ = *s++;
        }
    }

    // If no capability was added, write "none".
    int const nothing_detected = p == buf;
    if (nothing_detected) {
        char const *s = "none";
        while (*s && p < end - 1) *p++ = *s++;
    }

    // Null-terminate the string.
    *p = '\0';
    return buf;
}

#if defined(SZ_DYNAMIC_DISPATCH)

SZ_DYNAMIC int sz_dynamic_dispatch(void);
SZ_DYNAMIC int sz_version_major(void);
SZ_DYNAMIC int sz_version_minor(void);
SZ_DYNAMIC int sz_version_patch(void);
SZ_DYNAMIC sz_cptr_t sz_capabilities_to_string(sz_capability_t caps);

#else

SZ_DYNAMIC int sz_dynamic_dispatch(void) { return 0; }
SZ_PUBLIC int sz_version_major(void) { return STRINGZILLA_H_VERSION_MAJOR; }
SZ_PUBLIC int sz_version_minor(void) { return STRINGZILLA_H_VERSION_MINOR; }
SZ_PUBLIC int sz_version_patch(void) { return STRINGZILLA_H_VERSION_PATCH; }
SZ_PUBLIC sz_cptr_t sz_capabilities_to_string(sz_capability_t caps) {
    return _sz_capabilities_to_string_implementation(caps);
}

#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // STRINGZILLA_H_
