/**
 *  @file include/stringzilla/find.h
 *  @author Ash Vardanian
 *  @date August 14, 2020
 *  @brief Hardware-accelerated sub-string and character-set search utilities.
 *
 *  Includes core APIs:
 *
 *  - @c sz_find_best and reverse-order @c sz_rfind_best
 *  - @c sz_find_byte_best and reverse-order @c sz_rfind_byte_best
 *  - @c sz_find_byteset_best and reverse-order @c sz_rfind_byteset_best
 *
 *  Convenience functions for character-set matching, on the serial kernels:
 *
 *  - @c sz_find_byte_from shortcut for @c sz_find_byteset_serial
 *  - @c sz_find_byte_not_from shortcut for @c sz_find_byteset_serial with inverted set
 *  - @c sz_rfind_byte_from shortcut for @c sz_rfind_byteset_serial
 *  - @c sz_rfind_byte_not_from shortcut for @c sz_rfind_byteset_serial with inverted set
 */
#ifndef STRINGZILLA_FIND_H_
#define STRINGZILLA_FIND_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Locates the first byte equal to the one at @p needle in @p haystack, like @c memchr.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle Needle - single-byte substring to find.
 *  @param[out] match Address of the first match, or NULL if not found.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @see x86_64 implementation in glibc: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memchr.S
 *  @see AArch64 implementation in glibc: https://github.com/lattera/glibc/blob/master/sysdeps/aarch64/memchr.S
 */
STRINGZILLA_API sz_status_t sz_find_byte_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_cptr_t *match, sz_capability_t capabilities, void *stream);

/**
 *  @brief Locates the last byte equal to the one at @p needle in @p haystack, like @c memrchr.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle Needle - single-byte substring to find.
 *  @param[out] match Address of the last match, or NULL if not found.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  There is no AArch64 reference implementation to link here.
 *
 *  @see x86_64 implementation in glibc: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memrchr.S
 */
STRINGZILLA_API sz_status_t sz_rfind_byte_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_cptr_t *match, sz_capability_t capabilities, void *stream);

/**
 *  @brief Locates the first occurrence of @p needle in @p haystack, like @c memmem in LibC, or like
 *      @c strstr for strings of known length.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle Needle - substring to find.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @param[out] match Address of the first match, or NULL if not found.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_find_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length, sz_cptr_t *match, sz_capability_t capabilities,
                                         void *stream);

/**
 *  @brief Locates the last matching substring.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle Needle - substring to find.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @param[out] match Address of the last match, or NULL if not found.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_rfind_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length, sz_cptr_t *match, sz_capability_t capabilities,
                                          void *stream);

/**
 *  @brief Finds the first character present from the @p set, present in @p haystack.
 *
 *  @param[in] haystack Haystack - the string to scan.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] set Set of relevant characters.
 *  @param[out] match Address of the first matching character from @p set, or NULL if not found.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Equivalent to `strspn(haystack, accepted)` and `strcspn(haystack, rejected)` in LibC. May have
 *  identical implementation and performance to @c sz_rfind_byteset_best. Useful for parsing, when
 *  we want to skip a set of characters, such as:
 *
 *  - 6 whitespaces: " \t\n\r\v\f".
 *  - 16 digits forming a float number: "0123456789,.eE+-".
 *  - 5 HTML reserved characters: "\"'&<>", of which "<>" can be useful for parsing.
 *  - 2 JSON string special characters useful to locate the end of the string: "\"\\".
 */
STRINGZILLA_API sz_status_t sz_find_byteset_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set,
                                                 sz_cptr_t *match, sz_capability_t capabilities, void *stream);

/**
 *  @brief Finds the last character present from the @p set, present in @p haystack.
 *
 *  @param[in] haystack Haystack - the string to scan.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] set Set of relevant characters.
 *  @param[out] match Address of the last matching character from @p set, or NULL if not found.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Equivalent to `strspn(haystack, accepted)` and `strcspn(haystack, rejected)` in LibC. May have
 *  identical implementation and performance to @c sz_find_byteset_best. Useful for parsing, when we
 *  want to skip a set of characters, such as:
 *
 *  - 6 whitespaces: " \t\n\r\v\f".
 *  - 16 digits forming a float number: "0123456789,.eE+-".
 *  - 5 HTML reserved characters: "\"'&<>", of which "<>" can be useful for parsing.
 *  - 2 JSON string special characters useful to locate the end of the string: "\"\\".
 */
STRINGZILLA_API sz_status_t sz_rfind_byteset_best(sz_cptr_t haystack, sz_size_t haystack_length,
                                                  sz_byteset_t const *set, sz_cptr_t *match,
                                                  sz_capability_t capabilities, void *stream);

/** @copydoc sz_find_byte_best */
STRINGZILLA_API sz_status_t sz_find_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byte_best */
STRINGZILLA_API sz_status_t sz_rfind_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_cptr_t *match, void *stream);

/** @copydoc sz_find_best */
STRINGZILLA_API sz_status_t sz_find_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_best */
STRINGZILLA_API sz_status_t sz_rfind_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_find_byteset_best */
STRINGZILLA_API sz_status_t sz_find_byteset_serial(sz_cptr_t haystack, sz_size_t haystack_length,
                                                   sz_byteset_t const *set, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byteset_best */
STRINGZILLA_API sz_status_t sz_rfind_byteset_serial(sz_cptr_t haystack, sz_size_t haystack_length,
                                                    sz_byteset_t const *set, sz_cptr_t *match, void *stream);

#if STRINGZILLA_TARGET_WESTMERE

/** @copydoc sz_find_byte_best */
STRINGZILLA_API sz_status_t sz_find_byte_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                  sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byte_best */
STRINGZILLA_API sz_status_t sz_rfind_byte_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_cptr_t *match, void *stream);

/** @copydoc sz_find_best */
STRINGZILLA_API sz_status_t sz_find_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_best */
STRINGZILLA_API sz_status_t sz_rfind_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_size_t needle_length, sz_cptr_t *match, void *stream);
#endif

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_find_byte_best */
STRINGZILLA_API sz_status_t sz_find_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byte_best */
STRINGZILLA_API sz_status_t sz_rfind_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                  sz_cptr_t *match, void *stream);

/** @copydoc sz_find_best */
STRINGZILLA_API sz_status_t sz_find_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_best */
STRINGZILLA_API sz_status_t sz_rfind_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_find_byteset_best */
STRINGZILLA_API sz_status_t sz_find_byteset_haswell(sz_cptr_t haystack, sz_size_t haystack_length,
                                                    sz_byteset_t const *set, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byteset_best */
STRINGZILLA_API sz_status_t sz_rfind_byteset_haswell(sz_cptr_t haystack, sz_size_t haystack_length,
                                                     sz_byteset_t const *set, sz_cptr_t *match, void *stream);
#endif

#if STRINGZILLA_TARGET_SKYLAKE

/** @copydoc sz_find_byte_best */
STRINGZILLA_API sz_status_t sz_find_byte_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byte_best */
STRINGZILLA_API sz_status_t sz_rfind_byte_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                  sz_cptr_t *match, void *stream);

/** @copydoc sz_find_best */
STRINGZILLA_API sz_status_t sz_find_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_best */
STRINGZILLA_API sz_status_t sz_rfind_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_size_t needle_length, sz_cptr_t *match, void *stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_find_byteset_best */
STRINGZILLA_API sz_status_t sz_find_byteset_icelake(sz_cptr_t haystack, sz_size_t haystack_length,
                                                    sz_byteset_t const *set, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byteset_best */
STRINGZILLA_API sz_status_t sz_rfind_byteset_icelake(sz_cptr_t haystack, sz_size_t haystack_length,
                                                     sz_byteset_t const *set, sz_cptr_t *match, void *stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_find_byte_best */
STRINGZILLA_API sz_status_t sz_find_byte_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byte_best */
STRINGZILLA_API sz_status_t sz_rfind_byte_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_cptr_t *match, void *stream);

/** @copydoc sz_find_best */
STRINGZILLA_API sz_status_t sz_find_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_best */
STRINGZILLA_API sz_status_t sz_rfind_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_find_byteset_best */
STRINGZILLA_API sz_status_t sz_find_byteset_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set,
                                                 sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byteset_best */
STRINGZILLA_API sz_status_t sz_rfind_byteset_neon(sz_cptr_t haystack, sz_size_t haystack_length,
                                                  sz_byteset_t const *set, sz_cptr_t *match, void *stream);
#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_find_byte_best */
STRINGZILLA_API sz_status_t sz_find_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byte_best */
STRINGZILLA_API sz_status_t sz_rfind_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_cptr_t *match, void *stream);

/** @copydoc sz_find_best */
STRINGZILLA_API sz_status_t sz_find_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                        sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_best */
STRINGZILLA_API sz_status_t sz_rfind_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length, sz_cptr_t *match, void *stream);
#endif

#if STRINGZILLA_TARGET_SVE2

/** @copydoc sz_find_byteset_best */
STRINGZILLA_API sz_status_t sz_find_byteset_sve2(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set,
                                                 sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byteset_best */
STRINGZILLA_API sz_status_t sz_rfind_byteset_sve2(sz_cptr_t haystack, sz_size_t haystack_length,
                                                  sz_byteset_t const *set, sz_cptr_t *match, void *stream);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_find_byte_best */
STRINGZILLA_API sz_status_t sz_find_byte_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byte_best */
STRINGZILLA_API sz_status_t sz_rfind_byte_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_cptr_t *match, void *stream);

/** @copydoc sz_find_best */
STRINGZILLA_API sz_status_t sz_find_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                        sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_best */
STRINGZILLA_API sz_status_t sz_rfind_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_find_byteset_best */
STRINGZILLA_API sz_status_t sz_find_byteset_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set,
                                                sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byteset_best */
STRINGZILLA_API sz_status_t sz_rfind_byteset_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set,
                                                 sz_cptr_t *match, void *stream);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_find_byte_best */
STRINGZILLA_API sz_status_t sz_find_byte_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byte_best */
STRINGZILLA_API sz_status_t sz_rfind_byte_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_cptr_t *match, void *stream);

/** @copydoc sz_find_best */
STRINGZILLA_API sz_status_t sz_find_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_best */
STRINGZILLA_API sz_status_t sz_rfind_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_find_byteset_best */
STRINGZILLA_API sz_status_t sz_find_byteset_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set,
                                                 sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byteset_best */
STRINGZILLA_API sz_status_t sz_rfind_byteset_v128(sz_cptr_t haystack, sz_size_t haystack_length,
                                                  sz_byteset_t const *set, sz_cptr_t *match, void *stream);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_find_byte_best */
STRINGZILLA_API sz_status_t sz_find_byte_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                     sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byte_best */
STRINGZILLA_API sz_status_t sz_rfind_byte_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                      sz_cptr_t *match, void *stream);

/** @copydoc sz_find_best */
STRINGZILLA_API sz_status_t sz_find_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_best */
STRINGZILLA_API sz_status_t sz_rfind_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_find_byteset_best */
STRINGZILLA_API sz_status_t sz_find_byteset_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length,
                                                        sz_byteset_t const *set, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byteset_best */
STRINGZILLA_API sz_status_t sz_rfind_byteset_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length,
                                                         sz_byteset_t const *set, sz_cptr_t *match, void *stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX

/** @copydoc sz_find_byte_best */
STRINGZILLA_API sz_status_t sz_find_byte_loongsonasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                     sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byte_best */
STRINGZILLA_API sz_status_t sz_rfind_byte_loongsonasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                      sz_cptr_t *match, void *stream);

/** @copydoc sz_find_best */
STRINGZILLA_API sz_status_t sz_find_loongsonasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_best */
STRINGZILLA_API sz_status_t sz_rfind_loongsonasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_find_byteset_best */
STRINGZILLA_API sz_status_t sz_find_byteset_loongsonasx(sz_cptr_t haystack, sz_size_t haystack_length,
                                                        sz_byteset_t const *set, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byteset_best */
STRINGZILLA_API sz_status_t sz_rfind_byteset_loongsonasx(sz_cptr_t haystack, sz_size_t haystack_length,
                                                         sz_byteset_t const *set, sz_cptr_t *match, void *stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_find_byte_best */
STRINGZILLA_API sz_status_t sz_find_byte_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                  sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byte_best */
STRINGZILLA_API sz_status_t sz_rfind_byte_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_cptr_t *match, void *stream);

/** @copydoc sz_find_best */
STRINGZILLA_API sz_status_t sz_find_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_best */
STRINGZILLA_API sz_status_t sz_rfind_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_size_t needle_length, sz_cptr_t *match, void *stream);

/** @copydoc sz_find_byteset_best */
STRINGZILLA_API sz_status_t sz_find_byteset_powervsx(sz_cptr_t haystack, sz_size_t haystack_length,
                                                     sz_byteset_t const *set, sz_cptr_t *match, void *stream);

/** @copydoc sz_rfind_byteset_best */
STRINGZILLA_API sz_status_t sz_rfind_byteset_powervsx(sz_cptr_t haystack, sz_size_t haystack_length,
                                                      sz_byteset_t const *set, sz_cptr_t *match, void *stream);
#endif

/**
 *  @brief Finds the search kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_find_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                sz_kernel_punned_t *kernel, sz_capability_t *capability);

/*  @c sz_utf8_delimiters (UTF-8 punctuation/symbol/separator/whitespace enumeration) lives in
 *  "stringzilla/utf8_tokens.h" alongside its per-ISA backends and property tables. */

#pragma endregion Core API

#include "stringzilla/find/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/find/westmere.h"
#include "stringzilla/find/haswell.h"
#include "stringzilla/find/skylake.h"
#include "stringzilla/find/icelake.h"
#include "stringzilla/find/neon.h"
#include "stringzilla/find/sve.h"
#include "stringzilla/find/sve2.h"
#include "stringzilla/find/v128relaxed.h"
#include "stringzilla/find/v128.h"
#include "stringzilla/find/rvv.h"
#include "stringzilla/find/loongsonasx.h"
#include "stringzilla/find/powervsx.h"
#endif // STRINGZILLA_HEADER_ONLY

#pragma region Helper Shortcuts

/**
 *  @brief Finds the first byte in @p haystack that is present in @p needle.
 *
 *  @param[in] haystack String to be scanned.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle String whose bytes form the accepted set.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @return Pointer to the first matching byte, or NULL if not found.
 */
STRINGZILLA_INLINE sz_cptr_t sz_find_byte_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    return sz_find_byteset_serial_(haystack, haystack_length, &set);
}

/**
 *  @brief Finds the first byte in @p haystack that is not present in @p needle.
 *
 *  @param[in] haystack String to be scanned.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle String whose bytes form the rejected set.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @return Pointer to the first non-matching byte, or NULL if not found.
 */
STRINGZILLA_INLINE sz_cptr_t sz_find_byte_not_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    sz_byteset_invert(&set);
    return sz_find_byteset_serial_(haystack, haystack_length, &set);
}

/**
 *  @brief Finds the last byte in @p haystack that is present in @p needle.
 *
 *  @param[in] haystack String to be scanned.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle String whose bytes form the accepted set.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @return Pointer to the last matching byte, or NULL if not found.
 */
STRINGZILLA_INLINE sz_cptr_t sz_rfind_byte_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    return sz_rfind_byteset_serial_(haystack, haystack_length, &set);
}

/**
 *  @brief Finds the last byte in @p haystack that is not present in @p needle.
 *
 *  @param[in] haystack String to be scanned.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle String whose bytes form the rejected set.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @return Pointer to the last non-matching byte, or NULL if not found.
 */
STRINGZILLA_INLINE sz_cptr_t sz_rfind_byte_not_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                    sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    sz_byteset_invert(&set);
    return sz_rfind_byteset_serial_(haystack, haystack_length, &set);
}

#pragma endregion Helper Shortcuts

/*  Header-only builds link no library, so their dispatch points only report it missing. */
#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_find_byte_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_cptr_t *match, sz_capability_t capabilities, void *stream) {
    sz_unused_(haystack), sz_unused_(haystack_length), sz_unused_(needle), sz_unused_(match), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byte_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_cptr_t *match, sz_capability_t capabilities, void *stream) {
    sz_unused_(haystack), sz_unused_(haystack_length), sz_unused_(needle), sz_unused_(match), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_find_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length, sz_cptr_t *match, sz_capability_t capabilities,
                                         void *stream) {
    sz_unused_(haystack), sz_unused_(haystack_length), sz_unused_(needle), sz_unused_(needle_length), sz_unused_(match),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_rfind_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length, sz_cptr_t *match, sz_capability_t capabilities,
                                          void *stream) {
    sz_unused_(haystack), sz_unused_(haystack_length), sz_unused_(needle), sz_unused_(needle_length), sz_unused_(match),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_find_byteset_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set,
                                                 sz_cptr_t *match, sz_capability_t capabilities, void *stream) {
    sz_unused_(haystack), sz_unused_(haystack_length), sz_unused_(set), sz_unused_(match), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byteset_best(sz_cptr_t haystack, sz_size_t haystack_length,
                                                  sz_byteset_t const *set, sz_cptr_t *match,
                                                  sz_capability_t capabilities, void *stream) {
    sz_unused_(haystack), sz_unused_(haystack_length), sz_unused_(set), sz_unused_(match), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_find_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_FIND_H_
