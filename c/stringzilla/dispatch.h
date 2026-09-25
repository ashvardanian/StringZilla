/**
 *  @file c/stringzilla/dispatch.h
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief Shared dispatch table for the per-domain StringZilla shims.
 *
 *  The compiled StringZilla library is split into one translation unit per domain, so that touching
 *  a domain only recompiles that domain: `compare.c`, `memory.c`, `hash.c`, `cipher.c`, `find.c`,
 *  `sort.c`, `intersect.c`, `levenshtein.c`, `overlap.c`, `substrings.c`, `utf8_runes.c`,
 *  `utf8_tokens.c`, `utf8_wordbreaks.c`, `utf8_graphemes.c`, `utf8_sentences.c`,
 *  `utf8_linebreaks.c`, `utf8_uncased_fold.c`, and `utf8_uncased.c`. Each TU includes only its own
 *  domain header, fills its slice of the shared @c sz_dispatch_cpu_table via
 *  `sz_dispatch_<domain>_update_`, and defines the @c STRINGZILLA_API_RUNTIME public wrappers that
 *  call through the table. The thin `runtime.c` owns the table definition and initializes it once.
 */
#ifndef STRINGZILLA_DISPATCH_H_
#define STRINGZILLA_DISPATCH_H_

#if !STRINGZILLA_RUNTIME_DISPATCH
#error "The dispatch shims are compiled with `STRINGZILLA_RUNTIME_DISPATCH=1`, which the build passes."
#endif

#include <stringzilla/types.h> // Function-pointer typedefs, `sz_capability_t`, `STRINGZILLA_TARGET_*`

/** The dispatch table and per-domain updaters are shared across translation units, but must stay
 *  internal to the shared object to preserve the exported ABI. */
#if defined(_MSC_VER)
#define STRINGZILLA_DISPATCH_INTERNAL
#else
#define STRINGZILLA_DISPATCH_INTERNAL __attribute__((visibility("hidden")))
#endif

typedef struct sz_implementations_t {
    sz_equal_t equal;
    sz_order_t order;

    sz_copy_t copy;
    sz_move_t move;
    sz_fill_t fill;
    sz_lookup_t lookup;

    sz_bytesum_t bytesum;
    sz_hash_t hash;
    sz_hash_multiseed_t hash_multiseed;
    sz_hash_state_init_t hash_state_init;
    sz_hash_state_update_t hash_state_update;
    sz_hash_state_digest_t hash_state_digest;
    sz_fill_random_t fill_random;

    sz_sha256_state_init_t sha256_state_init;
    sz_sha256_state_update_t sha256_state_update;
    sz_sha256_state_digest_t sha256_state_digest;
    sz_sha256_multistate_update_t sha256_multistate_update;
    sz_sha256_multistate_digest_t sha256_multistate_digest;

    sz_aes256_key_init_t aes256_key_init;
    sz_aes256_gcm_key_init_t aes256_gcm_key_init;
    sz_aes256_ctr_xor_t aes256_ctr_xor;
    sz_aes256_gcm_encrypt_t aes256_gcm_encrypt;
    sz_aes256_gcm_decrypt_t aes256_gcm_decrypt;
    sz_aes256_gcm_encryptor_init_t aes256_gcm_encryptor_init;
    sz_aes256_gcm_encryptor_associate_t aes256_gcm_encryptor_associate;
    sz_aes256_gcm_encryptor_update_t aes256_gcm_encryptor_update;
    sz_aes256_gcm_encryptor_digest_t aes256_gcm_encryptor_digest;
    sz_aes256_gcm_decryptor_init_t aes256_gcm_decryptor_init;
    sz_aes256_gcm_decryptor_associate_t aes256_gcm_decryptor_associate;
    sz_aes256_gcm_decryptor_update_unverified_t aes256_gcm_decryptor_update_unverified;
    sz_aes256_gcm_decryptor_verify_t aes256_gcm_decryptor_verify;

    sz_find_byte_t find_byte;
    sz_find_byte_t rfind_byte;
    sz_find_t find;
    sz_find_t rfind;
    sz_find_byteset_t find_byteset;
    sz_find_byteset_t rfind_byteset;

    sz_utf8_count_t utf8_count;
    sz_utf8_seek_t utf8_seek;
    sz_utf8_segmenter_t utf8_newlines;
    sz_utf8_segmenter_t utf8_whitespaces;
    sz_utf8_segmenter_t utf8_delimiters;
    sz_utf8_decode_t utf8_decode;

    sz_utf8_norm_t utf8_norm;
    sz_utf8_find_denormalized_t utf8_find_denormalized;
    sz_utf8_uncased_fold_t utf8_uncased_fold;
    sz_utf8_uncased_search_t utf8_uncased_search;

    sz_utf8_segmenter_t utf8_wordbreaks;
    sz_utf8_segmenter_t utf8_graphemes;
    sz_utf8_segmenter_t utf8_sentences;
    sz_utf8_segmenter_t utf8_linebreaks;
    sz_utf8_uncased_order_t utf8_uncased_order;

    sz_sequence_argsort_t sequence_argsort;
    sz_sequence_argsort_t sequence_argsort_uncased;
    sz_sequence_intersect_t sequence_intersect;

    sz_levenshtein_distances_t levenshtein_distances;
    sz_levenshtein_distances_t levenshtein_distances_utf8;

    sz_overlap_scores_t overlap_scores;

    sz_substrings_counts_t substrings_counts;
    sz_substrings_find_t substrings_find;
    sz_substrings_replace_t substrings_replace;
    sz_substrings_bm25_scores_t substrings_bm25_scores;

} sz_implementations_t;

/**
 *  @brief The global "virtual table" of supported @b CPU backends, defined in `stringzilla.c`
 *      and populated by the per-domain updaters below.
 *
 *  Holds CPU tiers only. A device backend is reached through @ref sz_dispatch_gpu_table instead,
 *  picked by the engine's own capability rather than by the machine's - so an engine built for the
 *  host scores on the host even where a device is present.
 */
extern STRINGZILLA_DISPATCH_INTERNAL sz_implementations_t sz_dispatch_cpu_table;

/**
 *  @brief The cross-product engines a device can run, defined in `stringzilla.c`.
 *
 *  Only the families with a device backend appear, and every slot is null until a GPU runtime is
 *  compiled in and a device answers. Populated by @ref sz_dispatch_gpu_table_init.
 */
typedef struct sz_implementations_gpu_t {
    sz_levenshtein_distances_t levenshtein_distances;
    sz_levenshtein_distances_t levenshtein_distances_utf8;
    sz_overlap_scores_t overlap_scores;
    sz_substrings_counts_t substrings_counts;
    sz_substrings_find_t substrings_find;
    sz_substrings_replace_t substrings_replace;
    sz_substrings_bm25_scores_t substrings_bm25_scores;
} sz_implementations_gpu_t;

extern STRINGZILLA_DISPATCH_INTERNAL sz_implementations_gpu_t sz_dispatch_gpu_table;

/*  Each updater fills only its own fields, defaulting to the serial backend and then
 *  overriding for the most capable enabled SIMD generation matching @p caps. */
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_compare_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_memory_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_hash_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_cipher_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_find_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_sort_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_intersect_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_levenshtein_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_overlap_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_substrings_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_utf8_runes_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_utf8_tokens_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_utf8_wordbreaks_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_utf8_graphemes_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_utf8_sentences_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_utf8_linebreaks_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_utf8_uncased_fold_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_utf8_norm_update_(sz_capability_t caps);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_utf8_uncased_update_(sz_capability_t caps);

STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_levenshtein_gpu_update_(void);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_overlap_gpu_update_(void);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_substrings_gpu_update_(void);
STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_gpu_table_init(void);

#if STRINGZILLA_ARCH_ARM64_ && (STRINGZILLA_TARGET_SVE || STRINGZILLA_TARGET_SVE2) && !defined(_MSC_VER)
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

/**
 *  @brief Whether the running CPU's SVE registers are wider than NEON's 128 bits - the point where
 *      the scalable kernels start outrunning NEON in the length-sensitive families. The runtime
 *      counterpart of the compile-time @c STRINGZILLA_SVE_WIDER_THAN_NEON_ in `types.h`.
 *
 *  The Arm tie-break policy, per Graviton 5 measurements: a scalable kernel dispatches
 *  unconditionally only when it wins at the minimal 128-bit length on the mixed multilingual corpus
 *  (byte search with its predicated heads, byteset scans, UTF-8 delimiters and sentences); kernels
 *  that only win with wider registers (substring find, memory ops, argsort, newlines, whitespaces,
 *  line breaks) stay behind this gate; and families whose scalar walk beats every 128-bit front
 *  (graphemes) install no Arm SIMD at all until the width flips the economics.
 */
STRINGZILLA_MAYBE_UNUSED_ STRINGZILLA_C_INLINE_ sz_bool_t sz_sve_wider_than_neon_(void) {
    return svcntb() > 16 ? sz_true_k : sz_false_k;
}
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_ARM64_ && (STRINGZILLA_TARGET_SVE || STRINGZILLA_TARGET_SVE2) && !defined(_MSC_VER)

#endif // STRINGZILLA_DISPATCH_H_
